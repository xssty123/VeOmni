"""Host-resident persistent embedding shards with row-sparse local gradients."""

from functools import lru_cache

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from ..utils.device import create_event, create_stream, get_current_stream, switch_to_specified_stream


class HostEmbedding(nn.Embedding):
    """Keep the local FP32 table on CPU across the model's FSDP materialization."""

    def __init__(self, num_embeddings, embedding_dim, *, dtype):
        device = "meta" if torch.empty(0).is_meta else "cpu"
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)

    def _apply(self, fn, recurse=True):
        # Other modules follow the accelerator placement of model.to_empty/to.
        # The host table has a fixed dtype/device; only meta storage needs allocation.
        if self.weight.is_meta and not fn(torch.empty(0, device="meta")).is_meta:
            old = self.weight
            self.weight = nn.Parameter(torch.empty_like(old, device="cpu"), requires_grad=old.requires_grad)
            if hasattr(old, "spec_info"):
                self.weight.spec_info = old.spec_info
        return self


def local_row_leaf(weight):
    """Share the DCP parameter storage with a leaf whose gradient can be sparse COO.

    DTensor sparse embedding backward is unsupported by PyTorch 2.10. The
    optimizer and clipper read this leaf's gradient; DCP sees the global DTensor.
    """
    leaf = getattr(weight, "_host_row_leaf", None)
    if leaf is None:
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        if local.device.type != "cpu":
            raise RuntimeError("Host PLE shards must be materialized on the CPU before lookup")
        leaf = local.detach().requires_grad_(weight.requires_grad)
        leaf._pending_host_rows = []
        weight._host_row_leaf = leaf
    return leaf


@lru_cache(maxsize=8)
def _gradient_stream(device):
    return create_stream(device)


class _HostRows(torch.autograd.Function):
    """Defer the host read of row gradients until clipping or the optimizer step."""

    @staticmethod
    def forward(ctx, device, dtype, ids, *leaves):
        ctx.ids = ids
        ctx.leaves = leaves
        pieces = [weight.index_select(0, indices) for weight, indices in zip(leaves, ids, strict=True)]
        return torch.cat(pieces).to(dtype=dtype).pin_memory().to(device, non_blocking=True)

    @staticmethod
    def backward(ctx, grad):
        stream = _gradient_stream(grad.device)
        stream.wait_stream(get_current_stream(grad.device))
        with switch_to_specified_stream(stream):
            buffer = torch.empty(grad.shape, dtype=grad.dtype, device="cpu", pin_memory=True)
            buffer.copy_(grad, non_blocking=True)
            grad.record_stream(stream)
            event = create_event()
            event.record(stream)
        parts = buffer.split([ids.numel() for ids in ctx.ids])
        for leaf, ids, values in zip(ctx.leaves, ctx.ids, parts, strict=True):
            if leaf.requires_grad and ids.numel():
                leaf._pending_host_rows.append((event, ids, values))
        return (None,) * (3 + len(ctx.leaves))


def drain_row_gradients(leaf):
    """Wait for D2H completion before the CPU reads or accumulates any pinned values."""
    for event, ids, values in leaf._pending_host_rows:
        event.synchronize()
        grad = torch.sparse_coo_tensor(ids.unsqueeze(0), values.to(leaf.dtype), leaf.shape)
        leaf.grad = grad if leaf.grad is None else leaf.grad + grad
    leaf._pending_host_rows.clear()


def lookup_host_rows(shards, shard_ids, row_ids, output_dtype, shard_stride):
    """Deduplicate owner-local requests and transfer only the requested row values."""
    device = row_ids.device
    keys = (shard_ids * shard_stride + row_ids).cpu()
    unique, inverse = keys.unique(sorted=True, return_inverse=True)
    owners = unique // shard_stride
    counts = torch.bincount(owners, minlength=len(shards)).tolist()
    requests = [
        (local_row_leaf(table.weight), ids)
        for table, ids in zip(shards.values(), unique.remainder(shard_stride).split(counts), strict=True)
        if ids.numel()
    ]
    if not any(leaf.requires_grad for leaf, _ in requests):
        # Keep reverse collectives aligned on empty owners or partially frozen tables.
        anchor = next((table for table in shards.values() if table.weight.requires_grad), shards["shard_0"])
        requests.append((local_row_leaf(anchor.weight), unique[:0]))
    if device.type == "cpu":
        rows = torch.cat([nn.functional.embedding(ids, leaf, sparse=True) for leaf, ids in requests]).to(output_dtype)
    else:
        leaves, ids = zip(*requests, strict=True)
        rows = _HostRows.apply(device, output_dtype or leaves[0].dtype, ids, *leaves)
    return rows.index_select(0, inverse.to(device))
