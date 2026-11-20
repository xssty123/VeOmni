"""Two-chunk expert parallelism using asynchronous differentiable all-to-all."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd

from .....distributed.moe.moe_utils import sort_chunks_by_idxs
from .....distributed.parallel_state import get_parallel_state
from .....utils.device import stream_synchronize
from . import npu


@dataclass
class _Chunk:
    """Split metadata and inverse permutation for one EP token chunk."""

    send: list[int]
    recv: list[int]
    counts: torch.Tensor
    totals: torch.Tensor
    permuted: torch.Tensor
    inverse: torch.Tensor
    probs: torch.Tensor


def wrapper(
    hidden_states,
    routing_weights,
    selected_experts,
    fc1_1_weight,
    fc1_2_weight,
    fc2_weight,
    fc1_1_2_weight,
    *,
    num_experts,
    swiglu_limit=None,
    assume_distinct_experts=False,
    block_size=128,
    expert_dtype="fp8",
):
    """Run two ordered dispatches and overlap their combines with the other chunk's expert GEMMs."""
    import torch_npu

    ps = get_parallel_state()
    if not ps.ep_enabled:
        return npu.wrapper(
            hidden_states,
            routing_weights,
            selected_experts,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            num_experts=num_experts,
            swiglu_limit=swiglu_limit,
        )
    group = ps.ep_group
    local_experts = num_experts // dist.get_world_size(group)
    order = torch.arange(num_experts).reshape(-1, local_experts).T.flatten().tolist()
    inverse_order = torch.arange(num_experts).reshape(local_experts, -1).T.flatten().tolist()
    up = npu._fc1_weight(*(w if w.numel() else None for w in (fc1_1_weight, fc1_2_weight, fc1_1_2_weight))).transpose(
        1, 2
    )
    down = fc2_weight.transpose(1, 2)
    chunks = []
    hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    indices = selected_experts.reshape(hidden.size(0), selected_experts.size(-1))
    weights = routing_weights.reshape_as(indices).to(hidden.dtype)
    for x, ids, probs in zip(hidden.tensor_split(2), indices.tensor_split(2), weights.tensor_split(2), strict=True):
        send, recv, counts, totals = npu.dispatch_preprocess(ids, num_experts, group)
        if ids.numel():
            permuted, inverse = torch_npu.npu_moe_token_permute(x.contiguous(), ids.int().contiguous())
        else:
            permuted, inverse = x, ids.flatten().int()
        chunks.append(_Chunk(send, recv, counts, totals, permuted, inverse, probs))
    # dispatch_preprocess copies the count matrices to the host asynchronously.
    stream_synchronize()
    dispatched = [all_to_all_single_autograd(c.permuted, c.recv, c.send, group) for c in chunks]
    combined = []
    for chunk, remote in zip(chunks, dispatched, strict=True):
        x = sort_chunks_by_idxs(remote, chunk.counts.flatten(), order)
        if x.size(0):
            x = npu.npu_group_gemm(x, up, chunk.totals)
            x = npu.npu_group_gemm(npu._swiglu(x, swiglu_limit), down, chunk.totals)
        else:
            # Keep the same collective graph and zero local weight gradients on empty owners.
            x = x + (up[:0].sum() + down[:0].sum())
        x = sort_chunks_by_idxs(x, chunk.counts.T.flatten(), inverse_order)
        combined.append(all_to_all_single_autograd(x, chunk.send, chunk.recv, group))
    outputs = [
        torch_npu.npu_moe_token_unpermute(x, chunk.inverse, probs=chunk.probs.contiguous())
        if chunk.inverse.numel()
        else x + chunk.probs.sum() * 0
        for x, chunk in zip(combined, chunks, strict=True)
    ]
    return torch.cat(outputs)
