"""Ascend vocabulary-tiled linear cross entropy for BF16 SFT."""

import torch


def wrapper(hidden, labels, weight, *, ignore_index=-100, num_items_in_batch=None):
    """Sum losses over supervised rows and apply the caller's token denominator."""
    from .cce_triton import ChunkLossCceFused

    if weight is None or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("cce_npu requires hidden states and a [vocabulary, hidden] projection weight")
    if hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype:
        raise ValueError("cce_npu requires BF16 hidden states and projection weight")
    labels = labels.reshape(-1)
    hidden = hidden.reshape(-1, weight.size(1))
    if labels.numel() != hidden.size(0):
        raise ValueError("One shifted target is required for each hidden row")
    valid = labels.ne(ignore_index)
    rows = valid.nonzero().flatten()
    if not rows.numel():
        return hidden[:0].sum(dtype=torch.float32) + weight[:0].sum(dtype=torch.float32)
    if rows.numel() != hidden.size(0):
        hidden = hidden.index_select(0, rows)
        labels = labels.index_select(0, rows)
    torch._assert_async(((labels >= 0) & (labels < weight.size(0))).all(), "CCE target outside vocabulary")
    loss = ChunkLossCceFused.apply(hidden, weight, labels, min(4096, weight.size(0)), ignore_index)
    denom = rows.numel() if num_items_in_batch is None else num_items_in_batch
    return loss / torch.as_tensor(denom, device=loss.device).clamp_min(1)
