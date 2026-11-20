# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CANN Lightning Indexer for Qwen's complete compressed causal blocks."""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def _metadata(batch, rows, blocks, heads, dim, topk, ratio, remainder, device):
    from cann_ops_transformer.ops import lightning_indexer_metadata

    residual = torch.full((batch,), remainder, dtype=torch.int32, device=device)
    metadata = lightning_indexer_metadata(
        num_heads_q=heads,
        num_heads_k=1,
        head_dim=dim,
        topk=topk,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=rows,
        max_seqlen_k=blocks,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    return residual, metadata


def _select_segment(q, keys, ratio, topk):
    batch, length, heads, dim = q.shape
    blocks = keys.shape[1]
    width = min(topk, blocks)
    # Until the (topk+1)th block is complete every visible block is selected.
    prefix = min(length, (topk + 1) * ratio - 1)
    slots = torch.arange(width, dtype=torch.int32, device=q.device)
    end = torch.arange(1, prefix + 1, dtype=torch.int32, device=q.device) // ratio
    dense = torch.where(slots < end[:, None], slots, -1).expand(batch, -1, -1)
    if prefix == length:
        return dense

    from cann_ops_transformer.ops import lightning_indexer

    rows = length - prefix
    residual, metadata = _metadata(batch, rows, blocks, heads, dim, topk, ratio, length % ratio, q.device)
    weights = torch.full((batch, rows, heads), dim**-0.5, dtype=torch.float32, device=q.device)
    indices, _ = lightning_indexer(
        q[:, prefix:].contiguous(),
        keys.unsqueeze(2).contiguous(),
        weights,
        topk,
        cmp_residual_k=residual,
        metadata=metadata,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
        return_value=0,
    )
    return torch.cat((dense, indices.squeeze(2)), dim=1)


@torch.no_grad()
def wrapper(q, pooled_keys, *, segments, compress_ratio, block_topk):
    """Return complete-block token IDs for local queries; the model appends each causal tail.

    At sequence parallel size one, pooled keys are already ordered by batch,
    segment and block. Each packed segment restarts CANN's causal block numbering.
    """
    batch, length, _, dim = q.shape
    if dim != 128 or compress_ratio not in (1, 4, 128) or block_topk < 1:
        raise ValueError("QSA CANN requires head_dim=128, compression ratio 1/4/128 and a positive block budget")
    if q.dtype not in (torch.float16, torch.bfloat16) or pooled_keys.dtype != q.dtype:
        raise ValueError("QSA CANN requires matching FP16/BF16 index queries and keys")

    lanes = torch.arange(compress_ratio, device=q.device, dtype=torch.int32)

    def tokens(indices, start):
        ids = start + indices[..., None] * compress_ratio + lanes
        return ids.masked_fill(indices[..., None] < 0, -1).flatten(-2)

    if len(segments) == batch:
        keys = pooled_keys.reshape(batch, length // compress_ratio, dim)
        return tokens(_select_segment(q, keys, compress_ratio, block_topk), 0)

    result = torch.full((batch, length, block_topk * compress_ratio), -1, dtype=torch.int32, device=q.device)
    offset = 0
    for row, start, end in segments:
        count = (end - start) // compress_ratio
        if count:
            keys = pooled_keys[offset : offset + count].unsqueeze(0)
            selected = tokens(_select_segment(q[row : row + 1, start:end], keys, compress_ratio, block_topk), start)
            result[row, start:end, : selected.size(-1)] = selected[0]
        offset += count
    return result
