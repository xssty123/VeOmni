"""Qwen QSA complete-block selection, including right-aligned sparse query suffixes."""

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import resolve_op
from veomni.ops.kernels.qsa_attention import lightning_indexer
from veomni.utils.device import IS_NPU_AVAILABLE


def _mock_lightning_indexer(q, keys, weights, topk, *, cmp_residual_k, metadata, **kwargs):
    batch, rows, heads, dim = q.shape
    blocks = keys.size(1)
    ratio = kwargs["cmp_ratio"]
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["mask_mode"], kwargs["return_value"]) == (
        "BSND",
        "BSND",
        3,
        0,
    )
    assert (metadata["batch_size"], metadata["max_seqlen_q"], metadata["max_seqlen_k"]) == (batch, rows, blocks)
    assert metadata["num_heads_k"] == 1 and metadata["num_heads_q"] == heads and metadata["head_dim"] == dim
    torch.testing.assert_close(weights, torch.full_like(weights, dim**-0.5))
    length = blocks * ratio + cmp_residual_k[:, None]
    visible = (length - rows + torch.arange(1, rows + 1)) // ratio
    scores = (torch.einsum("brhd,bkd->brhk", q.float(), keys.squeeze(2).float()).relu() * weights[..., None]).sum(2)
    scores.masked_fill_(torch.arange(blocks) >= visible[..., None], -torch.inf)
    return scores.topk(topk).indices.int().unsqueeze(2), None


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))),
        pytest.param(
            "npu",
            marks=[
                pytest.mark.launch_plan(provider=LaunchSpec()),
                pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_lightning_complete_blocks(device, packed, monkeypatch):
    if device == "cpu":
        module = SimpleNamespace(
            lightning_indexer=_mock_lightning_indexer, lightning_indexer_metadata=lambda **kwargs: kwargs
        )
        monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", module)
        indexer = lightning_indexer.wrapper
    else:
        pytest.importorskip("cann_ops_transformer.ops")
        indexer = resolve_op("qsa_indexer", "complete_blocks", "cann").wrapper

    batch, length, heads, dim, ratio, topk = 2, 75, 32, 128, 4, 8
    segments = ((0, 0, 3), (0, 3, 75), (1, 0, 37), (1, 37, 75)) if packed else ((0, 0, 75), (1, 0, 75))
    # Even queries rank the newest block first, odd queries rank the oldest.
    q = torch.zeros(batch, length, heads, dim, dtype=torch.bfloat16)
    q[:, ::2, :, 0] = 1
    q[:, 1::2, :, 1] = 1
    keys = []
    expected = torch.full((batch, length, topk * ratio), -1, dtype=torch.int32)
    for row, start, end in segments:
        count = (end - start) // ratio
        segment_keys = torch.zeros(count, dim, dtype=q.dtype)
        segment_keys[:, 0] = torch.arange(1, count + 1)
        segment_keys[:, 1] = torch.arange(count, 0, -1)
        keys.append(segment_keys)
        for pos in range(start, end):
            visible = (pos - start + 1) // ratio
            first = max(0, visible - topk) if pos % 2 == 0 else 0
            num_tokens = min(topk, visible) * ratio
            expected[row, pos, :num_tokens] = start + first * ratio + torch.arange(num_tokens)
    selected = indexer(
        q.to(device), torch.cat(keys).to(device), segments=segments, compress_ratio=ratio, block_topk=topk
    )
    torch.testing.assert_close(selected.cpu().sort(-1).values, expected.sort(-1).values)
