"""Head-first AscendC training adapter versus independent depthwise PyTorch convolution."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, resolve_op
from veomni.utils.device import IS_NPU_AVAILABLE


def _native_reference(x, weight, bias, *, heads, bounds):
    # Simulate the native [W,C] filter and NTD/BNSD output ABI with sliding windows.
    parts = [x] if bounds is None else [x[start:end].unsqueeze(0) for start, end in zip(bounds, bounds[1:])]
    outputs = []
    for part in parts:
        windows = F.pad(part, (0, 0, weight.size(0) - 1, 0)).unfold(1, weight.size(0), 1)
        y = (windows * weight.t()).sum(-1)
        outputs.append(y if bias is None else y + bias)
    y = torch.cat(outputs, dim=1).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
    return y if bounds is None else y.squeeze(0)


def _native_forward(x, weight, bias, *, activation, head_num, query_start_loc_cpu):
    assert activation is None
    return _native_reference(x, weight, bias, heads=head_num, bounds=query_start_loc_cpu)


def _native_backward(*, x, y, weight, dy, query_start_loc, activation, input_layout):
    assert input_layout == ("BNSD" if query_start_loc is None else "NTD")
    if activation:
        sigmoid = y.sigmoid()
        dy = dy * sigmoid * (1 + y * (1 - sigmoid))
    else:
        assert y is None
    with torch.enable_grad():
        x, weight = (value.detach().requires_grad_() for value in (x, weight))
        bias = x.new_zeros(x.shape[-1], requires_grad=True)
        raw = _native_reference(x, weight, bias, heads=dy.shape[-3], bounds=query_start_loc)
        return *torch.autograd.grad(raw, (x, weight, bias), dy), None


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
@pytest.mark.parametrize(
    "packed,bias,activation",
    [(False, False, None), (False, True, "silu"), (True, False, "swish"), (True, True, "silu")],
)
def test_head_first_convolution(device, packed, bias, activation, monkeypatch):
    if device == "cpu":
        module = SimpleNamespace(causal_conv1d_fn=_native_forward, causal_conv1d_bwd=_native_backward)
        monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", module)
        op = OP_REGISTRY._entries["causal_conv1d", "head_first", "fla_npu", "npu"].wrapper
    else:
        pytest.importorskip("fla_npu.ops.ascendc")
        op = resolve_op("causal_conv1d", "head_first", "fla_npu").wrapper

    heads, channels, width = 4, 512, 4
    lengths = (17, 2, 45) if packed else None
    dtype = torch.bfloat16 if device == "npu" else torch.float32
    rng = torch.Generator().manual_seed(28)
    shapes = [(1 if packed else 2, 64, channels), (channels, width)] + ([(channels,)] if bias else [])
    ref = [torch.randn(shape, generator=rng).to(dtype).float().requires_grad_() for shape in shapes]
    inputs = [item.detach().to(device=device, dtype=dtype).requires_grad_() for item in ref]
    actual = op(*inputs, heads=heads, packed_lengths=lengths, activation=activation)
    x, weight, *optional_bias = ref
    parts = x.split(lengths, dim=1) if packed else (x,)
    ys = [
        F.conv1d(F.pad(part.transpose(1, 2), (width - 1, 0)), weight[:, None], *optional_bias, groups=channels)
        for part in parts
    ]
    expected = torch.cat(ys, dim=-1).transpose(1, 2).unflatten(-1, (heads, -1)).transpose(1, 2)
    if activation:
        expected = F.silu(expected)
    grad = torch.randn(expected.shape, generator=rng).to(dtype)
    tol = dict(rtol=0.025, atol=0.15) if device == "npu" else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.cpu().float(), expected, **tol)
    assert actual.is_contiguous()
    for got, want in zip(
        torch.autograd.grad(actual, inputs, grad.to(device)),
        torch.autograd.grad(expected, ref, grad.float()),
        strict=True,
    ):
        torch.testing.assert_close(got.cpu().float(), want, **tol)
