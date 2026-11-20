"""Two-rank chunked dispatch with real collectives and a dense expert oracle."""

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.ops.kernels.moe_experts.standard import chunk_npu
from veomni.utils.device import IS_NPU_AVAILABLE


def _worker(rank, device, endpoint):
    if device == "npu":
        torch.npu.set_device(rank)
    dist.init_process_group(
        "gloo" if device == "cpu" else "hccl",
        store=endpoint.connect(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )

    def permute(x, ids):
        order = ids.flatten().argsort(stable=True)
        return x[order // ids.size(1)], order

    def unpermute(x, order, *, probs):
        return (x[order.argsort()].reshape(*probs.shape, x.size(-1)) * probs[..., None]).sum(1)

    def gemm(x, weight, counts):
        return torch.cat([part @ w for part, w in zip(x.split(counts.tolist()), weight, strict=True)])

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.callback(dist.destroy_process_group)
        stack.enter_context(
            patch.object(
                chunk_npu,
                "get_parallel_state",
                return_value=SimpleNamespace(ep_enabled=True, ep_group=dist.group.WORLD),
            )
        )
        if device == "cpu":
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    torch_npu=SimpleNamespace(npu_moe_token_permute=permute, npu_moe_token_unpermute=unpermute),
                )
            )
            stack.enter_context(patch.object(chunk_npu.npu, "npu_group_gemm", gemm))
            stack.enter_context(
                patch.object(chunk_npu.npu, "_swiglu", lambda x, limit: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
            )
        dtype = torch.bfloat16 if device == "npu" else torch.float32
        torch.manual_seed(94)
        up, down = [
            torch.randn(shape).mul_(0.05).to(device=device, dtype=dtype).requires_grad_()
            for shape in [(4, 64, 16), (4, 16, 32)]
        ]
        tokens = 5 if rank == 0 else 1
        x = torch.randn(tokens, 16, device=device, dtype=dtype, requires_grad=True)
        ids = torch.tensor([[0, 1], [2, 1], [0, 1], [0, 1], [0, 1]][:tokens], device=device)
        probs = torch.rand(ids.shape, device=device, requires_grad=True)
        u, d = (w.detach().chunk(2)[rank].contiguous().requires_grad_() for w in (up, down))
        empty = x.new_empty(0)
        actual = chunk_npu.wrapper(x, probs, ids, empty, empty, d, u, num_experts=4)
        gate, value = torch.einsum("td,tkod->tko", x, up[ids]).chunk(2, -1)
        ref = (torch.einsum("tki,tkdi->tkd", F.silu(gate) * value, down[ids]) * probs.to(dtype)[..., None]).sum(1)
        tol = dict(atol=0.002, rtol=0.03) if device == "npu" else dict(atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual, ref, **tol)
        grad = torch.randn_like(actual)
        a = torch.autograd.grad(actual, (x, probs, u, d), grad)
        b = torch.autograd.grad(ref, (x, probs, up, down), grad)
        for i, (got, want) in enumerate(zip(a, b, strict=True)):
            if i >= 2:
                dist.all_reduce(want)
                want = want.chunk(2)[rank]
            torch.testing.assert_close(got, want, **tol)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))),
        pytest.param(
            "npu",
            marks=[
                pytest.mark.launch_plan(provider=LaunchSpec(nproc=2)),
                pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend"),
            ],
        ),
    ],
)
def test_two_chunk_ep(device):
    spawn_with_rendezvous(_worker, args=(device,), nprocs=2)
