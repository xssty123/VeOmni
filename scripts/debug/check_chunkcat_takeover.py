"""Exercise installed FSDP on CPU with simulated communication; never start training."""

import contextlib
import gc
import inspect
import io
import os
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from check_chunkcat_sizes import load_entry, records
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as param_group


def check_path(call_id, fail_copy=False):
    entry = load_entry()
    entry._call_count = call_id - 1
    inputs = [torch.arange(9, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 20]
    expected = torch.empty((2, 7))
    torch._chunk_cat(inputs, 0, 2, out=expected)
    allocations, communicated, order, events = [], [], [], []

    class Comm:
        def allocate(self, size, *, dtype, device):
            tensor = torch.full(size, -999, dtype=dtype, device=device)
            allocations.append(tensor)
            return tensor

        def __call__(self, *, output_tensor, input_tensor, group, op):
            assert not torch.is_grad_enabled()
            assert "wait_stream" in order
            assert not grads  # FSDP must release the source list only after copying.
            torch.testing.assert_close(input_tensor.view_as(expected), expected, rtol=0, atol=0)
            communicated.append(weakref.ref(input_tensor))
            output_tensor.copy_(input_tensor.view(2, -1).sum(0))
            order.append("communicate")

    def record_event():
        event = object()
        events.append(event)
        return event

    stream = SimpleNamespace(wait_stream=lambda other: order.append("wait_stream"), record_event=record_event)
    device_handle = SimpleNamespace(current_stream=lambda: stream, stream=lambda other: contextlib.nullcontext())
    params = [
        SimpleNamespace(
            fsdp_placement=SimpleNamespace(dim=0),
            sharded_size=(size,),
            contiguous_sharded_stride=(1,),
            sharded_param=SimpleNamespace(grad=None),
            offload_to_cpu=False,
            to_sharded_dtensor=lambda tensor: tensor,
        )
        for size in (5, 2)
    ]
    captured = io.StringIO()
    original = collectives.foreach_reduce
    original_copy = collectives.foreach_reduce_scatter_copy_in
    grads = list(inputs)
    with (
        patch.object(collectives, "_get_device_handle", return_value=device_handle),
        patch.object(collectives, "_get_gradient_divide_factors", return_value=(1, 1, None, None)),
        contextlib.redirect_stdout(captured),
    ):
        try:
            entry.install_chunkcat_eager()
            assert param_group.foreach_reduce is collectives.foreach_reduce
            assert inspect.unwrap(collectives.foreach_reduce).__code__ is not inspect.unwrap(original).__code__
            if fail_copy:
                entry._copy_row = lambda *args: (_ for _ in ()).throw(RuntimeError("injected fresh copy failure"))
            try:
                result = param_group.foreach_reduce(
                    fsdp_params=params,
                    unsharded_grads=grads,
                    reduce_scatter_group=SimpleNamespace(size=lambda: 2),
                    reduce_scatter_comm=Comm(),
                    reduce_scatter_stream=stream,
                    orig_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    device=torch.device("cpu"),
                    gradient_divide_factor=None,
                    all_reduce_group=None,
                    all_reduce_stream=stream,
                    all_reduce_grads=False,
                    partial_reduce_output=None,
                    all_reduce_hook=None,
                )
            except RuntimeError as error:
                assert fail_copy and str(error) == "injected fresh copy failure"
                assert not communicated and len(grads) == 2
                print("EXPECTED_COPY_FAILURE")
                return
            assert not fail_copy
        finally:
            collectives.foreach_reduce = param_group.foreach_reduce = original
            collectives.foreach_reduce_scatter_copy_in = original_copy

    output = result[0]
    assert output is communicated[0]() and result[1] is events[0]
    assert output.shape == (14,) and output.dtype == torch.float32 and output.is_contiguous()
    torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
    actual_grad = torch.cat([p.sharded_param.grad for p in params])
    torch.testing.assert_close(actual_grad.float(), expected.sum(0), rtol=0, atol=0)
    # Exclude the install banner from JSON parsing.
    json_lines = io.StringIO(
        "\n".join(x for x in captured.getvalue().splitlines() if not x.startswith("[chunkcat-eager] installed"))
    )
    log = records(json_lines)
    selected = [r for r in log if r["event"] == "takeover_selected"]
    if call_id == 545:
        assert len(selected) == 1
        assert output.data_ptr() != allocations[0].data_ptr()
        assert (allocations[0] == -999).all()  # No writes to original at all.
        assert [r["target"] for r in log if r["event"] == "target_complete"] == ["fresh"]
        for event in ("communication_input", "lifetime_input"):
            assert next(r["input"]["data_ptr"] for r in log if r["event"] == event) == output.data_ptr()
    else:
        assert output is allocations[0] and not selected
    pointer = weakref.ref(output)
    allocations.clear()
    del output
    gc.collect()
    assert pointer() is result[0]  # Only FSDP's returned tuple must retain the input.
    del result
    gc.collect()
    assert pointer() is None  # No diagnostic strong-reference leak.
    print(f"PASS: call {call_id}, packed values, actual communication input, gradient output, event and lifetime")


def main():
    with patch.dict(
        os.environ,
        {
            "VEOMNI_CHUNKCAT_MODE": "rowcopy_takeover",
            "VEOMNI_CHUNKCAT_SYNC": "1",
            "VEOMNI_CHUNKCAT_PROBE_CALL": "545",
            "VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES": "0",
            "VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES": "0",
        },
    ):
        for call_id in (544, 545, 546):
            check_path(call_id)
        check_path(545, fail_copy=True)
        entry = load_entry()
        for key in ("VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES", "VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES"):
            with patch.dict(os.environ, {key: "64"}):
                try:
                    entry._size_options()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Takeover must reject size experiments")
    print("PASS: fail-stop and size guards; CPU only, simulated communication, no training or NPU operation")


if __name__ == "__main__":
    main()
