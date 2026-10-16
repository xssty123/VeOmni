"""Manual packing parity check. Defaults to CPU; NPU requires --device npu:0."""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import torch


def load_entry():
    path = Path(__file__).resolve().parents[2] / "tasks/train_vlm_chunkcat_eager.py"
    spec = importlib.util.spec_from_file_location("chunkcat_eager_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_diagnostic_boundaries(entry):
    """Inject failures on CPU to verify attribution and fail-stop behavior."""
    tensor = torch.arange(12, dtype=torch.float32)[2:10]
    output = torch.empty(8)
    real_pack = entry.eager_chunk_cat
    for failure in (None, "pre_sync", "pack", "post_sync", "disabled"):
        calls = []
        error = RuntimeError(f"injected {failure}")

        def synchronize(calls=calls, failure=failure, error=error):
            stage = "pre_sync" if not calls else "post_sync"
            calls.append(stage)
            if failure == stage:
                raise error

        def pack(*args, calls=calls, failure=failure, error=error, **kwargs):
            calls.append("pack")
            if failure == "pack":
                raise error
            return real_pack(*args, **kwargs)

        captured = io.StringIO()
        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_SYNC="0" if failure == "disabled" else "1"),
            patch.dict(os.environ, VEOMNI_CHUNKCAT_LOG_INPUTS="1"),
            patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="eager"),
            patch.object(entry, "_device_context", return_value=(synchronize, "mock-stream")) as context,
            patch.object(entry, "eager_chunk_cat", side_effect=pack),
            contextlib.redirect_stdout(captured),
        ):
            try:
                entry.eager_reduce_scatter_copy_in([tensor], output, 2)
            except RuntimeError as caught:
                assert failure in ("pre_sync", "pack", "post_sync") and caught is error
            else:
                assert failure in (None, "disabled")
                torch.testing.assert_close(output, tensor)
            context.assert_called_once()
        records = [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]
        metadata = next(r for r in records if "inputs" in r)
        assert metadata["inputs"][0]["data_ptr"] == tensor.data_ptr()
        assert metadata["inputs"][0]["storage_offset"] == 2
        assert metadata["stream"] == "mock-stream"
        failed = [r for r in records if r["event"] == "failed"]
        expected_calls = {
            None: ["pre_sync", "pack", "post_sync"],
            "pre_sync": ["pre_sync"],
            "pack": ["pre_sync", "pack"],
            "post_sync": ["pre_sync", "pack", "post_sync"],
            "disabled": ["pack"],
        }
        assert calls == expected_calls[failure]
        if failure in (None, "disabled"):
            assert not failed
        else:
            assert len(failed) == 1 and failed[0]["stage"] == failure
            assert records[-1] == failed[0]
    print("PASS: CPU diagnostic attribution, fail-stop, metadata, and sync opt-out")


def check_precast_boundaries(entry):
    """Check that each split operation and its synchronization can fail independently."""
    inputs = [torch.arange(5, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16)]
    expected = torch.empty((2, 5), dtype=torch.float32)
    torch._chunk_cat(inputs, 0, 2, out=expected)
    stages = ["pre_sync", "pad_and_reshape", "cast_tensor_0", "cast_tensor_1", "same_dtype_cat", "post_sync"]
    for failure in (None, "pad_and_reshape", "cast_tensor_0", "cast_tensor_1", "same_dtype_cat", "sync:cast_tensor_1"):
        output = torch.full((10,), float("nan"))
        captured = io.StringIO()
        error = RuntimeError(f"injected {failure}")

        def current_stage(captured=captured):
            return json.loads(captured.getvalue().splitlines()[-1].split(" ", 2)[2])["stage"]

        def synchronize(failure=failure, error=error):
            if failure == "sync:" + current_stage():
                raise error

        def inject(function, failure=failure, error=error):
            def invoke(*args, **kwargs):
                if current_stage() == failure:
                    raise error
                return function(*args, **kwargs)

            return invoke

        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="precast", VEOMNI_CHUNKCAT_SYNC="1"),
            patch.object(entry, "_device_context", return_value=(synchronize, "mock-stream")),
            patch.object(entry, "_pad_chunk", side_effect=inject(entry._pad_chunk)),
            patch.object(entry, "_cast_tensor", side_effect=inject(entry._cast_tensor)),
            patch.object(torch, "cat", side_effect=inject(torch.cat)),
            contextlib.redirect_stdout(captured),
        ):
            try:
                assert entry.eager_reduce_scatter_copy_in(inputs, output, 2) is None
            except RuntimeError as caught:
                assert failure is not None and caught is error
            else:
                assert failure is None
                torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
        records = [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]
        completed = [r["stage"] for r in records if r["event"] == "complete"]
        if failure is None:
            assert completed == stages
        else:
            failed_stage = failure.removeprefix("sync:")
            assert completed == stages[: stages.index(failed_stage)]
            assert records[-1]["event"] == "failed" and records[-1]["stage"] == failed_stage
            if failed_stage.startswith("cast_tensor_"):
                assert records[-1]["tensor_index"] == int(failed_stage.rsplit("_", 1)[1])
    print("PASS: CPU split-phase parity and operation/synchronization failure attribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    entry = load_entry()
    check_diagnostic_boundaries(entry)
    check_precast_boundaries(entry)
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    original = entry.install_chunkcat_eager()
    original_chunk_cat = torch._chunk_cat
    cases = [
        [(1,), (3,)],
        [(5,), (11,)],
        [(8,), (16,)],
        [(5, 3), (2, 7)],
        [(3, 2, 5), (7, 4, 2)],
        [(5,), (3, 2), (2, 3, 4)],
    ]
    checked = 0
    try:
        for shapes in cases:
            for noncontiguous in (False, True):
                for input_dtype, output_dtype in (
                    (torch.bfloat16, torch.float32),
                    (torch.bfloat16, torch.bfloat16),
                    (torch.float32, torch.float32),
                ):
                    inputs = []
                    for shape in shapes:
                        # Slicing makes even one-dimensional input noncontiguous.
                        x = (torch.arange(math.prod(shape) * 2, dtype=torch.float32) % 31).reshape(*shape, 2)
                        x = x.to(input_dtype)[..., 0]
                        inputs.append(x if noncontiguous else x.contiguous())
                    for chunks in (1, 2, 4, 8):
                        width = sum(math.ceil(x.shape[0] / chunks) * math.prod(x.shape[1:]) for x in inputs)
                        expected = torch.empty((chunks, width), dtype=output_dtype)
                        # Always use the native CPU implementation as the reference,
                        # including for NPU checks: never launch the failing NPU kernel.
                        original_chunk_cat(inputs, 0, chunks, out=expected)
                        actual = torch.full((chunks * width,), float("nan"), dtype=output_dtype, device=args.device)
                        ptr = actual.data_ptr()
                        device_inputs = [
                            torch.stack((x, x), dim=-1).to(args.device)[..., 0] if noncontiguous else x.to(args.device)
                            for x in inputs
                        ]
                        for mode in ("eager", "precast"):
                            actual.fill_(float("nan"))
                            with patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE=mode, VEOMNI_CHUNKCAT_SYNC="1"):
                                result = _fsdp_collectives.foreach_reduce_scatter_copy_in(
                                    device_inputs, actual, chunks
                                )
                            assert result is None
                            assert actual.data_ptr() == ptr
                            assert actual.dtype == output_dtype
                            torch.testing.assert_close(actual.cpu().view(chunks, width), expected, rtol=0, atol=0)
                            checked += 1
        assert torch._chunk_cat is original_chunk_cat
    finally:
        _fsdp_collectives.foreach_reduce_scatter_copy_in = original
    print(f"PASS: {checked} packing cases on {args.device}; exact values, dtype, buffer identity, and None return")


if __name__ == "__main__":
    main()
