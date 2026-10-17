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


def check_rowcopy_boundaries(entry):
    """Verify rank-major placement, offset output guards, and copy fail-stop behavior."""
    inputs = [torch.arange(5, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 20]
    expected = torch.empty((2, 5), dtype=torch.float32)
    torch._chunk_cat(inputs, 0, 2, out=expected)
    copy_stages = [f"copy_tensor_{index}_row_{row}" for index in range(2) for row in range(2)]
    for failure in (None, copy_stages[0], copy_stages[-1], "sync:" + copy_stages[1]):
        backing = torch.full((14,), -999.0)
        output = backing[2:12]
        captured = io.StringIO()
        error = RuntimeError(f"injected {failure}")
        copies = []

        def current_stage(captured=captured):
            return json.loads(captured.getvalue().splitlines()[-1].split(" ", 2)[2])["stage"]

        def synchronize(failure=failure, error=error):
            if failure == "sync:" + current_stage():
                raise error

        def copy(destination, source, failure=failure, error=error, copies=copies):
            copies.append(current_stage())
            assert destination.is_contiguous() and source.is_contiguous()
            assert destination.dtype == source.dtype == torch.float32
            if current_stage() == failure:
                raise error
            destination.copy_(source)

        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="rowcopy", VEOMNI_CHUNKCAT_SYNC="1"),
            patch.object(entry, "_device_context", return_value=(synchronize, "mock-stream")),
            patch.object(entry, "_copy_row", side_effect=copy),
            patch.object(torch, "cat", side_effect=AssertionError("rowcopy must not call cat")),
            contextlib.redirect_stdout(captured),
        ):
            try:
                assert entry.eager_reduce_scatter_copy_in(inputs, output, 2) is None
            except RuntimeError as caught:
                assert failure is not None and caught is error
            else:
                assert failure is None
                torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
        assert backing[:2].tolist() == [-999.0] * 2 and backing[12:].tolist() == [-999.0] * 2
        records = [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]
        if failure is None:
            assert copies == copy_stages
            assert records[-1]["stage"] == "post_sync" and records[-1]["event"] == "complete"
        else:
            stage = failure.removeprefix("sync:")
            assert copies == copy_stages[: copy_stages.index(stage) + 1]
            assert records[-1]["stage"] == stage and records[-1]["event"] == "failed"
            metadata = records[-1]["copy"]
            index, row = metadata["tensor_index"], metadata["row"]
            column = 0 if index == 0 else 3
            assert metadata["column_start"] == column
            assert metadata["destination"]["storage_offset"] == 2 + row * 5 + column
            assert metadata["destination"]["data_ptr"] == output.data_ptr() + (row * 5 + column) * 4
            assert metadata["source"]["dtype"] == metadata["destination"]["dtype"] == "torch.float32"
    print("PASS: CPU rowcopy placement, output guards, metadata, and operation/synchronization failure attribution")


def check_fresh_output_boundaries(entry):
    """Probe one call only; compare identical sources and preserve FSDP's output."""
    inputs = [torch.arange(5, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 20]
    expected = torch.empty((2, 5), dtype=torch.float32)
    torch._chunk_cat(inputs, 0, 2, out=expected)
    original_stages = [f"copy_tensor_{index}_row_{row}" for index in range(2) for row in range(2)]
    fresh_stages = ["fresh_" + stage for stage in original_stages]
    stages = ["allocate_fresh_output", *fresh_stages, *original_stages]
    failures = [None, *stages, "sync:allocate_fresh_output", "sync:" + fresh_stages[-1], "sync:" + original_stages[0]]
    for failure in failures:
        backing = torch.full((14,), -999.0)
        output = backing[2:12]
        captured = io.StringIO()
        error = RuntimeError(f"injected {failure}")
        calls, copies, allocations = [], [], []
        allocate = torch.empty_like

        def current_stage(captured=captured):
            return json.loads(captured.getvalue().splitlines()[-1].split(" ", 2)[2])["stage"]

        def synchronize(failure=failure, error=error):
            if failure == "sync:" + current_stage():
                raise error

        def allocate_output(
            out, failure=failure, error=error, calls=calls, allocate=allocate, allocations=allocations
        ):
            calls.append(current_stage())
            if failure == current_stage():
                raise error
            fresh = allocate(out)
            assert fresh.data_ptr() != out.data_ptr()
            allocations.append(fresh)
            return fresh

        def copy(destination, source, failure=failure, error=error, calls=calls, copies=copies):
            stage = current_stage()
            calls.append(stage)
            copies.append((stage, source.data_ptr(), destination.data_ptr(), source.numel()))
            if failure == stage:
                raise error
            destination.copy_(source)

        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="rowcopy_fresh", VEOMNI_CHUNKCAT_SYNC="1"),
            patch.dict(os.environ, VEOMNI_CHUNKCAT_PROBE_CALL="545"),
            patch.object(entry, "_call_count", 544),
            patch.object(entry, "_device_context", return_value=(synchronize, "mock-stream")),
            patch.object(entry, "_copy_row", side_effect=copy),
            patch.object(torch, "empty_like", side_effect=allocate_output),
            patch.object(torch, "cat", side_effect=AssertionError("fresh probe must not call cat")),
            contextlib.redirect_stdout(captured),
        ):
            try:
                assert entry.eager_reduce_scatter_copy_in(inputs, output, 2) is None
            except RuntimeError as caught:
                assert failure is not None and caught is error
            else:
                assert failure is None
                torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
                torch.testing.assert_close(allocations[0], expected, rtol=0, atol=0)
                for fresh_copy, original_copy in zip(copies[:4], copies[4:]):
                    assert fresh_copy[1] == original_copy[1]  # identical source addresses
                    assert fresh_copy[2] != original_copy[2]
                    assert fresh_copy[3] == original_copy[3]
        assert backing[:2].tolist() == [-999.0] * 2 and backing[12:].tolist() == [-999.0] * 2
        records = [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]
        if failure is None:
            assert calls == stages
            assert [r["target"] for r in records if r["event"] == "target_complete"] == ["fresh", "original"]
        else:
            stage = failure.removeprefix("sync:")
            assert calls == stages[: stages.index(stage) + 1]
            assert records[-1]["event"] == "failed" and records[-1]["stage"] == stage
            if stage.startswith("fresh_") or stage == "allocate_fresh_output":
                assert (output == -999).all()  # no original-buffer writes after fresh failure
            if stage.startswith("fresh_"):
                assert records[-1]["copy"]["target"] == "fresh"
            elif stage.startswith("copy_"):
                assert records[-1]["copy"]["target"] == "original"
        metadata = next(r for r in records if "inputs" in r)
        assert metadata["probe_call"] == 545 and metadata["probe_active"]

    for call_id in (544, 546):
        captured = io.StringIO()
        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="rowcopy_fresh", VEOMNI_CHUNKCAT_SYNC="1"),
            patch.dict(os.environ, VEOMNI_CHUNKCAT_PROBE_CALL="545"),
            patch.object(entry, "_call_count", call_id - 1),
            patch.object(torch, "empty_like", side_effect=AssertionError("non-probe call must not allocate")),
            contextlib.redirect_stdout(captured),
        ):
            assert entry.eager_reduce_scatter_copy_in(inputs, output, 2) is None
        torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
        records = [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]
        assert not any(r.get("probe_active") for r in records)
        assert not any(r["stage"].startswith("fresh_") for r in records)

    with patch.dict(os.environ, VEOMNI_CHUNKCAT_MODE="rowcopy_fresh"):
        for value in ("0", "-1", "bad"):
            with patch.dict(os.environ, VEOMNI_CHUNKCAT_PROBE_CALL=value):
                try:
                    entry._probe_call()
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid probe call accepted")
    print(
        f"PASS: CPU fresh-output probe; {len(failures)} boundary cases, identical sources, call gating, output guards"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    entry = load_entry()
    check_diagnostic_boundaries(entry)
    check_precast_boundaries(entry)
    check_rowcopy_boundaries(entry)
    check_fresh_output_boundaries(entry)
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
    rejected_layouts = 0
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
                        for mode in ("eager", "precast", "rowcopy", "rowcopy_fresh"):
                            actual.fill_(float("nan"))
                            with patch.dict(
                                os.environ,
                                VEOMNI_CHUNKCAT_MODE=mode,
                                VEOMNI_CHUNKCAT_SYNC="1",
                                VEOMNI_CHUNKCAT_PROBE_CALL=str(entry._call_count + 1),
                            ):
                                try:
                                    result = _fsdp_collectives.foreach_reduce_scatter_copy_in(
                                        device_inputs, actual, chunks
                                    )
                                except ValueError as error:
                                    assert mode in ("rowcopy", "rowcopy_fresh")
                                    assert noncontiguous and input_dtype == output_dtype
                                    assert str(error) == "rowcopy requires contiguous packed tensors and output"
                                    packed = entry._pad_chunk(inputs, dim=0, num_chunks=chunks)
                                    assert any(not t.is_contiguous() for t in packed)
                                    assert torch.isnan(actual.cpu()).all()
                                    rejected_layouts += 1
                                    continue
                            assert result is None
                            assert actual.data_ptr() == ptr
                            assert actual.dtype == output_dtype
                            torch.testing.assert_close(actual.cpu().view(chunks, width), expected, rtol=0, atol=0)
                            checked += 1
        assert torch._chunk_cat is original_chunk_cat
    finally:
        _fsdp_collectives.foreach_reduce_scatter_copy_in = original
    print(f"PASS: {checked} packing cases on {args.device}; exact values, dtype, buffer identity, and None return")
    print(f"PASS: {rejected_layouts} noncontiguous same-dtype rowcopy cases rejected before output writes")


if __name__ == "__main__":
    main()
