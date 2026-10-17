"""CPU-only checks for opt-in copy-size and allocation-size probes; no training."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ENTRY = Path(__file__).resolve().parents[2] / "tasks/train_vlm_chunkcat_eager.py"


def load_entry():
    spec = importlib.util.spec_from_file_location("size_probe", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def records(captured):
    return [json.loads(line.split(" ", 2)[2]) for line in captured.getvalue().splitlines()]


def check_copy_sizes():
    inputs = [torch.arange(9, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 20]
    expected = torch.empty((2, 7))
    torch._chunk_cat(inputs, 0, 2, out=expected)
    count = 0
    for chunk_bytes in (4, 8, 12, 64):
        for call_id in (544, 545, 546):
            entry = load_entry()
            backing = torch.full((18,), -999.0)
            output = backing[2:16]
            captured = io.StringIO()
            with (
                patch.dict(os.environ, VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES=str(chunk_bytes)),
                patch.object(entry, "_call_count", call_id - 1),
                contextlib.redirect_stdout(captured),
            ):
                assert entry.eager_reduce_scatter_copy_in(inputs, output, 2) is None
            torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
            assert (backing[:2] == -999).all() and (backing[16:] == -999).all()
            rows = records(captured)
            parts = [r for r in rows if r["event"] == "buffers" and "row_byte_start" in r]
            if call_id != 545:
                assert not parts
                continue
            expected_starts = list(range(0, 20, chunk_bytes))
            assert [r["row_byte_start"] for r in parts] == expected_starts
            assert [r["row_byte_end"] for r in parts] == [min(n + chunk_bytes, 20) for n in expected_starts]
            for r in parts:
                assert r["destination"]["data_ptr"] == output.data_ptr() + 28 + r["row_byte_start"]
                assert r["source"]["bytes"] == r["destination"]["bytes"] == r["row_byte_end"] - r["row_byte_start"]
            fresh_done = next(i for i, r in enumerate(rows) if r["event"] == "target_complete")
            first_part = next(i for i, r in enumerate(rows) if "row_byte_start" in r)
            assert fresh_done < first_part
            assert [r["target"] for r in rows if r["event"] == "target_complete"] == ["fresh", "original"]
            count += 1

    for failure in ("copy", "sync"):
        entry = load_entry()
        captured = io.StringIO()
        calls = []
        error = RuntimeError("injected chunk failure")

        def stage(captured=captured):
            return records(captured)[-1]["stage"]

        def copy(destination, source, calls=calls, failure=failure, error=error):
            calls.append(stage())
            if failure == "copy" and stage().endswith("_bytes_8_16"):
                raise error
            destination.copy_(source)

        def synchronize(failure=failure, error=error):
            if failure == "sync" and stage().endswith("_bytes_8_16"):
                raise error

        with (
            patch.dict(os.environ, VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES="8"),
            patch.object(entry, "_call_count", 544),
            patch.object(entry, "_copy_row", side_effect=copy),
            patch.object(entry, "_device_context", return_value=(synchronize, "mock")),
            contextlib.redirect_stdout(captured),
        ):
            try:
                entry.eager_reduce_scatter_copy_in(inputs, torch.empty(14), 2)
            except RuntimeError as caught:
                assert caught is error
            else:
                raise AssertionError("chunk failure was swallowed")
        rows = records(captured)
        assert calls[-1] == "copy_tensor_0_row_1_bytes_8_16"
        assert rows[-1]["event"] == "failed" and rows[-1]["stage"] == calls[-1]
        assert rows[-1]["copy_part"]["row_byte_start"] == 8
        submitted = any(r["event"] == "submitted" and r["stage"] == calls[-1] for r in rows)
        assert submitted == (failure == "sync")
    print(f"PASS: copy sizes ({count} selected cases), neighboring calls, exact placement, guards, and fail-stop")


class DefaultAllocMixin:
    def allocate(self, size, *, dtype, device):
        return torch.empty(*size, dtype=dtype, device=device)


class DefaultReduceScatter(DefaultAllocMixin):
    pass


def check_allocation_sizes():
    inputs = [torch.arange(9, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 20]
    expected = torch.empty((2, 7))
    torch._chunk_cat(inputs, 0, 2, out=expected)
    for extra_bytes in (4, 64, 256):
        entry = load_entry()
        outputs = []

        def foreach_reduce(reduce_scatter_comm, outputs=outputs, entry=entry):
            reduce_scatter_input = reduce_scatter_comm.allocate((14,), dtype=torch.float32, device="cpu")
            outputs.append(reduce_scatter_input)
            assert entry.eager_reduce_scatter_copy_in(inputs, reduce_scatter_input, 2) is None
            communication_output = reduce_scatter_comm.allocate((7,), dtype=torch.float32, device="cpu")
            assert communication_output.untyped_storage().nbytes() == 28
            return reduce_scatter_input

        collectives = SimpleNamespace(
            DefaultReduceScatter=DefaultReduceScatter,
            DefaultAllocMixin=DefaultAllocMixin,
            foreach_reduce=foreach_reduce,
        )
        captured = io.StringIO()
        with (
            patch.dict(
                os.environ, VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES="0", VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES=str(extra_bytes)
            ),
            contextlib.redirect_stdout(captured),
        ):
            entry._install_allocation_size_probe(collectives, extra_bytes)
            try:
                # A call outside foreach_reduce must not be enlarged, even at the selected count.
                entry._call_count = 544
                unrelated = DefaultReduceScatter().allocate((14,), dtype=torch.float32, device="cpu")
                assert unrelated.untyped_storage().nbytes() == 56
                for call_id in (544, 545, 546):
                    entry._call_count = call_id - 1
                    output = foreach_reduce(DefaultReduceScatter())
                    torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)
                    assert output.shape == (14,) and output.storage_offset() == 0
                    assert output.untyped_storage().nbytes() == 56 + (extra_bytes if call_id == 545 else 0)
                    assert output.data_ptr() == output.untyped_storage().data_ptr()
                    # The larger backing remains alive through the logical view.
                    output.add_(1)
                    torch.testing.assert_close(output.view_as(expected), expected + 1, rtol=0, atol=0)
                assert entry._allocation_probe[0]() is outputs[1]
            finally:
                del DefaultReduceScatter.allocate
        rows = records(captured)
        resized = [r for r in rows if r.get("event") == "allocation_context" and r["call"] == 545]
        assert len(resized) == 1 and resized[0]["out_storage"]["nbytes"] == 56 + extra_bytes
        assert sum("requested_storage_bytes" in r and "storage" in r for r in rows) == 1
    print("PASS: allocation capacities, original caller scope, logical shape, retained storage, exact packing")

    entry = load_entry()
    with (
        patch.dict(os.environ, VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES="64"),
        patch.object(entry, "_call_count", 544),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        try:
            entry.eager_reduce_scatter_copy_in(inputs, torch.empty(14), 2)
        except RuntimeError as error:
            assert "did not come from" in str(error)
        else:
            raise AssertionError("silently accepted unmodified allocation")
    print("PASS: allocation hook bypass rejected before packing")


def check_validation():
    entry = load_entry()
    for chunk, extra in (("8", "64"), ("-1", "0"), ("0", "-1"), ("bad", "0")):
        with patch.dict(os.environ, VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES=chunk, VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES=extra):
            try:
                entry._size_options()
            except ValueError:
                pass
            else:
                raise AssertionError("invalid size options accepted")
    print("PASS: mutually exclusive controls and invalid values")


def check_installed_contract():
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    entry = load_entry()
    original_copy = _fsdp_collectives.foreach_reduce_scatter_copy_in
    original_allocate = _fsdp_collectives.DefaultReduceScatter.allocate
    assert "allocate" not in vars(_fsdp_collectives.DefaultReduceScatter)
    with patch.dict(os.environ, VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES="64"):
        try:
            entry.install_chunkcat_eager()
            output = _fsdp_collectives.DefaultReduceScatter().allocate((7,), dtype=torch.float32, device="cpu")
            assert output.untyped_storage().nbytes() == 28
            assert _fsdp_collectives.DefaultAllocMixin.allocate is original_allocate
        finally:
            _fsdp_collectives.foreach_reduce_scatter_copy_in = original_copy
            if "allocate" in vars(_fsdp_collectives.DefaultReduceScatter):
                del _fsdp_collectives.DefaultReduceScatter.allocate
    print("PASS: installed torch FSDP signature/source guard and process-local hook installation")


def main():
    # Never inherit an experiment switch from the invoking shell during CPU validation.
    with patch.dict(
        os.environ,
        {
            "VEOMNI_CHUNKCAT_MODE": "rowcopy_fresh",
            "VEOMNI_CHUNKCAT_SYNC": "1",
            "VEOMNI_CHUNKCAT_PROBE_CALL": "545",
            "VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES": "0",
            "VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES": "0",
            "VEOMNI_CHUNKCAT_COPY_SLICE": "0:1",
        },
    ):
        check_validation()
        check_copy_sizes()
        check_allocation_sizes()
        check_installed_contract()
    print("PASS: CPU size diagnostics only; no NPU operation or training executed")


if __name__ == "__main__":
    main()
