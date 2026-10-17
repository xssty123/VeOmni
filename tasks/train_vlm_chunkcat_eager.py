"""Opt-in A5 diagnostic entry point; does not modify the installed torch files."""

import ast
import inspect
import json
import math
import os
import textwrap
import time
import weakref

import torch
from torch._decomp.decompositions import _chunk_cat as eager_chunk_cat
from torch._decomp.decompositions import _pad_chunk


_call_count = 0
_allocation_probe = None


def _size_options():
    chunk = int(os.environ.get("VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES", "0"))
    extra = int(os.environ.get("VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES", "0"))
    if chunk < 0 or extra < 0 or (chunk and extra):
        raise ValueError("Copy chunk and allocation extra bytes must be nonnegative and mutually exclusive")
    if (chunk or extra) and _packing_mode() != "rowcopy_fresh":
        raise ValueError("Size probes require rowcopy_fresh with synchronization")
    return chunk, extra


def _copy_slice():
    value = os.environ.get("VEOMNI_CHUNKCAT_COPY_SLICE")
    if value is not None:
        parts = value.split(":")
        if len(parts) != 2 or any(not part.isdecimal() for part in parts):
            raise ValueError("VEOMNI_CHUNKCAT_COPY_SLICE must be tensor_index:row_index")
        return tuple(map(int, parts))
    rank = int(os.environ.get("RANK", "-1"))
    if not 0 <= rank < 8:
        raise ValueError("Default copy slice requires rank 0..7; otherwise set VEOMNI_CHUNKCAT_COPY_SLICE")
    return (7, 5) if rank < 4 else (8, 4)


def _install_allocation_size_probe(collectives, extra_bytes):
    """Enlarge one original allocation, returning only its original logical view."""
    communication = collectives.DefaultReduceScatter
    original = communication.allocate
    if original is not collectives.DefaultAllocMixin.allocate:
        raise RuntimeError("Allocation size probe requires unpatched DefaultAllocMixin.allocate")
    if list(inspect.signature(original).parameters) != ["self", "size", "dtype", "device"]:
        raise RuntimeError("Unexpected FSDP allocation signature")
    body = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0].body
    expected = ast.parse("return torch.empty(*size, dtype=dtype, device=device)").body
    if [ast.dump(node) for node in body] != [ast.dump(node) for node in expected]:
        raise RuntimeError("Unexpected FSDP allocation implementation; refusing to replace it")
    probe_call = _probe_call()
    foreach_code = collectives.foreach_reduce.__code__

    def allocate(self, size, *, dtype, device):
        global _allocation_probe
        if _call_count + 1 != probe_call:
            return original(self, size, dtype=dtype, device=device)
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        try:
            selected = (
                type(self) is communication
                and caller is not None
                and caller.f_code is foreach_code
                and caller.f_locals.get("reduce_scatter_comm") is self
                # foreach_reduce also allocates the communication output after copy-in.
                and "reduce_scatter_input" not in caller.f_locals
            )
        finally:
            del caller, frame
        if not selected:
            return original(self, size, dtype=dtype, device=device)
        if _allocation_probe is not None:
            raise RuntimeError("Selected allocation was invoked more than once")
        shape = tuple(int(n) for n in size)
        numel = math.prod(shape)
        element_bytes = torch.empty((), device="meta", dtype=dtype).element_size()
        if numel <= 0 or extra_bytes % element_bytes:
            raise ValueError("Allocation probe requires nonempty output and element-aligned extra bytes")
        record = {
            "rank": os.environ.get("RANK", "?"),
            "call": probe_call,
            "logical_shape": shape,
            "logical_bytes": numel * element_bytes,
            "extra_bytes": extra_bytes,
            "requested_storage_bytes": numel * element_bytes + extra_bytes,
        }
        print(f"[chunkcat-eager] allocation_resize_begin {json.dumps(record)}", flush=True)
        try:
            backing = original(self, (numel + extra_bytes // element_bytes,), dtype=dtype, device=device)
            output = backing[:numel].view(shape)
            record["storage"] = _storage_metadata(output)
        except Exception as error:
            print(
                f"[chunkcat-eager] allocation_resize_failed {json.dumps({**record, 'message': str(error)})}",
                flush=True,
            )
            raise
        # The view owns the storage; do not extend its lifetime with a global strong reference.
        _allocation_probe = (weakref.ref(output), record)
        print(f"[chunkcat-eager] allocation_resize_returned {json.dumps(record)}", flush=True)
        return output

    communication.allocate = allocate
    return original


def _packing_mode():
    mode = os.environ.get("VEOMNI_CHUNKCAT_MODE", "eager")
    if mode not in ("eager", "precast", "rowcopy", "rowcopy_fresh"):
        raise ValueError("VEOMNI_CHUNKCAT_MODE must be eager, precast, rowcopy, or rowcopy_fresh")
    if mode in ("precast", "rowcopy", "rowcopy_fresh") and os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") != "1":
        raise ValueError(f"{mode} diagnosis requires VEOMNI_CHUNKCAT_SYNC=1")
    return mode


def _probe_call():
    call = int(os.environ.get("VEOMNI_CHUNKCAT_PROBE_CALL", "545"))
    if call < 1:
        raise ValueError("VEOMNI_CHUNKCAT_PROBE_CALL must be a positive integer")
    return call


def _cast_tensor(tensor, dtype):
    return tensor.to(dtype=dtype)


def _copy_row(destination, source):
    destination.copy_(source)


def _tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _device_context(device):
    if device.type == "cpu":
        return None, None
    if device.type != "npu":
        raise ValueError(f"This diagnostic supports CPU/NPU only, found {device}")
    # The training runtime registers torch.npu before the first FSDP call.
    stream = repr(torch.npu.current_stream(device))
    return lambda: torch.npu.synchronize(device), stream


def _storage_metadata(tensor):
    """Describe storage bounds without reading tensor contents or allocating device memory."""
    storage = tensor.untyped_storage()
    start = tensor.data_ptr()
    base = storage.data_ptr()
    size = storage.nbytes()
    contiguous = tensor.is_contiguous()
    end = start + tensor.numel() * tensor.element_size() if contiguous else None
    return {
        "base_ptr": base,
        "nbytes": size,
        "tensor_ptr": start,
        "contiguous": contiguous,
        "logical_end_ptr": end,
        "within_storage": base <= start <= end <= base + size if contiguous else None,
    }


def _allocation_context(output):
    """Inspect only the direct FSDP caller; never invoke or replace its allocator."""
    frame = inspect.currentframe()
    caller = None
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        if caller is None:
            return {"available": False, "reason": "no_caller_frame"}
        if caller.f_code.co_name != "foreach_reduce" or caller.f_locals.get("reduce_scatter_input") is not output:
            return {"available": False, "reason": "not_direct_fsdp_caller"}
        comm = caller.f_locals.get("reduce_scatter_comm")
        if comm is None:
            return {"available": False, "reason": "no_reduce_scatter_comm"}
        allocate = comm.allocate
        function = getattr(allocate, "__func__", allocate)
        code = getattr(function, "__code__", None)
        flags = vars(comm) if hasattr(comm, "__dict__") else {}
        return {
            "available": True,
            "comm_type": f"{type(comm).__module__}.{type(comm).__qualname__}",
            "allocate_module": getattr(function, "__module__", None),
            "allocate_qualname": getattr(function, "__qualname__", None),
            "allocate_file": code.co_filename if code is not None else None,
            "allocate_line": code.co_firstlineno if code is not None else None,
            "use_mem_cache": flags.get("_use_mem_cache") if isinstance(flags.get("_use_mem_cache"), bool) else None,
            "mem_cache_flag": flags.get("_mem_cache_flag") if isinstance(flags.get("_mem_cache_flag"), bool) else None,
        }
    finally:
        # Do not retain frames, comm objects, or their live tensors across calls.
        del caller, frame


def eager_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size) -> None:
    """Preserve FSDP's output buffer and packing order using torch's decomposition."""
    global _call_count
    _call_count += 1
    call_id = _call_count
    mode = _packing_mode()
    chunk_bytes, extra_bytes = _size_options()
    probe_call = _probe_call() if mode == "rowcopy_fresh" else None
    sync_enabled = os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") == "1"
    verbose = call_id == 1 or os.environ.get("VEOMNI_CHUNKCAT_LOG_INPUTS") == "1" or sync_enabled
    stage = "metadata"
    snapshot = {"num_chunks": world_size, "sync_enabled": sync_enabled, "packing_mode": mode}
    if probe_call is not None:
        snapshot.update(probe_call=probe_call, probe_active=call_id == probe_call)
    if chunk_bytes or extra_bytes:
        snapshot.update(copy_chunk_bytes=chunk_bytes, alloc_extra_bytes=extra_bytes)

    def report(event, **details):
        # Use cached Python data only, including after a device failure.
        record = {
            "event": event,
            "rank": os.environ.get("RANK", "?"),
            "pid": os.getpid(),
            "call": call_id,
            "stage": stage,
            "time": time.time(),
            **details,
        }
        print(f"[chunkcat-eager] {event} {json.dumps(record)}", flush=True)

    def run_phase(name, function, *args, **kwargs):
        nonlocal stage
        stage = name
        report("begin")
        result = function(*args, **kwargs)
        report("submitted")
        if synchronize:
            synchronize()
        report("complete")
        return result

    try:
        out = reduce_scatter_input.view(world_size, -1)
        snapshot["inputs"] = [_tensor_metadata(t) for t in unsharded_grads]
        snapshot["out"] = _tensor_metadata(out)
        if call_id == probe_call:
            stage = "allocation_metadata"
            snapshot["allocation_context"] = _allocation_context(reduce_scatter_input)
            snapshot["out_storage"] = _storage_metadata(out)
            report("allocation_context", **snapshot["allocation_context"], out_storage=snapshot["out_storage"])
            if extra_bytes:
                if _allocation_probe is None or _allocation_probe[0]() is not reduce_scatter_input:
                    raise RuntimeError("Selected output did not come from the allocation size probe")
                snapshot["allocation_resize"] = _allocation_probe[1]
                if snapshot["out_storage"]["nbytes"] != out.numel() * out.element_size() + extra_bytes:
                    raise RuntimeError("Selected output storage does not match requested enlarged capacity")
        stage = "stream_context"
        synchronize, stream = _device_context(out.device)
        snapshot.update(stream=stream, sync_scope="device" if synchronize and sync_enabled else "none")
        if verbose:
            report("begin", **snapshot)
        if sync_enabled and synchronize:
            stage = "pre_sync"
            report("begin")
            synchronize()
            report("complete")
        if mode in ("precast", "rowcopy", "rowcopy_fresh"):
            packed = run_phase("pad_and_reshape", _pad_chunk, unsharded_grads, dim=0, num_chunks=world_size)
            snapshot["packed_inputs"] = [_tensor_metadata(t) for t in packed]
            snapshot["converted"] = []
            report("buffers", **snapshot)
            converted = []
            for index, tensor in enumerate(packed):
                snapshot["tensor_index"] = index
                value = run_phase(f"cast_tensor_{index}", _cast_tensor, tensor, out.dtype)
                converted.append(value)
                snapshot["converted"].append(_tensor_metadata(value))
                report("buffers", tensor_index=index, converted=snapshot["converted"][-1])
            snapshot.pop("tensor_index", None)
            if mode == "precast":
                run_phase("same_dtype_cat", torch.cat, converted, dim=1, out=out)
            else:
                stage = "validate_copy_layout"
                if not out.is_contiguous() or any(
                    t.ndim != 2 or t.shape[0] != world_size or not t.is_contiguous() for t in converted
                ):
                    raise ValueError("rowcopy requires contiguous packed tensors and output")
                if sum(t.shape[1] for t in converted) != out.shape[1]:
                    raise ValueError("Packed widths do not cover the FSDP output")
                selected_slice = _copy_slice() if chunk_bytes and call_id == probe_call else None
                if selected_slice is not None:
                    index, row = selected_slice
                    if index >= len(converted) or row >= world_size or converted[index].shape[1] == 0:
                        raise ValueError("Selected copy slice does not exist in this packing call")
                    if chunk_bytes % out.element_size():
                        raise ValueError("Copy chunk bytes must be a multiple of output element size")
                targets = [("original", out)]
                if call_id == probe_call:
                    fresh = run_phase("allocate_fresh_output", torch.empty_like, out)
                    stage = "validate_fresh_output"
                    snapshot["fresh_out"] = _tensor_metadata(fresh)
                    snapshot["fresh_storage"] = _storage_metadata(fresh)
                    if fresh.data_ptr() == out.data_ptr():
                        raise ValueError("Fresh output must have a different address from the live FSDP output")
                    report("buffers", **snapshot)
                    # Keep both outputs alive and replay identical sources/copy sizes.
                    # The original output must still be filled for FSDP's caller.
                    targets.insert(0, ("fresh", fresh))
                for target_name, target in targets:
                    offset = 0
                    for index, tensor in enumerate(converted):
                        width = tensor.shape[1]
                        for row in range(world_size):
                            prefix = "fresh_" if target_name == "fresh" else ""
                            stage = f"{prefix}copy_tensor_{index}_row_{row}"
                            source = tensor[row]
                            destination = target[row, offset : offset + width]
                            snapshot["copy"] = {
                                "target": target_name,
                                "tensor_index": index,
                                "row": row,
                                "column_start": offset,
                                "column_end": offset + width,
                                "source": _tensor_metadata(source),
                                "destination": _tensor_metadata(destination),
                            }
                            report("buffers", **snapshot["copy"])
                            if target_name == "original" and selected_slice == (index, row):
                                row_stage = stage
                                chunk_elements = chunk_bytes // out.element_size()
                                for start in range(0, width, chunk_elements):
                                    end = min(start + chunk_elements, width)
                                    part_source = source[start:end]
                                    part_destination = destination[start:end]
                                    stage = (
                                        f"{row_stage}_bytes_{start * out.element_size()}_{end * out.element_size()}"
                                    )
                                    snapshot["copy_part"] = {
                                        "row_byte_start": start * out.element_size(),
                                        "row_byte_end": end * out.element_size(),
                                        "source": _tensor_metadata(part_source),
                                        "destination": _tensor_metadata(part_destination),
                                    }
                                    report("buffers", **snapshot["copy_part"])
                                    run_phase(stage, _copy_row, part_destination, part_source)
                                snapshot.pop("copy_part")
                                del part_destination, part_source
                                stage = row_stage
                                report("split_row_complete", total_bytes=width * out.element_size())
                            else:
                                run_phase(stage, _copy_row, destination, source)
                        offset += width
                    if call_id == probe_call:
                        report("target_complete", target=target_name)
                snapshot.pop("copy", None)
                del source, destination, target, targets
                if call_id == probe_call:
                    del fresh
            del tensor, value, converted, packed
        else:
            stage = "pack"
            if verbose:
                report("begin")
            eager_chunk_cat(unsharded_grads, dim=0, num_chunks=world_size, out=out)
            if verbose:
                report("submitted")
        if sync_enabled and synchronize:
            stage = "post_sync"
            report("begin")
            synchronize()
            report("complete")
    except Exception as error:
        report("failed", error_type=type(error).__name__, message=str(error), **snapshot)
        raise


def install_chunkcat_eager():
    """Replace only the FSDP copy-in helper, independently in each worker process."""
    if os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") not in ("0", "1"):
        raise ValueError("VEOMNI_CHUNKCAT_SYNC must be 0 or 1")
    mode = _packing_mode()
    chunk_bytes, extra_bytes = _size_options()
    if chunk_bytes:
        _copy_slice()
    probe_call = _probe_call() if mode == "rowcopy_fresh" else None
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "10"]:
        raise RuntimeError(f"This private FSDP diagnostic targets torch 2.10, found {torch.__version__}")

    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    original = _fsdp_collectives.foreach_reduce_scatter_copy_in
    if original is eager_reduce_scatter_copy_in:
        return original
    expected = ["unsharded_grads", "reduce_scatter_input", "world_size"]
    if list(inspect.signature(original).parameters) != expected:
        raise RuntimeError("FSDP copy-in signature changed; refusing to install the diagnostic replacement")
    if extra_bytes:
        _install_allocation_size_probe(_fsdp_collectives, extra_bytes)
    _fsdp_collectives.foreach_reduce_scatter_copy_in = eager_reduce_scatter_copy_in
    print(
        f"[chunkcat-eager] installed rank={os.environ.get('RANK', '?')} torch={torch.__version__} "
        f"packing_mode={mode} "
        f"boundary_sync={os.environ.get('VEOMNI_CHUNKCAT_SYNC', '1')} probe_call={probe_call} schema=7 "
        f"copy_chunk_bytes={chunk_bytes} alloc_extra_bytes={extra_bytes}",
        flush=True,
    )
    return original


if __name__ == "__main__":
    install_chunkcat_eager()

    from veomni.arguments import parse_args
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer

    args = parse_args(VeOmniVLMArguments)
    trainer = VLMTrainer(args)
    trainer.train()
