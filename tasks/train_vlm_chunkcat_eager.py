"""Opt-in A5 diagnostic entry point; does not modify the installed torch files."""

import inspect
import json
import os
import time

import torch
from torch._decomp.decompositions import _chunk_cat as eager_chunk_cat
from torch._decomp.decompositions import _pad_chunk


_call_count = 0


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
    probe_call = _probe_call() if mode == "rowcopy_fresh" else None
    sync_enabled = os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") == "1"
    verbose = call_id == 1 or os.environ.get("VEOMNI_CHUNKCAT_LOG_INPUTS") == "1" or sync_enabled
    stage = "metadata"
    snapshot = {"num_chunks": world_size, "sync_enabled": sync_enabled, "packing_mode": mode}
    if probe_call is not None:
        snapshot.update(probe_call=probe_call, probe_active=call_id == probe_call)

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
    _fsdp_collectives.foreach_reduce_scatter_copy_in = eager_reduce_scatter_copy_in
    print(
        f"[chunkcat-eager] installed rank={os.environ.get('RANK', '?')} torch={torch.__version__} "
        f"packing_mode={mode} "
        f"boundary_sync={os.environ.get('VEOMNI_CHUNKCAT_SYNC', '1')} probe_call={probe_call} schema=6",
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
