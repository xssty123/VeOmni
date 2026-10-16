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
    if mode not in ("eager", "precast"):
        raise ValueError("VEOMNI_CHUNKCAT_MODE must be eager or precast")
    if mode == "precast" and os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") != "1":
        raise ValueError("precast diagnosis requires VEOMNI_CHUNKCAT_SYNC=1")
    return mode


def _cast_tensor(tensor, dtype):
    return tensor.to(dtype=dtype)


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


def eager_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size) -> None:
    """Preserve FSDP's output buffer and packing order using torch's decomposition."""
    global _call_count
    _call_count += 1
    call_id = _call_count
    mode = _packing_mode()
    sync_enabled = os.environ.get("VEOMNI_CHUNKCAT_SYNC", "1") == "1"
    verbose = call_id == 1 or os.environ.get("VEOMNI_CHUNKCAT_LOG_INPUTS") == "1" or sync_enabled
    stage = "metadata"
    snapshot = {"num_chunks": world_size, "sync_enabled": sync_enabled, "packing_mode": mode}

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
        if mode == "precast":
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
            run_phase("same_dtype_cat", torch.cat, converted, dim=1, out=out)
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
        f"boundary_sync={os.environ.get('VEOMNI_CHUNKCAT_SYNC', '1')} schema=3",
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
