"""Opt-in A5 diagnostic entry point; does not modify the installed torch files."""

import inspect
import os

import torch
from torch._decomp.decompositions import _chunk_cat as eager_chunk_cat


_call_count = 0


def eager_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size) -> None:
    """Preserve FSDP's output buffer and packing order using torch's decomposition."""
    global _call_count
    _call_count += 1
    out = reduce_scatter_input.view(world_size, -1)
    log_inputs = os.environ.get("VEOMNI_CHUNKCAT_LOG_INPUTS") == "1"

    def report(event):
        # Metadata only: never copy gradient values to the host to produce a log.
        inputs = [(tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in unsharded_grads]
        print(
            f"[chunkcat-eager] {event} rank={os.environ.get('RANK', '?')} call={_call_count} "
            f"num_chunks={world_size} inputs={inputs} out_shape={tuple(out.shape)} out_dtype={out.dtype}",
            flush=True,
        )

    if _call_count == 1 or log_inputs:
        report("begin")
    try:
        eager_chunk_cat(unsharded_grads, dim=0, num_chunks=world_size, out=out)
    except Exception:
        report("failed")
        raise
    if log_inputs:
        # Submission completed; asynchronous device execution may still be pending.
        print(f"[chunkcat-eager] submitted call={_call_count}", flush=True)


def install_chunkcat_eager():
    """Replace only the FSDP copy-in helper, independently in each worker process."""
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
        "implementation=torch._decomp.decompositions._chunk_cat",
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
