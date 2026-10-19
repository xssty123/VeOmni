"""Bounded, per-rank observation of native FSDP copy-in (private A5 diagnostic)."""

import contextvars
import functools
import inspect
import json
import os
import time
from pathlib import Path


def tensor_metadata(tensor):
    """Read host metadata only: no tensor values, clones, or device synchronization."""
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "nbytes": tensor.numel() * tensor.element_size(),
    }


def callable_name(function):
    return f"{function.__module__}.{function.__qualname__}"


class FSDPObserver:
    """Keep only Python metadata between calls; never retain a tensor or event."""

    def __init__(self, directory):
        rank = os.environ.get("RANK", "unknown")
        self.path = Path(directory) / f"rank-{rank}-pid-{os.getpid()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", buffering=1)
        self.rank = rank
        self.calls = 0
        self.updates = 0
        self.groups = {}
        self.bytes_written = 0
        self.limit = 8 * 1024 * 1024
        self.context = contextvars.ContextVar("fsdp_observer", default=None)

    def emit(self, event, **details):
        if self.bytes_written >= self.limit:
            return
        row = {
            "event": event,
            "rank": self.rank,
            "time_ns": time.time_ns(),
            "completed_optimizer_steps": self.updates,
            **details,
        }
        line = json.dumps(row) + "\n"
        if self.bytes_written + len(line.encode()) >= self.limit:
            line = json.dumps({"event": "trace_limit", "limit_bytes": self.limit}) + "\n"
            self.bytes_written = self.limit
        else:
            self.bytes_written += len(line.encode())
        self.file.write(line)

    def install(self, collectives, param_group):
        original_reduce = collectives.foreach_reduce
        original_copy = collectives.foreach_reduce_scatter_copy_in
        if param_group.foreach_reduce is not original_reduce:
            raise RuntimeError("FSDP aliases differ; do not combine observation with eager/takeover patches")
        signature = inspect.signature(original_reduce)
        required = {"fsdp_params", "unsharded_grads", "reduce_scatter_group", "reduce_scatter_comm"}
        if not required.issubset(signature.parameters):
            raise RuntimeError("Unsupported FSDP foreach_reduce signature")
        if tuple(inspect.signature(original_copy).parameters) != (
            "unsharded_grads",
            "reduce_scatter_input",
            "world_size",
        ):
            raise RuntimeError("Unsupported FSDP copy-in signature")
        self.emit("installed", reduce=callable_name(original_reduce), copy_in=callable_name(original_copy))

        @functools.wraps(original_reduce)
        def observe_reduce(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            params = bound.arguments["fsdp_params"]
            names = tuple(getattr(param, "_param_fqn", None) for param in params)
            if not names or any(not isinstance(name, str) for name in names):
                raise RuntimeError("FSDP parameter FQNs unavailable; refusing an unidentifiable trace")
            if len(names) != len(bound.arguments["unsharded_grads"]):
                raise RuntimeError("FSDP parameter/gradient order cannot be mapped")
            group = bound.arguments["reduce_scatter_group"]
            world_size = group.size() if group is not None else 1
            key = (names, id(group))
            if key not in self.groups:
                if len(self.groups) >= 256:
                    raise RuntimeError("Unexpectedly many FSDP groups for this bounded diagnostic")
                self.groups[key] = len(self.groups)
                comm = bound.arguments["reduce_scatter_comm"]
                self.emit(
                    "group",
                    group_id=self.groups[key],
                    parameter_fqns=names,
                    parameter_owners=sorted({name.rsplit(".", 1)[0] for name in names}),
                    reduce_scatter_world_size=world_size,
                    reduce_scatter_stream=repr(bound.arguments.get("reduce_scatter_stream")),
                    comm_type=f"{type(comm).__module__}.{type(comm).__qualname__}",
                    allocate=callable_name(comm.allocate),
                )
            self.calls += 1
            metadata = {"call": self.calls, "group_id": self.groups[key], "world_size": world_size}
            # Do not keep extra parameter/gradient tensor references in the observer.
            del bound, params, group
            token = self.context.set(metadata)
            self.emit("reduce_enter", **metadata)
            try:
                result = original_reduce(*args, **kwargs)
                self.emit("reduce_return", **metadata)
                return result
            except Exception as error:
                self.emit("reduce_exception", **metadata, error_type=type(error).__name__, message=str(error)[:2048])
                raise
            finally:
                self.context.reset(token)

        @functools.wraps(original_copy)
        def observe_copy(unsharded_grads, reduce_scatter_input, world_size):
            metadata = self.context.get()
            if metadata is None:
                raise RuntimeError("Copy-in was called outside the observed FSDP reduction")
            # Full metadata at each copy-in makes the last submitted target identifiable
            # even when an asynchronous error surfaces in a later operation.
            self.emit(
                "copy_enter",
                **metadata,
                inputs=[tensor_metadata(tensor) for tensor in unsharded_grads],
                out=tensor_metadata(reduce_scatter_input),
                num_chunks=world_size,
            )
            try:
                result = original_copy(unsharded_grads, reduce_scatter_input, world_size)
                self.emit("copy_return", **metadata)
                return result
            except Exception as error:
                # No device metadata reads after failure.
                self.emit("copy_exception", **metadata, error_type=type(error).__name__, message=str(error)[:2048])
                raise

        collectives.foreach_reduce = param_group.foreach_reduce = observe_reduce
        collectives.foreach_reduce_scatter_copy_in = observe_copy

    def attach_optimizer(self, optimizer):
        if not optimizer.param_groups or any(
            group.get("foreach") is not False or group.get("fused") is not False for group in optimizer.param_groups
        ):
            raise RuntimeError("This comparison requires actual AdamW foreach=False, fused=False")
        self.emit("optimizer_verified", foreach=False, fused=False, groups=len(optimizer.param_groups))

        def before_step(optimizer, args, kwargs):
            self.emit("optimizer_enter", last_reduce_call=self.calls)

        def after_step(optimizer, args, kwargs):
            self.updates += 1
            self.emit("optimizer_return", last_reduce_call=self.calls)

        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)


def limit_training_loop(args, smoke_steps):
    # train_steps is a read-only property; its supported cap is train.max_steps.
    args.train.max_steps = min(args.train_steps, smoke_steps)
    args.train.num_train_epochs = 1


if __name__ == "__main__":
    import torch
    from train_vlm_chunkcat_eager import install_adamw_scalar_diagnostic

    from veomni.arguments import parse_args
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer

    if not torch.__version__.startswith("2.10."):
        raise RuntimeError(f"This private FSDP diagnostic targets torch 2.10, found {torch.__version__}")
    # Import the normal training stack first, including its platform patches.
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    import veomni.optim.optimizer as optimizer_module

    for function in (_fsdp_collectives.foreach_reduce, _fsdp_collectives.foreach_reduce_scatter_copy_in):
        if function.__module__ != _fsdp_collectives.__name__:
            raise RuntimeError("Native FSDP bindings required; refusing a prepatched copy/reduce function")
    observer = FSDPObserver(os.environ["VEOMNI_FSDP_OBSERVE_DIR"])
    observer.install(_fsdp_collectives, _fsdp_param_group)
    verified = install_adamw_scalar_diagnostic(optimizer_module)
    args = parse_args(VeOmniVLMArguments)
    trainer = VLMTrainer(args)
    if len(verified) != 1:
        raise RuntimeError("This update counter expects exactly one verified AdamW optimizer")
    observer.attach_optimizer(trainer.base.model.optimizer)
    smoke_steps = int(os.environ.get("VEOMNI_FSDP_OBSERVE_STEPS", "3"))
    if smoke_steps < 1 or trainer.base.start_step != 0 or trainer.base.start_epoch != 0:
        raise RuntimeError("This bounded observation requires a fresh run and a positive step limit")
    # The scheduler is already built against the original training horizon.
    # Cap only the outer loop so the first updates retain that LR schedule.
    observer.emit(
        "loop_cap",
        original_train_steps=args.train_steps,
        original_epochs=args.train.num_train_epochs,
        requested_steps=smoke_steps,
        scheduler_horizon_preserved=True,
    )
    limit_training_loop(args, smoke_steps)
    observer.emit("training_enter")
    try:
        trainer.train()
        observer.emit("training_return", reduce_calls=observer.calls)
    except Exception as error:
        observer.emit("training_exception", error_type=type(error).__name__, message=str(error)[:2048])
        raise
    finally:
        observer.file.close()
