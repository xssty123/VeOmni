"""CPU storage for immutable Ascend attention and GDN outputs during SAC."""

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from ..utils.device import create_event, create_stream, get_current_stream, switch_to_specified_stream


# Both functional kernels return fresh tensors. Scalar RNG metadata and None
# are replayed along with tensors, without issuing a device transfer.
HOST_CACHE_OPS = frozenset(
    {"fla_npu_stable.npu_chunk_gated_delta_rule_fwd.default", "npu.npu_fusion_attention.default"}
)


@lru_cache(maxsize=8)
def _copy_stream(device):
    return create_stream(device)


@dataclass
class _HostOutput:
    data: torch.Tensor
    device: torch.device
    ready: object

    def restore(self):
        get_current_stream(self.device).wait_event(self.ready)
        return self.data.to(self.device, non_blocking=True)


def _pack(value):
    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach()
    if value.device.type == "cpu":
        return value
    stream = _copy_stream(value.device)
    stream.wait_stream(get_current_stream(value.device))
    with switch_to_specified_stream(stream):
        buffer = torch.empty_like(value, device="cpu", pin_memory=True)
        buffer.copy_(value, non_blocking=True)
        value.record_stream(stream)
        event = create_event()
        event.record(stream)
    return _HostOutput(buffer, value.device, event)


class _KernelOutputMode(TorchDispatchMode):
    def __init__(self, entries, replay):
        super().__init__()
        self.entries = entries
        self.replay = replay

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) not in HOST_CACHE_OPS:
            return func(*args, **(kwargs or {}))
        queue = self.entries[func]
        if self.replay:
            if not queue:
                raise RuntimeError(f"Checkpoint replay has no saved output for {func}")
            return tree_map(lambda x: x.restore() if isinstance(x, _HostOutput) else x, queue.popleft())
        out = func(*args, **(kwargs or {}))
        queue.append(tree_map(_pack, out))
        return out


def host_cache_contexts(policy):
    # Native SAC handles every other selected op. Exclude these two there so
    # a second accelerator-resident reference cannot defeat the CPU offload.
    def device_policy(ctx, op, *args, **kwargs):
        if str(op) in HOST_CACHE_OPS:
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        return policy(ctx, op, *args, **kwargs)

    forward, backward = torch.utils.checkpoint.create_selective_checkpoint_contexts(device_policy)
    entries = defaultdict(deque)

    @contextmanager
    def combined(native, replay):
        with native, _KernelOutputMode(entries, replay):
            yield

    return combined(forward, False), combined(backward, True)
