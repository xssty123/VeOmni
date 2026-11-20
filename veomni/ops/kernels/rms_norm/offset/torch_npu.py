# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""offset RMSNorm torch_npu adapter (scale is 1 + weight)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from ..group import group_last_dim
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Last-dimension grouping applied by the native forward."""

    group_size: int | None = None


def forward(x: Tensor, weight: Tensor, *, eps: float, group_size: int | None = None) -> tuple[Tensor, SavedState]:
    """NPU fused affine RMSNorm. The fused scale is ``1 + weight``.

    Empty ``x`` falls back to the eager pair. Backward also passes
    ``1 + weight`` into ``npu_rms_norm_backward``.
    """
    if x.numel() == 0:
        return _eager.forward(x, weight, eps=eps, group_size=group_size)

    import torch_npu

    if group_size is not None:
        grouped = group_last_dim(x.float(), group_size)
        unit = grouped.new_ones(group_size)
        normalized, rstd = torch_npu.npu_rms_norm(grouped, unit, eps)
        output = normalized.reshape_as(x) * (1.0 + weight.float())
        return output.to(x.dtype), SavedState((x, weight, rstd, unit), _Meta(group_size))
    output, rstd = torch_npu.npu_rms_norm(x, 1.0 + weight, eps)
    return output, SavedState((x, weight, rstd), _Meta())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_x, grad_weight)``. Empty inputs reuse the eager backward."""
    meta = saved.metadata
    if not isinstance(meta, _Meta):
        return _eager.backward(grad_output, saved)
    x, weight, *optional_rstd = saved.tensors

    import torch_npu

    if meta.group_size is not None:
        rstd, unit = optional_rstd
        grouped = group_last_dim(x.float(), meta.group_size)
        grad = grad_output.float()
        scaled = (grad * (1.0 + weight.float())).reshape_as(grouped)
        dx, _ = torch_npu.npu_rms_norm_backward(scaled.contiguous(), grouped, unit, rstd)
        dw = (grad * (grouped * rstd).reshape_as(x)).sum_to_size(weight.shape)
        return dx.reshape_as(x).to(x.dtype), dw.to(weight.dtype)
    (rstd,) = optional_rstd
    scale = 1.0 + weight
    return torch_npu.npu_rms_norm_backward(grad_output.contiguous(), x, scale, rstd)
