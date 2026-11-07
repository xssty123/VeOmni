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

"""NPU end-to-end coverage for the model ``return_log_probs`` contract.

The CUDA suite remains the bitwise reference.  These tests exercise the same
plain-model integration on the independently generated NPU Qwen3 and Qwen3-VL
modeling files: model forward -> loss dispatch -> ``fused_linear_aux`` and
backward.  Kernel-only arithmetic is already covered by the shared chunked
loss tests, so this module keeps the NPU-specific surface intentionally small.
"""

import gc
import os

import pytest
import torch
import torch.nn.functional as F

from veomni.utils.device import IS_NPU_AVAILABLE, empty_cache, get_device_type, synchronize

from ..tools.training_utils import make_npu_ops_config
from .base import test_return_log_probs_e2e as _base


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="return_log_probs NPU tests require torch_npu")

_MODELS = [
    pytest.param(_base.TOY_QWEN3, "qwen3", id="qwen3-text"),
    pytest.param(_base.TOY_QWEN3_VL, "qwen3_vl", id="qwen3_vl-vlm"),
]

# The full-logits model path projects a 3-D tensor while the streamed path
# projects a flattened 2-D tensor.  Ascend may select different matmul tilings
# for those shapes, so this is a numerical integration check rather than the
# CUDA suite's batch-invariant bitwise check.  Both paths run in fp32 here.
_NPU_RTOL = 1e-3
_NPU_ATOL = 1e-3


def _release_npu() -> None:
    synchronize()
    gc.collect()
    empty_cache()


@pytest.fixture(autouse=True)
def _scope_npu_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _build_npu_model(toy_path: str, family: str, ce_impl: str = "chunk_loss"):
    from veomni.models.auto import build_foundation_model

    ops_implementation = make_npu_ops_config(
        family,
        attn_implementation="eager",
        cross_entropy_loss_implementation=ce_impl,
    )
    return build_foundation_model(
        config_path=toy_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        init_device=get_device_type(),
        ops_implementation=ops_implementation,
    )


def _assert_reference_close(actual: torch.Tensor, expected: torch.Tensor, family: str, field: str) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        rtol=_NPU_RTOL,
        atol=_NPU_ATOL,
        msg=lambda msg: f"[{family}] {field} differs from full-logits reference: {msg}",
    )


@pytest.mark.parametrize("toy_path,family", _MODELS)
@pytest.mark.parametrize("ce_impl", ["chunk_loss", "eager"])
def test_return_log_probs_matches_logits_reference_npu(toy_path: str, family: str, ce_impl: str) -> None:
    """NPU model forward returns the expected per-token log-probs payload."""
    assert os.path.isdir(toy_path), f"Path not found: {toy_path}"

    torch.manual_seed(0)
    model = _build_npu_model(toy_path, family, ce_impl).eval()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = _base.IGNORE_INDEX
    labels[1, ::5] = _base.IGNORE_INDEX

    with torch.no_grad():
        ref_logits = model(input_ids=input_ids, use_cache=False).logits
        ref_log_probs, ref_entropy = _base._reference_log_probs_and_entropy_from_logits(ref_logits, labels)
        out = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_log_probs=True,
            chunk_size=batch_size * seq_len + 1,
        )

    assert out.loss is None, "loss must be None when return_log_probs=True"
    assert out.logits is None, "logits must be cleared when return_log_probs=True"
    assert out.fused_linear_aux is not None, "fused_linear_aux must be populated"
    assert out.fused_linear_aux.log_probs.shape == labels.shape
    assert out.fused_linear_aux.entropy.shape == labels.shape
    assert torch.isfinite(out.fused_linear_aux.log_probs).all()
    assert torch.isfinite(out.fused_linear_aux.entropy).all()

    _assert_reference_close(out.fused_linear_aux.log_probs, ref_log_probs, family, "log_probs")
    _assert_reference_close(out.fused_linear_aux.entropy, ref_entropy, family, "entropy")

    shifted_target_is_ign = F.pad(labels[..., 1:] == _base.IGNORE_INDEX, (0, 1), value=True)
    assert torch.all(out.fused_linear_aux.log_probs[shifted_target_is_ign] == 0)
    assert torch.all(out.fused_linear_aux.entropy[shifted_target_is_ign] == 0)
    assert torch.all(out.fused_linear_aux.log_probs[~shifted_target_is_ign] < 0)
    assert torch.all(out.fused_linear_aux.entropy[~shifted_target_is_ign] > 0)

    del model, out, ref_logits, ref_log_probs, ref_entropy
    _release_npu()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_return_log_probs_consumer_backward_and_distill_npu(toy_path: str, family: str) -> None:
    """Exercise the shared consumer, gradient and distillation contracts on NPU."""
    model = _build_npu_model(toy_path, family)
    try:
        _base._check_plain_verl_consumer_contract(model, family)
        _base._check_log_probs_backward(model, family)
        _base._check_topk_distill_forward_and_backward(model, family)
    finally:
        del model
        _release_npu()
