# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""NPU eager FP32 forward/loader parity against independent Hugging Face models."""

import gc
import tempfile
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from tests.models.base import test_low_precision_oracle as oracle
from veomni.utils.device import IS_NPU_AVAILABLE, empty_cache, get_device_type, get_dist_comm_backend, get_torch_device


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU model parity needs torch_npu")


def _case(family, arch, factory, kind="causal_lm"):
    return oracle.Case(
        family + "-eager",
        arch,
        kind,
        factory,
        oracle._hf(family, arch),
        attn_implementation="eager",
        dtype=torch.float32,
    )


_CASES = [
    _case("qwen3", "Qwen3ForCausalLM", oracle.tiny_qwen3_config),
    _case("qwen3_moe", "Qwen3MoeForCausalLM", oracle.tiny_qwen3_moe_config),
    _case("deepseek_v3", "DeepseekV3ForCausalLM", oracle.tiny_deepseek_v3_config),
    _case("glm_moe_dsa", "GlmMoeDsaForCausalLM", oracle.tiny_glm_moe_dsa_config),
    _case(
        "qwen3_5",
        "Qwen3_5ForCausalLM",
        oracle._qwen3_5_full_attention(oracle.tiny_qwen3_5_text_config),
        "qwen3_5_text",
    ),
    _case(
        "qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        oracle._qwen3_5_full_attention(oracle.tiny_qwen3_5_moe_text_config),
        "qwen3_5_text",
    ),
    replace(
        _case("qwen3_vl", "Qwen3VLForConditionalGeneration", oracle.tiny_qwen3_vl_config, "vlm_full"),
        single_image=True,
    ),
]
_LOADER_CASES = [
    case for case in _CASES if case.case_id in {"qwen3_moe-eager", "deepseek_v3-eager", "glm_moe_dsa-eager"}
]


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    if not IS_NPU_AVAILABLE:
        yield
        return
    initialized_here = not dist.is_initialized()
    with tempfile.TemporaryDirectory() as directory:
        if initialized_here:
            get_torch_device().set_device(0)
            dist.init_process_group(
                get_dist_comm_backend(), init_method=f"file://{directory}/init", rank=0, world_size=1
            )
        try:
            yield
        finally:
            if initialized_here:
                dist.destroy_process_group()


def _check(case, from_disk):
    config = oracle._make_config(case)
    inputs, shared, veomni_only = oracle._make_inputs(case, config, get_device_type(), case.dtype)
    expected, weights, buffers = oracle._hf_logits(case, config, inputs, shared, case.dtype)
    with tempfile.TemporaryDirectory() as directory:
        model = (
            oracle._build_veomni_model_from_disk(case, config, weights, buffers, directory)
            if from_disk
            else oracle._build_veomni_model(case, config, weights)
        )
        assert type(model).__module__.endswith("_npu"), type(model).__module__
        with torch.no_grad():
            actual = model(input_ids=inputs, use_cache=False, **shared, **veomni_only).logits
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    del model, expected, actual, weights, buffers
    gc.collect()
    empty_cache()


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
def test_logits_equal_v5_npu(case):
    _check(case, from_disk=False)


@pytest.mark.parametrize("case", _LOADER_CASES, ids=lambda case: case.case_id)
def test_logits_equal_v5_via_loader_npu(case):
    _check(case, from_disk=True)
