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

"""Fail loudly when candidate CI lacks the target-specific FLA 26.10 runtime."""

import importlib
import os
from importlib import metadata
from pathlib import Path

import pytest
from packaging.version import Version

from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.skipif(
    not IS_NPU_AVAILABLE or os.environ.get("VEOMNI_NPU_TARGET") not in ("a3", "a5"), reason="A3/A5 candidate test"
)
_DISTRIBUTIONS = (
    "fla-npu",
    "flash-linear-attention-npu",
    "flash-linear-attention-npu-a2",
    "flash-linear-attention-npu-a3",
    "flash-linear-attention-npu-a5",
)


def test_fla_npu_distribution_and_api_contract():
    installed = {}
    for name in _DISTRIBUTIONS:
        try:
            installed[name] = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            pass
    assert len(installed) == 1, f"Expected exactly one FLA distribution: {list(installed)}"
    name, distribution = next(iter(installed.items()))
    assert name in _DISTRIBUTIONS[-2:], f"Legacy FLA distribution: {name}"
    assert Version(distribution.version).release == (26, 10, 0), distribution.version
    target = os.environ.get("VEOMNI_NPU_TARGET")
    if target:
        assert name == f"flash-linear-attention-npu-{target}", (name, target)
    module = importlib.import_module("fla_npu.ops.ascendc")
    for symbol in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd"):
        assert callable(getattr(module, symbol, None)), symbol
    package = importlib.import_module("fla_npu")
    expected = Path(distribution.locate_file("fla_npu")).resolve()
    assert Path(package.__file__).resolve().parent == expected, package.__file__
