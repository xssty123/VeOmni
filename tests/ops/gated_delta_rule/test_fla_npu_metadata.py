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

"""CPU contract for the host metadata consumed by FLA NPU 26.10 GDN."""

from itertools import accumulate

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import fla_npu


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

_CASES = [
    pytest.param([128, 256], 4, 512, id="aligned-four-heads"),
    pytest.param([64, 128, 32], 8, 256, id="short-tail-eight-heads"),
    pytest.param([65, 67], 4, 512, id="packed-two-partial-chunks"),
]


def _expected_rows(lengths, chunk_size):
    # Enumerate chunk start positions independently of the production cdiv
    # and repeat_interleave implementation; sequence IDs must restart neither
    # the cumulative boundaries nor another sequence's local chunk numbering.
    return [[seq, chunk] for seq, length in enumerate(lengths) for chunk, _ in enumerate(range(0, length, chunk_size))]


def _unexpected_work(*args, **kwargs):
    pytest.fail("Runtime metadata resolution must reuse supplied host tables without tensor reads or uploads")


@pytest.mark.parametrize("lengths,num_heads,cumsum_size", _CASES)
def test_precompute_tables_match_sequence_boundaries(lengths, num_heads, cumsum_size):
    boundaries = [0, *accumulate(lengths)]
    cu = torch.tensor(boundaries, dtype=torch.int32)
    host_cu, tensor_tables, host_tables = fla_npu.precompute_varlen_metadata(cu, num_heads, device="cpu")

    assert host_cu == boundaries
    expected_keys = {str(size) for size in (16, 32, 64, 128, 1216, cumsum_size)}
    assert tensor_tables.keys() == host_tables.keys() == expected_keys
    for key in expected_keys:
        expected = _expected_rows(lengths, int(key))
        table = tensor_tables[key]
        assert table.dtype == cu.dtype
        assert table.device == cu.device
        assert table.tolist() == expected
        assert host_tables[key] == [value for row in expected for value in row]


@pytest.mark.parametrize("lengths,num_heads,cumsum_size", _CASES)
def test_runtime_fallback_builds_only_active_host_table(monkeypatch, lengths, num_heads, cumsum_size):
    boundaries = [0, *accumulate(lengths)]
    cu = torch.tensor(boundaries, dtype=torch.int32)
    g = torch.empty(1, sum(lengths), num_heads)
    monkeypatch.setattr(fla_npu, "_prepare_chunk_indices", _unexpected_work)
    monkeypatch.setattr(torch.Tensor, "to", _unexpected_work)

    resolved_cu, host_cu, tensor_tables, host_tables = fla_npu._ensure_varlen_metadata(g, cu, 64)

    assert resolved_cu is cu
    assert host_cu == boundaries
    assert tensor_tables == {}
    assert host_tables == {"64": [value for row in _expected_rows(lengths, 64) for value in row]}


@pytest.mark.parametrize("lengths,num_heads,cumsum_size", _CASES)
def test_precomputed_metadata_is_reused_without_tensor_sync(monkeypatch, lengths, num_heads, cumsum_size):
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    g = torch.empty(1, sum(lengths), num_heads)
    host_cu, tensor_tables, host_tables = fla_npu.precompute_varlen_metadata(cu, num_heads)
    monkeypatch.setattr(fla_npu, "_prepare_chunk_indices", _unexpected_work)
    monkeypatch.setattr(fla_npu, "_prepare_chunk_indices_list", _unexpected_work)
    monkeypatch.setattr(torch.Tensor, "tolist", _unexpected_work)
    monkeypatch.setattr(torch.Tensor, "to", _unexpected_work)

    resolved_cu, resolved_host, resolved_tensors, resolved_tables = fla_npu._ensure_varlen_metadata(
        g, cu, 64, cu_seqlens_list=host_cu, chunk_indices=tensor_tables, chunk_indices_list=host_tables
    )

    assert resolved_cu is cu
    assert resolved_host == host_cu
    assert resolved_tables == host_tables
    assert resolved_tensors.keys() == tensor_tables.keys()
    assert all(resolved_tensors[key] is table for key, table in tensor_tables.items())


def test_partial_metadata_fills_active_table_and_preserves_other_sizes(monkeypatch):
    cu = torch.tensor([0, 65, 132], dtype=torch.int32)
    g = torch.empty(1, 132, 4)
    legacy_tensor = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32)
    host_tables = {"64": None, "128": [0, 0, 1, 0]}
    monkeypatch.setattr(fla_npu, "_prepare_chunk_indices", _unexpected_work)
    monkeypatch.setattr(torch.Tensor, "tolist", _unexpected_work)
    monkeypatch.setattr(torch.Tensor, "to", _unexpected_work)

    resolved_cu, host_cu, tensor_tables, resolved_tables = fla_npu._ensure_varlen_metadata(
        g, cu, 64, cu_seqlens_list=[0, 65, 132], chunk_indices={"128": legacy_tensor}, chunk_indices_list=host_tables
    )

    assert resolved_cu is cu
    assert host_cu == [0, 65, 132]
    assert tensor_tables["128"] is legacy_tensor
    assert set(tensor_tables) == {"128"}
    assert resolved_tables == {"64": [0, 0, 0, 1, 1, 0, 1, 1], "128": [0, 0, 1, 0]}
    assert host_tables == {"64": None, "128": [0, 0, 1, 0]}
