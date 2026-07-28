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

"""Varlen-metadata memoization in the AscendC gated delta-rule entry.

``cu_seqlens`` is identical for every GDN layer of a micro-batch, so
``_ensure_varlen_metadata`` must build ``chunk_indices`` once and hand the same
objects to layers 2..N — each rebuild costs a ``.tolist()`` D2H sync plus a
Python loop over every chunk. The memoization keys on tensor *identity*, which
only holds together because the int32 -> int64 normalization is memoized too;
these tests pin both halves, since a regression there is silent (correct results,
one sync per layer).

The metadata helpers are plain torch, so they run on a CPU tensor — but the
module they live in imports ``torch_npu`` / ``fla_npu``, hence the importorskip.
"""

import pytest
import torch


flash_gdr = pytest.importorskip(
    "veomni.ops.kernels.gated_delta_rule._ascend.flash_gated_delta_rule",
    reason="AscendC GDN entry requires torch_npu + fla_npu + triton-ascend (NPU hosts only)",
)


_CHUNK_SIZE = 64
_NUM_LAYERS = 4


def _cu_seqlens(lengths):
    """Collator-shaped cu_seqlens: int32, one entry per boundary."""
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _dummy_g(num_heads=4, seq_len=1024):
    # _cumsum_block_t only reads g.shape[-1]; _ensure_varlen_metadata only reads g.device.
    return torch.zeros(1, seq_len, num_heads)


def _required_sizes(g, chunk_size=_CHUNK_SIZE):
    sizes = set(flash_gdr._DEFAULT_VARLEN_CHUNK_SIZES)
    sizes.add(chunk_size)
    sizes.add(flash_gdr._cumsum_block_t(g, chunk_size))
    return sizes


# ---------------------------------------------------------------------------
# normalization — the identity anchor everything else keys on
# ---------------------------------------------------------------------------


def test_normalized_cu_seqlens_returns_one_object_per_micro_batch():
    cu_seqlens = _cu_seqlens([128, 256, 640])
    device = torch.device("cpu")

    first = flash_gdr._normalized_cu_seqlens(cu_seqlens, device)
    second = flash_gdr._normalized_cu_seqlens(cu_seqlens, device)

    assert first is second, "int32 -> int64 cast must be memoized, else every downstream cache misses"
    assert first.dtype == torch.int64
    assert first.tolist() == cu_seqlens.tolist()


def test_normalized_cu_seqlens_passes_through_when_already_normalized():
    cu_seqlens = _cu_seqlens([64, 64]).to(torch.int64)

    assert flash_gdr._normalized_cu_seqlens(cu_seqlens, cu_seqlens.device) is cu_seqlens


def test_normalized_cu_seqlens_separates_distinct_inputs():
    a, b = _cu_seqlens([128, 128]), _cu_seqlens([64, 192])
    device = torch.device("cpu")

    out_a = flash_gdr._normalized_cu_seqlens(a, device)
    out_b = flash_gdr._normalized_cu_seqlens(b, device)

    assert out_a is not out_b
    assert out_a.tolist() == a.tolist()
    assert out_b.tolist() == b.tolist()


# ---------------------------------------------------------------------------
# chunk indices — a full sweep must fit in the cache
# ---------------------------------------------------------------------------


def test_prepare_chunk_indices_survives_a_full_sweep():
    """maxsize < len(required_sizes) would evict the entry the next sweep asks for first."""
    g = _dummy_g()
    cu_seqlens = flash_gdr._normalized_cu_seqlens(_cu_seqlens([512, 1024, 256]), g.device)
    sizes = sorted(_required_sizes(g))

    first_sweep = [flash_gdr.prepare_chunk_indices(cu_seqlens, size) for size in sizes]
    second_sweep = [flash_gdr.prepare_chunk_indices(cu_seqlens, size) for size in sizes]

    assert all(a is b for a, b in zip(first_sweep, second_sweep))
    assert len(sizes) <= flash_gdr._VARLEN_CACHE_MAXSIZE


def test_prepare_chunk_indices_cache_matches_recomputation():
    g = _dummy_g()
    cu_seqlens = flash_gdr._normalized_cu_seqlens(_cu_seqlens([300, 700]), g.device)

    for size in sorted(_required_sizes(g)):
        cached = flash_gdr.prepare_chunk_indices(cu_seqlens, size)
        fresh = flash_gdr.prepare_chunk_indices.__wrapped__(cu_seqlens, size)
        assert torch.equal(cached, fresh)


def test_prepare_chunk_indices_list_accepts_tensor_and_host_list():
    cu_seqlens = _cu_seqlens([128, 384])
    from_tensor = flash_gdr.prepare_chunk_indices_list(cu_seqlens, _CHUNK_SIZE)
    from_list = flash_gdr.prepare_chunk_indices_list(cu_seqlens.tolist(), _CHUNK_SIZE)

    assert from_tensor == from_list
    # Same host list content -> same cached object, whichever form the caller used.
    assert from_tensor is from_list


# ---------------------------------------------------------------------------
# end to end: N layers of a micro-batch pay for metadata once
# ---------------------------------------------------------------------------


def _ensure(g, cu_seqlens):
    return flash_gdr._ensure_varlen_metadata(
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=None,
        chunk_indices=None,
        chunk_indices_list=None,
        chunk_size=_CHUNK_SIZE,
    )


def test_layers_share_one_set_of_metadata_objects():
    g = _dummy_g()
    cu_seqlens = _cu_seqlens([512, 512, 1024])

    layers = [_ensure(g, cu_seqlens) for _ in range(_NUM_LAYERS)]
    head_cu, head_list, head_tensors, head_lists = layers[0]

    for cu, cu_list, tensors, lists in layers[1:]:
        assert cu is head_cu
        assert cu_list is head_list
        assert tensors.keys() == head_tensors.keys()
        for key in head_tensors:
            assert tensors[key] is head_tensors[key], f"chunk_indices['{key}'] rebuilt per layer"
            assert lists[key] is head_lists[key], f"chunk_indices_list['{key}'] rebuilt per layer"


def test_metadata_tracks_a_new_micro_batch():
    g = _dummy_g()
    first_cu, first_list, first_tensors, _ = _ensure(g, _cu_seqlens([256, 256]))
    second_cu, second_list, second_tensors, _ = _ensure(g, _cu_seqlens([128, 128, 256]))

    assert first_cu is not second_cu
    assert first_list != second_list
    key = str(_CHUNK_SIZE)
    assert not torch.equal(first_tensors[key], second_tensors[key])


def test_metadata_matches_uncached_reference():
    g = _dummy_g()
    cu_seqlens = _cu_seqlens([192, 832])
    cu, cu_list, tensors, lists = _ensure(g, cu_seqlens)

    assert cu_list == cu_seqlens.tolist()
    for key, indices in tensors.items():
        assert torch.equal(indices, flash_gdr.prepare_chunk_indices.__wrapped__(cu, int(key)))
        assert lists[key] == flash_gdr._prepare_chunk_indices_list_from_host.__wrapped__(cu_list, int(key))


def test_caller_supplied_metadata_still_wins():
    """The precompute-outside-the-model path (MM's) must keep working untouched."""
    g = _dummy_g()
    cu_seqlens = _cu_seqlens([64, 192])
    supplied = torch.zeros(2, 2, dtype=torch.int64)

    _, _, tensors, _ = flash_gdr._ensure_varlen_metadata(
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=None,
        chunk_indices={str(_CHUNK_SIZE): supplied},
        chunk_indices_list=None,
        chunk_size=_CHUNK_SIZE,
    )

    assert tensors[str(_CHUNK_SIZE)] is supplied
