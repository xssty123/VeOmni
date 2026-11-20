# Qwen4-Exp integration status

Supported in this correctness stage:

- GPU and NPU VLM supervised fine-tuning with Ulysses sequence parallelism, plus
  GPU context parallelism (`cp_size > 1`).
- GatedDeltaNet sequence-to-head all-to-all using the existing Qwen3.5
  causal-conv and gated-delta kernels over the active SP/CP group.
- VisionModel follows Qwen3.5's SP path: each rank runs patch embedding and
  merger on its local visual-token shard, vision attention scopes `use_parallel_state`
  to a temporary Ulysses SP state so standard attention performs sequence/head
  exchange over the active SP/CP group, and only the merged features are
  gathered for the global placeholder scatter.
- CP-style QSA selection plus main-attention KV all-gather. The indexer uses
  local block pooling, compressed-row all-gather, and local queries to return
  local `[B, S_local, K]` compact global token indices; `Qwen4ExpTextAttention`
  keeps `query_states` local (`[B, H_q, S_local, D]`) with all query heads intact
  and all-gathers only `key_states` and `value_states` (`[B, H_kv, S_global, D]`).
  `qsa_attention_implementation=eager` expands compact indices to a local-query
  `[B, 1, S_local, S_global]` mask, while `qsa_attention_implementation=tilelang`
  runs the gather-based TileLang sparse-attention kernel
  (`veomni/ops/kernels/qsa_attention/`, bf16, NVIDIA CUDA) directly on the local
  query shard and all-gathered KV with full GQA head grouping preserved.
  Explicit cu-seqlens isolate packed samples consistently for SP/CP=1 and SP/CP>1.
- PLE n-gram token halos and differentiable dilated-convolution halos across
  Ulysses or context-parallel shard boundaries, while persistent PLE table
  routing remains on the independent `ple_fsdp × ple` mesh. Packed metadata is
  computed once per model forward; n-gram histories reset at local sample
  offsets, and convolution correction is bounded to the first
  `(kernel_size - 1) * dilation` outputs of each sample in fixed-size chunks.
  Interior tokens retain the original convolution and SP halo, with no added
  collective or checkpoint parameter.
- VeOmni fused cross-entropy and fused MoE dispatch, with sequence-local router
  logits and padding masks for auxiliary load-balancing loss.
- Concurrent PLE and MoE expert parallelism: PLE tables use the persistent
  two-dimensional `ple_fsdp × ple` layout while expert tensors use the
  independent `ep_fsdp × ep` layout.
- Scalable loading and training of the released ~95 GiB PLE table:
  - preserve its 128 checkpoint-native embedding shards instead of
    concatenating them;
  - persistently shard every table by rows over `ple` and columns over the
    complementary `ple_fsdp` dimension;
  - stream only the local row-by-column rectangle and zero-pad final rows;
  - keep PLE weights outside FSDP2, eliminating their forward/backward
    parameter all-gathers;
  - route lookup requests and differentiable column slices over the flattened
    PLE mesh, so every rank may process a different sample.
- Explicitly discarding `mtp.*` checkpoint tensors; MTP loss is not supported.

The internal example uses `ple_size=8`, `ep_size=4`,
`broadcast_model_weights_from_rank0=false`, and
`ep_sharded_stream_load=true`. The world size and every padded PLE shard row
count must be divisible by `ple_size`; the number of experts must be divisible
by `ep_size`. PLE and EP may be enabled together because their parameter sets
and communication meshes are independent.

The implementation details and constraints of this Qwen4-Exp-only local
parallel path are documented in
[Qwen4-Exp PLE Two-Dimensional Parallelism](../../../../docs/design/qwen4_exp_ple_2d_parallelism.md).

Known limitations:

- Cache prefill/decode, HSDP replicas of persistent PLE, NPU context
  parallelism, and hybrid CP × Ulysses remain unsupported.
- The default `eager` QSA backend expands compact indices to a dense mask and
  materializes the `[B, H_q, S_local, S_global]` attention-score tensor. Use
  `qsa_attention_implementation=tilelang` (NVIDIA CUDA, bf16) for the 16K
  performance gate; the TileLang backend covers training forward/backward only
  and raises on KV-cache or additive-mask layouts. Nonzero attention dropout
  falls back to eager QSA and therefore loses the sparse kernel's memory
  advantage.
- Ulysses/CP GatedDeltaNet requires non-eager, varlen-capable causal-conv and
  chunk gated-delta-rule kernels. GatedDeltaNet head counts must be divisible by
  the active `ulysses_size`/`cp_size`; QSA attention has no head-divisibility
  constraint because it keeps all query heads local and all-gathers only KV.
  With `chunk_gated_delta_rule_implementation=flash_qla`, GatedDeltaNet
  preserves grouped Q/K heads (`num_k_heads < num_v_heads`) directly into the
  FlashQLA chunk kernel instead of expanding Q/K with `repeat_interleave`.
  Qwen4-Exp does the same for `fla_npu`, whose AscendC chunk kernel supports
  grouped value attention.
- PLE n-gram history and its dilated convolution reset at explicit
  `cu_seq_lens_q` packed-sample boundaries. The halo-only fallback remains for
  direct component calls that do not provide packing metadata.
- Distributed PLE training expects pretrained or DCP weights. Initializing from
  scratch after PLE parameters become DTensors is not supported by the upstream
  Hugging Face initializer.
- The real checkpoint schema and shard reads plus two-process lookup/backward
  semantics are validated without materializing the 95 GiB table. A real
  multi-node load-and-train smoke run remains the final deployment gate.

## Compact QSA indexing on Ascend

The eager indexer uses INT8 for NPU visibility gathers (the resulting mask is still
boolean) and bounded INT32 position arithmetic. Top-k scores and floating
attention arithmetic are unchanged. Positions including compression padding
must fit INT32, matching the output-index contract.

A two-entry host-only LRU caches immutable block metadata keyed by the complete
packed boundaries, compression ratio, local shard length and SP rank/group size.
Blocks are bucketed by owner in one pass, and counts stay on the host. One model
forward prepares the device metadata before the decoder loop and reuses it across
QSA layers and checkpoint recomputation. Only rank-local query segment maps are
materialized. Direct indexer calls prepare their own segment metadata through the
same validated path. The CANN backend separately caches up to eight Lightning
Indexer plans keyed by device and kernel shape.

## Qwen3.8-Flash-Next Ascend training

The [Ascend SFT configuration](../../../../configs/text/qwen38_flash_next_ascend.yaml) combines all seven adaptations. It uses BF16, FSDP2, EP=8, PLE=8, and CP=Ulysses=1. Set the checkpoint and dataset paths for your environment. Launch it with `bash train.sh tasks/train_vlm.py configs/text/qwen38_flash_next_ascend.yaml`. Individual kernel selectors can also be used independently. For example:

```yaml
model:
  ops_implementation:
    qsa_indexer_implementation: cann
    causal_conv1d_implementation: fla_npu
    chunk_gated_delta_rule_implementation: fla_npu
```

- Lightning Indexer requires `lightning_indexer` and `lightning_indexer_metadata`
  from `cann_ops_transformer.ops`. It ranks complete compressed blocks. VeOmni
  adds the causal tail and resets block numbering at packed sample boundaries.
  Use Ulysses and context parallel sizes of 1, FP16 or BF16, one index KV
  head, index head dimension 128 and compression ratio 1, 4 or 128.
  `qsa_attention_implementation` independently selects the attention kernel.
- Head-first convolution requires `causal_conv1d_fn` and `causal_conv1d_bwd`
  from `fla_npu.ops.ascendc`. It accepts fixed or packed training sequences
  with equal GDN key/value head dimensions and no recurrent cache. The A3/A5
  CANN 9.2.1 Dockerfiles pin FLA-NPU at `a7c608f2`, which exports both APIs.
  These backends use the repository's standard Transformers 5.16.1 pin.

- `rms_norm_implementation: torch_npu` supports grouped offset RMSNorm. The
  affine weight stays per feature, including when the PLE hyper-connection
  normalizes each hidden-size group separately.
- `moe_implementation: chunk_npu` overlaps the two EP token chunks' dispatch
  and return collectives with local grouped expert GEMMs.
- `cross_entropy_loss_implementation: cce_npu` requires Triton-Ascend. It
  projects supervised BF16 hidden rows in vocabulary tiles up to 4096 entries.
- Non-reentrant checkpointing with `selective_recompute_layers > 0` and
  `save_ops: [attn, gdn]` reuses forward kernel outputs. GDN must go through
  the dispatcher in the FLA-NPU stable launcher; a ctypes-only build cannot
  intercept the kernel. With `enable_async_activation: true`, the block input
  and native GDN/fusion-attention results can be offloaded to host memory.
- `model.model_config.text_config.ple_host_offload: true` keeps the persistent
  2D PLE shards on CPU. Lookup deduplicates rows before routing and again on
  each owner. A side stream prefetches one layer ahead. Backward copies only
  row gradients into pinned memory, and SparseAdam updates the touched rows
  after global gradient clipping. The tables and their two FP32 moment tensors
  require roughly three times the local padded table size in host RAM, plus
  transient row and activation buffers. Gloo must be available for the CPU
  mesh. Use a world-covering FSDP2 PLE mesh and streamed pretrained weights.
  DCP saves the full logical parameter and optimizer state across owners.

## Two-device pipeline regression

Run the daily toy regression with:

```bash
pytest -s tests/e2e/test_e2e_parallel.py::test_qwen4_exp_training_smoke
```

On a host with at least two CUDA or NPU devices, the test uses VLM dummy data,
`ple_size=2`, `ep_size=2`, and `ulysses_size=2` to cover FSDP2 streaming load
and a fused-MoE forward/backward training smoke with finite loss and gradient
norm. It does not load the released 95 GiB PLE table.
