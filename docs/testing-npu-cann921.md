# A3/A5 CANN 9.2.1 UT/ST migration

## Plan and allowed diff

Development follows `/veomni-develop`; an upstream PR will additionally require `/veomni-review`.

1. Develop Docker and UT/ST together on `codex/ascend-cann921` at `70360904` for one eventual PR. Port the legacy coverage from `ci/npu-ut-st-coverage` at `c5bf89b0`; the Docker branch originally used source base `3ba1d9e6`. Main `e6486d3847536d6a5cf72a433040914b69f1957d` was merged locally on 2026-11-07 (`7a8760db`).
2. Apply PR #1369 at `a8d357cec86643ab8fec12acc6c60a26bc7d1d29` as an explicit, unmerged dependency.
3. Restore the image wheel stack after uv sync, update FLA contracts, port legacy coverage and declare distributed resource plans.
4. Compare all 59 legacy UT selections and four ST selections against main; reuse shared tests, restore missing metadata contracts, and retain both SwapAdamW manifest additions.
5. Validate static checks and focused CPU tests locally, then run A3 and A5 UT/ST on hardware and retain scheduler reports. No push is authorized for this step.

Allowed environment changes: the candidate test setup restores Python 3.12, torch 2.10.0, torch-npu 2.10.0.post7, Triton 3.6.0 then Triton-Ascend 3.6.0, the target-specific FLA 26.10.0 wheel and TorchCodec 0.10.0 source from the image. The setup also installs TorchCodec build prerequisites (`pybind11==3.1.0`, ninja, cmake, setuptools, wheel) and Triton ELF parsing (`pyelftools`). Preserve the project lockfile, all other dependency choices, existing upstream workflows and Dockerfiles. No image publication is part of this change.

## Image baseline

- A3: `ghcr.io/xssty123/veomni-ascend-candidate@sha256:33a0777d684a06fb812992a06d87e533b6f6f8c58b382c66f11373450cd7b73f`
- A3 build: https://github.com/xssty123/VeOmni/actions/runs/42175438619 ; Docker/source commit `70360904303d914abab91bc0ad2101892ab05fa4`, FLA commit `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`.
- A5 user-provided image: `ghcr.io/xssty123/veomni-ascend-candidate:a5-cann9.2.1-local-20261105-j16`; resolved read-only from GHCR to `sha256:369b251a90d189051d0f95038345440869f105d6753395958223a11ed89aee95` on 2026-11-07. The local workflows pin this digest; retain wheel checksums on the test host.
- A3 packaging/import smoke passed; its evidence explicitly reports dependency conflicts and no NPU execution. This is not a UT/ST pass.

## Coverage migration

- Model forward/loader parity and return-log-probability tests follow the new `tests/models/base` helpers.
- EP LoRA trainer integration is now part of the shared save/load/resume test; its NPU entry requests 2/4 devices.
- Muon dense and MoE parity, both zero-comm modes and the real-builder smoke reuse `test_muon_fsdp2_parity`, each reserving four devices.
- Sequence-classification loss reuses the current model-loss and reduction suites. GDN metadata was missing: model tests mock `precompute_varlen_metadata`, so they cannot replace its real contract test. The restored CPU test imports the production FLA adapter directly, without mocking Torch/NPU/Triton packages.
- GDN uses instance-local `ops_config`, FLA `chunk_gated_delta_rule_fwd/bwd`, BF16 head dimensions 128, and chunk size 64. Numerics compare independent eager results, including packed boundaries and gradients.
- Existing A2 upstream workflows retain their images and runner labels. Candidate A3/A5 validation uses the shared suite runner inside the chosen image.

- The old context-parallel directory is now DeepSeek-V4 CUDA/TileLang coverage; it stays in GPU CI. Shared sequence-classification tests moved to `tests/models/base/test_loss_utils.py`; fused NPU LoRA numerics moved to `tests/ops/moe_experts_lora/`.

## Main/legacy coverage audit (2026-11-07)

Source: `ci/npu-ut-st-coverage@c5bf89b0`, compared with main `e6486d38` and the migrated candidate branch. The table follows the 59 commands in the legacy A5 UT workflow. Directory selection means the runner collects current tests and respects their launch plans/skips; it does not mean every case runs on NPU. Replacements below identify preserved coverage intent, not identical historical test counts or hardware results.

| # | Legacy UT path (under tests/) | Current coverage | Selection | Decision |
|---|---|---|---|---|
| 1 | `ops/test_fla_npu_installation_contract.py` | [ops/test_fla_npu_installation_contract.py](../tests/ops/test_fla_npu_installation_contract.py) | `npu-unit-29` | Adapted to target-specific FLA 26.10 distribution plus stable AscendC fwd/bwd API. Also used as fast-fail candidate preflight. |
| 2 | `parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses_npu.py` | [parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses_npu.py](../tests/parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses_npu.py) | `npu-gdn-ulysses` | Port instance-local ops config, SP/DP meshes, BF16 K=V=128, packed tail and forward/backward determinism; reserve two devices. |
| 3 | `parallel/ulysses/test_op_wrapper.py` | [ops/base/test_op_entry.py](../tests/ops/base/test_op_entry.py)<br>[ops/base/test_ops_config.py](../tests/ops/base/test_ops_config.py)<br>[ops/rms_norm/test_rms_norm.py](../tests/ops/rms_norm/test_rms_norm.py)<br>[ops/rope/test_rope.py](../tests/ops/rope/test_rope.py) | `npu-unit-29` | Use registry rejection/resolution and numerical tests; old global wrapper API was removed. |
| 4 | `parallel/context_parallel` | [parallel/context_parallel](../tests/parallel/context_parallel) | `GPU only` | Excluded from NPU: current directory is DeepSeek-V4 CUDA/TileLang coverage, retained in GPU CI. |
| 5 | `models/test_deepseek_v4_liger_ops.py` | [ops/moe_experts/test_npu_clamped_swiglu.py](../tests/ops/moe_experts/test_npu_clamped_swiglu.py)<br>[ops/rms_norm/test_rms_norm.py](../tests/ops/rms_norm/test_rms_norm.py)<br>[ops/swiglu_mlp/test_swiglu_mlp.py](../tests/ops/swiglu_mlp/test_swiglu_mlp.py)<br>[models/transformers/deepseek_v4/test_deepseek_v4.py](../tests/models/transformers/deepseek_v4/test_deepseek_v4.py) | `npu-unit-08`, `npu-unit-29` | Use current per-op dispatch/numerics and model tests; old OpSlot/Liger binding assertions no longer describe the runtime. |
| 6 | `models/test_generated_call_site_signatures.py` | [models/transformers/test_generated_call_site_signatures.py](../tests/models/transformers/test_generated_call_site_signatures.py) | `device_api_check.yml` | Moved to device_api_check.yml CPU gate; intentionally excluded from both accelerator manifests. |
| 7 | `models/test_models_logits_equal_v5_npu.py` | [models/test_models_logits_equal_v5_npu.py](../tests/models/test_models_logits_equal_v5_npu.py) | `npu-unit-08` | Retain explicit NPU forward/loader tests with shared oracle helpers. Qwen3.5 here uses full_attention, so it does not count as GDN coverage. |
| 8 | `models/test_return_log_probs_e2e_npu.py` | [models/test_return_log_probs_e2e_npu.py](../tests/models/test_return_log_probs_e2e_npu.py) | `npu-unit-08` | Retain NPU Qwen3/Qwen3-VL consumer/backward/distillation checks; share main helper assertions. Main base entry is CUDA-only. |
| 9 | `utils/test_rank0_load_and_broadcast_weights.py` | [utils/test_rank0_load_and_broadcast_weights.py](../tests/utils/test_rank0_load_and_broadcast_weights.py) | `npu-rank0-broadcast` | Reuse current file and its current parameters. |
| 10 | `ops/test_chunk_loss.py` | [ops/loss/test_cross_entropy_loss.py](../tests/ops/loss/test_cross_entropy_loss.py)<br>[models/base/test_chunk_logprobs.py](../tests/models/base/test_chunk_logprobs.py) | `npu-unit-08`, `npu-unit-29` | Reuse current loss/log-probability tests. |
| 11 | `ops/test_chunk_topk_distill.py` | [models/base/test_chunk_topk_distill.py](../tests/models/base/test_chunk_topk_distill.py) | `npu-unit-08` | Reuse current model-loss helper tests. |
| 12 | `ops/test_chunked_fused_linear_grad_gating.py` | [models/base/test_chunked_fused_linear_grad_gating.py](../tests/models/base/test_chunked_fused_linear_grad_gating.py) | `npu-unit-08` | Reuse current gradient gating tests. |
| 13 | `ops/test_eager_kernels_cpu.py` | [ops/loss/test_load_balancing_loss.py](../tests/ops/loss/test_load_balancing_loss.py)<br>[ops/moe_experts/test_moe_experts.py](../tests/ops/moe_experts/test_moe_experts.py)<br>[ops/base/test_op_entry.py](../tests/ops/base/test_op_entry.py) | `npu-unit-29` | Reuse eager loss semantics, routing-order references and registry resolution; old bind flags were removed. |
| 14 | `ops/test_expert_scatter_index.py` | [ops/moe_experts/test_expert_scatter_index.py](../tests/ops/moe_experts/test_expert_scatter_index.py) | `npu-unit-29` | Moved unchanged in purpose. |
| 15 | `ops/test_fused_load_balancing_loss.py` | [ops/loss/test_load_balancing_loss.py](../tests/ops/loss/test_load_balancing_loss.py) | `npu-unit-29` | Reuse current eager/fused forward/backward comparisons. |
| 16 | `ops/test_gdn_ascend_gate.py` | [ops/base/test_builtin_entries.py](../tests/ops/base/test_builtin_entries.py)<br>[ops/base/test_op_entry.py](../tests/ops/base/test_op_entry.py) | `npu-unit-29` | Use exact registered rows plus device/package resolution; old npu_ascendc name is now fla_npu. |
| 17 | `ops/test_gdn_varlen_metadata.py` | [ops/gated_delta_rule/test_fla_npu_metadata.py](../tests/ops/gated_delta_rule/test_fla_npu_metadata.py) | `npu-unit-29` | RESTORED: real CPU metadata, aligned and partial packed sequences; current host-only runtime contract, without hardware-module mocks. |
| 18 | `ops/test_kernel_registry_sanity.py` | [ops/base/test_builtin_entries.py](../tests/ops/base/test_builtin_entries.py)<br>[ops/base/test_op_import.py](../tests/ops/base/test_op_import.py)<br>[ops/base/test_op_config_names.py](../tests/ops/base/test_op_config_names.py) | `npu-unit-29` | Reuse current exact catalog, import and configured-name guards. |
| 19 | `ops/test_npu_kernels_extended.py` | [ops/rms_norm/test_rms_norm.py](../tests/ops/rms_norm/test_rms_norm.py)<br>[ops/rope/test_rope.py](../tests/ops/rope/test_rope.py)<br>[ops/gated_delta_rule/test_gated_delta_rule.py](../tests/ops/gated_delta_rule/test_gated_delta_rule.py)<br>[ops/moe_experts/test_npu_clamped_swiglu.py](../tests/ops/moe_experts/test_npu_clamped_swiglu.py) | `npu-unit-29` | Reuse NPU production shapes, zero/epsilon/gating cases, partial RoPE and clamped SwiGLU in current family suites; not a promise of identical legacy parameter counts. |
| 20 | `ops/test_seqcls_loss_npu.py` | [models/base/test_loss_utils.py](../tests/models/base/test_loss_utils.py)<br>[distributed/test_reduce_loss.py](../tests/distributed/test_reduce_loss.py) | `npu-unit-08`, `npu-unit-21` | Do not duplicate: logits/hidden-weight gradients, priority, valid-token reduction, validation and all-padding graph-zero already covered. These are CPU semantics, not fused NPU acceptance. |
| 21 | `distributed/test_muon_zero_comm_shard.py` | [distributed/test_muon_zero_comm_shard.py](../tests/distributed/test_muon_zero_comm_shard.py) | `npu-muon-shards` | Reuse current file and its current parameters. |
| 22 | `optim/test_muon_unit.py` | [optim/test_muon_unit.py](../tests/optim/test_muon_unit.py) | `npu-muon-unit` | Reuse main strict tolerance; do not carry over unsupported relaxation to 5e-3. |
| 23 | `optim/test_muon_fsdp2_parity.py` | [optim/test_muon_fsdp2_parity.py](../tests/optim/test_muon_fsdp2_parity.py) | `npu-muon-dense-parity` | Select only dense parity here; MoE zero-comm modes and builder smoke run through the NPU wrapper to replace fused_triton with fused_npu. |
| 24 | `optim/test_muon_fsdp2_smoke_npu.py` | [optim/test_muon_fsdp2_smoke_npu.py](../tests/optim/test_muon_fsdp2_smoke_npu.py) | `npu-muon-smoke` | Retain NPU wrapper around main MoE parity and builder smoke, both zero-comm modes; reserve four devices. |
| 25 | `optim/test_optimizer_vlm_param_groups.py` | [optim/test_optimizer_vlm_param_groups.py](../tests/optim/test_optimizer_vlm_param_groups.py) | `npu-optimizer-vlm` | Reuse current file and its current parameters. |
| 26 | `lora/test_target_mapping.py` | [lora/test_target_mapping.py](../tests/lora/test_target_mapping.py) | `npu-lora-target-mapping` | Reuse current file and its current parameters. |
| 27 | `lora/test_veomni_lora_moe_native.py` | [lora/test_veomni_lora_moe_native.py](../tests/lora/test_veomni_lora_moe_native.py) | `npu-unit-36` | Reuse current file and its current parameters. |
| 28 | `lora/test_moe_lora_eager.py` | [lora/test_moe_lora_eager.py](../tests/lora/test_moe_lora_eager.py) | `npu-lora-eager` | Reuse current file and its current parameters. |
| 29 | `lora/test_moe_lora_trainer.py` | [lora/test_moe_lora_trainer.py](../tests/lora/test_moe_lora_trainer.py) | `npu-lora-trainer` | Reuse current file and its current parameters. |
| 30 | `lora/test_moe_lora_ep2_npu.py` | [lora/test_moe_lora_ep2_npu.py](../tests/lora/test_moe_lora_ep2_npu.py) | `npu-lora-ep` | Retain NPU fused shared/independent EP and reset tests using shared alignment helpers; reserve 2/4 devices. |
| 31 | `parallel/ulysses/test_ulysses.py` | [parallel/ulysses/test_ulysses.py](../tests/parallel/ulysses/test_ulysses.py) | `npu-unit-01` | Reuse current file and its current parameters. |
| 32 | `parallel/ulysses/test_all_gather.py` | [parallel/ulysses/test_all_gather.py](../tests/parallel/ulysses/test_all_gather.py) | `npu-unit-02` | Reuse current file and its current parameters. |
| 33 | `parallel/ulysses/test_slice_input_tensor.py` | [parallel/ulysses/test_slice_input_tensor.py](../tests/parallel/ulysses/test_slice_input_tensor.py) | `npu-unit-03` | Reuse current file and its current parameters. |
| 34 | `parallel/encoder_data_balance/test_balance_reverse.py` | [parallel/encoder_data_balance/test_balance_reverse.py](../tests/parallel/encoder_data_balance/test_balance_reverse.py) | `npu-unit-04` | Reuse current file and its current parameters. |
| 35 | `parallel/encoder_data_balance/test_balance_sorting_algo.py` | [parallel/encoder_data_balance/test_balance_sorting_algo.py](../tests/parallel/encoder_data_balance/test_balance_sorting_algo.py) | `npu-unit-05` | Reuse current file and its current parameters. |
| 36 | `parallel/test_ddp_sp_grad_sync.py` | [parallel/test_ddp_sp_grad_sync.py](../tests/parallel/test_ddp_sp_grad_sync.py) | `npu-unit-06` | Reuse current file and its current parameters. |
| 37 | `parallel/test_parallel_state_registry.py` | [parallel/test_parallel_state_registry.py](../tests/parallel/test_parallel_state_registry.py) | `npu-unit-07` | Reuse current file and its current parameters. |
| 38 | `models/test_model_registry.py` | [models/base/test_auto_registry.py](../tests/models/base/test_auto_registry.py) | `npu-unit-08` | Use current model registry suite. |
| 39 | `models/test_models_patch.py` | [models/test_models_patch.py](../tests/models/test_models_patch.py) | `npu-unit-08` | Reuse current file and its current parameters. |
| 40 | `models/test_padded_packed_loss.py` | [models/base/test_loss_utils.py](../tests/models/base/test_loss_utils.py) | `npu-unit-08` | Use packed-target loss/gradient semantics. Legacy whole-model FlashAttention test was CUDA-only and skipped on NPU; this is not an identical full-model replacement. |
| 41 | `models/test_vlm_trainer.py` | [trainer/test_vlm_trainer.py](../tests/trainer/test_vlm_trainer.py) | `npu-unit-10` | Reuse relocated trainer suite. |
| 42 | `trainer/test_aux_metrics.py` | [trainer/test_aux_metrics.py](../tests/trainer/test_aux_metrics.py) | `npu-unit-11` | Reuse current file and its current parameters. |
| 43 | `lora/test_vlm_lora.py` | [lora/test_vlm_lora.py](../tests/lora/test_vlm_lora.py) | `npu-unit-14` | Reuse current file and its current parameters. |
| 44 | `models/test_checkpoint_tensor_converter.py` | [models/base/test_checkpoint_tensor_converter.py](../tests/models/base/test_checkpoint_tensor_converter.py) | `npu-unit-08` | Reuse relocated converter suite. |
| 45 | `utils/test_extra_parallel_clip_grad_norm.py` | [utils/test_extra_parallel_clip_grad_norm.py](../tests/utils/test_extra_parallel_clip_grad_norm.py) | `npu-unit-16` | Reuse current file and its current parameters. |
| 46 | `utils/test_moe_ep_sharded_load_matrix.py` | [utils/test_moe_ep_sharded_load_matrix.py](../tests/utils/test_moe_ep_sharded_load_matrix.py) | `npu-unit-17` | Reuse current file and its current parameters. |
| 47 | `checkpoints/test_checkpoint_callback.py` | [checkpoints/test_checkpoint_callback.py](../tests/checkpoints/test_checkpoint_callback.py) | `npu-unit-19` | Reuse current file and its current parameters. |
| 48 | `checkpoints/test_dcp_checkpointer.py` | [checkpoints/test_dcp_checkpointer.py](../tests/checkpoints/test_dcp_checkpointer.py) | `npu-unit-19` | Reuse current file and its current parameters. |
| 49 | `checkpoints/test_trainer_saveload.py` | [checkpoints/test_trainer_saveload.py](../tests/checkpoints/test_trainer_saveload.py) | `npu-unit-19` | Reuse current file and its current parameters. |
| 50 | `distributed/test_async_offload_unit.py` | [distributed/test_async_offload_unit.py](../tests/distributed/test_async_offload_unit.py) | `npu-unit-20` | Reuse current file and its current parameters. |
| 51 | `distributed/test_dummy_forward.py` | [distributed/test_dummy_forward.py](../tests/distributed/test_dummy_forward.py) | `npu-unit-22` | Reuse current file and its current parameters. |
| 52 | `data` | [data](../tests/data) | `npu-unit-27` | Reuse current file and its current parameters. |
| 53 | `ops/test_comp.py` | [models/transformers/qwen/test_qwen3_vl_vision_positions.py](../tests/models/transformers/qwen/test_qwen3_vl_vision_positions.py) | `npu-unit-08` | Reuse on-device position interpolation/rotary reference comparisons; no need to restore old timed benchmark wrapper. |
| 54 | `ops/test_moe_hw_gate.py` | [ops/base/test_op_entry.py](../tests/ops/base/test_op_entry.py)<br>[ops/base/test_builtin_entries.py](../tests/ops/base/test_builtin_entries.py)<br>[ops/base/test_platform.py](../tests/ops/base/test_platform.py) | `npu-unit-29` | Use current device/package/architecture requirements and catalog; removed legacy bind API tests are retired. |
| 55 | `ops/test_npu_kernels.py` | [ops/rms_norm/test_rms_norm.py](../tests/ops/rms_norm/test_rms_norm.py)<br>[ops/rope/test_rope.py](../tests/ops/rope/test_rope.py)<br>[ops/gated_delta_rule/test_gated_delta_rule.py](../tests/ops/gated_delta_rule/test_gated_delta_rule.py)<br>[distributed/test_hccl_premul_sum.py](../tests/distributed/test_hccl_premul_sum.py)<br>[ops/base/test_builtin_entries.py](../tests/ops/base/test_builtin_entries.py) | `npu-unit-23`, `npu-unit-29` | Reuse current family numerical tests, HCCL patch tests and catalog. |
| 56 | `utils/test_count_flops.py` | [utils/test_count_flops.py](../tests/utils/test_count_flops.py) | `npu-unit-30` | Reuse current file and its current parameters. |
| 57 | `lora/test_lora_trainer_saveload.py` | [lora/test_lora_trainer_saveload.py](../tests/lora/test_lora_trainer_saveload.py) | `npu-unit-33` | Reuse current file and its current parameters. |
| 58 | `lora/test_moe_lora_fused_npu.py` | [ops/moe_experts_lora/test_moe_experts_lora.py](../tests/ops/moe_experts_lora/test_moe_experts_lora.py) | `npu-unit-29` | Reuse single-rank NPU LoRA numerics; keep separate EP trainer/save-load tests. |
| 59 | `lora/test_moe_lora_ep_sharded_stream_load.py` | [lora/test_moe_lora_ep_sharded_stream_load.py](../tests/lora/test_moe_lora_ep_sharded_stream_load.py) | `npu-unit-35` | Reuse current file and its current parameters. |

### Changes to existing legacy test files

Besides the seven added files accounted for above, the old branch modified nine existing files:

| Legacy modification | Resolution |
|---|---|
| `distributed/test_fsdp_equivalence.py`, `e2e/test_e2e_parallel.py` | Main already removed Qwen3.5 NPU skip marks; reuse the current cases. |
| `lora/utils.py` | Main already shares eager ops configuration helpers; do not copy older field names. |
| `ops/test_gdn_varlen_metadata.py` | Restore real metadata coverage under the new adapter path; drop obsolete import mocks and full-keyset equality between precompute and runtime fallback. |
| `parallel/ulysses/test_op_wrapper.py` | Old wrapper API removed; current registry/config and numerical suites own those contracts. |
| `optim/test_muon_unit.py` | Keep main's numerical assertion, without the legacy tolerance relaxation. |
| `tools/training_utils.py` | Use current backend names and a scoped candidate override for Qwen3.5 dense/MoE; validate override values in the shared helper test. |
| `distributed/test_torch_compile.py`, `parallel/ulysses/test_ulysses.py` | Old diff was formatting only; retain current main versions. |

### ST and manifest reconciliation

| Legacy ST command | Current behavior |
|---|---|
| FLA installation contract | Run candidate preflight before either suite, failing on missing/wrong distribution or API. |
| `tests/e2e/test_e2e_parallel.py` | Shared `npu-e2e-40`, explicit Triton baseline; rerun only Qwen3.5 dense/MoE with `fla_npu`. |
| `tests/distributed/test_fsdp_equivalence.py` | Shared `npu-e2e-41`, same two-backend policy. |
| Explicit `test_wan_dit_parallel_align` rerun | Already included in `npu-e2e-40`; remove the duplicate execution. |

The manifest merge retains all existing names and both upstream additions,
`swap-adamw-unit` and `swap-adamw-device-unit`: 56 selections total.
Current SeedOmni additions are collected through the existing directory entry.
No duplicate NPU-only copy of main's generic loss, op or optimizer tests is added.
The inherited padding-mask entry has an explicit `TRITON_ASCEND_ARCH` environment,
so its overlap with `tests/ops` is intentionally retained. Installation-contract
preflight and suite execution are also intentional: fail early, then record it
in the suite inventory.

### GDN metadata contract

The restored test covers ten CPU cases: both legacy shapes (`[128, 256]` with
four heads and `[64, 128, 32]` with eight heads), packed partial chunks
(`[65, 67]`), and incomplete precomputed metadata. It checks sequence IDs,
per-sequence chunk numbering, all precomputed tensor/list tables, and unchanged
legacy tensor objects. Expected rows come from sequence chunk start positions,
not from another production metadata helper.

FLA 26.10's stable runtime consumes host metadata for chunk size 64. Its fallback
must not create/upload device chunk tables; precomputation still provides the
legacy table superset. Warm resolution must not call Tensor.tolist(), upload
metadata, or rebuild chunk indices. Consequently, requiring cold fallback and
precomputation to have identical keysets would assert the obsolete API behavior.
These checks validate metadata only. NPU FLA forward/backward, packed numerical
comparisons, SP, and training remain separate hardware acceptance requirements.

## Validation status

Main synchronization and the coverage audit are implemented locally. Hardware UT/ST results are pending.

The focused production metadata suite passed **10 tests** on CPU using an isolated
`/tmp/veomni-gdn-metadata-env` with Torch 2.10.0 and Transformers 5.16.1, without
changing the repository environment or lockfile. After the merge, **49 registry/catalog/platform
tests** and **12 CPU scheduler tests** also passed, as did `make quality` (1059
formatted files). The 56-entry manifest was checked as a lossless union of main
and the prior candidate manifest, with unique names and existing paths. Earlier
validation passed Bash syntax, four workflow YAMLs, helper references, and
device-API scans for `tests/`, `veomni/` and `tasks/`.
Full candidate collection/execution still requires the target image and NPU host;
this Mac result is not an A3/A5 numerical or distributed pass.

## Run on the server

Run inside the matching image, with the current checkout mounted writable,
eight NPUs exposed, the host Ascend driver mounted, and CI assets available.
These commands do not build or push Docker images:

```bash
# A3 container
bash scripts/ci/npu/setup_candidate.sh a3
bash scripts/ci/npu/run_candidate.sh a3 unit
bash scripts/ci/npu/run_candidate.sh a3 e2e

# A5 container
bash scripts/ci/npu/setup_candidate.sh a5
bash scripts/ci/npu/run_candidate.sh a5 unit
bash scripts/ci/npu/run_candidate.sh a5 e2e
```

Before testing, set `CI_HF_MODELS_DIR`, `CI_SAMPLES_DIR` and `CI_DATASET_DIR`
to the mounted assets. Local workflow defaults use `/mnt/veomni_ci` on A3
and `/home/z00832855/ci/ut` on A5; adapt mounts/paths to the actual host.
The A5 workflow retains the existing server's device/library mounts. For
`act`, use `act workflow_dispatch -W npu_unit_tests_a3.yml` (or the desired
root YAML) with a compatible ARM runner mapping. The shared scripts also work
without `act` in a running container.

Do not run UT and ST simultaneously in the same checkout: setup recreates
`.venv`, and the runner owns the declared device set. `--collect-only` can be
passed to `run_candidate.sh` to inspect the inventory before execution;
collection is not hardware acceptance. `--serial` runs one batch at a time.
No workflow kills unrelated host Python processes; the scheduler owns cleanup.

Results go to `outputs/npu-candidate/<target>-<suite>-<UTC-time>-<pid>/`.
Each invocation captures source SHA/diff, package freeze, wheel hashes, NPU
information and scheduler inventories/results. Record the pulled image's
RepoDigest alongside these artifacts, especially for the user-built A5 tag.
A passing preflight/import check is not a numerical or distributed test pass.
Review failed cases and skipped reasons before declaring either target accepted.

The A5 setup retains the previous host-scoped `uv sync` certificate exceptions
for GitHub/object storage/PyTorch download hosts. The previous remote wheel
installations are replaced by checksum-checked `/app` wheels, so their pip
TLS exceptions no longer apply. The mounted `/tmp/latest_proxy` is honored;
no proxy credentials are printed. System certificates are used for build tools.
