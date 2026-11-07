# A3/A5 CANN 9.2.1 UT/ST migration

## Plan and allowed diff

Development follows `/veomni-develop`; an upstream PR will additionally require `/veomni-review`.

1. Develop Docker and UT/ST together on `codex/ascend-cann921` at `70360904` for one eventual PR. Port the legacy coverage from `ci/npu-ut-st-coverage` at `c5bf89b0`; the current Docker branch source base is `3ba1d9e6`.
2. Apply PR #1369 at `a8d357cec86643ab8fec12acc6c60a26bc7d1d29` as an explicit, unmerged dependency.
3. Restore the image wheel stack after uv sync, update FLA contracts, port legacy coverage and declare distributed resource plans.
4. Validate static checks and CPU collection locally, then run A3 and A5 UT/ST on hardware and retain scheduler reports.

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
- Sequence-classification loss and GDN metadata checks moved into the current model-loss and op suites; obsolete wrappers must not reference deleted modules.
- GDN uses instance-local `ops_config`, FLA `chunk_gated_delta_rule_fwd/bwd`, BF16 head dimensions 128, and chunk size 64. Numerics compare independent eager results, including packed boundaries and gradients.
- Existing A2 upstream workflows retain their images and runner labels. Candidate A3/A5 validation uses the shared suite runner inside the chosen image.

- The old context-parallel directory is now DeepSeek-V4 CUDA/TileLang coverage; it stays in GPU CI. Shared sequence-classification tests moved to `tests/models/base/test_loss_utils.py`; fused NPU LoRA numerics moved to `tests/ops/moe_experts_lora/`.

## Validation status

Local implementation and static validation are complete. Hardware UT/ST results are pending.

Local checks: `make quality`, Bash syntax, four workflow YAMLs, manifest paths and helper references; 12 CPU scheduler tests passed. Device-API scans passed for `tests/`, `veomni/` and `tasks/`. Torch-dependent pytest collection and execution are unavailable in this Mac environment (Torch is absent; installed Transformers is 5.9.0 rather than the image pin).

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
