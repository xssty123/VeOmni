# A3 CANN 9.2.0 build comparison

Status: local checks passed; build, packaging smoke and A3 NPU validation pending.

## Allowed-diff contract (2026-11-03)

Request: build an A3 candidate to compare with the A5 failures on a hosted ARM runner.
Use the `veomni-ascend-docker` validate workflow: verify base/source, implement the
target substitutions, check locally, push to the private fork and dispatch once.

- Comparison commit: `ac1d8171f58327f872e6a40058972fdbf2ed0735`.
  Its A5 CANN 9.2.0 run
  [41480937068](https://github.com/xssty123/VeOmni/actions/runs/41480937068)
  lost runner communication during FLA compilation. This is not a known-good image.
- Historical known-good Dockerfile:
  `docker/ascend/Dockerfile.ascend_9.2.0_torch_npu2.10.0.post6_a5`
  at `2d66d74c9f3b05a95d44af1ac0a3f2af02b7e26d`,
  [run 35588936993](https://github.com/xssty123/VeOmni/actions/runs/35588936993).
  Image: `ghcr.io/xssty123/veomni-a5:a5-cann9.2.0-beta.2-torch_npu2.10.0.post6-2d66d74c9f3b-35588936993-1`,
  digest `sha256:6655d18e4a6cc2f0ee27f76344803b8fb1c9a60b10e939c11753ae5c6c47fcbd`.
  Its beta CANN and FLA 26.6.0 stack does not validate this new A3 candidate.
- Branch: `codex/a3-cann920-compare`; retain the original A5 branch and files.
- Target: A3 / `ascend910_93` / `linux/arm64`.
- Base: `quay.io/ascend/cann:9.2.0-a3-ubuntu22.04-py3.12`.
  Registry preflight found ARM64 manifest
  `sha256:e1888c48f85a1eba328a513d6f70eb5a8e84e68179ce1ac9d208969137fba476`.
  The config build history records Python 3.12.15, matching the A5 9.2.0 base;
  the runtime smoke checks this exact version.
- Fixed FLA source `a7c608f2e3f828779ceeec1fcf155a6e37ede83c` maps
  `ascend910_93` to distribution `flash-linear-attention-npu-a3` in
  `scripts/fla_npu_artifacts.py:125-155`. Its published minimum-version checks
  permit the selected versions; this is source-contract evidence, not runtime validation.

| File | Authorized change |
| --- | --- |
| New `Dockerfile.ascend_9.2.0_torch_npu2.10.0.post7_a3` | Copy the A5 9.2.0 candidate; substitute the A3 base, SOC, tiered wheel names and artifact labels; remove stale A3 and A5 FLA copies before installation |
| New `tests/a3_image_smoke.py` | Copy the A5 checks; verify the A3 distribution and artifact names, base Python version, and report A3 identity |
| `.github/workflows/docker-build-ascend-a5-candidate.yml` on this branch only | Reuse the registered dispatch entry point, with A3 display name, Dockerfile, smoke script, unique tag/cache/artifacts and result identity |
| This record | Record scope, evidence and local validation |

The workflow filename remains registered on fork `main`, so this comparison needs
no default-branch changes. Dispatch that existing path with `--ref codex/a3-cann920-compare`.
The original A5 workflow on `codex/a5-cann921-main` is unaffected. Do not edit the
formal A3 Quay build workflow or publish to Quay.

Preserve Torch 2.10.0, torch-npu 2.10.0.post7, Triton/Triton-Ascend 3.6.0,
Transformers 5.16.1, TorchCodec v0.10.0, all other dependency pins, installation
order and workarounds. Preserve the exact FLA source and full-operator build,
`FLA_BUILD_JOBS=4`, `ubuntu-22.04-arm`, six-hour timeout, resource logging and
BuildKit configuration. Preserve UT/ST, pyproject.toml, uv.lock and model code.

## Interpretation

This changes both the target-specific CANN base image and generated FLA kernels.
An A3 success would support investigating the A5 build path but would not prove
an A5 hardware defect or isolate a single kernel. Both hosted builds run without
NPU devices. Packaging smoke does not establish real A3 forward/backward correctness.

## Validation plan

Verify the exact Dockerfile substitution against the A5 comparison base, YAML,
embedded Bash and Python syntax, smoke lint/format, repository `make quality`,
and whitespace. Verify the remote SHA before dispatch and report the run URL.
Freeze a new known-good baseline only if build and packaging smoke both succeed.

Local checks passed: repository-pinned Ruff 0.13.2 `make quality`, explicit A3
smoke lint/format, Python syntax, YAML parsing, Bash syntax for all eight workflow
run blocks and 19 Docker RUN commands, exact allowed Docker/workflow substitutions,
Dockerfile/smoke target wiring and unchanged `-j4`. No local image build or NPU
execution was performed. The forthcoming Actions result is the acceptance evidence.
