# A5 CANN 9.2.0 base-image experiment

Status: prepared for a private-fork build; image and NPU validation pending.

## Allowed-diff contract (2026-11-03)

- Request: change the candidate base image to CANN 9.2.0 and try a build.
- Comparison base: `83e80c73d292707ff3b5f5f8560198b5a7481a36`.
  Run [41452062154](https://github.com/xssty123/VeOmni/actions/runs/41452062154)
  lost hosted-runner communication during FLA compilation. No successful image
  exists for that stack; neither OOM nor CANN incompatibility was established.
- Historical known-good Dockerfile:
  `docker/ascend/Dockerfile.ascend_9.2.0_torch_npu2.10.0.post6_a5`
  at `2d66d74c9f3b05a95d44af1ac0a3f2af02b7e26d`,
  [run 35588936993](https://github.com/xssty123/VeOmni/actions/runs/35588936993).
- Historical image:
  `ghcr.io/xssty123/veomni-a5:a5-cann9.2.0-beta.2-torch_npu2.10.0.post6-2d66d74c9f3b-35588936993-1`,
  digest `sha256:6655d18e4a6cc2f0ee27f76344803b8fb1c9a60b10e939c11753ae5c6c47fcbd`.
  This older beta image does not validate the current dependency stack.
- Target: A5 / `linux/arm64`, Ubuntu 22.04, Python 3.12.

| File | Allowed change |
| --- | --- |
| New `Dockerfile.ascend_9.2.0_torch_npu2.10.0.post7_a5` | Copy the 9.2.1 candidate; change only its header and base tag to `quay.io/ascend/cann:9.2.0-950-ubuntu22.04-py3.12` |
| `docker-build-ascend-a5-candidate.yml` | Select the new Dockerfile; update candidate tag, cache scope, summary and result path; pass expected CANN 9.2.0 to smoke |
| `tests/a5_image_smoke.py` | Parameterize the expected CANN version, retaining 9.2.1 as the default; match the base-image Python patch version (9.2.0: 3.12.15, 9.2.1: 3.12.16) |
| This record | Describe scope and validation evidence |

Preserve the 9.2.1 Dockerfile, all dependency pins and installation commands/order,
FLA revision `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`, full-operator build,
`FLA_BUILD_JOBS=4`, workarounds, wheel retention and API checks. Preserve the runner,
timeout, resource logging and BuildKit settings. No UT/ST, formal publication,
model code, pyproject.toml or uv.lock changes. Push only to the existing private-fork
candidate branch and dispatch its candidate workflow; no upstream PR or Quay promotion.

The base-image replacement can change packages inherited from that image. It is a
base-image comparison, not proof isolating one CANN shared library. Validate the
resulting package state and exact image digest if the build succeeds.

Registry preflight confirmed an ARM64 manifest for the requested 9.2.0 tag:
`sha256:5c7a84895b65f89030b0b13c80ca9e1436b7f4ba6da8312dc9c9f6c535fa4070`.
Its config build history records Python 3.12.15; runtime smoke must verify it.

## Mechanical validation

Compare the new Dockerfile with the 9.2.1 file after replacing `9.2.1` with `9.2.0`;
the contents must be identical. Check workflow YAML, embedded Bash, smoke-script
syntax, repository quality and whitespace before committing.

Local checks passed: repository-pinned Ruff 0.13.2 `make quality`, explicit smoke
script lint/format, YAML parsing, Bash syntax for all eight workflow run blocks
and 19 Docker RUN commands, Python syntax, exact allowed Dockerfile comparison,
workflow file/CANN expectation alignment, and `git diff --check`.
No local image build or NPU execution was performed; the Mac has no Docker CLI.
The dispatched Actions run must provide the image-build and packaging-smoke result.
