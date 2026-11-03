# Unified A3/A5 CANN 9.2.1 candidates

## Scope and allowed diff (2026-11-03)

The shared development branch is `codex/ascend-cann921`, based on upstream
`3ba1d9e621c95592352a33fb23287dfaf4b35dc9`. A3 and A5 Docker work, followed by
future UT/ST adaptation, belong on this branch. This change covers Docker only;
existing UT/ST branches, workflows and tests are not merged or changed yet.

| Item | Baseline | Allowed change |
| --- | --- | --- |
| A3 Dockerfile | `4e1ea454da0fc69ef1535189ee97b3626cb5f1a5`, CANN 9.2.0 | Add versioned 9.2.1 file; replace only base tag/header |
| A5 Dockerfile | `83e80c73d292707ff3b5f5f8560198b5a7481a36`, CANN 9.2.1 | Carry over unchanged |
| Candidate workflow | A3 workflow at `4e1ea454` | One manual entry point selecting `a3` or `a5`; target-specific paths, tag/cache/artifacts; CANN 9.2.1 throughout |
| Packaging smoke | A3/A5 scripts at `4e1ea454` | One script selecting the target distribution/artifacts; require CANN 9.2.1 and the matching Python version |
| This document | Historical candidate records | Consolidate branch usage, validation boundaries and immutable evidence |

Preserve torch 2.10.0, torch-npu 2.10.0.post7, Triton/Triton-Ascend 3.6.0,
Transformers 5.16.1, TorchCodec v0.10.0, all remaining dependency versions,
installation commands/order and workarounds. Preserve FLA source
`a7c608f2e3f828779ceeec1fcf155a6e37ede83c`, full-operator compilation,
`FLA_BUILD_JOBS=4`, runner, timeout and resource diagnostics. Do not alter
pyproject.toml, uv.lock, formal publishing workflows, UT/ST or model code.
Upstream changes since the experiment baseline do not change pyproject.toml,
uv.lock or the TorchCodec installer. Image construction still consumes the
selected branch's source, so the new image must be validated again.

## Targets

| Target | Base image | FLA SOC | Distribution | Dockerfile |
| --- | --- | --- | --- | --- |
| A3 | `quay.io/ascend/cann:9.2.1-a3-ubuntu22.04-py3.12` | `ascend910_93` | `flash-linear-attention-npu-a3` | `Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a3` |
| A5 | `quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12` | `ascend950` | `flash-linear-attention-npu-a5` | `Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5` |

Each target uses ARM64 and Python 3.12. The candidate does not claim NPU
correctness from a hosted-runner build or import check.

The A3 9.2.1 registry preflight found ARM64 manifest
`sha256:dd0630abf2f152d4e86f7358b335c754f9c98a5d0f3ab52753f9ab016a8b5d5d`;
its image config records Python 3.12.16. The shared smoke requires Python
3.12.16 and CANN 9.2.1 for both targets. The Dockerfiles retain version tags;
this preflight digest is evidence, not an additional base-image pin.

## Historical evidence and current acceptance

The known-good A3 **9.2.0 packaging baseline** remains frozen:

- Source: `4e1ea454da0fc69ef1535189ee97b3626cb5f1a5`.
- Run: [41572349826](https://github.com/xssty123/VeOmni/actions/runs/41572349826).
- Tag: `ghcr.io/xssty123/veomni-ascend-candidate:a3-cann9.2.0-torch_npu2.10.0.post7-4e1ea454da0f-41572349826-1`.
- Digest: `sha256:3b8cdbf335b1586d86a05894c23a911c38ffaa9b1ab1bf7c1ff71e01fa6cfc12`.
- Observed: CANN 9.2.0, Python 3.12.15, torch 2.10.0, torch-npu 2.10.0.post7,
  Triton/Triton-Ascend 3.6.0, Transformers 5.16.1,
  `flash-linear-attention-npu-a3==26.10.0+unknown.deva7c608f`.
- Build, wheel checksums, Stable-ABI loading and fresh-container imports passed.
  `pip check` reported missing packages/version conflicts; NPU UT/ST were not run.

That record is not acceptance of A3 9.2.1. The A5 9.2.1 run `41452062154`
and A5 9.2.0 run `41480937068` lost hosted-runner communication during FLA
compilation; neither proves OOM or CANN incompatibility. Both new 9.2.1 targets
remain unvalidated until their own build and smoke records pass.

The original `codex/a3-cann920-compare` and `codex/a5-cann921-main` branches
remain historical references. Continue new Docker and future UT/ST work here.

## Candidate workflow

The workflow display name is **docker-build-ascend-candidate**. Its existing
registered filename stays `docker-build-ascend-a5-candidate.yml` to retain the
fork's manual dispatch entry without changing the default branch. The name in
the Actions sidebar may still reflect the default branch's older A5 title.
Select this development branch and choose the `target` input (`a3` by default).
The workflow verifies one target per invocation, on a fresh runner by digest,
and records separate target tags, cache scopes and artifact names.

After the branch is pushed, explicit dispatch commands are:

```bash
gh workflow run docker-build-ascend-a5-candidate.yml --repo xssty123/VeOmni \
  --ref codex/ascend-cann921 -f target=a3
gh workflow run docker-build-ascend-a5-candidate.yml --repo xssty123/VeOmni \
  --ref codex/ascend-cann921 -f target=a5
```

Building and packaging smoke remain separate from target-hardware UT/ST.
Future tests must identify the exact image digest and use the preinstalled
environment when validating that image. Do not substitute a fresh `uv sync`
environment and label it an image acceptance result. Preserve real numerical
checks and record skips, dependency consistency and hardware results separately.

This is a unified development branch, not an already validated upstream PR.
Before the final PR, retain formal Docker/UT/ST content and remove private-fork
candidate machinery and historical execution records from the upstream diff.
No Quay publication, workflow dispatch or PR creation is part of this change.

## Local checks

Passed repository-pinned Ruff 0.13.2 `make quality`, explicit smoke lint/format,
strict YAML parsing, Bash syntax for all eight workflow run blocks and both
Dockerfiles' 19 RUN commands, embedded Python syntax and smoke CLI parsing.
Mechanical comparison confirms the A3 base/header-only version change and the
unchanged A5 Dockerfile. Local execution checks cover both target source/tag
selectors and success/missing-smoke result records; these use synthetic metadata
only and are not image smoke or NPU results. No dependencies were installed into
the project environment and no full training/test suite was run on this Mac.
