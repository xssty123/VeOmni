# Unified A3/A5 CANN 9.2.1 candidates

## Full FLA diagnostic (2026-11-03)

Allowed diff from `115ace9e1f10c5bf9cccd9e3b3475bc497ac2c12`: extend the A5
workflow and diagnostic script with `diagnose-fla`, and update this record.
No Dockerfile, dependency, full-image command, A3 or UT/ST changes.
Plan: implement, check locally, push the unified branch, dispatch one full
FLA diagnostic. Use the same Docker/development skills as the single-op run.

The successful single-op baseline is run
[41603511535](https://github.com/xssty123/VeOmni/actions/runs/41603511535)
at that commit: pinned FLA `a7c608f`, CANN 9.2.1, `ascend950`, `-j4`, exit 0,
260 seconds. Recorded container memory.peak reached 2.79 GiB and all sampled
OOM counters were zero. This produced a single-op native package, not a full
image or full FLA wheel; there is no published image tag/digest for that run.

`diagnose-fla` uses the same prepared container and live resource monitor,
unsets the operator filter and runs the Dockerfile's full `build_wheel.py`
command with four jobs. It retains the wheel, SHA256 and build log, then runs
the same zip, installation and packaged API checks as the Dockerfile. Build
and validation exit codes are recorded separately. It does not build
TorchCodec, publish an image or run NPU computation. Select this mode on the
existing A5 workflow; its artifact is `a5-diagnose-fla`.

Local checks passed: pinned `make quality`, YAML and mode/job isolation,
Bash/embedded Python syntax, and mocked full-wheel success, compile failure
and API failure. The latter checks verify that the operator filter is unset,
`-j4` reaches the builder, and compile/validation exits and logs are preserved.

## A5 single-operator diagnostic (2026-11-03)

Plan: record the allowed diff, implement and locally check the diagnostic,
then push `codex/ascend-cann921` to the personal fork and dispatch one
`diagnose-kda` run. Skills: `veomni-ascend-docker`, `veomni-develop`.

Comparison base: `0f199b87`. The A5 source-build baseline is `83e80c73`
(not a successful image); runs `41452062154` and `41480937068` lost runner
communication. The known-good A3 packaging commit/run/digest below remain
historical evidence, not an A5 acceptance baseline.

Allowed changes:

| File | Allowed change |
| --- | --- |
| A5 9.2.1 Dockerfile | Name the existing dependency stage `fla-build-env`; inherit it in the final stage before FLA compilation |
| A5 candidate workflow | Add `full` (default) / `diagnose-kda` modes; isolated diagnostic job; fix kernel-log timestamp |
| `docker/ascend/tests/diagnose_a5_kda.sh` | Add single-op build and live host/container resource capture, exit status and artifacts |
| This document | Record diagnostic scope, execution and evidence |

Preserve all full-image dependency versions, install commands/order, FLA SHA,
full-operator compilation and `-j4`. Preserve A3, UT/ST and release workflows.
The diagnostic uses the same dependency stage, then the same source checkout,
requirements installation and build-only preflight. It calls native `build.sh`
with `--ops=chunk_kda_bwd_intra`, `--soc=ascend950`, `--pkg`, and `-j4`.
It neither produces a validated full wheel nor publishes an image.

Select `docker-build-ascend-a5-candidate`, branch `codex/ascend-cann921`,
mode `diagnose-kda`. No new default-branch workflow registration is needed.
Host memory, highest-RSS processes and container cgroup counters are sampled
every five seconds and streamed to Actions. These are samples, not a guaranteed
instantaneous peak. A lost runner can still prevent final artifact upload.
Diagnosis records are separate from full-image packaging smoke records.

Local checks: repository-pinned `make quality`, YAML/Bash/embedded Python
syntax, and a mechanical comparison proving the Dockerfile differs only in
stage declarations passed. Mocked host execution verified success, compiler
exit 137, and preparation failure preserve their exit codes, distinguish an
unstarted compiler, write result records and clean up the container. These
checks do not validate the ARM64 compiler; the dispatched run supplies that
evidence.

## Scope and allowed diff (2026-11-03)

The shared development branch is `codex/ascend-cann921`, based on upstream
`3ba1d9e621c95592352a33fb23287dfaf4b35dc9`. A3 and A5 Docker work, followed by
future UT/ST adaptation, belong on this branch. This change covers Docker only;
existing UT/ST branches, workflows and tests are not merged or changed yet.

| Item | Baseline | Allowed change |
| --- | --- | --- |
| A3 Dockerfile | `4e1ea454da0fc69ef1535189ee97b3626cb5f1a5`, CANN 9.2.0 | Add versioned 9.2.1 file; replace only base tag/header |
| A5 Dockerfile | `83e80c73d292707ff3b5f5f8560198b5a7481a36`, CANN 9.2.1 | Carry over unchanged |
| Candidate workflows | Unified workflow at `d5511f5e` | Separate A3 and A5 manual entry points, each with a fixed target; preserve target-specific paths, tag/cache/artifacts and CANN 9.2.1 |
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

## Candidate workflows

Select the target by choosing its independent workflow; neither entry has a
`target` input:

| Target | Workflow display name and filename |
| --- | --- |
| A3 | `docker-build-ascend-a3-candidate` / `docker-build-ascend-a3-candidate.yml` |
| A5 | `docker-build-ascend-a5-candidate` / `docker-build-ascend-a5-candidate.yml` |

Both workflows use the same development branch and packaging-smoke script.
Each fixes its own `TARGET` internally, verifies the image on a fresh runner by
digest, and retains separate tags, cache scopes and artifact names. The original
formal A2/A3 build workflows are unchanged.

Before manual dispatch, the workflow file must exist on the fork's default
branch with `workflow_dispatch`. Register the new A3 candidate entry there when
publishing these workflow changes; this local change does not modify fork `main`.
After branch push and entry registration, use:

```bash
gh workflow run docker-build-ascend-a3-candidate.yml --repo xssty123/VeOmni \
  --ref codex/ascend-cann921
gh workflow run docker-build-ascend-a5-candidate.yml --repo xssty123/VeOmni \
  --ref codex/ascend-cann921
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
strict YAML parsing, Bash syntax for each workflow's eight run blocks and both
Dockerfiles' 19 RUN commands, embedded Python syntax and smoke CLI parsing.
Mechanical comparison confirms the A3 base/header-only version change and the
unchanged A5 Dockerfile. Local execution checks cover both target source/tag
selectors and success/missing-smoke result records; these use synthetic metadata
only and are not image smoke or NPU results. No dependencies were installed into
the project environment and no full training/test suite was run on this Mac.
