# A5 CANN 9.2.1 candidate for VeOmni #1369

Status: local draft; no build, registry push, or NPU validation has been run.
Scope: A5 ARM64 Dockerfile and manual private-fork build workflow only.

## Allowed-diff contract

- Original draft base: `ci/npu-ut-st-coverage`, `c5bf89b068d0fdc4aae243edc59337af2c0df0d4`.
- Migration target: `codex/a5-cann921-main`, based on upstream `main`
  `f1534a207cdc2d050dcefd5cee24fdbf70a22831`, fetched on 2026-10-31.
  Cherry-pick only these four candidate files; do not carry over historical
  Docker or UT/ST commits. The fixed PR build source and dependency pins stay unchanged.
- Known-good Docker build: `2d66d74c9f3b05a95d44af1ac0a3f2af02b7e26d`,
  [Actions run 35588936993](https://github.com/xssty123/VeOmni/actions/runs/35588936993).
  The old A5 Dockerfile and workflow match the original draft base at `c5bf89b0`.
- Known-good image: `ghcr.io/xssty123/veomni-a5:a5-cann9.2.0-beta.2-torch_npu2.10.0.post6-2d66d74c9f3b-35588936993-1`.
- Recorded baseline digest: `sha256:6655d18e4a6cc2f0ee27f76344803b8fb1c9a60b10e939c11753ae5c6c47fcbd`.
  This is historical build evidence, not acceptance of the new stack.
- New Dockerfile: `docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5`.

| Change | Previous | Candidate |
| --- | --- | --- |
| Base | CANN 9.2.0-beta.2 / Ubuntu 22.04 / Python 3.12 | CANN 9.2.1 / Ubuntu 22.04 / Python 3.12.16; versioned image tag |
| VeOmni source | build context from old coverage branch | PR #1369 head `0ec0d8829a85fa82abcf545857517d01542e703c` |
| torch-npu | 2.10.0.post6 | 2.10.0.post7 |
| Triton / Triton-Ascend | 3.5.0 / 3.2.2 | 3.6.0 / 3.6.0 (not post2) |
| Transformers | 5.9.0 | 5.16.1, matching PR source |
| FLA source | `cf1947bff0e75522f9cc0bba5814e6705dc80aae` | `a7c608f2e3f828779ceeec1fcf155a6e37ede83c` |
| FLA build/check | legacy dispatcher API, untiered wheel | `ascend950`, tiered A5 wheel, stable GDN forward/backward API |
| Artifact provenance | freeze + wheel checksums | retain those, add source/base provenance and dependency-check report |
| Build workflow | old manual A5 build | separate manual candidate build, fresh-runner digest smoke, result artifact |

Preserved: Torch 2.10.0, TorchCodec v0.10.0 and the baseline ARM installer,
uv 0.11.16, PyArrow 21.0.0, pandas 2.3.2, Ubuntu package list, mirror override,
thread limits, community-Triton-before-Ascend installation order, and `--no-deps`
for replacement wheels. Dependencies supplied by VeOmni follow the fixed PR
source; there is no extra independent dependency upgrade. Existing Dockerfiles,
build/publish workflows, UT/ST workflows, pyproject.toml, uv.lock and model/kernel
code are unchanged.

## Simplification contract (2026-10-31)

Starting draft: `546a312b1074faac853f88e13370061bead0755d`; the known-good
historical build above remains the validation baseline. Follow the historical
Dockerfile's section order and direct installation style.

Allowed changes: remove candidate commit arguments and source gating from the
Dockerfile; pin FLA directly in its installation block; move candidate provenance
and full smoke execution to the workflow's fresh-container validation. Keep
simple final version assertions, package freeze, wheel checksums, and the upstream
FLA build/API checks. Adapt only this candidate workflow, smoke script and document
to the new locations. Preserve all dependency versions, installation
commands/order, workarounds, fixed PR source, UT/ST and other workflows.
The subsequent user-requested cleanup starts from `bbf9f245`: remove the two
header comments about PR provenance and validation status, and use the CANN
version tag without a base-image digest. All other Dockerfile content stays unchanged.

## Implementation and validation sequence

1. Add the versioned Dockerfile, stable-API image smoke script, and candidate workflow.
2. Check Python/shell/YAML/workflow syntax and the diff against the allowed scope.
3. After separate build authorization, run the manual workflow in the private fork.
4. Run A3/A5 UT/ST separately after integrating #1369. A hosted ARM smoke does
   not execute NPU kernels or validate numerics, device communication, or training.
5. Freeze the successful commit, source commit, tag, digest, reports and runtime
   results as the next baseline. Quay promotion is outside this workflow.

## Build context and outputs

Use `.github/workflows/docker-build-ascend-a5-candidate.yml`. It checks out this
Docker branch and the fixed upstream PR source separately. It stages only the
historical TorchCodec installer into the PR context;
the new Dockerfile stays on the Docker branch. This avoids installing new FLA
against the old coverage branch's legacy GDN adapter. Both commit identities are
recorded. No floating PR head, `main`, or FLA branch is used for the build.

The candidate is pushed only to `ghcr.io/xssty123/veomni-ascend-candidate` under
a unique A5 tag. A second ARM runner pulls that exact digest into a fresh
container and checks its preinstalled Python environment without `uv sync`.
The final job uploads `ascend-candidate-result.json` plus available smoke reports.
Publication occurs only if a human later dispatches the workflow; editing these
files does not dispatch it. The workflow must first be made available on the
fork's default branch before GitHub can offer manual dispatch.

Retained under `/app`: TorchNPU/Triton/Triton-Ascend/FLA A5 wheels,
`torchcodec/` source and installer, `a5-image-requirements.txt`,
`a5-ci-wheels.sha256`, and the upstream `check_packaged_wheel_api.py`.
The workflow records source commits as image labels and saves `docker image inspect`
output. The fresh-container job mounts the smoke script and inspection JSON read-only;
the smoke report includes source identities and `pip check` output/exit status.
The Dockerfile has no dependency on the candidate workflow's smoke script or commit
arguments and keeps only inline build/API/version checks.
FLA distribution: `flash-linear-attention-npu-a5`; Python import: `fla_npu`.
Do not require a literal `26.10.0` version string: the selected source declares
`26.10.0.dev0`, and the build may add a local version suffix. The commit and
wheel checksum are the identity.

## Known limits retained for this draft

- CANN 9.2.1 + Triton-Ascend 3.6.0 + this FLA commit is not yet validated here.
- Triton-Ascend's SciPy/decorator pins conflict with VeOmni's requirements.
  Keep the existing no-deps overlay policy for this draft; record `pip check`
  output and exit status rather than claiming dependency consistency. This
  report is informational, while version/API/library smoke failures are fatal.
- The PR source declares torch-npu 2.10.0; the selected post7 wheel is installed
  afterward. Any resulting metadata inconsistency is recorded, not hidden.
- No other transitive or build-tool versions are newly frozen in this draft.
- Old UT/ST scripts still match the untiered FLA wheel and legacy API. They
  cannot be used unchanged to accept this candidate. CI integration is separate.

## Mechanical comparison

Compare the new Dockerfile with the historical baseline from
`git show 2d66d74c9f3b05a95d44af1ac0a3f2af02b7e26d:docker/ascend/Dockerfile.ascend_9.2.0_torch_npu2.10.0.post6_a5`,
and run `git diff --check`. That historical A5 file is absent from upstream main.
Expected tracked scope: this document, the new versioned Dockerfile,
`docker/ascend/tests/a5_image_smoke.py`, and the new candidate workflow only.

Initial draft checks on 2026-10-31 (before simplification): actionlint 1.7.12 (shellcheck/pyflakes unavailable),
YAML parse, `bash -n` for all six workflow shell blocks and 22 Docker RUN blocks,
embedded Python syntax, smoke-script Ruff lint/format, and new-file whitespace
checks passed. A wrong-platform invocation failed and wrote a failure report as
expected. The baseline Dockerfile/workflow and UT/ST files are unchanged.
Docker is unavailable on this execution host, so no image build or runtime
import check was performed. TorchCodec's v0.10.0 source declares 0.10.0a0 and
appends a Git suffix; the smoke checks that source-build version, not a release
wheel's literal 0.10.0 metadata.

Simplified draft checks on 2026-10-31: repository-pinned Ruff 0.13.2
`make quality`, smoke lint/format, actionlint 1.7.12, YAML parse, all six workflow
shell blocks, 19 Docker RUN blocks, embedded Python syntax, and whitespace checks
passed. A CPU-only check confirmed rejection of mismatched Docker commit labels.
Dependency pins and the FLA source commit are unchanged. No Docker build or NPU
validation has been run.
