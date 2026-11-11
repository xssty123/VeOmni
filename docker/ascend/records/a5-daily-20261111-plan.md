# A5 daily-wheel candidate plan and allowed diff

## Plan

1. Use `veomni-ascend-docker` for version selection, coordinated image changes, GHCR build and packaging smoke.
2. Resolve immutable publisher artifacts and review the six-component version table before any image build or compilation diagnostic.
3. Add the versioned A5 Dockerfile, reuse the already registered fork candidate workflow, provide matching smoke checks and documentation.
4. Run static validation and `make quality`, commit and push only the task branch, verify its remote SHA, then dispatch the confirmed candidate build.
5. Use `veomni-ci` after the user supplies this run's A5 SSH host, own task directory, devices and asset mounts. Pin fresh upstream main for both unchanged uv and image UT/ST; record all four results.
6. Report build, dependency consistency, packaging smoke and NPU acceptance separately. No Quay publication or upstream PR is requested.

## Source and scope

- Upstream main resolved and fetched on 2026-11-11: `e5b027cc6986337025c3afc27c54a0864a37e3ac`.
- Task branch: `codex/ascend-a5-daily-20261111` in the user's `myfork` remote, `https://github.com/xssty123/VeOmni`. The fork is public; no claim of private visibility is made.
- Isolated worktree: `outputs/worktrees/ascend-a5-daily-20261111`. Original checkout and its untracked work are unchanged.
- Upstream reference Dockerfile: `docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3` at the above SHA.
- Upstream references: `.github/workflows/docker-build-ascend-a3.yml`, `docker/README.md`, `docs/hardware_support/AscendDockerUsage/`.
- Reuse upstream standalone Dockerfile layout, ARM pip installation, parquet workaround, base-Triton-before-overlay order and repository-root context.
- Historical private-fork reference: `298394aee6806472065142d86539308ed36b1101`, A5 CANN 9.2.1 candidate Dockerfile, workflow and smoke. This is implementation evidence, not proof of a successful image.
- New file: `docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5_daily20261111`.
- Candidate output: unique tags in `ghcr.io/xssty123/veomni-ascend-candidate`, `linux/arm64` on `ubuntu-22.04-arm`.
- Candidate workflow `.github/workflows/docker-build-ascend-a5-candidate.yml` is registered on the fork. No default-branch registration change is required.
- No local/remote image build, source compilation, GHCR push or workflow dispatch has run yet.

## Confirmed target versions

| Component | Proposed version | Route and evidence |
|---|---|---|
| CANN | 9.2.1, 950, Ubuntu 22.04, Python 3.12 | `quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12`; publisher registry index digest `sha256:1fa0e2b6754034d9683c5718dfc12a666fc340dafb15966bdf4641ed1045c1b1` |
| torch_npu | 2.10.0.post7 | Official Ascend wheel, `--no-deps` override after `npu_aarch64` |
| triton-ascend | 3.6.0.post2 | ARM64 Python 3.12 wheel after base Triton 3.6.0; actual cp312/ARM64 wheel metadata requires base Triton 3.6.0; SHA256 pinned |
| transformers | 5.16.1 | Explicit pip pin matching upstream VeOmni's default group |
| fla_npu | `flash-linear-attention-npu-a5==0.dev20261111+main.4fc94af` | Prebuilt `py3-none-manylinux_2_34_aarch64` main daily; exact URL/SHA256 pinned; source prefix resolved, no full-commit binary attestation |
| TorchCodec | 0.10.0 | no Linux ARM64 wheel on PyPI/official CPU index; v0.10.0 source commit 0b261b98080925f2b709712a5491a1e8dd817065 confirmed |

Coupled target: torch 2.10.0, Python 3.12, Ubuntu 22.04/glibc 2.35, ARM64, Ascend A5/950.

The user explicitly confirmed this version set, the main daily wheel, its experimental CANN 9.2.1 compatibility pending hardware tests, and the ARM64 TorchCodec source route on 2026-11-11, then instructed construction to start. Hardware allocation is deliberately deferred until the candidate is built, as the user requested.

## Allowed changes

- Add one versioned A5 daily-wheel Dockerfile; do not change historical A2/A3/A5 Dockerfiles.
- Add/adapt the A5 candidate workflow for the new file, immutable wheel inputs, isolated cache, unique tags, provenance and machine-readable result.
- Normal wheel build has no source-only FLA diagnostic modes. Historical diagnostic implementation remains on its original task branch and is not repointed at the new file.
- Add a profile-driven A5 smoke check for the selected versions, retained wheel hash, packaged APIs/native launcher and imports. Record `pip check` conflicts without making them a separate gate.
- Add A5 build/usage guide and navigation; update Docker README with the new candidate route and status.
- Prepare image UT/ST counterparts from the recorded upstream workflows without changing suite selection. Actual run preparation will re-pin current upstream main at hardware-validation time.
- Preserve `pyproject.toml`, `uv.lock`, production model/runtime code, all test manifests and existing uv workflows. First hardware uv setup follows upstream unchanged.

## Pending decisions

- At hardware stage only: designated SSH host, user-owned task root, NPU allocation, driver/container mounts and test assets.

## Local verification

- Build contract and CPU-only contract regression tests run before dispatch.
- `make quality` passes with the repository's locked Ruff 0.13.2. System Ruff 0.15.20 differs on two unrelated upstream files; no unrelated reformat was applied.
- Python syntax, workflow YAML and shell syntax are checked locally.
- Hardware tests and image runtime checks require their actual candidate/host; local static checks do not substitute for those results.
