# A5 daily-wheel candidate build and validation

This is an experimental ARM64 A5/950 candidate, not a published Quay release or
a hardware-qualified image. The candidate pipeline builds a unique GHCR tag,
runs packaging/import smoke against its immutable digest, then records separate
hardware acceptance results. Quay publication and an upstream delivery PR are
outside this workflow.

## Selected environment

| Component | Pinned input | Installation |
|---|---|---|
| CANN | 9.2.1, 950, Ubuntu 22.04, Python 3.12 | Official `quay.io/ascend/cann` base, pinned by manifest digest |
| torch / torch-npu | 2.10.0 / 2.10.0.post7 | ARM extra, then selected torch-npu wheel with `--no-deps` |
| Triton / Triton-Ascend | 3.6.0 / 3.6.0.post2 | Base first, checksum-verified Ascend overlay second |
| Transformers | 5.16.1 | Explicit pip pin matching upstream VeOmni |
| FLA NPU | `flash-linear-attention-npu-a5==0.dev20261111+main.4fc94af` | Checksum-verified prebuilt main daily wheel, no source compilation |
| TorchCodec | v0.10.0, commit `0b261b98080925f2b709712a5491a1e8dd817065` | ARM64 source build using the existing Ascend installer |

The TorchCodec source tag reports `0.10.0a0+<git suffix>` after installation.
No Linux ARM64 0.10.0 wheel was found in PyPI or the official PyTorch CPU index
on 2026-11-11. The image also includes pytest 7.4.4, expecttest 0.3.0 and rich
14.2.0 from the upstream lockfile for the image-environment test lane.

Exact URLs, hashes, base digest and expected installed versions live in
[`a5-daily-20261111.json`](../../../docker/ascend/profiles/a5-daily-20261111.json).
The main daily stream is intentionally selected, not the separate `26.10.0.*`
release-branch daily stream. See the publisher's
[Daily and Release guide](https://github.com/flashserve/flash-linear-attention-npu/blob/main/docs/Daily%E5%92%8CRelease%E5%8C%85%E5%8F%91%E5%B8%83%E4%B8%8E%E5%AE%89%E8%A3%85%E8%AF%B4%E6%98%8E.md).

### Evidence and limits

- The FLA SHA256 was measured from the actual official mirror download, not
  obtained from a signed publisher checksum. The wheel's `4fc94af` source prefix
  resolves to commit `4fc94af96fc181274cdfc5c1760a2bb0dc9d76f2`, but the binary
  carries no full-commit build attestation.
- FLA uses `py3-none-manylinux_2_34_aarch64`; Ubuntu 22.04 supplies glibc 2.35.
  Its embedded OPP compiler version is 9.1.0. The publisher's pinned validation
  matrix lists CANN 9.1.0, not this exact CANN 9.2.1 combination. Successful
  installation alone cannot establish A5 numerical or training compatibility.
- Triton-Ascend metadata pins several older dependencies and pytest 8.x, while
  VeOmni's test environment uses pytest 7.x. `--no-deps` preserves the selected
  VeOmni runtime. Full `pip check` output is recorded separately and does not
  alone fail packaging smoke. Missing imports, native library errors and actual
  test failures remain failures.
- The normal FLA wheel path has no source-compilation diagnostic mode. Historical
  source diagnostics stay on their original branch and are not reused here.

## Build a candidate

Confirm the six-component version set and unresolved installation choices before
starting a build. The build context is the repository root. The standalone file is:

`docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5_daily20261111`

Run the static input check first:

```bash
python3 docker/ascend/tests/check_daily_build_contract.py
```

The fork's registered manual workflow builds on native ARM64 and pushes only a
unique GHCR candidate tag. After committing and verifying the task branch SHA:

```bash
gh workflow run docker-build-ascend-a5-candidate.yml \
  --repo xssty123/VeOmni \
  --ref codex/ascend-a5-daily-20261111 \
  -f mode=full
```

The image repository is `ghcr.io/xssty123/veomni-ascend-candidate`. Tags include
the selected daily date, source commit, run ID and attempt. Cache scope is
separate from historical A5 source builds. GHCR access/visibility must be checked
independently from the fork's public visibility.

An explicitly requested local build can use:

```bash
docker build --platform linux/arm64 \
  -f docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5_daily20261111 \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  -t veomni:a5-daily20261111 .
```

This still compiles TorchCodec, not FLA. An image tag or successful command start
is not a completed build record.

## Packaging smoke and records

The workflow pulls the just-built immutable digest into a fresh container and
checks architecture, Python, CANN, source labels, final installed versions,
retained artifact checksums, all 38 publisher Stable operator registrations,
OPP payload layout and torch-npu/Triton-Ascend/TorchCodec library imports.
The standalone publisher API checker is fetched at a fixed commit and verified
against a previously measured SHA256. No test step installs packages or runs
`uv sync` to repair the candidate.

Artifacts:

- `a5-build-diagnostics`: resource and available kernel logs.
- `a5-candidate-smoke`: observed image metadata, smoke JSON/log, package snapshot
  and retained artifact checksums.
- `a5-candidate-result`: source/run/tag/digest identity, build status, packaging
  status, dependency consistency, and four initially `not_run` hardware results.

The candidate may already exist in GHCR if a later smoke job fails. Always read
build, smoke and workflow outcomes separately; never promote a failed or
incomplete candidate.

## A5 hardware acceptance

After the image is built, obtain the run-specific SSH host, user-owned task root,
NPU allocation, driver mounts and test assets. Do not infer them from an earlier
run. All remote checkouts, generated YAML, scripts, caches and logs stay under
that user-owned task root in an isolated run directory.

Resolve latest upstream main once again for the test source and pin its full SHA.
It can differ from the image build SHA. Both lanes import this same checkout:

| Lane | UT | ST | Environment |
|---|---|---|---|
| uv | `npu_unit_tests.yml` | `npu_e2e_test.yml` | Unchanged upstream setup first, including its extra and package versions |
| image | `npu_unit_tests_image.yml` | `npu_e2e_test_image.yml` | Image Python and installed dependencies, no environment reconstruction |

Prepared image YAML under `docker/ascend/ci/a5-daily-20261111/` is derived from
upstream `e5b027cc6986337025c3afc27c54a0864a37e3ac`. It preserves test selection,
parallel parameters and assertions; only environment preparation, tool entry
points and display names differ. Re-derive from the pinned test SHA if main has
advanced before hardware validation. The original uv workflow files remain
unchanged in `.github/workflows/`.

These YAML files are plans, not executable SSH configuration. Translate their
ordered `run`/`uses` steps and host-only mounts/device configuration explicitly
for the allocated machine. Never blindly use the upstream A2 runner's paths,
16-device mount list or default card selection on A5. Use separate clean
containers for uv and image, both from the candidate digest. Preserve the first
unchanged uv attempt and diagnose actual errors before changing dependencies or
extras. Record all four results, collected/skipped tests, failures and logs.
Only four completed passing results establish full acceptance for this run.
