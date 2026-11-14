# Build the Ascend A5 CANN 9.2.1 image

This recipe restores the dependency stack of the recorded A5 image
`veomni-a5:cann9.2.1-local-20261105-j16`, using the current repository checkout
as its VeOmni source. It targets **Linux ARM64 / Ascend 950 (A5)**.
It is not interchangeable with the A2 or A3 image.

The [Dockerfile](../../../docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5)
and [Quay build workflow](../../../.github/workflows/docker-build-ascend-a5.yml)
preserve the historical package versions. This is not a security dependency
upgrade. Building this branch produces a new image, not a reproduction of
the historical image digest: the VeOmni checkout is newer and unpinned apt/pip
dependencies can resolve differently.

## Versions and installation order

| Component | A5 selection |
| --- | --- |
| Base | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.1-950-ubuntu22.04-py3.12` |
| OS / architecture / Python | Ubuntu 22.04 / `linux/arm64` / Python 3.12 |
| torch / torch-npu | `2.10.0` / `2.10.0.post7` |
| Triton / Triton-Ascend | `3.6.0` / `3.6.0` |
| Transformers / uv | `5.16.1` / `0.11.16` |
| PyArrow / pandas | `21.0.0` / `2.3.2` |
| FLA source | `a7c608f2e3f828779ceeec1fcf155a6e37ede83c` |
| FLA target / distribution | `ascend950` / `flash-linear-attention-npu-a5` |
| TorchCodec source | `0b261b98080925f2b709712a5491a1e8dd817065` (`v0.10.0`) |

The recipe installs VeOmni's `npu_aarch64` extra into system Python, applies
the PyArrow/pandas pins, replaces torch-npu, installs base Triton before the
Ascend overlay, builds the A5 FLA wheel, then builds TorchCodec. The
TorchCodec source reports `0.10.0a0` (optionally with a commit suffix).
The FLA wheel and selected torch-npu/Triton wheels remain in `/app`, alongside
`a5-ci-wheels.sha256` and `a5-image-requirements.txt`.

A2 ARM and A3 use Triton `3.5.0` / Triton-Ascend `3.2.2`; A2 x86 uses
`3.2.0` / `3.2.2`. Do not substitute these pairs into the A5 recipe without
separately validating the changed environment. See the
[A2](build_a2_docker.md) and [A3](build_a3_docker.md) guides for the corresponding Dockerfiles.

Unlike the A2/A3 recipes from PR #1427, this A5 baseline does not prepopulate
`/opt/veomni/uv-cache` or install the complete locked dev group into system
Python. A new test venv needs its own dependencies. `uv sync` creates a
different environment from the image's preinstalled stack.

## Build locally

Run from the repository root on a native Linux ARM64 host with Docker and
enough disk/memory for CANN/FLA compilation. A cross-architecture emulation
build is not the validated historical path.

```bash
docker build --platform linux/arm64 \
  -f docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5 \
  --build-arg FLA_BUILD_JOBS=4 \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" \
  --label ai.veomni.docker-revision="$(git rev-parse HEAD)" \
  --label ai.veomni.fla-revision=a7c608f2e3f828779ceeec1fcf155a6e37ede83c \
  --label ai.veomni.target=a5 \
  -t veomni:cann921-a5-local .
```

`FLA_BUILD_JOBS` defaults to 4; the successful historical server run used
16 on a larger machine. Compiler memory requirements must be considered
before increasing it. `PIP_INDEX` and `APT_SOURCE` can select package mirrors;
`BUILD_PREFER_IPV4=1` is an opt-in workaround for hosts with unusable IPv6.
For an older Docker daemon, the preserved recipe can also be built natively
with `DOCKER_BUILDKIT=0`, omitting `--platform` and selecting `--network=host`
only when required by that host's package access.

The historical server bundle supplied pre-fetched FLA/TorchCodec sources.
This repository-root recipe clones the same pinned revisions instead; it
requires network access for Git, package indexes and FLA's build dependencies.

## Upstream Quay build

The [A5 builder](../../../.github/workflows/docker-build-ascend-a5.yml) follows
the upstream A3 publication flow. It requires the `ByteDance-Seed` repository
owner and runs on `ubuntu-22.04-arm`, using `QUAY_USERNAME` and
`QUAY_PASSWORD` Actions secrets to publish to `quay.io/ascend/veomni`.
It supports manual dispatch, a published release, and pushes to `main` that
change its Dockerfile or workflow. Fork branch pushes do not
build or publish this image.

The builder uses GitHub Actions caching like A3, with scope
`ascend-a5-cann921` to keep A5's cache separate. Final images are published
to Quay using the secrets above.

| Trigger | Output tag |
| --- | --- |
| Push/dispatch on `main` | `veomni-9.2.1-a5-ubuntu22.04-py3.12-latest` |
| Published release | `veomni-9.2.1-a5-ubuntu22.04-py3.12-<release-tag>[.postN]` |
| Manual dispatch on another upstream branch | `veomni-9.2.1-a5-ubuntu22.04-py3.12-<normalized-branch>` |

The release collision handling and main/branch tag layout match the existing
A2/A3 workflows. The CANN base uses `950`; the published image tag uses `a5`.
These automated builder tags coexist with the previously published
product-based tags below. The workflow does not overwrite the historical
`v0.1.12-cann...` product tag.

After the workflow has landed upstream, an authorized maintainer can dispatch:

```bash
gh workflow run docker-build-ascend-a5.yml \
  --repo ByteDance-Seed/VeOmni --ref main
```

Before treating a new build as hardware-validated, run A5 UT/ST against its
exact digest, recording the test commit and environment. Keep the upstream
uv UT/ST and image-environment UT/ST results separate; do not run `uv sync`
in the latter and describe it as testing the unchanged image.
Launching an A5 hardware container also requires the host's actual driver,
device and URMA mounts; the A2 device example must not be assumed sufficient.

## Historical published image and provenance

The previously built A5 artifact remains available under the product tag:

```bash
docker pull quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni
```

Its manifest digest is
`sha256:369b251a90d189051d0f95038345440869f105d6753395958223a11ed89aee95`;
its image/config ID is
`sha256:112a21cff75515133b0229ebfc21b411f017e06a0342cd92294e04352943a203`.
It was built on 2026-11-05 from VeOmni source
`273d31cbf9eebd05913c38ed15388447a0f094f2`, with server-build adaptations at
`87f92b2df8d2fb59398f96deb7dd73a7eadb13a1`. The archived Dockerfile SHA256 is
`c5e1e26ee38719c7407b184d90d1fa18a46603cf0a6bcbae30aaaa449a90de9a`.

The recorded build passed packaging/import/API checks but had unresolved
`pip check` conflicts and did not execute NPU UT/ST. Its Quay vulnerability
scan also reports known issues. Neither its published tag nor a later test
using an overridden venv establishes acceptance of this new branch's image.
