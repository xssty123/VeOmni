# Ascend Docker Supported Tags

A full list of tags published at
[quay.io/ascend/veomni](https://quay.io/repository/ascend/veomni?tab=tags).

## Tag Naming Schemes

Two tag schemes coexist on quay.io:

- **Product-based** (current): `<veomni_version>-cann<CANN>-torch_npu<torch_npu>-<chip_series>-<os>-py<python>-veomni`
- **Legacy**: `veomni-<cann>-<chip_series>-<os>-py<python>[-torch<torch>][-<suffix>]`

> Product-based tags use dedicated CANN / torch-npu Dockerfiles that include the
> triton-ascend and fla_npu GDN stack. General-purpose Dockerfiles remain
> available for the legacy tags.

## CANN 9.2.1

These are previously published artifacts, not outputs of rebuilding this
branch. A tag's availability does not establish NPU acceptance or a clean
vulnerability scan. Use the recorded digest to identify an exact artifact.

| Product tag | Platform | Manifest digest |
| --- | --- | --- |
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni` | AMD64 + ARM64 | `sha256:59701c8721756b2fe9a2773bc6dbf238b60a2f60353e0d94235968b0084cd7ff` |
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-veomni` | ARM64 | `sha256:89fa9e4515070c0fb54c210048cbcd7a39d8076a0a59d36b38862f4be0309976` |
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni` | ARM64 | `sha256:369b251a90d189051d0f95038345440869f105d6753395958223a11ed89aee95` |

Prefix tags with `quay.io/ascend/veomni:` when pulling. A2/A3 were built from
PR #1427 at `29eeb215bf3a2e6ed73a5345c00d046d0418a10e`; the A5 artifact uses
older VeOmni source and the Triton 3.6.0 stack. See the
[A2](build_a2_docker.md), [A3](build_a3_docker.md)
and [A5 recipe and provenance](build_a5_docker.md) guides.

## CANN 9.1.0

### Product-based tags

| Tag | Dockerfile | Content |
|---|---|---|
| `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12-veomni` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |

## CANN 9.0.0

### Product-based tags

| Tag | Dockerfile | Content |
|---|---|---|
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.11-veomni` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.11-veomni` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A2-ubuntu22.04-py3.11-veomni-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A3-ubuntu22.04-py3.11-veomni-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |

### Legacy tags

| Tag | Dockerfile | Content |
|---|---|---|
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-torch2.10.0-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.10.0-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-torch2.9.0-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.9.0-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.7.1-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |

## CANN 8.3.rc2

### Legacy tags

| Tag | Dockerfile | Content |
|---|---|---|
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-a3-ubuntu22.04-py3.11-latest` | [Dockerfile] [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a3) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-torch2.7.1-latest` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a5` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a4` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a3` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a2` | [Dockerfile] [a2.x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [a2.arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
