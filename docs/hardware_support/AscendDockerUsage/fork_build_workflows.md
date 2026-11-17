# Fork image build workflows

`codex/ascend-docker-build` collects the existing CANN 9.2.1 recipes, GHCR candidate builds and product-based Quay naming. The fork's default `main` remains the Actions registration branch. The source tips, pending-file hashes and allowed changes are fixed in `docker/ascend/records/branch-consolidation-20261117.json` and its accompanying contract.

## GHCR candidates

`.github/workflows/docker-build-ascend-candidate.yml` supports `a2-amd64`, `a2-arm64`, `a3-arm64` and `a5-arm64`. It builds the selected recipe, pushes to `ghcr.io/xssty123/veomni-ascend-candidate`, pulls the exact digest in a fresh job and checks installed packages and FLA/TorchCodec imports. Candidate tags retain the unique target, source SHA, run ID and attempt suffix. The result artifact records build and packaging-smoke results separately from NPU UT/ST.

The smoke profiles follow the retained recipes: A2 x86 uses Triton 3.2.0 / Ascend 3.2.2; A2 ARM and A3 use 3.5.0 / 3.2.2; A5 uses 3.6.0 / 3.6.0. No recipe installation command or version is changed by the consolidation.

## Quay build and naming

The existing `docker-build-ascend-a2.yml`, `docker-build-ascend-a3.yml` and `docker-build-ascend-a5.yml` retain their build and architecture handling, but use the fork owner, manual triggers and the `quay-production` environment. They directly build and publish to Quay, so run them only for an explicitly requested publication after the applicable NPU validation. They are separate from candidate-first delivery.

`image_version` is explicit. `docker/ascend/image_tag.py` preserves the format:

```text
{version}-cann{CANN}-torch_npu{PTA}-{product}-{OS}-{Python}-veomni
v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni
```

A2 uses product `910b` and combines AMD64/ARM64 under one tag. A3 uses `a3`; the A5 CANN base's `950` identifier becomes `a5`. Existing release tags are mutable with the historical direct-build route; choose a new version prefix when retaining an older image is required. No workflow was dispatched by the consolidation.

## Historical publishing sources

The original digest-promotion and two fixed-image copy YAML files are retained byte for byte in `docker/ascend/legacy/`. They are historical references, not registered workflows: their old dependency expectations, artifact schemas and `main` guards must be re-evaluated before reuse. Their original commits also remain reachable through the preservation merge.

GitHub manual dispatch requires the workflow filename to be registered on the default branch. Existing A2/A3/A5 formal names and the generic candidate name are already registered there; select this branch as the run ref and supply inputs supported by this branch's YAML. Default-branch files were not changed, and the newly added image-test entrypoints on the test branch may require separate registration.

The preserved six-file worktree remains intact. Its copied prebuilt-wheel recipes have not been built or validated on NPU by this task. Future image changes start from a newly verified upstream base and a new task branch; this branch preserves the existing work as a reusable reference.
