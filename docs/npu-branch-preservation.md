# Preserved NPU test branches

`codex/ascend-cann921` retains the existing pending NPU tests, adaptations and suite manifest. No existing test, selection, uv workflow or candidate setup command was removed or changed during the branch consolidation.

## Four test lanes

| Environment | Unit tests | System tests |
| --- | --- | --- |
| Upstream-style uv | `.github/workflows/npu_unit_tests.yml` | `.github/workflows/npu_e2e_test.yml` |
| Installed image | `.github/workflows/npu_unit_tests_image.yml` | `.github/workflows/npu_e2e_test_image.yml` |

The image counterparts preserve the runner/container mounts and the common `scripts.ci.accelerator` suite selections. They use the image interpreter and do not run `uv sync` or install packages. Missing runtime dependencies remain failures of that image environment. These manual workflows require default-branch registration before GitHub dispatch; this consolidation leaves the fork default branch unchanged. They can also be used through the existing local `act` execution route.

For the historical A3/A5 candidate environments, the four existing root YAML files and `run_candidate.sh` are unchanged. Their new `_image.yml` counterparts call `run_image.sh` without creating a project environment, and retain the same full-suite invocation plus the additional FLA e2e pass. Set `IMAGE_PYTHON` when the image interpreter is not the first `python3` on PATH. The uv and image lanes must use the same checkout and suite manifest.

## Historical coverage

The four earlier CI branch tips are parents of the preservation merge, so their original commits and test implementations remain reachable from this branch. `scripts/ci/npu/preservation.json` fixes their exact source SHAs. The original image UT/ST YAML files are retained byte for byte under `scripts/ci/npu/legacy/`; their old test paths and commented container configuration are historical references, not the maintained image entrypoints.

The separate worktree with an unfinished upstream merge remains untouched. Its index, working files, merge metadata and patches were saved in the local reorganization backup. Hardware UT/ST were not run by the consolidation, and neither old workflow success nor local structural checks establish NPU acceptance.
