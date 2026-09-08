### Adding a New Workflow

When adding a new workflow for continuous integration (CI), you have two runner options: a fixed runner or a machine from the vemlp.

- **Fixed Runner**: To use a self-hosted fixed runner, specify it via the `runs-on` keyword. Common label sets:
  - GPU, any free L20-8 host: `runs-on: [self-hosted, l20-8]`
  - GPU, a specific host (e.g. for repro): `runs-on: [self-hosted, l20-3]`
  - NPU, any free 910B-8 slice: `runs-on: [self-hosted, 910b-8]`
  - NPU, a specific 8-card slice: `runs-on: [self-hosted, 910b-2]`
  - NPU, a specific physical machine (either of its two splits): `runs-on: [self-hosted, 910b-host-1]`

  See `github_runner/README.md` for the full label scheme.
- **Vemlp Runner**: Opting for a Vemlp machine allows you to launch tasks elastically.

Here is a template to assist you. This template is designed for using Vemlp machines. Currently, for each workflow, you need to create a `setup` and a `cleanup` job. When using this template, the main parts you need to modify are the `IMAGE` environment variable and the specific `job steps`.

```yaml
name: Your Default Workflow

on:
  push:
    branches:
      - main
      - v0.*
  pull_request:
    branches:
      - main
      - v0.*
    paths:
      - "**/*.py"
      - ".github/workflows/template.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

permissions:
  contents: read

env:
  IMAGE: "your vemlp image" # e.g. "verl-ci-cn-beijing.cr.volces.com/verlai/verl:sgl055.dev2"
  DYNAMIC_RUNNER_URL: "https://sd4hav466omp034ocfn0g.apigateway-cn-beijing.volceapi.com/veomni/runner" # public veFaas api

jobs:
  setup:
    if: github.repository_owner == 'ByteDance-Seed'
    runs-on: ubuntu-latest
    outputs:
      runner-label: ${{ steps.create-runner.outputs.runner-label }}
      task-id: ${{ steps.create-runner.outputs.task-id }}
    steps:
      - uses: actions/checkout@v4
      - id: create-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "create"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          image: "${{ env.DEFAULT_IMAGE }}"

  your_job:
    needs: setup
    runs-on: ["${{ needs.setup.outputs.runner-label || 'default-runner' }}"]
    steps:
      xxxx # your jobs

  cleanup:
    runs-on: ubuntu-latest
    needs: [setup, your_job]
    if: always()
    steps:
      - id: destroy-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "destroy"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          task-id: "${{ needs.setup.outputs.task-id }}"
```

### Model and Dataset
To avoid CI relies on network, we pre-download dataset on a NFS on the CI machine. The path for models are \${HOME}/models and the path for dataset is \${HOME}/models/hf_data.

### Private-fork image UT/ST with upstream test parity

`npu_image_unit_tests.yml` and `npu_image_e2e_tests.yml` run in an **already
started Ascend image container**, using act's host executor. They do not pull an
image, create a nested container, or install/synchronize Python dependencies.
The container must already provide CANN, eight visible NPUs, Python, pytest,
torch, torch-npu, torchrun, and the dependencies/assets required by the tests.
Do not run UT and ST concurrently on the same NPU slice.

The test-command baseline is upstream commit
`1f6674f0a2556702f4db6ac5a51e0fb02c20d84a`:

| Image workflow | Upstream source | Pytest invocations |
| --- | --- | --- |
| `npu_image_unit_tests.yml` | `npu_unit_tests.yml` | 29 |
| `npu_image_e2e_tests.yml` | `npu_e2e_test.yml` | 3 |

Only the launcher changes from `uv run --frozen pytest` to
`"$IMAGE_PYTHON" -m pytest`; test arguments and ordering are identical. These
counts are commands, not individual parametrized test cases. The final Wan
invocation is retained exactly as in upstream, even though the first ST command
also collects that test. Test files, model code, skip conditions, numerical
tolerances, and dependency versions are unchanged. No optional GDN enablement
or extra test coverage is carried over from other private branches. Actual
skips still depend on the packages/hardware available in the image.

Run from the repository inside the existing image container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -o pipefail

act workflow_dispatch -W .github/workflows/npu_image_unit_tests.yml \
  -j npu_image_unit_tests -P 910b-8=-self-hosted \
  --input python=/usr/bin/python 2>&1 | tee npu_image_ut.log

act workflow_dispatch -W .github/workflows/npu_image_e2e_tests.yml \
  -j npu_image_system_tests -P 910b-8=-self-hosted \
  --input python=/usr/bin/python 2>&1 | tee npu_image_st.log
```

Change the `python` input to the interpreter actually installed in the image
(for example `/app/.venv/bin/python`); its default is `python` resolved on PATH.
The interpreter directory is prepended to PATH, while torchrun may also be
resolved from another existing PATH entry. Existing device visibility is
preserved; only when both Ascend visibility variables are absent does the
workflow derive the upstream eight-card slice from the runner name.
Pass `--env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` to act if needed.

Asset defaults remain `/mnt/veomni_ci/models`,
`/mnt/veomni_ci/datasets/custom_data/samples`, and `/mnt/veomni_ci/datasets`.
For other existing mounts, pass act `--env CI_HF_MODELS_DIR=...`,
`--env CI_SAMPLES_DIR=...`, and `--env CI_DATASET_DIR=...`. See the
[act runner documentation](https://nektosact.com/usage/runners.html) for host
executor mapping and the [usage guide](https://nektosact.com/usage/index.html)
for workflow/job selection and inputs.

Unlike the upstream container workflow, these local-host workflows do not kill
all visible pytest/torchrun processes. Inspect remaining workers after a failed
run before starting another test; do not terminate unrelated jobs. Checkout uses
`clean: false` to preserve local assets. Logs include the checked-out commit,
interpreter, package versions, NPU state, and pip inventory; record the tested
container's image digest separately because these workflows do not select it.

When refreshing this branch from upstream, recheck both ordered pytest command
lists against their source workflows, in addition to syncing the test code.
