#!/usr/bin/env bash
# Execute in an already configured candidate container; never resync during tests.
set -eo pipefail
case "${1:-}" in a3|a5) target=$1 ;; *) echo 'Usage: bash scripts/ci/npu/run_candidate.sh a3|a5 unit|e2e [runner options]' >&2; exit 2 ;; esac
case "${2:-}" in unit|e2e) suite=$2 ;; *) echo 'Choose unit or e2e.' >&2; exit 2 ;; esac
shift 2
for option in "$@"; do
    case "$option" in --collect-only|--serial|--keep-going) ;; *) echo "Unsupported runner option: $option" >&2; exit 2 ;; esac
done
cd "$(dirname "$0")/../../.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VEOMNI_NPU_TARGET=$target
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
if [[ $target == a5 ]]; then export HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable; fi
output=${VEOMNI_NPU_OUTPUT_DIR:-"$PWD/outputs/npu-candidate/${target}-${suite}-$(date -u +%Y%m%dT%H%M%SZ)-$$"}
mkdir -p "$output"
python_bin="$PWD/.venv/bin/python"
"$python_bin" - "$target" <<'PY' | tee "$output/preflight.log"
import sys
from importlib.metadata import version
import torch
import torch_npu
assert torch.npu.is_available(), 'NPU unavailable; cannot qualify this image'
assert torch.npu.device_count() >= 8, 'Full UT/ST qualification requires eight visible NPUs'
assert sys.version_info[:2] == (3, 12)
for name, expected in {'torch': '2.10.0', 'torch-npu': '2.10.0.post7', 'triton': '3.6.0', 'triton-ascend': '3.6.0', 'transformers': '5.16.1'}.items():
    assert version(name).split('+')[0] == expected, (name, version(name), expected)
print('NPU:', torch.npu.get_device_name(0), 'count:', torch.npu.device_count(), 'target:', sys.argv[1])
PY
# Record runtime, source and retained-wheel evidence with every invocation.
git rev-parse HEAD > "$output/veomni-commit.txt"
git diff --binary HEAD > "$output/working-tree.patch"
git ls-files --others --exclude-standard > "$output/untracked-files.txt"
npu-smi info > "$output/npu-smi.txt"
uv pip freeze --python "$python_bin" > "$output/requirements.txt"
sha256sum /app/torch_npu-*.whl /app/triton-*.whl /app/triton_ascend-*.whl /app/flash_linear_attention_npu_"$target"-*.whl > "$output/wheels.sha256"
# A missing/incompatible FLA is a failure, rather than a passing suite of skips.
"$python_bin" -m pytest -q tests/ops/test_fla_npu_installation_contract.py --junitxml="$output/fla-contract.xml"
VEOMNI_TEST_GDN_IMPLEMENTATION=triton "$python_bin" -m scripts.ci.accelerator --backend npu --suite "$suite" --include-conditional --output-dir "$output/default" "$@"
if [[ $suite == e2e ]]; then
    VEOMNI_TEST_GDN_IMPLEMENTATION=fla_npu "$python_bin" -m scripts.ci.accelerator \
        --backend npu --suite e2e --include-conditional --output-dir "$output/fla-npu" \
        --only npu-e2e-40,npu-e2e-41 --pytest-arg=-k --pytest-arg=qwen3_5 "$@"
fi
printf 'Completed %s %s; evidence: %s\n' "$target" "$suite" "$output"
