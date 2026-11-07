#!/usr/bin/env bash
# Run inside the selected A3/A5 image from the VeOmni checkout.
set -eo pipefail
case "${1:-}" in a3|a5) target=$1 ;; *) echo 'Usage: bash scripts/ci/npu/setup_candidate.sh a3|a5' >&2; exit 2 ;; esac
cd "$(dirname "$0")/../../.."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
[[ $(uname -m) == aarch64 ]] || { echo 'These candidates require aarch64.' >&2; exit 1; }
image_python=$(command -v python3)
"$image_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
shopt -s nullglob
npu_wheels=(/app/torch_npu-2.10.0.post7-*.whl)
triton_wheels=(/app/triton-3.6.0-*.whl)
ascend_wheels=(/app/triton_ascend-3.6.0-*.whl)
fla_wheels=(/app/flash_linear_attention_npu_"$target"-26.10.0*.whl)
for name in npu_wheels triton_wheels ascend_wheels fla_wheels; do
    declare -n wheels=$name
    [[ ${#wheels[@]} == 1 ]] || { echo "Expected one retained wheel in $name, got ${#wheels[@]}" >&2; exit 1; }
done
[[ -f /app/torchcodec/pyproject.toml ]] || { echo 'Missing retained TorchCodec source.' >&2; exit 1; }
if [[ -f /app/${target}-ci-wheels.sha256 ]]; then
    sha256sum -c "/app/${target}-ci-wheels.sha256"
fi
if [[ -r /tmp/latest_proxy ]]; then
    proxy=$(cat /tmp/latest_proxy)
    export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" http_proxy="$proxy" https_proxy="$proxy"
fi
sync_tls=(--system-certs)
if [[ $target == a5 ]]; then
    # Retain the existing A5 runner's host-scoped CA workaround for uv sync.
    sync_tls+=(--allow-insecure-host github.com --allow-insecure-host objects.githubusercontent.com --allow-insecure-host download-r2.pytorch.org)
fi
uv sync "${sync_tls[@]}" --frozen --python "$image_python" --extra npu_aarch64 --group dev
python_bin="$PWD/.venv/bin/python"
# Both distributions own libtriton.so; the Ascend overlay must be installed last.
uv pip install --python "$python_bin" --no-deps --reinstall "${npu_wheels[0]}" "${triton_wheels[0]}"
uv pip install --python "$python_bin" --no-deps --reinstall "${ascend_wheels[0]}"
uv pip install --python "$python_bin" --no-deps --reinstall "${fla_wheels[0]}"
uv pip install --system-certs --index-url https://pypi.org/simple --python "$python_bin" pybind11==3.1.0 ninja cmake setuptools wheel pyelftools
# Reuse the image's pinned Ascend TorchCodec source, without changing its revision.
export PATH="$PWD/.venv/bin:$PATH"
export CMAKE_PREFIX_PATH="$("$python_bin" -m pybind11 --cmakedir):${CMAKE_PREFIX_PATH:-}"
export LIBRARY_PATH="/usr/local/lib:/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 uv pip install --python "$python_bin" --no-deps --no-build-isolation --editable /app/torchcodec
"$python_bin" -c "from importlib.metadata import version; from triton._C.libtriton import ascend; assert version('torch').split('+')[0] == '2.10.0'; assert version('torch-npu') == '2.10.0.post7'; assert version('triton') == version('triton-ascend') == '3.6.0'; assert version('transformers') == '5.16.1'; from packaging.version import Version; assert Version(version('torchcodec')).release == (0, 10, 0); from torchcodec.decoders import VideoDecoder"
