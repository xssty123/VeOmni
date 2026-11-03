#!/usr/bin/env bash
# Host entry point; --inside runs only in the prepared ARM64 dependency image.
set -eo pipefail
export DIAG_MODE=${DIAG_MODE:-diagnose-kda}
case "$DIAG_MODE" in
    diagnose-kda|diagnose-fla) ;;
    *) echo "Unsupported diagnostic mode: $DIAG_MODE" >&2; exit 2 ;;
esac

if [[ "${1:-}" == --inside ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export FLA_NPU_SOC=ascend950 TORCH_DEVICE_BACKEND_AUTOLOAD=0
    git clone https://github.com/flashserve/flash-linear-attention-npu /tmp/fla-npu
    cd /tmp/fla-npu
    git checkout --detach "$FLA_COMMIT"
    test "$(git rev-parse HEAD)" = "$FLA_COMMIT"
    python -m pip install -r requirements.txt 'setuptools>=70.1' wheel
    python -m pip freeze > /reports/requirements.txt
    python scripts/check_npu_env.py --build-only
    if [[ "$DIAG_MODE" == diagnose-kda ]]; then
        python scripts/check_build_ops.py --ops=chunk_kda_bwd_intra
    fi
    unset FLA_NPU_OPS FLA_NPU_BUILD_ARGS
    date -u +%Y-%m-%dT%H:%M:%SZ > /reports/compile-started-at.txt
    started=$SECONDS
    set +e
    if [[ "$DIAG_MODE" == diagnose-fla ]]; then
        PIP_VERBOSE=1 FLA_NPU_BUILD_ARGS="-j2" \
            python -u scripts/build_wheel.py --no-progress --wheel-dir /app
    else
        bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=chunk_kda_bwd_intra -j4
    fi
    status=$?
    set -e
    printf '%s\n' "$status" > /reports/compile-exit-code.txt
    printf '%s\n' "$((SECONDS - started))" > /reports/compile-seconds.txt
    if [[ -f /app/fla-npu-build.log ]]; then
        cp /app/fla-npu-build.log /reports/
    fi
    if [[ "$DIAG_MODE" == diagnose-fla && "$status" == 0 ]]; then
        mkdir -p /reports/wheels
        cp /app/flash_linear_attention_npu_a5-*.whl /reports/wheels/
        (cd /reports/wheels && sha256sum ./*.whl) > /reports/wheel-sha256.txt
        validate_wheel() {
            python -m zipfile -t /app/flash_linear_attention_npu_a5-*.whl &&
                python -m pip uninstall -y fla-npu flash-linear-attention-npu flash-linear-attention-npu-a5 &&
                python -m pip install --force-reinstall --no-deps /app/flash_linear_attention_npu_a5-*.whl &&
                python -c "from fla_npu.ops import ascendc; assert all(callable(getattr(ascendc, n, None)) for n in ('chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd'))" &&
                python scripts/check_packaged_wheel_api.py
        }
        set +e
        validate_wheel
        status=$?
        set -e
        printf '%s\n' "$status" > /reports/validation-exit-code.txt
        python -m pip freeze > /reports/requirements-after-validation.txt
    fi
    exit "$status"
fi

set -u
: "${FLA_COMMIT:?}" "${DIAG_IMAGE:?}" "${REPORT_DIR:?}" "${GITHUB_SHA:?}"
mkdir -p "$REPORT_DIR"
REPORT_DIR=$(cd "$REPORT_DIR" && pwd)
export REPORT_DIR
container=a5-${DIAG_MODE}
started_at=$(date +%s)
monitor_pid=

sample_resources() {
    date -u +%Y-%m-%dT%H:%M:%SZ
    free -b
    df -h / /var/lib/docker
    ps -eo pid,ppid,comm,%cpu,%mem,rss --sort=-rss | head -n 26 || true
    cat /proc/pressure/memory || true
    # Read the actual container cgroup; the host root alone can miss its OOMs.
    pid=$(docker inspect --format '{{.State.Pid}}' "$container" 2>/dev/null) || return 0
    if [[ "$pid" -gt 0 && -r "/proc/$pid/cgroup" ]]; then
        cgroup=$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")
        if [[ -n "$cgroup" ]]; then
            for name in memory.current memory.peak memory.max memory.events; do
                file="/sys/fs/cgroup${cgroup}/$name"
                if [[ -r "$file" ]]; then
                    printf '%s: ' "$name"
                    cat "$file"
                fi
            done
        fi
    fi
}

finish() {
    status=$?
    trap - EXIT
    set +e
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null
        wait "$monitor_pid" 2>/dev/null
    fi
    sample_resources >> "$REPORT_DIR/resources.log" 2>&1
    docker inspect "$container" > "$REPORT_DIR/container-inspect.json" 2>/dev/null
    sudo journalctl -k --since "@$started_at" --no-pager > "$REPORT_DIR/kernel.log" 2>&1
    # Container-created files are root-owned; the host writes the result beside them.
    DIAG_EXIT_CODE="$status" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['REPORT_DIR'])
def read_int(name):
    path = root / name
    return int(path.read_text()) if path.exists() else None

result = {
    'mode': os.environ['DIAG_MODE'],
    'veomni_commit': os.environ['GITHUB_SHA'],
    'fla_commit': os.environ['FLA_COMMIT'],
    'soc': 'ascend950',
    'operator': 'all' if os.environ['DIAG_MODE'] == 'diagnose-fla' else 'chunk_kda_bwd_intra',
    'jobs': 2 if os.environ['DIAG_MODE'] == 'diagnose-fla' else 4,
    'container_exit_code': int(os.environ['DIAG_EXIT_CODE']),
    'compile_exit_code': read_int('compile-exit-code.txt'),
    'compile_seconds': read_int('compile-seconds.txt'),
    'validation_exit_code': read_int('validation-exit-code.txt'),
    'npu_ut_st': 'not_run',
}
(root / 'diagnostic-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
    docker rm -f "$container" >/dev/null 2>&1
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker image inspect "$DIAG_IMAGE" > "$REPORT_DIR/image-inspect.json"
docker create --name "$container" --entrypoint /bin/bash \
    --mount "type=bind,src=$REPORT_DIR,dst=/reports" \
    --mount "type=bind,src=$(cd "$(dirname "$0")" && pwd)/$(basename "$0"),dst=/diagnose_a5_kda.sh,readonly" \
    -e FLA_COMMIT -e DIAG_MODE -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    "$DIAG_IMAGE" /diagnose_a5_kda.sh --inside

(
    while true; do
        sample_resources 2>&1 | tee -a "$REPORT_DIR/resources.log"
        sleep 5
    done
) &
monitor_pid=$!

# Run foreground in the same Actions step as the sampler, preserving both streams.
set +e
docker start --attach "$container" 2>&1 | tee "$REPORT_DIR/compile.log"
attach_status=${PIPESTATUS[0]}
set -e
container_status=$(docker inspect --format '{{.State.ExitCode}}' "$container")
if [[ "$attach_status" -ne 0 ]]; then
    exit "$attach_status"
fi
exit "$container_status"
