#!/usr/bin/env bash
# Manual launch only. Stop at the first failing subprocess; never start training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mode=${1:-eager}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    all) modes=(eager cast precast-cat mixed-cat) ;;
    eager|cast|precast-cat|mixed-cat) modes=("$mode") ;;
    *) echo "Usage: bash $0 [all|eager|cast|precast-cat|mixed-cat] [Python options]" >&2; exit 2 ;;
esac

root=${REPRO_LOG_ROOT:-../diagnostics}
mkdir -p "$root"
run_dir=$(mktemp -d "$root/cast-repro-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Logs: $run_dir"
git rev-parse HEAD > "$run_dir/commit.txt"
git diff HEAD > "$run_dir/worktree.diff"
cp scripts/debug/reproduce_a5_cast.py scripts/debug/run_a5_cast_repro.sh "$run_dir/"
export ASCEND_LAUNCH_BLOCKING=1
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
for item in "${modes[@]}"; do
    set +e
    python3 -u scripts/debug/reproduce_a5_cast.py --device "${REPRO_DEVICE:-npu:0}" \
        --mode "$item" "$@" --log-dir "$run_dir/$item" 2>&1 | tee "$run_dir/$item.full.log"
    status=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${status[0]}" > "$run_dir/$item.exit_code.txt"
    if [[ ${status[0]} -ne 0 ]]; then exit "${status[0]}"; fi
    if [[ ${status[1]} -ne 0 ]]; then exit "${status[1]}"; fi
done
