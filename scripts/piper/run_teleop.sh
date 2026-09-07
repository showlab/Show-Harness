#!/usr/bin/env bash
# Terminal 3 (single command): DUAL-arm keyboard teleop + rollout recorder.
# Usage: run_teleop.sh [save_dir] [extra collect_rollouts_piper.py args...]
#   default save_dir: <repo>/rollouts/dual
#   e.g.  run_teleop.sh data/my_task --mode A
# Needs Terminals 1 (cameras) + 2 (BOTH arms: run_arm.sh) up.
# Runs the Show-Harness .venv (topics only).
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

cleanup() {
    trap - INT TERM EXIT
    echo "[teleop-dual] stopped."
}
trap cleanup INT TERM EXIT

source_ros
source_piper_ws
ensure_roscore

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || _die "venv python not found at $VENV_PY (install requirements-real.txt)"

SAVE="${1:-$REPO_ROOT/rollouts/dual}"
[ "$#" -gt 0 ] && shift

echo "[teleop-dual] collecting -> $SAVE"
# The teleop script handles its own KeyboardInterrupt (stops the recording, closes
# the session/subscribers), so Ctrl+C exits cleanly with no orphaned nodes.
"$VENV_PY" "$REPO_ROOT/scripts/trajectory/collect_rollouts_piper.py" "$SAVE" "$@"
