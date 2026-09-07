#!/usr/bin/env bash
# Autonomous closed-loop rollout on the left Piper. Pass-through args go to
# scripts/run_real.py (e.g. --vlm-url http://localhost:8000/v1). Needs the cameras + arm
# node up. Runs the Show-Harness .venv (topics only); scripts/run_real.py handles its own
# Ctrl+C (compiles the video, homes the arm) so there are no orphaned processes.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

source_ros
source_piper_ws
ensure_roscore

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || _die "venv python not found at $VENV_PY (install requirements-real.txt)"

exec "$VENV_PY" "$REPO_ROOT/scripts/run_real.py" \
    --robot-config "$REPO_ROOT/configs/robot_piper.yaml" "$@"
