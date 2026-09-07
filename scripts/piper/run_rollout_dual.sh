#!/usr/bin/env bash
# Autonomous DUAL-ARM closed-loop rollout on the Piper rig. Pass-through args go to
# scripts/run_real_dual.py (e.g. --mode A|B, --task-left/--task-right, --vlm-url ...). Needs
# the cameras + BOTH arm nodes up (scripts/piper/run_can.sh + run_arm.sh per arm).
# scripts/run_real_dual.py handles its own Ctrl+C (compiles the video(s), homes both arms).
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

source_ros
source_piper_ws
ensure_roscore

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || _die "venv python not found at $VENV_PY (install requirements-real.txt)"

exec "$VENV_PY" "$REPO_ROOT/scripts/run_real_dual.py" \
    --robot-config "$REPO_ROOT/configs/robot_piper.yaml" "$@"
