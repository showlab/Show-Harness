#!/usr/bin/env bash
# Capture ONE arm's Z safety floor (minimum EEF height). Per-arm by design: you rest a
# single gripper on the table at a time.
# Usage: capture_z_floor.sh --arm <left|right> [--write]
#   capture_z_floor.sh --arm left            -> print that arm's current EEF height
#   capture_z_floor.sh --arm left --write    -> save it as arms.left.z_floor_m
# Rest that arm's gripper ON THE TABLETOP first: the captured height becomes the floor.
# Needs the arm nodes up (scripts/piper/run_arm.sh). Runs the Show-Harness .venv.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

trap 'trap - INT TERM EXIT' INT TERM EXIT  # capture_z_floor.py cleans up its own connection

source_ros
source_piper_ws
ensure_roscore

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || _die "venv python not found at $VENV_PY (install requirements-real.txt)"

"$VENV_PY" "$REPO_ROOT/scripts/piper/capture_z_floor.py" "$@"
