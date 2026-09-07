#!/usr/bin/env bash
# Send BOTH Piper arms to their idle/park REST pose, SIMULTANEOUSLY.
# The counterpart of go_begin.sh: `rest_joints` is reached ON DEMAND only -- it is never
# auto-entered (init and the post-recording reset both go to the BEGIN pose).
# Usage: go_rest.sh [extra go_begin.py args...]
#   go_rest.sh                              -> both arms -> rest_joints (at the same time)
#   go_rest.sh --arm left --capture --write -> save that arm's CURRENT joints as its rest pose
# Needs BOTH arm nodes up (scripts/piper/run_arm.sh). Runs the Show-Harness .venv.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

trap 'trap - INT TERM EXIT' INT TERM EXIT  # go_begin.py cleans up its own connections

source_ros
source_piper_ws
ensure_roscore

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || _die "venv python not found at $VENV_PY (install requirements-real.txt)"

# --rest targets rest_joints; pass-through args (e.g. --arm right, --capture --write).
"$VENV_PY" "$REPO_ROOT/scripts/piper/go_begin.py" --rest "$@"
