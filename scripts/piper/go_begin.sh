#!/usr/bin/env bash
# Reset: send BOTH Piper arms to their BEGIN pose, SIMULTANEOUSLY.
# Usage: go_begin.sh [extra go_begin.py args...]
#   go_begin.sh                             -> both arms -> begin_joints (at the same time)
#   go_begin.sh --arm right                 -> just the right arm
#   go_begin.sh --arm left --capture --write -> save that arm's CURRENT joints as its
#                                               begin pose (does not move)
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

"$VENV_PY" "$REPO_ROOT/scripts/piper/go_begin.py" "$@"
