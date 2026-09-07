#!/usr/bin/env bash
# Capture the Franka Z safety floor (minimum EEF height) as a NAMED setting.
# Usage: capture_z_floor.sh [--name <setting>] [--write] [--activate]
#   capture_z_floor.sh                            -> print the current EEF height (read-only)
#   capture_z_floor.sh --name drawer --write      -> save it as z_floors.drawer
#   capture_z_floor.sh --name drawer --write --activate -> + set z_floor_name: drawer
# Rest the (closed) gripper ON THE WORK SURFACE first: the captured height becomes the
# floor for that setting. Read-only robot query; the arm never moves. Runs the Show-Harness .venv.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "ERROR: venv python not found at $VENV_PY" >&2; exit 1; }

exec "$VENV_PY" "$REPO_ROOT/scripts/franka/capture_z_floor.py" "$@"
