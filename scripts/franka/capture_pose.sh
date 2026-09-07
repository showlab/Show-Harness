#!/usr/bin/env bash
# Capture the Franka's CURRENT joint configuration as a NAMED start pose.
# Usage: capture_pose.sh [--name <pose>] [--write] [--activate]
#   capture_pose.sh                              -> print the current joints (read-only)
#   capture_pose.sh --name chess_ready --write   -> save as poses.chess_ready
#   capture_pose.sh --name chess_ready --write --activate
#       -> + begin_pose: chess_ready + move_to_begin_on_init: true (every rollout
#          homes to it before starting)
# Hand-guide/jog the arm to the wanted configuration first. Read-only robot query;
# the arm never moves. poses.default is protected (= go_home_client.py HOME, the
# lab-shared reset -- never modified from here). Runs the Show-Harness .venv.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "ERROR: venv python not found at $VENV_PY" >&2; exit 1; }

exec "$VENV_PY" "$REPO_ROOT/scripts/franka/capture_pose.py" "$@"
