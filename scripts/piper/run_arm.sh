#!/usr/bin/env bash
# Terminal 2 (single command): BOTH Piper arm nodes in software-control mode, via
# cobot_magic's stock start_ms_piper.launch (left = can_left, right = can_right).
# Usage: run_arm.sh [mode] [auto_enable]   (defaults: mode=1 auto_enable=true)
# Activate BOTH CAN buses first (can_muti_activate.sh).
# WARNING: enabling closes BOTH grippers to width 0 -- clear all fingers first.
# Needs the 'aloha' conda env (piper_sdk). Ctrl+C stops cleanly (no orphaned nodes).
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

TAG="arm-dual"
MODE="${1:-1}"
AUTO_ENABLE="${2:-true}"
ARM_PATTERN="piper_start_ms_node"
LAUNCHED=0
WATCHER_PID=""

cleanup() {
    trap '' INT TERM
    trap - EXIT
    [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null
    [ "$LAUNCHED" = "1" ] || return 0
    info "stopping ..."
    kill_pattern "$ARM_PATTERN"
}
trap cleanup INT TERM EXIT

banner "Piper arms — BOTH nodes (start_ms_piper.launch)" \
       "mode=$MODE auto_enable=$AUTO_ENABLE  (left=can_left, right=can_right)"

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || _die "conda.sh not found at $CONDA_SH (the arm nodes need the 'aloha' env)"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate aloha || _die "could not 'conda activate aloha'"

source_ros
source_piper_ws
command -v roslaunch >/dev/null 2>&1 || _die "roslaunch not on PATH after sourcing ROS"
python3 -c "import piper_sdk" 2>/dev/null || _die "piper_sdk not importable in this env (is 'aloha' active?)"
rospack find piper >/dev/null 2>&1 || _die "piper package not found (source the Piper workspace)"
ensure_roscore

# Clear any stale arm node (single- OR dual-arm) from a previous run.
kill_pattern "$ARM_PATTERN"

info "launching BOTH arms ..."
warn "enabling CLOSES both grippers to width 0 -- clear all fingers now."
LAUNCHED=1
# Watch in the background; the green box below prints once BOTH arms stream
# joint states (proof the CAN buses and both nodes are actually alive).
watch_ready 60 "ARMS UP -- left + right joint states streaming" \
    "/puppet/joint_left + /puppet/joint_right alive" \
    /puppet/joint_left /puppet/joint_right &
WATCHER_PID=$!
roslaunch piper start_ms_piper.launch mode:="$MODE" auto_enable:="$AUTO_ENABLE"
