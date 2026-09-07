#!/usr/bin/env bash
# Terminal 1 (single command): DaBai DC cameras via astra_camera.
# Publishes /camera_{f,l,r}/color/image_raw. Ctrl+C stops cleanly (no orphaned nodes).
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_piper_env.sh
source "$HERE/_piper_env.sh"

TAG="cameras"
CAM_PATTERN="astra_camera|multi_camera.launch"
LAUNCHED=0  # set to 1 only once we actually start the cameras
WATCHER_PID=""

cleanup() {
    # Ignore repeat INT/TERM so the kill escalation always runs to completion;
    # clear EXIT so this does not re-enter.
    trap '' INT TERM
    trap - EXIT
    [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null
    # Only tear down nodes if THIS invocation started them. A failed prerequisite or a
    # Ctrl+C during startup must not kill healthy camera nodes another terminal is running.
    [ "$LAUNCHED" = "1" ] || return 0
    info "stopping ..."
    kill_pattern "$CAM_PATTERN"
}
trap cleanup INT TERM EXIT

banner "Piper cameras — astra multi_camera" \
       "publishes /camera_f (front) + /camera_l /camera_r (wrists)"

source_ros
source_camera_ws
command -v roslaunch >/dev/null 2>&1 || _die "roslaunch not on PATH after sourcing ROS"
rospack find astra_camera >/dev/null 2>&1 || _die "astra_camera package not found (source the camera_ws)"
ensure_roscore

# Clear any stale camera nodes from a previous crashed run before starting fresh (the
# intended replace-then-relaunch; runs only after every prerequisite passed).
kill_pattern "$CAM_PATTERN"

info "launching astra_camera multi_camera.launch ..."
LAUNCHED=1
# Watch in the background; the green box below prints once all three streams
# actually deliver an image (not merely once the nodes register).
watch_ready 90 "CAMERAS UP -- all 3 streams publishing" \
    "front /camera_f + wrists /camera_l /camera_r" \
    /camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw &
WATCHER_PID=$!
roslaunch astra_camera multi_camera.launch
