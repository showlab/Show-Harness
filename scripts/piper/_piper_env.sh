# Shared helpers for the Piper launchers. SOURCED by the run_*.sh scripts, not run.
#
# We deliberately avoid `set -u` here: ROS/catkin setup.bash files reference unset
# variables and would abort under it. Robustness comes from explicit checks + traps
# in each launcher instead.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COBOT="${COBOT_MAGIC_DIR:-$HOME/cobot_magic}"
PIPER_WS="$COBOT/Piper_ros_private-ros-noetic/devel/setup.bash"
CAMERA_WS="$COBOT/camera_ws/devel/setup.bash"
ROSCORE_LOG="/tmp/showharness_roscore.log"

# --- Console styling (colors only on a TTY; NO_COLOR disables them) -----------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_DIM=""
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_CYAN=""
fi

TAG="piper"  # launchers override this so every line is attributable

info() { echo "${C_CYAN}[${TAG}]${C_RESET} $*"; }
warn() { echo "${C_YELLOW}${C_BOLD}[${TAG}] WARNING:${C_RESET}${C_YELLOW} $*${C_RESET}"; }
_die() { echo "${C_RED}${C_BOLD}[${TAG}] ERROR:${C_RESET}${C_RED} $*${C_RESET}" >&2; exit 1; }

# Header printed once at launch: what this terminal runs and how to stop it.
# NOTE: success_box pads with printf %-54s, which counts BYTES -- keep the box
# text pure ASCII (use "--", not an em dash) or the right border drifts.
banner() {
    echo "${C_BOLD}${C_CYAN}────────────────────────────────────────────────────────────${C_RESET}"
    echo "${C_BOLD}${C_CYAN} $1${C_RESET}"
    [ $# -gt 1 ] && echo "${C_DIM} $2${C_RESET}"
    echo "${C_DIM} Ctrl+C stops cleanly (no orphaned nodes)${C_RESET}"
    echo "${C_BOLD}${C_CYAN}────────────────────────────────────────────────────────────${C_RESET}"
}

success_box() {
    echo "${C_BOLD}${C_GREEN}"
    echo "  ┌──────────────────────────────────────────────────────────┐"
    printf "  │  ✓ %-54s │\n" "$1"
    [ $# -gt 1 ] && printf "  │    %-54s │\n" "$2"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo "${C_RESET}"
}

# Background readiness watcher: wait until every given ROS topic delivers one
# message, then print the distinct green success box (interleaved with the
# roslaunch log). Warns instead if not confirmed before the deadline. Launchers
# run this with `&` BEFORE the blocking roslaunch and kill it in cleanup.
watch_ready() {
    local deadline_s="$1" what="$2" detail="$3"; shift 3
    local deadline=$(( $(date +%s) + deadline_s )) topic
    for topic in "$@"; do
        while ! timeout 5 rostopic echo -n1 --noarr "$topic" >/dev/null 2>&1; do
            if [ "$(date +%s)" -ge "$deadline" ]; then
                warn "not confirmed within ${deadline_s}s: no message on $topic yet -- check the log above."
                return 1
            fi
            sleep 1
        done
    done
    success_box "$what" "$detail"
}

source_ros() {
    [ -f /opt/ros/noetic/setup.bash ] || _die "ROS Noetic not found at /opt/ros/noetic"
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash
}

source_piper_ws() {
    [ -f "$PIPER_WS" ] || _die "Piper workspace not built at $PIPER_WS"
    # shellcheck disable=SC1091
    source "$PIPER_WS"
}

source_camera_ws() {
    [ -f "$CAMERA_WS" ] || _die "camera workspace not built at $CAMERA_WS"
    # shellcheck disable=SC1091
    source "$CAMERA_WS"
}

# Ensure a roscore is up WITHOUT owning it: start a detached one only if none is
# running, so the three terminals share one master and no launcher kills it out
# from under the others on exit. `pkill -f roscore` clears it when you are done.
ensure_roscore() {
    if timeout 3 rosnode list >/dev/null 2>&1; then
        return 0
    fi
    echo "[piper] no roscore running; starting one (detached, log: $ROSCORE_LOG) ..."
    nohup roscore >"$ROSCORE_LOG" 2>&1 &
    for _ in $(seq 1 30); do
        timeout 2 rosnode list >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    _die "roscore did not come up within 15s (see $ROSCORE_LOG)"
}

# Kill processes matching a pattern, escalating SIGINT -> SIGTERM -> SIGKILL, so no
# orphaned nodes are left behind. Never matches roscore/rosmaster.
kill_pattern() {
    local pat="$1"
    pgrep -f "$pat" >/dev/null 2>&1 || return 0
    pkill -INT -f "$pat" 2>/dev/null || true
    for _ in $(seq 1 10); do
        pgrep -f "$pat" >/dev/null 2>&1 || return 0
        sleep 0.3
    done
    pkill -TERM -f "$pat" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "$pat" 2>/dev/null || true
}
