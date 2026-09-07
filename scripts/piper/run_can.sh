#!/usr/bin/env bash
# One-click: bring up BOTH Piper CAN buses -- can_left + can_right @ 1 Mbaud.
#
# Each arm's USB-CAN adapter is matched by its STABLE SERIAL, not its USB port, so
# re-plugging into a different port never breaks this (the vendor can_config.sh /
# can_muti_activate.sh key on port paths, which drift -- that is why they kept failing
# with "USB port ... not found"). The left/right serials below were confirmed against
# cobot_magic's can_config.sh (1-8:1.0=can_left, 1-10:1.0=can_right) on 2026-07-04.
#
# Usage:
#   scripts/piper/run_can.sh           # activate can_left + can_right
#   scripts/piper/run_can.sh --list    # just list every CAN adapter (no sudo)
#
# Needs sudo (loads gs_usb, renames + activates the interfaces). If sudo fails with a
# "nosuid / effective uid is not 0" error, your shell has no_new_privs set -- open a
# plain login terminal (check: grep NoNewPrivs /proc/self/status  ==> want 0).
set -o pipefail

# ---- per-arm adapter identity (edit only if you swap an adapter; --list to rediscover)
LEFT_SERIAL="003F00414148571320343133"
RIGHT_SERIAL="003100194148571320343133"
BITRATE=1000000

declare -A WANT=( ["can_left"]="$LEFT_SERIAL" ["can_right"]="$RIGHT_SERIAL" )
ARPHRD_CAN=280

_arm_of() { case "$1" in can_left) echo left;; can_right) echo right;; *) echo "$1";; esac; }

# CAN iface -> its USB interface path (e.g. "1-8:1.0"). Pure sysfs, no sudo.
_port_of_iface() { basename "$(readlink -f "/sys/class/net/$1/device" 2>/dev/null)"; }

# CAN iface -> the stable USB device serial. Walks up to the usb device dir. No sudo.
_serial_of_iface() {
    local d; d="$(readlink -f "/sys/class/net/$1/device" 2>/dev/null)"
    while [ -n "$d" ] && [ "$d" != "/" ]; do
        [ -f "$d/serial" ] && { cat "$d/serial"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}

_can_ifaces() {
    local iface
    for iface in $(ls /sys/class/net/ 2>/dev/null); do
        [ "$(cat "/sys/class/net/$iface/type" 2>/dev/null)" = "$ARPHRD_CAN" ] && echo "$iface"
    done
}

_iface_for_serial() {
    local want="$1" iface
    for iface in $(_can_ifaces); do
        [ "$(_serial_of_iface "$iface")" = "$want" ] && { echo "$iface"; return 0; }
    done
    return 1
}

list_adapters() {
    echo "[can-dual] CAN adapters currently present:"
    local iface any=0
    for iface in $(_can_ifaces); do
        any=1
        local port serial tag=""
        port="$(_port_of_iface "$iface")"; serial="$(_serial_of_iface "$iface")"
        [ "$serial" = "$LEFT_SERIAL" ] && tag="  <- left arm (can_left)"
        [ "$serial" = "$RIGHT_SERIAL" ] && tag="  <- right arm (can_right)"
        printf "  %-10s USB %-8s serial %s%s\n" "$iface" "$port" "$serial" "$tag"
    done
    [ "$any" = 1 ] || echo "  (none -- is gs_usb loaded and are the adapters plugged in?)"
}

_preflight_sudo() {
    sudo -n true 2>/dev/null && return 0
    if ! sudo true; then
        echo "[can-dual] ERROR: sudo cannot escalate. If it said 'nosuid / effective uid is not 0',"
        echo "           your shell has no_new_privs set -- run this from a plain login terminal"
        echo "           (check: grep NoNewPrivs /proc/self/status  ==> must be 0)."
        exit 1
    fi
}

activate_one() {
    local name="$1" serial="$2" iface port
    iface="$(_iface_for_serial "$serial")"
    if [ -z "$iface" ]; then
        echo "[can-dual] ✗ $name: no adapter with serial $serial found -- is the $(_arm_of "$name") arm's USB plugged in? (try: $0 --list)"
        return 1
    fi
    port="$(_port_of_iface "$iface")"

    # Fast path: the right adapter is already named + UP at the right bitrate.
    if [ "$iface" = "$name" ] \
        && [ "$(ip -br link show "$name" 2>/dev/null | awk '{print $2}')" = "UP" ] \
        && [ "$(ip -details link show "$name" 2>/dev/null | grep -oP 'bitrate \K[0-9]+')" = "$BITRATE" ]; then
        echo "[can-dual] ✓ $name already UP @ $BITRATE (USB $port)"
        return 0
    fi

    echo "[can-dual] → $name: adapter at USB $port (currently '$iface')"
    # If the target name is held by a DIFFERENT (stale/wrong) iface, park it aside so
    # the rename cannot collide.
    if [ "$iface" != "$name" ] && ip link show "$name" >/dev/null 2>&1; then
        echo "[can-dual]   '$name' is taken by another iface; parking it as ${name}_stale"
        sudo ip link set "$name" down 2>/dev/null || true
        sudo ip link set "$name" name "${name}_stale" 2>/dev/null || true
    fi
    sudo ip link set "$iface" down 2>/dev/null || true
    sudo ip link set "$iface" type can bitrate "$BITRATE" || { echo "[can-dual] ✗ $name: set bitrate failed"; return 1; }
    if [ "$iface" != "$name" ]; then
        sudo ip link set "$iface" name "$name" || { echo "[can-dual] ✗ $name: rename $iface -> $name failed"; return 1; }
    fi
    sudo ip link set "$name" up || { echo "[can-dual] ✗ $name: bring up failed"; return 1; }
    echo "[can-dual] ✓ $name UP @ $BITRATE (USB $port)"
}

main() {
    if [ "${1:-}" = "--list" ] || [ "${1:-}" = "list" ]; then
        list_adapters
        return 0
    fi

    _preflight_sudo
    # gs_usb provides the candleLight USB-CAN interfaces; harmless if already loaded.
    sudo modprobe gs_usb || { echo "[can-dual] ERROR: could not load gs_usb"; exit 1; }

    local rc=0 name
    for name in can_left can_right; do
        activate_one "$name" "${WANT[$name]}" || rc=1
    done

    echo
    echo "[can-dual] result:"
    for name in can_left can_right; do
        local state; state="$(ip -br link show "$name" 2>/dev/null | awk '{print $2}')"
        printf "  %-10s %s\n" "$name" "${state:-MISSING}"
    done
    if [ "$rc" = 0 ] \
        && [ "$(ip -br link show can_left 2>/dev/null | awk '{print $2}')" = "UP" ] \
        && [ "$(ip -br link show can_right 2>/dev/null | awk '{print $2}')" = "UP" ]; then
        echo "[can-dual] ✓ both arms' CAN buses are UP. Verify traffic with:  candump can_left   /   candump can_right"
        return 0
    fi
    echo "[can-dual] ✗ not all interfaces came up -- see above (run '$0 --list' to inspect adapters)."
    return 1
}

main "$@"
