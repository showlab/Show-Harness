#!/usr/bin/env python3
"""Move the Piper arms to a saved pose. DUAL-ARM: both arms move SIMULTANEOUSLY.

The rig is dual-arm, so by default this drives BOTH arms at once (one thread each) to
the pose in the unified configs/robot_piper.yaml. This is the reset command -- the same
move the teleop collector runs when a recording stops and run_real runs on init.

START POSES ARE NAMED. An arm can have any number of them under ``arms.<side>.poses``;
the shared ``begin_pose`` key (or ``--begin-pose`` on the runners) picks the active one.
Capturing a pose on ONE arm also writes the OTHER arm's MIRROR of it by default
(``mirror_capture`` in the config, or --no-mirror here), so a start pose is defined once.

    # reset BOTH arms to the active BEGIN pose (what every recording/rollout starts from)
    python scripts/piper/go_begin.py

    # ... a specific named pose / the idle-park REST pose (backs scripts/piper/go_rest.sh)
    python scripts/piper/go_begin.py --pose wide
    python scripts/piper/go_begin.py --rest

    # list the poses defined for each arm
    python scripts/piper/go_begin.py --list

    # define a pose = an arm's CURRENT joints (jog it there first; does not move):
    # writes arms.left.poses.wide AND the mirrored arms.right.poses.wide
    python scripts/piper/go_begin.py --arm left --pose wide --capture --write
    python scripts/piper/go_begin.py --arm right --rest --capture --write

Prerequisites: BOTH arm nodes running in mode 1 (scripts/piper/run_arm.sh), with ROS +
the Piper workspace sourced so rospy / piper_msgs import.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.piper.config import (
    POSE_MAP_KEY,
    SIDES,
    arm_blocks,
    pose_joints,
    pose_names,
)
from core.piper.poses import go_begin, go_begin_dual, mirror_joints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move the Piper arms to a saved pose (both by default).")
    p.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_piper.yaml"))
    p.add_argument(
        "--arm",
        default="both",
        choices=["both", "left", "right"],
        help="Which arm(s) to act on. 'both' moves them SIMULTANEOUSLY.",
    )
    p.add_argument(
        "--rest",
        action="store_true",
        help="Target the REST (idle/park) pose instead of the BEGIN start pose.",
    )
    p.add_argument(
        "--pose",
        default=None,
        help="Name of the start pose under arms.<side>.poses to move to / capture into. "
        "Default: the config's `begin_pose` (or `rest_pose` with --rest).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_poses",
        help="List the poses defined for each arm and exit (no hardware needed).",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="With --capture --write, do NOT also write the other arm's mirrored pose "
        "(overrides `mirror_capture` in the config).",
    )
    p.add_argument("--time-s", type=float, default=3.0, help="Move duration (s).")
    p.add_argument(
        "--keep-grip",
        action="store_true",
        help="Do NOT open the gripper on arrival (BEGIN opens it by default so a reset "
        "always leaves the hand empty).",
    )
    p.add_argument("--mock-robots", action="store_true", help="Simulated arms (no ROS/CAN).")
    p.add_argument(
        "--capture",
        action="store_true",
        help="Read the arm's CURRENT joints and print them as the pose (does not move). "
        "Requires a single --arm (a captured pose is per-arm).",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="With --capture, write the captured joints into that arm's block in the config.",
    )
    return p.parse_args()


def _block_end(text: str, start: int, indent: int) -> int:
    """Where the block opened at ``indent`` ends: the first later line indented <= it."""
    for line in re.finditer(r"^(?P<ind>[ \t]*)(?P<content>\S.*)$", text[start:], re.MULTILINE):
        if len(line.group("ind")) <= indent:
            return start + line.start()
    return len(text)


def _find_key(text: str, start: int, end: int, key: str):
    """The ``key:`` line inside ``text[start:end]`` (indented; first match), or None."""
    pat = re.compile(rf"^(?P<indent>[ \t]+){re.escape(key)}:.*$", re.MULTILINE)
    return pat.search(text, start, end)


def _render_missing(keys: Sequence[str], value: str, indent: int) -> str:
    """YAML lines creating a missing (possibly nested) key path, e.g. ``poses:\\n  x: [...]``."""
    lines = []
    for depth, key in enumerate(keys):
        pad = " " * (indent + 2 * depth)
        last = depth == len(keys) - 1
        lines.append(f"{pad}{key}: {value}" if last else f"{pad}{key}:")
    return "\n".join(lines) + "\n"


def _write_arm_joints(config_path: Path, side: str, keys: Sequence[str], joints: list) -> bool:
    """Set ``arms.<side>.<keys...>`` to an inline list, preserving the rest of the config.

    ``keys`` is a key PATH, so this writes both a flat pose (``("rest_joints",)``) and a
    named one (``("poses", "default")``), creating any missing level. Anchors on the arm's
    block so the left and right entries (identical key names) are never confused, and
    re-parses the result before writing as a safety net: the file is never overwritten
    with YAML that no longer loads or that lost the value.
    """
    text = config_path.read_text(encoding="utf-8")
    rounded = [round(float(v), 5) for v in joints]
    value = f"[{', '.join(str(v) for v in rounded)}]"
    # Find the `  <side>:` line inside `arms:`, then walk the key path under it.
    arm_pat = re.compile(rf"^(?P<indent>[ \t]+){re.escape(side)}:[ \t]*$", re.MULTILINE)
    m = arm_pat.search(text)
    if not m:
        print(f"[go-begin] WARNING: no `{side}:` block found under `arms:` in {config_path}")
        return False
    indent = len(m.group("indent"))
    start, end = m.end(), _block_end(text, m.end(), len(m.group("indent")))

    new_text = None
    for depth, key in enumerate(keys):
        km = _find_key(text, start, end, key)
        last = depth == len(keys) - 1
        if km is None:
            # This level (and everything under it) is absent -> append it to the block.
            head = text[:end]
            if not head.endswith("\n"):
                head += "\n"
            new_text = head + _render_missing(keys[depth:], value, indent + 2) + text[end:]
            break
        if last:
            new_text = (
                text[: km.start()] + f"{km.group('indent')}{key}: {value}" + text[km.end() :]
            )
            break
        indent = len(km.group("indent"))
        start = km.end()
        end = _block_end(text, start, indent)

    if new_text is None:  # unreachable for a non-empty key path; be explicit anyway
        return False
    path = ".".join(keys)
    try:
        parsed = yaml.safe_load(new_text)
        got: Any = ((parsed or {}).get("arms") or {}).get(side, {})
        for key in keys:
            got = (got or {}).get(key) if isinstance(got, dict) else None
        ok = isinstance(got, list) and len(got) == len(rounded)
    except yaml.YAMLError:
        ok = False
    if not ok:
        print(
            f"[go-begin] WARNING: editing arms.{side}.{path} would corrupt {config_path}; "
            f"not writing. Set it by hand: {keys[-1]}: {value}"
        )
        return False
    config_path.write_text(new_text, encoding="utf-8")
    return True


def _make_robot(side: str, mock: bool):
    if mock:
        from core.piper.piper_interface import MockPiperRobot

        robot = MockPiperRobot(arm=side)
    else:
        from core.piper.piper_interface import PiperInterface

        robot = PiperInterface(arm=side)
    robot.connect()
    return robot


def _print_poses(cfg: dict[str, Any]) -> None:
    """Show every named pose per arm, marking the ones the config currently selects."""
    active = {
        "begin": str(cfg.get("begin_pose") or ""),
        "rest": str(cfg.get("rest_pose") or ""),
    }
    for side in SIDES:
        names = pose_names(cfg, side)
        block = arm_blocks(cfg)[side]
        print(f"[go-begin] {side} arm:")
        for name in names:
            marks = [role.upper() for role, sel in active.items() if sel == name]
            tag = f"  <- active {'/'.join(marks)}" if marks else ""
            joints = [round(float(v), 5) for v in block[POSE_MAP_KEY][name]]
            print(f"[go-begin]   poses.{name}: {joints}{tag}")
        if not names:
            print("[go-begin]   (no named poses -- using the flat begin_joints/rest_joints)")
        for flat in ("begin_joints", "rest_joints"):
            if block.get(flat) is not None:
                print(f"[go-begin]   {flat}: {[round(float(v), 5) for v in block[flat]]}")
    for role, name in active.items():
        if name:
            print(f"[go-begin] active {role} pose: {name!r} (set `{role}_pose` in the config)")


def _target_keys(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[tuple[str, ...], str]:
    """Resolve what this invocation targets: the config key PATH under ``arms.<side>``
    and a human label. A named pose (--pose, else the config's begin_pose/rest_pose)
    lives under ``poses.<name>``; with no name selected we fall back to the flat
    ``begin_joints`` / ``rest_joints`` key."""
    selector = "rest_pose" if args.rest else "begin_pose"
    flat_key = "rest_joints" if args.rest else "begin_joints"
    name = args.pose or cfg.get(selector)
    if name:
        return (POSE_MAP_KEY, str(name)), f"{'rest' if args.rest else 'begin'}:{name}"
    return (flat_key,), ("rest" if args.rest else "begin")


def _target_joints(
    cfg: dict[str, Any], side: str, keys: Sequence[str], args: argparse.Namespace
):
    """The saved joints this invocation should move to (None -> not configured yet)."""
    if keys[0] == POSE_MAP_KEY:
        return pose_joints(cfg, side, keys[1])
    return arm_blocks(cfg)[side].get(keys[0])


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    blocks = arm_blocks(cfg)  # validates that BOTH arms are configured

    if args.list_poses:  # config-only; no hardware needed
        _print_poses(cfg)
        return 0

    keys, label = _target_keys(cfg, args)
    path = ".".join(keys)
    # BEGIN always leaves the hand empty: the gripper is opened once the arm arrives, so a
    # reset can never start the next recording/rollout still holding something. --keep-grip
    # opts out; the REST (park) pose leaves the gripper as-is unless asked.
    open_gripper = (not args.keep_grip) and not args.rest
    # Capturing one arm's pose also saves the OTHER arm's mirror of it, so a start pose is
    # defined once (the arms face each other: mirroring negates the yaw joints -- see
    # core.piper.poses.MIRROR_SIGNS). Config `mirror_capture`, --no-mirror opts out.
    mirror = bool(cfg.get("mirror_capture", True)) and not args.no_mirror

    sides = list(SIDES) if args.arm == "both" else [args.arm]
    if args.capture and len(sides) != 1:
        print("[go-begin] ERROR: --capture needs a single --arm (left or right).")
        return 1

    robots: dict[str, object] = {}
    exit_code = 0
    try:
        for side in sides:
            robots[side] = _make_robot(side, args.mock_robots)

        if args.capture:
            side = sides[0]
            other = next(s for s in SIDES if s != side)
            joints = np.asarray(robots[side].get_joint_positions(), dtype=float)[:6]  # type: ignore[attr-defined]
            rounded = [round(float(v), 5) for v in joints]
            print(f"[go-begin] {side} current joints (rad): {rounded}")
            print(f"[go-begin] YAML (arms.{side}.{path}): {keys[-1]}: [{', '.join(str(v) for v in rounded)}]")
            mirrored = mirror_joints(joints, cfg.get("mirror_signs"))
            if mirror:
                print(f"[go-begin] mirrored for the {other} arm: {mirrored}")
            if args.write:
                config_path = Path(args.robot_config)
                if _write_arm_joints(config_path, side, keys, list(joints)):
                    print(f"[go-begin] wrote arms.{side}.{path} into {args.robot_config}")
                else:
                    exit_code = 1
                if mirror and exit_code == 0:
                    if _write_arm_joints(config_path, other, keys, mirrored):
                        # How to drive the OTHER arm to the pose we just mirrored, so the
                        # verification step is a copy-paste (a named pose vs the flat rest).
                        target = (
                            f"--pose {keys[1]}" if keys[0] == POSE_MAP_KEY else "--rest"
                        )
                        print(
                            f"[go-begin] wrote arms.{other}.{path} (MIRRORED) into "
                            f"{args.robot_config} -- verify it on hardware "
                            f"(`go_begin.sh --arm {other} {target}`), and re-capture that "
                            f"arm directly if the mount is not symmetric."
                        )
                    else:
                        exit_code = 1
        elif len(sides) == 1:
            side = sides[0]
            moved = go_begin(
                robots[side], _target_joints(cfg, side, keys, args), time_to_go=args.time_s,
                label=f"{side} {label}", open_gripper=open_gripper,
            )
            if not moved:
                exit_code = 1
        else:
            # The reset: BOTH arms move at the same time.
            print(f"[go-begin] moving BOTH arms to {label} simultaneously ...")
            moved = go_begin_dual(
                robots,
                {s: _target_joints(cfg, s, keys, args) for s in sides},
                time_to_go=args.time_s,
                label=label,
                open_gripper=open_gripper,
            )
            if not moved:
                exit_code = 1
    except Exception as exc:  # noqa: BLE001 - surface clearly
        print(f"[go-begin] ERROR: {exc}")
        exit_code = 1
    finally:
        for robot in robots.values():
            try:
                robot.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
