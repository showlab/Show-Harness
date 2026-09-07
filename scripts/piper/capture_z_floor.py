#!/usr/bin/env python3
"""Capture the Z safety floor (minimum EEF height) for ONE Piper arm.

The z-floor is the base-frame height of the end effector when that arm's gripper just
touches the tabletop; the controller blocks any downward MOVE P below it so the stiff
Piper position control never presses into the table. It is a PER-ARM calibration value
(``arms.<side>.z_floor_m`` in the unified configs/robot_piper.yaml) -- you rest one
gripper on the table at a time, so this always targets a single arm.

    # rest the LEFT gripper on the tabletop first, then read + persist its height
    python scripts/piper/capture_z_floor.py --arm left --write

    # just print it (does not touch the config)
    python scripts/piper/capture_z_floor.py --arm right

This overwrites that arm's z_floor_m in place, so re-capturing never means editing the
config by hand.

Prerequisites: the arm node must be running (scripts/piper/run_arm.sh) with ROS + the
Piper workspace sourced. The measured pose is published in any node mode, so hand-guide /
jog the gripper onto the table first.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.piper.config import SIDES, arm_blocks

KEY = "z_floor_m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture one Piper arm's Z safety floor.")
    p.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_piper.yaml"))
    p.add_argument("--arm", required=True, choices=list(SIDES), help="Which arm's floor to capture.")
    p.add_argument("--mock-robot", action="store_true", help="Use a simulated arm (no ROS/CAN).")
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the captured height into arms.<side>.z_floor_m (overwrites in place).",
    )
    return p.parse_args()


def _write_arm_scalar(config_path: Path, side: str, key: str, value: float) -> bool:
    """Set ``arms.<side>.<key>`` to ``value``, preserving the rest of the config.

    Anchors on the arm's block so the left and right entries (identical key names) are
    never confused; appends the key if that block does not have it yet. Re-parses the
    result before writing so the file is never left un-loadable or missing the value.
    """
    text = config_path.read_text(encoding="utf-8")
    rounded = round(float(value), 5)
    arm_pat = re.compile(rf"^(?P<indent>[ \t]+){re.escape(side)}:[ \t]*$", re.MULTILINE)
    m = arm_pat.search(text)
    if not m:
        print(f"[capture-z-floor] WARNING: no `{side}:` block found under `arms:` in {config_path}")
        return False
    arm_indent = len(m.group("indent"))
    start = m.end()
    end = len(text)
    for line in re.finditer(r"^(?P<ind>[ \t]*)(?P<content>\S.*)$", text[start:], re.MULTILINE):
        if len(line.group("ind")) <= arm_indent:
            end = start + line.start()
            break
    block = text[start:end]
    key_pat = re.compile(rf"^(?P<indent>[ \t]+){re.escape(key)}:.*$", re.MULTILINE)
    km = key_pat.search(block)
    if km:
        new_block = block[: km.start()] + f"{km.group('indent')}{key}: {rounded}" + block[km.end():]
    else:
        indent = " " * (arm_indent + 2)
        new_block = block.rstrip("\n") + f"\n{indent}{key}: {rounded}\n"
    new_text = text[:start] + new_block + text[end:]
    try:
        parsed = yaml.safe_load(new_text)
        got = ((parsed or {}).get("arms") or {}).get(side, {}).get(key)
        ok = isinstance(got, (int, float)) and not isinstance(got, bool) and abs(float(got) - rounded) < 1e-9
    except yaml.YAMLError:
        ok = False
    if not ok:
        print(
            f"[capture-z-floor] WARNING: editing arms.{side}.{key} would corrupt {config_path}; "
            f"not writing. Set it by hand: {key}: {rounded}"
        )
        return False
    config_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    blocks = arm_blocks(cfg)
    side = args.arm

    if args.mock_robot:
        from core.piper.piper_interface import MockPiperRobot

        robot: object = MockPiperRobot(arm=side)
    else:
        from core.piper.piper_interface import PiperInterface

        robot = PiperInterface(arm=side)
    robot.connect()  # type: ignore[attr-defined]

    exit_code = 0
    try:
        z = round(float(np.asarray(robot.get_ee_pose(), dtype=float)[2]), 5)  # type: ignore[attr-defined]
        prev = blocks[side].get(KEY)
        prev_str = f" (was {float(prev):.5f})" if isinstance(prev, (int, float)) else ""
        print(f"[capture-z-floor] {side} arm: current EEF height = {z:.5f} m{prev_str}")
        print(f"[capture-z-floor] YAML (arms.{side}): {KEY}: {z}")
        print(
            "[capture-z-floor] this is the floor -- downward motion below it is blocked. "
            f"Make sure the {side} gripper was resting on the tabletop when captured."
        )
        if args.write:
            if _write_arm_scalar(Path(args.robot_config), side, KEY, z):
                print(f"[capture-z-floor] wrote arms.{side}.{KEY}: {z} into {args.robot_config}")
            else:
                exit_code = 1
    except Exception as exc:  # noqa: BLE001 - surface clearly
        print(f"[capture-z-floor] ERROR: {exc}")
        exit_code = 1
    finally:
        try:
            robot.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
