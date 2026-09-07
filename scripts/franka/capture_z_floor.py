#!/usr/bin/env python3
"""Capture the Franka Z safety floor (minimum EEF height) as a NAMED setting.

The z-floor is the base-frame height of the end effector when the gripper just
touches the work surface; the controller blocks any downward motion below it. One
config carries a calibrated floor per table/task setting under ``z_floors:`` in
configs/robot_franka.yaml (the Franka analog of robot_piper.yaml's per-pose
z_floor_m variants); ``z_floor_name`` (or ``scripts/run_real.py --z-floor-name``) picks the
active one.

    # rest the (closed) gripper on the surface first, then read + persist the height
    python scripts/franka/capture_z_floor.py --name drawer --write

    # also make it the active floor (sets z_floor_name: drawer)
    python scripts/franka/capture_z_floor.py --name drawer --write --activate

    # just print the current height (does not touch the config)
    python scripts/franka/capture_z_floor.py

Read-only by design: ``get_ee_pose()`` never moves the arm and no impedance
controller is started, so this is safe to run while the arm is hand-guided or
parked. Re-capturing a name overwrites it in place, so calibration never means
editing the config by hand.
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

FLOORS_KEY = "z_floors"
ACTIVE_KEY = "z_floor_name"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture a named Franka Z safety floor.")
    p.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_franka.yaml"))
    p.add_argument(
        "--name",
        default=None,
        help="Which named floor (z_floors.<name>) to capture. Default: the config's "
        "z_floor_name, else 'default'.",
    )
    p.add_argument("--nuc-ip", default=None, help="Override the Franka NUC IP.")
    p.add_argument("--nuc-port", type=int, default=None, help="Override the NUC port.")
    p.add_argument("--mock-robot", action="store_true", help="Use a simulated robot (no NUC).")
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the captured height into z_floors.<name> (overwrites in place).",
    )
    p.add_argument(
        "--activate",
        action="store_true",
        help="With --write: also set z_floor_name to this name, making it the active floor.",
    )
    return p.parse_args()


def _block_span(text: str) -> tuple[int, int] | None:
    """``(start, end)`` of the top-level ``z_floors:`` block body, or None."""
    m = re.search(rf"^{FLOORS_KEY}:[ \t]*$", text, re.MULTILINE)
    if not m:
        return None
    start = m.end()
    end = len(text)
    for line in re.finditer(r"^(?P<ind>[ \t]*)\S.*$", text[start:], re.MULTILINE):
        if len(line.group("ind")) == 0:
            end = start + line.start()
            break
    return start, end


def _edit_named_floor(text: str, name: str, value: float) -> str:
    """Set ``z_floors.<name>`` to ``value``, preserving the rest of the file.

    Replaces the entry in place, appends it to the block, or appends a whole
    ``z_floors:`` block at EOF when the config has none yet."""
    span = _block_span(text)
    if span is None:
        return text.rstrip("\n") + f"\n\n{FLOORS_KEY}:\n  {name}: {value}\n"
    start, end = span
    block = text[start:end]
    km = re.search(rf"^(?P<indent>[ \t]+){re.escape(name)}:.*$", block, re.MULTILINE)
    if km:
        new_block = block[: km.start()] + f"{km.group('indent')}{name}: {value}" + block[km.end():]
    else:
        new_block = block.rstrip("\n") + f"\n  {name}: {value}\n"
    return text[:start] + new_block + text[end:]


def _edit_active_name(text: str, name: str) -> str:
    """Set the top-level ``z_floor_name`` to ``name`` (insert before z_floors / at EOF)."""
    line = f"{ACTIVE_KEY}: {name}"
    m = re.search(rf"^{ACTIVE_KEY}:.*$", text, re.MULTILINE)
    if m:
        return text[: m.start()] + line + text[m.end():]
    fm = re.search(rf"^{FLOORS_KEY}:[ \t]*$", text, re.MULTILINE)
    if fm:
        return text[: fm.start()] + line + "\n" + text[fm.start():]
    return text.rstrip("\n") + f"\n{line}\n"


def _write_config(config_path: Path, name: str, value: float, activate: bool) -> bool:
    """Apply the edits, but only if the result re-parses with the expected values."""
    text = config_path.read_text(encoding="utf-8")
    new_text = _edit_named_floor(text, name, value)
    if activate:
        new_text = _edit_active_name(new_text, name)
    try:
        parsed = yaml.safe_load(new_text) or {}
        got = (parsed.get(FLOORS_KEY) or {}).get(name)
        ok = (
            isinstance(got, (int, float))
            and not isinstance(got, bool)
            and abs(float(got) - value) < 1e-9
        )
        if activate:
            ok = ok and str(parsed.get(ACTIVE_KEY)) == name
    except yaml.YAMLError:
        ok = False
    if not ok:
        print(
            f"[capture-z-floor] WARNING: editing {FLOORS_KEY}.{name} would corrupt "
            f"{config_path}; not writing. Set it by hand: {name}: {value}"
        )
        return False
    config_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    rb = cfg.get("robot", {}) or {}
    name = args.name or str(cfg.get(ACTIVE_KEY) or "default")
    floors = cfg.get(FLOORS_KEY) or {}

    if args.mock_robot:
        from core.franka.franka_interface import MockRobot

        robot: object = MockRobot(ip="mock", port=0)
    else:
        from core.franka.franka_interface import FrankaInterface

        nuc_ip = args.nuc_ip or str(rb.get("nuc_ip", ""))
        nuc_port = int(args.nuc_port if args.nuc_port is not None else rb.get("nuc_port", 4242))
        print(f"[capture-z-floor] connecting to Franka NUC at {nuc_ip}:{nuc_port} (read-only) ...")
        robot = FrankaInterface(ip=nuc_ip, port=nuc_port)

    exit_code = 0
    try:
        z = round(float(np.asarray(robot.get_ee_pose(), dtype=float)[2]), 5)  # type: ignore[attr-defined]
        prev = floors.get(name)
        prev_str = f" (was {float(prev):.5f})" if isinstance(prev, (int, float)) else " (new setting)"
        print(f"[capture-z-floor] current EEF height = {z:.5f} m -> {FLOORS_KEY}.{name}{prev_str}")
        print(f"[capture-z-floor] YAML: {FLOORS_KEY}: {{{name}: {z}}}")
        print(
            "[capture-z-floor] this is the floor -- downward motion below it is blocked. "
            "Make sure the gripper was resting on the work surface when captured."
        )
        if args.write:
            if _write_config(Path(args.robot_config), name, z, args.activate):
                active = " and set it active" if args.activate else ""
                print(
                    f"[capture-z-floor] wrote {FLOORS_KEY}.{name}: {z} into "
                    f"{args.robot_config}{active}"
                )
                if not args.activate and str(cfg.get(ACTIVE_KEY)) != name:
                    print(
                        f"[capture-z-floor] note: the active floor is still "
                        f"{cfg.get(ACTIVE_KEY)!r}; use it per run with "
                        f"--z-floor-name {name}, or re-run with --activate."
                    )
            else:
                exit_code = 1
        elif args.activate:
            print("[capture-z-floor] --activate does nothing without --write.")
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
