#!/usr/bin/env python3
"""Capture the Franka's CURRENT joint configuration as a NAMED start pose.

The pose analog of ``capture_z_floor.py``: hand-guide (or jog) the arm to the
configuration you want, then read + persist it under ``poses.<name>`` in
configs/robot_franka.yaml. ``--activate`` additionally makes it the DEFAULT
initial position: ``begin_pose: <name>`` plus ``move_to_begin_on_init: true``,
so every ``scripts/run_real.py`` rollout homes to it before starting (via the session's
own ``move_to_joint_positions`` -- the lab-shared ``franka_server/go_home_client.py``
reset is NOT involved and is never modified).

    # just print the current joints (does not touch the config)
    python scripts/franka/capture_pose.py

    # save the current configuration as poses.chess_ready
    python scripts/franka/capture_pose.py --name chess_ready --write

    # ... and make it the default initial position for every rollout
    python scripts/franka/capture_pose.py --name chess_ready --write --activate

Read-only by design: ``get_joint_positions()`` never moves the arm. Re-capturing
a name overwrites it in place. ``poses.default`` is REFUSED as a target: by
convention it IS go_home_client.py's HOME (home = rest), so the base pose can
never drift from the lab's shared reset.
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

POSES_KEY = "poses"
ACTIVE_KEY = "begin_pose"
INIT_KEY = "move_to_begin_on_init"
# By convention poses.default IS franka_server/go_home_client.py's HOME (the lab's
# shared reset, which doubles as home and rest). Never captured over.
PROTECTED = "default"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture the current joints as a named Franka start pose.")
    p.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_franka.yaml"))
    p.add_argument(
        "--name",
        default=None,
        help="Which named pose (poses.<name>) to capture. Required with --write. "
        f"{PROTECTED!r} is refused (pinned to go_home_client.py's HOME).",
    )
    p.add_argument("--nuc-ip", default=None, help="Override the Franka NUC IP.")
    p.add_argument("--nuc-port", type=int, default=None, help="Override the NUC port.")
    p.add_argument("--mock-robot", action="store_true", help="Use a simulated robot (no NUC).")
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the captured joints into poses.<name> (overwrites in place).",
    )
    p.add_argument(
        "--activate",
        action="store_true",
        help="With --write: set begin_pose to this name AND move_to_begin_on_init: true, "
        "so every rollout homes to it before starting.",
    )
    return p.parse_args()


def _fmt_joints(joints: list[float]) -> str:
    return "[" + ", ".join(f"{j:.5f}" for j in joints) + "]"


def _block_span(text: str) -> tuple[int, int] | None:
    """``(start, end)`` of the top-level ``poses:`` block body, or None."""
    m = re.search(rf"^{POSES_KEY}:[ \t]*$", text, re.MULTILINE)
    if not m:
        return None
    start = m.end()
    end = len(text)
    for line in re.finditer(r"^(?P<ind>[ \t]*)\S.*$", text[start:], re.MULTILINE):
        if len(line.group("ind")) == 0:
            end = start + line.start()
            break
    return start, end


def _edit_named_pose(text: str, name: str, joints_str: str) -> str:
    """Set ``poses.<name>`` to ``joints_str``, preserving the rest of the file."""
    span = _block_span(text)
    if span is None:
        return text.rstrip("\n") + f"\n\n{POSES_KEY}:\n  {name}: {joints_str}\n"
    start, end = span
    block = text[start:end]
    km = re.search(rf"^(?P<indent>[ \t]+){re.escape(name)}:.*$", block, re.MULTILINE)
    if km:
        new_block = block[: km.start()] + f"{km.group('indent')}{name}: {joints_str}" + block[km.end():]
    else:
        new_block = block.rstrip("\n") + f"\n  {name}: {joints_str}\n"
    return text[:start] + new_block + text[end:]


def _edit_top_scalar(text: str, key: str, value: str, anchor_key: str) -> str:
    """Set top-level ``key: value`` (replace in place; insert before ``anchor_key``
    or append at EOF when the key is absent)."""
    line = f"{key}: {value}"
    m = re.search(rf"^{re.escape(key)}:.*$", text, re.MULTILINE)
    if m:
        return text[: m.start()] + line + text[m.end():]
    am = re.search(rf"^{re.escape(anchor_key)}:[ \t]*$", text, re.MULTILINE)
    if am:
        return text[: am.start()] + line + "\n" + text[am.start():]
    return text.rstrip("\n") + f"\n{line}\n"


def _write_config(config_path: Path, name: str, joints: list[float], activate: bool) -> bool:
    """Apply the edits, but only if the result re-parses with the expected values."""
    text = config_path.read_text(encoding="utf-8")
    joints_str = _fmt_joints(joints)
    new_text = _edit_named_pose(text, name, joints_str)
    if activate:
        new_text = _edit_top_scalar(new_text, ACTIVE_KEY, name, POSES_KEY)
        new_text = _edit_top_scalar(new_text, INIT_KEY, "true", POSES_KEY)
    try:
        parsed = yaml.safe_load(new_text) or {}
        got = (parsed.get(POSES_KEY) or {}).get(name)
        ok = (
            isinstance(got, list)
            and len(got) == len(joints)
            and all(abs(float(a) - round(b, 5)) < 1e-9 for a, b in zip(got, joints))
        )
        if activate:
            ok = ok and str(parsed.get(ACTIVE_KEY)) == name and parsed.get(INIT_KEY) is True
    except (yaml.YAMLError, TypeError, ValueError):
        ok = False
    if not ok:
        print(
            f"[capture-pose] WARNING: editing {POSES_KEY}.{name} would corrupt "
            f"{config_path}; not writing. Set it by hand: {name}: {joints_str}"
        )
        return False
    config_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    rb = cfg.get("robot", {}) or {}
    poses = cfg.get(POSES_KEY) or {}

    if args.write and not args.name:
        print("[capture-pose] --write needs --name <pose>.")
        return 1
    if args.name == PROTECTED:
        print(
            f"[capture-pose] refusing to overwrite poses.{PROTECTED}: it is pinned to "
            "franka_server/go_home_client.py's HOME (the lab-shared reset). "
            "Pick another name."
        )
        return 1

    if args.mock_robot:
        from core.franka.franka_interface import MockRobot

        robot: object = MockRobot(ip="mock", port=0)
    else:
        from core.franka.franka_interface import FrankaInterface

        nuc_ip = args.nuc_ip or str(rb.get("nuc_ip", ""))
        nuc_port = int(args.nuc_port if args.nuc_port is not None else rb.get("nuc_port", 4242))
        print(f"[capture-pose] connecting to Franka NUC at {nuc_ip}:{nuc_port} (read-only) ...")
        robot = FrankaInterface(ip=nuc_ip, port=nuc_port)

    exit_code = 0
    try:
        joints = [round(float(j), 5) for j in np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)]  # type: ignore[attr-defined]
        if len(joints) != 7:
            raise RuntimeError(f"expected 7 joints, got {len(joints)}: {joints}")
        name = args.name or "(unnamed)"
        prev = poses.get(args.name) if args.name else None
        prev_str = " (overwrites the existing pose)" if isinstance(prev, list) else " (new pose)"
        print(f"[capture-pose] current joints -> {POSES_KEY}.{name}{prev_str if args.name else ''}")
        print(f"[capture-pose] YAML: {name}: {_fmt_joints(joints)}")
        if args.write:
            if _write_config(Path(args.robot_config), args.name, joints, args.activate):
                print(f"[capture-pose] wrote {POSES_KEY}.{args.name} into {args.robot_config}")
                if args.activate:
                    print(
                        f"[capture-pose] activated: {ACTIVE_KEY}: {args.name} + {INIT_KEY}: true "
                        "-- every rollout now homes to this pose before starting."
                    )
                else:
                    print(
                        f"[capture-pose] use it per run with  python scripts/run_real.py --begin-pose "
                        f"{args.name}  (or re-run with --activate to make it the default)."
                    )
            else:
                exit_code = 1
        elif args.activate:
            print("[capture-pose] --activate does nothing without --write.")
    except Exception as exc:  # noqa: BLE001 - surface clearly
        print(f"[capture-pose] ERROR: {exc}")
        exit_code = 1
    finally:
        try:
            robot.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
