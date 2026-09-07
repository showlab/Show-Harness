#!/usr/bin/env python3
"""Measure ``tasks.FLANGE_TO_FINGERTIP_M`` for whatever gripper is currently spawned.

The planner in ``real2sim/robolab/tasks.py`` positions the FLANGE (``rl_tcp`` reports
``panda_hand``) and needs to know how far below it the fingers actually close. That number
is a property of the gripper geometry, and this repo's rule for it is that it is MEASURED,
never derived: the stock Panda's 0.1034 m was confirmed by a height sweep on a 58 mm cube,
and the Robotiq's datasheet figure (0.1628) was found to be wrong for RoboLab's flattened
USD by ~45 mm -- it aimed high and closed on thin air, silently, with every episode still
reporting a clean token sequence.

The sweep is behavioural because that is the only thing that cannot lie: at each candidate
flange height it descends, closes, lifts, and asks whether the OBJECT came up with the
hand. A grasp that merely reports a plausible finger width is not a grasp -- an empty
gripper closing on air and a gripper holding a cube differ in width, but a gripper pinching
the cube's top corner reads like a good grasp and drops it on the first carry token.

Usage (RoboLab's interpreter, from the repo root)::

    CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=YES \
    LD_LIBRARY_PATH=$ROBOLAB_ROOT/.deps/lib:$LD_LIBRARY_PATH \
    $ROBOLAB_ROOT/.venv/bin/python \
        scripts/trajectory/real2sim/robolab/calibrate_fingertip.py \
        --task RubiksCubeTask --device cuda:0

It prints a table of (flange height -> width, object lift) and the offset to write into
``tasks.FLANGE_TO_FINGERTIP_M``. It changes no files: the constant is edited by hand, so
the value in the source always has a measurement behind it.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.config import camera_contract, load_yaml  # noqa: E402

# A grasp counts as HOLDING when the object rises with the hand by at least this much over
# the lift. Well above settling noise, well below the commanded lift.
LIFT_HELD_M = 0.03


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", default="RubiksCubeTask")
    ap.add_argument("--sim", default="robolab")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--robot-config", default="configs/robot_robolab.yaml")
    ap.add_argument("--instruction-type", default="default")
    ap.add_argument("--camera-preset", default="WRIST_LEFT")
    ap.add_argument("--z-min", type=float, default=0.06,
                    help="lowest flange height above the object centroid to try (m)")
    ap.add_argument("--z-max", type=float, default=0.16)
    ap.add_argument("--z-step", type=float, default=0.005)
    ap.add_argument("--lift-tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--gui", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from core.sim.robolab_task import launch_isaac

    simulation_app = launch_isaac(headless=not args.gui, device=args.device)
    try:
        return _sweep(args)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


def _sweep(args: argparse.Namespace) -> int:
    from scripts.trajectory.real2sim.atomic_tokenizer import AtomicExec
    from scripts.trajectory.real2sim.backends import make_backend
    from scripts.trajectory.real2sim.robolab import tasks
    from scripts.trajectory.real2sim.robolab.tasks import UnsupportedTask

    robot_cfg = load_yaml(ROOT / args.robot_config)
    backend = make_backend(
        args.sim, task=args.task, device=args.device,
        instruction_type=args.instruction_type, camera_preset=args.camera_preset,
        # Layout randomisation OFF, unlike generation: a sweep compares heights against
        # ONE object pose. With it on, every reset re-samples the object's XY (and lands
        # it at a slightly different height), so consecutive rows differ by the layout as
        # much as by the height being tested, and the "held range" is noise.
        randomize_xy_m=None,
        **camera_contract(robot_cfg),
    )
    # The sweep drops the object on purpose at the heights that do not work, so RoboLab
    # must not end the episode underneath it.
    suspend = getattr(backend, "suspend_task_termination", None)
    if callable(suspend):
        suspend()

    backend.reset(args.seed)
    try:
        plan = tasks.resolve_plan(backend)
    except UnsupportedTask as exc:
        print(f"[calib] {args.task}: {exc}")
        return 2
    obj = plan["object"]
    print(f"[calib] task={args.task} object={obj} "
          f"({tasks.describe(backend, plan)})")
    print(f"[calib] current constant FLANGE_TO_FINGERTIP_M = "
          f"{tasks.FLANGE_TO_FINGERTIP_M:.4f} m")

    heights = np.arange(args.z_min, args.z_max + 1e-9, args.z_step)
    executor = AtomicExec(backend, step_m=0.02, max_cmd_m=0.02, max_ctrl_steps=24,
                          gripper_steps=tasks.GRIPPER_SETTLE_STEPS)
    rows = []
    for offset in heights:
        backend.reset(args.seed)
        centroid = backend.object_centroid(obj)
        obj_z0 = float(centroid[2])
        flange_z = obj_z0 + float(offset)

        # Approach from above, then straight down -- never sideways at grasp height, or the
        # hand sweeps the object off its spot before the sweep measures anything.
        executor.move_to(np.array([centroid[0], centroid[1], flange_z + 0.08]), budget=80)
        executor.move_to(np.array([centroid[0], centroid[1], flange_z]), budget=60)
        executor.quiesce()
        reached = float(backend.tcp_pos()[2])

        width = executor.grasp()
        for _ in range(args.lift_tokens):
            executor.move("MV_UP")
        executor.quiesce()
        obj_z1 = float(backend.object_centroid(obj)[2])
        lift = obj_z1 - obj_z0
        held = lift >= LIFT_HELD_M
        # The offset that MATTERS is the one the flange actually reached, not the one it
        # was asked for. Below a certain height the fingertip is on the table and the arm
        # simply stops: several commanded heights then collapse onto the same real one,
        # and reporting the command would put a "held at 50 mm" row in the table for a
        # grasp that happened at 98 mm.
        actual = reached - obj_z0
        short = abs(reached - flange_z) > 0.002
        rows.append((float(offset), actual, reached, width, lift, held))
        print(f"[calib] asked {offset*1000:6.1f} mm -> reached {actual*1000:6.1f} mm "
              f"(flange z {reached:.4f}{', BLOCKED' if short else ''}): "
              f"width {width*1000:6.2f} mm, object lifted {lift*1000:7.2f} mm  "
              f"{'HELD' if held else '--'}", flush=True)
        executor.release()

    backend.close()

    held = [r for r in rows if r[5]]
    print()
    if not held:
        print("[calib] NOTHING HELD at any height. The finger geometry cannot grasp this "
              "object at this orientation -- widen the sweep, or the tip is mounted wrong "
              "(check the placement constants in make_short_finger_asset.py).")
        return 2
    actuals = sorted(r[1] for r in held)
    lo, hi = actuals[0], actuals[-1]
    mid = 0.5 * (lo + hi)
    print(f"[calib] held at REACHED offsets {lo*1000:.1f} .. {hi*1000:.1f} mm "
          f"({len(held)}/{len(rows)} heights)")
    print(f"[calib] widths while holding: {[round(r[3]*1000, 1) for r in held]} mm")
    print()
    print(f"[calib] FLANGE_TO_FINGERTIP_M = {mid:.4f}   <- write this into "
          f"scripts/trajectory/real2sim/robolab/tasks.py")
    if hi >= max(r[1] for r in rows) - 1e-9:
        print("[calib] WARNING: the held range runs to the TOP of the sweep, so this "
              "midpoint is the midpoint of the sweep, not of the range. Re-run with a "
              "higher --z-max -- this is exactly how the stock finger's first estimate "
              "(0.1113) came out wrong.")
    if lo <= min(r[1] for r in rows) + 1e-9:
        print("[calib] NOTE: the held range runs to the BOTTOM of the reachable sweep. "
              "That bottom is usually the fingertip touching the table rather than a "
              "sweep bound, in which case it is a real limit and the midpoint stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
