#!/usr/bin/env python3
"""Record CONTINUOUS scripted demos and export their TCP tracks -- input stage for Scheme D.

This only records: it drives the task with a smooth privileged waypoint servo -- continuous,
multi-axis motion, the same character as a motion-planning or human demo -- and writes each
successful episode's TCP path + gripper events to ``<out>/tracks/<task>/track_epNNN.json``.
``follow_tokenize.py`` then reads those tracks and RE-EXECUTES them as single-axis 2 cm
tokens, so the recorded frames come from the tokenised rollout and sit on the deployment
lattice.

The scripted servo replaces the official mplib motion planner, which segfaults when
constructing ``PandaArmMotionPlanningSolver`` in this conda env (mplib 0.1.1). The data
character is equivalent: continuous, multi-axis, smooth.

NOTE: this does NOT tokenise or write MVTOKEN rollouts. The earlier offline-decomposition
variants (accumulator / waypoint-Manhattan, and the RL-h5 path) were removed: their frames
came from the continuous demo, so a token labelled MV_LEFT often coincided with a diagonal
frame-to-frame move (measured: ~65-72% off-axis, ~40-65% of steps >1 cm off-axis). Only the
closed-loop methods (Scheme A oracle, Scheme D follower) produce strictly single-axis frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.atomic_tokenizer import (  # noqa: E402
    DemoRecorder,
    gripper_events,
)
from scripts.trajectory.real2sim.backends import make_backend  # noqa: E402
from scripts.trajectory.real2sim.maniskill import tasks  # noqa: E402

# -------------------------------------------------------------- scripted demos
def _demo_pickcube(rec: DemoRecorder, env, hover: float) -> None:
    cube, goal = tasks.actor_pos(env, "cube"), tasks.actor_pos(env, "goal_site")
    rec.servo_to(cube + [0, 0, hover])
    rec.servo_to(cube + [0, 0, 0.002])
    rec.set_gripper(close=True)
    rec.servo_to(goal)
    rec.hold_until_success(14)


def _demo_stackcube(rec: DemoRecorder, env, hover: float) -> None:
    a, b = tasks.actor_pos(env, "cubeA"), tasks.actor_pos(env, "cubeB")
    rec.servo_to(a + [0, 0, hover])
    rec.servo_to(a + [0, 0, 0.002])
    rec.set_gripper(close=True)
    rec.servo_to(a + [0, 0, 0.10])
    rec.servo_to(np.array([b[0], b[1], a[2] + 0.10]))
    rec.servo_to(b + [0, 0, 0.04])
    rec.set_gripper(close=False)
    rec.servo_to(rec._tcp() + [0, 0, 0.08])
    rec.hold_until_success(10)


def _demo_pick_and_place(rec: DemoRecorder, env, hover: float, task: str) -> None:
    """Shared BlockPAP / BlockStack demo: grasp the carried block, seat it on the target.

    Steering is by the CARRIED object, not the TCP: the block can sit a few mm off-centre
    in the fingers, and both rigs' success checks are tight enough to care.
    """
    spec = tasks.task_spec(task)
    carried, target_name = spec["carried"], spec["target"]
    obj = tasks.actor_pos(env, carried)
    tgt = tasks.actor_pos(env, target_name)

    rec.servo_to(obj + [0, 0, hover])
    rec.servo_to(obj + [0, 0, 0.002])          # TCP at the block centre
    rec.set_gripper(close=True)
    rec.servo_to(obj + [0, 0, 0.12])           # lift clear of both objects

    off = tasks.actor_pos(env, carried)[:2] - rec._tcp()[:2]
    tx, ty = tgt[0] - off[0], tgt[1] - off[1]
    rec.servo_to(np.array([tx, ty, tgt[2] + 0.12]))
    drop_z = spec["drop_z"](env, tgt)
    tcp_above_block = rec._tcp()[2] - tasks.actor_pos(env, carried)[2]
    rec.servo_to(np.array([tx, ty, drop_z + tcp_above_block]))
    rec.set_gripper(close=False)
    rec.servo_to(rec._tcp() + [0, 0, 0.08])
    rec.hold_until_success(14)


DEMOS = {
    "pickcube": _demo_pickcube,
    "stackcube": _demo_stackcube,
    "blockpap": lambda rec, env, hover: _demo_pick_and_place(rec, env, hover, "blockpap"),
    "blockstack": lambda rec, env, hover: _demo_pick_and_place(rec, env, hover, "blockstack"),
}


def run_scripted_demo(task: str, rec: DemoRecorder, rng: np.random.Generator) -> bool:
    """Continuous privileged demo; returns env success."""
    hover = 0.05 + rng.uniform(0, 0.02)
    DEMOS[task](rec, rec.backend.env, hover)
    return rec.backend.success()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(DEMOS), default="pickcube")
    ap.add_argument("--sim", default="maniskill", help="discretiser backend (see backends/)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", required=True,
                    help="dataset root; tracks -> <out>/tracks/<task>/")
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--step-m", type=float, default=0.02,
                    help="atomic step the layout is snapped to where the task needs it "
                         "(see tasks.randomize_layout)")
    ap.add_argument("--no-snap", action="store_true",
                    help="sample the layout freely instead of snapping it to the atomic "
                         "lattice (blockstack: lower follower success on its 1 cm tolerance)")
    args = ap.parse_args()

    spec = tasks.task_spec(args.task)
    # Whether this task's layout is ours to sample (see tasks.randomize_layout). Every
    # registered task is an RLinf rig now, but keying off the layout spec keeps the check
    # about what it actually gates.
    samples_layout = bool(spec.get("layout"))
    snap = bool(spec.get("snap_layout", False)) and not args.no_snap
    backend = make_backend(args.sim, **tasks.backend_kwargs(args.task))

    tracks_dir = Path(args.out) / "tracks" / args.task
    tracks_dir.mkdir(parents=True, exist_ok=True)

    kept, tried = 0, 0
    while kept < args.episodes and tried < args.episodes * 3:
        seed = args.seed0 + tried
        tried += 1
        # The env's own layout sampling draws from the GLOBAL np.random; seed it for
        # reproducibility even when it is overridden below.
        np.random.seed(seed)
        backend.reset(seed)
        rec = DemoRecorder(backend)
        rng = np.random.default_rng(seed)
        for _ in range(8):
            rec.step(np.zeros(3))  # settle, mirrors deployment num_steps_wait
        layout: dict = {}
        if samples_layout:
            # Anchor the lattice at the SETTLED gripper XY, then let the objects come to
            # rest before the demo reads their poses.
            layout = tasks.randomize_layout(backend.env, args.task, rng, rec._tcp()[:2],
                                            args.step_m, snap=snap)
            for _ in range(6):
                rec.step(np.zeros(3))
        rec.tcp.clear()
        rec.grip_cmds.clear()
        rec.capture()
        if not run_scripted_demo(args.task, rec, rng):
            print(f"[record:{args.task}] seed={seed} demo failed; skip", flush=True)
            continue
        (tracks_dir / f"track_ep{kept:03d}.json").write_text(json.dumps({
            "seed": seed,
            "sim": args.sim,
            "env_id": spec["env_id"],
            "task": tasks.task_text(args.task),
            "task_key": args.task,
            "tcp": [[round(float(v), 5) for v in q] for q in rec.tcp],
            "grip_cmds": rec.grip_cmds,
            "events": gripper_events(rec.grip_cmds),
            # Absolute object poses, so the follower reproduces the exact same scene
            # without having to re-derive the sampling (see tasks.apply_layout).
            **({"layout": layout} if layout else {}),
            "robot_uids": tasks.task_robot_uids(args.task),
        }))
        print(f"[record:{args.task}] seed={seed} demo={len(rec.tcp)} ctrl-steps -> "
              f"track_ep{kept:03d}.json", flush=True)
        kept += 1

    backend.close()
    print(f"[record:{args.task}] kept {kept}/{tried} tracks -> {tracks_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
