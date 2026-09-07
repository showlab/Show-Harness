#!/usr/bin/env python3
"""Record CONTINUOUS scripted demos on RoboLab and export their TCP tracks -- Scheme D step 1.

RoboLab sibling of ``real2sim/maniskill/record_demos.py``. It only records: a smooth
privileged waypoint servo drives the task -- continuous, multi-axis motion, the same
character a motion planner or a human demo has -- and each successful episode's TCP path +
gripper events go to ``<out>/tracks/<Task>/track_epNNN.json``. ``follow_tokenize.py`` then
RE-EXECUTES those paths as single-axis 2 cm tokens, so the recorded frames come from the
tokenised rollout and sit on the deployment lattice.

Almost nothing here is RoboLab-specific: :class:`~...atomic_tokenizer.DemoRecorder` (the
servo + capture loop) lives in the sim-agnostic core, because it only calls ``tcp_pos`` /
``apply_delta`` / ``success`` on the backend. What this file adds is the scripted demo --
which privileged poses to servo to, in what order -- and RoboLab lets even that be generic:
the plan comes from the task's own ``subtasks`` declaration via ``robolab/tasks.py``, so
any single-object pick-and-place task among RoboLab-120 records without new code.

The recorded track does NOT store a layout: unlike the ManiSkill rigs (where this repo
samples object poses and replays them), a RoboLab scene's layout is produced by the env's
own reset events, so the follower reproduces it by resetting with the same seed.

Usage:
    export OMNI_KIT_ACCEPT_EULA=YES
    export LD_LIBRARY_PATH=$ROBOLAB_ROOT/.deps/lib:$LD_LIBRARY_PATH
    $ROBOLAB_ROOT/.venv/bin/python \
        scripts/trajectory/real2sim/robolab/record_demos.py \
        --task BananaInBowlTask --episodes 20 --out <stage>
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="RoboLab Task class name.")
    ap.add_argument("--sim", default="robolab", help="discretiser backend (see backends/)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", required=True, help="tracks land in <out>/tracks/<task>/")
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--instruction-type", default="default")
    ap.add_argument("--camera-preset", default="WRIST_LEFT")
    ap.add_argument("--gui", action="store_true", help="run with the Isaac Sim viewport")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Isaac Sim first: nothing may import isaaclab/robolab before the Kit app is up.
    from core.sim.robolab_task import launch_isaac

    simulation_app = launch_isaac(headless=not args.gui, device=args.device)
    try:
        return _record(args)
    except Exception:
        # Isaac Sim's SimulationApp.close() can terminate the process outright, which
        # swallows BOTH the traceback and the exit code -- a failure then looks like a
        # clean exit 0 that silently produced nothing. Print before closing.
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


def _record(args: argparse.Namespace) -> int:
    from scripts.trajectory.real2sim.atomic_tokenizer import DemoRecorder, gripper_events
    from scripts.trajectory.real2sim.backends import make_backend
    from scripts.trajectory.real2sim.robolab import tasks
    from scripts.trajectory.real2sim.robolab.tasks import GRASPED_WIDTH_M, UnsupportedTask

    def scripted_demo(rec: DemoRecorder, plan: dict, rng: np.random.Generator) -> bool:
        """Continuous privileged demo: grasp the object, carry it, seat it on the target.

        Steering is by the CARRIED OBJECT, not the flange: after a grasp the object can sit
        a centimetre off-centre in the fingers, and RoboLab's containment predicates check
        where the OBJECT is. Heights all come through ``tasks``, which adds the Robotiq
        flange->fingertip offset (see its docstring).
        """
        backend = rec.backend
        obj, target = plan["object"], plan["target"]
        hover = 0.05 + float(rng.uniform(0, 0.02))

        grasp = tasks.grasp_tcp(backend, obj)
        rec.servo_to(grasp + [0, 0, hover])
        rec.servo_to(grasp)
        # The Robotiq is a binary joint command and takes far longer than the default 10
        # steps to travel: sampled too early the width still reads near-open, so an EMPTY
        # grasp passes the check and the failure only shows up later as a dropped object.
        grasp_flange_z = float(rec._tcp()[2])
        rec.set_gripper(close=True, steps=tasks.GRIPPER_SETTLE_STEPS)
        if backend.gripper_width() < GRASPED_WIDTH_M:
            return False
        # ONE carry height for the whole transport, computed from the scene and the height
        # the grasp happened at. Recomputing it per leg used to ratchet the arm upwards
        # (see tasks.carry_z), so the lift was still running when the horizontal leg began
        # and the two blended into a diagonal.
        travel_z = tasks.carry_z(backend, obj, target, grasp_flange_z)
        rec.servo_to(np.array([grasp[0], grasp[1], travel_z]))

        place = tasks.place_tcp(backend, obj, target, plan["target_kind"])
        rec.servo_to(np.array([place[0], place[1], travel_z]))
        rec.servo_to(place)
        rec.set_gripper(close=False)
        rec.servo_to(rec._tcp() + [0, 0, 0.08])
        return rec.hold_until_success(20)

    def _layout_of(backend, plan: dict) -> dict:
        """Object/target centroids at reset -- the layout this demo assumes."""
        out: dict = {}
        for key in ("object", "target"):
            name = plan.get(key)
            if name:
                try:
                    out[name] = [round(float(v), 5)
                                 for v in backend.object_centroid(name)]
                except Exception:  # noqa: BLE001 -- prims without queryable geometry
                    pass
        return out

    backend = make_backend(
        args.sim,
        task=args.task,
        device=args.device,
        instruction_type=args.instruction_type,
        camera_preset=args.camera_preset,
    )
    tracks_dir = Path(args.out) / "tracks" / args.task
    tracks_dir.mkdir(parents=True, exist_ok=True)

    kept, tried = 0, 0
    layouts: list[dict] = []
    while kept < args.episodes and tried < args.episodes * 3:
        seed = args.seed0 + tried
        tried += 1
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        backend.reset(seed)

        try:
            plan = tasks.resolve_plan(backend)
        except UnsupportedTask as exc:
            print(f"[record:{args.task}] {exc}")
            return 2
        if tried == 1:
            print(f"[record:{args.task}] plan: {tasks.describe(backend, plan)}")

        layout = _layout_of(backend, plan)
        layouts.append(layout)
        rec = DemoRecorder(backend)
        for _ in range(8):
            rec.step(np.zeros(3))  # settle, mirrors deployment num_steps_wait
        # Only the path AFTER settling is the demo; drop the settle samples.
        rec.tcp.clear()
        rec.grip_cmds.clear()
        rec.capture()

        if not scripted_demo(rec, plan, rng):
            print(f"[record:{args.task}] seed={seed} demo failed; skip", flush=True)
            continue

        (tracks_dir / f"track_ep{kept:03d}.json").write_text(json.dumps({
            "seed": seed,
            "sim": args.sim,
            "env_id": backend.env_id,
            "task": backend.task_description,
            "task_key": args.task,
            "plan": plan,
            "camera_preset": args.camera_preset,
            "instruction_type": args.instruction_type,
            "tcp": [[round(float(v), 5) for v in q] for q in rec.tcp],
            "grip_cmds": rec.grip_cmds,
            "events": gripper_events(rec.grip_cmds),
            # The LAYOUT this demo was recorded against. A RoboLab scene's object
            # placement comes from the env's own reset events, so the follower reproduces
            # it by resetting with the same seed -- this is stored so follow_tokenize can
            # CHECK that it did, rather than trust it. If the reproduction ever breaks
            # (a RoboLab change to how the events draw, a different task_dirs, an
            # embodiment whose reset consumes a different number of random draws), the
            # follower would chase waypoints aimed at where the objects USED to be, and
            # the failure would look like a clumsy demo rather than a broken assumption.
            "layout": layout,
        }))
        print(f"[record:{args.task}] seed={seed} demo={len(rec.tcp)} ctrl-steps -> "
              f"track_ep{kept:03d}.json", flush=True)
        kept += 1

    backend.close()
    print(f"[record:{args.task}] kept {kept}/{tried} tracks -> {tracks_dir}")

    # Layout diversity is the whole reason randomisation exists, and its failure mode is
    # SILENT: every episode still records, succeeds and looks healthy, it is just the same
    # episode N times. That is exactly what happened when the randomisation event was
    # merged alongside RoboLab's default reset instead of overriding it -- the default ran
    # second and undid it. Check the thing we actually want, not that the code ran.
    unique = {tuple(round(v, 4) for xyz in lay.values() for v in xyz) for lay in layouts}
    if layouts and len(unique) == 1:
        print(f"[record:{args.task}] WARNING: all {len(layouts)} episodes share ONE layout. "
              "If randomisation was requested it is NOT taking effect -- the dataset has "
              "no layout diversity.", flush=True)
    else:
        print(f"[record:{args.task}] layouts: {len(unique)} distinct across {len(layouts)} "
              "episodes", flush=True)
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
