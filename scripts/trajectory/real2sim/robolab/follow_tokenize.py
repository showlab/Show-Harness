#!/usr/bin/env python3
"""Scheme D on RoboLab -- closed-loop follower: re-execute a demo's path with 2 cm tokens.

RoboLab sibling of ``real2sim/maniskill/follow_tokenize.py``. Reads the TCP tracks exported
by ``record_demos.py``, resets the same scene to the same seed, reduces each track to RDP
waypoints + gripper events, then chases each waypoint with dominant-axis 2 cm tokens. Every
recorded frame is a state the DISCRETE controller actually reached, so every frame-to-frame
move is strictly single-axis and sits exactly on the deployment lattice. Episodes are kept
only if RoboLab's own success predicate passes at the end -- the decomposition is thereby
VALIDATED, not assumed.

Why this is worth having on top of the oracle (Scheme A): the oracle's paths are Manhattan by
construction (align XY, then descend), while a follower inherits the SHAPE of a continuous
demo -- diagonal approaches, curved carries, varied ordering -- and only the execution is
quantised. Training on both gives the policy token sequences that do not all look like the
same state machine.

Scene reproduction differs from ManiSkill: there the recorder sampled object poses and
stored them in the track for ``apply_layout`` to replay. A RoboLab scene's layout comes
from the env's own reset events, so resetting with the recorded seed is what reproduces it
-- ``RobolabBackend.reset`` seeds the global RNG the events draw from.

Output: ``<out>/<Task>_follow/rollout_NNN`` in the teleop/MVTOKEN rollout format.

Usage:
    export OMNI_KIT_ACCEPT_EULA=YES
    export LD_LIBRARY_PATH=$ROBOLAB_ROOT/.deps/lib:$LD_LIBRARY_PATH
    $ROBOLAB_ROOT/.venv/bin/python \
        scripts/trajectory/real2sim/robolab/follow_tokenize.py \
        --tracks <stage>/tracks/BananaInBowlTask --out <stage> --agentview-square 256
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.config import camera_contract, load_yaml  # noqa: E402

# Control steps to hold still, after the last token, waiting for the success predicate.
# RoboLab's is static -- object_in_container with require_contact_with and
# require_gripper_detached -- so it cannot fire until the RELEASED object has come to
# rest in the container. Measured on RubiksCubeTask: ~40 steps. The old value of 20 was
# under it, and the shortfall did not surface as a failure; it surfaced as a corrective
# re-chase that happened to burn enough time for the predicate to fire (see follow_track).
SETTLE_STEPS = 90

# How far to lift straight up after opening the fingers. RoboLab's success predicate
# requires the gripper to be OUT OF CONTACT with the placed object, so this is part of
# completing the task, not a flourish. Three tokens clears a 58 mm cube plus the fingers'
# own travel with margin, and being a whole number of tokens keeps every stored frame on
# the 2 cm deployment lattice.
RELEASE_RETREAT_M = 0.06

# How far an object may sit from where the demo saw it before the track is unusable.
# The follower chases waypoints recorded against a specific LAYOUT; if the reset did not
# reproduce it, every waypoint points at where the object used to be. That failure is
# nasty precisely because it still produces a full, plausible-looking episode -- the arm
# moves smoothly to a spot the object is not, closes on nothing, and the run is written
# off as a clumsy demo. 2 cm is one token: beyond that the discrepancy is real, not noise.
LAYOUT_TOL_M = 0.02



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracks", required=True,
                    help="tracks/<Task> dir written by record_demos.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", default="", help="backend; default: the one the track used")
    ap.add_argument("--step-m", type=float, default=0.02,
                    help="physical metres per atomic token (the MVTOKEN contract is 0.02)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--robot-config", default="configs/robot_robolab.yaml",
                    help="THE source of truth for the camera transform contract "
                         "(which cameras, rotation, flip, crop). The deployment runner "
                         "reads the same file, which is what keeps stored frames "
                         "byte-identical to the ones the policy is served at inference.")
    ap.add_argument("--wrist-flip", default=None,
                    help="override the config's wrist_flip (none | vertical | horizontal "
                         "| both). Leave unset -- the config is the contract.")
    ap.add_argument("--crop-aspect", type=float, default=None,
                    help="override the config's *_crop_aspect (0 disables cropping). "
                         "Leave unset -- the config is the contract.")
    ap.add_argument("--agentview-square", type=int, default=0,
                    help="store agentview as an N x N letterbox (resize_with_pad); 256 "
                         "matches the real-robot datasets and the deployment runner.")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--gui", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    from core.sim.robolab_task import launch_isaac

    simulation_app = launch_isaac(headless=not args.gui, device=args.device)
    try:
        return _follow_all(args)
    except Exception:
        # Isaac Sim's SimulationApp.close() can terminate the process outright, which
        # swallows BOTH the traceback and the exit code -- a failure then looks like a
        # clean exit 0 that silently produced nothing. Print before closing.
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()



def write_preview(ds_dir: Path) -> None:
    """Render the preview video for a freshly generated dataset. ALWAYS.

    Generation produces a video every single time on purpose. Every defect that has cost
    real time on this integration passed all of the automatic checks and was only visible
    by looking: a gripper 44 deg off vertical, an object knocked clean out of the
    container, an arm curling backwards after RELEASE, a wrist camera rotated 90 deg. In
    each case the episode was marked ``success``, the token sequence was clean, the
    per-token displacements were in range and the gripper widths were right. Watching the
    rollout is the check those cases fail, so it is not an optional extra step.

    Never fatal: a dataset that generated fine must not be lost to a rendering problem.
    """
    try:
        # preview.py is the sim-agnostic dataset preview renderer --
        # it only reads the rollout directory layout, which every backend shares.
        from scripts.trajectory.real2sim.preview import make_preview_mp4

        path = make_preview_mp4(Path(ds_dir))
        if path is None:
            print(f"[viz] WARNING: no episodes to render under {ds_dir}", flush=True)
            return
        # Check the file is actually a video. imageio picks its backend from the
        # extension AND what is installed: with no imageio-ffmpeg in the interpreter it
        # silently falls through to the TIFF writer and leaves an 8-byte ".mp4" behind,
        # having raised nothing useful. A truncated file here means no video for a
        # dataset that was supposed to always have one, so say so loudly.
        size = path.stat().st_size if path.exists() else 0
        if size < 10_000:
            print(f"[viz] WARNING: {path} is only {size} B -- not a playable video. "
                  f"Is imageio-ffmpeg installed in {sys.executable}?", flush=True)
        else:
            print(f"[viz] preview -> {path} ({size / 1e6:.1f} MB)", flush=True)
    except Exception as exc:  # noqa: BLE001 -- rendering must never lose the data
        print(f"[viz] preview FAILED for {ds_dir}: {exc!r}", flush=True)



def _layout_drift(backend, layout: dict) -> dict:
    """{asset: metres} between where the demo saw each object and where it is now.

    Empty when the track predates layout recording, so old tracks still run.
    """
    drift: dict = {}
    for name, xyz in (layout or {}).items():
        try:
            now = backend.object_centroid(name)
        except Exception:  # noqa: BLE001 -- prims without queryable geometry
            continue
        drift[name] = float(np.linalg.norm(now - np.asarray(xyz, dtype=np.float64)))
    return drift


def _segment_moves(seg: np.ndarray, step_m: float) -> bool:
    """Did the DEMO actually move during this segment, or is it a parked hold?

    ``record_demos`` steps every episode to a fixed length, so the stretch after the last
    gripper event is typically the arm sitting still -- ``tcp[-1]`` is wherever it was
    parked, not a waypoint. Chasing a parked pose is the wrong way to end an episode:
    ``retarget_place`` deliberately puts the follower's FLANGE where the demo's was not (it
    aims the carried OBJECT at the target, correcting for the follower's own grasp offset),
    so chasing the demo's flange UNDOES that correction and drags the arm back toward the
    base -- the visible "curl inward" at the end of a rollout.

    What it is NOT is pointless. The motion was doing a real job by accident: RoboLab's
    success predicate carries ``require_gripper_detached``, so SOME retreat after RELEASE
    is required to complete the task at all -- dropping it outright made every episode
    fail. Only the DIRECTION was an artefact of where the demo happened to stop. The
    retreat is now explicit and upward (:data:`RELEASE_RETREAT_M`), which detaches for a
    reason a policy can see in the image.

    Measured against the demo's own motion rather than special-cased to the tail, so a
    mid-track pause is skipped for the same reason.
    """
    if len(seg) < 2:
        return False
    return bool(np.max(np.linalg.norm(seg - seg[0], axis=1)) > 0.5 * float(step_m))


def _follow_all(args: argparse.Namespace) -> int:
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        GRASP,
        RELEASE,
        AtomicExec,
        EpisodeComplete,
        RolloutWriter,
        TokenBudgetExceeded,
        TokenEpisode,
        rdp,
    )
    from scripts.trajectory.real2sim.backends import make_backend
    from scripts.trajectory.real2sim.robolab import tasks

    def retarget_place(ep: TokenEpisode, plan: dict) -> None:
        """Re-aim the placement from LIVE state just before releasing.

        The demo's placement waypoint came from the DEMO's own grasp offset, but the
        follower grasps from a lattice position and can hold the object with a different
        offset -- blindly chasing the recorded waypoint then leaves it off-centre. Steer
        the CARRIED OBJECT (not the flange) onto the target, and take the drop height from
        the object's actual underside.
        """
        place = tasks.place_tcp(ep.backend, plan["object"], plan["target"],
                                plan["target_kind"])
        tcp = ep.backend.tcp_pos()
        ep.chase(np.array([place[0], place[1], tcp[2]]), tol=0.008)
        place = tasks.place_tcp(ep.backend, plan["object"], plan["target"],
                                plan["target_kind"])
        tcp = ep.backend.tcp_pos()
        ep.chase(np.array([tcp[0], tcp[1], place[2]]), tol=0.008)

    def follow_track(backend, track: dict, out_dir: Path,
                     agentview_square: Optional[int]) -> dict:
        tcp = np.asarray(track["tcp"], dtype=np.float64)
        events = [(int(i), str(t)) for i, t in track["events"]]
        bounds = [0] + [i for i, _ in events] + [len(tcp) - 1]
        plan = track.get("plan") or {}

        writer = RolloutWriter(out_dir, agentview_square=agentview_square or None)
        ep = TokenEpisode(
            backend, writer,
            # gripper_steps is NOT optional here. With the default 10 the Robotiq has not
            # finished travelling when the next token starts, so the lift begins on a
            # half-closed gripper and the object slips out mid-carry -- while the recorded
            # tokens still claim a carry. Measured: with 10 the cube was dropped at z=0.32
            # (width jumped 0.056 -> 0.085) and the episode logged 12 further MV_UP tokens
            # holding nothing; with GRIPPER_SETTLE_STEPS the same lift keeps it held.
            AtomicExec(backend, step_m=args.step_m, max_cmd_m=args.step_m,
                       max_ctrl_steps=24, gripper_steps=tasks.GRIPPER_SETTLE_STEPS),
            max_tokens=args.max_tokens,
        )
        def dropped() -> bool:
            """Commanded closed but holding nothing -> the object slipped out."""
            return (ep.exec.gripper_closed
                    and backend.gripper_width() > tasks.DROPPED_WIDTH_M)

        try:
            for si in range(len(bounds) - 1):
                a, b = bounds[si], bounds[si + 1]
                if b > a and _segment_moves(tcp[a:b + 1], args.step_m):
                    seg = tcp[a:b + 1]
                    corners = rdp(seg)
                    for k, ci in enumerate(corners[1:], 1):
                        last = si == len(bounds) - 2 and k == len(corners) - 1
                        # tighter tolerance right before a gripper event / at the goal
                        tol = 0.008 if (last or si < len(events)) and k == len(corners) - 1 \
                            else 0.012
                        ep.chase(seg[ci], tol=tol)
                        # Bail the moment the object is gone: every token recorded after a
                        # drop is a transport label over an empty gripper.
                        if dropped():
                            return {"success": False, "steps": writer.step,
                                    "reason": "dropped_in_transit", "writer": writer}
                if si < len(events):
                    _idx, tok = events[si]
                    if tok == RELEASE and plan:
                        retarget_place(ep, plan)
                    ep.emit(tok, "grasp" if tok == GRASP else "release")
                    if tok == RELEASE:
                        # Lift clear of what was just placed. NOT decoration: RoboLab's
                        # success predicate carries require_gripper_detached, which is
                        # literally `not in_contact(object, gripper)`, so an episode that
                        # opens the fingers and stays put NEVER succeeds -- the fingers
                        # are still straddling the object in the container.
                        #
                        # This used to happen by accident. The follower chased the demo's
                        # parked tail pose, which lay behind the follower's own placement,
                        # so the arm dragged BACKWARD and incidentally detached. That is
                        # the "curl inward" at the end of every rollout: the direction was
                        # an artefact of where the demo happened to stop, not of the task.
                        # Retreating straight UP is the same detachment with a reason a
                        # policy can see in the image.
                        ep.go_z(ep.backend.tcp_pos()[2] + RELEASE_RETREAT_M)

            # Let the scene come to rest BEFORE concluding anything. RoboLab's predicate
            # for this task is static (object_in_container with require_contact_with and
            # require_gripper_detached), so it cannot fire until the released object has
            # actually stopped moving -- measured on RubiksCubeTask, that takes ~40
            # control steps, twice the old budget of 20.
            settled_in = ep.settle_steps(SETTLE_STEPS)
            success = success0 = settled_in is not None

            # Same rule as the segment loop, for the same reason (see _segment_moves):
            # only re-chase a tail the demo actually travelled.
            tail_moved = _segment_moves(tcp[bounds[-2]:], args.step_m)
            if not success and tail_moved:
                # One corrective round: the IK can drift the last cm; re-chase the final
                # waypoint (lattice-limited) and settle again before giving up.
                ep.chase(tcp[-1], tol=0.011)
                settled_in = ep.settle_steps(SETTLE_STEPS)
                success = settled_in is not None
        except EpisodeComplete:
            # The task finished mid-plan and RoboLab froze the scene. That is a normal,
            # successful end -- the remaining planned tokens exist only because the
            # planner had no way to know. Fall through to scoring; emitting them would
            # append frames that cannot change (measured: 8 MV_UP at exactly 0.00 mm).
            success = success0 = backend.success()
            settled_in = 0
        except TokenBudgetExceeded as exc:
            return {"success": False, "steps": writer.step, "reason": str(exc),
                    "writer": writer}
        # Where everything actually ended up. Recorded on success AND failure: "the
        # predicate did not fire" says nothing about WHY, and the frames only show it if
        # you happen to look. Cheap to store, and the first thing anyone needs.
        geom = {}
        try:
            if plan:
                obj_c = backend.object_centroid(plan["object"])
                tgt_c = backend.object_centroid(plan["target"])
                geom = {
                    "final_object_xyz": [round(float(v), 4) for v in obj_c],
                    "final_target_xyz": [round(float(v), 4) for v in tgt_c],
                    "final_object_to_target_xy_m": round(
                        float(np.linalg.norm(obj_c[:2] - tgt_c[:2])), 4),
                    "final_tcp_xyz": [round(float(v), 4) for v in backend.tcp_pos()],
                    "final_ee_tilt_deg": round(backend.ee_tilt_deg(), 2),
                }
        except Exception:  # noqa: BLE001 -- diagnostics must never fail the episode
            geom = {}
        return {"success": bool(success), "steps": writer.step, "reason": "", **geom,
                # How much of the settle budget was actually needed, and whether the
                # corrective re-chase ran at all. Both are cheap to record and make the
                # margin visible in metadata instead of leaving it to be rediscovered.
                "settle_steps_used": settled_in, "settle_budget": SETTLE_STEPS,
                "tail_rechased": bool(not success0 and tail_moved),
                "writer": writer}

    track_files = sorted(Path(args.tracks).glob("track_ep*.json"))
    if not track_files:
        raise SystemExit(f"no tracks under {args.tracks}")
    first = json.loads(track_files[0].read_text())
    task_key = first.get("task_key", Path(args.tracks).name)
    sim = args.sim or first.get("sim", "robolab")

    # The camera contract comes from the SAME file the deployment runner reads, so the
    # frames stored here match the ones the policy will be served. CLI flags stay as
    # overrides for one-off probing, but the config is what a real dataset is built from.
    robot_cfg = load_yaml(ROOT / args.robot_config)
    backend_kwargs = {
        "task": task_key,
        "device": args.device,
        "instruction_type": first.get("instruction_type", "default"),
        "camera_preset": first.get("camera_preset", "WRIST_LEFT"),
        **camera_contract(robot_cfg),
    }
    if args.wrist_flip is not None:
        backend_kwargs["wrist_flip"] = args.wrist_flip
    if args.crop_aspect is not None:
        backend_kwargs["agentview_crop_aspect"] = args.crop_aspect or None
        backend_kwargs["wrist_crop_aspect"] = args.crop_aspect or None
    # Same story for the letterbox: --agentview-square overrides, else the config's.
    agentview_square = (
        args.agentview_square if args.agentview_square
        else robot_cfg.get("agentview_square_size")
    )
    backend = make_backend(sim, **backend_kwargs)
    # Same generation-time rule as the oracle: RoboLab freezes an env the instant its
    # success predicate fires, which here is DURING the post-RELEASE retreat below -- so
    # the retreat tokens land on a frozen scene, `emit` refuses them, and the episode ends
    # on RELEASE. Suspending the predicate (time_out stays) keeps the retreat real;
    # success is still RoboLab's own predicate, evaluated on demand.
    suspend = getattr(backend, "suspend_task_termination", None)
    if callable(suspend):
        suspend()
    print(
        f"[follow:{task_key}] camera contract from {args.robot_config}: "
        f"agentview={backend.agentview_camera} rot={backend.agentview_rotation_degrees} "
        f"flip={backend.agentview_flip} crop={backend.agentview_crop_aspect} "
        f"square={agentview_square} | wrist={backend.wrist_camera} "
        f"rot={backend.wrist_rotation_degrees} flip={backend.wrist_flip} "
        f"crop={backend.wrist_crop_aspect}",
        flush=True,
    )

    out_root = Path(args.out) / f"{task_key}_follow"
    kept = 0
    for tf in track_files:
        track = json.loads(tf.read_text())
        # The env's reset events draw from the GLOBAL np.random; seed it before reset so
        # the follower sees the same scene the recorder did.
        np.random.seed(int(track["seed"]))
        backend.reset(int(track["seed"]))

        drift = _layout_drift(backend, track.get("layout") or {})
        if drift:
            worst = max(drift.values())
            if worst > LAYOUT_TOL_M:
                print(f"[follow:{task_key}] {tf.name} SKIPPED: reset did not reproduce "
                      f"the recorded layout (max drift {worst * 100:.1f} cm > "
                      f"{LAYOUT_TOL_M * 100:.0f} cm): "
                      + ", ".join(f"{k} {v * 100:.1f}cm" for k, v in drift.items()),
                      flush=True)
                continue

        out_dir = out_root / f"rollout_{kept:03d}"
        result = follow_track(backend, track, out_dir, agentview_square)
        writer = result.pop("writer")
        writer.close({
            "source": f"follow_{task_key}",
            "method": "closed_loop_follower",
            "sim": sim,
            "env_id": track["env_id"],
            "task": track["task"],
            "task_key": task_key,
            "plan": track.get("plan"),
            "seed": track["seed"],
            "step_m": args.step_m,
            # The contract AS APPLIED (read off the backend), not the CLI flags -- a
            # dataset has to be able to state how its own pixels were produced.
            "agentview_camera": backend.agentview_camera,
            "agentview_rotation_degrees": backend.agentview_rotation_degrees,
            "agentview_flip": backend.agentview_flip,
            "agentview_crop_aspect": backend.agentview_crop_aspect,
            "agentview_square": agentview_square or None,
            "wrist_camera": backend.wrist_camera,
            "wrist_rotation_degrees": backend.wrist_rotation_degrees,
            "wrist_flip": backend.wrist_flip,
            "wrist_crop_aspect": backend.wrist_crop_aspect,
            **{k: v for k, v in result.items()},
        })
        print(f"[follow:{task_key}] {tf.name} steps={result['steps']} "
              f"success={result['success']} {result['reason']}", flush=True)
        if result["success"] or args.keep_failures:
            kept += 1
            if kept == 1:
                # Render as soon as the FIRST episode lands, not only at the end. A batch
                # is tens of minutes of GPU time and every defect that has mattered on
                # this project was visible in the frames and invisible in the statistics
                # -- so make the first episode checkable while the rest are still running,
                # instead of discovering a 90 deg wrist or a 44 deg tilt an hour later.
                write_preview(out_root)
        else:
            shutil.rmtree(out_dir, ignore_errors=True)

    backend.close()
    print(f"[follow:{task_key}] kept {kept}/{len(track_files)} -> {out_root}")
    write_preview(out_root)
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
