#!/usr/bin/env python3
"""Scheme A on RoboLab -- scripted privileged oracle emitting 2 cm atomic tokens.

Sibling of ``real2sim/maniskill/oracle.py``, driving a RoboLab (Isaac Lab) Franka +
Robotiq 2F-85 instead of a ManiSkill Panda. The state machine is the same shape -- align
XY in Manhattan runs, descend, grasp, carry, place, release, all as single-axis 2 cm tokens
recorded frame-before-action -- and it reuses the identical
``AtomicExec`` / ``TokenEpisode`` / ``RolloutWriter`` core. What changes is that the plan is
not hand-written per task: it is read off the RoboLab task's OWN ``subtasks`` declaration
(see ``real2sim/robolab/tasks.py``), so any single-object ``pick_and_place`` task among
RoboLab-120 can be generated without touching this file.

Because the data is produced BY the same discrete dynamics the policy runs at deployment,
every frame sits on the deployment lattice by construction -- the whole reason the
discretiser executes tokens instead of labelling a continuous demo.

Episodes end with RoboLab's own termination predicate; failures are discarded unless
``--keep-failures``.

Isaac Sim start-up is expensive, so ONE process generates many episodes against one env
(the same reason ``scripts/run_robolab_mvtoken.py --episodes`` exists).

Usage (RoboLab's interpreter, from the repo root):
    $ROBOLAB_ROOT/.venv/bin/python \
        scripts/trajectory/real2sim/robolab/oracle.py \
        --task BananaInBowlTask --episodes 20 --agentview-square 256 --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from core.config import camera_contract, load_yaml  # noqa: E402


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

# A recorded MV_* token that achieved less than this is a lie about the physics: the frame
# is labelled with a 2 cm move the arm did not make. Matches the executor's own stall
# threshold (0.3 x step_m) and the dataset gate in scripts/robolab/check_dataset.py.
MIN_TOKEN_TRAVEL_M = 0.006


def stalled_token_count(rollout_dir: Path) -> int:
    """How many recorded move tokens moved less than :data:`MIN_TOKEN_TRAVEL_M`.

    Read back off what was WRITTEN rather than tracked during execution, so it measures the
    dataset as the trainer will see it -- ``ee_pose`` is the pose before each token, so
    token i's displacement is pose[i+1] - pose[i].
    """
    import json as _json

    path = rollout_dir / "actions.jsonl"
    if not path.exists():
        return 0
    recs = [_json.loads(line) for line in path.open() if line.strip()]
    stalled = 0
    for cur, nxt in zip(recs, recs[1:]):
        if cur.get("kind") != "move":
            continue
        d = float(np.linalg.norm(
            np.asarray(cur["ee_pose"][:3]) - np.asarray(nxt["ee_pose"][:3])
        ))
        if d < MIN_TOKEN_TRAVEL_M:
            stalled += 1
    return stalled


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True,
                    help="RoboLab Task class name, e.g. BananaInBowlTask.")
    ap.add_argument("--sim", default="robolab", help="discretiser backend (see backends/)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--step-m", type=float, default=0.02,
                    help="physical metres per atomic token (the MVTOKEN contract is 0.02)")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--drop-margin-m", type=float, default=None,
                    help="metres between the carried object's underside and the drop "
                         "target when the fingers open (default: "
                         "real2sim.robolab.tasks.DROP_MARGIN_M). Lower = the arm descends "
                         "further before RELEASE.")
    ap.add_argument("--max-stalled-frac", type=float, default=0.05,
                    help="discard an episode when more than this fraction of its recorded "
                         "move tokens achieved under 6 mm of travel. A blocked arm still "
                         "records full 2 cm labels over frames that barely change.")
    ap.add_argument("--retreat-tokens", type=int, default=2,
                    help="MV_UP tokens recorded AFTER release. Must be >= 1: an episode "
                         "that ends on RELEASE makes its final frame mean both RELEASE "
                         "and the synthesised terminal DONE, which is what taught the "
                         "policy to stop above the container.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--instruction-type", default="default")
    ap.add_argument("--camera-preset", default="WRIST_LEFT")
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
                         "matches the real-robot datasets and the deployment runner. "
                         "0 = keep the raw render.")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--gui", action="store_true", help="run with the Isaac Sim viewport")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Isaac Sim first: nothing may import isaaclab/robolab before the Kit app is up.
    from core.sim.robolab_task import launch_isaac

    simulation_app = launch_isaac(headless=not args.gui, device=args.device)
    try:
        return _generate(args)
    except Exception:
        # Isaac Sim's SimulationApp.close() can terminate the process outright, which
        # swallows BOTH the traceback and the exit code -- a failure then looks like a
        # clean exit 0 that silently produced nothing. Print before closing.
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


def _generate(args: argparse.Namespace) -> int:
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        GRASP,
        RELEASE,
        AtomicExec,
        EpisodeComplete,
        RolloutWriter,
        TokenEpisode,
        TokenBudgetExceeded,
    )
    from scripts.trajectory.real2sim.backends import make_backend
    from scripts.trajectory.real2sim.robolab import tasks
    from scripts.trajectory.real2sim.robolab.tasks import GRASPED_WIDTH_M, UnsupportedTask

    class OracleEpisode(TokenEpisode):
        """One pick-and-place episode, emitted entirely as atomic tokens."""

        def __init__(self, backend, writer, rng, plan: dict, step_m: float, max_tokens: int,
                     drop_margin_m: float = tasks.DROP_MARGIN_M,
                     retreat_tokens: int = 2):
            super().__init__(
                backend,
                writer,
                # max_cmd_m == step_m: the executor never commands more than one token's
                # worth in a single control step, so the IK target stays inside the range
                # the relative-IK solver tracks cleanly. gripper_steps is raised well above
                # the default 10 because the Robotiq's binary command takes that long to
                # travel -- sampled too early the width still reads near-open and an EMPTY
                # grasp passes the check (see tasks.GRIPPER_SETTLE_STEPS).
                AtomicExec(backend, step_m=step_m, max_cmd_m=step_m, max_ctrl_steps=24,
                           gripper_steps=tasks.GRIPPER_SETTLE_STEPS),
                max_tokens=max_tokens,
                rng=rng,
            )
            self.plan = plan
            self.drop_margin_m = float(drop_margin_m)
            self.retreat_tokens = int(retreat_tokens)
            self.retreat_travel_m: list[float] = []
            self.stalled_tokens = 0

        def _grasp_with_retries(self) -> bool:
            """GRASP, with bounded empty-grasp retries (reopen, sink one token, retry)."""
            for _retry in range(3):
                self.emit(GRASP)
                if self.backend.gripper_width() > GRASPED_WIDTH_M:
                    return True
                self.emit(RELEASE)
                self.emit("MV_DOWN")
            return False

        def run(self) -> dict:
            obj, target = self.plan["object"], self.plan["target"]

            # 1) to the approach pose as ONE dominant-axis pursuit, then down onto the
            # object. `chase` commits to the axis of LARGEST error and stays on it until
            # that axis is done, so from a high start (|dz| >> |dxy|) the arm DESCENDS
            # first and only trims XY once it is near the approach height -- instead of
            # shuffling sideways at full altitude, which is what the previous
            # align_xy-then-go_z pair produced and what the recorded rollouts showed.
            #
            # Tolerances are clamped to >= 0.55 * step_m by TokenEpisode: anything tighter
            # than half a step makes 2 cm ping-pong geometrically inevitable.
            self.chase(tasks.approach_tcp(self.backend, obj), tol=0.012)
            self.align_xy(tasks.grasp_tcp(self.backend, obj))
            self.go_z(tasks.grasp_tcp(self.backend, obj, z_jitter=0.004, rng=self.rng)[2])
            self.align_xy(tasks.grasp_tcp(self.backend, obj))

            # 2) grasp -- and remember the height it happened at, because once the object
            # is in the fingers its pose follows the hand and can no longer be used to
            # work out how far to lift (see tasks.carry_z).
            grasp_flange_z = float(self.backend.tcp_pos()[2])
            if not self._grasp_with_retries():
                return {"success": False, "reason": "grasp_failed"}

            # 3) lift clear of both the pick site and the target rim. Deliberately a PURE
            # vertical run, and step 4 a PURE horizontal one: carrying at a slant would
            # interleave MV_UP with MV_RIGHT and produce a staircase, which is exactly
            # what the follower (Scheme D) inherited from its continuous demos.
            self.go_z(tasks.carry_z(self.backend, obj, target, grasp_flange_z))
            if self.backend.gripper_width() < GRASPED_WIDTH_M:
                return {"success": False, "reason": "dropped_on_lift"}

            # 4) carry over the target, aiming so the CARRIED OBJECT (not the flange) lands
            # on the target centre -- RoboLab's containment predicate checks the object.
            place = tasks.place_tcp(self.backend, obj, target, self.plan["target_kind"],
                                    drop_margin_m=self.drop_margin_m)
            self.align_xy(place)
            if self.backend.gripper_width() < GRASPED_WIDTH_M:
                return {"success": False, "reason": "dropped_in_transit"}

            # 5) descend to the drop height, release, retreat.
            self.go_z(place[2])
            self.emit(RELEASE)
            if not self._retreat():
                return {"success": False, "reason": "retreat_blocked"}
            # Containment/support predicates need a few quiet steps to turn true.
            settled = self.settle(20)
            return {"success": bool(settled or self.backend.success())}

        def _retreat(self) -> bool:
            """Post-RELEASE lift: at least ``retreat_tokens`` MV_UP that REALLY move.

            The episode must not end on RELEASE. Downstream, the final frame is consumed
            twice -- once with its own label and once as the synthesised terminal DONE
            (``rollout_to_llamafactory.py``) -- so an episode ending on RELEASE trains one
            image to mean both "open the fingers" and "the task is over", on a frame where
            the object is still in a closed gripper above the container. That is the
            "stops above the target" behaviour the deployed policy showed.

            Ending on MV_UP makes both labels agree with the picture: the fingers are open,
            the object is already in the container, and the arm is on its way out.

            REALLY move is the other half. RoboLab used to freeze the env at RELEASE, and
            the earlier version of this code emitted these tokens into the frozen scene --
            80 of 812 samples labelled MV_UP over EXACTLY 0.00 mm of travel. The freeze is
            gone (``RobolabBackend.suspend_task_termination``), so verify rather than
            assume: a retreat that does not lift discards the episode instead of writing a
            lie into the dataset.
            """
            self.retreat_travel_m = []
            for _ in range(self.retreat_tokens):
                before = float(self.backend.tcp_pos()[2])
                try:
                    self.emit("MV_UP", "move")
                except EpisodeComplete:
                    # time_out is the only DoneTerm left during generation, and it means
                    # the episode ran out of clock -- the frames after it are not real.
                    return False
                travel = float(self.backend.tcp_pos()[2]) - before
                self.retreat_travel_m.append(travel)
                if travel < 0.3 * self.exec.step_m:
                    return False
            return True

    # The camera contract comes from the SAME file the deployment runner reads, so the
    # frames stored here match the ones the policy will be served. CLI flags stay as
    # overrides for one-off probing, but the config is what a real dataset is built from.
    robot_cfg = load_yaml(ROOT / args.robot_config)
    backend_kwargs: dict[str, Any] = {
        "task": args.task,
        "device": args.device,
        "instruction_type": args.instruction_type,
        "camera_preset": args.camera_preset,
        # Point RoboLab's OWN artefacts (env_cfg.json, and the HDF5 episode recorder this
        # pipeline never reads) at this run's directory, as the eval runner already does.
        # Not tidiness: the recorder takes an EXCLUSIVE lock on its file, so two generator
        # processes leaving their output in RoboLab's default directory clobber each other
        # and report a corrupt file. With this, tasks can be generated in parallel.
        "output_dir": str(Path(args.out).resolve()),
        **camera_contract(robot_cfg),
    }
    if args.wrist_flip is not None:
        backend_kwargs["wrist_flip"] = args.wrist_flip
    if args.crop_aspect is not None:
        backend_kwargs["agentview_crop_aspect"] = args.crop_aspect or None
        backend_kwargs["wrist_crop_aspect"] = args.crop_aspect or None
    agentview_square = (
        args.agentview_square if args.agentview_square
        else robot_cfg.get("agentview_square_size")
    )
    backend = make_backend(args.sim, **backend_kwargs)
    # Generation only: stop RoboLab ending (and freezing) the episode the instant the
    # success predicate fires, so the post-release retreat is real motion over real
    # frames. Success is still RoboLab's own predicate, just evaluated on demand.
    suspend = getattr(backend, "suspend_task_termination", None)
    if callable(suspend):
        suspend()
    print(
        f"[oracle] camera contract from {args.robot_config}: "
        f"agentview={backend.agentview_camera} rot={backend.agentview_rotation_degrees} "
        f"flip={backend.agentview_flip} crop={backend.agentview_crop_aspect} "
        f"square={agentview_square} | wrist={backend.wrist_camera} "
        f"rot={backend.wrist_rotation_degrees} flip={backend.wrist_flip} "
        f"crop={backend.wrist_crop_aspect}",
        flush=True,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    kept, tried, reasons = 0, 0, {}
    while kept < args.episodes and tried < args.episodes * 4:
        seed = args.seed0 + tried
        tried += 1
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        backend.reset(seed)

        try:
            plan = tasks.resolve_plan(backend)
        except UnsupportedTask as exc:
            print(f"[oracle] {args.task}: {exc}")
            return 2
        if tried == 1:
            print(f"[oracle] plan: {tasks.describe(backend, plan)}")

        rollout_dir = out_root / f"rollout_{kept:03d}"
        writer = RolloutWriter(
            rollout_dir, agentview_square=agentview_square or None
        )
        episode = OracleEpisode(
            backend, writer, rng, plan, step_m=args.step_m, max_tokens=args.max_tokens,
            drop_margin_m=(args.drop_margin_m if args.drop_margin_m is not None
                           else tasks.DROP_MARGIN_M),
            retreat_tokens=args.retreat_tokens,
        )
        try:
            result = episode.run()
        except EpisodeComplete:
            # Task finished mid-plan and the sim froze; a normal end, not a failure.
            result = {"success": bool(backend.success()), "reason": ""}
        except TokenBudgetExceeded as exc:
            result = {"success": False, "reason": str(exc)}
        except UnsupportedTask as exc:
            print(f"[oracle] {exc}")
            return 2
        ok = bool(result.get("success"))
        # The ending is part of the contract, so check it here rather than trusting the
        # state machine: an episode whose last token is RELEASE hands the converter a
        # final frame that has to serve as both RELEASE and the terminal DONE.
        last_token = writer.tokens[-1] if writer.tokens else None
        if ok and last_token != "MV_UP":
            ok = False
            result = {"success": False, "reason": f"bad_ending:{last_token}"}
        # Reject episodes that spent a meaningful share of their tokens NOT MOVING.
        #
        # The planners' stall guards stop a single pursuit that goes nowhere, but they stop
        # it AFTER recording the blocked tokens, and an episode can stall in several
        # separate runs and still finish successfully. Measured on the 2026-08-12 batch:
        # CannedFoodInBinTask/rollout_007 carried its can into a bottle standing between it
        # and the bin and logged 15 of 68 tokens (22%) as full 2 cm moves over 3.8-5.6 mm of
        # real travel -- while succeeding, ending on MV_UP, and keeping the gripper closed
        # throughout. Every other check passes; only the per-token displacement shows it.
        stalled = stalled_token_count(rollout_dir)
        episode.stalled_tokens = stalled
        if ok and writer.step and stalled / writer.step > args.max_stalled_frac:
            ok = False
            result = {"success": False,
                      "reason": f"stalled:{stalled}/{writer.step}"}
        writer.close(
            {
                "drop_margin_m": episode.drop_margin_m,
                "retreat_tokens": args.retreat_tokens,
                "retreat_travel_m": [round(t, 5) for t in episode.retreat_travel_m],
                "last_token": last_token,
                "stalled_tokens": episode.stalled_tokens,
                "source": f"oracle_{args.task}",
                "method": "privileged_oracle",
                "sim": args.sim,
                "env_id": backend.env_id,
                "task": args.task,
                "task_text": backend.task_description,
                "plan": plan,
                "seed": seed,
                "step_m": args.step_m,
                # The contract AS APPLIED (read off the backend), not the CLI flags
                # -- a dataset has to be able to state how its own pixels were made.
                "agentview_camera": backend.agentview_camera,
                "agentview_rotation_degrees": backend.agentview_rotation_degrees,
                "agentview_flip": backend.agentview_flip,
                "agentview_crop_aspect": backend.agentview_crop_aspect,
                "agentview_square": agentview_square or None,
                "wrist_camera": backend.wrist_camera,
                "wrist_rotation_degrees": backend.wrist_rotation_degrees,
                "wrist_flip": backend.wrist_flip,
                "wrist_crop_aspect": backend.wrist_crop_aspect,
                "success": ok,
                "reason": result.get("reason"),
            }
        )
        if ok or args.keep_failures:
            kept += 1
            if kept == 1:
                # Render as soon as the FIRST episode lands, not only at the end. A batch
                # is tens of minutes of GPU time and every defect that has mattered on
                # this project was visible in the frames and invisible in the statistics
                # -- so make the first episode checkable while the rest are still running,
                # instead of discovering a 90 deg wrist or a 44 deg tilt an hour later.
                write_preview(out_root)
            # flush: a batch redirects stdout to a file, and without this the per-episode
            # lines sit in the buffer for tens of minutes -- long enough that a healthy run
            # looks hung while it is happily writing frames.
            print(f"[oracle] episode seed={seed}: success={ok} -> {rollout_dir.name} "
                  f"({writer.step} tokens)", flush=True)
        else:
            reason = str(result.get("reason", "not_successful"))
            reasons[reason] = reasons.get(reason, 0) + 1
            shutil.rmtree(rollout_dir, ignore_errors=True)
            print(f"[oracle] episode seed={seed}: FAILED ({reason}), discarded", flush=True)

    backend.close()
    print(f"[oracle] kept {kept}/{tried} attempted episodes -> {out_root}")
    write_preview(out_root)
    if reasons:
        print(f"[oracle] failure reasons: {reasons}")
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
