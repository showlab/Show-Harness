#!/usr/bin/env python3
"""Scheme A -- scripted privileged oracle: plan and execute directly in 2 cm atomic tokens.

A state machine reads the object poses (privileged sim state), aligns axis-by-axis in
Manhattan runs, grasps, carries, places and releases -- emitting exactly one atomic token
per recorded frame and executing it closed-loop. Because the data is produced BY the same
discrete dynamics the policy will run at deployment, every frame sits on the deployment
distribution by construction (no demo-path mismatch), and the generation process mirrors
how the real teleop data was collected: a human pressing one direction key at a time.

Episodes end with the env's own success check; failures are discarded unless
``--keep-failures``.

Two tasks share this file (they differed only in constants and two lines of placement
maths, and used to be two 300-line near-copies):

    blockpap    RLinf BlockPAP-v1  -- orange block onto the coaster
    blockstack  RLinf BlockStack-v1 -- white block onto the gray block (lattice-snapped
                layout: 1 cm success tolerance vs a 2 cm step, see tasks.py)

Usage (mimicgen conda env):
    python scripts/trajectory/real2sim/maniskill/oracle.py --task blockpap \
        --episodes 20 --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.atomic_tokenizer import (  # noqa: E402
    GRASP,
    RELEASE,
    AtomicExec,
    RolloutWriter,
    TokenEpisode,
)
from scripts.trajectory.real2sim.backends import make_backend  # noqa: E402
from scripts.trajectory.real2sim.maniskill import tasks  # noqa: E402
from scripts.trajectory.real2sim.maniskill.tasks import const  # noqa: E402

# Fingers stopped on the block instead of closing on air (both rigs hold 4 cm cubes).
GRASPED_WIDTH_M = 0.012

# Per-task oracle geometry. Kept as explicit per-task callables rather than one "smart"
# rule because these numbers reproduce the published ms_0717 datasets seed-for-seed; note
# the asymmetry in ``place_tcp_z`` -- BlockPAP descends to the seated-block height directly
# (the TCP sits at the block centre after a centred grasp), while BlockStack corrects for
# where the carried block actually ended up in the fingers, which its 1 cm tolerance needs.
SPECS: dict[str, dict[str, Any]] = {
    "blockpap": {
        "max_tokens": 140,
        # Grasp height jitter (metres, uniform +-): a little variety in the frames that
        # precede GRASP.
        "grasp_z_jitter": 0.003,
        "carry_offset": False,
        "carry_z": lambda env, obj, tgt: max(
            obj[2] + 0.03,
            tgt[2] + float(const(env, "COASTER_THICKNESS"))
            + float(const(env, "BLOCK_HALF_SIZE")[2]) + 0.03,
        ) + 0.06,
        "place_tcp_z": lambda env, tgt, tcp, obj: tasks.TASKS["blockpap"]["drop_z"](env, tgt),
        "final_hold": 10,
    },
    "blockstack": {
        "max_tokens": 160,
        "grasp_z_jitter": 0.0,
        "carry_offset": True,
        "carry_z": lambda env, obj, tgt: (
            tgt[2] + 4 * float(const(env, "BLOCK_HALF_SIZE")[2]) + 0.04
        ),
        "place_tcp_z": lambda env, tgt, tcp, obj: (
            tasks.TASKS["blockstack"]["drop_z"](env, tgt) + (tcp[2] - obj[2])
        ),
        "final_hold": 12,
    },
}


class OracleEpisode(TokenEpisode):
    """One pick-and-place episode, emitted entirely as atomic tokens."""

    def __init__(self, backend, writer, rng, task: str, step_m: float) -> None:
        spec = SPECS[task]
        super().__init__(backend, writer, AtomicExec(backend, step_m=step_m),
                         max_tokens=spec["max_tokens"], rng=rng)
        self.task = task
        self.spec = spec
        self.tspec = tasks.task_spec(task)

    def _pos(self, which: str) -> np.ndarray:
        return tasks.actor_pos(self.backend.env, self.tspec[which])

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
        env = self.backend.env
        obj0, tgt0 = self._pos("carried"), self._pos("target")
        jitter = self.spec["grasp_z_jitter"]
        grasp_z = obj0[2] + (float(self.rng.uniform(-jitter, jitter)) if jitter else 0.0)

        # 1) over the object, down to grasp height, then a fine align on its LIVE pose.
        # All tolerances stay >= 0.55 * step_m (enforced by TokenEpisode): tighter than
        # half a step makes 2 cm ping-pong around the target geometrically inevitable.
        # +-1.1 cm is plenty for an 8 cm finger opening.
        self.align_xy(obj0)
        self.go_z(grasp_z)
        self.align_xy(self._pos("carried"))

        # 2) grasp
        if not self._grasp_with_retries():
            return {"success": False, "reason": "grasp_failed"}

        # 3) lift clear
        self.go_z(self.spec["carry_z"](env, obj0, tgt0))
        if self.backend.gripper_width() < GRASPED_WIDTH_M:
            return {"success": False, "reason": "dropped_on_lift"}

        # 4) carry over the target. Where the tolerance is tight, aim so the CARRIED
        # object (which can sit a few mm off-centre in the fingers) lands on the target
        # centre, rather than aiming the TCP at it.
        target_xy = tgt0[:2]
        if self.spec["carry_offset"]:
            target_xy = target_xy - (self._pos("carried")[:2] - self.backend.tcp_pos()[:2])
        self.align_xy(target_xy)

        # 5) descend until the object is seated, release, retreat (gives the DONE-adjacent
        # frames some variety)
        self.go_z(self.spec["place_tcp_z"](env, tgt0, self.backend.tcp_pos(),
                                           self._pos("carried")))
        self.emit(RELEASE)
        self.run_tokens(["MV_UP", "MV_UP"])
        self.exec.hold(self.spec["final_hold"])
        return {"success": self.backend.success()}


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

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(SPECS), default="blockpap")
    ap.add_argument("--sim", default="maniskill", help="discretiser backend (see backends/)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--traj", default="random", help="scene trajectory preset, or 'random'")
    ap.add_argument(
        "--layout", choices=["wide", "native"], default="wide",
        help="wide (default): sample BOTH objects over the full reachable box after reset "
             "(tasks.WIDE_BOX). native: keep the env's own placement, which barely moves "
             "the target (BlockPAP's coaster spans ~4x2 cm; BlockStack pins the gray block).",
    )
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--no-snap", action="store_true",
                    help="sample the layout freely instead of snapping it to the atomic "
                         "lattice (blockstack: shows the raw, much lower success rate)")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--table-tex", default="white",
                    help="BlockPAP tabletop: 'white' | 'black' | a wood texture id "
                         "'001'..'021' ('006' is the BlockPAP-v1_Mix wood)")
    ap.add_argument("--agentview-square", type=int, default=0,
                    help="store agentview as an N x N letterbox (resize_with_pad); 256 "
                         "matches the real-robot datasets and the deployment runner. "
                         "0 = keep the raw render.")
    args = ap.parse_args()

    tspec = tasks.task_spec(args.task)
    snap = bool(tspec.get("snap_layout", False)) and not args.no_snap
    backend = make_backend(args.sim, **tasks.backend_kwargs(
        args.task, table_tex=args.table_tex, traj_id=args.traj))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    kept, tried = 0, 0
    while kept < args.episodes and tried < args.episodes * 4:
        seed = args.seed0 + tried
        tried += 1
        # The scene's own random layout draws from the GLOBAL np.random -- seed it for
        # reproducibility even when it is overridden below, then seed the env's own RNG
        # through reset().
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        backend.reset(seed)

        ex0 = AtomicExec(backend, step_m=args.step_m)
        layout_info: dict = {}
        if args.layout == "wide":
            if snap:
                # The lattice is anchored at the SETTLED gripper XY, so settle first.
                ex0.hold(8)
                layout_info = tasks.randomize_layout(
                    backend.env, args.task, rng, backend.tcp_pos()[:2], args.step_m,
                    snap=True)
                ex0.hold(6)  # let the objects come to rest before the oracle reads them
            else:
                layout_info = tasks.randomize_layout(
                    backend.env, args.task, rng, None, args.step_m, snap=False)
                ex0.hold(8)  # settle, mirrors deployment num_steps_wait
        else:
            ex0.hold(8)

        rollout_dir = out_root / f"rollout_{kept:03d}"
        writer = RolloutWriter(rollout_dir,
                               agentview_square=args.agentview_square or None)
        ep = OracleEpisode(backend, writer, rng, args.task, step_m=args.step_m)
        try:
            result = ep.run()
        except RuntimeError as exc:  # token budget exceeded
            result = {"success": False, "reason": str(exc)}
        ok = bool(result.get("success"))
        writer.close({
            "source": f"oracle_{args.task}",
            "method": "privileged_oracle",
            "env_id": tspec["env_id"],
            "seed": seed,
            "traj": args.traj,
            "layout": args.layout,
            "step_m": args.step_m,
            "table_tex": args.table_tex,
            "agentview_square": args.agentview_square or None,
            "task": tasks.task_text(args.task),
            "success": ok,
            "reason": result.get("reason", ""),
            # end-of-episode poses (the carried object has moved onto the target on success)
            "end_xy": {name: [round(float(v), 4)
                              for v in tasks.actor_pos(backend.env, name)[:2]]
                       for name, _z, _q in tspec["layout"]},
            **({"layout_sampled": layout_info} if layout_info else {}),
        })
        print(f"[oracle:{args.task}] seed={seed} steps={writer.step} success={ok} "
              f"{result.get('reason','')}", flush=True)
        if ok or args.keep_failures:
            kept += 1
            if kept == 1:
                # Render as soon as the FIRST episode lands, not only at the end. A batch
                # is tens of minutes of GPU time and every defect that has mattered on
                # this project was visible in the frames and invisible in the statistics
                # -- so make the first episode checkable while the rest are still running,
                # instead of discovering a 90 deg wrist or a 44 deg tilt an hour later.
                write_preview(out_root)
        else:
            shutil.rmtree(rollout_dir, ignore_errors=True)

    backend.close()
    print(f"[oracle:{args.task}] kept {kept}/{tried} episodes -> {out_root}")
    write_preview(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
