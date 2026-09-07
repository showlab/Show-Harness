#!/usr/bin/env python3
"""Scheme D -- closed-loop follower: re-execute a demo's path with 2 cm atomic tokens.

Reads the TCP tracks exported by ``record_demos.py``, resets the same scene to the same
seed and layout, reduces each track to RDP waypoints + gripper events, then chases each
waypoint with dominant-axis 2 cm tokens. Every recorded frame is a state the DISCRETE
controller actually reached, so every frame-to-frame move is strictly single-axis and sits
exactly on the deployment lattice. Episodes are kept only if the env's success check passes
at the end -- the decomposition is thereby VALIDATED, not assumed.

This is also the generic entry point of the whole approach: the tokeniser does not care
where the continuous track came from, so any source (a real teleop log, an official demo
dataset, another simulator's motion planner) can be turned into atomic-token training data
by writing a track file and pointing this script at the matching backend.

Output: ``<out>/<task>_follow/rollout_NNN`` in the teleop/MVTOKEN rollout format.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.atomic_tokenizer import (  # noqa: E402
    GRASP,
    RELEASE,
    AtomicExec,
    RolloutWriter,
    TokenEpisode,
    rdp,
)
from scripts.trajectory.real2sim.backends import make_backend  # noqa: E402
from scripts.trajectory.real2sim.maniskill import tasks  # noqa: E402


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

def retarget_place(ep: TokenEpisode, task: str) -> None:
    """Re-aim the placement from LIVE state just before releasing.

    The demo's placement waypoint was derived from the DEMO's own grasp offset, but the
    follower grasps from a lattice position and can end up holding the object with a
    different offset -- blindly chasing the recorded waypoint then leaves it off-centre.
    Measured on BlockStack (1 cm tolerance) that alone failed half the episodes, so steer
    the CARRIED OBJECT (not the TCP) onto the target and set the drop height from the
    object's actual underside.
    """
    spec = tasks.TASKS.get(task, {})
    if not spec.get("carried"):  # tasks whose objects the env places itself
        return
    env = ep.backend.env
    target = tasks.actor_pos(env, spec["target"])

    obj = tasks.actor_pos(env, spec["carried"])
    tcp = ep.backend.tcp_pos()
    carry_off = obj[:2] - tcp[:2]
    ep.chase(np.array([target[0] - carry_off[0], target[1] - carry_off[1], tcp[2]]),
             tol=0.006)

    obj = tasks.actor_pos(env, spec["carried"])
    tcp = ep.backend.tcp_pos()
    ep.chase(np.array([tcp[0], tcp[1], spec["drop_z"](env, target) + (tcp[2] - obj[2])]),
             tol=0.006)


def follow_track(backend, track: dict, out_dir: Path, step_m: float,
                 agentview_square: int | None = None) -> dict:
    tcp = np.asarray(track["tcp"], dtype=np.float64)
    events = [(int(i), str(t)) for i, t in track["events"]]
    bounds = [0] + [i for i, _ in events] + [len(tcp) - 1]
    task = track.get("task_key", "")

    writer = RolloutWriter(out_dir, agentview_square=agentview_square)
    ep = TokenEpisode(backend, writer, AtomicExec(backend, step_m=step_m), max_tokens=160)
    success = False
    try:
        for si in range(len(bounds) - 1):
            a, b = bounds[si], bounds[si + 1]
            if b > a:
                seg = tcp[a:b + 1]
                corners = rdp(seg)
                for k, ci in enumerate(corners[1:], 1):
                    last = si == len(bounds) - 2 and k == len(corners) - 1
                    # tighter tolerance right before a gripper event / at the goal
                    tol = 0.008 if (last or si < len(events)) and k == len(corners) - 1 \
                        else 0.012
                    ep.chase(seg[ci], tol=tol)
            if si < len(events):
                _idx, tok = events[si]
                if tok == RELEASE and track.get("layout"):
                    retarget_place(ep, task)
                ep.emit(tok, "grasp" if tok == GRASP else "release")

        success = ep.settle(16)
        if not success and len(tcp) > 1:
            # One corrective round: PD settle can drift the last cm; re-chase the final
            # waypoint (lattice-limited) and settle again before giving up.
            ep.chase(tcp[-1], tol=0.011)
            success = ep.settle(16)
    except RuntimeError as exc:  # token budget exceeded
        return {"success": False, "steps": writer.step, "reason": str(exc),
                "writer": writer}
    return {"success": success, "steps": writer.step, "reason": "", "writer": writer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True,
                    help="tracks/<task> dir written by record_demos.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", default="", help="discretiser backend; default: the one the "
                                              "track was recorded with (see backends/)")
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--table-tex", default="white",
                    help="RLinf rigs only: tabletop appearance -- 'white' | 'black' | a "
                         "wood texture id '001'..'021'. '006' is the wood used by the "
                         "BlockPAP-v1_Mix reference videos. The texture only matters at "
                         "RENDER time, so the same tracks can be re-rendered on any "
                         "tabletop. A featureless white table leaves the wrist view nearly "
                         "constant near the surface, which starves a wrist-dependent "
                         "policy of signal during fine alignment -- prefer the wood.")
    ap.add_argument("--agentview-square", type=int, default=0,
                    help="store agentview as an N x N letterbox (resize_with_pad) instead "
                         "of the raw 640x480 render; 256 matches the real-robot datasets "
                         "and the deployment runner. 0 = keep the raw render.")
    args = ap.parse_args()

    track_files = sorted(Path(args.tracks).glob("track_ep*.json"))
    if not track_files:
        raise SystemExit(f"no tracks under {args.tracks}")
    first = json.loads(track_files[0].read_text())
    task_key = first.get("task_key", Path(args.tracks).name)
    sim = args.sim or first.get("sim", "maniskill")

    backend = make_backend(sim, **tasks.backend_kwargs(task_key, table_tex=args.table_tex))
    out_root = Path(args.out) / f"{task_key}_follow"
    kept = 0
    for tf in track_files:
        track = json.loads(tf.read_text())
        # The env layout sampling draws from the GLOBAL np.random; seed it before reset.
        np.random.seed(int(track["seed"]))
        backend.reset(int(track["seed"]))
        ex0 = AtomicExec(backend, step_m=args.step_m)
        ex0.hold(8)
        if track.get("layout"):
            # Reproduce the recorded scene exactly (the recorder overrode the env's own
            # sampling), then let the objects settle before following the track.
            tasks.apply_layout(backend.env, track["layout"])
            ex0.hold(6)
        out_dir = out_root / f"rollout_{kept:03d}"
        result = follow_track(backend, track, out_dir, args.step_m,
                              agentview_square=args.agentview_square or None)
        writer = result.pop("writer")
        writer.close({
            "source": f"follow_{task_key}",
            "method": "closed_loop_follower",
            "sim": sim,
            "env_id": track["env_id"],
            "seed": track["seed"],
            "step_m": args.step_m,
            "table_tex": args.table_tex,
            "agentview_square": args.agentview_square or None,
            "task": track["task"],
            **{k: v for k, v in result.items()},
        })
        print(f"[follow:{task_key}] {tf.name} steps={result['steps']} "
              f"success={result['success']} {result['reason']}", flush=True)
        if result["success"]:
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
    print(f"[follow:{task_key}] kept {kept}/{len(track_files)}")
    write_preview(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
