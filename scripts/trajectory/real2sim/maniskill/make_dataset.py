#!/usr/bin/env python3
"""Convert generated real2sim rollout datasets to LlamaFactory MVTOKEN format + stats.

For every dataset directory under ``--root`` (each holding ``rollout_NNN/`` teleop-format
rollouts), this runs the standard converter
(``train/data_preparation/rollouts_to_alpaca.py --version <v> --task <text>``)
producing ``rollout_lite.json`` inside the dataset dir -- the exact training-sample form of
the existing MVTOKEN datasets -- and computes per-dataset stats into ``<root>/stats.json``
(episodes, samples, token histogram, episode length, per-token displacement measured from
the ee_pose deltas, adjacent opposite-token pairs).

The displacement stats are the quality gate for the whole pipeline: a closed-loop dataset
must show ~step_m per token with a sub-millimetre sigma and ZERO adjacent opposite pairs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.atomic_tokenizer import OPPOSITE, TOKEN_AXIS  # noqa: E402

CONVERTER = ROOT / "train" / "data_preparation" / "rollouts_to_alpaca.py"


def rollout_stats(ds_dir: Path) -> dict:
    episodes = sorted(d for d in ds_dir.iterdir()
                      if d.is_dir() and (d / "actions.jsonl").exists())
    tokens: Counter = Counter()
    lengths: list[int] = []
    axis_errs: list[float] = []
    flips = 0
    move_pairs = 0
    task = ""
    step_m = 0.02
    for ep in episodes:
        recs = [json.loads(l) for l in (ep / "actions.jsonl").open()]
        lengths.append(len(recs))
        toks = [r["token"] for r in recs]
        tokens.update(toks)
        flips += sum(1 for a, b in zip(toks, toks[1:]) if OPPOSITE.get(a) == b)
        move_pairs += sum(1 for a, b in zip(toks, toks[1:])
                          if a in TOKEN_AXIS and b in TOKEN_AXIS)
        meta = json.loads((ep / "metadata.json").read_text())
        task = meta.get("task", task)
        step_m = float(meta.get("step_m", step_m))
        # per-token main-axis displacement (only meaningful for executed rollouts)
        for a, b in zip(recs, recs[1:]):
            if a["token"] in TOKEN_AXIS:
                ax, sg = TOKEN_AXIS[a["token"]]
                d = np.array(b["ee_pose"][:3]) - np.array(a["ee_pose"][:3])
                axis_errs.append(float(d[ax] * sg))
    disp = np.asarray(axis_errs) if axis_errs else np.zeros(1)
    # A token whose realised displacement falls far short of step_m means the arm was
    # BLOCKED (fingers in contact with the object/table) while the label still claims a
    # full step -- i.e. a mislabelled frame. A fraction of a percent is normal (the last
    # descent token touching down); a spike means the trajectory source's approach path is
    # not lattice-friendly, e.g. it dives diagonally onto the grasp pose instead of
    # hovering directly above it, so the ±half-step lattice error lands the fingers ON the
    # object. Those episodes usually fail their success check and get discarded anyway,
    # but this number is what tells you WHY the yield dropped.
    blocked = int((disp < 0.75 * step_m).sum()) if axis_errs else 0
    return {
        "episodes": len(episodes),
        "samples": int(sum(lengths)) + len(episodes),  # + synthesized DONE per episode
        "tokens_per_episode_mean": round(float(np.mean(lengths)), 1) if lengths else 0,
        "token_hist": dict(sorted(tokens.items())),
        "per_token_disp_mm_mean": round(float(disp.mean() * 1000), 2),
        "per_token_disp_mm_std": round(float(disp.std() * 1000), 2),
        "tokens_blocked": blocked,
        "tokens_blocked_pct": round(100 * blocked / max(1, len(axis_errs)), 2),
        "adjacent_opposite_pairs": int(flips),
        "adjacent_move_pairs": int(move_pairs),
        "task": task,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    datasets = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name != "tracks"
        and any((c / "actions.jsonl").exists() for c in d.iterdir() if c.is_dir())
    )
    all_stats: dict = {}
    for ds in datasets:
        stats = rollout_stats(ds)
        if not args.skip_convert:
            out_json = ds / "rollout_lite.json"
            cmd = [sys.executable, str(CONVERTER), str(ds),
                   "--version", args.version,
                   "--task", stats["task"],
                   "--output", str(out_json)]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            if r.returncode != 0:
                print(f"[convert] {ds.name} FAILED:\n{r.stderr[-800:]}", flush=True)
            else:
                n = len(json.loads(out_json.read_text()))
                stats["lf_samples"] = n
                print(f"[convert] {ds.name}: {n} samples -> {out_json.name}", flush=True)
        all_stats[ds.name] = stats
    (root / "stats.json").write_text(json.dumps(all_stats, indent=2, sort_keys=True))
    print(json.dumps(all_stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
