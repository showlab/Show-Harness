#!/usr/bin/env python3
"""Per-step latency decomposition of recorded real rollouts.

Reads the timing fields the MVTOKEN runner writes into each rollout's
steps.jsonl (``ts`` = wall-clock step start, ``t_obs_ms`` camera read,
``t_decide_ms`` decision incl. the VLM call, ``t_exec_ms`` arm motion) and
reports, per rollout and aggregated per served model:

  period   = ts[i+1] - ts[i]          the true action-to-action cycle
  gap      = period - t_exec_ms[i]    everything BETWEEN two executions
             (camera + decision + logging/live-view overhead)

Rollouts recorded before the timing fields existed are listed with their
``vlm_ms`` only. Point it at rollout dirs or any parent to scan:

    python scripts/trajectory/step_timing.py rollouts/real/MVTOKEN/0904
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stats(xs: list[float]) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    return (f"mean {statistics.mean(xs):7.0f}  median {statistics.median(xs):7.0f}  "
            f"p90 {xs[round(0.9 * (len(xs) - 1))]:7.0f}  n={len(xs)}")


def _model_of(run_dir: Path) -> str:
    try:
        meta = json.loads((run_dir / "metadata.json").read_text())
        return str(meta.get("robot_config", {}).get("vlm", {}).get("model", "?"))
    except Exception:  # noqa: BLE001 - label only
        return "?"


def analyze(run_dir: Path):
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").open()]
    timed = [s for s in steps if "ts" in s and "t_exec_ms" in s]
    out = {
        "model": _model_of(run_dir),
        "vlm_ms": [float(s["vlm_ms"]) for s in steps if "vlm_ms" in s],
        "obs": [float(s["t_obs_ms"]) for s in timed],
        "decide": [float(s["t_decide_ms"]) for s in timed],
        "exec": [float(s["t_exec_ms"]) for s in timed],
        "period": [], "gap": [],
    }
    for a, b in zip(timed, timed[1:]):
        period = (b["ts"] - a["ts"]) * 1000.0
        if 0 < period < 600_000:  # a pause at the operator gate is not a step
            out["period"].append(period)
            out["gap"].append(period - float(a["t_exec_ms"]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="rollout dirs, or parents to scan")
    args = p.parse_args(argv)

    runs = []
    for path in map(Path, args.paths):
        runs += sorted(d.parent for d in path.rglob("steps.jsonl"))
    if not runs:
        print("no steps.jsonl found under the given paths")
        return 1

    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        r = analyze(run)
        print(f"\n{run}  (model {r['model']})")
        if not r["exec"]:
            print(f"  no timing fields (pre-instrumentation rollout); vlm_ms: {_stats(r['vlm_ms'])}")
            continue
        for key, label in (("period", "period ms"), ("gap", "gap ms"), ("obs", "camera ms"),
                           ("decide", "decide ms"), ("vlm_ms", "vlm ms"), ("exec", "motion ms")):
            print(f"  {label:10} {_stats(r[key])}")
            by_model[r["model"]][key] += r[key]

    if len(runs) > 1:
        print("\n=== aggregate per model ===")
        for model, agg in sorted(by_model.items()):
            print(f"\n{model}")
            for key, label in (("period", "period ms"), ("gap", "gap ms"), ("obs", "camera ms"),
                               ("decide", "decide ms"), ("vlm_ms", "vlm ms"), ("exec", "motion ms")):
                print(f"  {label:10} {_stats(agg[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
