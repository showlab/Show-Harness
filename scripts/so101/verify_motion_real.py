"""Quantitative motion check of the SO-101 interpreter on the REAL arm (the hardware
counterpart of scripts/trajectory/real2sim/so101_workshop/verify_motion.py).

Nothing is grasped and nothing should be touched: put the vials and the rack away. The arm
goes to the start pose, then to the begin pose, and runs:

  grid      each MV_* token from the begin pose and back: travel, off-axis, pitch change
  walk      N random feasible MV_* tokens, then the exact inverse sequence: closure error
  rotate    RT_PITCH_FWD / RT_PITCH_BACK pairs: achieved angle, TCP drift while turning

Distances come from forward kinematics on the measured joints, so they check the CONTROL
(does the hand go where the interpreter says), not the absolute calibration. Pass criteria are
the CRITERIA constants below, fixed before running and looser than the sim's (sag, P gain).

    python scripts/so101/verify_motion_real.py --port COM3 --out runs/verify_real
    python scripts/so101/verify_motion_real.py --mock --out /tmp/verify_mock     # no hardware
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.so101.lerobot_io import gripper_temperature  # noqa: E402
from core.so101.rig import DEFAULT_CONFIG, open_rig  # noqa: E402
from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, OPPOSITE  # noqa: E402

# Pass criteria. Set BEFORE looking at results; change only with a written reason.
CRITERIA = {
    "travel_mm": [16.0, 24.0],        # one MV_* token = 20 mm +-20 % (sim: +-10 %)
    "off_axis_mm_max": 4.0,           # sim: 2.0
    "pitch_change_deg_max": 3.0,      # hand pitch change during a translation (sim: 1.0)
    "walk_closure_mm_max": 5.0,       # sim: 2.0
    "rot_step_deg": [8.0, 12.0],      # one RT_PITCH token = 10 deg (sim: 9-11)
    "rot_tcp_drift_mm_max": 5.0,      # sim: 3.0
}


def pitch_deg(be) -> float:
    return float(np.degrees(be.kin.tool_pitch(be.q_obs())))


def run_token(ex, tok):
    """(executed, travel vector m, pitch change deg, blocked reason)."""
    ok, why = ex.check(tok)
    if not ok:
        return False, np.zeros(3), 0.0, why
    be = ex.backend
    p0, a0 = be.tcp_pos(), pitch_deg(be)
    ex.last_blocked = None
    ex.run(tok)
    return True, be.tcp_pos() - p0, pitch_deg(be) - a0, ex.last_blocked


def grid(ex) -> dict:
    rows = []
    for tok in ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"):
        for t in (tok, OPPOSITE[tok]):
            done, d, dp, why = run_token(ex, t)
            u = np.asarray(MOVE_DIRS[t], dtype=np.float64)
            along = float(d @ u)
            rows.append({"token": t, "executed": done, "blocked": why,
                         "travel_mm": round(1000 * along, 2),
                         "off_axis_mm": round(1000 * float(np.linalg.norm(d - along * u)), 2),
                         "pitch_change_deg": round(dp, 2)})
            print(f"[grid] {t:9s} {rows[-1]}", flush=True)
    ok = [r for r in rows if r["executed"] and not r["blocked"]]
    return {"tokens": rows,
            "travel_mm": [min(r["travel_mm"] for r in ok), max(r["travel_mm"] for r in ok)] if ok else None,
            "off_axis_max_mm": max((r["off_axis_mm"] for r in ok), default=None),
            "pitch_change_max_deg": max((abs(r["pitch_change_deg"]) for r in ok), default=None),
            "executed": len(ok), "refused_or_blocked": len(rows) - len(ok)}


def walk(ex, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    be = ex.backend
    start, done = be.tcp_pos(), []
    toks = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"]
    while len(done) < n:
        t = toks[int(rng.integers(len(toks)))]
        ok, _, _, why = run_token(ex, t)
        if ok and not why:
            done.append(t)
    for t in reversed(done):
        run_token(ex, OPPOSITE[t])
    err = float(np.linalg.norm(be.tcp_pos() - start))
    print(f"[walk] {n} tokens out and back: closure {1000 * err:.2f} mm", flush=True)
    return {"tokens": n, "return_error_mm": round(1000 * err, 2)}


def rotate(ex, pairs: int) -> dict:
    rows = []
    be = ex.backend
    for _ in range(pairs):
        for t in ("RT_PITCH_FWD", "RT_PITCH_BACK"):
            p0, a0 = be.tcp_pos(), pitch_deg(be)
            ok, why = ex.check(t)
            if not ok:
                rows.append({"token": t, "executed": False, "blocked": why})
                continue
            ex.last_blocked = None
            ex.run(t)
            rows.append({"token": t, "executed": True, "blocked": ex.last_blocked,
                         "angle_deg": round(abs(pitch_deg(be) - a0), 2),
                         "tcp_drift_mm": round(1000 * float(np.linalg.norm(be.tcp_pos() - p0)), 2)})
            print(f"[rotate] {rows[-1]}", flush=True)
    ok = [r for r in rows if r["executed"] and not r["blocked"]]
    return {"tokens": rows,
            "angle_deg": [min(r["angle_deg"] for r in ok), max(r["angle_deg"] for r in ok)] if ok else None,
            "tcp_drift_max_mm": max((r["tcp_drift_mm"] for r in ok), default=None)}


def verdict(rep: dict) -> dict:
    c, v = CRITERIA, {}
    g = rep.get("grid")
    if g and g["travel_mm"]:
        v["travel"] = c["travel_mm"][0] <= g["travel_mm"][0] and g["travel_mm"][1] <= c["travel_mm"][1]
        v["off_axis"] = g["off_axis_max_mm"] <= c["off_axis_mm_max"]
        v["pitch_change"] = g["pitch_change_max_deg"] <= c["pitch_change_deg_max"]
    if rep.get("walk"):
        v["walk_closure"] = rep["walk"]["return_error_mm"] <= c["walk_closure_mm_max"]
    r = rep.get("rotate")
    if r and r["angle_deg"]:
        v["rot_angle"] = c["rot_step_deg"][0] <= r["angle_deg"][0] and r["angle_deg"][1] <= c["rot_step_deg"][1]
        v["rot_drift"] = r["tcp_drift_max_mm"] <= c["rot_tcp_drift_mm_max"]
    v["PASS"] = bool(v) and all(v.values())
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--port", default=None, help="override robot.port")
    ap.add_argument("--joint_offset", default=None, help="override robot.joint_offset")
    ap.add_argument("--sections", default="grid,walk,rotate")
    ap.add_argument("--walk_tokens", type=int, default=20)
    ap.add_argument("--rot_pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true", help="MockFollower instead of the arm")
    args = ap.parse_args()

    rig = open_rig(args.config, port=args.port, mock=args.mock, joint_offset=args.joint_offset,
                   control_hz=1000.0 if args.mock else 30.0)
    rep = {"criteria": CRITERIA, "config": args.config, "mock": args.mock, "time": time.strftime("%F %T")}
    sections = args.sections.split(",")
    try:
        if not rig.go_begin():
            print("[verify] begin pose not reached (stopping short is tolerated, continuing)", flush=True)
        rep["begin_tcp_m"] = rig.backend.tcp_pos().round(4).tolist()
        if "grid" in sections:
            rep["grid"] = grid(rig.ex)
        if "walk" in sections:
            rep["walk"] = walk(rig.ex, args.walk_tokens, args.seed)
        if "rotate" in sections:
            rep["rotate"] = rotate(rig.ex, args.rot_pairs)
        rep["gripper_temp_c"] = gripper_temperature(rig.robot)
    except KeyboardInterrupt:
        print("\n[verify] interrupted: holding", flush=True)
        rig.arm.hold()
    finally:
        rep["verdict"] = verdict(rep)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "motion_report.json").write_text(json.dumps(rep, indent=1))
        print(f"[verify] {rep['verdict']}  -> {out / 'motion_report.json'}", flush=True)
        if not args.mock:
            input("[verify] the arm goes back to the start pose and then goes limp: support it and press Enter")
            rig.arm.move_to(rig.start_q, duration_s=4.0)
        rig.close()
    return 0 if rep["verdict"].get("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
