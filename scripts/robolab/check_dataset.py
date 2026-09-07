#!/usr/bin/env python3
"""Contract check over a generated RoboLab dataset directory.

Runs on plain Python (no Isaac Sim, no RoboLab venv) over ``<root>/<Task>/rollout_*/``.
Checks the things that have actually gone wrong on this pipeline and that no aggregate
statistic reveals on its own:

* **the episode must not end on RELEASE** -- the converter builds the terminal DONE sample
  from the last frame, so an episode ending on RELEASE trains one image to mean both
  "open the fingers" and "the task is over";
* **the post-release retreat must have moved** -- emitting MV_UP into a frozen scene
  produces frames labelled with a 2 cm move over 0.00 mm of travel;
* **no token may be recorded over a frame that did not change** -- same failure, anywhere
  in the episode, from a blocked planner rather than a frozen env;
* the gripper must actually be holding something between GRASP and RELEASE.

It is a gate, not a report: exit code 1 means do not train on this.

    python3 scripts/robolab/check_dataset.py rollouts/robolab/GEN2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# A recorded MV_* token that moved less than this is a lie about the physics. Well under
# one 2 cm token (the executor's own stall guard uses 0.3x) and well over solver noise.
MIN_TOKEN_TRAVEL_M = 0.006
# Fraction of an episode's move tokens allowed to be stalled before the episode is
# rejected. Matches ``oracle.py --max-stalled-frac``, so what the generator discards and
# what this gate rejects are the same rule. Not zero: a single blocked token at the end of
# a pursuit is the planner's stall guard working as designed, and rejecting whole episodes
# for one of those throws away good data. A CLUSTER is the real defect -- an arm wedged
# against scene furniture logs a fifth of its trajectory as moves it never made.
MAX_STALLED_FRAC = 0.05
# Finger opening (joint sum) below which a gripper commanded CLOSED is holding nothing.
GRASPED_WIDTH_M = 0.005


def check_episode(ep: Path) -> list[str]:
    problems: list[str] = []
    recs = [json.loads(line) for line in (ep / "actions.jsonl").open() if line.strip()]
    if not recs:
        return [f"{ep}: empty actions.jsonl"]
    meta_path = ep / "metadata.json"
    if not meta_path.exists():
        # Written last, so its absence means this episode is still being generated.
        # Reported as SKIPPED rather than failed so the gate can be run mid-batch.
        return ["__incomplete__"]
    meta = json.loads(meta_path.read_text())
    tokens = [r["token"] for r in recs]

    if tokens[-1] != "MV_UP":
        problems.append(f"{ep}: ends on {tokens[-1]}, expected MV_UP")
    if "RELEASE" not in tokens:
        problems.append(f"{ep}: never released")
    else:
        after = tokens[tokens.index("RELEASE") + 1:]
        if len(after) < 2:
            problems.append(f"{ep}: only {len(after)} token(s) after RELEASE, expected >= 2")
    travel = meta.get("retreat_travel_m") or []
    stalled = [round(t, 5) for t in travel if t < MIN_TOKEN_TRAVEL_M]
    if stalled:
        problems.append(f"{ep}: retreat token(s) barely moved: {stalled} m")

    # Frame-to-frame travel for every recorded move. ee_pose is the pose BEFORE the token,
    # so token i's displacement is pose[i+1] - pose[i].
    blocked = moves = 0
    for cur, nxt in zip(recs, recs[1:]):
        if cur["kind"] != "move":
            continue
        moves += 1
        d = sum((a - b) ** 2 for a, b in zip(cur["ee_pose"][:3], nxt["ee_pose"][:3])) ** 0.5
        if d < MIN_TOKEN_TRAVEL_M:
            blocked += 1
    if moves and blocked / moves > MAX_STALLED_FRAC:
        problems.append(f"{ep}: {blocked}/{moves} move tokens ({100*blocked/moves:.0f}%) "
                        f"over < {MIN_TOKEN_TRAVEL_M * 1000:.0f} mm of travel")

    # Between the LAST grasp and the LAST release the fingers must be on something.
    #
    # The last one, not the first: the oracle retries an empty grasp up to three times
    # (GRASP -> width 0 -> RELEASE -> MV_DOWN -> GRASP), so an episode that needed a retry
    # legitimately contains a closed-and-empty frame. Anchoring on the first GRASP reported
    # those retries as defects -- it flagged two healthy episodes in the 2026-08-12 batch.
    if "GRASP" in tokens and "RELEASE" in tokens:
        g = len(tokens) - 1 - tokens[::-1].index("GRASP")
        r = len(tokens) - 1 - tokens[::-1].index("RELEASE")
        if g < r:
            carried = [rec for rec in recs[g + 1:r + 1] if rec["gripper_closed"]]
            empty = [rec["step"] for rec in carried
                     if rec["gripper_width"] <= GRASPED_WIDTH_M]
            if empty:
                problems.append(f"{ep}: gripper closed on nothing at steps {empty[:5]}"
                                f"{'...' if len(empty) > 5 else ''}")
    if not meta.get("success"):
        problems.append(f"{ep}: metadata says success=false ({meta.get('reason')})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = ap.parse_args()

    root = Path(args.root)
    tasks = sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("rollout_*")))
    if not tasks:
        print(f"no task directories with rollouts under {root}")
        return 1

    all_problems: list[str] = []
    total = skipped = 0
    for task in tasks:
        eps = sorted(task.glob("rollout_*"))
        problems: list[str] = []
        endings: Counter = Counter()
        lengths = []
        n_skip = 0
        for ep in eps:
            if not (ep / "actions.jsonl").exists():
                continue
            found = check_episode(ep)
            if found == ["__incomplete__"]:
                n_skip += 1
                continue
            total += 1
            recs = [json.loads(x) for x in (ep / "actions.jsonl").open() if x.strip()]
            if recs:
                endings[recs[-1]["token"]] += 1
                lengths.append(len(recs))
            problems.extend(found)
        skipped += n_skip
        preview = task / "preview.mp4"
        size = preview.stat().st_size if preview.exists() else 0
        flag = "OK " if not problems else "BAD"
        span = (f"tokens {min(lengths)}-{max(lengths)} (mean {sum(lengths)/len(lengths):.1f})"
                if lengths else "no complete episodes")
        print(f"[{flag}] {task.name:26s} {len(lengths):3d} eps  {span}  "
              f"endings={dict(endings)}  preview={size//1000} KB"
              f"{f'  (+{n_skip} still generating)' if n_skip else ''}")
        if problems and not args.quiet:
            for p in problems[:10]:
                print(f"        {p}")
            if len(problems) > 10:
                print(f"        ... and {len(problems) - 10} more")
        all_problems.extend(problems)

    print()
    if skipped:
        print(f"({skipped} episode(s) skipped -- still being generated)")
    if all_problems:
        print(f"FAILED: {len(all_problems)} problem(s) across {total} episodes")
        return 1
    print(f"PASSED: {total} episodes across {len(tasks)} tasks satisfy the contract")
    print("This checks the DATA. Now watch the videos -- every defect that has cost real "
          "time here passed checks like these and was only visible by looking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
