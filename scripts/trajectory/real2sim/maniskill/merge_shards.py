#!/usr/bin/env python3
"""Merge sharded oracle runs into one dataset directory with sequential rollout numbering.

Generation is parallelised by running N generator processes (``oracle.py`` /
``follow_tokenize.py``) over disjoint seed ranges, each numbering its episodes from
``rollout_000``. This concatenates the shards into
a single dataset, renumbering as it goes (and recording the source shard + original seed in
each episode's metadata so any rollout stays traceable).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (faster, consumes the shards)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for shard in args.shards:
        sd = Path(shard)
        eps = sorted(d for d in sd.iterdir()
                     if d.is_dir() and (d / "actions.jsonl").exists())
        for ep in eps:
            dst = out / f"rollout_{n:03d}"
            if dst.exists():
                shutil.rmtree(dst)
            (shutil.move if args.move else shutil.copytree)(str(ep), str(dst))
            meta_path = dst / "metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["shard"] = sd.name
            meta["shard_rollout"] = ep.name
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            n += 1
        print(f"[merge] {sd.name}: {len(eps)} episodes", flush=True)
    print(f"[merge] total {n} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
