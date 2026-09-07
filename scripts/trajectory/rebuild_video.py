#!/usr/bin/env python3
"""Rebuild a rollout video from the saved PNG frames.

A live rollout writes every step's frame to ``<run>/images/agentview/NNNN.png`` (and
``<run>/images/wrist/NNNN.png``) but only muxes the ``.mp4`` once, at the very end. If
the run is interrupted (Ctrl+C) or the encoder is killed mid-write, the ``.mp4`` can be
left truncated/empty even though every frame is safely on disk. This script re-encodes a
playable video straight from those frames.

By default it reproduces the SAME annotated "analysis" video the logger makes (a header
with step / stage / gripper / reasoning above the agentview+wrist panels), reading the
per-step records from ``steps.json``. Use ``--view`` for a plainer layout.

Examples
--------
    # Rebuild the annotated video for one run (writes <run>/rollout_rebuilt.mp4):
    python scripts/trajectory/rebuild_video.py rollouts/real/Gemma-CoT/0620/task_0/16-14-09

    # Just stitch the raw agentview+wrist side by side at 5 fps:
    python scripts/trajectory/rebuild_video.py <run> --view side --fps 5

    # Overwrite the broken original instead of writing a new file:
    python scripts/trajectory/rebuild_video.py <run> --output <run>/rollout_failure.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# Make the repo importable when run as a standalone script (scripts/ is not a package).
# File is scripts/trajectory/rebuild_video.py, so repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.record.episode_logger import _make_analysis_frame  # reuse the logger's renderer
from core.record.images import save_mp4, to_uint8_hwc

FPS_DEFAULT = 2.0
FPS_MIN, FPS_MAX = 0.5, 30.0


def _resolve_dirs(path: Path) -> tuple[Path, Path]:
    """Map any of {run dir, images dir, images/agentview dir} to (run_dir, images_dir)."""
    path = path.resolve()
    if (path / "images" / "agentview").is_dir():
        return path, path / "images"
    if path.name == "images" and (path / "agentview").is_dir():
        return path.parent, path
    if path.name == "agentview" and path.is_dir():
        return path.parent.parent, path.parent
    raise SystemExit(
        f"Could not find frames under '{path}'. Expected a run directory containing "
        "images/agentview/*.png (or the images/ or images/agentview/ directory itself)."
    )


def _frames(view_dir: Path) -> list[Path]:
    return sorted(view_dir.glob("*.png"), key=lambda p: p.stem)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _records_by_index(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Map step index (record['i']) -> full record, from steps.json (fallback steps.jsonl)."""
    steps_json = run_dir / "steps.json"
    if steps_json.exists():
        try:
            records = json.loads(steps_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = []
        out: dict[int, dict[str, Any]] = {}
        for pos, rec in enumerate(records):
            if isinstance(rec, dict):
                out[int(rec.get("i", pos))] = rec
        if out:
            return out
    jsonl = run_dir / "steps.jsonl"
    out = {}
    if jsonl.exists():
        for pos, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(rec.get("i", pos))] = rec
    return out


def _auto_fps(run_dir: Path) -> float:
    for name in ("summary.json", "metadata.json"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = data.get("video_fps")
        if value is None and isinstance(data.get("config"), dict):
            value = data["config"].get("video_fps")
        if value is not None:
            return float(value)
    return FPS_DEFAULT


def _even(img: np.ndarray) -> np.ndarray:
    """Crop to even height/width so h.264 (which rejects odd dimensions) can encode it."""
    img = to_uint8_hwc(img)
    h, w = img.shape[:2]
    return np.ascontiguousarray(img[: h - (h % 2), : w - (w % 2)])


def _side_by_side(agent: np.ndarray, wrist: Optional[np.ndarray]) -> np.ndarray:
    agent = to_uint8_hwc(agent)
    if wrist is None:
        return _even(agent)
    wrist = to_uint8_hwc(wrist)
    if wrist.shape[0] != agent.shape[0]:  # match heights before stacking
        scale = agent.shape[0] / wrist.shape[0]
        wrist = np.asarray(
            Image.fromarray(wrist).resize(
                (max(1, round(wrist.shape[1] * scale)), agent.shape[0]),
                Image.Resampling.BILINEAR,
            )
        )
    return _even(np.concatenate([agent, wrist], axis=1))


def build_frames(view: str, run_dir: Path, images_dir: Path) -> list[np.ndarray]:
    agent_dir = images_dir / "agentview"
    wrist_dir = images_dir / "wrist"
    agent_paths = _frames(agent_dir)
    if not agent_paths:
        raise SystemExit(f"No agentview frames found in {agent_dir}")
    wrist_paths = {p.stem: p for p in _frames(wrist_dir)} if wrist_dir.is_dir() else {}

    if view == "wrist":
        if not wrist_paths:
            raise SystemExit(f"No wrist frames found in {wrist_dir}")
        return [_even(_load(p)) for _, p in sorted(wrist_paths.items())]
    if view == "agentview":
        return [_even(_load(p)) for p in agent_paths]
    if view == "side":
        frames = []
        for ap in agent_paths:
            wp = wrist_paths.get(ap.stem)
            frames.append(_side_by_side(_load(ap), _load(wp) if wp else None))
        return frames

    # annotated (default): reproduce the logger's analysis frame using steps.json records.
    records = _records_by_index(run_dir)
    if not records:
        print("  [warn] no steps.json/steps.jsonl records found; headers will be blank.")
    frames = []
    for ap in agent_paths:
        wp = wrist_paths.get(ap.stem)
        try:
            idx = int(ap.stem)
        except ValueError:
            idx = -1
        record = records.get(idx, {"i": idx if idx >= 0 else ap.stem})
        frames.append(
            _make_analysis_frame(
                agentview=_load(ap),
                wrist=_load(wp) if wp else None,
                record=record,
            )
        )
    return frames


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild a rollout video from its saved PNG frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory (containing images/), or the images/ or images/agentview/ dir.",
    )
    parser.add_argument(
        "--view",
        choices=("annotated", "side", "agentview", "wrist"),
        default="annotated",
        help="annotated = logger-style header + agentview/wrist (default); "
        "side = raw agentview|wrist; agentview/wrist = that single view.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"Frames per second (default: auto from summary/metadata, else {FPS_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 path (default: <run>/rollout_rebuilt.mp4).",
    )
    args = parser.parse_args(argv)

    run_dir, images_dir = _resolve_dirs(args.path)
    fps = args.fps if args.fps is not None else _auto_fps(run_dir)
    fps = min(FPS_MAX, max(FPS_MIN, float(fps)))
    output = args.output or (run_dir / "rollout_rebuilt.mp4")

    print(f"run:    {run_dir}")
    print(f"view:   {args.view}")
    frames = build_frames(args.view, run_dir, images_dir)
    print(f"frames: {len(frames)} @ {fps:g} fps -> {frames[0].shape[1]}x{frames[0].shape[0]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_mp4(output, frames, fps=fps)
    size = output.stat().st_size if output.exists() else 0
    if size < 1024:
        print(f"  [warn] output is only {size} bytes -- encoding may have failed.")
    print(f"wrote:  {output} ({size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
