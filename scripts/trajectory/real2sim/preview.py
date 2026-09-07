"""Per-dataset preview rendering for real2sim rollouts (sim-agnostic).

Renders one episode's agentview + wrist frames into ``<dataset>/preview.mp4``
with the step/token/gripper state stamped on each frame -- the quickest way to
eyeball whether a generated dataset's camera transforms and action labels line
up before converting it for training. Used by the ManiSkill and RoboLab
generators after each dataset finishes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_recs(ep_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (ep_dir / "actions.jsonl").open()]


def episode_dirs(ds_dir: Path) -> list[Path]:
    return sorted(d for d in ds_dir.iterdir()
                  if d.is_dir() and (d / "actions.jsonl").exists())


def make_preview_mp4(ds_dir: Path, fps: float = 3.0) -> Path | None:
    import imageio.v2 as iio

    eps = episode_dirs(ds_dir)
    if not eps:
        return None
    ep = eps[0]
    recs = load_recs(ep)
    font = _font(20)
    frames = []
    for r in recs:
        av = Image.open(ep / r["agentview"]).convert("RGB")
        wr = Image.open(ep / r["wrist"]).convert("RGB")
        h = 360
        # Scale BOTH views by height and keep their aspect. The wrist used to be forced
        # to a square, which is right for ManiSkill (256x256 hand_camera) but stretches
        # RoboLab's 960x720 wrist -- and a distorted preview is exactly the thing you
        # would use to judge whether the wrist flip/crop is correct.
        av = av.resize((int(av.width * h / av.height), h), Image.BILINEAR)
        wr = wr.resize((int(wr.width * h / wr.height), h), Image.BILINEAR)
        bar = 44
        canvas = Image.new("RGB", (av.width + wr.width, h + bar), (16, 18, 22))
        canvas.paste(av, (0, bar))
        canvas.paste(wr, (av.width, bar))
        d = ImageDraw.Draw(canvas)
        grip = "CLOSED" if r["gripper_closed"] else "OPEN"
        d.text((10, 10), f"#{r['step']:03d}  {r['token']:<9}  gripper={grip}  "
                         f"w={r['gripper_width']:.3f}", fill=(240, 240, 240), font=font)
        frames.append(np.asarray(canvas))
    # freeze the last frame a moment so the final state is readable
    frames.extend([frames[-1]] * int(fps))
    out = ds_dir / "preview.mp4"
    iio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7,
                 macro_block_size=1)
    return out
