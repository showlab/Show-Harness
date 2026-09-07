from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import StreamingVideoWriter, save_png, to_uint8_hwc


JSONL_REASON_LIMIT = 1500
_SENSITIVE_METADATA_KEYS = {
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "client_secret",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _redact_sensitive_metadata(value: Any) -> Any:
    """Return metadata with credential values removed recursively.

    Environment-variable *names* such as ``api_key_env`` remain useful for
    reproduction and are not credentials themselves.
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SENSITIVE_METADATA_KEYS:
                redacted[name] = "[REDACTED]"
            else:
                redacted[name] = _redact_sensitive_metadata(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive_metadata(item) for item in value]
    return value


def _truncate_record_reasoning(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the step record with each VLM reasoning string capped for the
    compact steps.jsonl. The full text is preserved separately in steps.json."""
    vlm = record.get("vlm")
    if not isinstance(vlm, dict):
        return record
    truncated_vlm: dict[str, Any] = {}
    for role, entry in vlm.items():
        if isinstance(entry, dict):
            entry = {
                key: (
                    value[:JSONL_REASON_LIMIT] + "..."
                    if key in ("why", "reasoning", "raw")
                    and isinstance(value, str)
                    and len(value) > JSONL_REASON_LIMIT
                    else value
                )
                for key, value in entry.items()
            }
        truncated_vlm[role] = entry
    out = dict(record)
    out["vlm"] = truncated_vlm
    return out


class EpisodeLogger:
    def __init__(
        self,
        root_dir: str | Path,
        task_id: int,
        variant: str | None = None,
        video_fps: float = 2.0,
    ) -> None:
        now = datetime.now(timezone(timedelta(hours=8)))
        date_stamp = now.strftime("%m%d")
        time_stamp = now.strftime("%H-%M-%S")
        # Group rollouts by model/CoT variant first (e.g. Gemma-CoT), then by date,
        # task, and run time.
        base_dir = Path(root_dir)
        if variant:
            base_dir = base_dir / variant
        self.run_dir = base_dir / date_stamp / f"task_{task_id}" / time_stamp
        self.agentview_dir = self.run_dir / "images" / "agentview"
        self.wrist_dir = self.run_dir / "images" / "wrist"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.run_dir / "steps.jsonl"
        self._steps_file = self._steps_path.open("w", encoding="utf-8")
        self._steps_json_path = self.run_dir / "steps.json"
        # Full per-step records with the complete (un-truncated) reasoning, dumped to
        # steps.json at close; steps.jsonl keeps the compact, reasoning-truncated form.
        self._full_records: list[dict[str, Any]] = []
        # Steps whose frames have been written to images/ (see log_step). The controller
        # prompt for step N is saved BEFORE step N's log_step runs, so the current frame is
        # legitimately still missing at that moment -- this set tells the two cases apart.
        self._logged_steps: set[int] = set()
        self.debug_dir = self.run_dir / "debug_payloads"
        # Analysis video: frames STREAM to disk as they are logged (crash-safe
        # fragmented mp4 -- a force-killed run leaves rollout_live.mp4 playable);
        # close() renames/remuxes it to rollout_success/failure.mp4.
        self.video = StreamingVideoWriter(
            self.run_dir / "rollout_live.mp4",
            fps=min(30.0, max(0.5, float(video_fps))),
        )

    def write_metadata(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                _jsonable(_redact_sensitive_metadata(data)), f, indent=2, sort_keys=True
            )

    def write_calibration(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "calibration.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def write_summary(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def write_plan(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "subgoals.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def write_planner_diagnostics(self, data: dict[str, Any]) -> None:
        """Persist planner routes/errors without mixing them into the plan contract."""
        with (self.run_dir / "planner_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def save_planner_prompt(self, attempt: int, prompt: str) -> None:
        prompts_dir = self.run_dir / "planner_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"attempt_{int(attempt):02d}.txt").write_text(
            str(prompt), encoding="utf-8"
        )

    def log_step(
        self,
        step_idx: int,
        agentview: np.ndarray,
        wrist: Optional[Any],
        record: dict[str, Any],
    ) -> None:
        """``wrist`` is one frame (single-arm), or a [left, right] pair (dual-arm) --
        the pair is stored side by side and rendered as separate analysis panels."""
        agentview_path = self.agentview_dir / f"{step_idx:04d}.png"
        save_png(agentview_path, agentview)
        record = dict(record)
        if wrist is not None:
            wrist_path = self.wrist_dir / f"{step_idx:04d}.png"
            stored = (
                np.concatenate([np.asarray(w) for w in wrist], axis=1)
                if isinstance(wrist, (list, tuple))
                else wrist
            )
            save_png(wrist_path, stored)
        self._logged_steps.add(int(step_idx))
        # steps.json keeps the full reasoning; steps.jsonl gets a reasoning-truncated copy.
        self._full_records.append(_jsonable(record))
        self._steps_file.write(
            json.dumps(_jsonable(_truncate_record_reasoning(record)), separators=(",", ":"))
            + "\n"
        )
        self._steps_file.flush()
        self.video.append(
            _make_analysis_frame(agentview=agentview, wrist=wrist, record=record)
        )

    def log_debug_payload(self, step_idx: int, payload: dict[str, Any]) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        with (self.debug_dir / f"{step_idx:04d}.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, sort_keys=True)

    def save_controller_prompt(
        self, step_idx: int, prompt: str, media: Optional[list[dict]] = None
    ) -> None:
        """Dump the exact controller request sent to the VLM at this step, for offline analysis.

        ``media`` is the role's ``last_media`` -- the media parts in WIRE ORDER (see
        ``core.record.images.image_manifest``). It is rendered into the header so the file
        states which frames were actually sent, rather than assuming a fixed camera set: a
        single-arm request sends agentview+wrist, a dual-arm one sends three views.

        Each frame carries a pixel digest, which is re-checked against the PNG saved under
        ``images/`` -- so the log proves the frame on disk IS the frame the model saw
        (``verified`` / ``MISMATCH`` / ``no file``) rather than merely naming a plausible path.
        """
        prompts_dir = self.run_dir / "controller_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# step {step_idx:04d} controller request"]
        if media:
            lines += self._render_media_header(step_idx, media)
        else:
            lines.append("# media: not reported by this controller (text-only record)")
        lines += ["#", "# ---- user message text (the part after the media above) ----", ""]
        (prompts_dir / f"{step_idx:04d}.txt").write_text(
            "\n".join(lines) + str(prompt), encoding="utf-8"
        )

    def _render_media_header(self, step_idx: int, media: list[dict]) -> list[str]:
        """Render the media manifest as the prompt file's header. See save_controller_prompt."""
        n_img = sum(1 for m in media if m.get("part_type") == "image_url")
        n_vid = sum(1 for m in media if m.get("part_type") == "video_url")
        shape = ", ".join(
            part for part, count in (("<image>", n_img), ("<video>", n_vid)) if count
        )
        lines = [
            "#",
            "# OpenAI content array actually sent (array order == placeholder order;",
            f"#   media first, then ONE text part). {len(media)} media part(s): {shape}",
            "#",
        ]
        for entry in media:
            head = (
                f"#   [{entry['slot']}] {entry['part_type']:<10} {entry['placeholder']:<8} "
                f"{entry.get('camera', '?'):<10}"
            )
            if entry.get("part_type") == "video_url":
                frames = entry.get("frames") or []
                lines.append(f"{head}  {len(frames)} frame(s), {entry.get('encoding', '')}")
                for frame in frames:
                    lines.append("#        " + self._render_frame(step_idx, entry, frame))
            else:
                lines.append(head + "  " + self._render_frame(step_idx, entry, entry))
        lines.append(f"#   [{len(media)}] text        <- the prompt body below")
        return lines

    # camera name -> (directory attribute, note). A note means the saved PNG is NOT a
    # byte-for-byte copy of what was sent, so the digest cannot be checked against it: the
    # dual-arm logger stores the two wrists side by side in ONE png (see log_step), while the
    # request sends them as two separate images.
    _CAMERA_SOURCES: dict[str, tuple[str, str]] = {
        "agentview": ("agentview_dir", ""),
        "wrist": ("wrist_dir", ""),
        "wrist_left": ("wrist_dir", "left half of the side-by-side png"),
        "wrist_right": ("wrist_dir", "right half of the side-by-side png"),
    }

    def _render_frame(self, step_idx: int, entry: dict, frame: dict) -> str:
        """One frame line: which step it came from, its digest, and whether the saved PNG matches."""
        t_offset = int(frame.get("t_offset", 0))
        src_step = step_idx + t_offset
        camera = entry.get("camera", "")
        attr, note = self._CAMERA_SOURCES.get(camera, ("wrist_dir", "unknown camera"))
        path = getattr(self, attr) / f"{src_step:04d}.png"
        rel = path.relative_to(self.run_dir)
        label = "t" if t_offset == 0 else f"t{t_offset:+d}"
        digest = frame.get("sha1", "?")
        shape = "x".join(str(d) for d in frame.get("shape", []))
        # With a note the stored png is a composite, so a digest comparison would always
        # "fail" for a reason that is not a defect -- say what it is instead of crying wolf.
        if note:
            status = note
        elif not path.exists() and t_offset == 0 and src_step not in self._logged_steps:
            # The runner saves this prompt mid-step, before log_step writes the frame. Not a
            # missing file -- it simply does not exist YET. (It stays missing only if the step
            # ends early, e.g. a DONE token breaks the loop before log_step.) Only the CURRENT
            # frame can be pending; a past frame that is absent is genuinely absent.
            status = "pending: written later in this step"
        else:
            status = self._verify_saved_frame(path, digest)
        return f"{label:<5} sha1={digest}  {shape:<12} {rel}  [{status}]"

    @staticmethod
    def _verify_saved_frame(path: Path, digest: str) -> str:
        """Compare the saved PNG's pixels against the digest the role recorded at request time.

        Hashes the decoded uint8 HWC bytes, the same canonical form the role hashed, so a match
        means the two are the same pixels (PNG is lossless, so a round trip is bit-exact). A
        MISMATCH is a real finding -- it means images/ does not hold what the model was shown.
        """
        if not path.exists():
            return "no file"
        try:
            import hashlib  # noqa: PLC0415

            saved = to_uint8_hwc(np.asarray(Image.open(path).convert("RGB")))
            actual = hashlib.sha1(saved.tobytes()).hexdigest()[:12]
        except Exception as exc:  # noqa: BLE001 - logging must never break a rollout
            return f"unverified: {type(exc).__name__}"
        return "verified" if actual == digest else f"MISMATCH saved={actual}"

    def close(self, success: bool, fps: float) -> Path:
        self._steps_file.close()
        with self._steps_json_path.open("w", encoding="utf-8") as f:
            json.dump(self._full_records, f, indent=2, ensure_ascii=False)
        video_path = self.run_dir / (
            "rollout_success.mp4" if success else "rollout_failure.mp4"
        )
        # Frames were already streamed at the writer's fps (fixed at open time);
        # a differing late fps request cannot re-time them, only note it.
        if self.video.frame_count and abs(float(fps) - self.video.fps) > 1e-6:
            print(
                f"[logger] note: video was streamed at {self.video.fps:g} fps "
                f"(close requested {float(fps):g})."
            )
        self.video.close(final_path=video_path)
        return video_path


# -- analysis-frame design ----------------------------------------------------------
# One video frame = header (step + telemetry) + reasoning strip + labeled camera
# panels + a status band DIRECTLY UNDER each arm's own view (left arm under the left
# wrist panel, right arm under the right wrist panel). All heights keep the canvas
# divisible by 16 so the mp4 encoder never resizes (752 = 64 + 96 + 32 + 512 + 48;
# width = 512 * panels).
_PANEL = 512
_HEADER_H = 64
_REASON_H = 96
_LABEL_H = 32
_BAND_H = 48

# Clean high-contrast light theme: near-white ground, crisp near-black text, and
# saturated (but not neon) accents so states pop at a glance.
BG_HEADER = (251, 251, 249)   # near-white
BG_REASON = (245, 245, 242)
BG_LABEL = (236, 237, 233)
BG_PANEL = (223, 225, 221)    # letterbox / gutters around the camera frames
RULE_C = (205, 208, 203)      # thin separators
FG = (30, 34, 39)             # crisp near-black (primary text)
MUT = (94, 101, 110)
FAINT = (150, 156, 163)
ACC_LEFT = (13, 148, 136)     # teal    -- LEFT arm
ACC_RIGHT = (124, 58, 237)    # violet  -- RIGHT arm (deliberately NOT red/pink:
                               # red is reserved for failure states)
ACC_NEUTRAL = (100, 116, 139) # slate   -- shared front view
_OK_C = (22, 163, 74)          # green
_WARN_C = (217, 119, 6)        # amber
_BAD_C = (220, 38, 38)         # red


def _make_analysis_frame(
    agentview: np.ndarray, wrist: Optional[Any], record: dict[str, Any]
) -> np.ndarray:
    """Compose one polished, information-dense frame of the saved rollout video."""
    if isinstance(wrist, (list, tuple)):
        # Left - Agent - Right: each wrist sits on its own arm's side of the
        # shared front view, so the layout mirrors the physical rig.
        wrist_left, wrist_right = (np.asarray(w) for w in wrist)
        views = [wrist_left, agentview, wrist_right]
        labels = ["WRIST · LEFT", "AGENT VIEW", "WRIST · RIGHT"]
        accents = [ACC_LEFT, ACC_NEUTRAL, ACC_RIGHT]
    else:
        views = [
            agentview,
            wrist if wrist is not None else np.zeros_like(to_uint8_hwc(agentview)),
        ]
        labels = ["AGENT VIEW", "WRIST"]
        accents = [ACC_NEUTRAL, ACC_NEUTRAL]

    width = _PANEL * len(views)
    panels_y = _HEADER_H + _REASON_H + _LABEL_H
    band_y = panels_y + _PANEL
    canvas = Image.new("RGB", (width, band_y + _BAND_H), color=BG_PANEL)
    draw = ImageDraw.Draw(canvas)
    big = _font(26, bold=True)
    std = _font(18)
    std_b = _font(18, bold=True)
    small = _font(15)

    # Camera panels: letterboxed (aspect preserved), never squeezed.
    for idx, view in enumerate(views):
        canvas.paste(_letterbox(view, _PANEL), (_PANEL * idx, panels_y))

    # Header: step number + right-aligned telemetry.
    draw.rectangle((0, 0, width, _HEADER_H), fill=BG_HEADER)
    draw.text((16, 16), f"STEP {int(record.get('i', 0)):03d}", fill=FG, font=big)
    _draw_right(draw, width - 16, 24, _header_facts(record), small)
    draw.line((0, _HEADER_H - 1, width, _HEADER_H - 1), fill=RULE_C, width=1)

    # Reasoning strip: the model's words, up to four wrapped lines (~full for the
    # concise CoT this stack runs; steps.json always keeps the untruncated text).
    draw.rectangle((0, _HEADER_H, width, _HEADER_H + _REASON_H), fill=BG_REASON)
    reason = _vlm_reason(record, "c")
    if reason != "-":
        draw.text((16, _HEADER_H + 6), "WHY", fill=FAINT, font=small)
        per_line = max(60, (width - 80) // 9)
        text = " ".join(str(reason).split())
        wrapped = textwrap.wrap(text, width=per_line)[:4]
        if len(wrapped) == 4 and len(text) > per_line * 4:
            wrapped[3] = wrapped[3][: per_line - 3] + "..."
        for i, line in enumerate(wrapped):
            draw.text((62, _HEADER_H + 6 + i * 21), line, fill=MUT, font=std)

    # Label bar: panel names with a per-panel accent underline.
    label_y = _HEADER_H + _REASON_H
    draw.rectangle((0, label_y, width, label_y + _LABEL_H), fill=BG_LABEL)
    for idx, (label, accent) in enumerate(zip(labels, accents)):
        x0 = _PANEL * idx
        draw.text((x0 + 16, label_y + 7), label, fill=MUT, font=small)
        draw.rectangle(
            (x0 + 12, label_y + _LABEL_H - 4, x0 + _PANEL - 12, label_y + _LABEL_H - 2),
            fill=accent,
        )

    # Status band: each arm's STAGE + action rendered DIRECTLY UNDER its own view.
    draw.rectangle((0, band_y, width, band_y + _BAND_H), fill=BG_HEADER)
    draw.line((0, band_y, width, band_y), fill=RULE_C, width=1)
    dual = isinstance(record.get("left"), dict) and isinstance(record.get("right"), dict)
    if dual:
        bands = {0: ("LEFT", ACC_LEFT, record["left"]), 2: ("RIGHT", ACC_RIGHT, record["right"])}
    else:
        bands = {1: ("ARM", ACC_NEUTRAL, record)}
    for idx, (name, accent, rec) in bands.items():
        x0 = _PANEL * idx + 16
        _draw_segments(
            draw, x0, band_y + 4,
            [(f"{name:<6}", accent, True), (_stage_label(rec), FG, False)],
            std, std_b,
        )
        act = str(rec.get("dec") or rec.get("act") or "-")
        line2 = [(f"{act:<10}", FG, True), (f"grip {rec.get('grip', '-')}", MUT, False)]
        for text, color in status_flags(rec):
            line2.append((f"   {text}", color, False))
        _draw_segments(draw, x0, band_y + 25, line2, std, std_b)

    # Thin separators between panels, spanning label bar through status band.
    for idx in range(1, len(views)):
        x0 = _PANEL * idx
        draw.line((x0, label_y, x0, band_y + _BAND_H), fill=RULE_C, width=1)
    return np.asarray(canvas)


def _letterbox(view: np.ndarray, size: int) -> Image.Image:
    """Fit a camera frame into a size x size tile, preserving its aspect ratio."""
    img = Image.fromarray(to_uint8_hwc(view))
    scale = min(size / img.width, size / img.height)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new, Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (size, size), BG_PANEL)
    tile.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return tile


def _stage_label(rec: dict[str, Any]) -> str:
    stage = str(rec.get("stage", "-"))
    if stage in ("FINISHED", "-"):
        return stage if stage == "FINISHED" else "STAGE -"
    try:
        idx = int(rec.get("sg", 0)) + 1
    except (TypeError, ValueError):
        idx = "-"
    total = rec.get("n_sg")
    count = f"{idx}/{int(total)}" if total else f"{idx}"
    return f"STAGE {count} · {stage}"


def status_flags(rec: dict[str, Any]) -> list[tuple[str, tuple[int, int, int]]]:
    flags: list[tuple[str, tuple[int, int, int]]] = []
    if rec.get("done"):
        flags.append(("✓ stage done", _OK_C))
    if rec.get("grasp_fail"):
        flags.append(("✕ empty close", _BAD_C))
    blocked = rec.get("blocked")
    if blocked == "z_floor":
        flags.append(("⚠ z-floor", _WARN_C))
    elif blocked == "reach":
        flags.append(("⚠ reach limit", _WARN_C))
    if rec.get("recover"):
        flags.append(("↺ recovery", _WARN_C))
    if rec.get("auto_release"):
        flags.append(("↺ auto-release", _WARN_C))
    view = rec.get("view")
    if view:
        flags.append((f"{str(view).lower()} guided", FAINT))
    return flags


def _header_facts(record: dict[str, Any]) -> list[tuple[str, tuple[int, int, int]]]:
    """Right-aligned run telemetry: per-step VLM latency, total elapsed time, and
    the running average time per executed step."""
    facts: list[tuple[str, tuple[int, int, int]]] = []
    if record.get("vlm_ms") is not None:
        facts.append((f"VLM {record['vlm_ms']} ms", MUT))
    t_s = record.get("t_s")
    if t_s is not None:
        t_s = float(t_s)
        facts.append((f"TOTAL {int(t_s // 60):02d}:{int(t_s % 60):02d}", MUT))
    if record.get("avg_s") is not None:
        facts.append((f"AVG {float(record['avg_s']):.1f} s/STEP", MUT))
    if record.get("replan"):
        facts.append((f"REPLAN {record['replan']}", _WARN_C))
    if record.get("done"):
        facts.append(("ALL STAGES DONE", _OK_C))
    return facts


def _draw_segments(draw: Any, x: int, y: int, segments: Any, font: Any, bold: Any) -> None:
    for text, color, is_bold in segments:
        active = bold if is_bold else font
        draw.text((x, y), text, fill=color, font=active)
        x += draw.textlength(text, font=active)


def _draw_right(draw: Any, right_x: int, y: int, facts: Any, font: Any) -> None:
    parts = [text for text, _ in facts]
    if not parts:
        return
    joined = "   ·   ".join(parts)
    x = right_x - draw.textlength(joined, font=font)
    for i, (text, color) in enumerate(facts):
        draw.text((x, y), text, fill=color, font=font)
        x += draw.textlength(text, font=font)
        if i < len(facts) - 1:
            draw.text((x, y), "   ·   ", fill=FAINT, font=font)
            x += draw.textlength("   ·   ", font=font)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for base in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation2",
        "/usr/share/fonts/truetype/liberation",
    ):
        for name in names:
            try:
                return ImageFont.truetype(f"{base}/{name}", size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _vlm_reason(record: dict[str, Any], role: str) -> str:
    vlm = record.get("vlm")
    if not isinstance(vlm, dict):
        return "-"
    role_aliases = {
        "c": ("c", "controller"),
        "g": ("g", "gripper"),
    }
    entry = None
    for key in role_aliases.get(role, (role,)):
        candidate = vlm.get(key)
        if isinstance(candidate, dict):
            entry = candidate
            break
    if not isinstance(entry, dict):
        return "-"
    return str(entry.get("why") or entry.get("reasoning") or "-")


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
