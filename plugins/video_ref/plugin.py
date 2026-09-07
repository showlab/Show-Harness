"""Reference-video replication tool (baseline): demo video -> planner brief.

Given a reference video of the task being demonstrated (a teleop recording's
visualization mp4, or any front-facing clip), this capability makes the subgoal
planner REPLICATE the demonstration instead of inventing its own plan: one dedicated
VLM call watches ``num_frames`` frames sampled uniformly in order and extracts a
structured DEMO BRIEF -- the ordered operations with the nuances that matter for
faithful replication (which arm, the exact part grasped, the exact destination and
placement detail). The brief renders as a compact block injected into the planner
prompt (``{video_ref}`` in ``subgoal_planner_dual.txt`` / ``subgoal_planner.txt``);
the planner still plans from the LIVE images, so demo-vs-live position differences
are absorbed there.

Two modes share the tool: **dual** (default; the Piper Mode B rig) attributes every
operation to an arm and renders the cross-arm sequencing rules; **single**
(``single=True``; the Franka / ``scripts/run_real.py`` path) drops the arm attribution --
one arm performs everything, so the analyst schema, prompt, and planner block carry
no arm field.

Baseline scope, on purpose:
  * The brief is TEXT; the demo frames are not re-sent with the planner call.
  * Replication is plan-level (operation order, arm, grasp part, destination).
    Trajectory-level mimicry is out of scope.
  * The extraction runs ONCE per run (before the first plan) and the block also
    feeds every replan through the planner agent it was mounted on.

Split-role rationale (two calls, not one): the analyst call sees only the video, so
its output is a loggable, operator-checkable artifact (printed at run start, saved in
run metadata) and the planner prompt stays small -- the planner never has to divide
attention between N demo frames and the live scene.

Disabled (or no video given) -> every method returns its empty value and the planner
prompt is unchanged. VLM-calling per the plugins convention: duck-typed ``client``
passed into :meth:`extract_brief`, prompt text co-located in ``video_ref.txt``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image

PROMPT_PATH = Path(__file__).with_name("video_ref.txt")
SINGLE_PROMPT_PATH = Path(__file__).with_name("video_ref_single.txt")

DEFAULT_NUM_FRAMES = 8

# Frames are downscaled before the VLM call (image_to_data_url sends them verbatim).
DEFAULT_MAX_SIDE = 512

ARMS = ("left", "right", "both")


def _operation_schema(single: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action": {"type": "string"},
        "object": {"type": "string"},
        "grasp": {"type": "string"},
        "destination": {"type": "string"},
    }
    if not single:
        properties = {"arm": {"type": "string", "enum": list(ARMS)}, **properties}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _brief_schema(single: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "operations": {
                "type": "array",
                "items": _operation_schema(single),
                "minItems": 1,
            },
        },
        "required": ["task", "operations"],
        "additionalProperties": False,
    }


# The dual-rig shapes, kept under their historical names.
_OPERATION_SCHEMA: dict[str, Any] = _operation_schema(single=False)
BRIEF_SCHEMA: dict[str, Any] = _brief_schema(single=False)


def load_video_ref_prompt(single: bool = False) -> str:
    """Return the co-located analyst prompt template for the requested mode."""
    path = SINGLE_PROMPT_PATH if single else PROMPT_PATH
    return path.read_text(encoding="utf-8").strip()


class VideoRefPlugin:
    """Extract a demo brief from a reference video and render the planner block."""

    def __init__(
        self,
        enabled: bool = False,
        video_path: str | Path | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        max_side: int = DEFAULT_MAX_SIDE,
        single: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.single = bool(single)
        self.video_path = Path(video_path) if video_path else None
        self.num_frames = max(2, int(num_frames))
        self.max_side = int(max_side)
        # Set by extract_brief(): the validated brief dict and the sampled indices.
        self.brief: Optional[dict[str, Any]] = None
        self.sampled_indices: list[int] = []

    # -- video ---------------------------------------------------------------
    def sample_frames(self) -> list[np.ndarray]:
        """Uniformly sample ``num_frames`` frames (first and last included), in order.

        One sequential decode pass keeping only the selected indices -- no random
        seeks (fragile across containers), no full-video buffering (long clips).
        When the container reports no usable frame count, falls back to buffering
        a bounded 1-of-N stride and subsampling that.
        """
        if self.video_path is None:
            raise ValueError("video_ref: no video_path configured")
        reader = imageio.get_reader(str(self.video_path), "ffmpeg")
        try:
            total = self._frame_count(reader)
            if total:
                wanted = sorted(
                    set(np.linspace(0, total - 1, self.num_frames).round().astype(int))
                )
                keep = set(wanted)
                frames = {
                    i: self._shrink(frame)
                    for i, frame in enumerate(reader)
                    if i in keep
                }
                self.sampled_indices = [i for i in wanted if i in frames]
                return [frames[i] for i in self.sampled_indices]
            # Unknown length: buffer a bounded stride (~4/s at common frame rates),
            # then subsample uniformly.
            meta = reader.get_meta_data()
            stride = max(1, int(round(float(meta.get("fps") or 12.0) / 4.0)))
            buffered: list[tuple[int, np.ndarray]] = []
            for i, frame in enumerate(reader):
                if i % stride == 0:
                    buffered.append((i, self._shrink(frame)))
                if len(buffered) >= 2400:  # ~10 min of video; demos are far shorter
                    break
            if not buffered:
                raise RuntimeError(f"video_ref: no frames decoded from {self.video_path}")
            picks = sorted(
                set(np.linspace(0, len(buffered) - 1, self.num_frames).round().astype(int))
            )
            self.sampled_indices = [buffered[p][0] for p in picks]
            return [buffered[p][1] for p in picks]
        finally:
            reader.close()

    @staticmethod
    def _frame_count(reader: Any) -> int:
        try:
            total = reader.count_frames()
            if np.isfinite(total) and int(total) > 0:
                return int(total)
        except Exception:  # noqa: BLE001 - fall through to the stride path
            pass
        return 0

    def _shrink(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(frame))
        image.thumbnail((self.max_side, self.max_side))
        return np.asarray(image)

    # -- demo brief extraction -------------------------------------------------
    def extract_brief(self, client: Any, debug: bool = False) -> dict[str, Any]:
        """One analyst VLM call: sampled frames -> validated demo brief (stored).

        Tries guided JSON first, then a free-JSON retry. Raises ``RuntimeError`` when
        both fail: a replication run without the demo brief would silently degrade
        into ordinary planning, which is exactly what the operator did not ask for.
        """
        frames = self.sample_frames()
        prompt = load_video_ref_prompt(self.single).format(num_frames=len(frames))
        errors: list[str] = []
        for schema in (_brief_schema(self.single), None):
            try:
                response = client.complete_json(
                    prompt,
                    frames[0],
                    wrist_image=frames[1:],
                    schema=schema,
                    max_tokens=None,
                    temperature=0.0,
                    chat_template_kwargs={},
                    debug=debug,
                )
                payload = response.payload.get("json")
                self.brief = _validate_brief(payload, single=self.single)
                return self.brief
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError(
            f"video_ref: could not extract a demo brief from {self.video_path}: "
            + " | ".join(_one_line(e)[:200] for e in errors)
        )

    # -- planner prompt block ----------------------------------------------------
    def render_prompt(self) -> str:
        """The REFERENCE DEMO block for the planner prompt ('' when disabled/empty)."""
        if not self.enabled or not self.brief:
            return ""
        lines = "\n".join(
            _op_line(i + 1, op) for i, op in enumerate(self.brief["operations"])
        )
        header = (
            "### REFERENCE DEMO -- a demonstration video of this task was analyzed. "
            "REPLICATE it.\n"
            f"Demo: {self.brief['task']}\n"
            "Operations in demo order:\n"
            f"{lines}\n"
        )
        if self.single:
            return header + (
                "Replicate faithfully on the LIVE images: perform the operations in "
                "the demo's order, grasp the same part (use it as the affordance), "
                "same destination and placement detail. Object positions may differ "
                "from the demo -- plan from where things are NOW; the rules below "
                "still govern stage segmentation."
            )
        return header + (
            "Replicate faithfully on the LIVE images: same arm per operation, grasp "
            "the same part (use it as the affordance), same destination and placement "
            "detail. The numbers are the demo's TIME ORDER across BOTH arms: an "
            "operation starts only after every lower-numbered operation is visibly "
            "finished -- when the preceding operation belongs to the OTHER arm, give "
            "this arm a WAIT stage first, completion = that operation's visible "
            "result. This sequencing OVERRIDES the keep-both-arms-busy preference. "
            "Object positions may differ from the demo -- plan from where things are "
            "NOW; the rules below still govern stage segmentation."
        )

    # -- operator-facing summaries -------------------------------------------------
    def summary_lines(self) -> list[str]:
        """Short per-operation lines for the run header ('' -> nothing to show)."""
        if not self.brief:
            return []
        return [_op_line(i + 1, op) for i, op in enumerate(self.brief["operations"])]

    def metadata(self) -> dict[str, Any]:
        """What the run log needs to reproduce/inspect the extraction."""
        if not self.enabled:
            return {}
        return {
            "video_path": str(self.video_path) if self.video_path else None,
            "num_frames": self.num_frames,
            "sampled_indices": list(self.sampled_indices),
            "brief": self.brief,
        }


def _validate_brief(parsed: Any, single: bool = False) -> dict[str, Any]:
    """Coerce/validate the analyst JSON into the canonical brief shape.

    Single mode keeps the same shape with ``arm: ""`` -- one arm performs
    everything, and the empty value makes ``_op_line`` render without a label."""
    if not isinstance(parsed, dict):
        raise RuntimeError(f"video_ref: analyst returned no JSON object: {parsed!r}")
    task = str(parsed.get("task") or "").strip()
    raw_ops = parsed.get("operations")
    if not task or not isinstance(raw_ops, list) or not raw_ops:
        raise RuntimeError(
            f"video_ref: analyst JSON missing task/operations: {_one_line(parsed)[:300]}"
        )
    operations: list[dict[str, str]] = []
    for item in raw_ops:
        if not isinstance(item, dict):
            continue
        arm = str(item.get("arm") or "").strip().lower()
        operations.append(
            {
                "arm": "" if single else (arm if arm in ARMS else "both"),
                "action": str(item.get("action") or "").strip() or "manipulate",
                "object": str(item.get("object") or "").strip() or "-",
                "grasp": str(item.get("grasp") or "").strip() or "-",
                "destination": str(item.get("destination") or "").strip() or "-",
            }
        )
    if not operations:
        raise RuntimeError(
            f"video_ref: analyst JSON has no parseable operations: {_one_line(parsed)[:300]}"
        )
    return {"task": task, "operations": operations}


def _op_line(index: int, op: dict[str, str]) -> str:
    arm_label = f"{op['arm'].upper()} " if op.get("arm") else ""
    parts = [f"{index}. {arm_label}{op['action']} {op['object']}"]
    if op.get("grasp") and op["grasp"] != "-":
        parts.append(f"-- grasp {op['grasp']}")
    if op.get("destination") and op["destination"] != "-":
        parts.append(f"-> {op['destination']}")
    return " ".join(parts)


def _one_line(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return " ".join(str(value).split())
