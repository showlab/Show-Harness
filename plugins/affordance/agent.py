"""AffordancePointer VLM role: task + instruction + image -> exact contact point(s).

The affordance capability's VLM sub-role (mirroring ``plugins/subgoal/agent.py``). One
``point`` call marks where the gripper should make contact, as [y, x] on a 0-1000 grid
(the pointing convention VLMs are trained on); an optional self-verification loop then
draws the dot, shows the ANNOTATED image back, and asks the model to confirm or correct
it -- pointing precision is judged far more reliably with the candidate visible than it
is produced blind. Prompts are co-located (``affordance_point.txt`` /
``affordance_verify.txt``); the ``client`` is duck-typed (``complete_json``) per the
plugins convention.

The same role serves two views. :meth:`locate` grounds a fresh point on a static view
(the front / AgentView, once per stage). :meth:`track` RE-locates that same part on a
MOVING view (the eye-in-hand wrist, per step): it seeds the previous dot onto the new
frame and runs a single verify-style call (confirm / correct / declare it gone), which
keeps the mark on one physical point across a shifting camera far more stably than an
independent re-ground each step. Both take a ``view_desc`` so the prompt names the view.

Degradation contract: :meth:`locate` returns ``[]`` and :meth:`track` returns ``None``
when the model abstains ("part not visible here") or every call fails -- the caller
renders no dot and the controller runs exactly as without this capability. Never raises
mid-rollout.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from core.record.images import to_uint8_hwc

POINT_PROMPT_PATH = Path(__file__).with_name("affordance_point.txt")
VERIFY_PROMPT_PATH = Path(__file__).with_name("affordance_verify.txt")

# No-think regime for local vLLM backends (same rationale as the subgoal planner: a
# busy scene sends a thinking template into a monologue that overruns the budget).
# Hosted providers drop chat_template_kwargs and keep their reasoning_effort setting.
NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}

# The 0-1000 pointing grid ([y, x], top-left origin) the prompts define.
GRID = 1000

# Default {view_desc} for the static front grounding; wrist callers pass their own.
FRONT_VIEW_DESC = "the front view of the workspace"

# Stage mode: exactly one point (the ABSTAIN rule in the prompt covers the empty case).
ONE_POINT_RULE = "Return EXACTLY ONE point, or an empty points list per the ABSTAIN rule."
# Task mode (offline test): the next manipulation's contact point(s).
TASK_POINTS_RULE = (
    "Return one point per gripper contact the task's NEXT manipulation needs "
    "(1 or 2 points): two arms acting at once = one point each; a two-arm grasp "
    "of one object = 2 points at opposite ends."
)

_POINT_FIELD_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}

# Field order matters: "part"/"why" precede "point" so the model commits to a named,
# visually-evidenced part BEFORE emitting coordinates (look-then-point).
POINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "part": {"type": "string"},
                    "why": {"type": "string"},
                    "point": _POINT_FIELD_SCHEMA,
                },
                "required": ["label", "part", "why", "point"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["points"],
    "additionalProperties": False,
}

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # present=false means the named part is not in THIS image (it left the moving
        # wrist frame, is occluded, or was never here) -> the caller drops the dot.
        "present": {"type": "boolean"},
        "on_target": {"type": "boolean"},
        "why": {"type": "string"},
        "point": _POINT_FIELD_SCHEMA,
    },
    "required": ["present", "on_target", "why", "point"],
    "additionalProperties": False,
}


def load_point_prompt() -> str:
    """Return the co-located pointing prompt template."""
    return POINT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_verify_prompt() -> str:
    """Return the co-located verification prompt template."""
    return VERIFY_PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass
class AffordancePoint:
    """One grounded contact point on the front view (0-1000 grid, top-left origin)."""

    label: str
    part: str
    why: str
    x: int  # 0 = left edge .. 1000 = right edge
    y: int  # 0 = top edge .. 1000 = bottom edge
    verified: bool = False  # a verify pass confirmed (or corrected) it
    calls: int = 1  # VLM calls spent grounding this point
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "part": self.part,
            "why": self.why,
            "point": [self.y, self.x],
            "verified": self.verified,
            "calls": self.calls,
            "latency_s": round(self.latency_s, 3),
        }


def draw_point(
    image: np.ndarray, x: int, y: int, color: tuple[int, int, int]
) -> np.ndarray:
    """A copy of ``image`` with a filled dot (white halo + colored core) at the
    0-1000 grid location ``(x, y)``. The halo keeps the dot visible on any
    background; the size scales with resolution so the marked point stays exact."""
    arr = to_uint8_hwc(np.asarray(image)).copy()
    h, w = arr.shape[:2]
    px = int(round(_clamp_grid(x) / GRID * (w - 1)))
    py = int(round(_clamp_grid(y) / GRID * (h - 1)))
    radius = max(4, int(round(0.012 * min(h, w))))
    halo = radius + max(2, radius // 3)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    draw.ellipse([px - halo, py - halo, px + halo, py + halo], fill=(255, 255, 255))
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)
    return np.asarray(pil)


class AffordancePointerAgent:
    """Point call + draw-and-verify loop; returns grounded :class:`AffordancePoint` s."""

    def __init__(self, client: Any, verify_rounds: int = 1) -> None:
        self.client = client
        # Self-verification rounds per point (0 = trust the first answer). Each round
        # is one extra VLM call on the annotated image; the loop exits early on the
        # first on_target verdict.
        self.verify_rounds = max(0, int(verify_rounds))

    # -- public entry -----------------------------------------------------------
    def locate(
        self,
        task: str,
        instruction: str,
        agentview: np.ndarray,
        stage_line: str = "",
        max_points: int = 1,
        color: tuple[int, int, int] = (255, 32, 32),
        color_name: str = "RED",
        view_desc: str = FRONT_VIEW_DESC,
        ref_image: Optional[np.ndarray] = None,
        ref_line: str = "",
        debug: bool = False,
    ) -> list[AffordancePoint]:
        """Ground ``instruction`` on ``agentview`` (``view_desc`` names it); [] on
        abstain or failure. Used for the static front grounding and the FIRST wrist
        grounding of a grasp (a fresh point call, then the verify loop).

        ``ref_image``/``ref_line``: an optional SECOND image attached to the call (a
        cross-view reference -- e.g. the scene view with the committed dot) with the
        prompt line describing it; default off -> the prompt and call are unchanged."""
        started = time.monotonic()
        count_rule = ONE_POINT_RULE if max_points <= 1 else TASK_POINTS_RULE
        prompt = load_point_prompt().format(
            task=task,
            instruction=instruction,
            stage_line=stage_line.strip(),
            view_desc=view_desc,
            count_rule=count_rule,
            context_lines=(ref_line.strip() if ref_image is not None else ""),
        )
        try:
            payload = self._complete(prompt, agentview, POINT_SCHEMA, debug, ref_image=ref_image)
        except RuntimeError as exc:
            print(f"[affordance] pointing failed ({_one_line(exc)[:160]})")
            return []
        points = _parse_points(payload, max_points)
        if not points and "points" not in payload:
            # Degenerate reply that parsed as JSON but not as points: distinct from a
            # deliberate abstain (an explicit empty "points" list), so don't cache it
            # as one -- the schema-fallback retry in _complete already ran.
            print(f"[affordance] unusable pointing reply ({_one_line(payload)[:160]})")
        for point in points:
            self._verify(
                task, instruction, point, agentview, color, color_name, view_desc, debug,
                ref_image=ref_image, ref_line=ref_line,
            )
            point.latency_s = time.monotonic() - started
        return points

    def track(
        self,
        task: str,
        part: str,
        view_image: np.ndarray,
        prev_x: int,
        prev_y: int,
        color: tuple[int, int, int],
        color_name: str,
        view_desc: str,
        ref_image: Optional[np.ndarray] = None,
        ref_line: str = "",
        debug: bool = False,
    ) -> Optional[AffordancePoint]:
        """Re-locate ``part`` on a MOVED view, seeded with the previous dot ``(prev_x,
        prev_y)``. One verify-style call: confirm, correct, or declare the part gone.

        Returns a fresh :class:`AffordancePoint` (``verified`` = the model confirmed it
        was already on target), or ``None`` when the part is no longer visible or the
        call fails -- the caller then drops that dot for the step. Seeding the prior
        keeps the mark on ONE physical point as the wrist camera shifts, instead of a
        noisy independent re-ground each step.
        """
        started = time.monotonic()
        annotated = draw_point(view_image, prev_x, prev_y, color)
        result = self._verify_call(
            task, _track_instruction(part), part, annotated, color_name, view_desc, debug,
            ref_image=ref_image, ref_line=ref_line,
        )
        if result is None:
            return None
        present, on_target, corrected = result
        if not present:
            return None
        if corrected is not None and not on_target:
            y, x = corrected
        else:
            x, y = prev_x, prev_y
        return AffordancePoint(
            label="",
            part=part,
            why="tracked on the wrist view",
            x=x,
            y=y,
            verified=on_target,
            calls=1,
            latency_s=time.monotonic() - started,
        )

    # -- VLM calls --------------------------------------------------------------
    def _verify(
        self,
        task: str,
        instruction: str,
        point: AffordancePoint,
        agentview: np.ndarray,
        color: tuple[int, int, int],
        color_name: str,
        view_desc: str,
        debug: bool,
        ref_image: Optional[np.ndarray] = None,
        ref_line: str = "",
    ) -> None:
        """Draw-and-check loop, mutating ``point`` toward the confirmed location.

        Each round redraws the dot at the CURRENT candidate on a fresh copy of the raw
        frame (one dot only, so the verdict is unambiguous). A final correction that
        exhausts the rounds is still adopted -- unverified is better than unmoved -- and
        any verify-call failure just keeps the current candidate.
        """
        for _ in range(self.verify_rounds):
            annotated = draw_point(agentview, point.x, point.y, color)
            result = self._verify_call(
                task, instruction, point.part, annotated, color_name, view_desc, debug,
                ref_image=ref_image, ref_line=ref_line,
            )
            if result is None:
                return
            point.calls += 1
            present, on_target, corrected = result
            if not present:
                # The part is not in this (static) view -- unexpected for a fresh
                # front grounding; keep the candidate rather than second-guess it.
                return
            if corrected is not None and not on_target:
                point.y, point.x = corrected
            point.verified = point.verified or on_target
            if on_target:
                return

    def _verify_call(
        self,
        task: str,
        instruction: str,
        part: str,
        annotated_image: np.ndarray,
        color_name: str,
        view_desc: str,
        debug: bool,
        ref_image: Optional[np.ndarray] = None,
        ref_line: str = "",
    ) -> Optional[tuple[bool, bool, Optional[tuple[int, int]]]]:
        """One verify/track VLM call on an image with the dot already drawn. Returns
        ``(present, on_target, corrected_yx)`` or ``None`` on failure. ``ref_image``
        (+ ``ref_line``) attaches a cross-view reference so the verdict is grounded
        in independent evidence instead of the drawn dot alone."""
        prompt = load_verify_prompt().format(
            task=task,
            instruction=instruction,
            part=part,
            color=color_name,
            view_desc=view_desc,
            context_lines=(ref_line.strip() if ref_image is not None else ""),
        )
        try:
            payload = self._complete(prompt, annotated_image, VERIFY_SCHEMA, debug, ref_image=ref_image)
        except RuntimeError as exc:
            print(f"[affordance] verify/track failed, keeping point ({_one_line(exc)[:120]})")
            return None
        present = bool(payload.get("present", True))
        on_target = bool(payload.get("on_target"))
        return present, on_target, _parse_grid_point(payload.get("point"))

    def _complete(
        self,
        prompt: str,
        image: np.ndarray,
        schema: dict[str, Any],
        debug: bool,
        ref_image: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Guided JSON first, then a free-JSON retry (the subgoal/video_ref ladder):
        a constrained decode can collapse into degenerate text on some hosted
        backends, while free generation with the prompt's explicit JSON shape still
        answers. Raises RuntimeError only when both attempts fail."""
        errors: list[str] = []
        for attempt_schema in (schema, None):
            try:
                response = self.client.complete_json(
                    prompt,
                    image,
                    wrist_image=ref_image,
                    schema=attempt_schema,
                    max_tokens=None,
                    temperature=0.0,
                    chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                    debug=debug,
                )
                payload = response.payload.get("json")
                if isinstance(payload, dict):
                    return payload
                errors.append(f"no JSON object: {str(response.raw_text)[:120]!r}")
            except RuntimeError as exc:
                errors.append(_one_line(exc)[:160])
        raise RuntimeError("affordance: " + " | ".join(errors))


def _parse_points(payload: dict[str, Any], max_points: int) -> list[AffordancePoint]:
    items = payload.get("points")
    if not isinstance(items, list):
        return []
    points: list[AffordancePoint] = []
    for item in items[: max(1, int(max_points))]:
        if not isinstance(item, dict):
            continue
        grid = _parse_grid_point(item.get("point"))
        if grid is None:
            continue
        y, x = grid
        points.append(
            AffordancePoint(
                label=str(item.get("label") or "").strip(),
                part=str(item.get("part") or "").strip(),
                why=str(item.get("why") or "").strip(),
                x=x,
                y=y,
            )
        )
    return points


def _parse_grid_point(value: Any) -> Optional[tuple[int, int]]:
    """``[y, x]`` clamped to the 0-1000 grid; None when the shape is unusable."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        y, x = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return _clamp_grid(y), _clamp_grid(x)


def _track_instruction(part: str) -> str:
    return f"keep the dot on {part} as the wrist camera moves"


def _clamp_grid(value: float) -> int:
    return int(min(max(round(float(value)), 0), GRID))


def _one_line(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return " ".join(str(value).split())
