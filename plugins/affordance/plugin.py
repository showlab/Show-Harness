"""Affordance grounding tool: premark each stage's exact contact point in BOTH views.

Motivation: the controller steers on the PLANNER'S WORDS ("affordance": a textual part
name), so tasks that hinge on WHERE exactly -- grasp a banana by one end for a handover,
set a plate down on an exact spot -- fail at the text-to-pixels step, and patching ever
more affordance rules into the planner/controller prompts erodes their instruction
following. This capability moves that responsibility into one dedicated, meticulously
prompted VLM pointing role (:class:`~plugins.affordance.agent.AffordancePointerAgent`).

Two-phase, matching how the controller steers (front for the approach, wrist for the
grasp):
  * FRONT (approach, rule B) -- when an arm ENTERS a spatial stage, the agent grounds
    the stage's (target, affordance) into an exact FRONT-view point ONCE (the front
    camera is static, so the pixel stays valid all stage). This picks WHICH part.
  * WRIST (grasp, rule A) -- once the arm is wrist-guided (the runner passes that
    signal), the agent RE-locates that same part on the moving wrist frame EVERY step,
    seeded with the previous dot (:meth:`AffordancePointerAgent.track`), so the mark
    stays locked to one physical point as the eye-in-hand camera shifts. Grasp
    precision is wrist-guided, so this is where the dot matters most.

The dot is drawn (single arm / LEFT: red; RIGHT: blue) on the front image every step and
on the wrist image while tracking; the controller's AFFORD field becomes "<part> = RED
dot in the LEFT Wrist view (also in Front View)" -- the committed part plus the dot
binding, replacing (not adding to) the planner's wording. Rule A ("judge AFFORD vs
fingertips") then reads the wrist dot; rule B reads the front dot. The prompt gets
SIMPLER, not longer.

Scope and degradation:
  * The front grounding is the semantic anchor; the wrist tracker re-locates THAT part,
    so a hedging planner text ("left end or middle") is sharpened to one committed part
    in both views.
  * Front grounding is once per (slot, stage); wrist tracking is per step but only while
    the arm is wrist-guided (no wrist VLM cost during the approach). A recovery clears
    the slot so a disturbed scene re-grounds.
  * Any abstain/failure drops that dot for the step (front) or the grasp (wrist): the
    controller falls back to the other view / the part text, identical in the limit to
    the tool being disabled. Never raises mid-rollout.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from plugins.affordance.agent import AffordancePoint, AffordancePointerAgent, draw_point
from plugins.prompt_text import fragment

# Slot -> (color name used in prompts, RGB). "arm" is the single-arm slot; the dual
# runner uses "left"/"right". Red/blue keeps the two dots unmistakable in one image.
DOT_COLORS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "arm": ("RED", (255, 32, 32)),
    "left": ("RED", (255, 32, 32)),
    "right": ("BLUE", (0, 96, 255)),
}

# Stage motions whose guidance is not toward a fixed scene point (vertical clearance,
# holding still, opening in place, plan sentinels) -- a structural property of the
# planner's stage vocabulary, not a scene threshold. PRESENT is here for a subtler
# reason, learned on hardware: a handover PRESENT ends when
# the object appears in the RECEIVER'S wrist view, so its endpoint is defined by the
# other arm, not by any fixed pixel -- a dot on "the handover region" turned that
# completion-driven stage into point-chasing (9x MV_RIGHT into the reach limit).
# Everything else (GRASP, MOVE, PLACE, PUSH, ...) is offered to the pointer, which
# may still abstain.
NON_SPATIAL_MOTIONS = frozenset(
    {"LIFT", "RETREAT", "WAIT", "RELEASE", "OPEN", "DONE", "REASON", "STILL", "PRESENT"}
)

# Track confirmations at the IDENTICAL pixel tolerated before the wrist seed is
# considered frozen and the part is re-grounded independently (see _wrist_stale).
_WRIST_STALE_LIMIT = 3


class AffordancePlugin:
    """Per-stage contact-point grounding + dot annotation + controller prompt block."""

    def __init__(
        self,
        enabled: bool = False,
        client: Any = None,
        verify_rounds: int = 1,
        view_name: str = "AgentView",
        view_desc: Optional[str] = None,
        wrist_tracking: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and client is not None
        # How the active controller prompt names the front image ("AgentView" in the
        # single-arm prompt, "Front View" in the dual one).
        self.view_name = str(view_name)
        # How the POINTER prompt describes that image ({view_desc}); None keeps the
        # dual rig's default wording. The Franka path passes its own (the AgentView
        # is an external upper camera, not the ego front camera).
        self.view_desc = str(view_desc) if view_desc else None
        # Master switch for the per-step WRIST dot (config: affordance_wrist_track).
        # Off -> the tool stays front-only: no wrist VLM calls ever, AFFORD keeps the
        # front-dot binding, and update_wrist() is an inert drop.
        self.wrist_tracking = bool(wrist_tracking)
        self.agent = (
            AffordancePointerAgent(client, verify_rounds=verify_rounds)
            if self.enabled
            else None
        )
        # Per-slot state. _stage_keys: the stage each slot was last front-grounded FOR
        # (an abstain/failed grounding is cached so the stage is not re-grounded).
        # _front_points: the static front dot (the semantic anchor: WHICH part).
        # _wrist_points: the per-step tracked wrist dot during the grasp phase.
        self._stage_keys: dict[str, str] = {}
        self._front_points: dict[str, AffordancePoint] = {}
        self._wrist_points: dict[str, AffordancePoint] = {}
        # Consecutive track() confirmations that returned the IDENTICAL pixel. Under a
        # moving eye-in-hand camera a fixed scene point cannot keep the same pixel, so
        # a run of verbatim confirms means the verifier is echoing its own drawn dot
        # (observed: frozen 7 steps at the same pixel through 7 MV_LEFTs);
        # at _WRIST_STALE_LIMIT the seed is dropped and the part re-grounded fresh.
        self._wrist_stale: dict[str, int] = {}
        # Every FRONT grounding + each wrist-tracking START this run, for the metadata /
        # offline inspection (per-step wrist tracks live in steps.jsonl, not here).
        self._history: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------------
    def wants(self, subgoal: Optional[dict[str, Any]]) -> bool:
        """Whether this stage moves toward a groundable scene point."""
        if not self.enabled or not isinstance(subgoal, dict):
            return False
        motion = str(subgoal.get("motion", "")).strip().upper()
        target = str(subgoal.get("target", "")).strip()
        return bool(target) and motion not in NON_SPATIAL_MOTIONS

    def ensure(
        self,
        slot: str,
        stage_key: str,
        task: str,
        subgoal: Optional[dict[str, Any]],
        agentview: np.ndarray,
        agentview_hd: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> None:
        """Ground ``slot``'s current stage once (no-op while ``stage_key`` is unchanged).

        Call every step BEFORE annotate(); grounding happens only on stage entry (a
        fresh ``stage_key``), on a non-spatial stage the slot's dot is dropped.
        ``agentview_hd`` (optional): a higher-resolution render of the SAME frame with
        the same pad geometry -- when given, the pointer/verify VLM calls see it
        instead of the small observation (glyph-level detail), while the returned
        0-1000 grid point is identical across both renders.
        """
        if not self.enabled or self.agent is None:
            return
        if not self.wants(subgoal):
            self._drop(slot)
            return
        if self._stage_keys.get(slot) == stage_key:
            return
        # A new stage: the previous stage's wrist track no longer applies.
        self._wrist_points.pop(slot, None)
        color_name, color = DOT_COLORS.get(slot, DOT_COLORS["arm"])
        locate_kwargs: dict[str, Any] = {}
        if self.view_desc:
            locate_kwargs["view_desc"] = self.view_desc
        points = self.agent.locate(
            task=task,
            instruction=_instruction(slot, subgoal),
            agentview=agentview_hd if agentview_hd is not None else agentview,
            stage_line=_stage_line(subgoal),
            max_points=1,
            color=color,
            color_name=color_name,
            debug=debug,
            **locate_kwargs,
        )
        self._stage_keys[slot] = stage_key
        record: dict[str, Any] = {"slot": slot, "stage_key": stage_key, "view": "front"}
        if points:
            self._front_points[slot] = points[0]
            record.update(points[0].to_dict())
            print(f"  [affordance] {slot} · {_describe(subgoal)} -> {_dot_line(points[0], color_name)}")
        else:
            self._front_points.pop(slot, None)
            record["point"] = None
            print(f"  [affordance] {slot} · {_describe(subgoal)} -> no point (abstained/failed)")
        self._history.append(record)

    def update_wrist(
        self,
        slot: str,
        wrist_image: Optional[np.ndarray],
        active: bool,
        task: str,
        agentview: Optional[np.ndarray] = None,
        wrist_hd: Optional[np.ndarray] = None,
        agentview_hd: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> None:
        """Per-step wrist dot for ``slot`` (call every step, BEFORE annotate_wrist()).

        ``active`` is the runner's "this arm is wrist-guided now" signal (dual: the
        arm's last reported guiding view was WRIST; single: the last target-in-wrist
        marker). When inactive, or with no committed front part, the wrist dot is
        dropped and no VLM call is made -- so the approach phase pays nothing. When
        active, the same part the front grounding committed is (re-)located on THIS
        wrist frame: a fresh grounding on the first step, then a seeded track.

        ``agentview`` (optional): the CURRENT raw front frame. When given, every wrist
        locate/track call also attaches the front view with the committed dot drawn as
        a cross-view REFERENCE image -- the wrist judgment is then anchored to the
        (independently verified) front grounding instead of only its own drawn dot,
        which is what kept a wrong wrist dot alive on look-alike scenes. A track that
        confirms the IDENTICAL pixel _WRIST_STALE_LIMIT times in a row is treated as
        frozen (impossible under a moving camera) and the part is re-grounded fresh.
        """
        if not self.enabled or self.agent is None:
            return
        front = self._front_points.get(slot)
        if not self.wrist_tracking or not active or front is None or wrist_image is None:
            self._wrist_points.pop(slot, None)
            self._wrist_stale.pop(slot, None)
            return
        color_name, color = DOT_COLORS.get(slot, DOT_COLORS["arm"])
        view_desc = _wrist_view_desc(slot)
        # HD renders (same pad geometry, same 0-1000 grid) take priority for every
        # VLM-facing image; the small observation stays the fallback.
        vlm_wrist = wrist_hd if wrist_hd is not None else wrist_image
        ref_front = agentview_hd if agentview_hd is not None else agentview
        ref_kwargs: dict[str, Any] = {}
        if ref_front is not None:
            ref_kwargs["ref_image"] = draw_point(ref_front, front.x, front.y, color)
            ref_kwargs["ref_line"] = (
                fragment(__file__, "affordance_field.txt", "cross_view_ref")
                .replace("{view_name}", self.view_name)
                .replace("{color_name}", color_name)
            )
        prev = self._wrist_points.get(slot)
        if prev is not None and self._wrist_stale.get(slot, 0) >= _WRIST_STALE_LIMIT:
            print(
                f"  [affordance] {slot} · wrist dot frozen at [y={prev.y},x={prev.x}] "
                f"for {_WRIST_STALE_LIMIT} steps -- re-grounding independently"
            )
            prev = None
            self._wrist_stale[slot] = 0
        if prev is None:
            points = self.agent.locate(
                task=task,
                instruction=f"{_arm_word(slot)}current target: {front.part}",
                agentview=vlm_wrist,
                max_points=1,
                color=color,
                color_name=color_name,
                view_desc=view_desc,
                debug=debug,
                **ref_kwargs,
            )
            point = points[0] if points else None
            self._history.append(
                {
                    "slot": slot,
                    "view": "wrist",
                    "part": front.part,
                    "point": point.to_dict()["point"] if point else None,
                }
            )
            if point is not None:
                print(f"  [affordance] {slot} · wrist tracking begins -> {_dot_line(point, color_name)}")
        else:
            point = self.agent.track(
                task=task,
                part=front.part,
                view_image=vlm_wrist,
                prev_x=prev.x,
                prev_y=prev.y,
                color=color,
                color_name=color_name,
                view_desc=view_desc,
                debug=debug,
                **ref_kwargs,
            )
            if point is not None:
                if point.x == prev.x and point.y == prev.y:
                    self._wrist_stale[slot] = self._wrist_stale.get(slot, 0) + 1
                else:
                    self._wrist_stale[slot] = 0
        if point is not None:
            point.part = front.part  # keep the committed part label stable across steps
            self._wrist_points[slot] = point
        else:
            self._wrist_points.pop(slot, None)
            self._wrist_stale.pop(slot, None)

    def clear(self, slot: Optional[str] = None) -> None:
        """Drop grounded state so the next ensure() re-grounds ``slot`` (None: all).

        Runners call this on a recovery intervention: an empty/lost grasp usually means
        the scene was disturbed, so the premarked pixel may no longer be the part.
        """
        if slot is None:
            self._stage_keys.clear()
            self._front_points.clear()
            self._wrist_points.clear()
            self._wrist_stale.clear()
            return
        self._drop(slot)

    def _drop(self, slot: str) -> None:
        self._stage_keys.pop(slot, None)
        self._front_points.pop(slot, None)
        self._wrist_points.pop(slot, None)
        self._wrist_stale.pop(slot, None)

    # -- per-step outputs ----------------------------------------------------------
    def annotate(self, agentview: np.ndarray) -> np.ndarray:
        """The front view with every active FRONT dot drawn; unchanged input when none.

        Thread-safe against concurrent ensure()/clear(): iterates a snapshot, so the
        live-display render thread can call it while the rollout thread re-grounds.
        """
        if not self.enabled or not self._front_points:
            return agentview
        annotated = agentview
        for slot, point in list(self._front_points.items()):
            _, color = DOT_COLORS.get(slot, DOT_COLORS["arm"])
            annotated = draw_point(annotated, point.x, point.y, color)
        return annotated

    def annotate_wrist(self, slot: str, wrist_image: Optional[np.ndarray]):
        """``wrist_image`` with ``slot``'s tracked wrist dot drawn; input unchanged when
        that slot has no active wrist dot (approach phase, or the part left the frame)."""
        point = self._wrist_points.get(slot)
        if not self.enabled or point is None or wrist_image is None:
            return wrist_image
        _, color = DOT_COLORS.get(slot, DOT_COLORS["arm"])
        return draw_point(wrist_image, point.x, point.y, color)

    def annotate_frames(
        self, frames: Optional[dict[str, np.ndarray]]
    ) -> Optional[dict[str, np.ndarray]]:
        """Live-display adapter: overlay the front dots on ``agentview`` and each arm's
        tracked wrist dot on ``wrist_left``/``wrist_right`` (``None``/dot-less input
        passes through untouched), so a streaming viewer shows exactly what the
        controller is being steered by in every panel."""
        if frames is None or not self.enabled:
            return frames
        if not self._front_points and not self._wrist_points:
            return frames
        out = dict(frames)
        if out.get("agentview") is not None:
            out["agentview"] = self.annotate(out["agentview"])
        # "wrist" is the single-arm stream key (slot "arm", the single-arm runner's
        # convention); wrist_left/right are the dual rig's.
        for key, slot in (("wrist", "arm"), ("wrist_left", "left"), ("wrist_right", "right")):
            if out.get(key) is not None:
                out[key] = self.annotate_wrist(slot, out[key])
        return out

    def afford_field(self, slot: str, planner_afford: str) -> str:
        """The controller's AFFORD field value for ``slot``.

        No dot -> the planner's text unchanged. With a dot, the POINTER'S committed
        part replaces the planner's wording (which may hedge, e.g. "left end or
        middle") and the dot binding is stated inline. Once the wrist dot is tracking,
        it is named FIRST (rule A / grasp reads the wrist), with the front dot noted:

            approach:  left end = RED dot in Front View
            grasp:     left end = RED dot in the LEFT Wrist view (also in Front View)

        One field, no extra prompt lines: rule A judges AFFORD vs the fingertips (wrist
        dot), rule B and the GRASP check use the front dot.
        """
        front = self._front_points.get(slot)
        wrist = self._wrist_points.get(slot)
        if not self.enabled or front is None:
            return planner_afford
        color_name, _ = DOT_COLORS.get(slot, DOT_COLORS["arm"])
        part = front.part.strip() or str(planner_afford).strip() or "the contact point"
        section = "with_wrist" if wrist is not None else "front_only"
        return (
            fragment(__file__, "affordance_field.txt", section)
            .replace("{part}", part)
            .replace("{color_name}", color_name)
            .replace("{wrist_view}", _wrist_view_name(slot))
            .replace("{view_name}", self.view_name)
        )

    def point_of(self, slot: str) -> Optional[dict[str, Any]]:
        """The active FRONT dot for the step log (None when slot has no front dot)."""
        point = self._front_points.get(slot)
        return point.to_dict() if self.enabled and point is not None else None

    def wrist_point_of(self, slot: str) -> Optional[dict[str, Any]]:
        """The active tracked WRIST dot for the step log (None when not tracking)."""
        point = self._wrist_points.get(slot)
        return point.to_dict() if self.enabled and point is not None else None

    def metadata(self) -> dict[str, Any]:
        """What the run log needs to inspect every grounding of this episode."""
        if not self.enabled:
            return {}
        return {
            "verify_rounds": self.agent.verify_rounds if self.agent else 0,
            "groundings": list(self._history),
        }


def _arm_word(slot: str) -> str:
    return f"the {slot.upper()} arm's " if slot in ("left", "right") else "the arm's "


def _wrist_view_name(slot: str) -> str:
    """How the controller prompt names this slot's wrist image."""
    if slot == "left":
        return "LEFT Wrist view"
    if slot == "right":
        return "RIGHT Wrist view"
    return "Wrist view"


def _wrist_view_desc(slot: str) -> str:
    """The {view_desc} for a wrist pointing/tracking call: a close-up with the
    fingertips visible, so the pointer marks the object part, not the gripper."""
    side = f"{slot.upper()} " if slot in ("left", "right") else ""
    return (
        f"the {side}wrist camera's close-up of the grasp; the gripper's two fingertips "
        "are visible -- mark the object part to grasp, not the fingers"
    )


def _instruction(slot: str, subgoal: dict[str, Any]) -> str:
    arm = _arm_word(slot)
    motion = str(subgoal.get("motion", "")).strip() or "current"
    target = str(subgoal.get("target", "")).strip()
    affordance = str(subgoal.get("affordance", "")).strip()
    line = f"{arm}{motion} stage -- target: {target}"
    if affordance:
        line += f"; contact part: {affordance}"
    return line


def _stage_line(subgoal: dict[str, Any]) -> str:
    description = str(subgoal.get("description", "")).strip()
    return f"Stage goal: {description}" if description else ""


def _describe(subgoal: dict[str, Any]) -> str:
    return f"{subgoal.get('motion', '?')} {subgoal.get('target', '?')}"


def _dot_line(point: AffordancePoint, color_name: str) -> str:
    mark = "verified" if point.verified else "unverified"
    return (
        f"{color_name} dot [y={point.y},x={point.x}] part='{point.part}' "
        f"({mark}, {point.calls} calls, {point.latency_s:.1f}s)"
    )
