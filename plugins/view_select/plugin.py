"""Multi-view action selection: the guiding view picks each move's motion frame.

The dual rig has two motion frames with opposite strengths. ``base`` executes MV_*
as AgentView-aligned vectors, so the front-view guidance (rule B) is exact but the
wrist-view guidance (rule A) drifts once an arm is yawed (the dual begin poses are
yawed +/-45 deg). Static ``motion_frame: wrist`` is the mirror image: rule A becomes
exact, and rule B needs the :class:`~plugins.wrist_frame.plugin.WristFramePlugin` prompt
rewrite. Neither static choice fits a whole episode -- grasping wants the wrist
frame, search/transport/placement want the base frame.

This tool makes the choice per arm, per step, by the VLM itself: the dual controller
already decides which view is the "primary guide" (rule A wrist vs rule B front), so
the output contract asks it to REPORT that choice, and the runner executes that arm's
move in the frame that realizes the chosen view's directions exactly:

    WRIST (rule A) -> ``wrist`` frame     FRONT (rule B) -> ``base`` frame

No prompt-direction rewrites are needed: the dual prompt stays in the base
convention, where rule B's image-edge lines are exact under base-frame execution and
rule A's wrist-image lines (as adapted by the ego swap, a camera-orientation
property) are exact under wrist-frame execution -- at any yaw, on either arm.
``motion_frame`` must therefore stay ``base``; ``run_real_dual`` refuses the
combination with ``wrist``, whose rule-B prompt rewrite would contradict the
base-frame execution of FRONT-guided moves.

Contract additions (rendered only when enabled; disabled -> every method returns its
empty value and the pipeline is byte-identical to today):

  * JSON mode: per-arm ``left_view`` / ``right_view`` fields ("WRIST" | "FRONT"),
    enforced by the guided-decoding schema.
  * CoT mode: the FINAL line carries the view per arm: ``FINAL: LEFT=<ACTION>@<VIEW>
    RIGHT=<ACTION>@<VIEW>``.

A missing/unparseable view (degraded output, strict-retry salvage) maps to no
override, so that move falls back to the controller's configured frame -- exactly
the pre-tool behavior.
"""
from __future__ import annotations

import re
from typing import Any, Optional

VIEW_WRIST = "WRIST"
VIEW_FRONT = "FRONT"
VIEW_TOKENS = (VIEW_WRIST, VIEW_FRONT)

_SIDES = ("left", "right")

# Guiding view -> the motion frame that executes that view's directions exactly.
_FRAME_BY_VIEW = {VIEW_WRIST: "wrist", VIEW_FRONT: "base"}

# Per-side recovery of the view from free text, covering both emit styles:
#   '"left_view": "WRIST"'  (malformed-JSON salvage)   'LEFT=MV_DOWN@WRIST'  (CoT FINAL)
_VIEW_ALTS = "|".join(VIEW_TOKENS)
_VIEW_PATTERNS = {
    side: re.compile(
        rf'(?:"?{side}_view"?\s*[=:]\s*"?\s*|{side.upper()}\s*=\s*[A-Z_]+\s*@\s*)'
        rf"({_VIEW_ALTS})\b",
        re.IGNORECASE,
    )
    for side in _SIDES
}


class ViewSelectPlugin:
    """Per-arm, per-step guiding-view report driving the executed motion frame."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    # -- output-contract contributions (consumed by core.vlm.dual_roles) --------------
    def json_example_field(self, side: str) -> str:
        """Fragment inserted after ``side``'s action in the JSON contract example."""
        if not self.enabled:
            return ""
        return f'"{side}_view":"{VIEW_WRIST}|{VIEW_FRONT}",'

    def schema_fields(self) -> dict[str, Any]:
        """Extra guided-JSON properties (added as required fields when enabled)."""
        if not self.enabled:
            return {}
        return {
            f"{side}_view": {"type": "string", "enum": list(VIEW_TOKENS)}
            for side in _SIDES
        }

    def cot_final_line(self) -> str:
        """The FINAL-line format replacing the default one in CoT mode."""
        return "FINAL: LEFT=<ACTION>@<VIEW> RIGHT=<ACTION>@<VIEW>"

    def contract_note(self, field_label: str) -> str:
        """One definition line tying the output field to the DIRECTION rules.

        ``field_label`` names the field as the surrounding contract spells it:
        ``left_view/right_view`` (JSON) or ``<VIEW>`` (CoT).
        """
        if not self.enabled:
            return ""
        return (
            f"{field_label} = which view guided that arm: "
            f"{VIEW_WRIST} (rule A) or {VIEW_FRONT} (rule B)."
        )

    # -- recovery of the views from the model output ----------------------------
    def views_from_payload(self, payload: dict[str, Any]) -> dict[str, Optional[str]]:
        """Per-arm views from a parsed decision JSON; invalid/missing -> ``None``."""
        if not self.enabled or not isinstance(payload, dict):
            return {}
        views: dict[str, Optional[str]] = {}
        for side in _SIDES:
            value = str(payload.get(f"{side}_view", "") or "").strip().upper()
            views[side] = value if value in VIEW_TOKENS else None
        return views

    def parse_views(self, text: str) -> dict[str, Optional[str]]:
        """Per-arm views from free text (CoT reasoning / malformed-JSON salvage).

        Uses the LAST match per side so the FINAL line wins over views quoted
        mid-reasoning; a side with no marker maps to ``None`` (no frame override).
        """
        if not self.enabled or not text:
            return {}
        views: dict[str, Optional[str]] = {}
        for side, pattern in _VIEW_PATTERNS.items():
            matches = pattern.findall(str(text))
            views[side] = matches[-1].upper() if matches else None
        return views

    # -- view -> executed motion frame -------------------------------------------
    def frame_for(self, view: Optional[str]) -> Optional[str]:
        """Motion frame realizing ``view`` exactly; ``None`` -> keep the configured frame."""
        if not self.enabled or not view:
            return None
        return _FRAME_BY_VIEW.get(str(view).strip().upper())
