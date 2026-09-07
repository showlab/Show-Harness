"""Recovery tool: measured-width grasp failure detection and subgoal rewind.

This tool is deliberately narrow. It does not inspect images, call a VLM, or own any
motion primitive. It watches the measured gripper width at stage boundaries and tells
the runner when a closed gripper is empty/lost, so the runner can reopen the gripper
and roll the active goal back to the relevant grasp stage.

The one numeric threshold here is a physical gripper calibration, not task logic:
near-zero closed width means the fingers touched each other, so nothing is between
them. Direction, alignment, and retry strategy are still left to the controller VLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.prompt_text import fragment


def _fragment(section: str) -> str:
    """This tool's prompt text lives in the co-located recovery.txt (the context
    wrapper and the per-event notes)."""
    return fragment(__file__, "recovery.txt", section)


RELEASE_TOKEN = "RELEASE"
GRASP_TOKEN = "GRASP"
STOP_TOKEN = "STOP"

# How many consecutive steps to HOLD while a just-commanded close has not settled (the
# width sensor still reads the open band right after the close). Bounded so a genuinely
# wide grasp -- which can legitimately read near the open band -- eventually proceeds.
MAX_UNSETTLED_HOLDS = 3


@dataclass(frozen=True)
class RecoveryDecision:
    """A runner-level recovery request.

    ``token`` is an optional action override for the current step. ``release`` requests
    an immediate release after the just-executed token. ``rollback_index`` moves the
    active subgoal pointer before the next VLM decision.
    """

    event: str
    reason: str
    token: str | None = None
    release: bool = False
    rollback_index: int | None = None
    block_done: bool = False
    reset_history: bool = False
    grasp_empty: bool = False
    grasp_unverified: bool = False
    prompt_note: str = ""


class RecoveryPlugin:
    """Classify measured gripper width and request recovery interventions."""

    def __init__(
        self,
        enabled: bool = True,
        empty_width_m: float = 0.005,
        open_width_m: float = 0.06,
    ) -> None:
        self.enabled = bool(enabled)
        self.empty_width_m = max(0.0, float(empty_width_m))
        self.open_width_m = max(self.empty_width_m, float(open_width_m))
        # Consecutive "close not settled yet" holds issued, bounded by MAX_UNSETTLED_HOLDS.
        self._unsettled_holds = 0

    def render_prompt_context(self, note: str) -> str:
        """Return a short controller context line for the last recovery event."""
        if not self.enabled:
            return ""
        text = " ".join(str(note or "").split())
        return _fragment("context").replace("{note}", text) if text else ""

    def phase_from_width(self, width_m: Any) -> str:
        """Classify measured width as ``empty``, ``holding``, ``open``, or ``unknown``."""
        try:
            width = float(width_m)
        except (TypeError, ValueError):
            return "unknown"
        if width <= self.empty_width_m:
            return "empty"
        if width >= self.open_width_m:
            return "open"
        return "holding"

    def before_decision(
        self,
        *,
        current_index: int,
        subgoals: Sequence[Any],
        measured_width_m: Any,
        gripper_closed: bool,
    ) -> RecoveryDecision | None:
        """Intervene before a VLM call based on the closed gripper's measured width.

        empty   -> reopen + roll back to the grasp stage (the close caught nothing).
        open    -> the async close has not settled (the sensor still reads the open band
                   right after a close); HOLD so a translation cannot slip in before the
                   grasp result is known. Bounded by MAX_UNSETTLED_HOLDS so a genuinely
                   wide grasp eventually proceeds.
        holding -> a verified hold; no intervention.
        """
        if not self.enabled or not gripper_closed:
            self._unsettled_holds = 0
            return None
        phase = self.phase_from_width(measured_width_m)

        if phase == "empty":
            self._unsettled_holds = 0
            rollback = _nearest_grasp_index(subgoals, current_index)
            if rollback is None:
                return None
            stage = (
                _motion(subgoals[current_index])
                if 0 <= current_index < len(subgoals)
                else ""
            )
            event = "empty_grasp" if stage == GRASP_TOKEN else "lost_grasp"
            reason = (
                f"closed gripper width {float(measured_width_m):.4f}m <= "
                f"empty threshold {self.empty_width_m:.4f}m"
            )
            return RecoveryDecision(
                event=event,
                reason=reason,
                token=RELEASE_TOKEN,
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note(event),
            )

        if phase != "open":  # holding or unknown -> proceed normally
            self._unsettled_holds = 0
            return None

        # phase == "open": commanded closed but the width still reads the open band, i.e.
        # the async close has not settled. Hold (do not move) until it settles, bounded.
        if self._unsettled_holds >= MAX_UNSETTLED_HOLDS:
            self._unsettled_holds = 0
            return None
        self._unsettled_holds += 1
        return RecoveryDecision(
            event="grasp_unsettled",
            reason=(
                f"closed gripper still reads open width {_fmt_width(measured_width_m)}; "
                "waiting for the gripper sensor to settle before the next move"
            ),
            token=STOP_TOKEN,
            block_done=True,
            prompt_note=_fragment("note_unsettled"),
        )

    def after_step(
        self,
        *,
        token: str,
        result: Any,
        current_index: int,
        subgoals: Sequence[Any],
        measured_width_m: Any,
        subgoal_done: bool,
        gripper_closed: bool,
    ) -> RecoveryDecision | None:
        """Validate a just-executed step before the runner advances subgoals."""
        if not self.enabled:
            return None
        token = str(token or "").strip().upper()
        rollback = _nearest_grasp_index(subgoals, current_index)
        if rollback is None:
            return None

        if bool(getattr(result, "grasp_empty", False)):
            return RecoveryDecision(
                event="empty_grasp",
                reason=str(getattr(result, "note", "") or "gripper closed empty"),
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note("empty_grasp"),
            )

        stage = (
            _motion(subgoals[current_index])
            if 0 <= current_index < len(subgoals)
            else ""
        )
        phase = self.phase_from_width(measured_width_m)

        if stage == GRASP_TOKEN and subgoal_done and phase != "holding":
            reason = (
                f"grasp DONE rejected because measured width phase is {phase}"
                f" (width={_fmt_width(measured_width_m)})"
            )
            return RecoveryDecision(
                event="unverified_grasp",
                reason=reason,
                release=bool(gripper_closed and phase == "empty"),
                rollback_index=current_index,
                block_done=True,
                reset_history=True,
                grasp_empty=(phase == "empty"),
                grasp_unverified=True,
                prompt_note=_prompt_note(
                    "empty_grasp" if phase == "empty" else "unverified_grasp"
                ),
            )

        if stage != GRASP_TOKEN and gripper_closed and phase == "empty":
            reason = (
                f"closed gripper became empty during {stage or 'post-grasp stage'} "
                f"(width={_fmt_width(measured_width_m)})"
            )
            return RecoveryDecision(
                event="lost_grasp",
                reason=reason,
                release=True,
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note("lost_grasp"),
            )

        return None


def _nearest_grasp_index(subgoals: Sequence[Any], current_index: int) -> int | None:
    upper = min(max(int(current_index), 0), len(subgoals) - 1)
    for idx in range(upper, -1, -1):
        if _motion(subgoals[idx]) == GRASP_TOKEN:
            return idx
    return 0 if subgoals else None


def _motion(subgoal: Any) -> str:
    return str(getattr(subgoal, "motion", "") or "").strip().upper()


def _fmt_width(value: Any) -> str:
    try:
        return f"{float(value):.4f}m"
    except (TypeError, ValueError):
        return "unknown"


def _prompt_note(event: str) -> str:
    if event == "lost_grasp":
        return _fragment("note_lost_grasp")
    if event == "unverified_grasp":
        return _fragment("note_unverified_grasp")
    return _fragment("note_empty_grasp")
