"""Rotation plugin: gripper yaw alignment for grasping, with eye-in-hand frame compensation.

The controller normally commands only translations (MV_*) and gripper events; the gripper
keeps its captured downward orientation. Some objects cannot be grasped that way -- an
elongated or angled object must have the gripper's two-finger opening rotated ACROSS its
narrow graspable width. This tool adds two atomic actions, ``ROTATE_CW`` / ``ROTATE_CCW``
(a fixed yaw step about the vertical), and offers them to the controller VLM as a
last-resort alignment move.

THE FRAME PROBLEM (the reason this is a tool, not just two tokens): the wrist camera is
eye-in-hand -- it rotates rigidly with the gripper. MV_* tokens are BASE-FRAME (a fixed
world direction), and the controller judges fine grasp moves from the WRIST view. At the
captured (reference) yaw the two agree, so "AFFORD to the gripper's left -> MV_LEFT" is
correct. Once the gripper has yawed by theta, the wrist image has rotated with it, so
"left in the wrist image" is no longer base -Y. Rather than make the VLM mentally re-map
directions (VLMs are weak at that) or switch fine moves to the coarser world-fixed
agentview, we compensate in CODE: :meth:`compensate_move` rotates a wrist-judged MV_*
vector by the KNOWN accumulated yaw before it is executed, so the VLM keeps reasoning in
the wrist frame exactly as today. When the gripper is unrotated (theta ~= 0) this is the
identity, so with the plugin off -- or before any rotation -- motion is byte-identical.

Disabled -> :meth:`action_tokens` is empty (the VLM is never offered ROTATE_*, so it never
emits one) and :meth:`compensate_move` returns the vector unchanged. Self-contained: it owns
its prompt (:meth:`render_prompt`) and the compensation geometry; the controller only calls it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

# The two yaw tokens, sourced from the hardware controller's own vocabulary so the strings
# the VLM is offered are exactly the ones franka_atomic_controller executes (single source
# of truth; importing avoids silent drift). ROTATE_CCW=+yaw, ROTATE_CW=-yaw (primitives_franka.yaml).
from core.action_units import ROTATE_ATOMS

# Co-located prompt file (the "when to rotate" block), loaded once and cached.
PROMPT_PATH = Path(__file__).with_name("rotation.txt")

DEFAULT_YAW_STEP_DEG = 30.0
# Soft guard so a run of same-direction rotations cannot drive the wrist joint into its
# hardware limit (Franka joint 7 ~= +/-166 deg). A safety bound like the Z floor, NOT a
# task gate -- it never counts or forbids a *kind* of action, only caps absolute travel.
DEFAULT_MAX_ACCUM_DEG = 150.0
# Below this residual yaw the gripper counts as back at neutral, so a lift proceeds.
DEFAULT_REALIGN_EPS_DEG = 2.0


class RotationPlugin:
    """Offer ROTATE_CW/CCW to the controller and compensate wrist-judged moves for yaw."""

    def __init__(
        self,
        enabled: bool = False,
        yaw_step_rad: float | None = None,
        compensation_sign: float = 1.0,
        max_accumulated_yaw_rad: float | None = None,
        realign_eps_rad: float | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        # Per-command yaw magnitude (default 30 deg). The controller uses this for ROTATE_*
        # and must widen its per-command yaw clamp to match, else the step is truncated.
        self.yaw_step_rad = (
            math.radians(DEFAULT_YAW_STEP_DEG) if yaw_step_rad is None else float(yaw_step_rad)
        )
        # +1 applies R_z(+accumulated_yaw); flip to -1 if a real-robot check shows a
        # post-rotation move goes the wrong way (the raw wrist image may be mirrored so the
        # apparent rotation sense is inverted). Sign is the one empirical unknown.
        self.compensation_sign = float(compensation_sign)
        self.max_accumulated_yaw_rad = (
            math.radians(DEFAULT_MAX_ACCUM_DEG)
            if max_accumulated_yaw_rad is None
            else float(max_accumulated_yaw_rad)
        )
        # Residual yaw below which the gripper is "back at neutral" for a lift.
        self.realign_eps_rad = (
            math.radians(DEFAULT_REALIGN_EPS_DEG)
            if realign_eps_rad is None
            else float(realign_eps_rad)
        )
        self._prompt = prompt_template

    def action_tokens(self) -> tuple[str, ...]:
        """The extra controller tokens to offer the VLM (empty when disabled)."""
        return tuple(ROTATE_ATOMS) if self.enabled else ()

    def render_prompt(self) -> str:
        """The 'when to rotate' controller-prompt block, or '' when disabled."""
        if not self.enabled:
            return ""
        if self._prompt is None:
            self._prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        return self._prompt

    def compensate_move(self, delta_xyz: Sequence[float], accumulated_yaw_rad: float):
        """Rotate a base-frame MV_* vector by the accumulated gripper yaw.

        The VLM judged this move in the (yawed) wrist image; rotating the base vector by
        the same yaw makes "left/right/forward/back in the wrist view" hit the correct
        base-frame direction. Vertical moves (x=y=0) are unaffected. Returns the vector
        unchanged when disabled or effectively unrotated (theta ~= 0), so the default MV
        path is untouched until a rotation has actually happened.
        """
        vec = np.asarray(delta_xyz, dtype=float).reshape(3)
        if not self.enabled:
            return vec
        theta = self.compensation_sign * float(accumulated_yaw_rad)
        if abs(theta) < 1e-6:
            return vec
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        return np.array([cos_t * x - sin_t * y, sin_t * x + cos_t * y, z], dtype=float)

    def yaw_within_limit(self, accumulated_yaw_rad: float) -> bool:
        """True while the accumulated yaw is inside the soft joint-travel guard."""
        return abs(float(accumulated_yaw_rad)) <= self.max_accumulated_yaw_rad

    def realign_delta_yaw(self, accumulated_yaw_rad: float) -> float:
        """Yaw change to return to the neutral (reference) orientation -- i.e.
        ``-accumulated_yaw`` -- or 0.0 when disabled or already ~neutral.

        Rotation is only a grasp-alignment move, so once an object is grasped the controller
        un-rotates back to neutral BEFORE lifting (an ``MV_UP`` while holding), keeping the
        carry/place phase in the clean un-rotated frame (and returning MV_* compensation to
        the identity). The controller applies this correction, clamped per command, so a
        large turn is undone over a step or two.
        """
        if not self.enabled:
            return 0.0
        acc = float(accumulated_yaw_rad)
        if abs(acc) <= self.realign_eps_rad:
            return 0.0
        return -acc
