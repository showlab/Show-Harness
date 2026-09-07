"""Auto-release tool: reflexively reopen a closed gripper that is holding nothing.

This tool is deliberately narrow, like :mod:`plugins.recovery`. It owns no motion
primitive, inspects no image, and calls no VLM. It answers a single physical
question each step -- "is the closed gripper empty?" -- by comparing the measured
gripper width against a calibration threshold. The runner acts on the answer by
issuing a ``RELEASE`` so the next decision starts from a clean, open gripper.

Unlike the controller's in-step empty-grasp check (which only fires at the moment a
GRASP settles) this rule is *reactive*: it is consulted after every step, so a grasp
that slips later -- the object falling out while the arm translates -- is also caught
and reopened. The threshold is the configured ``empty_width_m`` (a near-zero closed
width means the fingers touched each other, so nothing is between them); it is a
physical gripper calibration, not task logic.
"""
from __future__ import annotations


class AutoReleasePlugin:
    """Decide whether a closed gripper has collapsed below the empty-grasp width."""

    def __init__(self, enabled: bool = True, empty_width_m: float = 0.001) -> None:
        self.enabled = bool(enabled)
        self.empty_width_m = max(0.0, float(empty_width_m))

    def should_release(self, width_m: float, gripper_closed: bool) -> bool:
        """True when the rule should reopen the gripper.

        Fires only for a CLOSED gripper whose measured width is below
        ``empty_width_m`` -- an open gripper reads the wide (~0.08 m) band and is
        never touched. Disabled tool always returns ``False`` (no-op), so the
        runner behaves byte-identically to having no rule at all.
        """
        if not self.enabled or not bool(gripper_closed):
            return False
        try:
            width = float(width_m)
        except (TypeError, ValueError):
            return False
        return width < self.empty_width_m
