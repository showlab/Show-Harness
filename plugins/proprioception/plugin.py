"""Proprioception tool: controller *context providers*.

Renders prompt lines giving the VLM the gripper's measured state: end-effector height
above the table (plus the per-step move distances), whether the last descent actually
happened, and gripper finger width. The height + step-size line lets the VLM *reason*
about reach ("I'm 20 cm up and each step is ~3 cm, so I should descend") instead of
guessing distances. The gripper-width line lives under the controller prompt's
``GRIPPER:`` section, where it can inform grasp/done decisions.

The descent line closes a blind spot the images cannot: a MV_DOWN that moved almost
nothing looks identical to one that worked, so the VLM would keep commanding it into
whatever is under the gripper. Reported as a measurement plus its consequence, not as a
gate -- the VLM still chooses the next action.

Independence: this tool only adds prompt context. The Cartesian Z-floor *safety* limit
is enforced separately in ``interpreters.franka_atomic_controller`` (gated by
``enable_z_floor``) and is unaffected by whether this tool is enabled.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from plugins.prompt_text import fragment


def _fragment(section: str) -> str:
    """This tool's prompt text lives in the co-located proprioception.txt."""
    return fragment(__file__, "proprioception.txt", section)

# A MV_DOWN that travels less than this fraction of the distance it commanded did not
# move freely: something is under the gripper (an object it is already resting on, the
# Z floor, the arm's reach limit). Calibrated from hardware logs: a free descent lands
# at 90-105% of the command (n=21, min 0.90), while the descent that pressed into a
# held-down box read 0.65 and shrinking. 0.7 separates the two with margin on both
# sides. This only decides whether to SHOW the measurement -- the VLM still decides
# what to do about it.
DESCEND_STALL_RATIO = 0.7


class ProprioceptionPlugin:
    """Turn the per-step proprio reading into controller prompt fragments."""

    def __init__(
        self,
        enabled: bool = True,
        high_above_table_m: float = 0.10,
        fine_step_m: float = 0.02,
        coarse_step_m: Optional[float] = None,
        descend_stall_ratio: float = DESCEND_STALL_RATIO,
    ) -> None:
        self.enabled = bool(enabled)
        # X: above this height (m) above the table, tell the controller to descend first.
        # Configurable via robot_franka.yaml (high_above_table_m); shared with the variable-step plugin.
        self.high_above_table_m = max(0.0, float(high_above_table_m))
        # Per-step move distances surfaced to the VLM so it can gauge how far each move goes.
        # fine_step_m is the normal step; coarse_step_m is the larger far/high step (None when the
        # variable-step plugin is off, i.e. every step is fine).
        self.fine_step_m = max(0.0, float(fine_step_m))
        self.coarse_step_m = None if coarse_step_m is None else max(0.0, float(coarse_step_m))
        # A MV_DOWN that travelled below this fraction of what was commanded is reported
        # as stalled (see DESCEND_STALL_RATIO).
        self.descend_stall_ratio = float(descend_stall_ratio)

    def _step_sizes_text(self) -> str:
        """A short clause naming the per-step move distance(s) the controller uses."""
        fine_cm = f"{self.fine_step_m * 100.0:g}"
        if self.coarse_step_m is None or self.coarse_step_m <= self.fine_step_m:
            return _fragment("step_sizes_fine").replace("{fine_cm}", fine_cm)
        return (
            _fragment("step_sizes_coarse")
            .replace("{fine_cm}", fine_cm)
            .replace("{coarse_cm}", f"{self.coarse_step_m * 100.0:g}")
        )

    def render(
        self,
        proprio: Optional[Mapping[str, Any]],
        table_height_m: Optional[float],
        holding: bool = False,
    ) -> str:
        """Return the proprio prompt block, or ``""`` when disabled / data unavailable.

        ``holding`` makes the soft hint phase-aware: when reaching for an object a larger
        gap means "descend", but once an object is held the same gap is desirable clearance
        for lifting/carrying -- the old always-"descend" hint actively fought the lift.
        """
        if not self.enabled or not proprio or table_height_m is None:
            return ""
        eef = proprio.get("eef_pos") or []
        if len(eef) < 3:
            return ""
        try:
            eef_z = float(eef[2])
            table = float(table_height_m)
        except (TypeError, ValueError):
            return ""
        gap_cm = (eef_z - table) * 100.0
        if holding:
            hint = _fragment("hint_holding")
        else:
            hint = _fragment("hint_descend").replace(
                "{high_cm}", f"{self.high_above_table_m * 100.0:.0f}"
            )
        block = (
            _fragment("block")
            .replace("{gap_cm}", f"{gap_cm:.1f}")
            .replace("{step_sizes}", self._step_sizes_text())
            .replace("{hint}", hint)
        )
        stalled = self._descend_stall_line(proprio)
        return f"{block}\n{stalled}" if stalled else block

    def _descend_stall_line(self, proprio: Mapping[str, Any]) -> str:
        """The "the last descent did not happen" line, or ``""`` when it descended freely.

        The measurement is the runner's (commanded vs actually-travelled height for the
        step just executed); this only judges whether it is worth telling the VLM. A
        stalled descent means the gripper is already against something -- repeating
        MV_DOWN cannot lower it further and only presses harder, which is a real safety
        issue on a stiff position-controlled arm (observed: an arm asked to "hold" a box
        kept commanding MV_DOWN into it).
        """
        try:
            moved = float(proprio["descend_moved_m"])
            commanded = float(proprio["descend_commanded_m"])
        except (KeyError, TypeError, ValueError):
            return ""
        if commanded <= 0.0 or moved >= commanded * self.descend_stall_ratio:
            return ""
        return (
            _fragment("descend_stall")
            .replace("{moved_cm}", f"{moved * 100.0:.1f}")
            .replace("{commanded_cm}", f"{commanded * 100.0:.1f}")
        )

    def render_gripper(self, proprio: Optional[Mapping[str, Any]]) -> str:
        """Return the ``GRIPPER:`` prompt bullet for measured finger width.

        The numeric value is exposed as context, while the recovery plugin owns any hard
        thresholding. This keeps the prompt useful but non-brittle: the VLM can combine
        width with the images instead of treating a magic number as task logic.
        """
        if not self.enabled or not proprio:
            return ""
        try:
            width_m = float(proprio.get("gripper_width"))
        except (TypeError, ValueError):
            return ""
        command = str(proprio.get("gripper_command_name") or "").strip().upper()
        command_text = f", commanded {command}" if command else ""
        return (
            _fragment("gripper")
            .replace("{width_cm}", f"{width_m * 100.0:.1f}")
            .replace("{command_text}", command_text)
        )
