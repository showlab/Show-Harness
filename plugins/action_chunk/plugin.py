"""Action-chunk tool: let the model commit a short PLAN of moves per VLM call when far.

The controller normally decides one atomic move per VLM call. When the TARGET is not yet
visible in the wrist view the gripper is still far, so re-querying every step is wasteful.
With this tool enabled, while the TARGET is far the model PLANS its next few moves in one
call -- an ordered ``PLAN: M1, M2, ...`` of up to ``step_num`` distinct ``MV_`` moves -- and
the runner executes them in order open-loop before re-deciding. The moment the TARGET is in
the wrist view it falls back to one move per call for precise alignment.

Crucially the moves are the model's OWN choices (e.g. ``MV_FWD, MV_FWD, MV_DOWN``), not a
single action repeated; the prompt asks it to plan using the height + step size so it does
not overshoot (e.g. never planning more ``MV_DOWN`` than the height above the table allows).

The tool owns the prompt instruction (:meth:`render_prompt`) and the plan parser
(:meth:`parse_plan`); the runner owns the loop that executes the plan. "Far" is the shared
wrist-visibility signal from :mod:`core.prompting.wrist_marker` (``target_in_wrist`` False).

Disabled -> ``parse_plan`` returns ``[]`` and no prompt is added, i.e. one move per call.
"""
from __future__ import annotations

import re
from typing import List, Optional

from plugins.prompt_text import fragment


_MOVE_ATOMS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"}
# The model writes "PLAN: MV_FWD, MV_FWD, MV_DOWN" on one line; grab the rest of that line
# (no DOTALL -> stops at the newline) and pull the MV_ tokens from it in order.
_PLAN_LINE = re.compile(r"PLAN\s*[:=]\s*(.+)", re.IGNORECASE)
_MOVE_TOKEN = re.compile(r"MV_[A-Z]+", re.IGNORECASE)


class ActionChunkPlugin:
    """Render the plan instruction and parse the model's planned move sequence."""

    def __init__(self, enabled: bool = False, step_num: int = 3) -> None:
        self.enabled = bool(enabled)
        # Max moves the model may commit per VLM call while far (>=1; default 3).
        self.step_num = max(1, int(step_num))

    def render_prompt(self) -> str:
        """Controller-prompt line asking the model to plan its next moves when far, or ''.
        The text lives in the co-located action_chunk.txt."""
        if not self.enabled:
            return ""
        return fragment(__file__, "action_chunk.txt", "plan").replace(
            "{step_num}", str(self.step_num)
        )

    def parse_plan(self, text: str, target_in_wrist: Optional[bool]) -> List[str]:
        """Recover the planned move sequence from the model's reasoning/output text.

        Returns up to ``step_num`` ``MV_`` moves in order (the model's own distinct choices),
        or ``[]`` when disabled, when the TARGET is not far (``target_in_wrist`` is not False),
        or when no ``PLAN:`` marker is present -- in which case the runner takes one step.
        """
        if not self.enabled or target_in_wrist is not False or not text:
            return []
        marker = _PLAN_LINE.search(str(text))
        if not marker:
            return []
        moves = [mv.upper() for mv in _MOVE_TOKEN.findall(marker.group(1))]
        valid = [mv for mv in moves if mv in _MOVE_ATOMS]
        return valid[: self.step_num]
