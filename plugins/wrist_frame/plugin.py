"""Wrist-motion-frame prompt fix: front-view directions follow the gripper heading.

With ``motion_frame: wrist`` (an optional robot-config setting; the default is the
AgentView-centric ``base``) the controller executes MV_* relative to each arm's OWN
gripper heading -- MV_FWD moves along wherever that gripper points. The wrist-view
guidance (rule A) is exact by construction then, but the front-view guidance (rule B)
is written in the base convention: "TARGET near the image left -> MV_LEFT" etc. Under
wrist-frame execution those image-edge mappings are wrong whenever the arm is yawed
(the dual rig's begin poses are yawed +/-45 deg), so this tool rewrites ONLY the rule-B
TARGET mapping lines to judge relative to the gripper's visible pointing direction.

Mirrors :class:`plugins.ego.plugin.EgoPlugin`: a prompt-only transform driven by a hardware
property (here the configured motion frame), applied in memory before the prompt
reaches the controller agent; the source prompt files stay in the base convention.
Disabled -> the prompt is returned unchanged (the AgentView-centric default).

Ordering: apply BEFORE :class:`~plugins.ego.plugin.EgoPlugin`. The rewritten lines no longer
name an image edge, so the ego FWD/BACK swap correctly skips them (gripper-relative
directions are view-independent) while still fixing the wrist-view (rule A) lines.

For the PER-STEP dynamic frame (the guiding view picks each move's frame) see
``plugins.view_select``, which instead requires the untouched base-convention prompt
and therefore does not combine with ``motion_frame: wrist``.
"""
from __future__ import annotations

import re

# Image-edge front-view mappings -> gripper-heading mappings. Matched with flexible
# inner whitespace so cosmetic alignment in the prompt files never breaks the rewrite.
_LINE_REWRITES: tuple[tuple[str, str], ...] = (
    (
        r"-\s*TARGET near the image left\s+->\s*MV_LEFT",
        "- TARGET to the gripper-heading's left  -> MV_LEFT",
    ),
    (
        r"-\s*TARGET near the image right\s+->\s*MV_RIGHT",
        "- TARGET to the gripper-heading's right -> MV_RIGHT",
    ),
    (
        r"-\s*TARGET near the image bottom\s+->\s*MV_FWD",
        "- TARGET further ahead along the gripper's heading -> MV_FWD",
    ),
    (
        r"-\s*TARGET near the image top\s+->\s*MV_BACK",
        "- TARGET behind, opposite the gripper's heading -> MV_BACK",
    ),
)

# One-sentence definition of the heading, appended to rule B's intro line so the VLM
# knows what to anchor the four mappings to. Matches any rule-B intro phrasing that
# names the primary guide ("B) NO -> ..." in controller.txt, "B) Goal NOT in the wrist
# view -> ..." in controller_dual.txt), so a prompt rewording cannot silently detach it.
_INTRO_PATTERN = r"(B\) .*?primary guide\.)"
_INTRO_ADDENDUM = (
    " Moves follow the GRIPPER HEADING -- the direction the gripper points, visible in"
    " this view."
)


class WristFramePlugin:
    """Prompt-only front-view direction fix for wrist-frame (heading-relative) MV_*."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def apply(self, prompt_template: str) -> str:
        """Return ``prompt_template`` unchanged, or with the front-view TARGET mapping
        rewritten to gripper-heading-relative directions when enabled."""
        if not self.enabled:
            return prompt_template
        out = re.sub(_INTRO_PATTERN, r"\1" + _INTRO_ADDENDUM, prompt_template, count=1)
        for pattern, replacement in _LINE_REWRITES:
            out = re.sub(pattern, replacement, out)
        return out
