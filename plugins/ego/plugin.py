"""Egocentric-view prompt fix: invert the MV_FWD/MV_BACK <-> image mapping.

The controller prompt states each view's depth mapping in the Franka convention: a
target/affordance near the image BOTTOM means MV_FWD (drive away from the base), near
the image TOP means MV_BACK. Left/right and up/down are never touched.

Scope -- PER VIEW, not whole-prompt (learned on hardware):
the two views on the Piper rig do NOT share one depth convention, so the swap is
restricted to the WRIST-guidance section (rule A):

  * FRONT scene camera: the arms enter from the TOP of the image, so the base sits at
    the image top and "away from base" points toward the image BOTTOM -- the SAME
    convention the base prompt already encodes (bottom -> MV_FWD). Confirmed from the
    logged EEF: an executed MV_FWD (+X base) drives the gripper toward the image
    bottom. So rule B needs NO swap; inverting it sent front-guided approaches the
    wrong way (the observed regression, once affordance dots moved depth reasoning
    into the front view).
  * WRIST eye-in-hand camera: an executed MV_FWD advances the gripper so the target
    slides from the image TOP down toward the fingertips at the bottom -- the INVERSE
    of the base wrist rule, so rule A must be swapped (top -> MV_FWD).

This tool rewrites ONLY the wrist-section FWD/BACK direction lines in memory before the
prompt reaches the controller agent; the source prompt file is untouched (mirroring
:class:`plugins.coords.plugin.CoordsPlugin`). Driven by the hardware ``is_ego`` flag in the
robot config (a camera-orientation property), NOT a ``plugins:`` toggle: pass
``EgoPlugin(bool(cfg.get("is_ego", False)))``. Disabled -> the prompt is returned
unchanged, so any non-ego rig (Franka) is byte-identical.

Section detection anchors on the prompt's own guidance headers: a line naming the
"primary guide" opens the WRIST section when it mentions the wrist, else the FRONT
section; only lines inside the wrist section that name an image edge (``image bottom``/
``image top``) AND an MV_FWD/MV_BACK token are swapped. The ``{output_contract}`` token
list and the LEFT/RIGHT/UP/DOWN guidance are never matched.
"""
from __future__ import annotations

# Sentinel that cannot occur in the prompt, used to swap the two tokens in one pass
# without the second replace undoing the first.
_SENTINEL = "\x00"


class EgoPlugin:
    """Prompt-only FWD/BACK inversion for the egocentric wrist view."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def apply(self, prompt_template: str) -> str:
        """Return ``prompt_template`` unchanged, or with FWD/BACK swapped on the
        WRIST-section image-depth direction lines when enabled."""
        if not self.enabled:
            return prompt_template
        out = []
        in_wrist_section = False
        for line in prompt_template.split("\n"):
            lower = line.lower()
            # A guidance header ("... is the primary guide ...") opens the section it
            # names: the wrist rule (rule A) or the front rule (rule B / C ...).
            if "primary guide" in lower:
                in_wrist_section = "wrist" in lower
            names_edge = ("image bottom" in lower) or ("image top" in lower)
            names_depth = ("MV_FWD" in line) or ("MV_BACK" in line)
            swap = in_wrist_section and names_edge and names_depth
            out.append(_swap_fwd_back(line) if swap else line)
        return "\n".join(out)


def _swap_fwd_back(line: str) -> str:
    """Swap every MV_FWD<->MV_BACK on one line in a single pass."""
    return (
        line.replace("MV_FWD", _SENTINEL)
        .replace("MV_BACK", "MV_FWD")
        .replace(_SENTINEL, "MV_BACK")
    )
