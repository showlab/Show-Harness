"""Variable step-size tool: a controller *motion-magnitude* provider.

The controller normally moves a fixed ``step_m`` per atomic token. When this tool is
enabled it returns a COARSE step (e.g. 5 cm) instead of the fine ``step_m`` in these cases:

  HEIGHT-BASED (strictly followed):
    * the token is ``MV_UP`` -- lift clear of the table quickly; and
    * the gripper is high above the table (gap > ``high_above_table_m``) -- a coarse
      approach from altitude, paired with the proprioception "descend first" hint.

  WRIST-VISIBILITY (the VLM's distance signal):
    * the TARGET is NOT yet visible in the wrist view (``target_in_wrist`` False) -- the
      gripper is still far, so close distance with a big step. ``target_in_wrist`` is the
      shared wrist-visibility judgment produced by :mod:`core.prompting.wrist_marker` (the VLM's
      ``WRIST: YES/NO`` marker, rendered/parsed by the controller agent and forwarded by the
      runner); this tool only consumes it.

Effective rule: coarse if MV_UP OR gap > X OR (TARGET not in wrist); otherwise the fine
``default_step_m``. Once the TARGET is in the wrist view (close) the step is fine for
precise alignment. The whole behaviour is controlled by the single ``enabled`` flag.
"""
from __future__ import annotations

from typing import Optional


MV_UP = "MV_UP"


class VariableStepPlugin:
    """Pick the per-command translation magnitude from height, token, and wrist visibility."""

    def __init__(
        self,
        enabled: bool = False,
        coarse_step_m: float = 0.05,
        high_above_table_m: float = 0.10,
    ) -> None:
        self.enabled = bool(enabled)
        self.coarse_step_m = max(0.0, float(coarse_step_m))
        self.high_above_table_m = max(0.0, float(high_above_table_m))

    def step_m_for(
        self,
        token: str,
        default_step_m: float,
        eef_height_m: Optional[float] = None,
        table_height_m: Optional[float] = None,
        target_in_wrist: Optional[bool] = None,
    ) -> float:
        """Return the translation magnitude (meters) to use for ``token``.

        Coarse when lifting (MV_UP), when high above the table (height rules), or when the
        TARGET is not yet in the wrist view (``target_in_wrist`` is False). Otherwise the
        fine ``default_step_m``. ``target_in_wrist`` None (no marker) -> treated as fine.
        """
        default = float(default_step_m)
        if not self.enabled:
            return default
        # --- Height-based rules (strictly followed) ---
        if str(token or "").strip().upper() == MV_UP:
            return self.coarse_step_m
        if eef_height_m is not None and table_height_m is not None:
            try:
                gap = float(eef_height_m) - float(table_height_m)
            except (TypeError, ValueError):
                gap = None
            if gap is not None and gap > self.high_above_table_m:
                return self.coarse_step_m
        # --- Wrist-visibility rule: far (TARGET not in the wrist view) -> big step ---
        if target_in_wrist is False:
            return self.coarse_step_m
        return default
