"""4-dim ``pd_ee_delta_pos`` atomic controller for ManiSkill.

Sibling of :class:`interpreters.atomic_controller.AtomicController`, but for ManiSkill's
translation-only ``pd_ee_delta_pos`` control mode. That mode is exactly the "rotation is
locked, only XYZ translation" controller the MVTOKEN atomic-token policy assumes, so the
mapping is deliberately minimal: there is no yaw axis and no
agentview screen->base view mapping -- each ``MV_*`` token is a fixed base-frame step of
``step_m`` metres.

Action layout (matches ``gym.make(..., control_mode="pd_ee_delta_pos")`` on a Panda):

    [dx, dy, dz, gripper]   each in [-1, 1]

``dx/dy/dz`` are a delta-position target normalised by ``delta_bound_m`` (the arm's
``pos_upper``; 0.1 m on the stock Panda), so ``action == 1.0`` commands a
``delta_bound_m``-metre nudge in that axis per control step. ``gripper`` is the Panda mimic
command: ``open_gripper_action`` (+1 -> open) / ``close_gripper_action`` (-1 -> close).

The six ``move_vectors`` (base-frame unit vectors per token) come from config so the
axis/sign mapping stays tunable: ManiSkill's world frame at reset is +X forward (away from
the base), +Y robot-left, +Z up (verified by an axis probe). Flip a sign in
``configs/robot_maniskill.yaml`` if the zero-shot moves come out mirrored relative to how
the camera sees the scene.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.action_units import MOVE_ATOMS
from interpreters.sim_state import AtomicControllerState


class ManiskillAtomicController:
    """MVTOKEN atomic token -> 4-dim ``pd_ee_delta_pos`` action for a ManiSkill Panda."""

    def __init__(
        self,
        move_vectors: Mapping[str, Sequence[float]],
        step_m: float,
        delta_bound_m: float,
        open_gripper_action: float = 1.0,
        close_gripper_action: float = -1.0,
    ) -> None:
        missing = [name for name in MOVE_ATOMS if name not in move_vectors]
        if missing:
            raise ValueError(f"move_vectors is missing tokens: {missing}")
        self.move_vectors = {
            name: np.asarray(move_vectors[name], dtype=float) for name in MOVE_ATOMS
        }
        self.step_m = float(step_m)
        self.delta_bound_m = float(delta_bound_m)
        self.open_gripper_action = float(open_gripper_action)
        self.close_gripper_action = float(close_gripper_action)
        self.state = AtomicControllerState(
            gripper_command=self.open_gripper_action,
            gripper_name="OPEN",
        )

    # -- action construction ------------------------------------------------
    def _action(self, delta_m: np.ndarray) -> np.ndarray:
        normalized = np.clip(
            np.asarray(delta_m, dtype=float) / self.delta_bound_m, -1.0, 1.0
        )
        return np.asarray(
            [normalized[0], normalized[1], normalized[2], self.state.gripper_command],
            dtype=np.float32,
        )

    def action_for_atomic(self, token: str) -> np.ndarray:
        """One MV_* token -> a fixed ``step_m`` base-frame nudge (gripper held)."""
        if token not in self.move_vectors:
            raise ValueError(f"Unknown move token {token!r}; expected one of {MOVE_ATOMS}")
        self.state.last_atomic = token
        return self._action(self.move_vectors[token] * self.step_m)

    def hold_action(self) -> np.ndarray:
        """Zero translation, current gripper command (used to settle / open / close)."""
        self.state.last_atomic = None
        return self._action(np.zeros(3, dtype=float))

    def open_gripper(self) -> np.ndarray:
        self.state.gripper_command = self.open_gripper_action
        self.state.gripper_name = "OPEN"
        return self.hold_action()

    def close_gripper(self) -> np.ndarray:
        self.state.gripper_command = self.close_gripper_action
        self.state.gripper_name = "CLOSE"
        return self.hold_action()
