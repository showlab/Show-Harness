"""7-dim relative differential-IK atomic controller for RoboLab (Isaac Lab).

Sibling of
:class:`interpreters.maniskill_atomic_controller.ManiskillAtomicController` (ManiSkill,
``pd_ee_delta_pos``). RoboLab's ``DroidRelIKActionCfg`` action layout is::

    [dx, dy, dz, drx, dry, drz, gripper]

``dx/dy/dz`` are a base-frame end-effector displacement **divided by the action config's
``scale``** (RoboLab ships 0.5, i.e. an action of 1.0 asks for a 0.5 m IK target step), and
``drx/dry/drz`` are an axis-angle rotation delta. MVTOKEN holds the three rotation DOFs at
exactly zero, which makes the relative-IK controller re-target the CURRENT orientation
every step -- the "rotation is locked, only XYZ translation" contract the atomic-token
policy assumes, same as ManiSkill's ``pd_ee_delta_pos``.

Two differences from the ManiSkill controller, both easy to get wrong:

* **Gripper polarity is inverted.** ManiSkill's Panda mimic gripper is +1 open / -1 close.
  RoboLab drives a Robotiq 2F-85 through ``BinaryJointPositionZeroToOneAction``, whose rule
  is ``action > 0.5 -> CLOSE``: 1.0 closes, 0.0 opens. The defaults below encode that;
  flipping them would make every GRASP release and every RELEASE grab.
* **The delta is a per-control-step IK target, not a normalised velocity.** The action is
  divided by ``ik_scale`` rather than by a controller bound, and each ``env.step``
  re-issues "current pose + delta". A decision therefore travels ``step_m`` in total by
  splitting it over ``sim_steps_per_decision`` steps (the runner's job); ``max_delta_m``
  here is only a safety cap so a mis-set config cannot ask the IK solver for a lunge it
  will diverge on.

The six ``move_vectors`` (base-frame unit vector per token) come from config so the
axis/sign mapping stays tunable. RoboLab's Franka sits at the origin with +X forward,
+Y robot-left, +Z up, matching ``configs/primitives_franka.yaml`` -- but confirm it with
``scripts/run_robolab_mvtoken.py --probe-axes`` before trusting a rollout, exactly as on ManiSkill.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.action_units import MOVE_ATOMS
from interpreters.sim_state import AtomicControllerState
from core.sim.robolab_task import hold_orientation_rotvec

# RoboLab's BinaryJointPositionZeroToOneAction: ``actions > 0.5`` -> close command.
OPEN_GRIPPER_ACTION = 0.0
CLOSE_GRIPPER_ACTION = 1.0


class RobolabAtomicController:
    """MVTOKEN atomic token -> 7-dim relative-IK action for a RoboLab Franka/Robotiq."""

    def __init__(
        self,
        move_vectors: Mapping[str, Sequence[float]],
        step_m: float,
        ik_scale: float = 0.5,
        sim_steps_per_decision: int = 1,
        max_delta_m: float = 0.05,
        open_gripper_action: float = OPEN_GRIPPER_ACTION,
        close_gripper_action: float = CLOSE_GRIPPER_ACTION,
    ) -> None:
        missing = [name for name in MOVE_ATOMS if name not in move_vectors]
        if missing:
            raise ValueError(f"move_vectors is missing tokens: {missing}")
        self.move_vectors = {
            name: np.asarray(move_vectors[name], dtype=float) for name in MOVE_ATOMS
        }
        self.step_m = float(step_m)
        self.ik_scale = float(ik_scale)
        if self.ik_scale <= 0.0:
            raise ValueError(f"ik_scale must be > 0, got {ik_scale!r}")
        self.sim_steps_per_decision = max(1, int(sim_steps_per_decision))
        self.max_delta_m = float(max_delta_m)
        self.open_gripper_action = float(open_gripper_action)
        self.close_gripper_action = float(close_gripper_action)
        self.state = AtomicControllerState(
            gripper_command=self.open_gripper_action,
            gripper_name="OPEN",
        )
        # EE orientation every step is pulled back to; set by the runner at reset.
        self._quat_ref = None

    # -- geometry -----------------------------------------------------------
    @property
    def per_step_m(self) -> float:
        """Metres commanded per control step so one decision totals ``step_m``."""
        return self.step_m / self.sim_steps_per_decision

    # -- action construction ------------------------------------------------
    def _action(self, delta_m: np.ndarray) -> np.ndarray:
        delta = np.clip(
            np.asarray(delta_m, dtype=float), -self.max_delta_m, self.max_delta_m
        )
        scaled = delta / self.ik_scale
        return np.asarray(
            [
                scaled[0],
                scaled[1],
                scaled[2],
                # Rotation slots are filled in per control step by
                # with_orientation_hold(); zeros here are only the un-referenced
                # fallback. Zeros do NOT lock the orientation -- see that method.
                0.0,  # drx
                0.0,  # dry
                0.0,  # drz
                self.state.gripper_command,
            ],
            dtype=np.float32,
        )

    def set_orientation_reference(self, quat_wxyz) -> None:
        """Latch the orientation the episode is to be held at (None disables the hold)."""
        self._quat_ref = (
            None if quat_wxyz is None else np.asarray(quat_wxyz, dtype=np.float64)
        )

    def with_orientation_hold(self, action: np.ndarray, quat_cur) -> np.ndarray:
        """Fill ``action``'s rotation slots with the correction back to the reference.

        Must be applied PER CONTROL STEP, not once per decision: the correction is a
        function of the CURRENT orientation, so re-sending a stale one keeps commanding a
        rotation the arm has already made.

        Zeros in those slots do not mean "hold the orientation" -- RoboLab's relative IK
        reads them as "target whatever orientation you have now", so the solver's own error
        becomes the next setpoint and accumulates. Measured with zeros on RubiksCubeTask:
        the gripper left reset vertical and released 24.5 deg off, with no error raised and
        every displacement statistic in range. The generator applies the identical
        correction via the same
        :func:`core.sim.robolab_task.hold_orientation_rotvec`; keeping ONE implementation
        is what stops served rollouts drifting from the data the policy was trained on.
        """
        if self._quat_ref is None:
            return action
        out = np.array(action, dtype=np.float32, copy=True)
        out[3:6] = hold_orientation_rotvec(self._quat_ref, quat_cur) / self.ik_scale
        return out

    def action_for_atomic(self, token: str) -> np.ndarray:
        """One MV_* token -> the per-control-step share of a ``step_m`` base-frame nudge.

        The returned action is meant to be re-sent for ``sim_steps_per_decision`` steps;
        together they add up to ``step_m`` metres.
        """
        if token not in self.move_vectors:
            raise ValueError(f"Unknown move token {token!r}; expected one of {MOVE_ATOMS}")
        self.state.last_atomic = token
        return self._action(self.move_vectors[token] * self.per_step_m)

    def hold_action(self) -> np.ndarray:
        """Zero displacement, current gripper command (used to settle / open / close)."""
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
