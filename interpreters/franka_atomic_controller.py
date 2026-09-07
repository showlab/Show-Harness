"""Franka real-robot atomic-action controller.

The arm-agnostic token -> Cartesian-setpoint machinery lives in
:class:`interpreters.real_atomic_controller.RealAtomicController`; this module adds
what is specific to the Franka + Polymetis stack:

  * the physical calibration constants measured on THIS Franka + table
    (``TABLE_CONTACT_Z_M``, ``EMPTY_GRASP_WIDTH_M``),
  * recognition of the polymetis "no controller running" errors so the
    ``ensure_controller`` hook (``FrankaSession.start_impedance``) can self-heal a
    setpoint command after a joint move / server-side reflex killed the
    Cartesian-impedance controller.

Axis convention (Franka base frame, left/right verified on the real robot):

    MV_FWD  -> +X      MV_BACK  -> -X
    MV_LEFT -> -Y      MV_RIGHT -> +Y
    MV_UP   -> +Z      MV_DOWN  -> -Z
    ROTATE_CCW -> +yaw ROTATE_CW -> -yaw
    GRASP -> close gripper   RELEASE -> open gripper   DONE -> terminate

The actual MV_* unit vectors come from ``configs/primitives_franka.yaml`` so the mapping
stays the single source of truth; the lines above just document the current signs.

Everything historically importable from this module (token constants, thresholds,
``AtomicStepResult``, ``quat_to_euler`` ...) is re-exported so existing consumers
(scripts/run_real.py, core/runners/real.py, teleop, tests) keep working unchanged.
"""
from __future__ import annotations

# Re-exports: the shared vocabulary, types and helpers historically defined here.
from core.action_units import (  # noqa: F401
    MOVE_ATOMS,
    ROTATE_ATOMS,
    STOP_ATOM,
)
from interpreters.real_atomic_controller import (  # noqa: F401
    DEFAULT_MAX_POSITION_DELTA_M,
    DEFAULT_MAX_ROTATION_DELTA_RAD,
    DONE_ATOM,
    GRASP_ATOM,
    GRIPPER_ATOMS,
    GRIPPER_CLOSE_THRESHOLD_M,
    GRIPPER_MOTION_EPS_M,
    GRIPPER_STABLE_EPS_M,
    GRIPPER_SUSTAIN_S,
    RELEASE_ATOM,
    AtomicStepResult,
    RealAtomicController,
    euler_to_quat,
    quat_to_euler,
)

# --- Physical calibration (this Franka + table) -----------------------------
# Measured 2026-06-20 from the live robot with the EMPTY gripper fully closed and
# resting on the tabletop (cross-checked against data/teleoperation/test rollouts):
#   * EEF base-frame Z at table contact      = 0.1539 m  (stable to <0.01 mm)
#   * empty-closed gripper width              = 0.0002 m  (fingers touching)
#   * a held block read                       ~ 0.039 m
#   * fully-open width                        ~ 0.079-0.080 m
#
# TABLE_CONTACT_Z_M is the lowest EEF height the gripper should ever reach: at this
# height the fingertips just touch the table, so objects resting on it can still be
# grasped (rollout_000 grasped a block at Z=0.154) but the arm cannot be driven into
# the table. Used as the default Z safety floor for BOTH rollouts and teleop capture.
TABLE_CONTACT_Z_M = 0.154

# EMPTY_GRASP_WIDTH_M: a GRASP that settles at/below this width caught nothing and is
# reopened. The empty close reads ~0.0002 m, the thinnest object we expect to grasp is
# well above this, so 0.005 m (25x the empty width) flags an empty close robustly
# without false-flagging a real (thin) grasp.
EMPTY_GRASP_WIDTH_M = 0.005


def _is_controller_lost_error(exc: Exception) -> bool:
    """True when a setpoint command failed because no controller is running on the
    polymetis server (e.g. after a joint move preempted impedance, or a reflex
    terminated it). The grpc/zerorpc error text propagates to the client, so match it."""
    msg = str(exc).lower()
    return (
        "no controller running" in msg
        or "with no controller" in msg
        or "start_joint_impedance" in msg
        or "start_cartesian_impedance" in msg
    )


class FrankaAtomicController(RealAtomicController):
    """Maps atomic VLM tokens to Franka Cartesian-impedance setpoints.

    Expects a ``FrankaInterface`` or ``MockRobot`` (core.franka.franka_interface)
    as the ``robot`` and ``FrankaSession.start_impedance`` as the ``ensure_controller``
    recovery hook.
    """

    LOG_TAG = "franka-atomic"

    def _is_recoverable_command_error(self, exc: Exception) -> bool:
        # A lost Cartesian-impedance controller is restored by the ensure_controller
        # hook (terminate-then-start impedance) and the setpoint retried.
        return _is_controller_lost_error(exc)
