"""AgileX Piper 6-DoF kinematics: forward kinematics + damped-least-squares IK.

Why this exists: the arm node's Cartesian command path (``/puppet/pos_cmd`` ->
``EndPoseCtrl``) is firmware **MOVE P** -- every command is an independently *planned*
point-to-point move (accelerate, travel, decelerate). Streaming interpolated Cartesian
waypoints down that path makes the arm execute a chain of micro-moves: slow AND jerky.
The path that IS smooth on this hardware is the joint one (``/master/joint_<arm>`` ->
``JointCtrl``, firmware MOVE J at speed 100) -- it is exactly what the vendor's own
master-slave teleop and ``replay_data.py`` stream at high rate, and what our homing
moves already use. To route Cartesian motion through it we need IK on our side; this
module provides it.

FK: the vendor's own Denavit-Hartenberg model, vendored from
``piper_sdk/kinematics/piper_fk.py`` (C_PiperForwardKinematics) and re-expressed in
numpy/SI units. Both DH variants ship (firmware >= V1.5-2 offsets joints 2/3 by 2 deg);
:func:`select_dh_variant` picks the one that matches the arm's own pose feedback, so we
never guess the firmware version.

IK: damped least squares on the 6-D pose error (position + orientation rotation-vector),
numeric Jacobian, seeded from the previous solution. Our steps are millimetres, so it
converges in a couple of iterations; joint limits (vendored from
``piper_sdk/piper_param``) are enforced every iterate. A non-converged solve returns
None -- callers fall back to the firmware MOVE P path rather than stream a bad target.

Bounded-orientation IK (:meth:`PiperKinematics.ik_bounded`): holding the FULL captured
orientation exactly is what actually bounds the Piper's usable workspace -- the wrist
pitch (j5, only +/-70 deg) must absorb all orientation correction as the arm geometry
changes, so it saturates centimetres from the start pose while the POSITION workspace
extends much further (measured from the calibrated begin pose: strict full-pose IK dies
after 3 cm of MV_UP; position-only IK reaches everywhere we tested). ``ik_bounded``
therefore solves position EXACTLY and lets the orientation bend as far as (and no
farther than) a caller-set budget toward the orientation the mechanism naturally adopts
there: strict solve first (zero deviation in the healthy workspace), then a
position-primary solve to discover the natural orientation, budget-clamped.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

PIPER_DOF = 6

# Joint limits (rad), vendored from piper_sdk/piper_param/piper_param_manager.py.
JOINT_LIMITS_RAD = np.array(
    [
        [-2.6179, 2.6179],   # j1  +/-150 deg
        [0.0, 3.14],         # j2  0..180 deg
        [-2.967, 0.0],       # j3  -170..0 deg
        [-1.745, 1.745],     # j4  +/-100 deg
        [-1.22, 1.22],       # j5  +/-70 deg
        [-2.09439, 2.09439], # j6  +/-120 deg
    ],
    dtype=float,
)
# Stay a hair inside the hard limits so a streamed target never trips the firmware's
# joint-limit error mid-motion.
JOINT_LIMIT_MARGIN_RAD = math.radians(0.5)

# --- Vendored DH tables (units: a/d in mm, alpha/theta-offset in rad) -----------
# piper_sdk C_PiperForwardKinematics, dh_is_offset=0x00 (older firmware) and 0x01
# (firmware >= V1.5-2: joints 2/3 offset by 2 deg).
_DH_VARIANTS = {
    0x00: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 174.22 / 180, -100.78 / 180 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
    0x01: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 172.22 / 180, -102.78 / 180 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
}


class PiperKinematics:
    """FK + DLS IK for one DH variant. Positions in meters, angles in radians."""

    def __init__(self, dh_variant: int = 0x01) -> None:
        if dh_variant not in _DH_VARIANTS:
            raise ValueError(f"dh_variant must be 0x00 or 0x01, got {dh_variant!r}")
        self.dh_variant = dh_variant
        dh = _DH_VARIANTS[dh_variant]
        self._a = np.asarray(dh["a"], dtype=float) / 1000.0  # mm -> m
        self._alpha = np.asarray(dh["alpha"], dtype=float)
        self._theta0 = np.asarray(dh["theta"], dtype=float)
        self._d = np.asarray(dh["d"], dtype=float) / 1000.0  # mm -> m

    # -- forward -------------------------------------------------------------
    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Joints (6, rad) -> (position [x,y,z] m, rotation matrix 3x3), base frame.

        Modified-DH chain exactly as the vendor FK builds it; the firmware's own pose
        feedback (``/puppet/end_pose_euler``) is this transform's translation + euler.
        """
        q = np.asarray(q, dtype=float).reshape(-1)[:PIPER_DOF]
        T = np.eye(4)
        for i in range(PIPER_DOF):
            ct = math.cos(q[i] + self._theta0[i])
            st = math.sin(q[i] + self._theta0[i])
            ca = math.cos(self._alpha[i])
            sa = math.sin(self._alpha[i])
            Ti = np.array(
                [
                    [ct, -st, 0.0, self._a[i]],
                    [st * ca, ct * ca, -sa, -sa * self._d[i]],
                    [st * sa, ct * sa, ca, ca * self._d[i]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            T = T @ Ti
        return T[:3, 3].copy(), T[:3, :3].copy()

    def fk_pose7(self, q: np.ndarray) -> np.ndarray:
        """Joints -> [x,y,z,qx,qy,qz,qw] (the interface's pose convention)."""
        pos, rot = self.fk(q)
        return np.concatenate([pos, R.from_matrix(rot).as_quat()])

    # -- inverse -------------------------------------------------------------
    def ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
        pos_tol_m: float = 5e-4,
        ori_tol_rad: float = math.radians(0.5),
        max_iters: int = 16,
        damping: float = 0.05,
        max_seed_dist_rad: float = 0.35,
        _retry: bool = True,
    ) -> Optional[np.ndarray]:
        """Damped-least-squares IK. Returns joints (6, rad) or None (no convergence).

        Seeded from ``q_seed`` (the previous streamed solution / measured joints);
        consecutive targets are millimetres apart, so convergence is 1-3 iterations.
        Joint limits are clamped every iterate; None means "do not stream this" and the
        caller falls back to the firmware Cartesian path. A solve that stalls under the
        default damping is retried once with light damping (rescues slow convergence
        near a singular direction without destabilizing the common case).

        ``max_seed_dist_rad`` is a CONTINUITY guard: a converged solution farther than
        this from the seed (any joint) is rejected. Streamed waypoints are millimetres
        apart, so a legitimate solution is always near the seed; a distant one means
        DLS wandered to a different IK branch (e.g. a wrist flip near the j5=0
        singularity), and streaming it as a single MOVE J would swing the arm.
        """
        q = np.clip(
            np.asarray(q_seed, dtype=float).reshape(-1)[:PIPER_DOF].copy(),
            JOINT_LIMITS_RAD[:, 0] + JOINT_LIMIT_MARGIN_RAD,
            JOINT_LIMITS_RAD[:, 1] - JOINT_LIMIT_MARGIN_RAD,
        )
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3)

        for _ in range(max_iters):
            pos, rot = self.fk(q)
            e_pos = target_pos - pos
            e_ori = R.from_matrix(target_rot @ rot.T).as_rotvec()
            if float(np.linalg.norm(e_pos)) < pos_tol_m and float(np.linalg.norm(e_ori)) < ori_tol_rad:
                return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
            err = np.concatenate([e_pos, e_ori])

            J = self._jacobian(q)
            # dq = J^T (J J^T + lambda^2 I)^-1 err  -- damped, singularity-safe.
            JJt = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Trust region: a small target step never needs a big joint step; cap so a
            # near-singular pose cannot fling a joint.
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(
                q + dq,
                JOINT_LIMITS_RAD[:, 0] + JOINT_LIMIT_MARGIN_RAD,
                JOINT_LIMITS_RAD[:, 1] - JOINT_LIMIT_MARGIN_RAD,
            )

        # Final check (the loop may have converged on its last update).
        pos, rot = self.fk(q)
        if (
            float(np.linalg.norm(target_pos - pos)) < pos_tol_m
            and float(np.linalg.norm(R.from_matrix(target_rot @ rot.T).as_rotvec())) < ori_tol_rad
        ):
            return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
        if _retry:
            # One lightly-damped attempt from where the first solve stalled.
            return self.ik(
                target_pos, target_rot, q,
                pos_tol_m=pos_tol_m, ori_tol_rad=ori_tol_rad,
                max_iters=max_iters, damping=0.01,
                max_seed_dist_rad=max_seed_dist_rad, _retry=False,
            )
        return None

    def ik_position(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        pos_tol_m: float = 5e-4,
        max_iters: int = 24,
        damping: float = 0.03,
        max_seed_dist_rad: float = 0.35,
    ) -> Optional[np.ndarray]:
        """Position-only DLS IK: reach ``target_pos`` with whatever orientation results.

        The redundancy (6 joints, 3 constraints) resolves to the minimum-norm joint
        step each iterate, so consecutive millimetre targets stay on one smooth branch;
        the same seed-continuity guard as :meth:`ik` rejects a wandered solution.
        Used by :meth:`ik_bounded` to discover the orientation the mechanism naturally
        adopts at a position the strict full-pose solve cannot reach.
        """
        q = np.clip(
            np.asarray(q_seed, dtype=float).reshape(-1)[:PIPER_DOF].copy(),
            JOINT_LIMITS_RAD[:, 0] + JOINT_LIMIT_MARGIN_RAD,
            JOINT_LIMITS_RAD[:, 1] - JOINT_LIMIT_MARGIN_RAD,
        )
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        for _ in range(max_iters):
            pos, _ = self.fk(q)
            e_pos = target_pos - pos
            if float(np.linalg.norm(e_pos)) < pos_tol_m:
                return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
            J = self._jacobian(q)[:3]
            dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(3), e_pos)
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(
                q + dq,
                JOINT_LIMITS_RAD[:, 0] + JOINT_LIMIT_MARGIN_RAD,
                JOINT_LIMITS_RAD[:, 1] - JOINT_LIMIT_MARGIN_RAD,
            )
        pos, _ = self.fk(q)
        if float(np.linalg.norm(target_pos - pos)) < pos_tol_m:
            return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
        return None

    def ik_bounded(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
        max_ori_dev_rad: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Position-exact IK with the orientation held within ``max_ori_dev_rad`` of
        ``target_rot``. Returns ``(joints, ori_deviation_rad)`` or ``(None, 0.0)``.

        Cascade (each stage is the proven DLS solver, just re-targeted):
          1. strict full-pose :meth:`ik` -- zero deviation; taken everywhere the exact
             orientation is reachable, so the healthy workspace behaves as before.
          2. :meth:`ik_position` discovers the orientation the mechanism NATURALLY
             adopts at that position (typically a few degrees of wrist-pitch give near
             a joint limit). Within budget -> accepted as-is.
          3. Natural deviation over budget -> the reference orientation is bent by
             EXACTLY the budget toward the natural one (partial rotation of the
             deviation rotvec) and the full-pose solve retried at that target (2 deg
             slack: the bent orientation is only approximately achievable).
        ``(None, 0.0)`` means the POSITION itself is unreachable (workspace edge /
        joint limits) or off-branch -- the true boundary of the budgeted action space.
        """
        budget = max(0.0, float(max_ori_dev_rad))
        sol = self.ik(target_pos, target_rot, q_seed)
        if sol is not None:
            return sol, 0.0
        if budget <= 0.0:
            return None, 0.0
        q_free = self.ik_position(target_pos, q_seed)
        if q_free is None:
            return None, 0.0
        _, rot_free = self.fk(q_free)
        dev_vec = R.from_matrix(rot_free @ np.asarray(target_rot, float).T).as_rotvec()
        dev = float(np.linalg.norm(dev_vec))
        if dev <= budget:
            return q_free, dev
        bent_rot = R.from_rotvec(dev_vec / dev * budget).as_matrix() @ target_rot
        sol = self.ik(target_pos, bent_rot, q_seed, ori_tol_rad=math.radians(2.0))
        if sol is not None:
            return sol, budget
        return None, 0.0

    @staticmethod
    def _near_seed(q: np.ndarray, q_seed: np.ndarray, max_dist_rad: float) -> bool:
        if max_dist_rad <= 0.0:
            return True
        seed = np.asarray(q_seed, dtype=float).reshape(-1)[:PIPER_DOF]
        return float(np.max(np.abs(q - seed))) <= max_dist_rad

    def _jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numeric 6x6 Jacobian d[pos, rotvec]/dq (forward differences)."""
        pos0, rot0 = self.fk(q)
        J = np.zeros((6, PIPER_DOF))
        for i in range(PIPER_DOF):
            qp = q.copy()
            qp[i] += eps
            pos1, rot1 = self.fk(qp)
            J[:3, i] = (pos1 - pos0) / eps
            J[3:, i] = R.from_matrix(rot1 @ rot0.T).as_rotvec() / eps
        return J


def select_dh_variant(
    q_measured: np.ndarray, pose_measured7: np.ndarray
) -> Tuple[PiperKinematics, float]:
    """Pick the DH variant whose FK best matches the arm's OWN pose feedback.

    Returns ``(kinematics, position_error_m)`` for the better variant. The caller
    should refuse the joint-stream backend when the error is large (wrong model,
    uncalibrated arm) rather than stream IK targets computed from a bad FK.
    """
    q = np.asarray(q_measured, dtype=float).reshape(-1)[:PIPER_DOF]
    meas = np.asarray(pose_measured7, dtype=float).reshape(-1)[:7]
    best: Tuple[Optional[PiperKinematics], float] = (None, float("inf"))
    for variant in _DH_VARIANTS:
        kin = PiperKinematics(variant)
        pos, _ = kin.fk(q)
        err = float(np.linalg.norm(pos - meas[:3]))
        if err < best[1]:
            best = (kin, err)
    assert best[0] is not None
    return best[0], best[1]
