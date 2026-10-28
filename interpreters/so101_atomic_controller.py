"""SO-101 token-level arm controller, independent of where the arm lives.

One controller for every place the arm lives: the Isaac Lab Workshop sim
(scripts/trajectory/real2sim/backends/so101_workshop.py), a real LeRobot follower
(core/so101/real_backend.py, docs/so101.md) and a kinematic stand-in for unit tests (tests/test_so101_*.py).
Subclasses provide only I/O: read the joints, send one tick of joint targets, and the
world<-base rotation. Every decision about MOTION lives here, so what is verified in sim is
what runs on hardware. Nothing here imports a simulator or ``scripts/``.

The backend side of ``AtomicSimEnv`` (``tcp_pos / tcp_pose7 / gripper_width / apply_delta``)
is implemented here, so a sim backend only has to add cameras and success. It does NOT
claim ``supports_rotation``: that contract is an arbitrary world-axis rotation, which a
5-DoF arm cannot hold. RT_* units are interpreted by :class:`So101AtomicExec` instead (see
below), through :meth:`So101ArmController.apply_delta_pitch_yaw`.

Frames
------
TASK frame: world orientation, origin at the robot base. +X from the robot toward the
vials/rack, +Z up. MV_* tokens are unit steps along its axes (configs/primitives_so101.yaml). FK/IK work in the
base-link frame (yawed +90 deg in the Workshop world); conversion happens here.

Control
-------
One CONTROL step = ``ticks_per_ctrl`` ticks (a tick is one env.step, 60 Hz). Each control
step: TCP target = measured TCP + capped delta; IK from the measured joints holding the
references (tool pitch in the arm plane, plus EITHER the Wrist_Roll joint angle ``roll_ref``
(default) OR the closing-axis yaw ``yaw_ref``).

Which second reference: Pitch, Elbow and Wrist_Pitch are parallel, so at the constant roll
ROLL_CLOSING_HORIZONTAL (= -pi/2, measured) the jaw closes along the arm-plane normal, i.e.
HORIZONTALLY, in every pose. Holding the yaw instead keeps the closing direction's compass
heading but tilts it out of horizontal as the base turns (13-22 deg measured over the vial
area at pitch 47): the fixed finger dips and grasps fail. So roll is the default. The SO-101's PD gains are low, so gravity leaves a steady-state joint
offset; it is learned whenever the arm is at rest and added to the command (``q_bias``).

Exactness
---------
``feasible`` answers "can the hand be put THERE with THIS orientation, within tolerance and
off the joint limits" before a token runs; So101AtomicExec refuses tokens that fail it, so a
recorded token always means the motion it names.

Every tick is offered to ``step_hooks`` (video, trajectory recorder) with observed AND
commanded joints.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import yaml

from core.action_units import MOVE_ATOMS, RT_ATOMS
from interpreters.so101_kinematics import So101Kinematics, mat_to_quat_wxyz

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_HULL = CONFIGS / "so101_gripper_hull.json"
DEFAULT_PRIMITIVES = CONFIGS / "primitives_so101.yaml"

# Wrist_Roll at which the jaw's closing axis is the arm-plane normal (horizontal in any pose).
# Measured on the Workshop chain (scripts: tests/test_kinematics.py checks it).
ROLL_CLOSING_HORIZONTAL = -np.pi / 2


class So101ArmController(abc.ABC):
    delta_bound_m = 0.02
    supports_rotation = False  # no arbitrary world-axis rotation on 5 DoF; see module doc

    def __init__(
        self,
        kin: Optional[So101Kinematics] = None,
        ticks_per_ctrl: int = 2,
        tick_dt: float = 1.0 / 60.0,
        jaw_open_rad: float = 0.9,
        jaw_close_rad: float = -0.17,
        bias_rest_tol_rad: float = 2e-3,
        bias_alpha: float = 0.5,
        use_gravity_bias: bool = True,
        bias_max_rad: float = 0.08,
        bias_rate_rad: float = 0.01,
        contact_tol_rad: float = 0.08,
        fixed_finger_face_m: float = 0.007,
        hand_floor_task_z: Optional[float] = None,
        hull_path: Optional[Path] = DEFAULT_HULL,
        workspace: Optional[dict] = None,
        tol_pos_m: float = 1e-3,
        tol_ori_rad: float = float(np.radians(1.0)),
        min_joint_margin_rad: float = 0.02,
    ) -> None:
        self.kin = kin or So101Kinematics()
        self.ticks_per_ctrl = int(ticks_per_ctrl)
        self.tick_dt = float(tick_dt)
        self.jaw_open_rad = float(jaw_open_rad)
        self.jaw_close_rad = float(jaw_close_rad)
        self.bias_rest_tol_rad = float(bias_rest_tol_rad)
        self.bias_alpha = float(bias_alpha)
        self.use_gravity_bias = bool(use_gravity_bias)
        # Gravity sag measured on the Workshop SO-101 is <= 0.018 rad (the real arm may sag more
        # at full reach). The bias is learned at rest, rate-limited and bounded. It can no
        # longer cause a runaway (commands are relative to q_des, not the measured joints);
        # contact is reported by ``in_contact`` and the executor undoes what a push taught.
        self.bias_max_rad = float(bias_max_rad)
        self.bias_rate_rad = float(bias_rate_rad)
        self.contact_tol_rad = float(contact_tol_rad)
        # The SO-101 has ONE moving jaw. The TCP (lerobot URDF gripper_frame) is on the FIXED
        # finger; an object is held on the moving-jaw side of it, along the closing axis.
        # Measured from FK: closed moving-jaw tip at +0.007 m from the TCP along gripper +x.
        self.fixed_finger_face_m = float(fixed_finger_face_m)
        # True when in_contact() reads a physical sensor (sim); False = inferred from the joint
        # gap (real arm without force sensing), which then also needs "no progress".
        self.contact_is_measured = False
        # Floor for the WHOLE hand, not just the TCP: the lowest point of the gripper + jaw
        # collision hulls must stay above it (the fingertip region reaches 5.5 mm below the
        # TCP at pitch 47; a box model said 23 mm and was wrong). None = not checked.
        self.hand_floor_task_z = hand_floor_task_z
        self._hull = None
        if hull_path is not None and Path(hull_path).exists():
            h = json.loads(Path(hull_path).read_text())
            self._hull = {k: np.asarray(h[k], dtype=np.float64) for k in ("gripper", "jaw")}
        self.step_hooks: list[Callable[[dict], None]] = []
        # Labels for the current activity (token, phase, blocked reason): video + log.
        self.annotation: dict = {}
        # TASK-frame box; a token whose end point leaves it is refused.
        self.workspace = workspace or {"min": [0.05, -0.30, 0.0], "max": [0.45, 0.25, 0.30]}
        self.tol_pos_m = float(tol_pos_m)
        self.tol_ori_rad = float(tol_ori_rad)
        self.min_joint_margin_rad = float(min_joint_margin_rad)
        self.q_des = np.zeros(6)   # IK solution: where the arm should be
        self.q_bias = np.zeros(6)  # learned gravity sag, added to the command
        self.pitch_ref = 0.0
        # Small integral trim on the IK pitch target, set by the executor while settling: the
        # gravity bias leaves ~0.2 deg of pitch at full reach (measured). Never a reference.
        self.pitch_trim = 0.0
        self.yaw_ref: Optional[float] = None  # set: hold the closing-axis yaw (via Wrist_Roll)
        self.roll_ref: float = ROLL_CLOSING_HORIZONTAL  # used while yaw_ref is None
        self.tick_count = 0
        self.ctrl_step_count = 0
        self.decision_id: Optional[int] = None
        self.last_tcp_target_task: Optional[np.ndarray] = None

    # ------------------------------------------------------------ I/O (subclass)
    @abc.abstractmethod
    def _read_q(self) -> np.ndarray:
        """Measured joints, rad, Workshop order (Rotation..Jaw)."""

    @abc.abstractmethod
    def _send(self, q_cmd: np.ndarray) -> None:
        """Apply ``q_cmd`` for ONE tick and return when the tick is over."""

    @abc.abstractmethod
    def _r_world_base(self) -> np.ndarray:
        """3x3 rotation of the robot base link in the task/world frame."""

    def grasped(self) -> bool:
        return False

    def raw_frames(self):
        raise NotImplementedError

    # ------------------------------------------------------------ frames
    def _base_to_task(self, v: np.ndarray) -> np.ndarray:
        return self._r_world_base() @ v

    def _task_to_base(self, v: np.ndarray) -> np.ndarray:
        return self._r_world_base().T @ v

    # ------------------------------------------------------------ readback
    def q_obs(self) -> np.ndarray:
        return np.asarray(self._read_q(), dtype=np.float64)

    def tcp_pos(self) -> np.ndarray:
        return self._base_to_task(self.kin.tcp(self.q_obs())[0])

    def tcp_rotmat(self) -> np.ndarray:
        return self._r_world_base() @ self.kin.tcp(self.q_obs())[1]

    def tcp_pose7(self) -> list[float]:
        return [*self.tcp_pos().tolist(), *mat_to_quat_wxyz(self.tcp_rotmat()).tolist()]

    def gripper_width(self) -> float:
        """Approximate fingertip opening (m): chord of the ~0.08 m moving jaw swung from its
        closed limit. Good enough for empty-grasp detection; not a contact measurement."""
        jaw = self.q_obs()[5]
        return float(2 * 0.08 * np.sin(max(jaw - self.kin.lower[5], 0.0) / 2))

    def orientation_error(self) -> tuple[float, float]:
        """(pitch, yaw) error: references minus the measured hand, rad."""
        q = self.q_obs()
        dp = float((self.pitch_ref - self.kin.tool_pitch(q) + np.pi) % (2 * np.pi) - np.pi)
        dy = 0.0
        if self.yaw_ref is not None:
            dy = float((self.yaw_ref - self.kin.tool_yaw(q)[0] + np.pi / 2) % np.pi - np.pi / 2)
        else:
            dy = float(self.roll_ref - q[4])  # roll mode: the second reference is the joint
        return dp, dy

    # ------------------------------------------------------------ actuation
    def _tick(self, q_cmd: np.ndarray) -> None:
        self._send(q_cmd)
        self.tick_count += 1
        if self.step_hooks:
            rec = {
                "env_step": self.tick_count,
                "t": self.tick_count * self.tick_dt,
                "ctrl_step": self.ctrl_step_count,
                "decision_id": self.decision_id,
                "q_obs": self.q_obs(),
                "q_cmd": np.asarray(q_cmd, dtype=np.float64).copy(),  # what was sent
                "q_des": self.q_des.copy(),
                "tcp_obs": self.tcp_pose7(),
                "tcp_target": None if self.last_tcp_target_task is None
                else self.last_tcp_target_task.tolist(),
                "pitch_ref": self.pitch_ref,
                "yaw_ref": self.yaw_ref,
                "roll_ref": self.roll_ref,
                "annotation": dict(self.annotation),
                "contacts": self.contact_report() if hasattr(self, "contact_report") else [],
                "backend": self,
            }
            for hook in self.step_hooks:
                hook(rec)

    def _command(self) -> np.ndarray:
        q = self.q_des + self.q_bias
        q[:5] = np.clip(q[:5], self.kin.lower[:5], self.kin.upper[:5])
        return q

    def _update_bias(self, q_before: np.ndarray, q_cmd: np.ndarray) -> None:
        """At rest the measured joints sit ``sag`` below the command: learn it (EMA)."""
        if not self.use_gravity_bias:
            return
        q = self.q_obs()
        if np.max(np.abs(q[:5] - q_before[:5])) < self.bias_rest_tol_rad:
            sag = q_cmd - q
            sag[5] = 0.0  # the jaw stalls on objects by design; never "compensate" a grasp
            # learn gradually (<= bias_rate_rad per update) and within +-bias_max_rad. A push
            # against an obstacle would also look like sag; the executor restores the bias
            # from before the token whenever a token is aborted for contact or stall.
            step = np.clip(self.bias_alpha * (sag - self.q_bias), -self.bias_rate_rad, self.bias_rate_rad)
            self.q_bias = np.clip(self.q_bias + step, -self.bias_max_rad, self.bias_max_rad)

    def _ctrl_step(self, q_before: np.ndarray) -> None:
        q_cmd = self._command()
        self.ctrl_step_count += 1
        for _ in range(self.ticks_per_ctrl):
            self._tick(q_cmd)
        self._update_bias(q_before, q_cmd)

    def tcp_for_grasp(self, obj_center_task, width_m: float, gap_m: float = 0.002) -> np.ndarray:
        """TCP position that puts an object of ``width_m`` (centred at ``obj_center_task``)
        between the fingers, ``gap_m`` clear of the fixed finger, with the CURRENT orientation.

        The closing axis (gripper +x, from the fixed finger toward the moving jaw) is taken
        from the measured hand; with the yaw held it is horizontal.
        """
        c = self.tcp_rotmat()[:, 0]
        return np.asarray(obj_center_task, dtype=np.float64) - (self.fixed_finger_face_m + gap_m + width_m / 2) * c

    def hand_points(self, q) -> np.ndarray:
        """Gripper + jaw collision-hull vertices in the task frame at joints ``q`` (N, 3)."""
        poses = self.kin.link_poses(q)
        r = self._r_world_base()
        pts = [(r @ (poses[link][:3, :3] @ p.T + poses[link][:3, 3:4])).T for link, p in self._hull.items()]
        return np.vstack(pts)

    def hand_lowest_z(self, q) -> float:
        """Lowest task-frame z of the gripper + jaw collision hulls at joints ``q``."""
        poses = self.kin.link_poses(q)
        r = self._r_world_base()
        low = np.inf
        for link, pts in self._hull.items():
            t = poses[link]
            z = (r @ (t[:3, :3] @ pts.T + t[:3, 3:4]))[2]
            low = min(low, float(z.min()))
        return low

    def in_contact(self) -> bool:
        """Arm held away from its IK target by more than gravity can account for."""
        return bool(np.max(np.abs(self.q_des[:5] - self.q_obs()[:5])) > self.contact_tol_rad)

    def _solve(self, q_seed, target_task, pitch, yaw, roll=None):
        q_seed = np.asarray(q_seed, dtype=np.float64).copy()
        if yaw is None:
            q_seed[4] = self.roll_ref if roll is None else roll  # roll mode: the joint IS the reference
        q, err = self.kin.solve(q_seed, self._task_to_base(np.asarray(target_task)), pitch, yaw)
        return q, err

    def feasible(self, target_task, pitch: Optional[float] = None, yaw: Optional[float] = None,
                 use_yaw_ref: bool = True, roll: Optional[float] = None) -> tuple[bool, str]:
        """Can the hand be put at ``target_task`` with this orientation, exactly?

        Reasons for "no": outside the workspace box, IK residual above tolerance (unreachable
        / orientation cannot be held), or a joint within ``min_joint_margin_rad`` of a limit.
        """
        target_task = np.asarray(target_task, dtype=np.float64)
        pitch = self.pitch_ref if pitch is None else pitch
        if yaw is None and use_yaw_ref:
            yaw = self.yaw_ref
        lo, hi = np.asarray(self.workspace["min"]), np.asarray(self.workspace["max"])
        if np.any(target_task < lo) or np.any(target_task > hi):
            axis = "xyz"[int(np.argmax(np.maximum(lo - target_task, target_task - hi)))]
            return False, f"workspace({axis})"
        q, err = self._solve(self.q_obs(), target_task, pitch, yaw, roll)
        if yaw is None and not (self.kin.lower[4] + self.min_joint_margin_rad
                                <= q[4] <= self.kin.upper[4] - self.min_joint_margin_rad):
            return False, "joint_limit(Wrist_Roll)"
        if np.linalg.norm(err[:3]) > self.tol_pos_m:
            return False, f"ik_pos({np.linalg.norm(err[:3]) * 1000:.1f}mm)"
        if err.size > 3 and np.max(np.abs(err[3:])) > self.tol_ori_rad:
            return False, f"ik_ori({np.degrees(np.max(np.abs(err[3:]))):.1f}deg)"
        if self.hand_floor_task_z is not None and self._hull is not None:
            q_chk = q.copy()
            q_chk[5] = self.q_des[5]  # with the jaw as currently commanded
            low = self.hand_lowest_z(q_chk)
            if low < self.hand_floor_task_z:
                return False, f"hand_floor({(self.hand_floor_task_z - low) * 1000:.1f}mm)"
        margin = self.kin.joint_limit_margin(q)[:5].min()
        if margin < self.min_joint_margin_rad:
            return False, f"joint_limit({margin:.3f}rad)"
        return True, "ok"

    def commanded_tcp(self) -> np.ndarray:
        """TCP of the COMMANDED joints q_des (model, no sag): the setpoint the arm is chasing."""
        return self._base_to_task(self.kin.tcp(self.q_des)[0])

    def apply_delta(self, delta_m, grip_cmd: float, max_cmd_m: float) -> None:
        self.apply_delta_pitch_yaw(delta_m, 0.0, 0.0, grip_cmd, max_cmd_m, 0.0)

    def apply_delta_pitch_yaw(self, delta_m, d_pitch: float, d_yaw: float, grip_cmd: float,
                         max_cmd_m: float, max_cmd_rad: float) -> None:
        """ONE control step: the COMMANDED TCP moves by (capped) ``delta_m``; the pitch / yaw
        references move by (capped) ``d_pitch`` / ``d_yaw``. Orientation is in SO-101 terms
        (in-plane pitch, and the closing-axis yaw OR, in roll mode, the Wrist_Roll angle:
        +d_yaw = counter-clockwise seen from above for a downward hand).

        Everything here is relative to the COMMANDED state (q_des), never the measured one.
        Solving from the measured, sagging joints re-adds whatever sag the bias does not cover
        on every step, and that integrates: measured on the kinematic arm with 0.05 rad of
        unmodelled sag, the hand walked 278 mm off in 90 steps (and the rack-contact runaway
        was the same loop). From q_des, residual sag is a constant offset that the executor's
        bounded integral term closes.
        """
        delta = np.asarray(delta_m, dtype=np.float64)
        n = np.linalg.norm(delta)
        if n > max_cmd_m:
            delta = delta * (max_cmd_m / n)
        if d_pitch:
            self.pitch_ref += float(np.clip(d_pitch, -max_cmd_rad, max_cmd_rad))
        if d_yaw:
            step = float(np.clip(d_yaw, -max_cmd_rad, max_cmd_rad))
            if self.yaw_ref is not None:
                self.yaw_ref += step
            else:
                self.roll_ref += step
        q_before = self.q_obs()
        target_task = self.commanded_tcp() + delta
        self.last_tcp_target_task = target_task
        q_des, _ = self._solve(self.q_des, target_task, self.pitch_ref + self.pitch_trim, self.yaw_ref)
        q_des[5] = self.jaw_open_rad if grip_cmd > 0 else self.jaw_close_rad
        self.q_des = q_des
        self._ctrl_step(q_before)

    # ------------------------------------------------------------ lifecycle helpers
    def _begin_episode(self, settle_ctrl_steps: int = 30) -> None:
        """Adopt the current pose as the command and learn the gravity sag at rest."""
        self.q_des = self.q_obs()
        self.q_des[5] = self.jaw_open_rad
        self.q_bias = np.zeros(6)
        self.pitch_ref = self.kin.tool_pitch(self.q_des)
        self.yaw_ref = None
        self.roll_ref = float(self.q_des[4])  # keep the reset roll until told otherwise
        self.pitch_trim = 0.0
        self.annotation = {"phase": "reset"}
        self.tick_count = 0
        self.ctrl_step_count = 0
        self.decision_id = None
        self.last_tcp_target_task = None
        for _ in range(settle_ctrl_steps):
            self._ctrl_step(self.q_obs())


# ---------------------------------------------------------------------------- token executor
# Show-Harness ``AtomicExec`` for SO-101: tokens are checked before they run, executed at a
# speed the arm can follow, and finished only when the arm is at REST on the target.
#
# Translation (MV_*): one token == ``step_m`` of TCP travel in the task frame, orientation held.
# The END POINT (and midpoint) is checked first with ``backend.feasible``; an infeasible token
# is refused whole (the arm does not move, ``last_blocked`` says why). Executing it partway
# would bend the motion off its axis or drop the orientation: a label that lies.
#
# Rotation (RT_*): the SO-101 can only tilt the hand about the arm-plane normal, and turn it
# about its own tool axis through Wrist_Roll. So:
#   RT_PITCH_FWD / RT_PITCH_BACK  in-plane tool pitch -/+ ``rot_step_rad`` (FWD tips the
#                                 fingertips outward, away from the base). The axis is the
#                                 arm-plane normal: the shared world -Y only while the arm
#                                 points along +X (a 5-DoF hand cannot tilt about a fixed
#                                 world axis in every pose)
#   RT_YAW_CCW / RT_YAW_CW        Wrist_Roll +/- ``rot_step_rad`` in roll mode (the default),
#                                 the closing-axis yaw in yaw mode; sense seen from above. The
#                                 axis is the TOOL axis: the shared world vertical only while
#                                 the tool points straight down. In roll mode, at tool pitch p
#                                 a token is rot_step * sin(p) of yaw and the rest tips the
#                                 closing axis out of horizontal (pitch 47 deg: 7.3 deg of yaw
#                                 and ~7 deg of tilt per token), so every token moves the roll
#                                 away from ROLL_CLOSING_HORIZONTAL
#   RT_ROLL_*                     not realisable -> always refused
# The TCP position is held during a rotation.
#
# Why not Show-Harness's own loop as is (measured on the Workshop sim, 2026-10-03):
#   * it commands up to ``step_m`` (2 cm) per control step: the low-gain SO-101 joints swing
#     past the target, 5.5 mm overshoot on MV_UP/DOWN, and the wrist roll lags the base on
#     MV_LEFT/RIGHT (5 deg transient yaw error);
#   * its ``quiesce`` holds by commanding "zero delta from the MEASURED pose", so the setpoint
#     follows the hand and an overshoot is never pulled back;
#   * a token returns as soon as it is within tolerance, still moving, so the next frame is
#     taken mid-motion.
# Here every motion is one ``_servo`` toward a fixed TARGET with per-step caps
# (``max_cmd_m``, ``rot_rate_rad``), and it only returns once the arm has settled there.

ROT_TOKENS = {
    "RT_PITCH_FWD": (-1.0, 0.0),
    "RT_PITCH_BACK": (1.0, 0.0),
    "RT_YAW_CCW": (0.0, 1.0),
    "RT_YAW_CW": (0.0, -1.0),
}
# The shared vocabulary is unchanged; this embodiment states what it can realise. RT_ROLL_*
# tips the fingertips sideways, OUT of the arm plane. Pitch, Elbow and Wrist_Pitch are
# parallel, so the SO-101 can only tilt its tool axis WITHIN that plane: a roll needs a joint
# it does not have. RT_ROLL_* is therefore refused deterministically, every time, with this
# reason, and never approximated (check() and run() agree; the arm does not move).
UNSUPPORTED = ("RT_ROLL_LEFT", "RT_ROLL_RIGHT")
UNSUPPORTED_REASON = "unsupported_on_5dof"
assert set(ROT_TOKENS) | set(UNSUPPORTED) == set(RT_ATOMS)


def _wrap_pi(a: float) -> float:
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def load_primitives(path: Path | str = DEFAULT_PRIMITIVES) -> dict:
    return yaml.safe_load(Path(path).read_text())


class So101AtomicExec:
    """Duck-type compatible with ``atomic_tokenizer.AtomicExec`` (the generators and writers
    use the same attributes and methods), but self-contained: every motion is a servo to a
    fixed target that only returns once the arm is at rest, which the base class does not do
    (see the comment above for the measured reasons)."""

    def __init__(
        self,
        backend,
        primitives: Optional[dict] = None,
        step_m: Optional[float] = None,
        rot_step_rad: Optional[float] = None,
        tol_m: float = 0.001,
        rot_tol_rad: float = float(np.radians(0.2)),
        gripper_steps: int = 10,
        lattice_origin: Optional[np.ndarray] = None,
        max_cmd_m: float = 0.005,             # 5 mm per control step (30 Hz) = 15 cm/s
        rot_rate_rad: float = float(np.radians(0.5)),  # 0.5 deg per control step (1 deg: 3.04 mm TCP drift)
        settle_tol_m: float = 0.0003,         # "at rest": TCP moved < 0.3 mm in one control step
        settle_tol_rad: float = float(np.radians(0.05)),
        servo_max_ctrl_steps: int = 90,       # 3 s: give up and report
        i_max_m: float = 0.004,               # integral trim limits (position / pitch)
        i_max_rad: float = float(np.radians(1.0)),
        stall_steps_max: int = 20,            # at rest short of the target this long = stall
    ) -> None:
        prim = load_primitives() if primitives is None else primitives
        self.backend = backend
        self.move_dirs = {k: np.asarray(prim["atomic_primitives"][k], dtype=np.float64) for k in MOVE_ATOMS}
        rot = prim.get("rotation_units", {})
        if sorted(rot.get("supported", [])) != sorted(ROT_TOKENS) or sorted(rot.get("unsupported", [])) != sorted(UNSUPPORTED):
            raise ValueError("primitives rotation_units disagree with So101AtomicExec's ROT_TOKENS / UNSUPPORTED")
        self.step_m = float(prim["step_m"] if step_m is None else step_m)
        self.rot_step_rad = float(np.radians(prim["rot_step_deg"]) if rot_step_rad is None else rot_step_rad)
        self.tol_m = float(tol_m)
        self.rot_tol_rad = float(rot_tol_rad)
        self.max_cmd_m = float(max_cmd_m)
        self.gripper_steps = int(gripper_steps)
        # Same meaning as AtomicExec.lattice_origin: aim every token at a lattice node.
        self.lattice_origin = None if lattice_origin is None else np.asarray(lattice_origin, dtype=np.float64)
        self.grip_cmd = 1.0  # +1 open / -1 close, persists across tokens
        self.rot_rate_rad = float(rot_rate_rad)
        self.settle_tol_m = float(settle_tol_m)
        self.settle_tol_rad = float(settle_tol_rad)
        self.servo_max_ctrl_steps = int(servo_max_ctrl_steps)
        self.i_max_m = float(i_max_m)
        self.i_max_rad = float(i_max_rad)
        self.stall_steps_max = int(stall_steps_max)
        self.last_blocked: Optional[str] = None
        self.last_converged: bool = True
        self.hold_target: Optional[np.ndarray] = None  # TCP position held between tokens
        self.last_contact: bool = False
        self.last_stall: bool = False
        self.last_abort: Optional[dict] = None

    @property
    def gripper_closed(self) -> bool:
        return self.grip_cmd < 0

    def lattice_node(self, pos: np.ndarray) -> np.ndarray:
        """Nearest lattice node to ``pos`` (``pos`` itself without a lattice, or when the hand
        is more than a quarter step off it: something pushed it). As in AtomicExec."""
        pos = np.asarray(pos, dtype=np.float64)
        if self.lattice_origin is None:
            return pos
        node = self.lattice_origin + self.step_m * np.round((pos - self.lattice_origin) / self.step_m)
        return node if np.max(np.abs(node - pos)) <= 0.25 * self.step_m else pos

    def _label(self, token: str, **extra) -> None:
        self.backend.annotation = {"token": token, **extra}

    def _refuse(self, token: str, why: str) -> float:
        self.last_blocked = why
        self._label(token, blocked=why)
        return 0.0

    # -------------------------------------------------------------- the one primitive
    def _servo(self, target, pitch_t: Optional[float] = None, yaw_t: Optional[float] = None,
               budget: Optional[int] = None, roll_t: Optional[float] = None) -> bool:
        """Drive the TCP to ``target`` (and the references to pitch_t / yaw_t or roll_t),
        capped per control step, and return once the arm is AT REST there. Returns
        convergence. ``yaw_t`` switches the backend to yaw mode; ``roll_t`` (roll mode) moves
        the Wrist_Roll reference."""
        be = self.backend
        target = np.asarray(target, dtype=np.float64)
        pitch_t = be.pitch_ref if pitch_t is None else pitch_t
        if yaw_t is not None and be.yaw_ref is None:
            be.yaw_ref = be.kin.tool_yaw(be.q_obs())[0]
        self.hold_target = target
        prev_p, prev_o = be.tcp_pos(), self._orient()
        contact_steps = stall_steps = 0
        # Integral action for the last millimetre: gravity sag that the joint bias does not
        # remove leaves the arm AT REST 1.5 mm / 0.2 deg short (measured in free space). While
        # at rest and not at the target, aim past it by the accumulated residual.
        i_pos = np.zeros(3)  # per target; the pitch trim persists like the gravity bias
        bias_before, trim_before = be.q_bias.copy(), be.pitch_trim
        for _ in range(budget or self.servo_max_ctrl_steps):
            d_pitch = pitch_t - be.pitch_ref
            if yaw_t is not None:
                d_yaw = _wrap_pi(yaw_t - be.yaw_ref)
            elif roll_t is not None and be.yaw_ref is None:
                d_yaw = roll_t - be.roll_ref
            else:
                d_yaw = 0.0
            # The COMMANDED TCP walks to (target + integral trim) at <= 5 mm per control step,
            # decelerating over the last centimetre (a constant 5 mm step swung the low-gain
            # joints 2.3 mm past). The measured hand follows; it only feeds the integral.
            err = (target + i_pos) - be.commanded_tcp()
            cap = min(self.max_cmd_m, max(0.001, 0.5 * float(np.linalg.norm(err))))
            be.apply_delta_pitch_yaw(err, d_pitch, d_yaw, self.grip_cmd, cap, self.rot_rate_rad)
            p, o = be.tcp_pos(), self._orient()
            ep, ey = be.orientation_error()
            at_target = (np.linalg.norm(target - p) < self.tol_m
                         and abs(pitch_t - be.pitch_ref) < 1e-12
                         and (yaw_t is None or abs(_wrap_pi(yaw_t - be.yaw_ref)) < 1e-12)
                         and (roll_t is None or be.yaw_ref is not None or abs(roll_t - be.roll_ref) < 1e-12)
                         and abs(ep) < self.rot_tol_rad and abs(ey) < self.rot_tol_rad)
            at_rest = (np.linalg.norm(p - prev_p) < self.settle_tol_m
                       and abs(o[0] - prev_o[0]) < self.settle_tol_rad
                       and abs(_wrap_pi(o[1] - prev_o[1])) < self.settle_tol_rad)
            # Two signatures of "something is in the way": a joint held far from its IK target
            # while the hand makes no progress (hard push), or the hand at rest short of the
            # target (soft stop; the joint gap of a 1 cm deficit is only ~0.04 rad).
            # The gap alone is NOT contact: at full reach gravity sag + motion lag exceed any
            # fixed threshold while the hand moves exactly right (measured: a correct 20.2 mm
            # MV_FWD at x = 0.26 m was flagged, and so was the vial approach).
            progress = float(np.linalg.norm(target - prev_p) - np.linalg.norm(target - p))
            # measured contact (sim sensor) is trusted as is; the kinematic fallback (no sensor,
            # e.g. the real arm) additionally needs "no progress" to avoid full-reach false fires
            pushing = be.in_contact() and (be.contact_is_measured
                                           or (progress < 0.0005 and not at_target))
            contact_steps = contact_steps + 1 if pushing else 0
            short = at_rest and not at_target
            if short:  # integrate the residual (bounded: a few mm / a degree, never a reference)
                i_pos = np.clip(i_pos + 0.5 * (target - p), -self.i_max_m, self.i_max_m)
                be.pitch_trim = float(np.clip(be.pitch_trim + 0.5 * ep, -self.i_max_rad, self.i_max_rad))
            # a stall is "at rest and short of the target for stall_steps_max control steps"
            # (the integral had that long to close the gap) - not "short for a moment"
            stall_steps = stall_steps + 1 if short else 0
            prev_p, prev_o = p, o
            if contact_steps >= 3 or stall_steps >= self.stall_steps_max:
                # Something is holding the arm off its target. Stop pushing: adopt where the
                # hand IS as the target, and report. Never keep driving into it (the gravity
                # bias used to learn the push and the arm ran away).
                q = be.q_obs()
                self.last_abort = {
                    "kind": "contact" if contact_steps >= 3 else "stall",
                    "pos_err_mm": round(float(np.linalg.norm(target - p) * 1000), 2),
                    "pitch_err_deg": round(float(np.degrees(ep)), 3),
                    "second_err_deg": round(float(np.degrees(ey)), 3),
                    "joint_gap_rad": round(float(np.max(np.abs(be.q_des[:5] - q[:5]))), 4),
                    "bias_rad": [round(float(b), 4) for b in be.q_bias[:5]],
                    "contacts": be.contact_report() if hasattr(be, "contact_report") else [],
                }
                if be.hand_floor_task_z is not None and be._hull is not None:
                    self.last_abort["hand_low_above_floor_mm"] = round(
                        (be.hand_lowest_z(q) - be.hand_floor_task_z) * 1000, 2)
                self.hold_target = be.tcp_pos()
                be.pitch_ref, yaw_now = self._orient()
                if be.yaw_ref is not None:
                    be.yaw_ref = yaw_now
                else:
                    be.roll_ref = float(be.q_obs()[4])
                # undo what the push may have taught the bias / trim, then command = where the
                # arm IS (minus the sag): nothing left to push with
                be.q_bias, be.pitch_trim = bias_before, trim_before
                be.q_des[:5] = q[:5] - be.q_bias[:5]
                be.apply_delta_pitch_yaw(np.zeros(3), 0.0, 0.0, self.grip_cmd, self.max_cmd_m, 0.0)
                self.last_contact = self.last_abort["kind"] == "contact"
                self.last_stall = self.last_abort["kind"] == "stall"
                self.last_converged = False
                return False
            if at_target and at_rest:
                self.last_converged = True
                return True
        self.last_converged = False
        self.last_contact = False
        return False

    def _orient(self) -> tuple[float, float]:
        q = self.backend.q_obs()
        return self.backend.kin.tool_pitch(q), self.backend.kin.tool_yaw(q)[0]

    # -------------------------------------------------------------- dry check
    def check(self, token: str) -> tuple[bool, str]:
        """Would ``token`` be executed (not refused)? Nothing moves.

        Generators call this BEFORE recording a frame, so a refused token never becomes a
        training label. Gripper tokens are always executable.
        """
        be = self.backend
        if token in self.move_dirs:
            node = self.lattice_node(be.tcp_pos())
            target = node + self.move_dirs[token] * self.step_m
            for probe in (target, (node + target) / 2):
                ok, why = be.feasible(probe)
                if not ok:
                    return False, why
            return True, "ok"
        if token.startswith("RT_"):
            if token in UNSUPPORTED or token not in ROT_TOKENS:
                return False, UNSUPPORTED_REASON
            sp, sy = ROT_TOKENS[token]
            pitch_t = be.pitch_ref + sp * self.rot_step_rad
            yaw_t = None if be.yaw_ref is None else be.yaw_ref + sy * self.rot_step_rad
            roll_t = be.roll_ref + sy * self.rot_step_rad if be.yaw_ref is None else None
            return be.feasible(self.lattice_node(be.tcp_pos()), pitch_t, yaw_t,
                               use_yaw_ref=False, roll=roll_t)
        if token in ("GRASP", "RELEASE"):
            return True, "ok"
        return False, f"unknown_token({token})"

    # -------------------------------------------------------------- translation
    def move(self, token: str) -> float:
        be = self.backend
        start = be.tcp_pos()
        node = self.lattice_node(start)
        target = node + self.move_dirs[token] * self.step_m
        for probe in (target, (node + target) / 2):
            ok, why = be.feasible(probe)
            if not ok:
                return self._refuse(token, why)
        self.last_blocked = None
        self.last_contact = self.last_stall = False
        self.last_abort = None
        self._label(token)
        if not self._servo(target):
            # moved PART of the way: recording it as this token would be a lie
            self.last_blocked = ("contact" if self.last_contact else
                                 "stall" if self.last_stall else "not_converged")
            self._label(token, blocked=self.last_blocked)
        return float(np.linalg.norm(be.tcp_pos() - start))

    # -------------------------------------------------------------- rotation
    def rotate(self, token: str) -> float:
        be = self.backend
        if token in UNSUPPORTED or token not in ROT_TOKENS:
            return self._refuse(token, UNSUPPORTED_REASON)
        sp, sy = ROT_TOKENS[token]
        pitch_t = be.pitch_ref + sp * self.rot_step_rad
        # yaw mode: turn the closing-axis heading; roll mode: turn the hand about its own axis
        # (Wrist_Roll; +roll is counter-clockwise from above, measured)
        yaw_t = None if be.yaw_ref is None else be.yaw_ref + sy * self.rot_step_rad
        roll_t = be.roll_ref + sy * self.rot_step_rad if be.yaw_ref is None else None
        pos = self.lattice_node(be.tcp_pos())
        ok, why = be.feasible(pos, pitch_t, yaw_t, use_yaw_ref=False, roll=roll_t)
        if not ok:
            return self._refuse(token, why)
        self.last_blocked = None
        self.last_contact = False
        self._label(token)
        o0, r0 = self._orient(), be.q_obs()[4]
        if not self._servo(pos, pitch_t, yaw_t, roll_t=roll_t):
            self.last_blocked = ("contact" if self.last_contact else
                                 "stall" if self.last_stall else "not_converged")
            self._label(token, blocked=self.last_blocked)
        o1, r1 = self._orient(), be.q_obs()[4]
        turn = _wrap_pi(o1[1] - o0[1]) if be.yaw_ref is not None else r1 - r0
        return float(np.hypot(o1[0] - o0[0], turn))

    # -------------------------------------------------------------- holding / gripper
    def _hold_point(self) -> np.ndarray:
        return self.backend.tcp_pos() if self.hold_target is None else self.hold_target

    def quiesce(self, tol_m=None, max_steps=None) -> int:
        tok = self.backend.annotation.get("token", "-")
        self._label(tok, phase="settle")
        c0 = self.backend.ctrl_step_count
        self._servo(self._hold_point(), budget=max_steps)
        return self.backend.ctrl_step_count - c0

    def hold(self, n: int = 1) -> None:
        target = self._hold_point()
        for _ in range(n):
            self.backend.apply_delta_pitch_yaw(target - self.backend.commanded_tcp(), 0.0, 0.0,
                                          self.grip_cmd, self.max_cmd_m, 0.0)

    def _gripper(self, token: str, cmd: float) -> float:
        # At rest first: closing while moving drags the object, opening while moving throws
        # it (Show-Harness's own measured failure modes). Then hold the TCP still while the
        # jaw moves, and let it settle again.
        self._label(token, phase="settle")
        self._servo(self._hold_point())
        self._label(token)
        self.grip_cmd = cmd
        self.hold(self.gripper_steps)
        self._servo(self._hold_point())
        return self.backend.gripper_width()

    def grasp(self) -> float:
        return self._gripper("GRASP", -1.0)

    def release(self) -> float:
        return self._gripper("RELEASE", 1.0)

    def run(self, token: str) -> float:
        """Execute any vocabulary token; returns metres (MV), radians (RT) or width (gripper)."""
        if token in self.move_dirs:
            return self.move(token)
        if token.startswith("RT_"):
            return self.rotate(token)
        if token == "GRASP":
            return self.grasp()
        if token == "RELEASE":
            return self.release()
        raise ValueError(f"unknown token {token!r}")

    def goto_pose(self, pos_task, pitch: float, yaw: Optional[float] = None,
                  roll: Optional[float] = None, tol_m: float = 1e-3, budget: int = 400) -> bool:
        """Servo (NOT a token, never recorded as one) to a start pose with an orientation:
        pitch + yaw (yaw mode) or pitch + Wrist_Roll ``roll`` (roll mode, the default)."""
        self._label("-", phase="goto_pose")
        return self._servo(pos_task, pitch, yaw, budget=budget, roll_t=roll)
