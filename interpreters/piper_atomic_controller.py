"""AgileX/Songling Piper real-robot atomic-action controller.

The arm-agnostic token -> Cartesian-setpoint machinery lives in
:class:`interpreters.real_atomic_controller.RealAtomicController`; this subclass adds
only what is specific to the 6-DoF Piper driven over ROS topics
(:class:`core.piper.piper_interface.PiperInterface`):

* **Cartesian motion** -- MV_* tokens are straight base-frame translations (the
  primitives unit vectors) and ROTATE_* a base-Z gripper yaw; the Piper SDK's MOVE P
  does the IK. This matches the image-relative controller prompt exactly
  (MV_LEFT = move image-left), so it is the motion model used for autonomous rollouts.
  All the token->setpoint work is inherited unchanged from the base -- the subclass
  overrides no motion logic, only the two safety wrappers below.

* **Divergence guard** -- the Piper node silently DROPS commands when it is not
  enabled / not in mode 1, and EndPoseCtrl gives no feedback on unreachable (IK)
  targets. Pure open-loop setpoint integration would then accumulate phantom distance
  and, once commands land again, lunge the arm in one large stiff MOVE P. After every
  motion the measured pose is compared against the setpoint; beyond
  ``divergence_resync_m`` (position) or ``divergence_resync_rad`` (orientation) the
  setpoint is re-synced to the measured pose (loudly).

* **Dropped-gripper detection** -- a GRASP/RELEASE that produced no width change was
  likely dropped (the same not-enabled / wrong-mode failure); warn so a phantom grasp
  is never silently recorded as a successful one.

Gripper defaults are sized for the Piper gripper (~0.07 m stroke) but are
PLACEHOLDERS until measured on the real unit -- see configs/robot_piper.yaml.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from interpreters.real_atomic_controller import (
    AtomicStepResult,
    RealAtomicController,
    euler_to_quat,
    quat_to_euler,
)
from core.piper.kinematics import select_dh_variant

# --- Piper gripper placeholders (CALIBRATE on the real unit) -----------------
# Rated stroke ~0.07 m. Measure with the teleop script: fully-open width, the
# empty-closed width, and a typical held-object width, then set robot_piper.yaml.
PIPER_GRIPPER_CLOSE_THRESHOLD_M = 0.06  # width below this counts as "closed"
PIPER_GRASP_OPEN_WIDTH_M = 0.055        # width at/above this is still-open band
PIPER_EMPTY_GRASP_WIDTH_M = 0.005       # a close settling at/below this caught nothing

# Measured-vs-setpoint divergence (m) beyond which the setpoint is re-synced to the
# measured pose. Sized to ~3 silently-dropped fine steps: large enough that a slow
# MOVE P still converging is never mistaken for divergence.
DEFAULT_DIVERGENCE_RESYNC_M = 0.06


class PiperAtomicController(RealAtomicController):
    """Maps atomic VLM tokens to motion on the Piper. Cartesian semantics: MV_* are
    base-frame translations and ROTATE_* a base-Z yaw, inherited from
    :class:`RealAtomicController`.

    **Motion backend** (``motion_backend``): how a Cartesian target is physically
    reached.

    * ``"joint_stream"`` (DEFAULT -- the AgileX-specific smooth path): each move's
      Cartesian ramp is converted to joint-space waypoints (on-board FK/IK, vendored
      from piper_sdk; bounded-orientation solve, see ``ori_flex_rad``) and streamed
      over ``/master/joint_<arm>`` -> firmware **MOVE J** -- the same mechanism the
      vendor's own master-slave teleop and trajectory replay use, which tracks a dense
      stream continuously. This avoids the stutter inherent to the Cartesian path,
      where every command is an independently PLANNED point-to-point move. Selected
      only after the vendored FK is validated against the arm's own pose feedback at
      sync. A move that reaches the budgeted-workspace boundary is CLAMPED there
      (smooth stop + truthful setpoint + a step note); ``endpose`` remains only as
      insurance for a move whose first waypoint already fails (broken IK premise).
    * ``"endpose"``: the plain Cartesian path (``/puppet/pos_cmd`` -> EndPoseCtrl,
      firmware MOVE P) -- one planned move per command.

    Plus the Piper-node safety net (setpoint divergence re-sync + dropped-gripper
    detection) and the Piper gripper-size defaults.
    """

    LOG_TAG = "piper-atomic"

    def __init__(
        self,
        robot: Any,
        atomic_primitives: dict[str, Any],
        step_m: float,
        yaw_step_rad: float,
        motion_backend: str = "joint_stream",
        joint_stream_hz: float = 50.0,
        ori_flex_rad: float = np.radians(15.0),
        divergence_resync_m: Optional[float] = DEFAULT_DIVERGENCE_RESYNC_M,
        divergence_resync_rad: Optional[float] = None,
        gripper_close_threshold_m: float = PIPER_GRIPPER_CLOSE_THRESHOLD_M,
        grasp_open_width_m: float = PIPER_GRASP_OPEN_WIDTH_M,
        gripper_settle_s: float = 1.5,
        **kwargs: Any,
    ) -> None:
        """Args (Piper-specific; the rest match RealAtomicController):
            motion_backend: "joint_stream" (smooth MOVE J streaming; default) or
                "endpose" (plain Cartesian MOVE P). See the class docstring.
            joint_stream_hz: Waypoint rate of the joint-stream backend. 50 Hz matches
                the vendor teleop/replay and our homing moves.
            ori_flex_rad: Orientation-bend budget for the joint-stream IK
                (kinematics.ik_bounded). Holding the captured orientation EXACTLY is
                what bounds the usable workspace (the +/-70 deg wrist pitch saturates
                centimetres from the start pose); within this budget the gripper may
                tilt toward the orientation the mechanism naturally adopts, which
                multiplies the reachable range of MV_UP/MV_BACK/lateral moves. The
                orientation SETPOINT stays the captured reference, so the bend never
                accumulates -- the gripper re-levels wherever the workspace allows.
                0 disables (strict full-pose IK only).
            divergence_resync_m: Re-sync the setpoint from the measured pose when the
                POSITION diverges beyond this after a motion (None disables the position
                guard -- NOT recommended: the Piper node drops commands silently).
            divergence_resync_rad: Re-sync when the commanded orientation diverges from
                the measured orientation beyond this (radians). None -> 1.5x the yaw
                step (tolerates one step of normal MOVE-P lag); 0/negative disables. In
                cartesian operation the gripper orientation is held constant, so this
                only trips on a runaway from dropped ROTATE_* commands.
            gripper_close_threshold_m / grasp_open_width_m / gripper_settle_s: Piper
                gripper sizing (stroke ~0.07 m), forwarded to the base.
        """
        super().__init__(
            robot,
            atomic_primitives,
            step_m,
            yaw_step_rad,
            gripper_close_threshold_m=gripper_close_threshold_m,
            grasp_open_width_m=grasp_open_width_m,
            gripper_settle_s=gripper_settle_s,
            **kwargs,
        )
        motion_backend = str(motion_backend).strip().lower()
        if motion_backend not in ("joint_stream", "endpose"):
            raise ValueError(
                f"motion_backend must be 'joint_stream' or 'endpose', got {motion_backend!r}"
            )
        self.motion_backend = motion_backend
        self.joint_stream_hz = max(1.0, float(joint_stream_hz))
        self.ori_flex_rad = max(0.0, float(ori_flex_rad))
        # Joint-stream state, initialized on sync: the validated kinematics model, the
        # last streamed joint solution (IK seed), and whether the backend is usable
        # (FK validated against the arm's own feedback + robot supports streaming).
        self._kin = None
        self._q_cmd: Optional[np.ndarray] = None
        self._joint_stream_ok = False
        # Note produced by the last _drive_to_target (e.g. a boundary clamp), appended
        # to the step's AtomicStepResult by _apply_motion so the runner/logs see it.
        self._last_drive_note = ""
        # One-shot latch for the "start pose near a joint limit" hint (a property of the
        # begin pose; re-printing it on every re-sync is just noise).
        self._warned_tight_start = False
        self.divergence_resync_m: Optional[float] = (
            None if divergence_resync_m is None else float(divergence_resync_m)
        )
        # Orientation divergence threshold. Default 1.5x the yaw step so a single step of
        # normal MOVE-P lag never false-triggers, but a runaway (dropped ROTATE_*) is
        # caught. None -> derive; <=0 -> disabled.
        self.divergence_resync_rad: Optional[float] = (
            1.5 * float(yaw_step_rad)
            if divergence_resync_rad is None
            else (float(divergence_resync_rad) if float(divergence_resync_rad) > 0 else None)
        )

    # -- joint-stream backend --------------------------------------------------
    def sync_from_robot(self) -> None:
        """Sync the Cartesian setpoint (base), then (re)validate the joint backend.

        The vendored FK is trusted only if it reproduces the arm's OWN pose feedback at
        the current joints (the firmware computes that feedback with the same DH); a
        mismatch means the wrong model or an uncalibrated arm, and we refuse to stream
        IK output computed from a bad FK -- the endpose (MOVE P) path takes over.
        """
        super().sync_from_robot()
        self._joint_stream_ok = False
        self._kin = None
        self._q_cmd = None
        if self.motion_backend != "joint_stream":
            return
        get_q = getattr(self.robot, "get_joint_positions", None)
        stream = getattr(self.robot, "stream_joints", None)
        if not (callable(get_q) and callable(stream)):
            print(f"[{self.LOG_TAG}] joint_stream: robot lacks joint access -> endpose fallback")
            return
        try:
            q = np.asarray(get_q(), dtype=float).reshape(-1)[:6]
            pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
            kin, err_m = select_dh_variant(q, pose)
        except Exception as exc:  # noqa: BLE001 - never let validation kill a sync
            print(f"[{self.LOG_TAG}] joint_stream: FK validation failed ({exc}) -> endpose fallback")
            return
        if err_m > 0.01:
            print(
                f"[{self.LOG_TAG}] WARNING: joint_stream disabled -- FK(measured joints) is "
                f"{err_m * 1000:.1f} mm from the arm's own pose feedback (model mismatch). "
                "Falling back to the endpose (MOVE P) backend."
            )
            return
        self._kin = kin
        self._q_cmd = q
        self._joint_stream_ok = True
        if self.verbose:
            print(
                f"[{self.LOG_TAG}] joint_stream backend ON (DH variant "
                f"{kin.dh_variant:#04x}, FK-vs-feedback err {err_m * 1000:.1f} mm, "
                f"{self.joint_stream_hz:.0f} Hz, ori-flex "
                f"{np.degrees(self.ori_flex_rad):.0f} deg)"
            )
        # Start-pose quality hint: a joint that BEGINS near its limit (the wrist pitch
        # j5 is the usual culprit, +/-70 deg) is what shrinks the strict-orientation
        # workspace to centimetres. Purely informational -- the bounded IK works around
        # it -- but re-capturing a begin pose with more headroom expands the strict range.
        from core.piper.kinematics import JOINT_LIMITS_RAD

        margins = np.degrees(
            np.minimum(q - JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1] - q)
        )
        tight = [f"j{j + 1} {margins[j]:.1f} deg" for j in range(6) if margins[j] < 8.0]
        # Once per controller: every re-sync (homing, divergence) would otherwise repeat
        # this paragraph, and it is a property of the START POSE, not of the sync.
        if tight and not self._warned_tight_start:
            self._warned_tight_start = True
            print(
                f"[{self.LOG_TAG}] start pose is near a joint limit ({', '.join(tight)}); "
                f"the {np.degrees(self.ori_flex_rad):.0f} deg orientation-bend budget "
                "compensates, but a roomier begin pose would widen the reach."
            )

    def _joint_settle(self) -> None:
        """Settle for the joint backend: re-stream the last joint solution (MOVE J)
        instead of re-commanding the Cartesian pose (which would be a MOVE P plan)."""
        for _ in range(self.settle_steps):
            self.robot.stream_joints(self._q_cmd)
            if self.settle_dt_s > 0.0:
                time.sleep(self.settle_dt_s)

    def end_stream(self) -> None:
        if not self._stream_hot:
            return
        if self._joint_stream_ok and self._q_cmd is not None:
            self._stream_hot = False
            self._stream_dir = None
            self._joint_settle()
            return
        super().end_stream()

    def _drive_to_target(
        self,
        start_pos: np.ndarray,
        start_euler: np.ndarray,
        continuous: bool = False,
    ) -> None:
        """Joint-stream realization of one Cartesian move (see class docstring).

        The move's Cartesian ramp -- the smooth plugin's eased/chained profile when
        enabled, else a constant-rate line -- is solved to joint space waypoint by
        waypoint (bounded-orientation IK, seeded with the previous solution) and
        streamed as MOVE J targets at ``joint_stream_hz``. Chaining semantics are
        IDENTICAL to the base: a ``continuous`` move ends hot (no settle) and an
        aligned successor starts at cruise, so held keys / action chunks flow.

        Waypoints solve with :meth:`~core.piper.kinematics.PiperKinematics.ik_bounded`:
        exact position, orientation allowed to bend up to ``ori_flex_rad`` toward what
        the mechanism can do -- this is what keeps MV_UP/MV_BACK/lateral moves alive
        near the wrist-pitch limit instead of dying centimetres from the start pose.
        The first unreachable waypoint is the TRUE boundary of that budgeted workspace:
        the stream stops there smoothly and the position setpoint is re-synced to the
        achieved pose (a "clamp", noted on the step result), so the runner and the VLM
        see the truth instead of a phantom setpoint. No stiff MOVE P fallback
        mid-motion; the EndPoseCtrl path remains only as insurance when not even the
        FIRST waypoint solves (a broken IK premise, e.g. a bad seed after an external
        move -- the firmware planner then gets one chance at the whole move).
        """
        if not self._joint_stream_ok or self._q_cmd is None:
            super()._drive_to_target(start_pos, start_euler, continuous=continuous)
            return

        dpos = self._target_pos - start_pos
        deuler = self._target_euler - start_euler
        dist = float(np.linalg.norm(dpos))
        moved = dist > 1e-5 or float(np.linalg.norm(deuler)) > 1e-5
        if not moved:
            self.end_stream()
            self._joint_settle()
            return

        smooth = self.smooth_plugin is not None and getattr(self.smooth_plugin, "enabled", False)
        pure_translation = dist > 1e-9 and float(np.linalg.norm(deuler)) <= 1e-9
        unit = (dpos / dist) if dist > 1e-9 else None

        if smooth:
            blend = bool(getattr(self.smooth_plugin, "blend", False))
            cruise = float(getattr(self.smooth_plugin, "cruise", 1.0))
            v0 = 0.0
            if blend and self._stream_hot and pure_translation and self._stream_dir is not None:
                if float(np.dot(unit, self._stream_dir)) >= 0.9:
                    v0 = cruise
            if v0 == 0.0:
                self.end_stream()
            v1 = cruise if (blend and continuous and pure_translation) else 0.0
            fractions, tool_dt = self.smooth_plugin.plan(dist, v0, v1)
            # The joint stream has no command-resolution quantum (MOVE J tracks dense
            # targets), so re-densify to the stream rate: waypoints every 1/hz over the
            # SAME total duration the plugin planned.
            duration = len(fractions) * tool_dt
        else:
            v1 = 0.0
            fractions = None
            duration = max(dist / 0.05, 0.15)  # constant-rate fallback: ~5 cm/s
        n = max(2, int(round(self.joint_stream_hz * duration)))
        dt = duration / n

        def frac_at(i: int) -> float:
            t = i / n
            if fractions is None:
                return t
            # Resample the plugin's eased profile at the stream rate (linear in between).
            x = t * len(fractions)
            lo = int(x)
            if lo >= len(fractions):
                return fractions[-1]
            prev = fractions[lo - 1] if lo > 0 else 0.0
            return prev + (fractions[lo] - prev) * (x - lo)

        q = self._q_cmd
        streamed = 0
        max_dev = 0.0
        for i in range(1, n + 1):
            frac = frac_at(i)
            pos = start_pos + dpos * frac
            euler = start_euler + deuler * frac
            rot = R.from_euler("xyz", euler).as_matrix()
            sol, dev = self._kin.ik_bounded(pos, rot, q_seed=q, max_ori_dev_rad=self.ori_flex_rad)
            if sol is None:
                # The budgeted workspace ends here. A move that never even started is a
                # broken IK premise -> give the firmware planner one chance (MOVE P);
                # a move that DID progress is clamped at the boundary, smoothly.
                if streamed == 0:
                    self._endpose_fallback(n)
                else:
                    self._clamp_at_boundary(q, frac_at(i - 1) if i > 1 else 0.0, max_dev)
                return
            q = sol
            max_dev = max(max_dev, float(dev))
            streamed += 1
            self.robot.stream_joints(q)
            if dt > 0.0:
                time.sleep(dt)
        self._q_cmd = q
        if max_dev > np.radians(1.0) and self.verbose:
            print(
                f"[{self.LOG_TAG}] joint_stream: orientation bent up to "
                f"{np.degrees(max_dev):.1f} deg (budget {np.degrees(self.ori_flex_rad):.0f} deg) "
                "to keep the target reachable"
            )
        # Keep gripper-only re-commands consistent with where we actually drove the arm.
        note = getattr(self.robot, "note_commanded_pose", None)
        if callable(note):
            note(np.concatenate([self._target_pos, euler_to_quat(self._target_euler)]))

        if smooth and v1 > 0.0:
            self._stream_hot = True
            self._stream_dir = unit
        else:
            self._stream_hot = False
            self._stream_dir = None
            self._joint_settle()

    def _clamp_at_boundary(self, q_last: np.ndarray, frac: float, max_dev: float) -> None:
        """End a move at the last reachable waypoint (the budgeted-workspace boundary).

        The stream simply stops where IK ran out -- no MOVE P stutter, no phantom
        setpoint: the POSITION setpoint is re-synced to the achieved pose so the next
        command starts from the truth, while the ORIENTATION setpoint keeps the pristine
        captured reference (a bent orientation must re-level when the arm moves back
        into the healthy workspace, never ratchet). The step result carries a note so
        the runner logs why the move fell short; the VLM re-decides from the next
        (truthful) observation.
        """
        self._q_cmd = q_last
        pos, _ = self._kin.fk(q_last)
        self._target_pos = np.asarray(pos, dtype=float).copy()
        note = (
            f"reach clamp: move stopped at {frac * 100:.0f}% -- workspace boundary "
            f"(joint limits) at the orientation-bend budget "
            f"{np.degrees(self.ori_flex_rad):.0f} deg"
        )
        self._last_drive_note = note
        print(f"[{self.LOG_TAG}] {note}")
        # The interface's cached command pose must match where the arm actually is.
        note_pose = getattr(self.robot, "note_commanded_pose", None)
        if callable(note_pose):
            note_pose(self._kin.fk_pose7(q_last))
        self._stream_hot = False
        self._stream_dir = None
        self._joint_settle()

    def _endpose_fallback(self, n: int) -> None:
        """Last-resort MOVE P for a move whose FIRST waypoint already failed IK --
        either a broken premise (bad seed) or the arm ALREADY PINNED at the workspace
        boundary (the previous move ended in a reach clamp and the caller pushed the
        same direction again). The firmware planner gets the whole move; the seed
        re-syncs after. Noted on the step result: an unreachable target makes MOVE P
        move nothing, and the runner must be able to tell the VLM its command did
        not act (observed on hardware: 3 pinned MV_FWDs with zero feedback)."""
        self._last_drive_note = (
            "reach fallback: IK failed at the first waypoint (arm at its reach "
            "limit or bad seed) -> one EndPoseCtrl attempt"
        )
        print(
            f"[{self.LOG_TAG}] WARNING: IK failed at the move's first waypoint (0/{n} "
            "streamed); falling back to EndPoseCtrl for this move."
        )
        self._stream_hot = False
        self._stream_dir = None
        self._command_setpoint()
        self._settle(re_command=True)
        # The arm was just moved OUTSIDE the joint stream: re-seed the IK from the
        # measured joints, or a later _joint_settle would re-stream the stale
        # pre-fallback configuration and silently drive the arm back.
        try:
            self._q_cmd = np.asarray(
                self.robot.get_joint_positions(), dtype=float
            ).reshape(-1)[:6]
        except Exception:  # noqa: BLE001 - keep the old seed as a last resort
            pass

    # -- execution -----------------------------------------------------------
    def _on_gripper_settled(
        self, close: bool, pre_width: float, settled_width: float
    ) -> None:
        """Flag a gripper command that never moved the fingers.

        The Piper node silently DROPS commands when it is not enabled / not in mode 1
        (same failure the pose divergence guard covers). A dropped GRASP would leave
        the width at its open value, so the empty-grasp check would pass and the runner
        would record a phantom successful grasp. A genuine GRASP/RELEASE always moves
        the width well beyond the motion epsilon (open<->closed), so zero motion after
        a state-change command means the command did not take -- warn loudly."""
        if settled_width is None:
            return
        if abs(settled_width - pre_width) <= self.GRIPPER_MOTION_EPS_M:
            print(
                f"[{self.LOG_TAG}] WARNING: gripper {'CLOSE' if close else 'OPEN'} "
                f"produced no width change ({pre_width:.4f}->{settled_width:.4f} m). The "
                "command was likely DROPPED (arm not enabled / node not in mode 1); a "
                "recorded grasp may be phantom. Check the arm is enabled and in mode 1."
            )

    def _apply_motion(
        self,
        delta_pos: np.ndarray,
        yaw: float,
        kind: str,
        pre_pose: np.ndarray,
        token: str,
        continuous: bool = False,
    ) -> AtomicStepResult:
        self._last_drive_note = ""
        result = super()._apply_motion(
            delta_pos, yaw, kind=kind, pre_pose=pre_pose, token=token, continuous=continuous
        )
        # Surface a drive-level event (e.g. a reach clamp at the workspace boundary)
        # on the step record, so the runner/logs explain a move that fell short.
        if self._last_drive_note:
            result.note = (
                f"{result.note}; {self._last_drive_note}"
                if result.note
                else self._last_drive_note
            )
            result.target_pose = self.target_pose  # setpoint was re-synced by the clamp
            self._last_drive_note = ""
        return self._guard_divergence(result)

    def _guard_divergence(self, result: AtomicStepResult) -> AtomicStepResult:
        """Re-sync the setpoint to the measured pose when it has run away from the arm.

        The node drops commands silently (not enabled / wrong mode) and IK failures are
        unreported, so the open-loop setpoint can diverge and then slew in one violent
        catch-up move when commands resume. Monitor BOTH position (``divergence_resync_m``)
        AND orientation (quaternion angle, ``divergence_resync_rad``). In cartesian
        operation the orientation is held constant, so the orientation check only trips
        on a runaway from dropped ROTATE_* commands (which carry zero position delta and
        would otherwise be invisible to the position guard).
        """
        if result.post_pose is None:
            return result
        measured = np.asarray(result.post_pose, dtype=float)
        # What should the arm be at? On the joint-stream backend, compare against the
        # pose we actually STREAMED (FK of the last joint command): with a bounded
        # orientation bend the streamed pose legitimately deviates from the Cartesian
        # setpoint by up to ori_flex_rad, and only "measured vs streamed" isolates the
        # guard's real target -- dropped/ignored commands. The endpose backend has no
        # joint truth, so it keeps comparing against the setpoint.
        if self._joint_stream_ok and self._q_cmd is not None:
            expect_pos, expect_rot = self._kin.fk(self._q_cmd)
            expect_quat = R.from_matrix(expect_rot).as_quat()
        else:
            expect_pos = self._target_pos
            expect_quat = R.from_euler("xyz", self._target_euler).as_quat()
        pos_gap = float(np.linalg.norm(expect_pos - measured[:3]))
        ori_gap = float((R.from_quat(expect_quat) * R.from_quat(measured[3:]).inv()).magnitude())
        pos_diverged = self.divergence_resync_m is not None and pos_gap > self.divergence_resync_m
        ori_diverged = self.divergence_resync_rad is not None and ori_gap > self.divergence_resync_rad
        if pos_diverged or ori_diverged:
            self._target_pos = measured[:3].copy()
            self._target_euler = quat_to_euler(measured[3:])
            # Also drop the interface's cached command pose so a following gripper-only
            # command (set_gripper_position re-commands the last pose) falls back to the
            # measured pose instead of re-lunging to the diverged setpoint we abandoned.
            invalidate = getattr(self.robot, "invalidate_commanded_pose", None)
            if callable(invalidate):
                invalidate()
            # Joint backend: re-seed the IK from the MEASURED joints -- the last
            # streamed solution is wherever the dropped commands were headed, not
            # where the arm actually is.
            if self._joint_stream_ok:
                try:
                    self._q_cmd = np.asarray(
                        self.robot.get_joint_positions(), dtype=float
                    ).reshape(-1)[:6]
                except Exception:  # noqa: BLE001 - keep the old seed; IK still converges
                    pass
            # Keep the returned record consistent with the re-synced setpoint.
            result.target_pose = self.target_pose
            note = (
                f"divergence: measured pose {pos_gap:.3f}m / {np.rad2deg(ori_gap):.0f}deg "
                "from the commanded pose -> setpoint re-synced to measured. Commands may be "
                "getting dropped (arm not enabled / node not in mode 1) or the target is "
                "unreachable."
            )
            print(f"[{self.LOG_TAG}] WARNING: {note}")
            result.note = f"{result.note}; {note}" if result.note else note
        return result
