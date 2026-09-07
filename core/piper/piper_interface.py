"""AgileX/Songling Piper arm interface: a ROS-topic client mirroring FrankaInterface.

Talks to ONE ``piper_start_ms_node.py`` (cobot_magic Piper_ros stack) launched in
mode 1 (software control), e.g. via ``scripts/piper_left.launch``. The node owns the
CAN bus / piper_sdk; this class only publishes and subscribes ROS topics, so it works
from any Python 3.8 environment with the ROS + Piper workspaces sourced:

    source /opt/ros/noetic/setup.bash
    source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

Topic contract (per arm, ``<arm>`` in {left, right}; verified against
piper_start_ms_node.py):

  feedback (published by the node at 200 Hz in every mode):
    /puppet/joint_<arm>            sensor_msgs/JointState: position[0:6] joints (rad),
                                   position[6] gripper jaw opening (METERS)
    /puppet/end_pose_euler_<arm>   piper_msgs/PosCmd: x,y,z meters; roll,pitch,yaw
                                   RADIANS (extrinsic xyz); gripper meters
    /puppet/arm_status_<arm>       piper_msgs/PiperStatusMsg: ctrl_mode, err_code, ...
  commands (consumed by the node in mode 1 only, silently DROPPED otherwise or when
  the node's enable flag is off -- hence the freshness/divergence guards here and in
  PiperAtomicController):
    /puppet/pos_cmd_<arm>          piper_msgs/PosCmd: ABSOLUTE end pose (meters/radians)
                                   -> EndPoseCtrl (MOVE P, 1 mm resolution) AND
                                   GripperCtrl(gripper meters) -- every pose command
                                   also commands the gripper, so this class always
                                   fills PosCmd.gripper with its tracked target width.
    /master/joint_<arm>            sensor_msgs/JointState: joint-space command
                                   (position[0:6] rad -> JointCtrl MOVE J,
                                   position[6] gripper meters -> GripperCtrl)
    /enable_flag                   std_msgs/Bool: GLOBAL enable/disable. CAUTION: the
                                   node's enable path commands the gripper to width 0
                                   (both arms if both nodes run) -- see enable().

Conventions at this boundary (identical to FrankaInterface so the atomic controller
and runners work unchanged): poses are [x,y,z,qx,qy,qz,qw] in the Piper base frame,
meters; quaternion scipy xyzw from/to extrinsic-xyz euler (bit-identical to the
node's tf ``quaternion_from_euler``); joints radians; gripper is a single jaw-opening
width in meters (0 closed .. ~0.07-0.08 open).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

PIPER_DOF = 6

# Command-side clamp for the gripper width (meters). The node's own upper clamp is
# buggy (it compares meters against 80000), so we enforce the 80000 um CAN limit here.
PIPER_GRIPPER_MAX_CMD_M = 0.08

# Default "open" width commanded by control_gripper(False). The Piper gripper's rated
# stroke is ~0.07 m; read the true fully-open width off /puppet/joint_<arm> during
# calibration and override via config if it differs.
PIPER_DEFAULT_OPEN_WIDTH_M = 0.07


def _require_ros():
    """Import the ROS + piper message modules, with an actionable error if missing."""
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "rospy / ROS messages are not importable. Run inside a shell with ROS "
            "Noetic sourced: `source /opt/ros/noetic/setup.bash` (and install "
            "`rospkg`/`catkin_pkg` into this Python env: pip install rospkg catkin_pkg)."
        ) from exc
    try:
        from piper_msgs.msg import PiperStatusMsg, PosCmd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "piper_msgs is not importable. Source the built Piper workspace first: "
            "`source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash`."
        ) from exc
    return rospy, JointState, Bool, PosCmd, PiperStatusMsg


class PiperInterface:
    """ROS-topic client for one Piper arm, duck-type compatible with FrankaInterface."""

    def __init__(
        self,
        arm: str = "left",
        open_width_m: float = PIPER_DEFAULT_OPEN_WIDTH_M,
        feedback_timeout_s: float = 5.0,
        max_feedback_age_s: float = 0.5,
        joint_stream_hz: float = 50.0,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            arm: Which arm's topic namespace to use ("left" or "right").
            open_width_m: Gripper width (m) commanded by ``control_gripper(False)``.
            feedback_timeout_s: Max wait for the first feedback messages on connect.
            max_feedback_age_s: Reads/commands raise if the cached feedback is older
                than this -- a dead/restarted node otherwise freezes the cached pose
                while publishes silently go nowhere.
            joint_stream_hz: Interpolation rate for ``move_to_joint_positions``.
        """
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.open_width_m = float(open_width_m)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.max_feedback_age_s = float(max_feedback_age_s)
        self.joint_stream_hz = float(joint_stream_hz)
        self.verbose = bool(verbose)

        self._ros = _require_ros()
        rospy = self._ros[0]
        # One shared node per process; disable_signals so our own Ctrl+C / pygame
        # handling keeps working and rospy does not hijack SIGINT.
        if not rospy.core.is_initialized():
            rospy.init_node("showharness_piper", anonymous=True, disable_signals=True)

        # Latest feedback, cached as (msg, receipt_monotonic) tuples (atomic swaps).
        self._joint_state: Optional[tuple[Any, float]] = None
        self._end_pose: Optional[tuple[Any, float]] = None
        self._arm_status: Optional[tuple[Any, float]] = None
        self._last_err_code: Optional[int] = None

        # The gripper width every PosCmd carries (the node commands the gripper on
        # EVERY pose command). Initialized from the MEASURED width in connect() so the
        # first setpoint cannot yank the fingers open/closed unexpectedly.
        self._gripper_target_m: Optional[float] = None
        # Last Cartesian pose we commanded; control_gripper re-commands it so a
        # gripper-only action does not move the arm. Invalidated by joint-space moves.
        self._last_commanded_pose: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._connected = False

        rospy_, JointState, Bool, PosCmd, PiperStatusMsg = self._ros
        ns = self.arm
        self._sub_joint = rospy_.Subscriber(
            f"/puppet/joint_{ns}", JointState, self._on_joint, queue_size=1, tcp_nodelay=True
        )
        self._sub_pose = rospy_.Subscriber(
            f"/puppet/end_pose_euler_{ns}", PosCmd, self._on_end_pose, queue_size=1, tcp_nodelay=True
        )
        self._sub_status = rospy_.Subscriber(
            f"/puppet/arm_status_{ns}", PiperStatusMsg, self._on_status, queue_size=1, tcp_nodelay=True
        )
        self._pub_pos_cmd = rospy_.Publisher(
            f"/puppet/pos_cmd_{ns}", PosCmd, queue_size=1, tcp_nodelay=True
        )
        self._pub_joint_cmd = rospy_.Publisher(
            f"/master/joint_{ns}", JointState, queue_size=1, tcp_nodelay=True
        )
        self._pub_enable = rospy_.Publisher("/enable_flag", Bool, queue_size=1, tcp_nodelay=True)

    # -- callbacks -----------------------------------------------------------
    def _on_joint(self, msg: Any) -> None:
        self._joint_state = (msg, time.monotonic())

    def _on_end_pose(self, msg: Any) -> None:
        self._end_pose = (msg, time.monotonic())

    def _on_status(self, msg: Any) -> None:
        self._arm_status = (msg, time.monotonic())
        err = int(getattr(msg, "err_code", 0))
        if err != self._last_err_code:
            if err != 0:
                print(f"[piper-{self.arm}] WARNING: arm_status err_code={err}")
            self._last_err_code = err

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> "PiperInterface":
        """Block until feedback flows and the command topics have a subscriber."""
        rospy = self._ros[0]
        deadline = time.monotonic() + self.feedback_timeout_s
        while self._joint_state is None or self._end_pose is None:
            if rospy.is_shutdown():
                raise RuntimeError("ROS is shutting down")
            if time.monotonic() > deadline:
                missing = [
                    name
                    for name, cached in (
                        (f"/puppet/joint_{self.arm}", self._joint_state),
                        (f"/puppet/end_pose_euler_{self.arm}", self._end_pose),
                    )
                    if cached is None
                ]
                raise RuntimeError(
                    f"No Piper feedback on {missing} after {self.feedback_timeout_s:.0f}s. "
                    "Is the arm node running? Start it with: roslaunch "
                    "scripts/piper_left.launch mode:=1 auto_enable:=true (CAN activated first)."
                )
            time.sleep(0.02)
        # The node subscribes with queue_size=1; wait for the connection handshake so
        # the first command is not silently dropped mid-connect.
        conn_deadline = time.monotonic() + 3.0
        while (
            self._pub_pos_cmd.get_num_connections() < 1
            or self._pub_joint_cmd.get_num_connections() < 1
        ):
            if time.monotonic() > conn_deadline:
                print(
                    f"[piper-{self.arm}] WARNING: command topics have no subscriber -- "
                    "the arm node is probably NOT in mode 1 (software control). "
                    "Commands will be dropped."
                )
                break
            time.sleep(0.02)
        # Adopt the physical gripper width as the initial target so the first PosCmd
        # (which always carries a gripper width) does not move the fingers.
        self._gripper_target_m = float(self.get_gripper_position()[0])
        self._connected = True
        if self.verbose:
            pose = self.get_ee_pose()
            print(
                f"[piper-{self.arm}] connected: ee_pos={np.round(pose[:3], 4).tolist()} "
                f"gripper={self._gripper_target_m * 1000:.1f}mm"
            )
        return self

    def close(self) -> None:
        for handle in (
            self._sub_joint,
            self._sub_pose,
            self._sub_status,
            self._pub_pos_cmd,
            self._pub_joint_cmd,
            self._pub_enable,
        ):
            try:
                handle.unregister()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._connected = False

    # -- feedback ------------------------------------------------------------
    def _fresh(self, cached: Optional[tuple[Any, float]], topic: str) -> Any:
        if cached is None:
            raise RuntimeError(f"No message received yet on {topic}; call connect() first.")
        msg, received = cached
        age = time.monotonic() - received
        if age > self.max_feedback_age_s:
            raise RuntimeError(
                f"Feedback on {topic} is stale ({age:.2f}s old > "
                f"{self.max_feedback_age_s:.2f}s). The arm node has likely died or the "
                "CAN link dropped; NOT proceeding on frozen state."
            )
        return msg

    def get_ee_pose(self) -> np.ndarray:
        """Measured EEF pose [x,y,z,qx,qy,qz,qw] (meters, Piper base frame).

        Built from the node's euler feedback with scipy extrinsic-xyz, which matches
        the node's own tf ``quaternion_from_euler`` bit-for-bit. NOTE: the SDK reports
        orientation as RPY; if the working grasp orientation sits near pitch +/-90 deg
        the euler<->quat round trip is gimbal-degenerate -- verify on hardware before
        trusting yaw edits there.
        """
        msg = self._fresh(self._end_pose, f"/puppet/end_pose_euler_{self.arm}")
        quat = R.from_euler("xyz", [msg.roll, msg.pitch, msg.yaw], degrees=False).as_quat()
        return np.concatenate([[msg.x, msg.y, msg.z], quat])

    def get_joint_positions(self) -> np.ndarray:
        """Measured joint positions, 6-vector (radians)."""
        msg = self._fresh(self._joint_state, f"/puppet/joint_{self.arm}")
        return np.asarray(msg.position[:PIPER_DOF], dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        """Measured joint velocities, 6-vector (rad/s)."""
        msg = self._fresh(self._joint_state, f"/puppet/joint_{self.arm}")
        return np.asarray(msg.velocity[:PIPER_DOF], dtype=float)

    def get_gripper_position(self) -> np.ndarray:
        """Measured gripper jaw opening as a 1-vector [width_m]."""
        msg = self._fresh(self._joint_state, f"/puppet/joint_{self.arm}")
        return np.asarray([msg.position[PIPER_DOF]], dtype=float)

    def get_force_torque(self) -> np.ndarray:
        """The Piper exposes no EE wrench estimate; zeros for API parity."""
        return np.zeros(6, dtype=float)

    def arm_status(self) -> Optional[Any]:
        """Latest PiperStatusMsg (err_code, ctrl_mode, ...) or None."""
        return self._arm_status[0] if self._arm_status is not None else None

    # -- commands ------------------------------------------------------------
    def _assert_commandable(self) -> None:
        # Freshness first: a dead node means publishes go nowhere while the cached
        # state freezes -- fail loudly instead.
        self._fresh(self._joint_state, f"/puppet/joint_{self.arm}")
        if self._pub_pos_cmd.get_num_connections() < 1:
            raise RuntimeError(
                f"/puppet/pos_cmd_{self.arm} has no subscriber -- the arm node is not "
                "in mode 1 (software control). Relaunch with mode:=1."
            )

    def _publish_pos_cmd(self, pose7: np.ndarray) -> None:
        PosCmd = self._ros[3]
        pose7 = np.asarray(pose7, dtype=float).reshape(-1)
        euler = R.from_quat(pose7[3:7]).as_euler("xyz", degrees=False)
        msg = PosCmd()
        msg.x, msg.y, msg.z = float(pose7[0]), float(pose7[1]), float(pose7[2])
        msg.roll, msg.pitch, msg.yaw = (float(v) for v in euler)
        width = 0.0 if self._gripper_target_m is None else self._gripper_target_m
        msg.gripper = float(np.clip(width, 0.0, PIPER_GRIPPER_MAX_CMD_M))
        msg.mode1 = 0
        msg.mode2 = 0
        self._pub_pos_cmd.publish(msg)
        self._last_commanded_pose = pose7[:7].copy()

    def update_desired_ee_pose(self, pose: np.ndarray) -> None:
        """Command an ABSOLUTE end pose [x,y,z,qx,qy,qz,qw] (plus the tracked gripper
        width -- the node commands both on every PosCmd)."""
        self._assert_commandable()
        self._publish_pos_cmd(pose)

    def control_gripper(self, gripper_action: bool) -> None:
        """Binary gripper: True=CLOSE (width 0), False=OPEN (open_width_m)."""
        self.set_gripper_position(0.0 if gripper_action else self.open_width_m)

    def set_gripper_position(self, pos: float) -> None:
        """Command the gripper to an absolute jaw opening (meters, clamped >= 0).

        Re-commands the last Cartesian setpoint (fallback: the measured pose) with the
        new width, so a gripper-only action never moves the arm.
        """
        self._assert_commandable()
        self._gripper_target_m = float(np.clip(pos, 0.0, PIPER_GRIPPER_MAX_CMD_M))
        pose = (
            self._last_commanded_pose
            if self._last_commanded_pose is not None
            else self.get_ee_pose()
        )
        self._publish_pos_cmd(pose)

    def get_gripper_prev_cmd_success(self) -> bool:
        """GripperCtrl is fire-and-forget on CAN; always True (API parity)."""
        return True

    def invalidate_commanded_pose(self) -> None:
        """Forget the last commanded Cartesian pose, so the next gripper-only command
        (``set_gripper_position``) falls back to the measured pose instead of re-issuing
        a stale one. Called by the controller after a divergence re-sync."""
        self._last_commanded_pose = None

    def note_commanded_pose(self, pose: np.ndarray) -> None:
        """Record ``pose`` [x,y,z,qx,qy,qz,qw] as the last commanded Cartesian pose
        WITHOUT publishing anything. Used by the joint-stream motion backend: the arm
        was driven there via JointCtrl, so a following gripper-only command
        (``set_gripper_position`` re-commands the last pose) must reference this target
        rather than a stale pre-stream one."""
        self._last_commanded_pose = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()

    def _publish_joint_cmd(self, q: np.ndarray, width_m: float) -> None:
        """Publish one JointState command on /master/joint_<arm> (node: MOVE J,
        speed 100, plus GripperCtrl with position[6])."""
        rospy, JointState = self._ros[0], self._ros[1]
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"joint{j}" for j in range(PIPER_DOF + 1)]
        msg.position = [float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:PIPER_DOF]] + [
            float(np.clip(width_m, 0.0, PIPER_GRIPPER_MAX_CMD_M))
        ]
        msg.velocity = [0.0] * (PIPER_DOF + 1)
        msg.effort = [0.0] * (PIPER_DOF + 1)
        self._pub_joint_cmd.publish(msg)

    def stream_joints(self, positions: np.ndarray) -> None:
        """Stream ONE joint-space waypoint (6-vector, rad) -- the smooth command path.

        The node maps this to firmware MOVE J at speed 100 (the same mechanism the
        vendor's master-slave teleop and trajectory replay use), which tracks a stream
        of nearby targets continuously -- unlike EndPoseCtrl's MOVE P, which plans an
        independent accelerate/decelerate move per command. The tracked gripper width
        rides along on every message (the node commands the gripper with position[6]).
        """
        self._assert_commandable()
        width = (
            self._gripper_target_m
            if self._gripper_target_m is not None
            else float(self.get_gripper_position()[0])
        )
        self._publish_joint_cmd(positions, width)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float) -> None:
        """Joint-space move (homing): stream a linear interpolation from the measured
        joints to ``positions`` (6-vector, rad) on /master/joint_<arm>, then block
        until the arm converges (or a timeout elapses).

        Invalidates the cached Cartesian setpoint: the caller MUST re-sync its atomic
        controller (``sync_from_robot``) afterwards -- and the next gripper-only
        command falls back to the measured pose instead of lunging to the stale one.
        """
        self._assert_commandable()
        target = np.asarray(positions, dtype=float).reshape(-1)[:PIPER_DOF]
        start = self.get_joint_positions()
        width = (
            self._gripper_target_m
            if self._gripper_target_m is not None
            else float(self.get_gripper_position()[0])
        )
        steps = max(2, int(round(self.joint_stream_hz * max(0.1, float(time_to_go)))))
        period = max(0.1, float(time_to_go)) / steps
        for i in range(1, steps + 1):
            alpha = i / steps
            self._publish_joint_cmd((1.0 - alpha) * start + alpha * target, width)
            time.sleep(period)
        # Block until the measured joints settle at the target so the caller's
        # subsequent sync captures a converged pose, not an in-flight one.
        deadline = time.monotonic() + 3.0
        tol_rad = 0.05
        while time.monotonic() < deadline:
            if float(np.max(np.abs(self.get_joint_positions() - target))) < tol_rad:
                break
            time.sleep(0.05)
        else:
            print(
                f"[piper-{self.arm}] WARNING: joint move did not converge within "
                f"{tol_rad} rad; max err "
                f"{float(np.max(np.abs(self.get_joint_positions() - target))):.3f} rad"
            )
        self._last_commanded_pose = None

    def enable(self) -> None:
        """(Re)enable the arm motors via /enable_flag.

        CAUTION: the node's enable path also commands the gripper to width 0 (CLOSED),
        on EVERY arm node subscribed to the global /enable_flag. Never call this while
        the fingers hold an object or are near a pinch point. The measured width is
        re-adopted as the gripper target afterwards.
        """
        Bool = self._ros[2]
        self._pub_enable.publish(Bool(data=True))
        time.sleep(0.5)
        try:
            self._gripper_target_m = float(self.get_gripper_position()[0])
        except RuntimeError:
            pass

    def disable(self) -> None:
        """Disable the arm motors via /enable_flag (GLOBAL: affects every arm node).

        CAUTION: this de-energizes the gripper (a held object can drop) and, unless the
        joints are brake-held, the arm itself may fall -- support it first.
        """
        Bool = self._ros[2]
        self._pub_enable.publish(Bool(data=False))

    # -- API-parity no-ops (the Piper node has no server-side policy lifecycle) ----
    def start_cartesian_impedance(self, Kx: Any = None, Kxd: Any = None) -> None:  # noqa: N803
        """No-op: the Piper node consumes streamed PosCmd directly (MOVE P)."""

    def start_joint_impedance(self, Kq: Any = None, Kqd: Any = None) -> None:  # noqa: N803
        """No-op: joint commands stream to /master/joint_<arm> directly."""

    def terminate_current_policy(self) -> None:
        """No-op: there is no server-side policy to terminate."""


class MockPiperRobot:
    """Simulated Piper arm with the same interface (no ROS, no hardware).

    Synchronous: poses and gripper widths update instantly, so controllers should be
    configured with ``gripper_settle_s=0``.
    """

    # A typical mid-workspace ready pose (the left rig's begin pose): clear of joint
    # limits and the j5=0 wrist singularity, so IK from here behaves like on hardware.
    HOME_JOINTS = np.array([-0.44299, 0.85654, -0.9814, 0.02017, 0.92179, 0.0])

    def __init__(
        self, arm: str = "left", open_width_m: float = PIPER_DEFAULT_OPEN_WIDTH_M
    ) -> None:
        from core.piper.kinematics import PiperKinematics  # pure math, no ROS

        self.arm = arm
        self.open_width_m = float(open_width_m)
        self._kin = PiperKinematics(0x01)
        self._joints = self.HOME_JOINTS.copy()
        # FK-CONSISTENT feedback: the mock's pose is always the FK of its joints, so
        # the joint-stream motion backend (DH variant selection, IK seeding, the
        # divergence guard) exercises the same math it runs on hardware.
        self._pose = self._kin.fk_pose7(self._joints)
        # Start fully open so a fresh session reads OPEN (intuitive for teleop).
        self._gripper_m = self.open_width_m
        print(f"  [MockPiperRobot] Simulating Piper ({arm}) -- no CAN/ROS connection")

    def connect(self) -> "MockPiperRobot":
        return self

    def close(self) -> None:
        pass

    def get_ee_pose(self) -> np.ndarray:
        return self._pose.copy()

    def get_joint_positions(self) -> np.ndarray:
        return self._joints.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(PIPER_DOF)

    def get_gripper_position(self) -> np.ndarray:
        return np.asarray([self._gripper_m], dtype=float)

    def get_force_torque(self) -> np.ndarray:
        return np.zeros(6)

    def arm_status(self) -> None:
        return None

    def update_desired_ee_pose(self, pose: np.ndarray) -> None:
        self._pose = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()

    def control_gripper(self, gripper_action: bool) -> None:
        self._gripper_m = 0.0 if gripper_action else self.open_width_m

    def set_gripper_position(self, pos: float) -> None:
        self._gripper_m = float(np.clip(pos, 0.0, PIPER_GRIPPER_MAX_CMD_M))

    def get_gripper_prev_cmd_success(self) -> bool:
        return True

    def invalidate_commanded_pose(self) -> None:
        pass

    def note_commanded_pose(self, pose: np.ndarray) -> None:
        # Mock semantic mirrors update_desired_ee_pose: "the arm went where commanded".
        # The joint-stream backend notes the Cartesian target it drove to via IK; the
        # mock adopts it so pose feedback (and the divergence guard) track the stream.
        self._pose = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()

    def stream_joints(self, positions: np.ndarray) -> None:
        self._joints = np.asarray(positions, dtype=float).reshape(-1)[:PIPER_DOF].copy()
        self._pose = self._kin.fk_pose7(self._joints)  # feedback follows the stream

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float) -> None:
        self._joints = np.asarray(positions, dtype=float).reshape(-1)[:PIPER_DOF].copy()
        self._pose = self._kin.fk_pose7(self._joints)

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def start_cartesian_impedance(self, Kx: Any = None, Kxd: Any = None) -> None:  # noqa: N803
        pass

    def start_joint_impedance(self, Kq: Any = None, Kqd: Any = None) -> None:  # noqa: N803
        pass

    def terminate_current_policy(self) -> None:
        pass
