"""Piper real-robot session: connect, wait for feedback, capture observations.

The Piper counterpart of :class:`core.franka.franka_session.FrankaSession`,
exposing the SAME :meth:`get_observation` contract so the atomic controllers,
runners and teleop collector work unchanged:

    ``agentview``     front-camera RGB frame (HxWx3 uint8),
    ``wrist``         wrist-camera RGB frame,
    ``ee_pose``       measured EEF pose [x,y,z,qx,qy,qz,qw],
    ``gripper_width`` measured gripper jaw opening (m).

Responsibilities:
  * connect to one Piper arm node (or a MockPiperRobot) over ROS topics,
  * subscribe to the front + wrist camera color topics (or MockCameras) -- the rig
    uses Orbbec DaBai DC cameras driven by ``astra_camera`` (``roslaunch
    astra_camera multi_camera.launch``), so the feed is consumed as ROS topics
    (``/camera_f/color/image_raw`` etc.), NOT opened directly like RealSense.

There is no impedance controller to start: the arm node (mode 1) consumes
streamed absolute PosCmd setpoints directly. Like the Franka session, this
deliberately does **not** auto-home the arm: atomic commands move relative to
wherever the arm already is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import core.franka.camera_utils as camera_utils  # MockCamera + resize_with_pad
from core.piper.piper_interface import (
    PIPER_DEFAULT_OPEN_WIDTH_M,
    MockPiperRobot,
    PiperInterface,
)
from core.piper.ros_camera import RosImageCamera


@dataclass
class PiperSessionConfig:
    """Connection + camera settings for a Piper real-robot session."""

    # Arm (ROS-topic namespace; the node itself is launched separately, see
    # scripts/piper_left.launch).
    arm: str = "left"
    use_mock_robot: bool = False
    open_width_m: float = PIPER_DEFAULT_OPEN_WIDTH_M
    feedback_timeout_s: float = 5.0
    max_feedback_age_s: float = 0.5

    # Cameras: ROS color-image topics published by astra_camera multi_camera.launch
    # (front = camera_f, left wrist = camera_l, right wrist = camera_r).
    connect_cameras: bool = True
    use_mock_cameras: bool = False
    front_camera_topic: str = "/camera_f/color/image_raw"
    wrist_camera_topic: str = "/camera_l/color/image_raw"
    camera_max_age_s: float = 1.0
    camera_connect_timeout_s: float = 10.0
    # Frame size for the mock cameras only (real frames come sized from the driver).
    mock_camera_width: int = 640
    mock_camera_height: int = 480
    # Square crop/pad size for the views fed to the VLM (robot yaml camera_resolution).
    observation_resolution: Optional[int] = 256

    verbose: bool = True


class PiperSession:
    """Owns the robot + camera handles for a Piper atomic-control session."""

    def __init__(self, config: Optional[PiperSessionConfig] = None) -> None:
        self.config = config or PiperSessionConfig()
        self.robot: Any = None
        self.front_cam: Any = None
        self.wrist_cam: Any = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "PiperSession":
        """Connect the robot + cameras."""
        self._connect_robot()
        self._connect_cameras()
        self._connected = True
        return self

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _connect_robot(self) -> None:
        cfg = self.config
        if cfg.use_mock_robot:
            self._log(f"[session] MOCK Piper robot ({cfg.arm} arm)")
            self.robot = MockPiperRobot(arm=cfg.arm, open_width_m=cfg.open_width_m)
        else:
            self._log(f"[session] Connecting to Piper {cfg.arm} arm over ROS topics ...")
            self.robot = PiperInterface(
                arm=cfg.arm,
                open_width_m=cfg.open_width_m,
                feedback_timeout_s=cfg.feedback_timeout_s,
                max_feedback_age_s=cfg.max_feedback_age_s,
                verbose=cfg.verbose,
            )
        self.robot.connect()

        # Sanity read (mirrors FrankaSession).
        ee_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        gripper = float(self.robot.get_gripper_position()[0])
        self._log(
            f"[session]   ee_pose={np.round(ee_pose, 4).tolist()} gripper={gripper*1000:.1f}mm"
        )

    def _connect_cameras(self) -> None:
        cfg = self.config
        if not cfg.connect_cameras:
            self._log("[session] cameras disabled (connect_cameras=False)")
            return
        if cfg.use_mock_cameras:
            self._log("[session] MOCK cameras")
            self.front_cam = camera_utils.MockCamera(
                width=cfg.mock_camera_width, height=cfg.mock_camera_height
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=cfg.mock_camera_width, height=cfg.mock_camera_height
            )
            return

        self._log(
            f"[session] Subscribing to camera topics: front={cfg.front_camera_topic} "
            f"wrist={cfg.wrist_camera_topic}"
        )
        self.front_cam = RosImageCamera(
            cfg.front_camera_topic,
            max_age_s=cfg.camera_max_age_s,
            connect_timeout_s=cfg.camera_connect_timeout_s,
            verbose=cfg.verbose,
        )
        self.wrist_cam = RosImageCamera(
            cfg.wrist_camera_topic,
            max_age_s=cfg.camera_max_age_s,
            connect_timeout_s=cfg.camera_connect_timeout_s,
            verbose=cfg.verbose,
        )

    def close(self) -> None:
        """Release cameras and close the robot connection."""
        for cam in (self.front_cam, self.wrist_cam):
            if cam is not None:
                try:
                    cam.release()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    self._log(f"[session] camera release error: {exc}")
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[session] robot close error: {exc}")
        self._connected = False

    def __enter__(self) -> "PiperSession":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- observations ------------------------------------------------------
    def _read_rgb(self, cam: Any, name: str) -> np.ndarray:
        success, frame, _ = cam.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Failed to read a fresh frame from the {name} camera topic "
                "(no message, or the last frame is stale). Is astra_camera "
                "multi_camera.launch running?"
            )
        # RosImageCamera already returns RGB (and MockCamera's channel order is
        # irrelevant), so there is no BGR swap here -- unlike the RealSense path.
        frame = np.asarray(frame)
        res = self.config.observation_resolution
        if res:
            frame = camera_utils.resize_with_pad(frame, res, res)
        return np.ascontiguousarray(frame)

    def get_observation(self) -> dict[str, Any]:
        """Capture a fresh observation (same contract as FrankaSession)."""
        if not self._connected:
            raise RuntimeError("PiperSession.get_observation called before connect()")
        if self.front_cam is None or self.wrist_cam is None:
            raise RuntimeError(
                "get_observation requires cameras, but the session was created with "
                "connect_cameras=False"
            )
        agentview = self._read_rgb(self.front_cam, "front")
        wrist = self._read_rgb(self.wrist_cam, "wrist")
        ee_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        gripper_width = float(self.robot.get_gripper_position()[0])
        return {
            "agentview": agentview,
            "wrist": wrist,
            "ee_pose": ee_pose,
            "gripper_width": gripper_width,
        }
