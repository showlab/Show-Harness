"""Dual-Piper session: both arms + three cameras, one observation contract.

Owns the LEFT and RIGHT Piper arm connections (:class:`PiperInterface` or mocks)
plus the three rig cameras (front ``camera_f``, left wrist ``camera_l``, right
wrist ``camera_r``, all ROS color-image topics from ``astra_camera
multi_camera.launch``). One front subscription is shared by both arms.

Observation contract (:meth:`get_observation`):

    ``agentview``      front-camera RGB frame (HxWx3 uint8),
    ``wrist_left``     left-wrist RGB frame,
    ``wrist_right``    right-wrist RGB frame,
    ``left`` / ``right``   per-arm state: ``{"ee_pose": [x,y,z,qx,qy,qz,qw],
                           "gripper_width": m}``.

:meth:`arm_observation` re-slices a dual observation into the SINGLE-arm contract
(``agentview``/``wrist``/``ee_pose``/``gripper_width``) so the per-arm recorders
and any single-arm consumer work unchanged (dual-collection Mode A).

Like :class:`core.piper.piper_session.PiperSession`, this does not auto-home the
arms; the collection script owns homing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import core.franka.camera_utils as camera_utils  # MockCamera + resize_with_pad
from core.piper.config import SIDES  # single source of truth for ("left", "right")
from core.piper.piper_interface import (
    PIPER_DEFAULT_OPEN_WIDTH_M,
    MockPiperRobot,
    PiperInterface,
)
from core.piper.ros_camera import RosImageCamera

__all__ = ["SIDES", "DualPiperSessionConfig", "DualPiperSession", "ArmSessionView"]


@dataclass
class DualPiperSessionConfig:
    """Connection + camera settings for a dual-Piper session."""

    use_mock_robots: bool = False
    open_width_left_m: float = PIPER_DEFAULT_OPEN_WIDTH_M
    open_width_right_m: float = PIPER_DEFAULT_OPEN_WIDTH_M
    feedback_timeout_s: float = 5.0
    max_feedback_age_s: float = 0.5

    connect_cameras: bool = True
    use_mock_cameras: bool = False
    front_camera_topic: str = "/camera_f/color/image_raw"
    wrist_left_camera_topic: str = "/camera_l/color/image_raw"
    wrist_right_camera_topic: str = "/camera_r/color/image_raw"
    camera_max_age_s: float = 1.0
    camera_connect_timeout_s: float = 10.0
    mock_camera_width: int = 640
    mock_camera_height: int = 480
    # Square crop/pad size for stored/displayed views (robot yaml camera_resolution).
    observation_resolution: Optional[int] = 256

    verbose: bool = True


class DualPiperSession:
    """Owns both robot handles + the three cameras for dual-arm collection."""

    def __init__(self, config: Optional[DualPiperSessionConfig] = None) -> None:
        self.config = config or DualPiperSessionConfig()
        self.robots: dict[str, Any] = {"left": None, "right": None}
        self.front_cam: Any = None
        self.wrist_cams: dict[str, Any] = {"left": None, "right": None}
        self._connected = False
        # Serializes observation capture. Two independent single-arm runners (Mode A)
        # read through this session concurrently from their own threads; the ROS
        # camera/pose reads are cheap, so one lock keeps every capture consistent
        # without meaningfully costing the loop.
        self._obs_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "DualPiperSession":
        self._connect_robots()
        self._connect_cameras()
        self._connected = True
        return self

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _connect_robots(self) -> None:
        cfg = self.config
        widths = {"left": cfg.open_width_left_m, "right": cfg.open_width_right_m}
        for side in SIDES:
            if cfg.use_mock_robots:
                self._log(f"[dual-session] MOCK Piper robot ({side} arm)")
                robot: Any = MockPiperRobot(arm=side, open_width_m=widths[side])
            else:
                self._log(f"[dual-session] Connecting to Piper {side} arm over ROS topics ...")
                robot = PiperInterface(
                    arm=side,
                    open_width_m=widths[side],
                    feedback_timeout_s=cfg.feedback_timeout_s,
                    max_feedback_age_s=cfg.max_feedback_age_s,
                    verbose=cfg.verbose,
                )
            robot.connect()
            # Connection proof: the arm answers with a live pose + gripper reading. The
            # raw pose is not printed (the operator watches the arms, not coordinates);
            # a failed read raises here anyway, which is the thing worth knowing.
            robot.get_ee_pose()
            gripper = float(robot.get_gripper_position()[0])
            self._log(f"  {side.upper():<5} connected  ·  gripper {gripper * 1000:.0f} mm")
            self.robots[side] = robot

    def _make_camera(self, topic: str) -> Any:
        cfg = self.config
        if cfg.use_mock_cameras:
            return camera_utils.MockCamera(
                width=cfg.mock_camera_width, height=cfg.mock_camera_height
            )
        return RosImageCamera(
            topic,
            max_age_s=cfg.camera_max_age_s,
            connect_timeout_s=cfg.camera_connect_timeout_s,
            verbose=cfg.verbose,
        )

    def _connect_cameras(self) -> None:
        cfg = self.config
        if not cfg.connect_cameras:
            self._log("[dual-session] cameras disabled (connect_cameras=False)")
            return
        if cfg.use_mock_cameras:
            self._log("[dual-session] MOCK cameras (front + both wrists)")
        else:
            self._log(
                f"[dual-session] Subscribing to camera topics: front={cfg.front_camera_topic} "
                f"wrist_left={cfg.wrist_left_camera_topic} wrist_right={cfg.wrist_right_camera_topic}"
            )
        self.front_cam = self._make_camera(cfg.front_camera_topic)
        self.wrist_cams["left"] = self._make_camera(cfg.wrist_left_camera_topic)
        self.wrist_cams["right"] = self._make_camera(cfg.wrist_right_camera_topic)

    def close(self) -> None:
        for cam in (self.front_cam, self.wrist_cams["left"], self.wrist_cams["right"]):
            if cam is not None:
                try:
                    cam.release()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    self._log(f"[dual-session] camera release error: {exc}")
        for side in SIDES:
            if self.robots[side] is not None:
                try:
                    self.robots[side].close()
                except Exception as exc:  # noqa: BLE001
                    self._log(f"[dual-session] {side} robot close error: {exc}")
        self._connected = False

    def __enter__(self) -> "DualPiperSession":
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
        frame = np.asarray(frame)
        res = self.config.observation_resolution
        if res:
            frame = camera_utils.resize_with_pad(frame, res, res)
        return np.ascontiguousarray(frame)

    def get_observation(self) -> dict[str, Any]:
        """Capture one synchronized dual observation (see module docstring)."""
        self._require_cameras("get_observation")
        with self._obs_lock:
            obs: dict[str, Any] = {
                "agentview": self._read_rgb(self.front_cam, "front"),
                "wrist_left": self._read_rgb(self.wrist_cams["left"], "left wrist"),
                "wrist_right": self._read_rgb(self.wrist_cams["right"], "right wrist"),
            }
            for side in SIDES:
                robot = self.robots[side]
                obs[side] = {
                    "ee_pose": np.asarray(robot.get_ee_pose(), dtype=float),
                    "gripper_width": float(robot.get_gripper_position()[0]),
                }
        return obs

    def get_camera_frames(self) -> Optional[dict[str, np.ndarray]]:
        """Latest three camera frames for a LIVE DISPLAY poller.

        Cameras only (no robot state), sharing ``_obs_lock`` with the rollout's
        observations so the reads never interleave mid-frame. Any problem (stale
        topic, not connected yet, shutdown race) returns ``None`` instead of
        raising: a cosmetic display thread must never crash or perturb the rollout.
        """
        if not self._connected or self.front_cam is None:
            return None
        try:
            with self._obs_lock:
                return {
                    "agentview": self._read_rgb(self.front_cam, "front"),
                    "wrist_left": self._read_rgb(self.wrist_cams["left"], "left wrist"),
                    "wrist_right": self._read_rgb(self.wrist_cams["right"], "right wrist"),
                }
        except Exception:  # noqa: BLE001 - display is best-effort by contract
            return None

    def get_arm_observation(self, side: str) -> dict[str, Any]:
        """Capture ONE arm's single-arm observation (front + that wrist + that state).

        The Mode-A path: each independent runner observes only what its arm needs, so
        two concurrent runners do not pay for (or contend on) the other wrist camera.
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self._require_cameras("get_arm_observation")
        with self._obs_lock:
            robot = self.robots[side]
            return {
                "agentview": self._read_rgb(self.front_cam, "front"),
                "wrist": self._read_rgb(self.wrist_cams[side], f"{side} wrist"),
                "ee_pose": np.asarray(robot.get_ee_pose(), dtype=float),
                "gripper_width": float(robot.get_gripper_position()[0]),
            }

    def _require_cameras(self, caller: str) -> None:
        if not self._connected:
            raise RuntimeError(f"DualPiperSession.{caller} called before connect()")
        if self.front_cam is None:
            raise RuntimeError(
                f"{caller} requires cameras, but the session was created with "
                "connect_cameras=False"
            )

    @staticmethod
    def arm_observation(obs: dict[str, Any], side: str) -> dict[str, Any]:
        """Slice a dual observation into the single-arm contract for ``side``."""
        return {
            "agentview": obs["agentview"],
            "wrist": obs[f"wrist_{side}"],
            "ee_pose": obs[side]["ee_pose"],
            "gripper_width": obs[side]["gripper_width"],
        }


class ArmSessionView:
    """Single-arm facade over a shared :class:`DualPiperSession` (dual rollout Mode A).

    Exposes exactly the surface the single-arm stack consumes -- ``robot`` (for the
    controller build + homing) and ``get_observation()`` (the single-arm contract) --
    so an unchanged :class:`core.runners.real.RealEpisodeRunner` drives one arm while
    a sibling view drives the other from its own thread.

    Lifecycle stays with the OWNING dual session (``connect``/``close`` here are
    no-ops), and a shared ``stop_event`` turns the operator's Ctrl+C (delivered only
    to the main thread) into a KeyboardInterrupt at each runner's next observation, so
    both runners tear down exactly like a single-arm interrupt (video compiled,
    summary written).
    """

    def __init__(
        self,
        session: DualPiperSession,
        side: str,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.session = session
        self.side = side
        self.stop_event = stop_event

    @property
    def robot(self) -> Any:
        return self.session.robots[self.side]

    def get_observation(self) -> dict[str, Any]:
        if self.stop_event is not None and self.stop_event.is_set():
            raise KeyboardInterrupt(f"dual rollout stop requested ({self.side} arm)")
        return self.session.get_arm_observation(self.side)

    def connect(self) -> "ArmSessionView":
        return self  # the dual session owns connection

    def close(self) -> None:
        pass  # the dual session owns teardown
