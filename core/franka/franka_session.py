"""Franka real-robot session: connect, start impedance, capture observations.

This is the *environment setup* half of the real-robot loop (session construction
+ ``reset_to_initial_state``) for the real robot, without any policy/openpi
dependency. It is used by the atomic-action test script and by the real-robot
episode runner.

Responsibilities:
  * connect to the Franka NUC (or a MockRobot) via ZeroRPC,
  * start the Cartesian-impedance controller (eef control),
  * connect the external + wrist RealSense cameras (or MockCameras),
  * expose :meth:`get_observation` returning agentview/wrist RGB frames and pose.

It deliberately does **not** auto-home the arm: it starts impedance at the
current pose so atomic commands move relative to wherever the arm already is.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import core.franka.camera_utils as camera_utils
from core.franka.franka_interface import FrankaInterface, MockRobot

try:  # OpenCV is only needed for the real (BGR) camera path.
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - mock path does not need cv2
    _CV2_AVAILABLE = False

# Cartesian-impedance gains (position xyz + orientation rpy), from main_pi05.py.
DEFAULT_KX = (750.0, 750.0, 750.0, 15.0, 15.0, 15.0)
DEFAULT_KXD = (37.0, 37.0, 37.0, 2.0, 2.0, 2.0)


@dataclass
class FrankaSessionConfig:
    """Connection + camera settings for a real-robot session."""

    # Robot (NUC ZeroRPC server). Defaults match baselines.pi05.config.RobotConfig.
    nuc_ip: str = "127.0.0.1"  # placeholder; pass your NUC address (configs/site/)
    nuc_port: int = 4242
    use_mock_robot: bool = False
    start_impedance: bool = True
    kx: tuple[float, ...] = DEFAULT_KX
    kxd: tuple[float, ...] = DEFAULT_KXD

    # Cameras.
    connect_cameras: bool = True
    use_mock_cameras: bool = False
    external_camera_serial: Optional[str] = None
    wrist_camera_serial: Optional[str] = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_read_timeout_ms: int = 3000
    camera_read_retries: int = 2
    camera_read_retry_delay_s: float = 0.05
    camera_restart_on_read_failure: bool = True
    # Square crop/pad size for the views fed to the VLM (robot_franka.yaml camera_resolution).
    observation_resolution: Optional[int] = 256
    # Optional second, HIGH-RES square render of the same frames (``agentview_hd`` /
    # ``wrist_hd`` observation keys) for consumers that need legible detail -- the
    # affordance pointer reads glyphs the 256 px observation destroys. Same
    # resize_with_pad geometry, so the 0-1000 pointing grid is identical across both
    # renders. None -> the hd keys are absent and nothing else changes.
    hd_resolution: Optional[int] = None

    verbose: bool = True


class FrankaSession:
    """Owns the robot + camera handles for a real-robot atomic-control session."""

    def __init__(self, config: Optional[FrankaSessionConfig] = None) -> None:
        self.config = config or FrankaSessionConfig()
        self.robot: Any = None
        self.external_cam: Any = None
        self.wrist_cam: Any = None
        self._connected = False
        # Serializes camera reads between the rollout (get_observation) and a live
        # display poller (get_camera_frames), so the two never interleave mid-frame.
        self._camera_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "FrankaSession":
        """Connect the robot + cameras and start the impedance controller."""
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
            self._log(f"[session] MOCK robot at {cfg.nuc_ip}:{cfg.nuc_port}")
            self.robot = MockRobot(ip=cfg.nuc_ip, port=cfg.nuc_port)
        else:
            self._log(f"[session] Connecting to Franka NUC at {cfg.nuc_ip}:{cfg.nuc_port} ...")
            self.robot = FrankaInterface(ip=cfg.nuc_ip, port=cfg.nuc_port)

        # Sanity read + (optionally) start the Cartesian-impedance controller.
        ee_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        gripper = float(self.robot.get_gripper_position()[0])
        self._log(
            f"[session]   ee_pose={np.round(ee_pose, 4).tolist()} gripper={gripper*1000:.1f}mm"
        )
        if cfg.start_impedance:
            self._log("[session]   starting Cartesian impedance controller ...")
            self.start_impedance()

    def start_impedance(self) -> None:
        """(Re)start the Cartesian-impedance controller at the arm's current pose.

        Best-effort terminates any policy already running first, so this works both as the
        initial start and as a *restart* after the controller was lost -- e.g. a joint move
        (``move_to_joint_positions`` / go_home) preempted it, or a server-side reflex
        terminated it. Calling it with nothing running is safe (the spurious terminate is
        swallowed). Wired as the recovery hook for FrankaAtomicController so a teleop /
        rollout setpoint command self-heals instead of crashing with "no controller running".
        """
        cfg = self.config
        try:
            self.robot.terminate_current_policy()
        except Exception:  # noqa: BLE001 - nothing running to terminate is fine
            pass
        self.robot.start_cartesian_impedance(
            Kx=np.asarray(cfg.kx, dtype=float), Kxd=np.asarray(cfg.kxd, dtype=float)
        )

    def _connect_cameras(self) -> None:
        cfg = self.config
        if not cfg.connect_cameras:
            self._log("[session] cameras disabled (connect_cameras=False)")
            return
        if cfg.use_mock_cameras:
            self._log("[session] MOCK cameras")
            self.external_cam = camera_utils.MockCamera(
                width=cfg.camera_width, height=cfg.camera_height
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=cfg.camera_width, height=cfg.camera_height
            )
            return

        devices = camera_utils.list_realsense_devices()
        self._log(f"[session] Found {len(devices)} RealSense device(s)")
        if not devices:
            raise RuntimeError(
                "No RealSense cameras found. Pass use_mock_cameras=True for a dry run, "
                "or check the camera USB connection."
            )
        self.external_cam = camera_utils.RealSenseCamera(
            serial_number=cfg.external_camera_serial,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
            read_timeout_ms=cfg.camera_read_timeout_ms,
            read_retries=cfg.camera_read_retries,
            read_retry_delay=cfg.camera_read_retry_delay_s,
            restart_on_read_failure=cfg.camera_restart_on_read_failure,
        )
        self.wrist_cam = camera_utils.RealSenseCamera(
            serial_number=cfg.wrist_camera_serial,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
            read_timeout_ms=cfg.camera_read_timeout_ms,
            read_retries=cfg.camera_read_retries,
            read_retry_delay=cfg.camera_read_retry_delay_s,
            restart_on_read_failure=cfg.camera_restart_on_read_failure,
        )

    def close(self) -> None:
        """Release cameras and terminate the controller / close the robot connection."""
        for cam in (self.external_cam, self.wrist_cam):
            if cam is not None:
                try:
                    cam.release()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    self._log(f"[session] camera release error: {exc}")
        if self.robot is not None:
            try:
                self.robot.terminate_current_policy()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[session] terminate_current_policy error: {exc}")
            try:
                self.robot.close()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[session] robot close error: {exc}")
        self._connected = False

    def __enter__(self) -> "FrankaSession":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- observations ------------------------------------------------------
    def _read_raw_rgb(self, cam: Any, name: str) -> np.ndarray:
        """One camera read at native resolution, converted to RGB (no resize)."""
        success, frame, _ = cam.read()
        if not success or frame is None:
            raise RuntimeError(f"Failed to read frame from {name} camera")
        frame = np.asarray(frame)
        # Real RealSense frames are BGR (bgr8); convert to RGB. Mock frames are random
        # uint8 and channel order is irrelevant, but converting keeps a single path.
        if _CV2_AVAILABLE and not self.config.use_mock_cameras:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _resize(self, frame: np.ndarray, res: Optional[int]) -> np.ndarray:
        if res:
            frame = camera_utils.resize_with_pad(frame, res, res)
        return np.ascontiguousarray(frame)

    def _read_rgb(self, cam: Any, name: str) -> np.ndarray:
        return self._resize(self._read_raw_rgb(cam, name), self.config.observation_resolution)

    def get_observation(self) -> dict[str, Any]:
        """Capture a fresh observation.

        Returns a dict with:
            ``agentview``    external-camera RGB frame (HxWx3 uint8),
            ``wrist``        wrist-camera RGB frame (HxWx3 uint8),
            ``ee_pose``      measured EEF pose [x,y,z,qx,qy,qz,qw],
            ``gripper_width``measured gripper width (m).
        """
        if not self._connected:
            raise RuntimeError("FrankaSession.get_observation called before connect()")
        if self.external_cam is None or self.wrist_cam is None:
            raise RuntimeError(
                "get_observation requires cameras, but the session was created with "
                "connect_cameras=False"
            )
        obs_res = self.config.observation_resolution
        hd_res = self.config.hd_resolution
        with self._camera_lock:
            raw_agent = self._read_raw_rgb(self.external_cam, "external")
            raw_wrist = self._read_raw_rgb(self.wrist_cam, "wrist")
        ee_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        gripper_width = float(self.robot.get_gripper_position()[0])
        obs: dict[str, Any] = {
            "agentview": self._resize(raw_agent, obs_res),
            "wrist": self._resize(raw_wrist, obs_res),
            "ee_pose": ee_pose,
            "gripper_width": gripper_width,
        }
        if hd_res:
            # Same frames, same pad geometry, more pixels: the 0-1000 grid maps to
            # the same physical spot in both renders (one camera read per view --
            # the hd render is a resize, never a second capture).
            obs["agentview_hd"] = self._resize(raw_agent, hd_res)
            obs["wrist_hd"] = self._resize(raw_wrist, hd_res)
        return obs

    def get_camera_frames(self) -> Optional[dict[str, np.ndarray]]:
        """Latest two camera frames for a LIVE DISPLAY poller.

        Cameras only (no robot state), sharing ``_camera_lock`` with the rollout's
        observations so the reads never interleave mid-frame. Any problem (not
        connected yet, a read timeout, shutdown race) returns ``None`` instead of
        raising: a cosmetic display thread must never crash or perturb the rollout.
        Mirrors ``DualPiperSession.get_camera_frames`` with the single-arm keys
        (``agentview`` / ``wrist``), which is also what selects the live window's
        single-arm layout.
        """
        if not self._connected or self.external_cam is None or self.wrist_cam is None:
            return None
        try:
            with self._camera_lock:
                return {
                    "agentview": self._read_rgb(self.external_cam, "external"),
                    "wrist": self._read_rgb(self.wrist_cam, "wrist"),
                }
        except Exception:  # noqa: BLE001 - display is best-effort by contract
            return None
