"""ROS-topic RGB camera client for the Piper rig.

The AgileX Piper rig uses Orbbec DaBai DC cameras driven by the ``astra_camera``
ROS package (``roslaunch astra_camera multi_camera.launch``), NOT Intel RealSense.
So the camera feed, like the arm, is consumed as ROS **topics** rather than opened
directly -- this client subscribes to a ``sensor_msgs/Image`` color topic
(``/camera_f/color/image_raw`` etc.) and hands back RGB frames with the same
``read() -> (success, frame, depth)`` shape the RealSense camera used, so
``PiperSession`` stays symmetric with ``FrankaSession``.

Images are decoded manually (no ``cv_bridge`` dependency, which does not import
cleanly in a plain venv): ``sensor_msgs/Image`` carries raw ``data`` + ``encoding``
+ ``step``, so a reshape is all that is needed. Both ``rgb8`` and ``bgr8`` are
handled; the astra ``dabai.launch`` publishes ``rgb8`` (``color_format: RGB``).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np


def _require_ros():
    try:
        import rospy
        from sensor_msgs.msg import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "rospy / sensor_msgs are not importable. Run inside a shell with ROS "
            "Noetic sourced: `source /opt/ros/noetic/setup.bash`."
        ) from exc
    return rospy, Image


class RosImageCamera:
    """Subscribes to a color image topic and returns the latest frame as RGB."""

    def __init__(
        self,
        topic: str,
        max_age_s: float = 1.0,
        connect_timeout_s: float = 10.0,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            topic: ``sensor_msgs/Image`` color topic, e.g. ``/camera_f/color/image_raw``.
            max_age_s: ``read()`` reports failure if the newest frame is older than
                this (a dead camera would otherwise return a frozen frame forever).
            connect_timeout_s: how long to wait for the first frame on construction.
        """
        self.topic = topic
        self.max_age_s = float(max_age_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.verbose = bool(verbose)

        self._ros = _require_ros()
        rospy, Image = self._ros
        # One shared node per process; guarded so it composes with PiperInterface.
        if not rospy.core.is_initialized():
            rospy.init_node("showharness_piper", anonymous=True, disable_signals=True)

        self._latest: Optional[tuple[np.ndarray, float]] = None
        self._lock = threading.Lock()
        self._sub = rospy.Subscriber(topic, Image, self._on_image, queue_size=1, tcp_nodelay=True)

        deadline = time.monotonic() + self.connect_timeout_s
        while self._latest is None:
            if rospy.is_shutdown():
                raise RuntimeError("ROS is shutting down")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"No image on {topic} after {self.connect_timeout_s:.0f}s. Is the "
                    "camera driver running? Start it with: "
                    "roslaunch astra_camera multi_camera.launch (camera_ws sourced)."
                )
            time.sleep(0.02)
        if self.verbose:
            h, w = self._latest[0].shape[:2]
            print(f"[piper-cam] {topic} streaming ({w}x{h} RGB)")

    def _on_image(self, msg: Any) -> None:
        try:
            frame = self._decode(msg)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the callback
            if self.verbose:
                print(f"[piper-cam] {self.topic}: decode error: {exc}")
            return
        with self._lock:
            self._latest = (frame, time.monotonic())

    @staticmethod
    def _decode(msg: Any) -> np.ndarray:
        """sensor_msgs/Image -> HxWx3 uint8 RGB (manual, no cv_bridge)."""
        enc = (msg.encoding or "").lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        h, w, step = int(msg.height), int(msg.width), int(msg.step)
        if enc in ("rgb8", "bgr8"):
            arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[:, :, ::-1]
            return np.ascontiguousarray(arr)
        if enc in ("rgba8", "bgra8"):
            arr = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
            if enc == "bgra8":
                arr = arr[:, :, ::-1]
            return np.ascontiguousarray(arr)
        if enc == "mono8":
            gray = buf.reshape(h, step)[:, :w].reshape(h, w, 1)
            return np.ascontiguousarray(np.repeat(gray, 3, axis=2))
        raise ValueError(
            f"Unsupported image encoding {msg.encoding!r} on {getattr(msg, '_topic', 'topic')}; "
            "expected rgb8/bgr8/rgba8/bgra8/mono8."
        )

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(success, rgb_frame, None)``; success=False if no fresh frame."""
        with self._lock:
            cached = self._latest
        if cached is None:
            return False, None, None
        frame, stamp = cached
        if time.monotonic() - stamp > self.max_age_s:
            return False, None, None  # stale -> caller decides (raise / reuse last)
        return True, frame.copy(), None

    def release(self) -> None:
        try:
            self._sub.unregister()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
