"""OpenCV cameras for the real SO-101: a background thread keeps the newest frame, so a read
never waits on the device. Frames are RGB uint8 640x480, the Workshop camera size.

``open_cameras(front, wrist)`` takes device indices (or paths). With ``mock=True`` the two
arguments are video files that play once and then hold their last frame, so the whole loop
can run without hardware on recorded or simulated footage.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

IMAGE_W, IMAGE_H = 640, 480


class Grabber:
    def __init__(self, source, name: str) -> None:
        import cv2

        src = int(source) if str(source).isdigit() else str(source)
        # Windows' default backend (MSMF) opens slowly and ignores the size request.
        if os.name == "nt" and isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {name!r} ({source}) did not open; is another program using it?")
        if frame.shape[:2] != (IMAGE_H, IMAGE_W):
            raise RuntimeError(f"camera {name!r} gives {frame.shape[1]}x{frame.shape[0]}, expected {IMAGE_W}x{IMAGE_H}")
        self.name, self._cv2 = name, cv2
        self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb

    def read(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def close(self) -> None:
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


class VideoLoop:
    """Plays a video file once, then keeps returning its last frame."""

    def __init__(self, path) -> None:
        import cv2

        cap, self.frames = cv2.VideoCapture(str(path)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not self.frames:
            raise RuntimeError(f"no frames in {path}")
        self.i = 0

    def read(self) -> np.ndarray:
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f

    def close(self) -> None:
        pass


def open_cameras(front, wrist, mock: bool = False) -> dict:
    """{"external": ..., "wrist": ...}; each has read() -> RGB uint8 (480, 640, 3) and close()."""
    if mock:
        return {"external": VideoLoop(front), "wrist": VideoLoop(wrist)}
    return {"external": Grabber(front, "front"), "wrist": Grabber(wrist, "wrist")}
