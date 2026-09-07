"""Utilities for managing RealSense cameras."""
from __future__ import annotations

import time
from typing import Optional
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("WARNING: pyrealsense2 not installed. Install with: pip install pyrealsense2")


def hardware_reset_device(
    serial_number: Optional[str] = None,
    settle_s: float = 5.0,
    timeout_s: float = 20.0,
) -> bool:
    """Power-cycle a RealSense over USB so it re-enumerates cleanly.

    A camera left in a bad state by a previous run is the usual reason a manual
    unplug/replug is needed before it will stream again. ``rs.device.hardware_reset()``
    performs that power-cycle in software; we then wait for the device to disappear
    and come back. Returns True if the (matching) device was reset.

    Args:
        serial_number: Reset only this serial, or the first device if ``None``.
        settle_s: Initial wait after issuing the reset (the device drops off USB).
        timeout_s: Max time to wait for the serial to re-enumerate.
    """
    if not REALSENSE_AVAILABLE:
        return False
    try:
        target = None
        for dev in rs.context().query_devices():
            if serial_number is None or (
                dev.get_info(rs.camera_info.serial_number) == serial_number
            ):
                target = dev
                break
        if target is None:
            print(f"  [reset] device {serial_number or 'default'} not found; skipping reset")
            return False
        print(f"  [reset] hardware_reset on {serial_number or 'default'} (re-enumerating)...")
        target.hardware_reset()
    except Exception as exc:  # noqa: BLE001 - reset is best-effort
        print(f"  [reset] hardware_reset failed: {exc}")
        return False

    time.sleep(max(0.0, settle_s))
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        try:
            serials = {
                d.get_info(rs.camera_info.serial_number)
                for d in rs.context().query_devices()
            }
        except Exception:  # noqa: BLE001
            serials = set()
        if serial_number is None or serial_number in serials:
            print(f"  [reset] device {serial_number or 'default'} is back online")
            return True
        time.sleep(1.0)
    print(f"  [reset] timed out waiting for {serial_number or 'default'} to re-enumerate")
    return True



class RealSenseCamera:
    """Wrapper for Intel RealSense camera with retry logic."""
    
    def __init__(
        self, 
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        reset_on_failure: bool = True,
        read_timeout_ms: int = 3000,
        read_retries: int = 2,
        read_retry_delay: float = 0.05,
        restart_on_read_failure: bool = True,
        warmup_frames: int = 3,
    ):
        """
        Initialize RealSense camera with automatic retry on failure.
        
        Args:
            serial_number: Camera serial number (None for any camera)
            width: Image width
            height: Image height
            fps: Frames per second
            enable_depth: Whether to enable depth stream
            max_retries: Maximum number of initialization attempts
            retry_delay: Delay between retries in seconds
            reset_on_failure: On a failed attempt, power-cycle the camera over USB
                (``hardware_reset``) before retrying, so a stuck device recovers
                without a manual unplug/replug.
            read_timeout_ms: Per-frame wait timeout during normal reads.
            read_retries: Number of wait attempts before a read is considered failed.
            read_retry_delay: Delay between failed read attempts.
            restart_on_read_failure: Restart the RealSense pipeline once before
                returning a failed read. This handles transient librealsense stalls
                without tearing down the whole robot run.
            warmup_frames: Number of frames to wait for after starting/restarting
                the pipeline before it is considered ready.
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 is required. Install with: pip install pyrealsense2")
        
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.pipeline = None
        self.config = None
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        self.read_retries = max(1, int(read_retries))
        self.read_retry_delay = max(0.0, float(read_retry_delay))
        self.restart_on_read_failure = bool(restart_on_read_failure)
        self.warmup_frames = max(0, int(warmup_frames))
        
        # Try to initialize camera with retries
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Starting RealSense camera {serial_number or 'default'} (attempt {attempt}/{max_retries})...")
                self._stop_pipeline(silent=True)
                self._start_pipeline()
                print(f"✓ RealSense camera {serial_number or 'default'} ready!")
                return  # Success!
                
            except Exception as e:
                print(f"✗ Attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    # A stuck device usually needs a power-cycle, not just a wait.
                    # Do it in software so the operator doesn't have to replug.
                    if reset_on_failure:
                        hardware_reset_device(serial_number)
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"\n{'='*60}")
                    print("CAMERA INITIALIZATION FAILED")
                    print(f"{'='*60}")
                    print(f"Camera: {serial_number or 'default'}")
                    print(f"All {max_retries} attempts failed.")
                    print("\nTroubleshooting steps:")
                    print("1. Unplug and replug the camera USB cable")
                    print("2. Try a different USB 3.0 port (preferably directly on motherboard)")
                    print("3. Check USB cable quality (use USB 3.0 cables)")
                    print("4. Reset USB bus: sudo usbreset")
                    print("5. Check camera power: lsusb | grep Intel")
                    print("6. See CAMERA_TROUBLESHOOTING.md for permanent solutions")
                    print(f"{'='*60}\n")
                    raise RuntimeError(f"Failed to initialize RealSense camera after {max_retries} attempts")

    def _start_pipeline(self) -> None:
        """Create and start a fresh librealsense pipeline."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial_number:
            self.config.enable_device(self.serial_number)

        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        self.pipeline.start(self.config)

        # Warm up camera - try to read a few frames.
        if self.warmup_frames:
            print("  Warming up camera...")
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)

    def _stop_pipeline(self, *, silent: bool = False) -> None:
        """Stop the current pipeline if it exists."""
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except Exception as e:
            if not silent:
                print(f"Error stopping camera: {e}")
        finally:
            self.pipeline = None
            self.config = None

    def restart(self) -> None:
        """Restart the camera pipeline without power-cycling the USB device."""
        print(f"Restarting RealSense camera {self.serial_number or 'default'} pipeline...")
        self._stop_pipeline(silent=True)
        time.sleep(0.2)
        self._start_pipeline()

    def _read_once(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.pipeline is None:
            raise RuntimeError("camera pipeline is not started")

        frames = self.pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("no color frame in RealSense frameset")

        color_image = np.asanyarray(color_frame.get_data())

        depth_image = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image
    
    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frame from camera.
        
        Returns:
            (success, color_image, depth_image)
            depth_image is None if depth is not enabled
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.read_retries + 1):
            try:
                color_image, depth_image = self._read_once()
                return True, color_image, depth_image
            except Exception as e:
                last_error = e
                print(
                    f"Error reading from camera {self.serial_number or 'default'} "
                    f"(attempt {attempt}/{self.read_retries}): {e}"
                )
                if attempt < self.read_retries and self.read_retry_delay:
                    time.sleep(self.read_retry_delay)

        if self.restart_on_read_failure:
            try:
                self.restart()
                color_image, depth_image = self._read_once()
                print(f"Recovered RealSense camera {self.serial_number or 'default'} after pipeline restart.")
                return True, color_image, depth_image
            except Exception as e:
                last_error = e
                print(f"Error reading after camera restart {self.serial_number or 'default'}: {e}")

        print(f"Failed to read from camera {self.serial_number or 'default'}: {last_error}")
        return False, None, None
    
    def release(self):
        """Stop the camera pipeline."""
        self._stop_pipeline()
        print(f"RealSense camera {self.serial_number or 'default'} stopped.")


class MockCamera:
    """Mock camera for testing without real hardware."""
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        **kwargs
    ):
        self.width = width
        self.height = height
        print(f"Using mock camera ({width}x{height})")
    
    def read(self) -> tuple[bool, np.ndarray, None]:
        """Generate random test image."""
        # Create a colorful test pattern
        img = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        time.sleep(0.03)  # Simulate 30 FPS
        return True, img, None
    
    def release(self):
        """No-op for mock camera."""
        print("Mock camera released.")


def list_realsense_devices() -> list[dict]:
    """
    List all connected RealSense devices.
    
    Returns:
        List of dicts with keys: 'serial_number', 'name', 'firmware_version'
    """
    if not REALSENSE_AVAILABLE:
        print("pyrealsense2 not available")
        return []
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    device_list = []
    for dev in devices:
        info = {
            'serial_number': dev.get_info(rs.camera_info.serial_number),
            'name': dev.get_info(rs.camera_info.name),
            'firmware_version': dev.get_info(rs.camera_info.firmware_version),
        }
        device_list.append(info)
    
    return device_list


def resize_with_pad(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image with padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


if __name__ == "__main__":
    # Test script
    print("Listing RealSense devices:")
    devices = list_realsense_devices()
    
    if not devices:
        print("No RealSense devices found!")
    else:
        for i, dev in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Serial: {dev['serial_number']}")
            print(f"  Name: {dev['name']}")
            print(f"  Firmware: {dev['firmware_version']}")
