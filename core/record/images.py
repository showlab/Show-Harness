from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def rotate_and_flip(image: np.ndarray, degrees: int = 0, flip: str = "none") -> np.ndarray:
    """Rotate (0/90/180/270 CCW) then flip an HWC image.

    ``flip`` (case-insensitive): ``none`` | ``vertical`` (flipud) | ``horizontal``
    (fliplr) | ``both``. This is the camera transform CONTRACT shared by deployment and
    data generation: the runner applies it to every frame it sends
    (``core.sim.mvtoken_maniskill_runner``), and the real2sim generators apply the same one to
    every frame they store (``scripts/trajectory/real2sim/backends/``). Training images and
    inference images must be byte-identical, so both sides call this function rather than
    each rolling their own flip -- a second implementation could drift silently.

    E.g. the ManiSkill ``hand_camera`` comes out with the gripper fingertips at the BOTTOM
    and its image-right mirrored relative to the agentview, whereas the MVTOKEN training
    wrist has fingertips at the TOP with left/right matching -- that is ``flip="both"``.
    """
    img = to_uint8_hwc(image)
    k = (int(degrees) % 360) // 90
    if k:
        img = np.rot90(img, k=k)
    mode = (flip or "none").lower()
    if mode in ("vertical", "both"):
        img = img[::-1]
    if mode in ("horizontal", "both"):
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def center_crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    """Centre-crop an HWC image to ``aspect`` (= width / height), keeping the larger side.

    Part of the same camera transform CONTRACT as :func:`rotate_and_flip`: deployment and
    data generation must apply it identically or the training images stop matching the
    inference images. Applied BEFORE the letterbox, so the padding lands on the cropped
    geometry.

    The motivating case is RoboLab, whose cameras render 16:9 (1280x720) while the real
    Franka/Piper rigs -- and every MVTOKEN training set built from them -- are 4:3
    (640x480). Letterboxing 16:9 straight into a 256 square leaves 256x144 of picture with
    56 black rows top and bottom, i.e. the manipulated scene occupies noticeably less of
    the frame than in training. Cropping to 4:3 first (``aspect=1.3333``) throws away the
    left/right margin instead and reproduces the training framing. ``aspect <= 0`` or a
    value the image already matches is a no-op.
    """
    img = to_uint8_hwc(image)
    if aspect is None or float(aspect) <= 0:
        return img
    aspect = float(aspect)
    h, w = img.shape[:2]
    current = w / h
    if abs(current - aspect) < 1e-6:
        return img
    if current > aspect:  # too wide -> trim columns
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        img = img[:, x0 : x0 + new_w]
    else:  # too tall -> trim rows
        new_h = int(round(w / aspect))
        y0 = (h - new_h) // 2
        img = img[y0 : y0 + new_h, :]
    return np.ascontiguousarray(img)


def prepare_view(
    image: np.ndarray,
    rotation_degrees: int = 0,
    flip: str = "none",
    crop_aspect: Optional[float] = None,
    square_size: Optional[int] = None,
) -> np.ndarray:
    """THE camera transform contract, in one place: rotate/flip -> crop -> letterbox.

    Every view the policy ever sees goes through this, and **agentview and wrist go
    through the SAME function with the same argument shape** -- only the values differ.
    They used to be handled differently (agentview letterboxed, wrist not), which was an
    accident of the ManiSkill cameras happening to be 640x480 and 256x256: the wrist
    needed no letterbox, so the step was simply omitted for it. That left two code paths
    for what is conceptually one operation, and a camera whose resolution changes (a new
    sim, a re-configured sensor) silently gets the wrong treatment.

    Order matters and is part of the contract:

    1. ``rotation_degrees`` / ``flip`` -- fix how the sensor is mounted, so everything
       downstream sees the orientation the policy was trained on.
    2. ``crop_aspect`` -- centre-crop to width/height (e.g. 4/3) so a 16:9 sensor frames
       the scene the way the 4:3 training rigs did, instead of being letterboxed into a
       thin strip.
    3. ``square_size`` -- letterbox into an N x N square via the REAL rigs' own
       ``resize_with_pad``, LAST, so the black bars land on the final geometry.

    Each stage is skipped when its argument is falsy, so a camera that already matches
    the contract passes through untouched.
    """
    view = rotate_and_flip(image, rotation_degrees, flip)
    if crop_aspect:
        view = center_crop_to_aspect(view, crop_aspect)
    if square_size:
        # Deliberately THE function the real rigs capture with (the franka/piper sessions
        # call it via camera_utils), not a copy: it is a training contract -- a 640x480
        # frame becomes 256x192 centred in 256x256 with 32 black rows top and bottom, and
        # the ms_0717 sets were converted offline with this same function. A second
        # implementation could drift from it silently. (camera_utils only warns when
        # pyrealsense2 is absent; importing it inside a sim env is fine.)
        from core.franka.camera_utils import resize_with_pad  # noqa: PLC0415

        view = resize_with_pad(to_uint8_hwc(view), int(square_size), int(square_size))
    return np.ascontiguousarray(to_uint8_hwc(view))


def image_to_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(to_uint8_hwc(image)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def frame_fingerprint(frame) -> dict:
    """Identify ONE frame by its pixels, so a log can prove which saved PNG it was.

    The hash is over the canonical uint8 HWC bytes -- exactly what ``save_png`` writes -- so
    re-reading ``images/<camera>/<step>.png`` and hashing it the same way reproduces this
    digest bit for bit. Hashing the data URL instead would not: encoding differs per call site.
    """
    array = to_uint8_hwc(frame)
    return {
        "sha1": hashlib.sha1(array.tobytes()).hexdigest()[:12],
        "shape": list(array.shape),
    }


def image_manifest(cameras: Sequence[str], images: Sequence) -> list[dict]:
    """Media manifest for an image request: one entry per ``<image>`` slot, in WIRE ORDER.

    ``cameras`` names the views positionally in the order the request sends them -- a
    TRAINING CONTRACT, since the converter emitted the ``<image>`` slots in that same order.
    ``None`` images are skipped and do NOT consume a slot, matching how ``_message_content``
    builds the parts, so ``slot`` is always the real wire position.
    """
    manifest: list[dict] = []
    for camera, image in zip(cameras, images):
        if image is None:
            continue
        manifest.append(
            {
                "slot": len(manifest),
                "part_type": "image_url",
                "placeholder": "<image>",
                "camera": camera,
                "t_offset": 0,
                **frame_fingerprint(image),
            }
        )
    return manifest


def save_png(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_hwc(image)).save(path)


def save_mp4(path: str | Path, images: list[np.ndarray], fps: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not images:
        return
    imageio.mimwrite(path, [to_uint8_hwc(img) for img in images], fps=fps)


class StreamingVideoWriter:
    """Append-as-you-go MP4 writer whose file survives a killed process.

    Frames stream straight to disk as a FRAGMENTED MP4 (``-movflags
    frag_keyframe+empty_moov``, intra-only zero-latency x264, one flushed fragment
    per frame): every fragment is self-contained, so a run that is force-killed
    (SIGKILL, double Ctrl+C mid-encode, power loss) still leaves a playable file
    with every logged frame -- no lost moov atom, no
    scripts/trajectory/rebuild_video.py pass. ``close(final_path)`` finalizes the
    stream and renames it to the final name; on a crash the fragmented
    ``live_path`` simply remains, already watchable. (Deliberately NO faststart
    remux at close: ffmpeg 4.2's stream-copy from fragmented mp4 drops the final
    fragment's frame.)
    """

    def __init__(self, live_path: str | Path, fps: float) -> None:
        self.live_path = Path(live_path)
        self.fps = float(fps)
        self._writer = None
        self._count = 0

    @property
    def frame_count(self) -> int:
        return self._count

    def append(self, frame: np.ndarray) -> None:
        frame = to_uint8_hwc(frame)
        if self._writer is None:
            self.live_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self.live_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                # Intra-only + zerolatency: no encoder lookahead/GOP buffering, so
                # EVERY appended frame is flushed to disk as its own fragment --
                # the crash-safety guarantee. (Analysis videos are a few hundred
                # frames at ~2 fps; the intra-only size cost is irrelevant next to
                # the per-step PNGs already saved.)
                output_params=[
                    "-g", "1",
                    "-tune", "zerolatency",
                    "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                ],
            )
        self._writer.append_data(frame)
        self._count += 1

    def close(self, final_path: Optional[str | Path] = None) -> Optional[Path]:
        """Finalize the stream and (with ``final_path``) rename the video to its
        final name. Returns the written file's path, or None when no frame was
        ever appended."""
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        if final_path is None:
            return self.live_path
        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        self.live_path.replace(final)
        return final
