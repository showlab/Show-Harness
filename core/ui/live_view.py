"""Live side-by-side camera viewer for real-robot rollouts.

Shows the three camera views while a rollout runs. OpenCV in this environment is
a headless build (``cv2.imshow`` is unavailable), so the window is drawn with
pygame.

Two modes:

* **Streaming**: :meth:`start_stream` spawns a render thread that polls a frame
  source (e.g. ``DualPiperSession.get_camera_frames`` or
  ``FrankaSession.get_camera_frames``) at ~12 Hz and redraws continuously -- the
  feed stays LIVE while the VLM thinks and while the arms move. The runner only
  posts status (STAGE / tokens / reasoning / telemetry) via :meth:`show_dual`
  (three views) or :meth:`show_single` (front + one wrist). The layout is chosen
  from the frame keys, so one stream thread serves both rigs. All pygame calls
  happen inside that one thread.
* **Immediate** (legacy single-arm ``show``, or ``show_dual``/``show_single``
  without a stream): the caller's thread renders one frame per call, as before.

The viewer is **fail-safe**: if pygame or a display is unavailable (headless CI,
``SDL_VIDEODRIVER=dummy``, a mock dry run, etc.) it silently disables itself, so
the rollout is never blocked by the absence of a screen. Pass an instance to the
runner (``viewer=...``); ``None`` skips it.
"""
from __future__ import annotations

import os
import textwrap
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

# One palette with the analysis video, so the live window and the saved rollout
# read as the same interface.
from core.record.episode_logger import (
    ACC_LEFT,
    ACC_NEUTRAL,
    ACC_RIGHT,
    BG_HEADER,
    BG_LABEL,
    BG_PANEL,
    BG_REASON,
    FAINT,
    FG,
    MUT,
    RULE_C,
)

_TILE = 352      # each camera view, upscaled from the 256 px observation
_GAP = 10
_MARGIN = 12
_HEADER_H = 58
_LABEL_H = 24
_BAND_H = 56     # per-view status band (each arm directly under its own camera)
_WHY_H = 52      # persistent reasoning strip

# pygame's bundled font has no ✓/✕/⚠/↺ glyphs (they render as boxes); the
# PIL-drawn analysis video keeps them.
_GLYPH_ASCII = (("✓", "+"), ("✕", "x"), ("⚠", "!"), ("↺", "~"))


class LiveView:
    """A best-effort pygame window showing the rollout's camera views + status."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        title: str = "Show-Harness rollout",
        always_on_top: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._ok = self.enabled
        self._initialized = False
        self._pygame: Any = None
        self._screen: Any = None
        self._font: Any = None
        self._font_small: Any = None
        self._font_big: Any = None
        self._font_bold: Any = None
        self._size: Optional[tuple[int, int]] = None
        self._caption_h = 26
        self._title = title
        self._always_on_top = bool(always_on_top)
        self._topmost_warning_shown = False
        # Shared state between the runner (writer) and the render thread (reader).
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {}
        self._frames: Optional[dict[str, np.ndarray]] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Optional KEYDOWN sink (pygame key code -> None). The stream thread pumps
        # events continuously (~12 Hz) even while the VLM thinks, so a handler set
        # here (e.g. plugins.dagger) sees keys in near-real time. Exceptions in the
        # handler are swallowed: a consumer bug must never kill the live feed.
        self.key_handler: Optional[Callable[[int], None]] = None
        # Keyboard-focus tracking (only meaningful with a key_handler): the OS sends
        # keys to the FOCUSED window, so an unfocused live view silently captures
        # nothing -- observed on hardware: DAGGER keys echoed uselessly into the
        # terminal. The window banners the armed/inactive state and the transition
        # is announced in the terminal (once per change, never spammed).
        self._key_focus: Optional[bool] = None

    @property
    def active(self) -> bool:
        return self._ok

    @property
    def streaming(self) -> bool:
        return self._stream_thread is not None and self._stream_thread.is_alive()

    # -- lifecycle ---------------------------------------------------------------
    def _ensure_ui(self) -> bool:
        """Initialize pygame lazily, in whichever thread renders. Streaming mode
        starts the stream before any immediate render, so exactly one thread ever
        touches pygame."""
        if self._initialized:
            return self._ok
        self._initialized = True
        if not self.enabled:
            self._ok = False
            return False
        try:
            # Audio init can hang on some boxes; force a dummy audio driver and
            # only init the display + font subsystems (never pygame.init()).
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            import pygame  # noqa: PLC0415 - imported lazily so it is an optional dep

            pygame.display.init()
            pygame.font.init()
            self._pygame = pygame
            self._font = pygame.font.Font(None, 22)  # bundled font, no system scan
            self._font_small = pygame.font.Font(None, 19)
            self._font_big = pygame.font.Font(None, 30)
            self._font_big.set_bold(True)
            self._font_bold = pygame.font.Font(None, 22)
            self._font_bold.set_bold(True)
            self._ok = True
        except Exception as exc:  # noqa: BLE001 - any failure -> disable, never crash
            print(f"[live-view] disabled (no display: {exc})")
            self._ok = False
        return self._ok

    def _create_window(self, size: tuple[int, int]) -> None:
        """Create/resize the pygame window and apply its persistent window state."""
        self._screen = self._pygame.display.set_mode(size)
        self._pygame.display.set_caption(self._title)
        self._size = size
        if (
            self._always_on_top
            and not _set_pygame_window_always_on_top(self._pygame)
            and not self._topmost_warning_shown
        ):
            print("[live-view] warning: the window manager does not support always-on-top")
            self._topmost_warning_shown = True

    def start_stream(
        self,
        frame_source: Callable[[], Optional[dict[str, np.ndarray]]],
        fps: float = 12.0,
    ) -> None:
        """Start the continuous live feed: a daemon thread polls ``frame_source``
        (returning {"agentview","wrist_left","wrist_right"} or None to keep the
        last frames) and redraws at ~``fps``. Idempotent; no-op when disabled."""
        if not self.enabled or self.streaming:
            return
        self._stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(frame_source, max(1.0, float(fps))),
            name="live-view",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_loop(
        self, frame_source: Callable[[], Optional[dict[str, np.ndarray]]], fps: float
    ) -> None:
        period = 1.0 / fps
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._ok:
                return
            frames = None
            try:
                frames = frame_source()
            except Exception:  # noqa: BLE001 - a flaky source must not kill the feed
                frames = None
            with self._lock:
                if frames is not None:
                    self._frames = frames
                frames_now = self._frames
                status_now = dict(self._status)
            if frames_now is not None and self._ensure_ui():
                self._render(frames_now, status_now)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        thread = self._stream_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._stream_thread = None
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._ok = False

    # -- dual-arm dashboard --------------------------------------------------------
    def show_dual(
        self,
        *,
        step: int,
        agentview: Optional[np.ndarray] = None,
        wrist_left: Optional[np.ndarray] = None,
        wrist_right: Optional[np.ndarray] = None,
        arms: dict[str, dict[str, Any]],
        task: str = "",
        phase: str = "",
        reason: str = "",
        telemetry: str = "",
    ) -> None:
        """Post the runner's status (and, without a stream, render immediately).

        With :meth:`start_stream` running, this only updates the status overlay --
        the stream thread supplies fresh camera frames continuously, which is what
        makes the feed real-time rather than one snapshot per action.

        Fields a status update OMITS (the reasoning, an arm's token/flags) PERSIST
        from the previous update instead of blanking: the last decision stays on
        screen through the whole next thinking/motion phase, tagged with the step
        it came from."""
        if not self._ok:
            return
        status = {
            "step": int(step),
            "arms": {side: dict(info or {}) for side, info in (arms or {}).items()},
            "task": task,
            "phase": phase,
            "reason": reason,
            "telemetry": telemetry,
        }
        frames = None
        if agentview is not None and wrist_left is not None and wrist_right is not None:
            frames = {
                "agentview": agentview,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        self._post_status(status, frames)

    def show_single(
        self,
        *,
        step: int,
        agentview: Optional[np.ndarray] = None,
        wrist: Optional[np.ndarray] = None,
        arm: Optional[dict[str, Any]] = None,
        task: str = "",
        phase: str = "",
        reason: str = "",
        telemetry: str = "",
    ) -> None:
        """Single-arm dashboard: :meth:`show_dual`'s contract with two views.

        Post the runner's status (and, without a stream, render immediately). The
        arm's status payload rides in the reserved ``"arm"`` slot; persistence of
        omitted fields (reasoning, token/flags) matches the dual path."""
        if not self._ok:
            return
        status = {
            "step": int(step),
            "arms": {"arm": dict(arm or {})},
            "task": task,
            "phase": phase,
            "reason": reason,
            "telemetry": telemetry,
        }
        frames = None
        if agentview is not None:
            frames = {"agentview": agentview}
            if wrist is not None:
                frames["wrist"] = wrist
        self._post_status(status, frames)

    def _post_status(
        self, status: dict[str, Any], frames: Optional[dict[str, np.ndarray]]
    ) -> None:
        """Store the status (carrying over omitted fields), then render when the
        caller owns rendering (no stream thread)."""
        with self._lock:
            prev = self._status
            if status["reason"]:
                status["reason_step"] = status["step"]
            elif prev.get("reason"):
                status["reason"] = prev["reason"]
                status["reason_step"] = prev.get("reason_step")
            prev_arms = prev.get("arms") or {}
            for side, info in status["arms"].items():
                carry = prev_arms.get(side) or {}
                if "token" not in info and "token" in carry:
                    info["token"] = carry["token"]
                    if "flags" not in info and "flags" in carry:
                        info["flags"] = carry["flags"]
            self._status = status
            if frames is not None and not self.streaming:
                self._frames = frames
        if self.streaming:
            return
        if frames is not None and self._ensure_ui():
            self._render(frames, status)

    def _render(self, frames: dict[str, np.ndarray], status: dict[str, Any]) -> None:
        """Render whichever dashboard the frames describe (three views -> dual,
        front [+ one wrist] -> single). One dispatch point keeps the stream thread
        rig-agnostic."""
        if "wrist_left" in frames or "wrist_right" in frames:
            self._render_dual(frames, status)
        else:
            self._render_single(frames, status)

    def _render_single(self, frames: dict[str, np.ndarray], status: dict[str, Any]) -> None:
        try:
            pygame = self._pygame
            width = _MARGIN * 2 + _TILE * 2 + _GAP
            height = _HEADER_H + _LABEL_H + _TILE + _BAND_H + _WHY_H
            if self._screen is None or self._size != (width, height):
                self._create_window((width, height))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._ok = False
                    return
                if event.type == pygame.KEYDOWN and self.key_handler is not None:
                    try:
                        self.key_handler(event.key)
                    except Exception:  # noqa: BLE001 - handler bug must not kill the feed
                        pass
            if self.key_handler is not None:
                self._track_key_focus(bool(pygame.key.get_focused()))
            screen = self._screen
            screen.fill(BG_PANEL)

            # Header: task (left), STEP counter (right).
            pygame.draw.rect(screen, BG_HEADER, (0, 0, width, _HEADER_H))
            self._text(f"TASK  {_ellipsis(status.get('task', ''), 58)}", _MARGIN, 17, FG)
            step_label = self._font_big.render(
                f"STEP {int(status.get('step', 0)):03d}", True, FG
            )
            screen.blit(step_label, (width - _MARGIN - step_label.get_width(), 15))
            pygame.draw.line(screen, RULE_C, (0, _HEADER_H - 1), (width, _HEADER_H - 1))

            # Keyboard armed/inactive strip (DAGGER): mirror of the dual dashboard.
            if self.key_handler is not None:
                armed = bool(self._key_focus)
                strip_color = (39, 105, 62) if armed else (190, 42, 42)
                strip_text = (
                    "KEYS ARMED -- keyboard steers the arm"
                    if armed
                    else "KEYS INACTIVE -- CLICK THIS WINDOW TO STEER"
                )
                pygame.draw.rect(screen, strip_color, (0, 0, width, 15))
                strip = self._font_small.render(strip_text, True, (255, 255, 255))
                screen.blit(strip, ((width - strip.get_width()) // 2, 1))

            # Labeled camera tiles: the scene view and the arm's wrist view.
            tiles = (
                ("AGENT VIEW", ACC_NEUTRAL, frames.get("agentview")),
                ("WRIST", ACC_LEFT, frames.get("wrist")),
            )
            tile_y = _HEADER_H + _LABEL_H
            pygame.draw.rect(screen, BG_LABEL, (0, _HEADER_H, width, _LABEL_H))
            for idx, (label, accent, frame) in enumerate(tiles):
                x = _MARGIN + idx * (_TILE + _GAP)
                self._text(label, x + 4, _HEADER_H + 4, MUT, small=True)
                pygame.draw.rect(screen, accent, (x, _HEADER_H + _LABEL_H - 3, _TILE, 2))
                surface = self._tile_surface(frame)
                if surface is not None:
                    screen.blit(surface, (x, tile_y))

            # Status band: the arm's STAGE + action under the front view; run phase
            # + telemetry under the wrist view.
            band_y = tile_y + _TILE
            info = (status.get("arms") or {}).get("arm") or {}
            pygame.draw.rect(screen, BG_HEADER, (0, band_y, width, _BAND_H))
            x0 = _MARGIN
            x = self._text("ARM   ", x0 + 4, band_y + 7, ACC_LEFT, bold=True)
            self._text(str(info.get("stage", "-")), x, band_y + 7, FG)
            x = x0 + 4
            token = info.get("token")
            if token:
                x = self._text(f"{token:<9}", x, band_y + 31, FG, bold=True)
            x = self._text(f"grip {info.get('grip', '-')}", x, band_y + 31, MUT)
            for text, color in info.get("flags") or []:
                for glyph, ascii_ in _GLYPH_ASCII:
                    text = text.replace(glyph, ascii_)
                x = self._text(f"  {text}", x, band_y + 31, color)
            x1 = _MARGIN + _TILE + _GAP
            self._text(str(status.get("phase", "")), x1 + 4, band_y + 7, FAINT)
            self._text(
                str(status.get("telemetry", "")), x1 + 4, band_y + 31, MUT, small=True
            )

            # Persistent reasoning strip, tagged with the step the decision came from.
            why_y = band_y + _BAND_H
            pygame.draw.rect(screen, BG_REASON, (0, why_y, width, _WHY_H))
            pygame.draw.line(screen, RULE_C, (0, why_y), (width, why_y))
            reason = status.get("reason", "")
            if reason:
                reason_step = status.get("reason_step")
                label = (
                    f"WHY · STEP {int(reason_step):03d}"
                    if reason_step is not None
                    else "WHY"
                )
                self._text(label, _MARGIN, why_y + 8, FAINT, small=True)
                text = " ".join(str(reason).split())
                lines = textwrap.wrap(text, width=66)[:2]
                if len(lines) == 2 and len(text) > 66 * 2:
                    lines[1] = lines[1][:63] + "..."
                for i, line in enumerate(lines):
                    self._text(line, _MARGIN + 118, why_y + 6 + i * 20, MUT, small=True)
            pygame.display.flip()
        except Exception as exc:  # noqa: BLE001 - disable on first render error
            print(f"[live-view] disabled (render error: {exc})")
            self._ok = False

    def _render_dual(self, frames: dict[str, np.ndarray], status: dict[str, Any]) -> None:
        try:
            pygame = self._pygame
            width = _MARGIN * 2 + _TILE * 3 + _GAP * 2
            height = _HEADER_H + _LABEL_H + _TILE + _BAND_H + _WHY_H
            if self._screen is None or self._size != (width, height):
                self._create_window((width, height))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._ok = False
                    return
                if event.type == pygame.KEYDOWN and self.key_handler is not None:
                    try:
                        self.key_handler(event.key)
                    except Exception:  # noqa: BLE001 - handler bug must not kill the feed
                        pass
            if self.key_handler is not None:
                self._track_key_focus(bool(pygame.key.get_focused()))
            screen = self._screen
            screen.fill(BG_PANEL)

            # Header: task (left), STEP counter (right). Phase + telemetry live in
            # the center status band, under the shared front view.
            pygame.draw.rect(screen, BG_HEADER, (0, 0, width, _HEADER_H))
            self._text(f"TASK  {_ellipsis(status.get('task', ''), 92)}", _MARGIN, 17, FG)
            step_label = self._font_big.render(
                f"STEP {int(status.get('step', 0)):03d}", True, FG
            )
            screen.blit(step_label, (width - _MARGIN - step_label.get_width(), 15))
            pygame.draw.line(screen, RULE_C, (0, _HEADER_H - 1), (width, _HEADER_H - 1))

            # Keyboard armed/inactive strip: with a key consumer mounted (DAGGER),
            # the operator must SEE whether this window will receive their keys.
            if self.key_handler is not None:
                armed = bool(self._key_focus)
                strip_color = (39, 105, 62) if armed else (190, 42, 42)
                strip_text = (
                    "KEYS ARMED -- keyboard steers the arms"
                    if armed
                    else "KEYS INACTIVE -- CLICK THIS WINDOW TO STEER"
                )
                pygame.draw.rect(screen, strip_color, (0, 0, width, 15))
                strip = self._font_small.render(strip_text, True, (255, 255, 255))
                screen.blit(strip, ((width - strip.get_width()) // 2, 1))

            # Labeled camera tiles, Left - Agent - Right: each wrist sits on its own
            # arm's side of the shared front view, mirroring the physical rig.
            tiles = (
                ("WRIST · LEFT", ACC_LEFT, frames.get("wrist_left")),
                ("AGENT VIEW", ACC_NEUTRAL, frames.get("agentview")),
                ("WRIST · RIGHT", ACC_RIGHT, frames.get("wrist_right")),
            )
            tile_y = _HEADER_H + _LABEL_H
            pygame.draw.rect(screen, BG_LABEL, (0, _HEADER_H, width, _LABEL_H))
            for idx, (label, accent, frame) in enumerate(tiles):
                x = _MARGIN + idx * (_TILE + _GAP)
                self._text(label, x + 4, _HEADER_H + 4, MUT, small=True)
                pygame.draw.rect(screen, accent, (x, _HEADER_H + _LABEL_H - 3, _TILE, 2))
                surface = self._tile_surface(frame)
                if surface is not None:
                    screen.blit(surface, (x, tile_y))

            # Status bands: each arm's STAGE + action DIRECTLY UNDER its own view;
            # the center band (under the shared front view) carries phase + telemetry.
            band_y = tile_y + _TILE
            arms = status.get("arms") or {}
            pygame.draw.rect(screen, BG_HEADER, (0, band_y, width, _BAND_H))
            for idx, (side, accent) in enumerate(
                (("left", ACC_LEFT), (None, None), ("right", ACC_RIGHT))
            ):
                x0 = _MARGIN + idx * (_TILE + _GAP)
                if side is None:  # center band: run phase + telemetry
                    self._text(str(status.get("phase", "")), x0 + 4, band_y + 7, FAINT)
                    self._text(
                        str(status.get("telemetry", "")), x0 + 4, band_y + 31, MUT,
                        small=True,
                    )
                    continue
                info = arms.get(side) or {}
                x = self._text(f"{side.upper():<6}", x0 + 4, band_y + 7, accent, bold=True)
                self._text(str(info.get("stage", "-")), x, band_y + 7, FG)
                x = x0 + 4
                token = info.get("token")
                if token:
                    x = self._text(f"{token:<9}", x, band_y + 31, FG, bold=True)
                x = self._text(f"grip {info.get('grip', '-')}", x, band_y + 31, MUT)
                for text, color in info.get("flags") or []:
                    for glyph, ascii_ in _GLYPH_ASCII:
                        text = text.replace(glyph, ascii_)
                    x = self._text(f"  {text}", x, band_y + 31, color)

            # Persistent reasoning strip: the LAST decision stays visible through the
            # whole next thinking/motion phase, tagged with the step it came from.
            why_y = band_y + _BAND_H
            pygame.draw.rect(screen, BG_REASON, (0, why_y, width, _WHY_H))
            pygame.draw.line(screen, RULE_C, (0, why_y), (width, why_y))
            reason = status.get("reason", "")
            if reason:
                reason_step = status.get("reason_step")
                label = (
                    f"WHY · STEP {int(reason_step):03d}"
                    if reason_step is not None
                    else "WHY"
                )
                self._text(label, _MARGIN, why_y + 8, FAINT, small=True)
                text = " ".join(str(reason).split())
                lines = textwrap.wrap(text, width=104)[:2]
                if len(lines) == 2 and len(text) > 104 * 2:
                    lines[1] = lines[1][:101] + "..."
                for i, line in enumerate(lines):
                    self._text(line, _MARGIN + 118, why_y + 6 + i * 20, MUT, small=True)
            pygame.display.flip()
        except Exception as exc:  # noqa: BLE001 - disable on first render error
            print(f"[live-view] disabled (render error: {exc})")
            self._ok = False

    def _track_key_focus(self, focused: bool) -> None:
        """Announce keyboard-focus transitions (once per change) in the terminal."""
        if focused == self._key_focus:
            return
        self._key_focus = focused
        # if focused:
        #     print("[live-view] keyboard ARMED -- the window has focus; keys steer the arms")
        # else:
        #     print(
        #         "[live-view] keyboard INACTIVE -- click the live-view window: "
        #         "keys are currently going to another window"
        #     )

    def _tile_surface(self, frame: Optional[np.ndarray]):
        """A TILE x TILE surface: the frame smooth-scaled up, aspect preserved."""
        if frame is None:
            return None
        pygame = self._pygame
        array = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        surface = pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))
        h, w = array.shape[:2]
        scale = min(_TILE / w, _TILE / h)
        scaled = pygame.transform.smoothscale(
            surface, (max(1, round(w * scale)), max(1, round(h * scale)))
        )
        tile = pygame.Surface((_TILE, _TILE))
        tile.fill(BG_PANEL)
        tile.blit(
            scaled,
            ((_TILE - scaled.get_width()) // 2, (_TILE - scaled.get_height()) // 2),
        )
        return tile

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        *,
        bold: bool = False,
        small: bool = False,
    ) -> int:
        """Render one text run; returns the x where the next run should start."""
        font = self._font_bold if bold else (self._font_small if small else self._font)
        surface = font.render(text, True, color)
        self._screen.blit(surface, (x, y))
        return x + surface.get_width()

    # -- legacy single-arm path ----------------------------------------------------
    def show(
        self,
        agentview: Optional[np.ndarray],
        wrist: Optional[np.ndarray] = None,
        caption: str = "",
    ) -> None:
        """Render the latest frames side by side. No-op if the viewer is disabled."""
        if not self._ok or not self._ensure_ui():
            return
        try:
            pygame = self._pygame
            combined = self._compose(agentview, wrist)
            if combined is None:
                return
            h, w = combined.shape[:2]
            win = (w, h + self._caption_h)
            if self._screen is None or self._size != win:
                self._create_window(win)
            # Drain the event queue so the OS keeps the window responsive.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            # pygame surfaces are (w, h, c); our frames are (h, w, c).
            surface = pygame.surfarray.make_surface(np.transpose(combined, (1, 0, 2)))
            self._screen.fill(BG_PANEL)
            self._screen.blit(surface, (0, 0))
            if caption:
                text = self._font.render(caption, True, FG)
                self._screen.blit(text, (8, h + 4))
            pygame.display.flip()
        except Exception as exc:  # noqa: BLE001 - disable on first render error
            print(f"[live-view] disabled (render error: {exc})")
            self._ok = False

    @staticmethod
    def _compose(
        agentview: Optional[np.ndarray], wrist: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        frames = [
            np.ascontiguousarray(np.asarray(f, dtype=np.uint8))
            for f in (agentview, wrist)
            if f is not None
        ]
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        # Pad to a common height so the two views sit side by side.
        height = max(f.shape[0] for f in frames)
        padded = []
        for f in frames:
            if f.shape[0] != height:
                pad = np.zeros((height, f.shape[1], 3), dtype=np.uint8)
                pad[: f.shape[0]] = f
                f = pad
            padded.append(f)
        return np.concatenate(padded, axis=1)


def _ellipsis(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


# pygame 2.5/2.6's _sdl2 Window wrappers OWN the SDL window: letting one be
# garbage-collected destroys the live display window and the next event pump
# segfaults (verified on pygame 2.6.1). Park every wrapper here so none is ever
# collected while the process runs.
_WINDOW_REFS: list[Any] = []


def _set_pygame_window_always_on_top(pygame: Any) -> bool:
    """Ask SDL to keep pygame's current display window above normal windows.

    pygame 2.5/2.6 does not expose ``SDL_SetWindowAlwaysOnTop`` on its display
    API, even though the bundled SDL does. Load the exact SDL library already
    used by pygame (important when pygame ships a private copy), resolve the
    display window by its SDL id, and call the native API. Every step is
    best-effort so a headless or unusual desktop can still run a rollout.
    """
    try:
        import ctypes  # noqa: PLC0415 - only needed when a real window is created
        import ctypes.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        from pygame._sdl2.video import Window  # noqa: PLC0415

        window_obj = Window.from_display_module()
        _WINDOW_REFS.append(window_obj)  # see note above: must never be GC'd
        window_id = int(window_obj.id)
        candidates: list[str] = []

        # Linux wheels commonly bundle a hashed SDL library. Loading the system
        # SDL would create a second SDL instance, which cannot see pygame's
        # window; /proc identifies the already-loaded library precisely.
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/maps", encoding="utf-8") as maps:
                    for line in maps:
                        path = line.rsplit(maxsplit=1)[-1].strip()
                        if "libSDL2" in os.path.basename(path) and path.startswith("/"):
                            candidates.append(path)
            except OSError:
                pass

        found = ctypes.util.find_library("SDL2")
        if found:
            candidates.append(found)
        if sys.platform == "win32":
            candidates.append("SDL2.dll")

        for library_path in dict.fromkeys(candidates):
            try:
                sdl = ctypes.CDLL(library_path)
                get_window = sdl.SDL_GetWindowFromID
                get_window.argtypes = [ctypes.c_uint32]
                get_window.restype = ctypes.c_void_p
                set_topmost = sdl.SDL_SetWindowAlwaysOnTop
                set_topmost.argtypes = [ctypes.c_void_p, ctypes.c_int]
                set_topmost.restype = None
                window = get_window(window_id)
                if window:
                    set_topmost(window, 1)
                    return True
            except (AttributeError, OSError):
                continue
    except (AttributeError, ImportError, TypeError, ValueError):
        pass
    return False
