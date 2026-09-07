"""Shared keyboard-teleoperation + rollout-recording core (hardware-agnostic).

Used by the per-robot teleop scripts (scripts/trajectory/collect_rollouts.py for
the Franka, scripts/trajectory/collect_rollouts_piper.py for the AgileX Piper),
which only wire up their session/controller and hardware-specific homing.

Dataset layout (identical for every robot -- the action space IS the shared
atomic-token vocabulary): each recording is saved under ``<save_path>/rollout_NNN/``:

    agentview/0000.png, 0001.png, ...   external-camera frames (one per step)
    wrist/0000.png, ...                 wrist-camera frames
    actions.jsonl                       per-step token + ee_pose + gripper
    metadata.json                       rollout summary (tokens, counts, config)
    visualization.mp4                   both views + action overlay per step

Records store the state the action was taken FROM: obs_t is captured and written,
THEN the controller executes the token -- (obs_t, a_t) pairs.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    import cv2  # only cv2.putText is used (works with the headless build too)

    _CV2 = True
except ImportError:  # pragma: no cover
    _CV2 = False

from core.action_units import ROTATE_ATOMS
from interpreters.real_atomic_controller import GRASP_ATOM, RELEASE_ATOM
from core.record.images import save_mp4, save_png

# ---------------------------------------------------------------------------
# Frame composition (video / labels) -- pure numpy + cv2.putText, no GUI needed.
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX if _CV2 else None


def _to_square(rgb: np.ndarray, size: int) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    if _CV2:
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    # Fallback: crude nearest-neighbour resize without cv2.
    h, w = rgb.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return rgb[ys][:, xs]


def compose_step_frame(
    agentview: np.ndarray,
    wrist: np.ndarray,
    step: int,
    token: str,
    gripper_closed: Optional[bool],
    size: int = 256,
    bar_h: int = 32,
) -> np.ndarray:
    """Build one RGB video frame: [agentview | wrist] + a labelled status bar."""
    a = _to_square(agentview, size)
    w = _to_square(wrist, size)
    views = np.hstack([a, w])
    bar = np.zeros((bar_h, views.shape[1], 3), dtype=np.uint8)
    canvas = np.vstack([views, bar]).astype(np.uint8)
    if _CV2:
        grip = "CLOSED" if gripper_closed else "OPEN"
        cv2.putText(canvas, "AgentView", (6, 18), _FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Wrist", (size + 6, 18), _FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{step:03d}  {token:<9} gripper={grip}",
            (6, size + bar_h - 11),
            _FONT,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


# ---------------------------------------------------------------------------
# Rollout recorder
# ---------------------------------------------------------------------------
class RolloutRecorder:
    """Owns the on-disk layout for numbered rollouts and the per-step buffers."""

    def __init__(
        self,
        root: str | Path,
        prefix: str = "rollout_",
        video_fps: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.video_fps = float(video_fps)
        self.verbose = verbose
        self.session_meta: dict[str, Any] = {}

        self.active = False
        self.index = -1
        self.dir: Optional[Path] = None
        self.jsonl: Optional[Path] = None
        self.steps: list[dict[str, Any]] = []
        self.video_frames: list[np.ndarray] = []

    def _next_index(self) -> int:
        pattern = re.compile(re.escape(self.prefix) + r"(\d+)$")
        existing = [int(m.group(1)) for p in self.root.glob(self.prefix + "*") if p.is_dir() and (m := pattern.match(p.name))]
        return max(existing) + 1 if existing else 0

    def start(self) -> Path:
        if self.active:
            return self.dir  # type: ignore[return-value]
        self.index = self._next_index()
        width = max(3, len(str(self.index)))
        self.dir = self.root / f"{self.prefix}{self.index:0{width}d}"
        (self.dir / "agentview").mkdir(parents=True, exist_ok=True)
        (self.dir / "wrist").mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "actions.jsonl"
        self.jsonl.write_text("")  # truncate/create
        self.steps = []
        self.video_frames = []
        self.active = True
        if self.verbose:
            print(f"[rec] ● recording -> {self.dir}")
        return self.dir

    def add_step(self, token: str, kind: str, obs: dict[str, Any], gripper_closed: Optional[bool]) -> None:
        if not self.active or self.dir is None:
            return
        step = len(self.steps)
        name = f"{step:04d}.png"
        save_png(self.dir / "agentview" / name, obs["agentview"])
        save_png(self.dir / "wrist" / name, obs["wrist"])
        record = {
            "step": step,
            "token": token,
            "kind": kind,
            "gripper_closed": None if gripper_closed is None else bool(gripper_closed),
            "ee_pose": np.asarray(obs["ee_pose"], dtype=float).round(5).tolist(),
            "gripper_width": round(float(obs["gripper_width"]), 5),
            "agentview": f"agentview/{name}",
            "wrist": f"wrist/{name}",
            "time": round(time.time(), 3),
        }
        self.steps.append(record)
        with self.jsonl.open("a", encoding="utf-8") as f:  # type: ignore[union-attr]
            f.write(json.dumps(record) + "\n")
        self.video_frames.append(compose_step_frame(obs["agentview"], obs["wrist"], step, token, gripper_closed))

    def stop(self) -> Optional[Path]:
        if not self.active or self.dir is None:
            return None
        meta = {
            "rollout": self.dir.name,
            "index": self.index,
            "num_steps": len(self.steps),
            "tokens": [s["token"] for s in self.steps],
            "created": datetime.now().isoformat(timespec="seconds"),
            **self.session_meta,
        }
        (self.dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        if self.video_frames:
            try:
                save_mp4(self.dir / "visualization.mp4", self.video_frames, self.video_fps)
            except Exception as exc:  # noqa: BLE001 - video is best-effort
                print(f"[rec] WARNING: could not write visualization video: {exc}")
        done_dir = self.dir
        if self.verbose:
            print(f"[rec] ■ stopped -> {done_dir} ({len(self.steps)} steps)")
        self.active = False
        self.dir = None
        self.jsonl = None
        self.steps = []
        self.video_frames = []
        return done_dir


# ---------------------------------------------------------------------------
# Teleop collector (pygame window + keyboard)
# ---------------------------------------------------------------------------
class RolloutCollector:
    """Pygame keyboard teleop driving any RealAtomicController-shaped controller.

    ``home_fn``, when provided, is the hardware-specific "return the arm to its
    home configuration and restore its control mode" hook, run after a recording
    stops (P). The collector itself re-syncs the controller setpoint from the
    robot afterwards -- that re-sync is what prevents the first post-home command
    from jumping back toward the stale pre-home setpoint, so it must never be
    left to the individual scripts.
    """

    def __init__(
        self,
        session: Any,
        controller: Any,
        recorder: RolloutRecorder,
        move_interval: float = 0.12,
        display_scale: int = 2,
        target_fps: int = 30,
        home_fn: Optional[Callable[[], None]] = None,
        include_rotate_keys: bool = True,
        window_title: str = "Show-Harness - Rollout Collection",
    ) -> None:
        self.session = session
        self.controller = controller
        self.recorder = recorder
        self.move_interval = float(move_interval)
        self.display_scale = int(display_scale)
        self.target_fps = int(target_fps)
        self.home_fn = home_fn
        self.include_rotate_keys = bool(include_rotate_keys)
        self.window_title = window_title

        self.running = True
        self.latest_obs: Optional[dict[str, Any]] = None
        self._next_gripper = GRASP_ATOM
        self._held_move: Optional[tuple[int, str]] = None
        self._last_move_t = 0.0
        self.last_token = "-"

    # -- robot / data ------------------------------------------------------
    def capture(self) -> dict[str, Any]:
        try:
            self.latest_obs = self.session.get_observation()
        except Exception as exc:  # noqa: BLE001
            if self.latest_obs is None:
                raise
            print(f"[capture] WARNING: {exc}; reusing previous frame")
        return self.latest_obs  # type: ignore[return-value]

    def _record_and_step(self, token: str, kind: str) -> None:
        obs = self.latest_obs if self.latest_obs is not None else self.capture()
        # Record the state the action was taken FROM (obs_t, a_t), then execute.
        self.recorder.add_step(token, kind, obs, self.controller.gripper_closed)
        self.controller.step(token)
        self.last_token = token

    def do_move(self, token: str) -> None:
        self._record_and_step(token, "rotate" if token in ROTATE_ATOMS else "move")

    def do_gripper(self) -> None:
        token = self._next_gripper
        self._record_and_step(token, "gripper")
        # Track the controller's ACTUAL state, not the commanded token: a GRASP that
        # caught nothing auto-reopens (empty-grasp rule), so basing the next toggle on
        # the token would desync (the next Space would send a no-op RELEASE).
        self._next_gripper = RELEASE_ATOM if self.controller.gripper_closed else GRASP_ATOM

    def toggle_recording(self) -> None:
        if self.recorder.active:
            self.recorder.stop()
            # Auto-home the arm so the next demo starts from the same pose. The
            # just-saved rollout is unaffected; the homing motion is not recorded
            # (the recorder is already stopped).
            if self.home_fn is not None:
                self.reset_to_home()
        else:
            self.recorder.start()

    def reset_to_home(self) -> None:
        """Run the hardware homing hook, then resume teleop from a re-synced state.

        The homing motion happens outside this controller's setpoint stream, so the
        setpoint is re-synced to the measured (home) pose afterwards -- otherwise the
        next teleop command would move relative to the stale pre-home setpoint.
        Errors never kill the session.
        """
        if self.home_fn is None:
            return
        try:
            self.home_fn()
            self.controller.sync_from_robot()
            self._next_gripper = RELEASE_ATOM if self.controller.gripper_closed else GRASP_ATOM
            self._held_move = None
            self.capture()  # refresh the live view to the homed pose
            print("[reset] arm homed; ready for the next recording.")
        except Exception as exc:  # noqa: BLE001 - never let a reset error kill the session
            print(f"[reset] ERROR: homing failed: {exc}")

    # -- ui ----------------------------------------------------------------
    def _print_controls(self) -> None:
        rotate_line = "  Z/X          ROTATE_CCW / ROTATE_CW\n" if self.include_rotate_keys else ""
        home_note = " (stop auto-homes the arm)" if self.home_fn is not None else ""
        print(
            "\nControls (unified with DAGGER / the web teleop):\n"
            "  W/A/S/D      MV_FWD / MV_LEFT / MV_BACK / MV_RIGHT   (W = away from the base)\n"
            "  R/F          MV_UP / MV_DOWN\n"
            "  Arrows       aliases: Up/Down = MV_UP/MV_DOWN, Left/Right = MV_LEFT/MV_RIGHT\n"
            f"{rotate_line}"
            "  Space/LShift toggle GRASP <-> RELEASE\n"
            f"  P            start/stop recording{home_note}   Q/Esc  quit\n"
        )

    def _blit_view(self, screen, pygame, rgb, x, w, h, label, font) -> None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        if (w, h) != (rgb.shape[1], rgb.shape[0]):
            surf = pygame.transform.smoothscale(surf, (w, h))
        screen.blit(surf, (x, 0))
        screen.blit(font.render(label, True, (0, 255, 255)), (x + 6, 6))

    def _render(self, screen, pygame, font, big, vw_disp, vh_disp, bar_h) -> None:
        obs = self.latest_obs
        if obs is None:
            return
        self._blit_view(screen, pygame, obs["agentview"], 0, vw_disp, vh_disp, "AgentView", font)
        self._blit_view(screen, pygame, obs["wrist"], vw_disp, vw_disp, vh_disp, "Wrist", font)

        screen.fill((20, 20, 24), pygame.Rect(0, vh_disp, vw_disp * 2, bar_h))
        if self.recorder.active:
            status = f"● REC {self.recorder.dir.name}  step={len(self.recorder.steps)}"
            color = (255, 80, 80)
        else:
            status = "IDLE  (press P to start recording)"
            color = (180, 180, 180)
        screen.blit(big.render(status, True, color), (8, vh_disp + 6))

        grip = "CLOSED" if self.controller.gripper_closed else "OPEN"
        width_mm = ""
        if obs is not None and "gripper_width" in obs:
            width_mm = f" ({float(obs['gripper_width']) * 1000:.1f}mm)"
        info = f"last={self.last_token}   gripper={grip}{width_mm}   next Space={self._next_gripper}"
        screen.blit(font.render(info, True, (220, 220, 220)), (8, vh_disp + 32))
        # Live EEF pose + Z floor: read the Z here while jogging the gripper down to the
        # tabletop to calibrate z_floor_m (the height stops dropping at contact).
        pose_txt = ""
        if obs is not None and "ee_pose" in obs:
            x, y, z = (float(v) for v in np.asarray(obs["ee_pose"], dtype=float)[:3])
            pose_txt = f"eef z={z:+.3f} (x={x:+.3f} y={y:+.3f}) m"
        floor = getattr(self.controller, "z_floor_m", None)
        floor_txt = f"z-floor={floor:.3f}m" if floor is not None else "z-floor=off"
        rotate_help = "Z/X=rot " if self.include_rotate_keys else ""
        screen.blit(
            font.render(f"{pose_txt}   {floor_txt}    [{rotate_help}Space=grip P=rec Q=quit]", True, (150, 220, 150)),
            (8, vh_disp + 54),
        )

    def _on_keydown(self, key, move_keys, pygame) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False
        elif key == pygame.K_p:
            self.toggle_recording()
        elif key in getattr(self, "_gripper_keys", {pygame.K_SPACE}):
            self.do_gripper()
        elif key in move_keys:
            token = move_keys[key]
            self._held_move = (key, token)
            self.do_move(token)
            self._last_move_t = time.time()

    def run(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        # Avoid pulling in the audio subsystem (ALSA init can stall on headless
        # / audio-less machines). We only need video + font.
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(self.window_title)
        # Bundled default font (no system-font scan, which can stall).
        font = pygame.font.Font(None, 22)
        big = pygame.font.Font(None, 28)

        self.capture()
        vh, vw = self.latest_obs["agentview"].shape[:2]  # type: ignore[index]
        s = self.display_scale
        vw_disp, vh_disp = vw * s, vh * s
        bar_h = 80
        screen = pygame.display.set_mode((vw_disp * 2, vh_disp + bar_h))

        # THE single-arm layout (core.teleop.dual.build_single_keymaps) -- shared
        # verbatim with the DAGGER override so collection and intervention are one
        # muscle memory (W = MV_FWD, away from the base; R/F = up/down; arrows as
        # aliases). DONE is an intervention-only token: dropped here.
        from core.teleop.dual import build_single_keymaps  # noqa: PLC0415 - teleop_dual
        # imports this module's recorder, so the shared keymap is imported lazily.

        moves, gripper_keys, _still = build_single_keymaps(
            pygame, self.include_rotate_keys
        )
        move_keys = {k: t for k, t in moves.items() if t != "DONE"}
        self._gripper_keys = set(gripper_keys)
        clock = pygame.time.Clock()
        self._print_controls()

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self._on_keydown(event.key, move_keys, pygame)
                    elif event.type == pygame.KEYUP:
                        if self._held_move and event.key == self._held_move[0]:
                            self._held_move = None
                if not self.running:
                    break
                # Repeat the held movement key at the configured rate.
                if self._held_move and (time.time() - self._last_move_t) >= self.move_interval:
                    self.do_move(self._held_move[1])
                    self._last_move_t = time.time()

                self.capture()
                self._render(screen, pygame, font, big, vw_disp, vh_disp, bar_h)
                pygame.display.flip()
                clock.tick(self.target_fps)
        finally:
            if self.recorder.active:
                self.recorder.stop()
            pygame.quit()
