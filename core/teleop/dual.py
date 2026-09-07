"""Dual-arm keyboard teleoperation + rollout recording (Piper left + right).

Extends the single-arm core (:mod:`core.teleop.single`) to two arms driven from one
keyboard and one pygame window (front + both wrist views):

    LEFT arm    W/A/S/D  MV_FWD/MV_LEFT/MV_BACK/MV_RIGHT     R/F  MV_UP/MV_DOWN
                LShift   toggle GRASP <-> RELEASE            Z/X  ROTATE_CCW/CW
                LAlt     STILL (this arm does nothing this step)
    RIGHT arm   I/J/K/L  MV_FWD/MV_LEFT/MV_BACK/MV_RIGHT     U/H  MV_UP/MV_DOWN
                RShift   toggle GRASP <-> RELEASE            N/M  ROTATE_CCW/CW
                RAlt     STILL (this arm does nothing this step)
    P  start/stop recording        Esc  quit

The bindings live in :func:`build_dual_keymaps` -- the ONE source of truth shared by
this collector and the DAGGER rollout-override plugin (``plugins.dagger``), so corrections
made during inference use exactly the muscle memory learned collecting data.

Two storage modes:

* **Mode A (independent)** -- one :class:`core.teleop.single.RolloutRecorder` per arm under
  ``<save>/left/rollout_NNN`` and ``<save>/right/rollout_NNN``. A step is appended
  only to the acting arm's recorder, with the single-arm record schema (that arm's
  wrist as ``wrist``); each tree is indistinguishable from a single-arm dataset.

* **Mode B (synchronous)** -- one :class:`DualRolloutRecorder` per rollout. Every
  timestamp where EITHER arm acts stores ONE record holding both arms' tokens --
  the inactive arm gets the ``STILL`` pseudo-action -- plus the front + both wrist
  frames. Actions from both arms within the same UI tick share one timestamp.

Records store the state the action was taken FROM: obs_t is captured and written,
THEN the controllers execute -- (obs_t, a_t) pairs, same as the single-arm core.

Real-time behaviour: both arms' tokens at a timestamp execute SIMULTANEOUSLY (one
thread per acting arm -- each controller blocks for its whole motion, so sequential
execution made the arms visibly take turns), and key input is LATEST-INTENT, not a
queue: at most one pending move + one pending gripper toggle per side, newest wins,
so taps delivered while a previous motion blocked the loop never replay as a backlog
(which overshot the operator's target on hardware).

Synchronized stepping (``sync_steps``; ON by default in Mode B): a step is taken only
once BOTH arms have registered an intent for it, so every recorded timestamp carries a
DELIBERATE decision from each arm. An arm whose turn it is to do nothing says so
explicitly with its STILL key (Alt / Enter). Without this, any single keypress recorded
a timestamp and the idle arm was auto-filled with ``STILL`` -- driving one arm for a
while flooded the dataset with STILL labels for the other. Mode A (independent per-arm
datasets, no STILL records at all) keeps the ungated behaviour.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from core.action_units import ROTATE_ATOMS
from interpreters.real_atomic_controller import GRASP_ATOM, RELEASE_ATOM
from core.record.images import save_mp4, save_png
from core.piper.dual_session import SIDES
from core.teleop.single import RolloutRecorder, _CV2, _FONT, _to_square

# Pseudo-action recorded (Mode B) for an arm that did not act at a timestamp where
# the other arm did. Never sent to a controller.
STILL_ATOM = "STILL"


def build_dual_keymaps(
    pygame, include_rotate_keys: bool = True
) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
    """The dual-arm key bindings: (left_moves, right_moves, gripper->side, still->side).

    Single source of truth, shared by the teleop collector and the DAGGER
    inference-override tool -- change bindings HERE and both stay in sync.
    """
    left = {
        pygame.K_w: "MV_FWD",
        pygame.K_s: "MV_BACK",
        pygame.K_a: "MV_LEFT",
        pygame.K_d: "MV_RIGHT",
        pygame.K_r: "MV_UP",
        pygame.K_f: "MV_DOWN",
    }
    right = {
        pygame.K_i: "MV_FWD",
        pygame.K_k: "MV_BACK",
        pygame.K_j: "MV_LEFT",
        pygame.K_l: "MV_RIGHT",
        pygame.K_u: "MV_UP",
        pygame.K_h: "MV_DOWN",
    }
    if include_rotate_keys:
        left[pygame.K_z] = "ROTATE_CCW"
        left[pygame.K_x] = "ROTATE_CW"
        right[pygame.K_n] = "ROTATE_CCW"
        right[pygame.K_m] = "ROTATE_CW"
    grippers = {pygame.K_LSHIFT: "left", pygame.K_RSHIFT: "right"}
    # Explicit "this arm does nothing this step": each arm's OWN Alt key.
    still = {pygame.K_LALT: "left", pygame.K_RALT: "right"}
    return left, right, grippers, still


def build_single_keymaps(
    pygame, include_rotate_keys: bool = True
) -> tuple[dict[int, str], set[int], set[int]]:
    """Single-arm key bindings: ``(moves, gripper_keys, still_keys)``.

    THE single source of truth for every single-arm keyboard surface on the
    Franka -- the DAGGER override (plugins/dagger) AND the data-collection teleop
    (core/teleop/single.py) build from it, so the two never drift (they once disagreed
    on W: the collector's legacy layout had W=MV_BACK while DAGGER had W=MV_FWD).

    Layout: the dual rig's LEFT-hand cluster (same physical keys as
    :func:`build_dual_keymaps`'s left arm, so muscle memory transfers between
    rigs) -- W = MV_FWD, AWAY from the robot base. Arrow keys are aliases
    (Up/Down = MV_UP/MV_DOWN, Left/Right = MV_LEFT/MV_RIGHT) for operators used
    to the collector's / web page's arrows. SPACE joins LShift as the gripper
    toggle. Enter emits DONE (declare the current stage complete): with one arm
    there is no second-arm key conflict, and the operator is the ground truth for
    "this stage is visibly done" during an intervention (collectors drop DONE)."""
    left, _right, _grippers, _still = build_dual_keymaps(pygame, include_rotate_keys)
    moves = dict(left)
    moves[pygame.K_UP] = "MV_UP"
    moves[pygame.K_DOWN] = "MV_DOWN"
    moves[pygame.K_LEFT] = "MV_LEFT"
    moves[pygame.K_RIGHT] = "MV_RIGHT"
    moves[pygame.K_RETURN] = "DONE"
    moves[pygame.K_KP_ENTER] = "DONE"
    gripper_keys = {pygame.K_LSHIFT, pygame.K_SPACE}
    still_keys = {pygame.K_LALT}
    return moves, gripper_keys, still_keys


def compose_dual_step_frame(
    agentview: np.ndarray,
    wrist_left: np.ndarray,
    wrist_right: np.ndarray,
    step: int,
    token_left: str,
    token_right: str,
    size: int = 256,
    bar_h: int = 32,
) -> np.ndarray:
    """One RGB video frame: [agentview | left wrist | right wrist] + status bar."""
    import cv2  # matches core.teleop.single: cv2 presence is gated by _CV2

    views = np.hstack([_to_square(agentview, size), _to_square(wrist_left, size), _to_square(wrist_right, size)])
    bar = np.zeros((bar_h, views.shape[1], 3), dtype=np.uint8)
    canvas = np.vstack([views, bar]).astype(np.uint8)
    if _CV2:
        cv2.putText(canvas, "AgentView", (6, 18), _FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "L-Wrist", (size + 6, 18), _FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "R-Wrist", (2 * size + 6, 18), _FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{step:03d}  L:{token_left:<10} R:{token_right:<10}",
            (6, size + bar_h - 11),
            _FONT,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


class DualRolloutRecorder:
    """Mode B on-disk layout: one synchronized record per timestamp.

    <save_path>/rollout_NNN/
        agentview/0000.png ...       front-camera frames (one per step)
        wrist_left/0000.png ...      left-wrist frames
        wrist_right/0000.png ...     right-wrist frames
        actions.jsonl                per-step {left: {...}, right: {...}} records
        metadata.json                rollout summary
        visualization.mp4            three views + action overlay
    """

    def __init__(
        self,
        root: str | Path,
        prefix: str = "rollout_",
        video_fps: float = 10.0,
        verbose: bool = True,
    ) -> None:
        # Reuse the single-arm recorder for numbering/lifecycle bookkeeping only; the
        # dual layout is written by this class (different dirs and record schema).
        self._base = RolloutRecorder(root, prefix=prefix, video_fps=video_fps, verbose=False)
        self.verbose = verbose
        self.session_meta: dict[str, Any] = {}

    # lifecycle ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._base.active

    @property
    def dir(self) -> Optional[Path]:
        return self._base.dir

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self._base.steps

    def start(self) -> Path:
        if self.active:
            return self.dir  # type: ignore[return-value]
        d = self._base.start()
        # Replace the single-arm image dirs with the dual layout. The base recorder
        # just created an empty wrist/ in a brand-new numbered dir; remove it, but a
        # non-empty one (never expected) must not kill the recording.
        try:
            (d / "wrist").rmdir()
        except OSError:
            pass
        (d / "agentview").mkdir(exist_ok=True)
        (d / "wrist_left").mkdir(exist_ok=True)
        (d / "wrist_right").mkdir(exist_ok=True)
        if self.verbose:
            print(f"[rec] ● recording (dual, synchronized) -> {d}")
        return d

    def add_step(
        self,
        tokens: dict[str, str],
        kinds: dict[str, str],
        obs: dict[str, Any],
        grippers_closed: dict[str, Optional[bool]],
    ) -> None:
        """Append one synchronized record. ``tokens[side]`` is the executed atomic
        token or ``STILL`` for an arm that did not act at this timestamp."""
        if not self.active or self.dir is None:
            return
        step = len(self.steps)
        name = f"{step:04d}.png"
        save_png(self.dir / "agentview" / name, obs["agentview"])
        save_png(self.dir / "wrist_left" / name, obs["wrist_left"])
        save_png(self.dir / "wrist_right" / name, obs["wrist_right"])
        record: dict[str, Any] = {
            "step": step,
            "agentview": f"agentview/{name}",
            "wrist_left": f"wrist_left/{name}",
            "wrist_right": f"wrist_right/{name}",
            "time": round(time.time(), 3),
        }
        for side in SIDES:
            closed = grippers_closed[side]
            record[side] = {
                "token": tokens[side],
                "kind": kinds[side],
                "gripper_closed": None if closed is None else bool(closed),
                "ee_pose": np.asarray(obs[side]["ee_pose"], dtype=float).round(5).tolist(),
                "gripper_width": round(float(obs[side]["gripper_width"]), 5),
            }
        self.steps.append(record)
        with self._base.jsonl.open("a", encoding="utf-8") as f:  # type: ignore[union-attr]
            f.write(json.dumps(record) + "\n")
        self._base.video_frames.append(
            compose_dual_step_frame(
                obs["agentview"], obs["wrist_left"], obs["wrist_right"],
                step, tokens["left"], tokens["right"],
            )
        )

    def stop(self) -> Optional[Path]:
        if not self.active or self.dir is None:
            return None
        meta = {
            "rollout": self.dir.name,
            "index": self._base.index,
            "num_steps": len(self.steps),
            "tokens_left": [s["left"]["token"] for s in self.steps],
            "tokens_right": [s["right"]["token"] for s in self.steps],
            "created": datetime.now().isoformat(timespec="seconds"),
            **self.session_meta,
        }
        (self.dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        if self._base.video_frames:
            try:
                save_mp4(self.dir / "visualization.mp4", self._base.video_frames, self._base.video_fps)
            except Exception as exc:  # noqa: BLE001 - video is best-effort
                print(f"[rec] WARNING: could not write visualization video: {exc}")
        done_dir = self.dir
        if self.verbose:
            print(f"[rec] ■ stopped -> {done_dir} ({len(self.steps)} steps)")
        # Reset the base's lifecycle state (mirrors RolloutRecorder.stop without the
        # single-arm metadata/video, which we already wrote).
        self._base.active = False
        self._base.dir = None
        self._base.jsonl = None
        self._base.steps = []
        self._base.video_frames = []
        return done_dir


class DualRolloutCollector:
    """Pygame keyboard teleop driving two RealAtomicController-shaped controllers.

    ``mode`` selects the storage layout: ``"A"`` (independent; ``recorders`` is a
    ``{"left": RolloutRecorder, "right": RolloutRecorder}`` dict) or ``"B"``
    (synchronous; ``recorders`` is one :class:`DualRolloutRecorder`).

    ``home_fn`` resets BOTH arms to their BEGIN poses SIMULTANEOUSLY; it runs after a recording stops, after which both
    controller setpoints are re-synced from the robots (same contract as the
    single-arm collector -- the re-sync prevents the first post-home command from
    jumping toward the stale pre-home setpoint).
    """

    def __init__(
        self,
        session: Any,
        controllers: dict[str, Any],
        recorders: Any,
        mode: str = "B",
        move_interval: float = 0.12,
        display_scale: int = 2,
        target_fps: int = 30,
        home_fn: Optional[Callable[[], None]] = None,
        include_rotate_keys: bool = True,
        sync_steps: Optional[bool] = None,
        window_title: str = "Show-Harness - Dual Piper Rollout Collection",
    ) -> None:
        mode = str(mode).strip().upper()
        if mode not in ("A", "B"):
            raise ValueError(f"mode must be 'A' (independent) or 'B' (synchronous), got {mode!r}")
        if mode == "A" and not isinstance(recorders, dict):
            raise ValueError("Mode A needs a {'left','right'} dict of RolloutRecorders")
        if mode == "B" and isinstance(recorders, dict):
            raise ValueError("Mode B needs a single DualRolloutRecorder")
        self.session = session
        self.controllers = controllers
        self.recorders = recorders
        self.mode = mode
        # Synchronized stepping: hold every action until BOTH arms have registered an
        # intent for the step (an idle arm says STILL with its own key). Default: on in
        # Mode B, whose records carry both arms and would otherwise be padded with
        # auto-STILL for whichever arm the operator was not driving; off in Mode A,
        # where the per-arm datasets are independent and record no STILL at all.
        self.sync_steps = (mode == "B") if sync_steps is None else bool(sync_steps)
        self.move_interval = float(move_interval)
        self.display_scale = int(display_scale)
        self.target_fps = int(target_fps)
        self.home_fn = home_fn
        self.include_rotate_keys = bool(include_rotate_keys)
        self.window_title = window_title

        self.running = True
        self.latest_obs: Optional[dict[str, Any]] = None
        self._next_gripper = {side: GRASP_ATOM for side in SIDES}
        self._held_move: dict[str, Optional[tuple[int, str]]] = {side: None for side in SIDES}
        self._last_move_t = {side: 0.0 for side in SIDES}
        # LATEST-INTENT slots: at most ONE pending move and one pending gripper toggle
        # per side. A tick blocks for the whole motion it executes (~0.4 s smooth ramp;
        # a gripper settle longer), so key events delivered meanwhile arrive in a burst
        # at the next loop top -- with a FIFO the arm then REPLAYED the entire burst
        # over the following seconds and overshot the operator's intent (observed on
        # hardware). Newest-wins slots collapse such a burst to the latest press: what
        # executes is what the operator last asked for, never a backlog. The gripper
        # slot stays separate (a toggle must not be eaten by a later move tap) and is
        # resolved from _next_gripper at FLUSH time, so its GRASP/RELEASE direction is
        # decided when it actually runs.
        self._pending_move: dict[str, Optional[tuple[str, str]]] = {side: None for side in SIDES}
        self._pending_gripper: dict[str, bool] = {side: False for side in SIDES}
        self.last_token = {side: "-" for side in SIDES}

    # -- key maps ------------------------------------------------------------
    def _build_keymaps(
        self, pygame
    ) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
        """Return (left_move_keys, right_move_keys, gripper_keys, still_keys) --
        the latter two mapping key -> side. See :func:`build_dual_keymaps`."""
        return build_dual_keymaps(pygame, self.include_rotate_keys)

    # -- robot / data --------------------------------------------------------
    def capture(self) -> dict[str, Any]:
        try:
            self.latest_obs = self.session.get_observation()
        except Exception as exc:  # noqa: BLE001
            if self.latest_obs is None:
                raise
            print(f"[capture] WARNING: {exc}; reusing previous frame")
        return self.latest_obs  # type: ignore[return-value]

    def queue_move(self, side: str, token: str) -> None:
        # Newest-wins: replace any not-yet-executed move so a burst of taps delivered
        # while a previous motion blocked the loop never replays as a backlog.
        kind = "rotate" if token in ROTATE_ATOMS else "move"
        self._pending_move[side] = (token, kind)

    def queue_still(self, side: str) -> None:
        """This arm's deliberate no-op for the coming step (its STILL key).

        Overrides a pending move -- pressing STILL is also how the operator CANCELS an
        intent registered while waiting for the other arm.
        """
        self._pending_move[side] = (STILL_ATOM, "still")

    def queue_gripper(self, side: str) -> None:
        # A flag, not a token: the actual GRASP/RELEASE is resolved at flush time from
        # _next_gripper, which only advances after the previous toggle executed.
        self._pending_gripper[side] = True

    def has_intent(self, side: str) -> bool:
        """True once this arm has registered what it does next (move / gripper / STILL)."""
        return self._pending_gripper[side] or self._pending_move[side] is not None

    def _take_pending(self, side: str) -> Optional[tuple[str, str]]:
        """Consume this side's next action: the gripper toggle first (a deliberate,
        rare press must not be starved by movement taps), else the pending move."""
        if self._pending_gripper[side]:
            self._pending_gripper[side] = False
            return self._next_gripper[side], "gripper"
        move = self._pending_move[side]
        self._pending_move[side] = None
        return move

    def _flush_pending(self) -> None:
        """Execute + record the pending action per side as ONE synchronized timestamp.

        With ``sync_steps`` the step waits until BOTH arms have registered an intent,
        so every record carries a deliberate decision from each arm (an idle arm's is
        its explicit STILL). Otherwise a single arm's action takes a step on its own.

        Records the state the actions are taken FROM (obs_t, a_t): one fresh dual
        observation is captured and written, THEN the taken tokens execute -- BOTH
        arms SIMULTANEOUSLY, one thread per acting arm (each controller blocks for
        its whole motion; run in sequence the arms visibly took turns). Same
        join-then-reraise policy as :func:`core.piper.poses.go_begin_dual`, so one
        arm's fault never leaves the other mid-flight. STILL is a recorded label only:
        it is never sent to a controller. In Mode B the still arm records ``STILL``;
        in Mode A it records nothing (its dataset only holds steps where it acted).
        """
        if self.sync_steps and not all(self.has_intent(s) for s in SIDES):
            return  # wait for the other arm's confirmation; intents stay pending
        pending = {side: self._take_pending(side) for side in SIDES}
        if not any(pending.values()):
            return
        obs = self.capture()

        # Record first (obs_t, a_t) ...
        if self.mode == "B":
            tokens = {s: (pending[s][0] if pending[s] else STILL_ATOM) for s in SIDES}
            kinds = {s: (pending[s][1] if pending[s] else "still") for s in SIDES}
            grips = {s: self.controllers[s].gripper_closed for s in SIDES}
            self.recorders.add_step(tokens, kinds, obs, grips)
        else:
            for side in SIDES:
                # Mode A datasets are single-arm: a STILL is a confirmation to the UI,
                # not an action, so it is never written to that arm's rollout.
                if pending[side] is None or pending[side][1] == "still":
                    continue
                token, kind = pending[side]
                self.recorders[side].add_step(
                    token, kind, self.session.arm_observation(obs, side),
                    self.controllers[side].gripper_closed,
                )

        # ... then execute, both arms at once.
        def _continuous(side: str, token: str, kind: str) -> bool:
            # Smoothness: while the operator HOLDS a movement key, the same token keeps
            # repeating -- tell the controller another aligned move is coming so it ends
            # this one at cruise speed instead of decelerating to a stop. The arm then
            # flows continuously for as long as the key is down (the key-up handler
            # calls end_stream to bring it to rest).
            held = self._held_move[side]
            return kind == "move" and held is not None and held[1] == token

        errors: dict[str, BaseException] = {}

        def _run(side: str, token: str, kind: str) -> None:
            try:
                self.controllers[side].step(token, continuous=_continuous(side, token, kind))
            except BaseException as exc:  # noqa: BLE001 - re-raised after both join
                errors[side] = exc

        # STILL is a label, never a command: the arm simply holds its setpoint.
        for side in SIDES:
            if pending[side] is not None and pending[side][1] == "still":
                self.last_token[side] = STILL_ATOM
        acting = {
            side: act
            for side, act in pending.items()
            if act is not None and act[1] != "still"
        }
        if not acting:
            return
        if len(acting) == 1:
            side, (token, kind) = next(iter(acting.items()))
            _run(side, token, kind)
        else:
            threads = [
                threading.Thread(target=_run, args=(side, token, kind), name=f"teleop-{side}")
                for side, (token, kind) in acting.items()
            ]
            for t in threads:
                t.start()
            for t in threads:  # always join both, even if one already failed
                t.join()
        for side, (token, kind) in acting.items():
            if side in errors:
                continue
            self.last_token[side] = token
            if kind == "gripper":
                # Track the controller's ACTUAL state (empty-grasp auto-reopen safe).
                self._next_gripper[side] = (
                    RELEASE_ATOM if self.controllers[side].gripper_closed else GRASP_ATOM
                )
        if errors:
            side, exc = next(iter(errors.items()))
            raise RuntimeError(f"{side} arm teleop step failed: {exc}") from exc

    # -- recording lifecycle ---------------------------------------------------
    def _recording_active(self) -> bool:
        if self.mode == "B":
            return self.recorders.active
        return any(self.recorders[s].active for s in SIDES)

    def toggle_recording(self) -> None:
        if self._recording_active():
            if self.mode == "B":
                self.recorders.stop()
            else:
                for side in SIDES:
                    self.recorders[side].stop()
            # Auto-home both arms so the next demo starts from the same poses; the
            # homing motion is not recorded (recorders already stopped).
            if self.home_fn is not None:
                self.reset_to_home()
        else:
            if self.mode == "B":
                self.recorders.start()
            else:
                for side in SIDES:
                    self.recorders[side].start()

    def reset_to_home(self) -> None:
        if self.home_fn is None:
            return
        try:
            for side in SIDES:
                self.controllers[side].end_stream()  # never reset while a stream is in flight
            self.home_fn()
            for side in SIDES:
                self.controllers[side].sync_from_robot()
                self._next_gripper[side] = (
                    RELEASE_ATOM if self.controllers[side].gripper_closed else GRASP_ATOM
                )
                self._held_move[side] = None
                self._pending_move[side] = None
                self._pending_gripper[side] = False
            self.capture()  # refresh the live view to the homed poses
            print("[reset] both arms back at BEGIN; ready for the next recording.")
        except Exception as exc:  # noqa: BLE001 - never let a reset error kill the session
            print(f"[reset] ERROR: homing failed: {exc}")

    # -- ui --------------------------------------------------------------------
    def _print_controls(self) -> None:
        rotate_l = "   Z/X  ROTATE_CCW/CW" if self.include_rotate_keys else ""
        rotate_r = "   N/M  ROTATE_CCW/CW" if self.include_rotate_keys else ""
        home_note = " (stop resets both arms to BEGIN)" if self.home_fn is not None else ""
        sync_note = (
            "\n  STEPPING: a step is taken only when BOTH arms have chosen -- an idle arm\n"
            "            presses its STILL key (LAlt / RAlt). Hold a STILL key to keep\n"
            "            confirming while you drive the other arm. STILL also cancels an\n"
            "            intent you already registered, and stops that arm."
            if self.sync_steps
            else "\n  STEPPING: either arm's key takes a step on its own (the other records STILL)."
        )
        print(
            "\nControls (dual arm, mode "
            f"{self.mode} = {'independent' if self.mode == 'A' else 'synchronized + STILL'}):\n"
            f"  LEFT   W/A/S/D  MV_FWD/MV_LEFT/MV_BACK/MV_RIGHT   R/F  MV_UP/MV_DOWN"
            f"{rotate_l}\n"
            "         LShift   toggle GRASP <-> RELEASE           LAlt   STILL\n"
            f"  RIGHT  I/J/K/L  MV_FWD/MV_LEFT/MV_BACK/MV_RIGHT   U/H  MV_UP/MV_DOWN"
            f"{rotate_r}\n"
            "         RShift   toggle GRASP <-> RELEASE           RAlt   STILL\n"
            f"  P      start/stop recording{home_note}\n"
            "  Esc    quit"
            f"{sync_note}\n"
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
        self._blit_view(screen, pygame, obs["wrist_left"], vw_disp, vw_disp, vh_disp, "L-Wrist", font)
        self._blit_view(screen, pygame, obs["wrist_right"], 2 * vw_disp, vw_disp, vh_disp, "R-Wrist", font)

        screen.fill((20, 20, 24), pygame.Rect(0, vh_disp, vw_disp * 3, bar_h))
        if self._recording_active():
            if self.mode == "B":
                status = f"● REC {self.recorders.dir.name}  step={len(self.recorders.steps)}"
            else:
                nl = len(self.recorders["left"].steps)
                nr = len(self.recorders["right"].steps)
                status = f"● REC (independent)  L steps={nl}  R steps={nr}"
            color = (255, 80, 80)
        else:
            status = f"IDLE  mode {self.mode}  (press P to start recording)"
            color = (180, 180, 180)
        screen.blit(big.render(status, True, color), (8, vh_disp + 6))

        lines = []
        for side in SIDES:
            grip = "CLOSED" if self.controllers[side].gripper_closed else "OPEN"
            width_mm = f" ({float(obs[side]['gripper_width']) * 1000:.1f}mm)" if side in obs else ""
            z = float(np.asarray(obs[side]["ee_pose"], dtype=float)[2]) if side in obs else float("nan")
            floor = getattr(self.controllers[side], "z_floor_m", None)
            floor_txt = f"floor={floor:.3f}" if floor is not None else "floor=off"
            lines.append(
                f"{side[0].upper()}: last={self.last_token[side]:<10} grip={grip}{width_mm} "
                f"z={z:+.3f} {floor_txt}"
            )
        screen.blit(font.render("   ".join(lines), True, (220, 220, 220)), (8, vh_disp + 32))
        # Synchronized stepping: show which arm the step is still waiting on, so a
        # deliberately-gated step never looks like a frozen UI.
        if self.sync_steps:
            waiting = [s for s in SIDES if not self.has_intent(s)]
            if not waiting:
                hint, color = "both ready -> stepping", (150, 220, 150)
            elif len(waiting) == len(SIDES):
                hint, color = "waiting: both arms (LAlt = left STILL, RAlt = right STILL)", (180, 180, 180)
            else:
                other = waiting[0]
                key = "LAlt" if other == "left" else "RAlt"
                hint = f"waiting for the {other.upper()} arm ({key} = STILL)"
                color = (255, 200, 90)
            screen.blit(font.render(hint, True, color), (8, vh_disp + 54))
        else:
            screen.blit(
                font.render("[LShift/RShift=grip  P=rec  Esc=quit]", True, (150, 220, 150)),
                (8, vh_disp + 54),
            )

    def _on_keydown(self, key, left_keys, right_keys, gripper_keys, still_keys, pygame) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_p:
            self.toggle_recording()
        elif key in gripper_keys:
            self.queue_gripper(gripper_keys[key])
        elif key in still_keys:
            side = still_keys[key]
            # STILL replaces whatever that arm was doing: it is both the "do nothing
            # this step" confirmation and the stop key. Take over the held slot (so the
            # auto-repeat below keeps confirming while the key is down, letting the
            # OTHER arm be driven with held keys) and bring a coasting arm to rest.
            self._held_move[side] = (key, STILL_ATOM)
            self._last_move_t[side] = time.time()
            self.queue_still(side)
            self.controllers[side].end_stream()
        elif key in left_keys:
            self._held_move["left"] = (key, left_keys[key])
            self.queue_move("left", left_keys[key])
            self._last_move_t["left"] = time.time()
        elif key in right_keys:
            self._held_move["right"] = (key, right_keys[key])
            self.queue_move("right", right_keys[key])
            self._last_move_t["right"] = time.time()

    def run(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(self.window_title)
        font = pygame.font.Font(None, 22)
        big = pygame.font.Font(None, 28)

        self.capture()
        vh, vw = self.latest_obs["agentview"].shape[:2]  # type: ignore[index]
        s = self.display_scale
        vw_disp, vh_disp = vw * s, vh * s
        bar_h = 80
        screen = pygame.display.set_mode((vw_disp * 3, vh_disp + bar_h))

        left_keys, right_keys, gripper_keys, still_keys = self._build_keymaps(pygame)
        clock = pygame.time.Clock()
        self._print_controls()

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self._on_keydown(
                            event.key, left_keys, right_keys, gripper_keys, still_keys, pygame
                        )
                    elif event.type == pygame.KEYUP:
                        for side in SIDES:
                            if self._held_move[side] and event.key == self._held_move[side][0]:
                                self._held_move[side] = None
                                # The run of repeats is over: bring the arm (which has been
                                # flowing at cruise speed) gently to rest. No-op if it was
                                # not streaming.
                                self.controllers[side].end_stream()
                if not self.running:
                    break
                # Repeat held keys at the configured per-arm rate. A held STILL key
                # re-confirms too, so holding it while driving the other arm with held
                # keys produces a continuous stream of steps (this arm labelled STILL).
                now = time.time()
                for side in SIDES:
                    held = self._held_move[side]
                    if (
                        held
                        and self._pending_move[side] is None
                        and (now - self._last_move_t[side]) >= self.move_interval
                    ):
                        if held[1] == STILL_ATOM:
                            self.queue_still(side)
                        else:
                            self.queue_move(side, held[1])
                        self._last_move_t[side] = now
                # One synchronized timestamp for everything queued this tick.
                self._flush_pending()

                self.capture()
                self._render(screen, pygame, font, big, vw_disp, vh_disp, bar_h)
                pygame.display.flip()
                clock.tick(self.target_fps)
        finally:
            if self._recording_active():
                if self.mode == "B":
                    self.recorders.stop()
                else:
                    for side in SIDES:
                        self.recorders[side].stop()
            pygame.quit()
