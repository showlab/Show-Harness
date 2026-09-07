"""DualTeleopBackend: one worker thread owning BOTH arms, driven by (a_L, a_R) pairs.

The dual-arm build of the keyboard/web teleop, following the compose-and-commit
design (usage guide: gumi/README.md) with the vocabulary the IMPLEMENTED dual
stack settled on: the no-op token is ``STILL`` (core/teleop/dual.py, the Mode-B
dataset schema, prompts/controller_dual.txt) -- not the ``STAY`` placeholder the
single-arm kb build reserved; ``STAY`` is still accepted on the wire as an alias.

One time step = one PAIR
------------------------
A GUI agent's browser actions are strictly serial, so it never operates two arms
"at the same time" -- it SUBMITS a synchronized pair and the backend makes it
simultaneous. ``POST /api/step`` carries one or more pairs:

    {"left": "MV_FWD", "right": "STILL"}          one pair
    {"left": "MV_FWD*3", "right": "MV_UP"}        three pairs; short side padded STILL
    {"command": "L:w*3 R:u g"}                    the command box, same grammar

Each pair executes exactly like one tick of the pygame dual collector
(core.teleop.dual.DualRolloutCollector._flush_pending): record FIRST -- one dual
observation + both tokens, STILL included, via DualRolloutRecorder.add_step -- then
run both non-STILL tokens SIMULTANEOUSLY, one thread per acting arm (each
controller blocks for its whole motion; sequential execution makes the arms
visibly take turns), join both, re-raise the first error. STILL is a recorded
label only; it is never sent to a controller. A pair whose sides are both STILL is
skipped outright (nothing executed, nothing recorded): waiting is a decision an
arm takes WHILE the other acts, not a step of its own.

Grippers are explicit and IDEMPOTENT per arm (GRASP on an already-closed gripper
becomes that side's STILL for the pair, and is recorded as such) -- the same
reasoning as the single-arm kb build: an agent that misreads a screenshot must
never trigger the opposite action. There is no blind-toggle path at all here;
/api/gripper answers 400.

No collision gate. The backend publishes ``gap_m`` -- the live distance between
the two grippers -- and flags ``gap_warn`` under GAP_WARN_M so the UI (and any
screenshot of it) shows the arms getting close; deciding what to do about it is
the operator's reasoning job, exactly as on the pygame collector and the real rig.

Stop gating (real hardware): ANY arm with a completed pick-and-place unlocks Stop
-- "left holds the box, right fills it" tasks would deadlock an every-arm rule.
With a DualSimScene the scene's own task_success() gates instead.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.action_units import MOVE_ATOMS, ROTATE_ATOMS
from interpreters.real_atomic_controller import GRASP_ATOM, RELEASE_ATOM
from core.piper.config import SIDES
from core.teleop.dual import STILL_ATOM
# The command grammar (aliases, *N repeats, L:/R: prefixes) has exactly ONE
# implementation -- the single-arm kb module's. This build only adds the pairing.
from gumi.web_teleop.backend import (
    KEY_ALIASES,
    _split_words,
    expand_tokens,
    parse_command_text,
)

GRIPPER_TOKENS = (GRASP_ATOM, RELEASE_ATOM)

#: Executed pairs the UI history strip keeps.
HISTORY_LEN = 10

#: Upper bound on pairs per /api/step request (worst pair ~2 s on hardware).
MAX_PAIRS = 24

#: Grippers closer than this flag ``gap_warn`` in state() -- a signal, never a gate.
GAP_WARN_M = 0.08

#: Legacy wire spelling from the single-arm kb build, canonicalized on parse.
_NOOP_ALIASES = {"STAY": STILL_ATOM}


class DualTeleopBackend:
    """Single robot-owning worker thread for TWO arms + three camera streams.

    Same concurrency model as the single-arm TeleopBackend (worker queue, blocking
    submit, idle-loop capture feeding MJPEG frames); the command surface is the
    dual one: ``steps`` (a list of pairs) plus start/stop/cancel/task.
    """

    MOVE_TOKENS = tuple(MOVE_ATOMS) + tuple(ROTATE_ATOMS)
    #: Per-arm vocabulary accepted in a pair (STILL is wire-level, handled apart).
    STEP_TOKENS = MOVE_TOKENS + GRIPPER_TOKENS
    VIEWS = ("agentview", "wrist_left", "wrist_right")

    dual_arm = True

    def __init__(
        self,
        session: Any,
        controllers: Dict[str, Any],
        recorder: Any,                       # DualRolloutRecorder
        home_fn: Optional[Callable[[], None]] = None,
        scene: Any = None,                   # DualSimScene or None (real hardware)
        task_text: str = "",
        require_task_done: bool = True,
        capture_fps: float = 15.0,
        jpeg_quality: int = 85,
    ) -> None:
        missing = [s for s in SIDES if s not in controllers]
        if missing:
            raise ValueError(f"controllers must cover {SIDES}; missing {missing}")
        self.session = session
        self.controllers = controllers
        self.recorder = recorder
        self.home_fn = home_fn
        self.scene = scene
        self.task_text = task_text
        self.require_task_done = bool(require_task_done)
        self.capture_interval = 1.0 / float(capture_fps)
        self.jpeg_quality = int(jpeg_quality)

        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run, name="dual-teleop-worker", daemon=True)

        self.latest_obs: Optional[dict] = None
        self.frame_cond = threading.Condition()
        self.frames: Dict[str, bytes] = {}
        self.frame_seq = 0

        # Teleop / task state (only mutated on the worker thread).
        self.last_pair: Dict[str, str] = {s: "-" for s in SIDES}
        self.last_message = "Ready"
        self.history: deque = deque(maxlen=HISTORY_LEN)
        self.holding = {s: False for s in SIDES}
        self.picked = {s: 0 for s in SIDES}
        self.placed = {s: 0 for s in SIDES}
        self.pairs_executed = 0              # since boot; the agent's action-landed proof
        self.rollouts_saved = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._capture()  # fail fast if cameras/robots are not readable
        self._worker.start()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._worker.join(timeout=5.0)
        if self.recorder.active:
            self.recorder.stop()

    # -- public API (called from HTTP handler threads) ------------------------
    def submit(self, kind: str, arg: Any = None, timeout: float = 60.0) -> dict:
        done = threading.Event()
        slot: Dict[str, Any] = {}
        self._q.put((kind, arg, done, slot))
        if not done.wait(timeout):
            return {"ok": False, "status": 504, "message": f"Command {kind} timed out"}
        return slot

    def state(self) -> dict:
        obs = self.latest_obs
        ee = {s: None for s in SIDES}
        gap_m = None
        if obs is not None:
            pos = {}
            for s in SIDES:
                pos[s] = np.asarray(obs[s]["ee_pose"], dtype=float)[:3]
                ee[s] = [round(float(v), 4) for v in pos[s]]
            gap_m = round(float(np.linalg.norm(pos["left"] - pos["right"])), 4)
        st = {
            "recording": self.recorder.active,
            "rollout": self.recorder.dir.name if self.recorder.active and self.recorder.dir else None,
            "steps": len(self.recorder.steps) if self.recorder.active else 0,
            "total_pairs": self.pairs_executed,
            "rollouts_saved": self.rollouts_saved,
            "dual_arm": True,
            "arms": list(SIDES),
            "last": dict(self.last_pair),
            "gripper_closed": {s: bool(self.controllers[s].gripper_closed) for s in SIDES},
            "holding": dict(self.holding),
            "picked": dict(self.picked),
            "placed": dict(self.placed),
            "task_done": self._task_done(),
            "can_stop": self.recorder.active and (not self.require_task_done or self._task_done()),
            "require_task_done": self.require_task_done,
            "task_text": self.task_text,
            "ee_pos": ee,
            "gap_m": gap_m,
            "gap_warn": bool(gap_m is not None and gap_m < GAP_WARN_M),
            "message": self.last_message,
            "sim": self.scene is not None,
            "step_tokens": list(self.STEP_TOKENS),
            "still_token": STILL_ATOM,
            "aliases": dict(KEY_ALIASES),
            "history": [dict(p) for p in self.history],
        }
        meta = getattr(self.recorder, "session_meta", {}) or {}
        st["step_m"] = meta.get("step_m")
        st["yaw_step_rad"] = meta.get("yaw_step_rad")
        if self.scene is not None:
            st["sim_scene"] = self._scene_state()
        return st

    def _scene_state(self) -> dict:
        """Live scene geometry for API-side navigation and debugging.

        Read through the scene's locked snapshot: state() runs on HTTP handler
        threads while a controller thread may be resizing the cubes dict mid-grasp.
        """
        return {"z_table_m": float(self.scene.z_table), **self.scene.snapshot()}

    # -- worker loop -----------------------------------------------------------
    def _run(self) -> None:
        next_capture = 0.0
        while not self._stop_evt.is_set():
            timeout = max(0.005, next_capture - time.monotonic())
            try:
                kind, arg, done, slot = self._q.get(timeout=timeout)
            except queue.Empty:
                try:
                    self._capture()
                except Exception as exc:  # noqa: BLE001 - keep streaming best-effort
                    self.last_message = f"Frame capture failed: {exc}"
                next_capture = time.monotonic() + self.capture_interval
                continue
            try:
                slot.update(self._execute(kind, arg))
            except Exception as exc:  # noqa: BLE001 - a command must never kill the worker
                traceback.print_exc()
                self.last_message = f"{kind} failed: {exc}"
                slot.update({"ok": False, "status": 500, "message": self.last_message})
            finally:
                done.set()

    def _capture(self) -> dict:
        obs = self.session.get_observation()
        self.latest_obs = obs
        display = obs
        getter = getattr(self.session, "get_display_frames", None)
        if getter is not None:  # real DualPiperSession has none: stream the obs frames
            d = getter()
            if d:
                display = d
        encoded = {}
        for name in self.VIEWS:
            bgr = cv2.cvtColor(np.ascontiguousarray(display[name]), cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                encoded[name] = buf.tobytes()
        with self.frame_cond:
            self.frames.update(encoded)
            self.frame_seq += 1
            self.frame_cond.notify_all()
        return obs

    # -- command execution (worker thread only) ---------------------------------
    def _execute(self, kind: str, arg: Any) -> dict:
        if kind == "steps":
            return self._cmd_steps(list(arg or []))
        if kind == "start":
            return self._cmd_start()
        if kind == "stop":
            return self._cmd_stop()
        if kind == "cancel":
            return self._cmd_cancel()
        if kind == "task":
            return self._cmd_set_task(str(arg or ""))
        if kind in ("move", "gripper"):
            return {
                "ok": False,
                "status": 400,
                "message": f"/api/{kind} is single-arm; this is the DUAL rig -- use "
                "POST /api/step with {\"left\": ..., \"right\": ...} "
                "(one request = one synchronized time step).",
            }
        return {"ok": False, "status": 404, "message": f"Unknown command: {kind}"}

    def _cmd_steps(self, pairs: List[Tuple[str, str]]) -> dict:
        """Execute validated pairs in order; stop at the first failure.

        ``executed`` counts pairs that reached the robots (and the dataset);
        ``skipped`` counts pairs that collapsed to STILL/STILL (explicitly, or via
        idempotent grippers) -- nothing ran, nothing was recorded for those.
        """
        results: List[dict] = []
        for left, right in pairs:
            res = self._step_pair({"left": left, "right": right})
            results.append(res)
            if not res.get("ok"):
                return {
                    "ok": False,
                    "status": int(res.get("status", 500)),
                    "message": str(res.get("message", "step failed")),
                    "results": results,
                    "executed": _count(results, executed=True),
                    "skipped": _count(results, executed=False),
                }
        executed = _count(results, executed=True)
        skipped = _count(results, executed=False)
        tail = results[-1] if results else {}
        summary = f"executed {executed} pair(s)"
        if executed and tail.get("left") is not None:
            summary += f"; last L:{tail['left']} R:{tail['right']}"
        if skipped:
            summary += f" ({skipped} no-op)"
        self.last_message = summary
        return {"ok": True, "message": summary, "results": results,
                "executed": executed, "skipped": skipped}

    def _step_pair(self, pair: Dict[str, str]) -> dict:
        """ONE synchronized time step: record (obs_t, a_t pair), then run both arms."""
        notes: List[str] = []
        # Idempotent grippers: a side whose GRASP/RELEASE is already satisfied
        # becomes STILL for this pair -- it can never perform the opposite action.
        for side in SIDES:
            token = pair[side]
            if token in GRIPPER_TOKENS:
                closed = bool(self.controllers[side].gripper_closed)
                if (token == GRASP_ATOM and closed) or (token == RELEASE_ATOM and not closed):
                    pair[side] = STILL_ATOM
                    notes.append(
                        f"{side[0].upper()}:{token} -> STILL (gripper already "
                        f"{'CLOSED' if closed else 'OPEN'})"
                    )
        if all(pair[s] == STILL_ATOM for s in SIDES):
            msg = "STILL/STILL pair skipped (nothing executed, nothing recorded)"
            if notes:
                msg += ": " + "; ".join(notes)
            return {"ok": True, "noop": True, "left": pair["left"], "right": pair["right"],
                    "message": msg}

        kinds = {s: _kind_of(pair[s]) for s in SIDES}

        # Record the state the pair is taken FROM (obs_t, a_t), then execute --
        # identical order to the single-arm backend and the pygame dual collector.
        obs = self.latest_obs if self.latest_obs is not None else self._capture()
        self.recorder.add_step(
            dict(pair), kinds, obs, {s: self.controllers[s].gripper_closed for s in SIDES}
        )

        # Both non-STILL tokens run SIMULTANEOUSLY, one thread per acting arm; both
        # threads are always joined, then the first error is re-raised (same policy
        # as core.teleop.dual._flush_pending / go_begin_dual).
        acting = {s: pair[s] for s in SIDES if pair[s] != STILL_ATOM}
        errors: Dict[str, BaseException] = {}

        def _run_side(side: str, token: str) -> None:
            try:
                self.controllers[side].step(token)
            except BaseException as exc:  # noqa: BLE001 - re-raised after both join
                errors[side] = exc

        if len(acting) == 1:
            side, token = next(iter(acting.items()))
            _run_side(side, token)
        else:
            threads = [
                threading.Thread(target=_run_side, args=(s, t), name=f"dual-step-{s}")
                for s, t in acting.items()
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Bookkeeping from the controllers' ACTUAL post-step state (an empty GRASP
        # auto-reopens inside the controller, so gripper_closed is the truth).
        for side, token in acting.items():
            if side in errors or kinds[side] != "gripper":
                continue
            closed = bool(self.controllers[side].gripper_closed)
            if token == GRASP_ATOM:
                if closed:
                    self.holding[side] = True
                    if self.recorder.active:
                        self.picked[side] += 1
                    notes.append(f"{side[0].upper()}: grasped")
                else:
                    notes.append(f"{side[0].upper()}: empty grasp -- auto-released")
            else:  # RELEASE
                if self.holding[side] and self.recorder.active:
                    self.placed[side] += 1
                self.holding[side] = False
                notes.append(f"{side[0].upper()}: released")

        self.last_pair = dict(pair)
        self.history.append(dict(pair))
        self.pairs_executed += 1
        self._capture()  # refresh the streams right away for instant feedback

        if errors:
            side, exc = next(iter(errors.items()))
            return {"ok": False, "status": 500, "left": pair["left"], "right": pair["right"],
                    "message": f"{side} arm step failed: {exc}"}
        msg = f"L:{pair['left']}  R:{pair['right']}"
        if notes:
            msg += "  (" + "; ".join(notes) + ")"
        return {"ok": True, "left": pair["left"], "right": pair["right"], "message": msg}

    # -- task / recording lifecycle ---------------------------------------------
    def _task_done(self) -> bool:
        if self.scene is not None:
            return bool(self.scene.task_success())
        # Real rig: ANY arm with a completed pick-and-place. `all` would deadlock
        # legitimate asymmetric tasks (one arm holds, the other places).
        return any(c >= 1 for c in self.placed.values())

    def _cmd_set_task(self, text: str) -> dict:
        if self.recorder.active:
            return {"ok": False, "status": 409, "message": "Cannot change the task while recording"}
        text = text.strip()
        if not text:
            return {"ok": False, "status": 400, "message": "Task text must not be empty"}
        self.task_text = text
        self.recorder.session_meta["task"] = text
        self.last_message = f"Task set: {text}"
        return {"ok": True, "message": self.last_message, "task_text": text}

    def _cmd_start(self) -> dict:
        if self.recorder.active:
            return {"ok": False, "status": 409, "message": "Already recording"}
        for side in SIDES:
            self.picked[side] = 0
            self.placed[side] = 0
        self.recorder.session_meta["task"] = self.task_text
        self.recorder.session_meta["teleop"] = "web-dual"
        if self.scene is not None:
            self.recorder.session_meta.update(self.scene.to_meta())
        rollout_dir = self.recorder.start()
        self.last_message = f"REC {rollout_dir.name}"
        return {"ok": True, "message": self.last_message, "rollout": rollout_dir.name}

    def _cmd_stop(self) -> dict:
        if not self.recorder.active:
            return {"ok": False, "status": 409, "message": "Not recording"}
        if self.require_task_done and not self._task_done():
            hint = (
                "place BOTH target cubes on the blue plate" if self.scene is not None
                else "complete at least one pick-and-place with either arm"
            )
            self.last_message = f"Task not complete -- {hint} first (or Cancel to discard)"
            return {"ok": False, "status": 409, "message": self.last_message}
        num_steps = len(self.recorder.steps)
        self.recorder.session_meta["success"] = True
        done_dir = self.recorder.stop()
        self.rollouts_saved += 1
        self._home_and_resync()
        if self.scene is not None:
            self.scene.reset()
        self._capture()
        self.last_message = f"Saved {done_dir.name} ({num_steps} steps); both arms homed"
        return {"ok": True, "message": self.last_message, "saved": str(done_dir), "steps": num_steps}

    def _cmd_cancel(self) -> dict:
        if not self.recorder.active:
            self._home_and_resync()
            if self.scene is not None:
                self.scene.reset()
            self._capture()
            self.last_message = "Both arms homed (no recording in progress)"
            return {"ok": True, "message": self.last_message}
        done_dir = self.recorder.stop()
        if done_dir is not None:
            shutil.rmtree(done_dir, ignore_errors=True)
        self._home_and_resync()
        if self.scene is not None:
            self.scene.reset()
        self._capture()
        discarded = done_dir.name if done_dir else "rollout"
        self.last_message = f"Discarded {discarded}; both arms homed"
        return {"ok": True, "message": self.last_message}

    def _home_and_resync(self) -> None:
        """Home BOTH arms, then re-sync each controller setpoint from its robot."""
        for side in SIDES:
            self.picked[side] = 0
            self.placed[side] = 0
            self.holding[side] = False
        if self.home_fn is None:
            return
        try:
            self.home_fn()
            for side in SIDES:
                self.controllers[side].sync_from_robot()
        except Exception as exc:  # noqa: BLE001 - a reset error must not kill the session
            self.last_message = f"Home failed: {exc}"
            print(f"[web-teleop-dual] reset ERROR: {exc}")


def _kind_of(token: str) -> str:
    if token == STILL_ATOM:
        return "still"
    if token in GRIPPER_TOKENS:
        return "gripper"
    if token in ROTATE_ATOMS:
        return "rotate"
    return "move"


def _count(results: List[dict], executed: bool) -> int:
    """Count ok pairs that did (or did not) actually touch the robots."""
    return sum(1 for r in results if r.get("ok") and bool(r.get("noop")) is not executed)


# ---------------------------------------------------------------------------
# Request parsing (no robot access: runs on the HTTP thread, unit-testable)
# ---------------------------------------------------------------------------
def _expand_side(value: Any, label: str) -> tuple:
    """Words -> canonical token list for one arm (aliases, *N, STAY->STILL)."""
    words = _split_words(value)
    if words is None:
        return None, (400, f"'{label}': expected a token string or a list of token strings")
    tokens, err = expand_tokens(words)
    if err is not None:
        return None, (err[0], f"'{label}': {err[1]}")
    return [_NOOP_ALIASES.get(t, t) for t in tokens], None


def parse_dual_step_request(body: dict, backend: DualTeleopBackend) -> tuple:
    """Normalize a /api/step body to ``(pairs, error)`` where pairs is
    ``[(left_token, right_token), ...]``.

    Accepted shapes (case-insensitive; ``*N`` repeats and single-letter aliases
    are allowed everywhere; ``STAY`` is accepted as a spelling of ``STILL``):

        {"left": "MV_FWD", "right": "STILL"}
        {"left": "MV_FWD*3", "right": ["MV_UP", "GRASP"]}   -- zipped, STILL-padded
        {"command": "L:w*3 R:u g"}                          -- one command-box line

    A body with only ``token``/``tokens`` (no arm) is rejected: on a two-arm rig
    every action must say which arm it drives. ``error`` is ``(status, message)``.
    """
    if not isinstance(body, dict):
        return None, (400, "body must be a JSON object")

    if "command" in body:
        if any(k in body for k in ("token", "tokens", "left", "right")):
            return None, (400, "use either 'command' or {left,right}, not both")
        if not isinstance(body["command"], str):
            return None, (400, "'command' must be a string")
        parsed, err = parse_command_text(body["command"])
        if err is not None:
            return None, err
        if "tokens" in parsed:
            return None, (
                400,
                "this is the DUAL-ARM rig: prefix every group with an arm, e.g. "
                "'L:MV_FWD*2 R:STILL' or 'L:w*3 R:g' (an omitted arm is STILL).",
            )
        body = parsed

    if "token" in body or "tokens" in body:
        return None, (
            400,
            "this is the DUAL-ARM rig: use {\"left\": ..., \"right\": ...} "
            "(either side may be omitted -> STILL), not {token|tokens}.",
        )

    left, err = _expand_side(body.get("left"), "left")
    if err is not None:
        return None, err
    right, err = _expand_side(body.get("right"), "right")
    if err is not None:
        return None, err
    if not left and not right:
        return None, (
            400,
            "no tokens given; expected e.g. {\"left\": \"MV_FWD\", \"right\": \"STILL\"} "
            "or {\"command\": \"L:w*3 R:u\"}",
        )

    allowed = set(backend.STEP_TOKENS) | {STILL_ATOM}
    bad = [t for t in left + right if t not in allowed]
    if bad:
        aliases = ", ".join(f"{k}={v}" for k, v in KEY_ALIASES.items())
        return None, (
            400,
            f"illegal action token(s): {', '.join(sorted(set(bad)))}; allowed: "
            f"{', '.join(list(backend.STEP_TOKENS) + [STILL_ATOM])} (aliases: {aliases})",
        )

    n = max(len(left), len(right))
    pairs = [
        (left[i] if i < len(left) else STILL_ATOM,
         right[i] if i < len(right) else STILL_ATOM)
        for i in range(n)
    ]
    if all(l == STILL_ATOM and r == STILL_ATOM for l, r in pairs):
        return None, (400, "at least one arm must act (an all-STILL request is a no-op)")
    if len(pairs) > MAX_PAIRS:
        return None, (400, f"too many pairs ({len(pairs)}); max {MAX_PAIRS} per call")
    return pairs, None
