"""TeleopBackend: the single robot-owning worker thread behind the web UI.

Mirrors core.teleop.single.RolloutCollector semantics exactly -- each accepted command
records (obs_t, a_t) then executes the token; Stop auto-homes and re-syncs the
controller -- but is driven by HTTP requests instead of pygame key events.

Concurrency model: ALL robot/controller/recorder access happens on one worker
thread. HTTP handlers submit commands to a queue and block for the result, so
commands serialize naturally (a held button can never overlap a homing move).
Between commands the worker captures observations at ``capture_fps`` and keeps
JPEG-encoded frames for the MJPEG streams.

Task rule (gates Stop): a rollout may only be stopped after the task is done.
  * With a SimScene: the orange cube physically rests on the plate.
  * On real hardware: at least one successful (non-empty) GRASP followed by a
    RELEASE happened DURING the recording -- the empty-grasp auto-reopen means a
    surviving GRASP really caught something; where it was released is up to the
    operator (there is no scene ground truth on the real rig).
Cancel (discard the in-progress rollout and home the arm) is always allowed.

The task string can be changed at runtime (``set_task`` -> queued as a "task"
command), but NOT while a recording is open: the rollout metadata must keep
describing what the operator actually demonstrated.
"""
from __future__ import annotations

import queue
import re
import shutil
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Optional

import cv2
import numpy as np

from core.action_units import MOVE_ATOMS, ROTATE_ATOMS
from interpreters.real_atomic_controller import GRASP_ATOM, RELEASE_ATOM
from core.teleop.single import RolloutRecorder



# Placeholder no-op action for the not-yet-active arm of a synchronized commit.
STAY_TOKEN = "STAY"

GRIPPER_TOKENS = (GRASP_ATOM, RELEASE_ATOM)

#: How many executed tokens the UI history strip keeps.
HISTORY_LEN = 12

#: Single-letter aliases accepted by the command box and the API. They ARE the keyboard
#: keys, so "w*3 a g" is exactly the typed form of pressing 3-W, then A, then G.
KEY_ALIASES = {
    "W": "MV_FWD",
    "S": "MV_BACK",
    "A": "MV_LEFT",
    "D": "MV_RIGHT",
    "Q": "MV_UP",
    "E": "MV_DOWN",
    "Z": "ROTATE_CCW",
    "X": "ROTATE_CW",
    "G": GRASP_ATOM,
    "R": RELEASE_ATOM,
}

#: Upper bound on a single ``*N`` repeat. The per-request cap (MAX_BATCH_TOKENS in
#: server.py) still applies to the whole expanded sequence.
MAX_REPEAT = 64

# "MV_FWD", "MV_FWD*3", "w", "w*3" -- a name with an optional repeat count.
_TOKEN_RE = re.compile(r"^([A-Z_]+)(?:\*([0-9]+))?$")
# "L:" / "R:" -- an arm prefix, glued to its token ("L:MV_FWD") or standing alone.
_ARM_RE = re.compile(r"^([LR]):(.*)$")

class TeleopBackend:
    MOVE_TOKENS = tuple(MOVE_ATOMS) + tuple(ROTATE_ATOMS)

    #: Tokens accepted by /api/step (motion + rotation from the base class, plus the two
    #: gripper atoms; STAY is tolerated as a placeholder, see the module docstring).
    STEP_TOKENS = MOVE_TOKENS + GRIPPER_TOKENS

    #: Flip to True (and wire a right-hand controller) to enable compose-and-commit.
    dual_arm = False

    def __init__(
        self,
        session: Any,
        controller: Any,
        recorder: RolloutRecorder,
        home_fn: Optional[Callable[[], None]] = None,
        scene: Any = None,               # SimScene or None (real hardware)
        task_text: str = "",
        require_task_done: bool = True,
        capture_fps: float = 15.0,
        jpeg_quality: int = 85,
    ) -> None:
        self.session = session
        self.controller = controller
        self.recorder = recorder
        self.home_fn = home_fn
        self.scene = scene
        self.task_text = task_text
        self.require_task_done = bool(require_task_done)
        self.capture_interval = 1.0 / float(capture_fps)
        self.jpeg_quality = int(jpeg_quality)

        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run, name="teleop-worker", daemon=True)

        # Latest observation + encoded stream frames.
        self.latest_obs: Optional[dict] = None
        self.frame_cond = threading.Condition()
        self.frames: dict[str, bytes] = {}
        self.frame_seq = 0

        # Teleop / task state (only mutated on the worker thread).
        self.last_token = "-"
        self.last_message = "Ready"
        self._next_gripper = GRASP_ATOM
        self.picked_count = 0
        self.placed_count = 0
        self.holding = False
        self.rollouts_saved = 0

    # -- lifecycle -----------------------------------------------------------
        #: Last HISTORY_LEN tokens that actually ran (worker thread is the only writer).
        self.history: deque = deque(maxlen=HISTORY_LEN)

    def start(self) -> None:
        self._capture()  # fail fast if cameras/robot are not readable
        self._sync_gripper_toggle()
        self._worker.start()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._worker.join(timeout=5.0)
        if self.recorder.active:
            self.recorder.stop()

    # -- public API (called from HTTP handler threads) -------------------------
    def submit(self, kind: str, arg: Optional[str] = None, timeout: float = 60.0) -> dict:
        """Enqueue a command for the worker thread and block for its result."""
        done = threading.Event()
        slot: dict[str, Any] = {}
        self._q.put((kind, arg, done, slot))
        if not done.wait(timeout):
            return {"ok": False, "status": 504, "message": f"Command {kind} timed out"}
        return slot

    def set_task(self, text: str) -> dict:
        """Change the task string without restarting the server (worker-thread queued)."""
        return self.submit("task", text)

    def state(self) -> dict:
        obs = self.latest_obs
        ee = None
        if obs is not None:
            ee = [round(float(v), 4) for v in np.asarray(obs["ee_pose"], dtype=float)[:3]]
        st = {
            "recording": self.recorder.active,
            "rollout": self.recorder.dir.name if self.recorder.active and self.recorder.dir else None,
            "steps": len(self.recorder.steps) if self.recorder.active else 0,
            "rollouts_saved": self.rollouts_saved,
            "last_token": self.last_token,
            "gripper_closed": bool(self.controller.gripper_closed),
            "next_gripper": self._next_gripper,
            "holding": self.holding,
            "picked": self.picked_count,
            "placed": self.placed_count,
            "task_done": self._task_done(),
            "can_stop": self.recorder.active and (not self.require_task_done or self._task_done()),
            "require_task_done": self.require_task_done,
            "task_text": self.task_text,
            "ee_pos": ee,
            "message": self.last_message,
            "sim": self.scene is not None,
        }
        # Fields the folded keyboard UI reads on top of the web UI's base state.
        st["dual_arm"] = bool(self.dual_arm)
        st["arms"] = ["left", "right"] if self.dual_arm else ["left"]
        st["step_tokens"] = list(self.STEP_TOKENS)
        st["stay_token"] = STAY_TOKEN
        st["aliases"] = dict(KEY_ALIASES)
        st["history"] = list(self.history)
        meta = getattr(self.recorder, "session_meta", {}) or {}
        st["step_m"] = meta.get("step_m")
        st["yaw_step_rad"] = meta.get("yaw_step_rad")
        if self.scene is not None:
            st["sim_scene"] = self._scene_state()
        return st

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
        if getter is not None:
            d = getter()
            if d:
                display = d
        encoded = {}
        for name in ("agentview", "wrist"):
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
    def _execute(self, kind: str, arg: Optional[str]) -> dict:
        if kind == "step":
            return self._cmd_step(list(arg or []))
        if kind == "move":
            return self._cmd_move(arg or "")
        if kind == "gripper":
            return self._cmd_gripper()
        if kind == "start":
            return self._cmd_start()
        if kind == "stop":
            return self._cmd_stop()
        if kind == "cancel":
            return self._cmd_cancel()
        if kind == "task":
            return self._cmd_set_task(arg or "")
        return {"ok": False, "status": 404, "message": f"Unknown command: {kind}"}

    def _record_and_step(self, token: str, kind: str) -> None:
        obs = self.latest_obs if self.latest_obs is not None else self._capture()
        # Record the state the action was taken FROM (obs_t, a_t), then execute.
        self.recorder.add_step(token, kind, obs, self.controller.gripper_closed)
        self.controller.step(token)
        self.last_token = token
        self._capture()  # refresh the stream right away so a click gives instant feedback

        self.history.append(token)  # only tokens that really ran land here

    def _scene_state(self) -> dict:
        """Live scene geometry (updates as cubes are picked/placed) for API navigation."""
        from gumi.web_teleop import sim

        scene = self.scene
        return {
            "z_table_m": float(scene.z_table),
            "plate_xy_m": [round(float(v), 4) for v in scene.plate_xy],
            "cubes_xy_m": {
                k: [round(float(v), 4) for v in xy] for k, xy in scene.cubes.items()
            },
            "held": scene.held,
            "held_xy_m": ([round(float(v), 4) for v in scene.held_xy] if scene.held else None),
            "target": "orange_cube",
            "place_radius_m": sim.PLACE_RADIUS_M,
            "grasp_xy_radius_m": sim.GRASP_XY_RADIUS_M,
            "grasp_max_height_m": sim.GRASP_MAX_HEIGHT_M,
        }

    def _cmd_step(self, tokens: list) -> dict:
        """Execute a validated token sequence in order; stop at the first failure.

        ``executed`` counts tokens that reached the robot (and therefore the dataset);
        ``skipped`` counts no-ops -- STAY, plus an idempotent GRASP/RELEASE that was
        already satisfied. Together they account for every accepted token.
        """
        results: list[dict] = []
        for token in tokens:
            if token == STAY_TOKEN:
                results.append(
                    {"token": token, "ok": True, "noop": True,
                     "message": "STAY (no-op, not recorded)"}
                )
                continue
            if token in GRIPPER_TOKENS:
                res = self._cmd_gripper_token(token)
            elif token in self.MOVE_TOKENS:
                res = self._cmd_move(token)
            else:  # pre-validated by parse_step_request, so this is belt-and-braces
                res = {"ok": False, "status": 400, "message": f"illegal action token: {token}"}
            results.append({"token": token, **res})
            if not res.get("ok"):
                return {
                    "ok": False,
                    "status": int(res.get("status", 400)),
                    "message": str(res.get("message", "step failed")),
                    "results": results,
                    "executed": _count(results, executed=True),
                    "skipped": _count(results, executed=False),
                }
        executed = _count(results, executed=True)
        skipped = _count(results, executed=False)
        summary = f"executed {executed} token(s): {' '.join(tokens) if tokens else '-'}"
        if skipped:
            summary += f" ({skipped} no-op)"
        self.last_message = summary
        return {
            "ok": True,
            "message": summary,
            "results": results,
            "executed": executed,
            "skipped": skipped,
        }

    def _cmd_gripper_token(self, token: str) -> dict:
        """Explicit, IDEMPOTENT GRASP / RELEASE (the inherited /api/gripper toggles blindly).

        An explicit token that is already satisfied does nothing at all: not executed,
        not recorded, not added to the history -- so pressing G twice can never drop what
        the first G caught. A real attempt still goes through the base toggle, so all of
        ITS bookkeeping (empty-grasp auto-reopen, holding / picked / placed counters, the
        next-toggle sync) stays in exactly one place.
        """
        closed = bool(self.controller.gripper_closed)
        if (token == GRASP_ATOM and closed) or (token == RELEASE_ATOM and not closed):
            self._sync_gripper_toggle()
            self.last_message = (
                f"{token} ignored: gripper already {'CLOSED' if closed else 'OPEN'} (no-op)"
            )
            return {
                "ok": True,
                "noop": True,
                "message": self.last_message,
                "gripper_closed": closed,
            }
        self._next_gripper = token
        return self._cmd_gripper()

    def _cmd_move(self, token: str) -> dict:
        if token not in self.MOVE_TOKENS:
            return {"ok": False, "status": 400, "message": f"Invalid action token: {token}"}
        self._record_and_step(token, "rotate" if token in ROTATE_ATOMS else "move")
        self.last_message = f"Executed {token}"
        return {"ok": True, "message": self.last_message}

    def _cmd_set_task(self, text: str) -> dict:
        """Retarget the task string; refused mid-rollout so metadata stays truthful."""
        if self.recorder.active:
            return {
                "ok": False,
                "status": 409,
                "message": "Cannot change the task while recording",
            }
        text = str(text or "").strip()
        if not text:
            return {"ok": False, "status": 400, "message": "Task text must not be empty"}
        self.task_text = text
        self.recorder.session_meta["task"] = text
        self.last_message = f"Task set: {text}"
        return {"ok": True, "message": self.last_message, "task_text": text}

    def _cmd_gripper(self) -> dict:
        token = self._next_gripper
        was_holding = self.holding
        self._record_and_step(token, "gripper")
        closed = bool(self.controller.gripper_closed)
        # Track the ACTUAL state (empty grasps auto-reopen), same as the pygame collector.
        self._next_gripper = RELEASE_ATOM if closed else GRASP_ATOM
        if token == GRASP_ATOM and closed:
            self.holding = True
            if self.recorder.active:
                self.picked_count += 1
            self.last_message = "Grasped ✊"
        elif token == GRASP_ATOM:
            self.last_message = "Empty grasp — auto-released"
        elif token == RELEASE_ATOM:
            self.holding = False
            if was_holding and self.recorder.active:
                self.placed_count += 1
            self.last_message = "Released ✋"
        return {"ok": True, "message": self.last_message, "gripper_closed": closed}

    def _task_done(self) -> bool:
        if self.scene is not None:
            return bool(self.scene.task_success())
        return self.placed_count >= 1

    def _cmd_start(self) -> dict:
        if self.recorder.active:
            return {"ok": False, "status": 409, "message": "Already recording"}
        self.picked_count = 0
        self.placed_count = 0
        self.recorder.session_meta["task"] = self.task_text
        self.recorder.session_meta["teleop"] = "web"
        if self.scene is not None:
            self.recorder.session_meta.update(self.scene.to_meta())
        rollout_dir = self.recorder.start()
        self.last_message = f"● Recording {rollout_dir.name}"
        return {"ok": True, "message": self.last_message, "rollout": rollout_dir.name}

    def _cmd_stop(self) -> dict:
        if not self.recorder.active:
            return {"ok": False, "status": 409, "message": "Not recording"}
        if self.require_task_done and not self._task_done():
            hint = (
                "place the orange cube on the blue plate" if self.scene is not None
                else "pick the object up and set it down"
            )
            self.last_message = (
                f"Task not complete — {hint} first (or Cancel to discard)"
            )
            return {"ok": False, "status": 409, "message": self.last_message}
        num_steps = len(self.recorder.steps)
        self.recorder.session_meta["success"] = True
        done_dir = self.recorder.stop()
        self.rollouts_saved += 1
        self._home_and_resync()
        if self.scene is not None:
            self.scene.reset()
        self._capture()
        self.last_message = f"✔ Saved {done_dir.name} ({num_steps} steps); arm homed"
        return {"ok": True, "message": self.last_message, "saved": str(done_dir), "steps": num_steps}

    def _cmd_cancel(self) -> dict:
        if not self.recorder.active:
            self._home_and_resync()
            if self.scene is not None:
                self.scene.reset()
            self._capture()
            self.last_message = "Arm homed (no recording in progress)"
            return {"ok": True, "message": self.last_message}
        done_dir = self.recorder.stop()
        if done_dir is not None:
            shutil.rmtree(done_dir, ignore_errors=True)
        self._home_and_resync()
        if self.scene is not None:
            self.scene.reset()
        self._capture()
        discarded = done_dir.name if done_dir else "rollout"
        self.last_message = f"✖ Discarded {discarded}; arm homed"
        return {"ok": True, "message": self.last_message}

    def _home_and_resync(self) -> None:
        """Home the arm, then re-sync the controller setpoint (mirrors reset_to_home)."""
        self.picked_count = 0
        self.placed_count = 0
        self.holding = False
        if self.home_fn is None:
            return
        try:
            self.home_fn()
            self.controller.sync_from_robot()
            self._sync_gripper_toggle()
        except Exception as exc:  # noqa: BLE001 - a reset error must not kill the session
            self.last_message = f"Home failed: {exc}"
            print(f"[web-teleop] reset ERROR: {exc}")

    def _sync_gripper_toggle(self) -> None:
        self._next_gripper = RELEASE_ATOM if self.controller.gripper_closed else GRASP_ATOM


def _count(results: list, executed: bool) -> int:
    """Count ok results that did (or did not) actually touch the robot."""
    return sum(1 for r in results if r.get("ok") and bool(r.get("noop")) is not executed)


# ---------------------------------------------------------------------------
# Request / command parsing
# (no robot access, so it runs on the HTTP thread -- and is unit-testable standalone)
# ---------------------------------------------------------------------------
def expand_tokens(parts: list) -> tuple:
    """Resolve aliases and ``*N`` repeats in a list of raw words -> ``(tokens, error)``.

    ``MV_FWD*3`` -> three MV_FWD; ``w`` -> MV_FWD; ``g`` -> GRASP. Case-insensitive.
    ``error`` is ``(status, message)`` or None. Nothing is checked against the token
    vocabulary here -- that happens once, centrally, in :func:`parse_step_request`.
    """
    out: list = []
    for part in parts:
        word = part.strip().upper()
        if not word:
            continue
        m = _TOKEN_RE.match(word)
        if not m:
            return None, (400, f"cannot parse '{part}'; expected NAME or NAME*N (e.g. MV_FWD*3)")
        name, count_s = m.group(1), m.group(2)
        name = KEY_ALIASES.get(name, name)
        count = int(count_s) if count_s else 1
        if count < 1 or count > MAX_REPEAT:
            return None, (400, f"repeat count in '{part}' must be 1..{MAX_REPEAT}")
        out.extend([name] * count)
    return out, None


def _split_words(value: Any) -> Optional[list]:
    """Flatten a token string / list of token strings into raw words (comma or space sep)."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(",", " ").split()
    if isinstance(value, (list, tuple)):
        out: list = []
        for v in value:
            if not isinstance(v, str):
                return None
            out.extend(v.replace(",", " ").split())
        return out
    return None


def _as_token_list(value: Any) -> tuple:
    """``_split_words`` + alias/repeat expansion -> ``(tokens, error)``."""
    words = _split_words(value)
    if words is None:
        return None, (400, "expected a token string or a list of token strings")
    return expand_tokens(words)


def parse_command_text(text: str) -> tuple:
    """Parse one command-box line -> ``(body, error)``, where ``body`` is a /api/step body.

    Grammar (case-insensitive, comma or whitespace separated):

        MV_FWD MV_FWD GRASP          full token names
        MV_FWD*3 MV_LEFT GRASP       repeat syntax
        w*3 a g                      single-letter aliases (i.e. the keyboard keys)
        L:MV_FWD*2 R:STAY            the dual-arm commit shape (right must be STAY here)

    Returns ``{"tokens": [...]}`` or ``{"left": [...], "right": [...]}``. The arm rules and
    the token vocabulary are enforced afterwards by :func:`parse_step_request`, so there
    is exactly one gate no matter which entry point was used.
    """
    words = (text or "").replace(",", " ").split()
    if not words:
        return None, (400, "empty command")

    if not any(_ARM_RE.match(w.upper()) for w in words):
        tokens, err = expand_tokens(words)
        if err is not None:
            return None, err
        return {"tokens": tokens}, None

    sides: dict = {"L": [], "R": []}
    side: Optional[str] = None
    for word in words:
        m = _ARM_RE.match(word.upper())
        if m:
            side, rest = m.group(1), m.group(2)
            if rest:
                sides[side].append(rest)
            continue
        if side is None:
            return None, (
                400,
                f"'{word}' comes before any arm prefix; write e.g. 'L:MV_FWD R:STAY'",
            )
        sides[side].append(word)

    left, err = expand_tokens(sides["L"])
    if err is not None:
        return None, err
    right, err = expand_tokens(sides["R"])
    if err is not None:
        return None, err
    return {"left": left, "right": right}, None


def parse_step_request(body: dict, backend: TeleopBackend) -> tuple:
    """Normalize a /api/step body to ``(tokens, error)``.

    Accepted shapes (all case-insensitive, whitespace/comma separated; ``*N`` repeats and
    single-letter aliases are allowed everywhere):
        {"token":   "MV_FWD"}
        {"tokens":  ["MV_FWD", "MV_FWD", "GRASP"]}   or  {"tokens": "MV_FWD*2 GRASP"}
        {"command": "w*3 a g"}                      -- one raw command-box line
        {"left":    "MV_FWD", "right": "STAY"}      -- the dual-arm commit shape
    ``error`` is ``(status, message)`` or None.
    """
    if not isinstance(body, dict):
        return None, (400, "body must be a JSON object")

    if "command" in body:
        if any(k in body for k in ("token", "tokens", "left", "right")):
            return None, (400, "use either 'command' or {token|tokens|left,right}, not both")
        if not isinstance(body["command"], str):
            return None, (400, "'command' must be a string")
        parsed, err = parse_command_text(body["command"])
        if err is not None:
            return None, err
        body = parsed

    has_arm_shape = "left" in body or "right" in body
    if has_arm_shape and ("token" in body or "tokens" in body):
        return None, (400, "use either {token|tokens} or {left,right}, not both")

    if has_arm_shape:
        right, err = _as_token_list(body.get("right"))
        if err is not None:
            return None, (err[0], f"'right': {err[1]}")
        if not backend.dual_arm and any(t != STAY_TOKEN for t in right):
            return None, (
                400,
                "right arm not supported in single-arm mode: this rig has one arm, so "
                "'right' may only be omitted, null or 'STAY'. The {left,right} commit "
                "shape is reserved for the dual-arm build (collect_rollouts_web_dual.py).",
            )
        tokens, err = _as_token_list(body.get("left"))
        if err is not None:
            return None, (err[0], f"'left': {err[1]}")
    else:
        raw = body.get("tokens") if "tokens" in body else body.get("token")
        tokens, err = _as_token_list(raw)
        if err is not None:
            return None, (err[0], f"'tokens': {err[1]}")

    if not tokens:
        return None, (400, "no tokens given; expected e.g. {\"tokens\": [\"MV_FWD\", \"GRASP\"]}")

    allowed = set(backend.STEP_TOKENS) | {STAY_TOKEN}
    bad = [t for t in tokens if t not in allowed]
    if bad:
        aliases = ", ".join(f"{k}={v}" for k, v in KEY_ALIASES.items())
        return None, (
            400,
            f"illegal action token(s): {', '.join(bad)}; allowed: "
            f"{', '.join(list(backend.STEP_TOKENS) + [STAY_TOKEN])} (aliases: {aliases})",
        )
    return tokens, None
