"""Closed-loop GPT VLM operator for the single- and dual-arm web teleop APIs.

The Claude browser extension used screenshots and UI clicks.  This operator keeps
the same visual feedback loop, but talks to the web teleop's HTTP API directly:

    state + fresh camera JPEGs -> one structured GPT decision -> /api/step

Direct API control removes browser focus/click ambiguity while preserving the web
page as a human control surface.  The operator starts paused and rejects stale,
low-confidence, collision-prone, malformed, and oscillating decisions before an
action can reach the robot.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from core.vlm.vlm_client import VLMClient, VLMParseError

ACTIONS: Tuple[str, ...] = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
    "ROTATE_CCW",
    "ROTATE_CW",
    "GRASP",
    "RELEASE",
    "STILL",
)
PHASES: Tuple[str, ...] = (
    "search",
    "approach",
    "align",
    "descend",
    "grasp",
    "lift",
    "transport",
    "place",
    "verify",
    "recover",
)
SINGLE_REPEAT_ACTIONS = frozenset(
    ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP")
)
INVERSE_ACTION = {
    "MV_FWD": "MV_BACK",
    "MV_BACK": "MV_FWD",
    "MV_LEFT": "MV_RIGHT",
    "MV_RIGHT": "MV_LEFT",
    "MV_UP": "MV_DOWN",
    "MV_DOWN": "MV_UP",
    "ROTATE_CCW": "ROTATE_CW",
    "ROTATE_CW": "ROTATE_CCW",
    "STILL": "STILL",
}

ARM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "repeat": {"type": "integer", "minimum": 1, "maximum": 3},
    },
    "required": ["action", "repeat"],
    "additionalProperties": False,
}
DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "phase": {"type": "string", "enum": list(PHASES)},
        "evidence": {"type": "string"},
        "next_goal": {"type": "string"},
        "left": ARM_SCHEMA,
        "right": ARM_SCHEMA,
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "finish": {"type": "boolean"},
        "pause": {"type": "boolean"},
    },
    "required": [
        "phase",
        "evidence",
        "next_goal",
        "left",
        "right",
        "confidence",
        "finish",
        "pause",
    ],
    "additionalProperties": False,
}


class DecisionError(ValueError):
    """A model decision failed an operator-side safety or schema check."""


@dataclass
class OperatorConfig:
    target_url: str
    interval_s: float = 0.25
    request_timeout_s: float = 30.0
    confidence_threshold: float = 0.55
    max_repeat: int = 3
    max_steps: int = 150
    max_output_tokens: int = 512
    image_max_side: int = 768
    image_detail: str = "high"
    auto_record: bool = True
    auto_save: bool = True
    dry_run: bool = False
    trace_root: Path = Path("data/gpt_operator_traces")

    def __post_init__(self) -> None:
        self.target_url = self.target_url.rstrip("/")
        self.interval_s = max(0.0, float(self.interval_s))
        self.request_timeout_s = max(1.0, float(self.request_timeout_s))
        self.confidence_threshold = min(1.0, max(0.0, float(self.confidence_threshold)))
        self.max_repeat = min(3, max(1, int(self.max_repeat)))
        self.max_steps = max(1, int(self.max_steps))
        self.max_output_tokens = max(128, int(self.max_output_tokens))
        self.image_max_side = max(256, int(self.image_max_side))
        self.trace_root = Path(self.trace_root)


class TeleopHTTPClient:
    """Small, persistent HTTP client for either web teleop server."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def state(self) -> dict:
        response = self.session.get(
            self.base_url + "/api/state", timeout=self.timeout_s
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("teleop /api/state did not return a JSON object")
        return data

    def post(self, path: str, body: Optional[dict] = None) -> dict:
        response = self.session.post(
            self.base_url + path,
            json=body or {},
            timeout=max(self.timeout_s, 60.0),
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"teleop {path} returned HTTP {response.status_code} with non-JSON body"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"teleop {path} did not return a JSON object")
        data.setdefault("ok", response.status_code < 400)
        data["http_status"] = response.status_code
        return data

    def snapshot_bytes(self, name: str) -> bytes:
        # Use a request-local connection: the dashboard can proxy live frames while
        # the control thread fetches a synchronized observation in parallel.
        response = requests.get(
            self.base_url + "/snapshot/" + name,
            timeout=self.timeout_s,
            headers={"Cache-Control": "no-cache"},
        )
        response.raise_for_status()
        return response.content

    def snapshots(self, dual_arm: bool) -> Tuple[Dict[str, bytes], Dict[str, np.ndarray]]:
        names = (
            ("agentview", "wrist_left", "wrist_right")
            if dual_arm
            else ("agentview", "wrist")
        )

        def fetch(name: str) -> Tuple[str, bytes, np.ndarray]:
            raw = self.snapshot_bytes(name)
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"teleop snapshot {name!r} was not a valid JPEG")
            return name, raw, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # The views describe the same instant closely enough for this controller, and
        # parallel downloads avoid adding one network round trip per camera.
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            rows = list(pool.map(fetch, names))
        return (
            {name: raw for name, raw, _ in rows},
            {name: image for name, _, image in rows},
        )


def state_marker(state: dict) -> Tuple[Any, ...]:
    """Fields that must not change while GPT is deciding from an observation."""
    if state.get("dual_arm"):
        progress = state.get("total_pairs")
    else:
        progress = state.get("steps")
    return (
        bool(state.get("recording")),
        progress,
        json.dumps(state.get("gripper_closed"), sort_keys=True),
        json.dumps(state.get("holding"), sort_keys=True),
    )


def compact_model_state(state: dict) -> dict:
    """Expose operational feedback, not simulator ground truth, to the model."""
    keys = (
        "recording",
        "sim",
        "steps",
        "total_pairs",
        "task_text",
        "last_token",
        "last",
        "gripper_closed",
        "holding",
        "picked",
        "placed",
        "task_done",
        "can_stop",
        "ee_pos",
        "gap_m",
        "gap_warn",
        "step_m",
        "yaw_step_rad",
        "history",
        "message",
    )
    return {key: state.get(key) for key in keys if key in state}


def _arm_decision(value: Any, label: str, max_repeat: int) -> dict:
    if not isinstance(value, dict):
        raise DecisionError(f"{label} must be an object")
    action = str(value.get("action", "")).upper().strip()
    if action not in ACTIONS:
        raise DecisionError(f"{label}.action is invalid: {action!r}")
    try:
        repeat = int(value.get("repeat", 1))
    except (TypeError, ValueError) as exc:
        raise DecisionError(f"{label}.repeat must be an integer") from exc
    if repeat < 1 or repeat > max_repeat:
        raise DecisionError(f"{label}.repeat must be between 1 and {max_repeat}")
    # Closed-loop precision: only clear translational travel may be chunked.  Down,
    # rotation and gripper commands always need another image before repeating.
    if action not in SINGLE_REPEAT_ACTIONS:
        repeat = 1
    if action == "STILL":
        repeat = 1
    return {"action": action, "repeat": repeat}


def normalize_decision(
    raw: Any,
    *,
    dual_arm: bool,
    state: dict,
    confidence_threshold: float,
    max_repeat: int,
) -> dict:
    """Validate and normalize a model decision before it can reach /api/step."""
    if not isinstance(raw, dict):
        raise DecisionError("model output must be a JSON object")
    phase = str(raw.get("phase", "")).strip().lower()
    if phase not in PHASES:
        raise DecisionError(f"invalid phase: {phase!r}")
    evidence = " ".join(str(raw.get("evidence", "")).split())
    next_goal = " ".join(str(raw.get("next_goal", "")).split())
    if not evidence or not next_goal:
        raise DecisionError("evidence and next_goal must be non-empty")
    evidence = evidence[:500]
    next_goal = next_goal[:300]
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise DecisionError("confidence must be a number") from exc
    if not 0.0 <= confidence <= 1.0:
        raise DecisionError("confidence must be between 0 and 1")

    left = _arm_decision(raw.get("left"), "left", max_repeat)
    right = _arm_decision(raw.get("right"), "right", max_repeat)
    finish = bool(raw.get("finish", False))
    pause = bool(raw.get("pause", False))

    if not dual_arm and right["action"] != "STILL":
        raise DecisionError("single-arm mode requires right.action=STILL")
    moving = [arm for arm in (left, right) if arm["action"] != "STILL"]
    if not finish and not pause and not moving:
        raise DecisionError("a non-finish decision must move at least one arm")
    if finish or pause:
        left = {"action": "STILL", "repeat": 1}
        right = {"action": "STILL", "repeat": 1}
    if dual_arm and state.get("gap_warn") and len(moving) > 1:
        raise DecisionError(
            "both arms were commanded while gap_warn is active; only one may move"
        )
    if confidence < confidence_threshold and not finish:
        pause = True
        left = {"action": "STILL", "repeat": 1}
        right = {"action": "STILL", "repeat": 1}

    return {
        "phase": phase,
        "evidence": evidence,
        "next_goal": next_goal,
        "left": left,
        "right": right,
        "confidence": round(confidence, 3),
        "finish": finish,
        "pause": pause,
    }


def action_body(decision: dict, dual_arm: bool) -> dict:
    def repeated(arm: dict) -> List[str]:
        return [arm["action"]] * int(arm["repeat"])

    if dual_arm:
        return {"left": repeated(decision["left"]), "right": repeated(decision["right"])}
    return {"tokens": repeated(decision["left"])}


def _resize_for_vlm(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return np.ascontiguousarray(image)
    scale = float(max_side) / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceWriter:
    """Append-only JSONL trace plus the exact JPEG observations used per cycle."""

    def __init__(self, root: Path, prompt: str, model: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root) / f"{stamp}_{model.replace('/', '_')}"
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.events_path = self.session_dir / "events.jsonl"
        self._lock = threading.Lock()

    def write(self, event: dict, frames: Optional[Dict[str, bytes]] = None) -> None:
        row = dict(event)
        if frames:
            saved = {}
            index = int(row.get("cycle", 0))
            for name, data in frames.items():
                rel = Path("frames") / f"{index:05d}_{name}.jpg"
                (self.session_dir / rel).write_bytes(data)
                saved[name] = str(rel)
            row["frames"] = saved
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)


class GPTWebOperator:
    """Background GPT observation/action loop with a thread-safe dashboard state."""

    def __init__(
        self,
        target: TeleopHTTPClient,
        vlm: VLMClient,
        prompt: str,
        config: OperatorConfig,
    ) -> None:
        self.target = target
        self.vlm = vlm
        self.prompt = prompt.strip()
        self.config = config
        self.trace_writer = TraceWriter(config.trace_root, self.prompt, vlm.model)

        self._cond = threading.Condition()
        self._shutdown = False
        self._mode = "paused"  # paused | continuous
        self._pending_once = 0
        self._abort_before_action = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="gpt-web-operator", daemon=True
        )
        self._worker_started = False

        self.status_text = "Paused — review the views, then press Run or Step once"
        self.busy = False
        self.cycle = 0
        self.executed_steps = 0
        self.last_error: Optional[str] = None
        self.last_latency_s: Optional[float] = None
        self.last_decision: Optional[dict] = None
        self.last_result: Optional[dict] = None
        self.target_state: dict = {}
        self.frame_bytes: Dict[str, bytes] = {}
        self.events: Deque[dict] = deque(maxlen=80)
        self.recent_actions: Deque[Tuple[str, str]] = deque(maxlen=8)

    def _complete_decision(
        self, prompt: str, vlm_images: Sequence[np.ndarray]
    ) -> Tuple[Any, int, Optional[str]]:
        """Request one decision, retrying one truncated/non-JSON answer.

        The normal 512-token budget keeps the control loop responsive. A reasoning
        model can occasionally spend that entire budget before emitting its JSON;
        retry exactly once with a larger cap instead of abandoning a valid rollout.
        Authentication, HTTP, and safety errors are not hidden by this retry.
        """
        kwargs = {
            "prompt": prompt,
            "agentview_image": vlm_images[0],
            "wrist_image": list(vlm_images[1:]),
            "schema": DECISION_SCHEMA,
            "max_tokens": self.config.max_output_tokens,
            "image_detail": self.config.image_detail,
        }
        try:
            return self.vlm.complete_json(**kwargs), 1, None
        except VLMParseError as exc:
            retry_reason = " ".join(str(exc).split())[:300]
            self.status_text = "GPT output was incomplete; retrying once"
            kwargs["max_tokens"] = max(1024, self.config.max_output_tokens * 2)
            return self.vlm.complete_json(**kwargs), 2, retry_reason

    # -- lifecycle / controls -------------------------------------------------
    def start_worker(self) -> None:
        with self._cond:
            if self._worker_started:
                return
            self._worker_started = True
            self._worker.start()

    def shutdown(self) -> None:
        self._abort_before_action.set()
        with self._cond:
            self._shutdown = True
            self._mode = "paused"
            self._pending_once = 0
            self._cond.notify_all()
        if self._worker_started:
            self._worker.join(timeout=5.0)

    def resume(self) -> None:
        self.start_worker()
        with self._cond:
            self._abort_before_action.clear()
            self._mode = "continuous"
            self.status_text = "Running"
            self.last_error = None
            self._cond.notify_all()

    def pause(self, message: str = "Paused by operator") -> None:
        self._abort_before_action.set()
        with self._cond:
            self._mode = "paused"
            self._pending_once = 0
            self.status_text = message
            self._cond.notify_all()

    def step_once(self) -> None:
        self.start_worker()
        with self._cond:
            self._abort_before_action.clear()
            self._pending_once += 1
            self.status_text = "One GPT step queued"
            self.last_error = None
            self._cond.notify_all()

    def target_command(self, path: str, body: Optional[dict] = None) -> dict:
        """Pause automation, then run an explicit recording/task control command."""
        self.pause("Paused for manual target command")
        result = self.target.post(path, body)
        if isinstance(result.get("state"), dict):
            self.target_state = result["state"]
        self.last_result = result
        self.status_text = str(result.get("message") or path)
        return result

    # -- public monitor state -------------------------------------------------
    def snapshot(self) -> dict:
        with self._cond:
            mode = self._mode
            pending_once = self._pending_once
            return {
                "mode": mode,
                "busy": self.busy,
                "pending_once": pending_once,
                "status": self.status_text,
                "cycle": self.cycle,
                "executed_steps": self.executed_steps,
                "last_error": self.last_error,
                "last_latency_s": self.last_latency_s,
                "last_decision": self.last_decision,
                "last_result": self.last_result,
                "target_state": self.target_state,
                "views": list(self.frame_bytes),
                "events": list(self.events),
                "model": self.vlm.model,
                "backend": self.vlm.provider,
                "target_url": self.target.base_url,
                "target_ui_url": self.target.base_url + "/",
                "dry_run": self.config.dry_run,
                "auto_record": self.config.auto_record,
                "auto_save": self.config.auto_save,
                "confidence_threshold": self.config.confidence_threshold,
                "trace_dir": str(self.trace_writer.session_dir.resolve()),
                "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12],
            }

    def frame(self, name: str) -> Optional[bytes]:
        with self._cond:
            return self.frame_bytes.get(name)

    # -- worker ---------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._shutdown
                    or self._mode == "continuous"
                    or self._pending_once > 0
                )
                if self._shutdown:
                    return
                single = self._mode != "continuous"
                if single:
                    self._pending_once -= 1
                self.busy = True
            try:
                self.run_cycle()
            except Exception as exc:  # noqa: BLE001 - never kill the supervisor
                self._record_error(exc)
                self.pause(f"Paused after error: {exc}")
            finally:
                with self._cond:
                    self.busy = False
                    if single and self._pending_once <= 0 and self._mode != "continuous":
                        self.status_text = self.status_text or "Step complete"
            if not single and self.config.interval_s:
                with self._cond:
                    self._cond.wait(timeout=self.config.interval_s)

    def _record_error(self, exc: Exception) -> None:
        message = " ".join(str(exc).split())
        self.last_error = message
        event = {
            "cycle": self.cycle,
            "time": _utc_now(),
            "kind": "error",
            "message": message,
        }
        self.events.appendleft(event)
        self.trace_writer.write(event)

    def _event(self, event: dict, frames: Optional[Dict[str, bytes]] = None) -> None:
        self.events.appendleft(event)
        self.trace_writer.write(event, frames)

    def _pause_from_cycle(self, message: str) -> None:
        with self._cond:
            self._mode = "paused"
            self._pending_once = 0
            self.status_text = message

    def _prompt_for(self, state: dict, dual_arm: bool) -> str:
        recent = [
            {"left": left, "right": right}
            for left, right in list(self.recent_actions)[-6:]
        ]
        mode = (
            "DUAL ARM: Image A=front, Image B=left wrist, Image C=right wrist."
            if dual_arm
            else "SINGLE ARM: Image A=front, Image B=wrist. Use left for the active arm "
            "and always return right={action:STILL,repeat:1}."
        )
        dynamic = {
            "mode": mode,
            "state": compact_model_state(state),
            "recent_executed_actions": recent,
        }
        return self.prompt + "\n\nCURRENT INPUT (variable):\n" + json.dumps(
            dynamic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def _oscillates(self, decision: dict) -> bool:
        candidate = (
            decision["left"]["action"],
            decision["right"]["action"],
        )
        inverse = tuple(INVERSE_ACTION.get(action, action) for action in candidate)
        if candidate == inverse or len(self.recent_actions) < 4:
            return False
        tail = list(self.recent_actions)[-4:]
        return tail == [candidate, inverse, candidate, inverse]

    def run_cycle(self) -> dict:
        """Perform one observe-decide-(optionally)-act transaction."""
        self.cycle += 1
        cycle = self.cycle
        started = time.monotonic()
        state = self.target.state()
        dual_arm = bool(state.get("dual_arm"))
        self.target_state = state
        if self.executed_steps >= self.config.max_steps:
            raise RuntimeError(f"operator max_steps={self.config.max_steps} reached")

        if self.config.auto_record and not state.get("recording") and not self.config.dry_run:
            started_result = self.target.post("/api/start")
            if not started_result.get("ok"):
                raise RuntimeError(started_result.get("message") or "could not start recording")
            state = started_result.get("state") or self.target.state()
            self.target_state = state

        raw_frames, images = self.target.snapshots(dual_arm)
        with self._cond:
            self.frame_bytes = dict(raw_frames)

        order = (
            ("agentview", "wrist_left", "wrist_right")
            if dual_arm
            else ("agentview", "wrist")
        )
        vlm_images = [
            _resize_for_vlm(images[name], self.config.image_max_side) for name in order
        ]
        prompt = self._prompt_for(state, dual_arm)
        self.status_text = "GPT is analyzing the current camera views"
        model_started = time.monotonic()
        response, decision_attempts, retry_reason = self._complete_decision(
            prompt, vlm_images
        )
        # Include every model attempt in the operator-visible latency.
        self.last_latency_s = round(time.monotonic() - model_started, 3)
        decision = normalize_decision(
            response.payload.get("json"),
            dual_arm=dual_arm,
            state=state,
            confidence_threshold=self.config.confidence_threshold,
            max_repeat=self.config.max_repeat,
        )
        self.last_decision = decision

        event = {
            "cycle": cycle,
            "time": _utc_now(),
            "kind": "decision",
            "model": self.vlm.model,
            "latency_s": self.last_latency_s,
            "decision_attempts": decision_attempts,
            "wall_s": round(time.monotonic() - started, 3),
            "decision": decision,
            "state_before": compact_model_state(state),
        }
        if retry_reason:
            event["retry_reason"] = retry_reason

        if decision["pause"]:
            reason = (
                "GPT requested a pause"
                if decision["confidence"] >= self.config.confidence_threshold
                else "confidence below threshold"
            )
            event.update({"outcome": "paused", "message": reason})
            self._event(event, raw_frames)
            self._pause_from_cycle(f"Paused: {reason} — {decision['next_goal']}")
            return event

        if decision["finish"]:
            if not state.get("can_stop"):
                event.update(
                    {
                        "outcome": "finish_rejected",
                        "message": "GPT saw completion but the teleop completion gate is closed",
                    }
                )
                self._event(event, raw_frames)
                self._pause_from_cycle(
                    "Paused: GPT reported completion, but CAN_STOP is false"
                )
                return event
            if self.config.auto_save and not self.config.dry_run:
                result = self.target.post("/api/stop")
                self.last_result = result
                if not result.get("ok"):
                    raise RuntimeError(result.get("message") or "stop/save failed")
                event["result"] = result
                event["outcome"] = "saved"
                self.target_state = result.get("state") or self.target.state()
                message = str(result.get("message") or "rollout saved")
            else:
                event["outcome"] = "complete_not_saved"
                message = "GPT verified completion; auto-save is disabled"
            self._event(event, raw_frames)
            self._pause_from_cycle(message)
            return event

        if self._oscillates(decision):
            event.update(
                {
                    "outcome": "oscillation_rejected",
                    "message": "alternating inverse-action loop detected",
                }
            )
            self._event(event, raw_frames)
            self._pause_from_cycle("Paused: inverse-action oscillation detected")
            return event

        if self.config.dry_run:
            event["outcome"] = "dry_run"
            self._event(event, raw_frames)
            self.status_text = "Dry run: decision generated; no robot action sent"
            return event

        if self._abort_before_action.is_set():
            event["outcome"] = "discarded_by_pause"
            self._event(event, raw_frames)
            self.status_text = "Paused before action; GPT decision discarded"
            return event

        current = self.target.state()
        if state_marker(current) != state_marker(state):
            event.update(
                {
                    "outcome": "stale_observation",
                    "message": "teleop state changed while GPT was deciding; re-observe",
                }
            )
            self._event(event, raw_frames)
            self.status_text = "State changed during inference; discarded stale action"
            return event

        body = action_body(decision, dual_arm)
        self.status_text = "Executing " + json.dumps(body, separators=(",", ":"))
        result = self.target.post("/api/step", body)
        self.last_result = result
        if not result.get("ok"):
            raise RuntimeError(result.get("message") or "teleop action failed")
        self.executed_steps += int(result.get("executed") or 1)
        signature = (decision["left"]["action"], decision["right"]["action"])
        self.recent_actions.append(signature)
        self.target_state = result.get("state") or self.target.state()
        event.update(
            {
                "outcome": "executed",
                "request": body,
                "result": result,
                "state_after": compact_model_state(self.target_state),
                "wall_s": round(time.monotonic() - started, 3),
            }
        )
        self._event(event, raw_frames)
        self.status_text = str(result.get("message") or "Action executed")
        return event
