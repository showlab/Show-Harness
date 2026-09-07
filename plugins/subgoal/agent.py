"""SubgoalPlanner VLM role: task + image -> raw subgoal-plan JSON.

The single VLM call for the subgoal capability. It is self-contained: it owns its
prompt file (``subgoal_planner.txt``, loaded by default) and its output schema, and
talks to a duck-typed VLM ``client`` (``complete_json``) rather than a concrete client
class, so the capability stays portable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Co-located prompt file: the capability owns its prompt rather than relying on the
# top-level prompts/ directory.
PROMPT_PATH = Path(__file__).with_name("subgoal_planner.txt")

# No-think regime: the planner emits the plan JSON directly (no chain-of-thought). With
# thinking on, a busy scene sends the model into a monologue that overruns the budget
# before any JSON appears.
NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}


@dataclass
class PlannerResponse:
    token: str
    raw_text: str
    payload: dict[str, Any]


# Output contract: an ordered list of subgoals, each fully specified. additionalProperties
# is forbidden and every field required so guided/structured decoding stays strict.
SUBGOAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "target": {"type": "string"},
                    "affordance": {"type": "string"},
                    "motion": {"type": "string"},
                    "description": {"type": "string"},
                    "completion": {"type": "string"},
                },
                "required": [
                    "id",
                    "target",
                    "affordance",
                    "motion",
                    "description",
                    "completion",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}


def load_prompt() -> str:
    """Return the co-located subgoal-planner prompt template (with ``{task}``)."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _join_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


class SubgoalPlannerAgent:
    """One VLM call that turns a task + image(s) into a subgoal-plan JSON object."""

    def __init__(
        self,
        client: Any,
        common_context: str,
        prompt_template: str | None = None,
        video_ref_block: str = "",
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.common_context = common_context
        # Defaults to the co-located prompt; pass prompt_template to override.
        self.prompt_template = prompt_template if prompt_template is not None else load_prompt()
        # Reference-demo block from plugins.video_ref ("" -> the {video_ref} placeholder
        # renders empty and the prompt is unchanged). Riding on the agent means the
        # initial plan AND every replan replicate the same demonstration.
        self.video_ref_block = str(video_ref_block or "")
        # Output budget for the plan JSON. 2048 proved too small the moment a plan
        # grew past ~10 stages on a pretty-printing hosted backend (rollout
        # Observed: four attempts all returned VALID fenced JSON, all cut off
        # mid-stage -> silent task_fallback). Config: planner_max_tokens.
        self.max_tokens = int(max_tokens)
        self._last_prompt = ""
        self._last_diagnostics: dict[str, Any] = {}

    def plan(
        self,
        task: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
        image_roles: "list[str] | None" = None,
    ):
        started = time.monotonic()
        prompt = _join_prompt_parts(
            self.common_context,
            _image_roles_block(image_roles),
            self.prompt_template.format(task=task, video_ref=self.video_ref_block),
        )
        self._last_prompt = prompt
        errors: list[str] = []
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=SUBGOAL_PLAN_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            response = _require_plan_response(response)
            return self._finish(response, "guided_json", errors, image_roles, started)
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"guided_json: {_one_line(str(exc))}")
            # Fallback drops guided_json: if the server's guided decoder is choking on
            # the nested schema (it returned empty content), free generation with the
            # explicit JSON format from the prompt still yields a parseable plan.
            strict_prompt = (
                prompt
                + "\n\nReturn only the JSON object in the format shown above. "
                "Do not include thought, reasoning, markdown, or prose."
            )
        try:
            response = self.client.complete_json(
                strict_prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=None,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            response = _require_plan_response(response)
            return self._finish(response, "free_json", errors, image_roles, started)
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"free_json: {_one_line(str(exc))}")

        if hasattr(self.client, "complete_text"):
            # Text retries preserve the ENTIRE rendered prompt (image-role block,
            # demo brief included). Dropping to a task-only generic prompt has made
            # a failed plan look like a successful pick-and-place plan.
            text_prompt = _text_retry_prompt(prompt)
            for label, chat_kwargs in (
                ("text_no_think", NO_THINK_CHAT_TEMPLATE_KWARGS),
                ("text_default", None),
            ):
                try:
                    response = self.client.complete_text(
                        text_prompt,
                        agentview_image,
                        wrist_image=wrist_image,
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                        chat_template_kwargs=chat_kwargs,
                        debug=debug,
                        strip_reasoning=False,
                    )
                    parsed = _require_plan_json(_parse_json_object(response.raw_text))
                    response = _json_response(
                        parsed, response, source=label, errors=errors
                    )
                    return self._finish(response, label, errors, image_roles, started)
                except RuntimeError as exc:
                    if "VLM returned" not in str(exc):
                        raise
                    errors.append(f"{label}: {_one_line(str(exc))}")

        response = _fallback_response(task, errors)
        return self._finish(response, "task_fallback", errors, image_roles, started)

    def _finish(
        self,
        response: Any,
        route: str,
        errors: "list[str]",
        image_roles: "list[str] | None",
        started: float,
    ) -> PlannerResponse:
        payload = dict(getattr(response, "payload", {}) or {})
        payload["planner_retry"] = route
        if errors:
            payload["planner_errors"] = list(errors)
        finished = PlannerResponse(
            token=str(getattr(response, "token", "")),
            raw_text=str(getattr(response, "raw_text", "")),
            payload=payload,
        )
        self._last_diagnostics = {
            "route": route,
            "errors": list(errors),
            "latency_s": payload.get("latency_s"),
            "total_latency_s": round(time.monotonic() - started, 3),
            "response_chars": len(finished.raw_text),
            "max_tokens": self.max_tokens,
            "image_roles": list(image_roles or []),
        }
        return finished

    def last_prompt(self) -> str:
        return self._last_prompt

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)


def _image_roles_block(image_roles: "list[str] | None") -> str:
    if not image_roles:
        return ""
    lines = ["IMAGE ORDER (same order as the attached images):"]
    lines.extend(f"{i}. {role}" for i, role in enumerate(image_roles, 1))
    return "\n".join(lines)


def _text_retry_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nRETRY FORMAT: Return exactly one JSON object matching the requested "
        "subgoals format. Preserve every task and scene-specific rule above. "
        "Do not include thought, reasoning, markdown, or prose."
    )


def _parse_json_object(raw_text: Any) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    decoder = json.JSONDecoder()
    for match_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[match_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise RuntimeError(f"VLM returned non-JSON text: {text!r}")


def _require_plan_response(response: Any) -> Any:
    payload = getattr(response, "payload", {}) or {}
    parsed = payload.get("json") if isinstance(payload, dict) else None
    _require_plan_json(parsed)
    return response


def _require_plan_json(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict) and _has_subgoals(parsed):
        return parsed
    raise RuntimeError(
        "VLM returned JSON without non-empty subgoals: "
        f"{json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed!r}"
    )


def _has_subgoals(parsed: dict[str, Any]) -> bool:
    items = parsed.get("subgoals")
    if isinstance(items, list):
        return any(isinstance(item, dict) for item in items)
    return isinstance(items, dict)


def _json_response(
    parsed: dict[str, Any],
    response: Any,
    *,
    source: str,
    errors: list[str],
) -> PlannerResponse:
    payload = dict(getattr(response, "payload", {}) or {})
    payload["json"] = parsed
    payload["planner_retry"] = source
    if errors:
        payload["planner_errors"] = list(errors)
    return PlannerResponse(
        token="",
        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        payload=payload,
    )


def _fallback_response(task: str, errors: list[str]) -> PlannerResponse:
    print(
        "[subgoal] WARNING: planner degraded to a single whole-task stage after "
        f"{len(errors)} failed attempts; first error: {errors[0][:160] if errors else '?'}"
    )
    parsed: dict[str, Any] = {
        "planner_fallback": "subgoal planner VLM returned empty/non-JSON after retries",
        "planner_errors": list(errors),
        "subgoals": [
            {
                "id": "task_fallback",
                "target": task,
                "affordance": "task-relevant visible object or region",
                "motion": "TASK",
                "description": f"Complete the task directly: {task}",
                "completion": f"the task is visibly complete: {task}",
            }
        ],
    }
    return PlannerResponse(
        token="",
        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        payload={"json": parsed, "planner_fallback": True},
    )


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())
