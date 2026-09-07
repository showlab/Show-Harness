"""DualSubgoalPlanner VLM role: task + three images -> per-arm subgoal tracks.

The dual-arm counterpart of :mod:`plugins.subgoal.agent`. One VLM call sees the front
view plus BOTH wrist views and returns ``{"left": [...], "right": [...]}`` -- one
ordered subgoal track per arm, planned to run CONCURRENTLY (parallel planning is the
planner's job; per-step coordination is the dual controller's). Each item uses the
same six fields as the single-arm plan, so :class:`core.v0_types.Subgoal` parses both.

Self-contained like the single-arm agent: owns its prompt (``subgoal_planner_dual.txt``)
and schema, and talks to a duck-typed VLM ``client``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import (
    NO_THINK_CHAT_TEMPLATE_KWARGS,
    PlannerResponse,
    _join_prompt_parts,
    _one_line,
    _parse_json_object,
)

DUAL_PROMPT_PATH = Path(__file__).with_name("subgoal_planner_dual.txt")

_SUBGOAL_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "target": {"type": "string"},
        "affordance": {"type": "string"},
        "motion": {"type": "string"},
        "description": {"type": "string"},
        "completion": {"type": "string"},
    },
    "required": ["id", "target", "affordance", "motion", "description", "completion"],
    "additionalProperties": False,
}

# One ordered track per arm; an empty track means that arm has no work. minItems 0 on
# purpose -- forcing a fake stage onto an idle arm would just make the controller
# hallucinate work for it.
DUAL_SUBGOAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "left": {"type": "array", "items": _SUBGOAL_ITEM_SCHEMA},
        "right": {"type": "array", "items": _SUBGOAL_ITEM_SCHEMA},
    },
    "required": ["left", "right"],
    "additionalProperties": False,
}


def load_dual_prompt() -> str:
    """Return the co-located dual-planner prompt template (with ``{task}``)."""
    return DUAL_PROMPT_PATH.read_text(encoding="utf-8").strip()


class DualSubgoalPlannerAgent:
    """One VLM call that turns a task + three images into per-arm subgoal tracks."""

    def __init__(
        self,
        client: Any,
        common_context: str,
        prompt_template: str | None = None,
        video_ref_block: str = "",
    ) -> None:
        self.client = client
        self.common_context = common_context
        self.prompt_template = (
            prompt_template if prompt_template is not None else load_dual_prompt()
        )
        # Reference-demo block from plugins.video_ref ("" -> the {video_ref} placeholder
        # renders empty and the prompt is the ordinary free plan). Mounted once, so
        # every replan through this agent replicates the same demo.
        self.video_ref_block = str(video_ref_block or "")

    def plan(
        self,
        task: str,
        agentview_image,
        wrist_left_image=None,
        wrist_right_image=None,
        debug: bool = False,
    ):
        prompt = _join_prompt_parts(
            self.common_context,
            self.prompt_template.format(task=task, video_ref=self.video_ref_block),
        )
        wrists = [wrist_left_image, wrist_right_image]
        errors: list[str] = []
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrists,
                schema=DUAL_SUBGOAL_PLAN_SCHEMA,
                max_tokens=4096,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            return _require_dual_plan_response(response)
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"guided_json: {_one_line(str(exc))}")
            # Same fallback ladder as the single-arm agent: drop guided_json first.
            strict_prompt = (
                prompt
                + "\n\nReturn only the JSON object in the format shown above. "
                "Do not include thought, reasoning, markdown, or prose."
            )
        try:
            response = self.client.complete_json(
                strict_prompt,
                agentview_image,
                wrist_image=wrists,
                schema=None,
                max_tokens=4096,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            return _require_dual_plan_response(response)
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"free_json: {_one_line(str(exc))}")

        if hasattr(self.client, "complete_text"):
            for label, chat_kwargs in (
                ("text_no_think", NO_THINK_CHAT_TEMPLATE_KWARGS),
                ("text_default", None),
            ):
                try:
                    response = self.client.complete_text(
                        prompt,
                        agentview_image,
                        wrist_image=wrists,
                        max_tokens=4096,
                        temperature=0.0,
                        chat_template_kwargs=chat_kwargs,
                        debug=debug,
                        strip_reasoning=False,
                    )
                    parsed = _require_dual_plan_json(
                        _parse_json_object(response.raw_text)
                    )
                    payload = dict(getattr(response, "payload", {}) or {})
                    payload["json"] = parsed
                    payload["planner_retry"] = label
                    payload["planner_errors"] = list(errors)
                    return PlannerResponse(
                        token="",
                        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                        payload=payload,
                    )
                except RuntimeError as exc:
                    if "VLM returned" not in str(exc):
                        raise
                    errors.append(f"{label}: {_one_line(str(exc))}")

        return _dual_fallback_response(task, errors)


def _require_dual_plan_response(response: Any) -> Any:
    payload = getattr(response, "payload", {}) or {}
    parsed = payload.get("json") if isinstance(payload, dict) else None
    _require_dual_plan_json(parsed)
    return response


def _require_dual_plan_json(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        left = parsed.get("left")
        right = parsed.get("right")
        both_lists = isinstance(left, list) and isinstance(right, list)
        if both_lists and (left or right):
            return parsed
    raise RuntimeError(
        "VLM returned JSON without non-empty left/right subgoal tracks: "
        f"{json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed!r}"
    )


def _dual_fallback_response(task: str, errors: list[str]) -> PlannerResponse:
    """Degraded plan when every retry failed: one whole-task stage PER ARM. The dual
    controller can still keep an arm STILL if the live scene shows it has no work."""
    stage = {
        "id": "task_fallback",
        "target": task,
        "affordance": "task-relevant visible object or region",
        "motion": "TASK",
        "description": f"Complete this arm's share of the task: {task}",
        "completion": f"the task is visibly complete: {task}",
    }
    parsed: dict[str, Any] = {
        "planner_fallback": "dual subgoal planner VLM returned empty/non-JSON after retries",
        "planner_errors": list(errors),
        "left": [dict(stage)],
        "right": [dict(stage)],
    }
    return PlannerResponse(
        token="",
        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        payload={"json": parsed, "planner_fallback": True},
    )
