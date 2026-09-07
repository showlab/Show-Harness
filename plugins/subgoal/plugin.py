"""SubgoalPlanner: drive the VLM planner agent and post-process its plan.

This is the subgoal capability's primary module (the public tool, mirroring every other
``plugins/<name>/plugin.py``). It wraps a :class:`~plugins.subgoal.agent.SubgoalPlannerAgent`
call: parses the raw plan JSON into :class:`~core.v0_types.Subgoal` objects and applies
the pre-grasp merge heuristic (fold a pure reach/align stage into the grasp stage for the
same target). ``Subgoal`` itself is a shared domain type that lives in ``core`` because
the runners and the controller path consume it; this capability only produces it.
"""
from __future__ import annotations

import re
from typing import Any

from core.v0_types import Subgoal


class SubgoalPlanner:
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._last_diagnostics: dict[str, Any] = {}
        self._last_prompt = ""

    def plan(
        self,
        task: str,
        agentview,
        wrist=None,
        debug: bool = False,
        image_roles: "list[str] | None" = None,
    ) -> tuple[list[Subgoal], str]:
        response = self.agent.plan(
            task,
            agentview,
            wrist_image=wrist,
            debug=debug,
            image_roles=image_roles,
        )
        diagnostics_fn = getattr(self.agent, "diagnostics", None)
        self._last_diagnostics = diagnostics_fn() if diagnostics_fn is not None else {}
        prompt_fn = getattr(self.agent, "last_prompt", None)
        self._last_prompt = str(prompt_fn() if prompt_fn is not None else "")
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Planner did not return JSON object: {response.raw_text!r}")
        items = _subgoal_items(payload)
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"Planner JSON has no subgoals: {response.raw_text!r}")
        subgoals = [Subgoal.from_dict(item, index=i) for i, item in enumerate(items)]
        subgoals = _merge_pregrasp_stages(subgoals)
        return subgoals, response.raw_text

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def last_prompt(self) -> str:
        return self._last_prompt


def _subgoal_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("subgoals")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    if isinstance(items, dict):
        return [items]
    item = payload.get("subgoal")
    if isinstance(item, dict):
        return [item]
    if _looks_like_subgoal(payload):
        return [payload]
    return []


def _looks_like_subgoal(payload: dict[str, Any]) -> bool:
    required = {"id", "target", "affordance", "motion", "description", "completion"}
    return required.issubset(payload)


def _merge_pregrasp_stages(subgoals: list[Subgoal]) -> list[Subgoal]:
    normalized: list[Subgoal] = []
    i = 0
    while i < len(subgoals):
        current = subgoals[i]
        if i + 1 < len(subgoals):
            next_subgoal = subgoals[i + 1]
            if _should_merge_pregrasp(current, next_subgoal):
                normalized.append(_merge_pregrasp(current, next_subgoal))
                i += 2
                continue
        normalized.append(current)
        i += 1
    return normalized


def _should_merge_pregrasp(current: Subgoal, next_subgoal: Subgoal) -> bool:
    # A DeepPlan checkpoint is a structural sentinel, not a motion stage: never fold it
    # (or a stage into it). Its branch-rule description mentions "grasp"/"lift", so the
    # keyword heuristic below would otherwise misclassify it. Guard structurally on the
    # motion so the pivot survives intact whether or not DeepPlan is enabled.
    if _is_reason_pivot(current) or _is_reason_pivot(next_subgoal):
        return False
    if not _same_target(current.target, next_subgoal.target):
        return False
    if not _is_grasp_stage(next_subgoal):
        return False
    return _is_pure_reach_or_align(current)


def _merge_pregrasp(reach: Subgoal, grasp: Subgoal) -> Subgoal:
    return Subgoal(
        id=grasp.id,
        target=grasp.target or reach.target,
        affordance=grasp.affordance or reach.affordance,
        motion=grasp.motion,
        description=_join_descriptions(reach.description, grasp.description),
        completion=grasp.completion,
    )


def _is_pure_reach_or_align(subgoal: Subgoal) -> bool:
    text = _subgoal_text(subgoal)
    if not _contains_any(
        text,
        (
            "approach",
            "reach",
            "align",
            "center",
            "position",
            "move to",
            "move toward",
            "above",
            "over",
        ),
    ):
        return False
    return not _contains_any(
        text,
        (
            "grasp",
            "close",
            "clamp",
            "secure",
            "lift",
            "release",
            "open",
            "place",
            "drop",
        ),
    )


def _is_grasp_stage(subgoal: Subgoal) -> bool:
    return _contains_any(
        _subgoal_text(subgoal),
        ("grasp", "close", "clamp", "secure", "pick"),
    )


def _is_reason_pivot(subgoal: Subgoal) -> bool:
    """True for a DeepPlan ``REASON`` checkpoint (a deferred-branch sentinel, not motion)."""
    return str(getattr(subgoal, "motion", "") or "").strip().upper() == "REASON"


def _same_target(a: str, b: str) -> bool:
    left = _norm_target(a)
    right = _norm_target(b)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _norm_target(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _subgoal_text(subgoal: Subgoal) -> str:
    return " ".join(
        str(getattr(subgoal, field, "") or "").lower()
        for field in ("id", "target", "affordance", "motion", "description", "completion")
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text)
        for term in terms
    )


def _join_descriptions(first: str, second: str) -> str:
    parts = []
    for text in (first, second):
        stripped = " ".join(str(text or "").split())
        if stripped:
            parts.append(stripped.rstrip("."))
    if not parts:
        return ""
    return ". Then ".join(parts) + "."
