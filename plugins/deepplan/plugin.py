"""DeepPlan tool: a deferred-branch checkpoint for conditional (IF-THEN) tasks.

The linear planner expands a task into one frozen ordered list of subgoals. That cannot
express a task whose plan depends on something only observable AFTER acting -- e.g.
"find and grasp the object under one of the blocks", where which block hides the object
is unknown until one is lifted. DeepPlan adds ONE structural sentinel to close that gap:
a subgoal with ``motion == "REASON"``.

With the tool ENABLED, the SAME :class:`~plugins.subgoal.SubgoalPlanner` (the only initial
planning call) is given a prompt ADDENDUM (:meth:`render_planner_addendum`) that lets it,
for a conditional task, emit an info-gathering PREFIX, then exactly ONE ``REASON`` pivot
carrying the branch rule, then a thin placeholder suffix. When the runner reaches the
pivot it calls :meth:`resolve`, which fires this tool's own VLM sub-role
(:class:`~plugins.deepplan.agent.DeepPlanResolverAgent`) on the LIVE images to decide the
branch and return the concrete stages to run next. The RUNNER owns the splice; the tool
only returns a frozen :class:`DeepPlanDecision`, exactly like :class:`plugins.recovery`.

Disabled -> :meth:`render_planner_addendum` is empty (the planner prompt is byte-identical
to today, so no ``REASON`` is ever emitted) and the runner is handed ``deepplan_plugin=None``
(the resolve path is unreachable). There are no task gates or step thresholds here: looping
(check the next candidate) is expressed by the resolver re-emitting a fresh peek + ``REASON``,
and termination is the resolver's own ``FOUND`` / ``EXHAUSTED`` judgment, bounded only by the
global ``max_steps`` backstop the runner already enforces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.v0_types import Subgoal

from .agent import DeepPlanResolverAgent

REASON_MOTION = "REASON"

# Co-located planner addendum: the REASON rules appended to the subgoal planner prompt
# only when the plugin is enabled (loaded once, cached).
PLANNER_ADDENDUM_PATH = Path(__file__).with_name("deepplan_planner.txt")


@dataclass(frozen=True)
class DeepPlanDecision:
    """The result of resolving a reached ``REASON`` checkpoint.

    ``resolved_subgoals`` non-empty -> the runner replaces the pivot (and the placeholder
    suffix) with these and continues. Empty -> resolution failed; the runner ends the
    episode rather than splicing a guessed plan (a blind grasp at an unrevealed target
    would otherwise spin the recovery loop).
    """

    event: str  # "branch_resolved" | "resolve_failed"
    branch: str  # FOUND | NOT_FOUND | EXHAUSTED | "" (failed)
    reason: str
    resolved_subgoals: list[Subgoal] = field(default_factory=list)
    raw_text: str = ""


class DeepPlanPlugin:
    """Augment the planner with a ``REASON`` pivot and resolve it on the live scene."""

    def __init__(
        self,
        enabled: bool = False,
        client: Any = None,
        common_context: str = "",
        resolver_prompt: str | None = None,
        max_resolve_retries: int = 2,
    ) -> None:
        self.enabled = bool(enabled)
        self.client = client
        self.common_context = common_context
        self._addendum: str | None = None
        # A single empty/garbled resolver reply must not kill the episode. On failure the
        # runner re-observes and re-resolves the SAME checkpoint next step; this counts the
        # consecutive failures so we eventually give up (a deterministic backend that always
        # returns empty would otherwise retry forever). Reset on any success. Tool-owned
        # state, like recovery's _unsettled_holds -- it reads no other tool's state.
        self.max_resolve_retries = max(0, int(max_resolve_retries))
        self._resolve_failures = 0
        # The resolver sub-role is only built when the plugin is live; a disabled tool holds
        # no VLM state, mirroring how a disabled tool never reads another tool's state.
        self.agent = (
            DeepPlanResolverAgent(
                client=client,
                common_context=common_context,
                prompt_template=resolver_prompt,
            )
            if self.enabled and client is not None
            else None
        )

    def render_planner_addendum(self) -> str:
        """The REASON-pivot rules to append to the subgoal planner prompt, or '' when off."""
        if not self.enabled:
            return ""
        if self._addendum is None:
            self._addendum = PLANNER_ADDENDUM_PATH.read_text(encoding="utf-8").strip()
        return self._addendum

    def is_pivot(self, subgoal: Any) -> bool:
        """True when execution has reached a DeepPlan checkpoint that must be resolved now."""
        return self.enabled and _motion(subgoal) == REASON_MOTION

    def resolve(
        self,
        *,
        task: str,
        subgoal: Subgoal,
        subgoals: Sequence[Subgoal],
        current_index: int,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> DeepPlanDecision:
        """Fire the resolver on the live scene and return the stages to run next.

        The runner owns all list mutation; this returns a frozen decision. On any
        VLM/parse failure -- or if the resolved tail has no runnable (non-REASON) head --
        the decision carries no subgoals so the runner can end cleanly.
        """
        if not self.enabled or self.agent is None:
            return DeepPlanDecision(
                event="resolve_failed", branch="", reason="deepplan disabled"
            )
        remaining = list(subgoals[current_index + 1 :])
        remaining_goal = (
            json.dumps([sg.to_prompt_dict() for sg in remaining], ensure_ascii=False)
            if remaining
            else "(no remaining stages were planned; produce the full finish)"
        )
        try:
            parsed = self.agent.resolve(
                task=task,
                branch_rule=subgoal.description,
                observe_condition=subgoal.completion,
                remaining_goal=remaining_goal,
                agentview_image=agentview_image,
                wrist_image=wrist_image,
                debug=debug,
            )
        except Exception as exc:  # noqa: BLE001 -- any resolver failure -> bounded retry/end
            return self._failure("", f"resolver call failed: {_one_line(exc)}", "")

        branch = str(parsed.get("branch", "")) if isinstance(parsed, dict) else ""
        reasoning = str(parsed.get("reasoning", "")) if isinstance(parsed, dict) else ""
        raw_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True) if isinstance(parsed, dict) else ""

        resolved = _parse_subgoals(parsed)
        # The spliced HEAD must be a concrete action: drop any leading REASON nodes so the
        # next iteration cannot re-land on a checkpoint with the SAME (unchanged) image and
        # spin with no progress. A trailing re-emitted REASON (the loop case) is kept.
        while resolved and _motion(resolved[0]) == REASON_MOTION:
            resolved.pop(0)
        if not resolved:
            return self._failure(branch, "resolver produced no runnable stage", raw_text)
        self._resolve_failures = 0
        return DeepPlanDecision(
            event="branch_resolved",
            branch=branch,
            reason=reasoning,
            resolved_subgoals=resolved,
            raw_text=raw_text,
        )

    def _failure(self, branch: str, reason: str, raw_text: str) -> DeepPlanDecision:
        """Count a failed resolution and choose retry-vs-give-up.

        Under the retry budget -> ``resolve_retry`` (the runner re-observes and re-resolves
        the same checkpoint next step). Budget exhausted -> ``resolve_failed`` (the runner
        ends the episode rather than splicing a guessed plan).
        """
        self._resolve_failures += 1
        attempts = f"attempt {self._resolve_failures}/{self.max_resolve_retries + 1}"
        if self._resolve_failures <= self.max_resolve_retries:
            return DeepPlanDecision(
                event="resolve_retry",
                branch=branch,
                reason=f"{reason} ({attempts}; retrying the checkpoint)",
                raw_text=raw_text,
            )
        return DeepPlanDecision(
            event="resolve_failed",
            branch=branch,
            reason=f"{reason} ({attempts}; giving up)",
            raw_text=raw_text,
        )


def _parse_subgoals(parsed: Any) -> list[Subgoal]:
    items = parsed.get("subgoals") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    out: list[Subgoal] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(Subgoal.from_dict(item, index=i))
        except (ValueError, TypeError):
            # Skip a malformed stage rather than abort; an empty result -> clean end.
            continue
    return out


def _motion(subgoal: Any) -> str:
    return str(getattr(subgoal, "motion", "") or "").strip().upper()


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())
