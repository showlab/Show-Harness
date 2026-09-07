"""DualSubgoalPlanner: drive the dual VLM planner agent and post-process both tracks.

The dual counterpart of :class:`plugins.subgoal.plugin.SubgoalPlanner`: parses the raw
``{"left": [...], "right": [...]}`` plan into per-arm :class:`core.v0_types.Subgoal`
lists, applying the same pre-grasp merge heuristic per track. Planner sloppiness on a
``WAIT`` stage (empty affordance/description) is backfilled rather than fatal -- a wait
stage's substance is its completion condition, not a graspable part.
"""
from __future__ import annotations

from typing import Any

from core.v0_types import Subgoal

from .plugin import _merge_pregrasp_stages

SIDES = ("left", "right")


class DualSubgoalPlanner:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def plan(
        self,
        task: str,
        agentview,
        wrist_left=None,
        wrist_right=None,
        debug: bool = False,
    ) -> tuple[dict[str, list[Subgoal]], str]:
        response = self.agent.plan(
            task,
            agentview,
            wrist_left_image=wrist_left,
            wrist_right_image=wrist_right,
            debug=debug,
        )
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dual planner did not return JSON object: {response.raw_text!r}")
        tracks: dict[str, list[Subgoal]] = {}
        for side in SIDES:
            items = payload.get(side)
            if not isinstance(items, list):
                raise RuntimeError(
                    f"Dual planner JSON has no {side!r} track: {response.raw_text!r}"
                )
            subgoals = [
                Subgoal.from_dict(_backfill_item(item), index=i)
                for i, item in enumerate(items)
                if isinstance(item, dict)
            ]
            tracks[side] = _merge_pregrasp_stages(subgoals)
        if not tracks["left"] and not tracks["right"]:
            raise RuntimeError(
                f"Dual planner returned two empty tracks: {response.raw_text!r}"
            )
        return tracks, response.raw_text


def _backfill_item(item: dict[str, Any]) -> dict[str, Any]:
    """Fill missing affordance/description from the item's own fields so a lean WAIT
    stage parses; stages with real manipulation keep whatever the planner wrote."""
    out = dict(item)
    if not str(out.get("affordance") or "").strip():
        out["affordance"] = str(out.get("target") or "the awaited object").strip()
    if not str(out.get("description") or "").strip():
        out["description"] = (
            str(out.get("completion") or "").strip()
            or f"{out.get('motion', 'stage')} for {out.get('target', 'the target')}"
        )
    return out
