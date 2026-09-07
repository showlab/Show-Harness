from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.v0_types import SkillContext


class Controller:
    """One VLM call per step that returns the executed base/grasp token."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def last_prompt(self) -> str:
        """The most recent fully-rendered controller prompt (for periodic logging)."""
        return getattr(self.agent, "last_prompt", "")

    def decide(
        self,
        ctx: SkillContext,
        recent_moves: str,
        previous_direction: str,
        gripper_state: str,
        recovery_context: str = "",
        prev_agentview: Any = None,
    ):
        return self.agent.decide(
            task=ctx.task,
            subgoal=ctx.subgoal.to_prompt_dict(),
            recent_moves=recent_moves,
            previous_direction=previous_direction,
            gripper_state=gripper_state,
            agentview_image=ctx.agentview,
            wrist_image=ctx.wrist,
            # Frame captured BEFORE the previous action executed (action-ablation
            # blind review); None everywhere else, incl. the sim runner.
            prev_agentview_image=prev_agentview,
            proprio=ctx.proprio,
            recovery_context=recovery_context,
            debug=ctx.debug,
        )


@dataclass(frozen=True)
class StageControlSuite:
    controller: Controller
