"""Subgoal planning capability: expand a task + image into an ordered subgoal plan.

Public API:
  * ``SubgoalPlanner``      -- orchestration (call the agent, parse + merge stages).
  * ``SubgoalPlannerAgent`` -- the single VLM call (owns the co-located prompt + schema).
  * ``SUBGOAL_PLAN_SCHEMA`` -- the plan output schema.
  * ``load_prompt``         -- read the co-located ``subgoal_planner.txt`` template.

The produced :class:`core.v0_types.Subgoal` is a shared domain type (consumed by the
runners and the controller), so it lives in ``core``, not here.

Implementation: the public tool is in :mod:`plugins.subgoal.plugin`; its VLM sub-role is in
:mod:`plugins.subgoal.agent`; the prompt is the co-located ``subgoal_planner.txt``.

Dual-arm counterparts (``DualSubgoalPlanner`` / ``DualSubgoalPlannerAgent`` /
``subgoal_planner_dual.txt``) plan one CONCURRENT track per arm; see
:mod:`plugins.subgoal.dual_agent`.
"""
from .agent import SUBGOAL_PLAN_SCHEMA, SubgoalPlannerAgent, load_prompt
from .dual_agent import (
    DUAL_SUBGOAL_PLAN_SCHEMA,
    DualSubgoalPlannerAgent,
    load_dual_prompt,
)
from .dual_plugin import DualSubgoalPlanner
from .plugin import SubgoalPlanner

__all__ = [
    "SubgoalPlanner",
    "SubgoalPlannerAgent",
    "SUBGOAL_PLAN_SCHEMA",
    "load_prompt",
    "DualSubgoalPlanner",
    "DualSubgoalPlannerAgent",
    "DUAL_SUBGOAL_PLAN_SCHEMA",
    "load_dual_prompt",
]
