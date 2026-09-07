"""DeepPlan capability: a deferred-branch <REASON> checkpoint for conditional tasks.

Public API:
  * ``DeepPlanPlugin``     -- augments the planner prompt with the REASON-pivot rules and
    resolves a reached pivot into concrete subgoals on the live scene.
  * ``DeepPlanDecision`` -- the runner-level splice request returned by a resolution.
  * ``DeepPlanResolverAgent`` / ``DEEPPLAN_RESOLVE_SCHEMA`` -- the VLM sub-role + its
    output schema.

Disabled -> the planner prompt is byte-identical to today (no REASON is ever emitted) and
the runner is handed ``deepplan_plugin=None``. See :mod:`plugins.deepplan.plugin`.
"""
from .agent import DEEPPLAN_RESOLVE_SCHEMA, DeepPlanResolverAgent
from .plugin import DeepPlanDecision, DeepPlanPlugin

__all__ = [
    "DeepPlanPlugin",
    "DeepPlanDecision",
    "DeepPlanResolverAgent",
    "DEEPPLAN_RESOLVE_SCHEMA",
]
