"""Affordance grounding capability.

Public API:
  * :class:`~plugins.affordance.plugin.AffordancePlugin` -- two-phase contact-point grounding
    (a static front dot per stage + a per-step tracked wrist dot during the grasp),
    dot annotation on both views, and the AFFORD-field rewrite.
  * :class:`~plugins.affordance.agent.AffordancePointerAgent` -- the VLM pointing role:
    :meth:`locate` (fresh point + draw-and-verify) and :meth:`track` (seeded per-step
    re-location on a moving view). Used directly by the offline test script.
  * :class:`~plugins.affordance.agent.AffordancePoint` -- one grounded point (0-1000 grid).
"""
from plugins.affordance.agent import AffordancePoint, AffordancePointerAgent
from plugins.affordance.plugin import AffordancePlugin

__all__ = ["AffordancePoint", "AffordancePointerAgent", "AffordancePlugin"]
