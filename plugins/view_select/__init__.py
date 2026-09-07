"""Multi-view action selection (dual mode B): the VLM reports each arm's guiding
view (WRIST rule A / FRONT rule B) and that view picks the move's motion frame.

Thin re-export of the public API. See :mod:`plugins.view_select.plugin`.
"""
from plugins.view_select.plugin import VIEW_FRONT, VIEW_TOKENS, VIEW_WRIST, ViewSelectPlugin

__all__ = ["ViewSelectPlugin", "VIEW_TOKENS", "VIEW_WRIST", "VIEW_FRONT"]
