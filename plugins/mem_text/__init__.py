"""Move-memory text capability: the controller's recent-move history + history rules.

Public API:
  * ``MemTextPlugin`` -- renders the "Recent moves" line and the move-history rule bullets.

See :mod:`plugins.mem_text.plugin` for the implementation.
"""
from .plugin import MemTextPlugin

__all__ = ["MemTextPlugin"]
