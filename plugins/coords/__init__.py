"""Coordinate-system capability for controller prompts.

Public API:
  * ``CoordsPlugin`` -- prompt transformer for DIRECTION/REMARK sections.

See :mod:`plugins.coords.plugin` for the implementation.
"""
from .plugin import CoordsPlugin

__all__ = ["CoordsPlugin"]
