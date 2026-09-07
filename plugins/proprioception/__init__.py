"""Proprioception capability: measured robot-state context for the controller.

Public API:
  * ``ProprioceptionPlugin`` -- renders controller prompt proprio/gripper blocks.

See :mod:`plugins.proprioception.plugin` for the implementation.
"""
from .plugin import ProprioceptionPlugin

__all__ = ["ProprioceptionPlugin"]
