"""Auto-release capability: reflexively reopen an empty closed gripper.

Public API:
  * ``AutoReleasePlugin`` -- decides when a closed gripper is below the empty width.

See :mod:`plugins.auto_release.plugin` for the implementation.
"""
from .plugin import AutoReleasePlugin

__all__ = ["AutoReleasePlugin"]
