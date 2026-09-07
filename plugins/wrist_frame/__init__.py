"""Wrist-motion-frame prompt fix (front-view directions follow the gripper heading).

Thin re-export of the public API. See :mod:`plugins.wrist_frame.plugin`.
"""
from plugins.wrist_frame.plugin import WristFramePlugin

__all__ = ["WristFramePlugin"]
