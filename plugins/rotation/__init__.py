"""Rotation capability: gripper yaw alignment tokens + eye-in-hand frame compensation.

Public API:
  * ``RotationPlugin`` -- offers ROTATE_CW/CCW to the controller VLM (a last-resort grasp
    alignment move) and rotates a wrist-judged MV_* vector by the accumulated gripper yaw
    so wrist-frame directions stay correct after the gripper has turned.

Disabled -> no ROTATE token is offered and ``compensate_move`` is the identity, so motion
is byte-identical to today. See :mod:`plugins.rotation.plugin`.
"""
from .plugin import RotationPlugin

__all__ = ["RotationPlugin"]
