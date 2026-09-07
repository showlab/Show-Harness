"""Smooth-motion capability: ramp the impedance setpoint along a min-jerk profile.

Public API:
  * ``SmoothPlugin`` -- yields the min-jerk interpolation fractions + per-waypoint delay.

See :mod:`plugins.smooth.plugin` for the implementation.
"""
from .plugin import SmoothPlugin

__all__ = ["SmoothPlugin"]
