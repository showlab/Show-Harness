"""DAGGER: real-time human keyboard override during autonomous rollouts, with the
teleop key bindings; in-flight VLM decisions superseded by input are dropped.

Thin re-export of the public API. See :mod:`plugins.dagger.plugin`.
"""
from plugins.dagger.plugin import KIND_GRIPPER, KIND_MOVE, KIND_STILL, DaggerPlugin

__all__ = ["DaggerPlugin", "KIND_MOVE", "KIND_GRIPPER", "KIND_STILL"]
