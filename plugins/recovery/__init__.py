"""Recovery capability: measured-width grasp recovery for the real runner.

Public API:
  * ``RecoveryPlugin`` -- classifies gripper width and requests release/rollback.
  * ``RecoveryDecision`` -- runner-level recovery intervention record.

See :mod:`plugins.recovery.plugin` for the implementation.
"""
from .plugin import RecoveryDecision, RecoveryPlugin

__all__ = ["RecoveryPlugin", "RecoveryDecision"]
