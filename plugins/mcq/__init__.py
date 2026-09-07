"""MCQ capability: a multiple-choice answer protocol for the controller.

Public API:
  * ``McqPlugin``        -- the answer protocol (option labelling + letter<->action map).
  * ``ATOMIC_ACTIONS`` -- the labelled atomic action vocabulary.

See :mod:`plugins.mcq.plugin` for the implementation.
"""
from .plugin import ATOMIC_ACTIONS, McqPlugin

__all__ = ["McqPlugin", "ATOMIC_ACTIONS"]
