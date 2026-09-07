"""Simulator backends for the atomic-action discretiser.

Each backend implements :class:`~scripts.trajectory.real2sim.atomic_tokenizer.AtomicSimEnv`
for one simulator; the discretisation core never imports any of them directly. Selecting a
different simulator is therefore a one-line change at the call site::

    backend = make_backend("maniskill", env_id="BlockPAP-v1",
                           robot_uids="panda_high_friction_wristcam")

Adding a simulator: write ``backends/<sim>.py`` with a class implementing the six abstract
methods, register it below, and the generators, planners, writer and dataset tooling all
work unchanged.
"""
from __future__ import annotations

from typing import Any

from scripts.trajectory.real2sim.atomic_tokenizer import AtomicSimEnv

# Lazily imported: a backend's sim package (mani_skill, ...) should not be a hard
# dependency of merely importing this module.
BACKENDS: dict[str, tuple[str, str]] = {
    "maniskill": ("scripts.trajectory.real2sim.backends.maniskill", "ManiSkillBackend"),
    # RoboLab additionally requires Isaac Sim to be launched (core.sim.robolab_task.launch_isaac)
    # BEFORE make_backend is called -- see backends/robolab.py.
    "robolab": ("scripts.trajectory.real2sim.backends.robolab", "RobolabBackend"),
}


def make_backend(sim: str, **kwargs: Any) -> AtomicSimEnv:
    """Construct the backend named ``sim`` (see :data:`BACKENDS`) via its ``make``."""
    import importlib

    if sim not in BACKENDS:
        raise SystemExit(f"unknown sim {sim!r}; available: {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[sim]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls.make(**kwargs)
