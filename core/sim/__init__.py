"""Simulator-side task glue and rollout runners.

Everything in this package talks to a SIMULATOR; nothing in it touches real hardware.
``core/`` proper keeps the shared pieces (``config``, ``images``, ``episode_logger``,
``v0_types``, ...) and the real-robot runners (``real_runner``, ``mvtoken_runner``,
``dual_runner``, ``teleop*``, ``franka/``, ``piper/``).

Two simulators are wired, each with the same two-file shape -- a task module (env
construction + observation/success/TCP/gripper accessors + axis probe) and an MVTOKEN
rollout runner:

===========  ==========================  ===============================
Simulator    task module                 MVTOKEN runner
===========  ==========================  ===============================
ManiSkill    ``maniskill_task``          ``mvtoken_maniskill_runner``
RoboLab      ``robolab_task``            ``mvtoken_robolab_runner``
===========  ==========================  ===============================

Plus ``maniskill_scenes`` -- the ManiSkill scene table, one row per environment (see its
docstring).

Nothing is re-exported here on purpose: importing a simulator's task module pulls that
simulator's heavyweight dependencies in, and the two cannot generally coexist in one
interpreter (ManiSkill wants its own conda env, RoboLab its own Python 3.11 venv).
Import the specific module you need::

    from core.sim.maniskill_task import make_maniskill_task
"""
