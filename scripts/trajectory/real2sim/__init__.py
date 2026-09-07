"""real2sim -- simulator-side collection of MVTOKEN atomic-action training data.

Sibling of the real-robot collectors in ``scripts/trajectory/`` (``collect_rollouts.py``,
``collect_rollouts_piper.py``, ``webui/``): those record a human pressing one direction key
at a time on a real arm, these produce the same rollouts from a simulator. Both write the
identical teleop layout, so sim and real data mix without a special case.

    atomic_tokenizer.py   the reusable, simulator-agnostic core: token vocabulary,
                          closed-loop 2 cm execution, planners, rollout writer
    backends/             one adapter per simulator (``make_backend("maniskill", ...)``)
    maniskill/            ManiSkill task construction + the generators and dataset tooling

Everything sim-specific is confined to ``backends/<sim>.py`` and ``<sim>/``; changing
simulator does not touch the discretiser. See README.md for the pipeline and commands.
"""
