"""Embodiment interpreters: deterministic grounding of semantic action units.

Each interpreter g_E maps the shared vocabulary (core.action_units) onto one
embodiment's control interface -- ManiSkill
delta-position and Isaac-Lab differential-IK (maniskill/robolab), and the real
arms via absolute Cartesian setpoints (real_atomic_controller base class;
franka_atomic_controller adds the Polymetis impedance specifics,
piper_atomic_controller the AgileX joint-streaming backend). Motion metrics
(unit vectors, step sizes, rotation signs) come from
configs/primitives_<embodiment>.yaml; per-rig calibration from the robot
configs. Moving to a new embodiment means writing a new interpreter -- the
model-facing vocabulary and prompts stay unchanged.
"""
