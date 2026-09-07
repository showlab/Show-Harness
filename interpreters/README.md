# Interpreters

An interpreter grounds the semantic action units (`core/action_units.py`) into
motion on one embodiment, deterministically. The model never sees joints,
frames, or gains; moving to a new robot means writing a new interpreter while
the vocabulary, prompts, and policies stay unchanged.

## The two families

**Simulators** emit normalized per-step actions consumed by `env.step`:

| File | Embodiment | Action |
| --- | --- | --- |
| `maniskill_atomic_controller.py` | ManiSkill | 4-D `pd_ee_delta_pos` (translation only) |
| `robolab_atomic_controller.py` | Isaac Lab | 7-D relative differential-IK with per-step orientation hold |

**Real arms** integrate absolute Cartesian setpoints against a duck-typed
robot object:

- `real_atomic_controller.py` — the arm-agnostic base: setpoint integration,
  per-command clamps, the Z safety floor, base/wrist motion frames, gripper
  settle + empty-grasp detection, and the mount points for the `smooth`,
  `variable_step`, and `rotation` plugins.
- `franka_atomic_controller.py` — Franka via Polymetis impedance
  (`core/franka/`), plus lost-controller self-healing.
- `piper_atomic_controller.py` — AgileX Piper via bounded-orientation DLS IK
  and 50 Hz joint streaming (`core/piper/`), with divergence re-sync and
  dropped-command detection.

## The robot duck type

A real-arm backend implements:

```
get_ee_pose() -> [x, y, z, qx, qy, qz, qw]
get_gripper_position() -> [width_m]
control_gripper(close: bool)
update_desired_ee_pose(pose7)
```

plus optionally `get_joint_positions` / `stream_joints` (joint-streaming
backends). `core/franka/franka_interface.py` (ZeroRPC) and
`core/piper/piper_interface.py` (ROS topics) are the two shipped backends;
both have mock counterparts for hardware-free runs.

## Metric realization and calibration

Unit vectors, rotation signs, and default step sizes live in
`configs/primitives_<embodiment>.yaml` — the single source of truth for
per-robot axis conventions (its comments record how each sign was verified).
Step sizes are overridden per run by the robot config (`fine_step_m`,
`coarse_step_m`, `up_step_m`), and safety comes from the calibrated Z floor:
once captured with the gripper resting on the tabletop, no command may drive
the setpoint below it. Capture floors and poses with the scripts under
`scripts/franka/` and `scripts/piper/` — calibration never means editing a
config by hand.
