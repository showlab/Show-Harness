"""Unified dual-arm Piper config: shared keys + a per-arm ``arms:`` block.

``configs/robot_piper.yaml`` is the single source of truth for BOTH arms. Everything
outside ``arms:`` is shared; only the values that genuinely differ per arm live under
``arms.left`` / ``arms.right``:

    begin_pose: default        # which named start pose is active (shared by both arms)

    arms:
      left:
        wrist_camera_topic: /camera_l/color/image_raw
        z_floor_m: 0.19443
        poses:                 # NAMED start poses -- capture as many as you like
          default: [...]       #   go_begin.py --arm left --pose <name> --capture --write
          high:    [...]
        rest_joints:  [...]    # idle/park pose, on demand (go_rest)
      right:
        ...

Start poses: an arm can have any number of named poses under ``arms.<side>.poses``. The
shared top-level ``begin_pose`` (and optional ``rest_pose``) selects which one is active;
:func:`arm_config` resolves it into the flat ``begin_joints`` / ``rest_joints`` keys that
every consumer already reads, so switching start pose is a one-word config change (or
``--begin-pose <name>`` on the runners) and nothing downstream needs to know about names.
A config with no ``poses:`` block still works: plain ``begin_joints`` / ``rest_joints``
under the arm remain valid (and are the fallback when no name is selected).

:func:`arm_config` flattens all of that into the SINGLE-arm view every downstream consumer
(run_real's helpers, the controllers, the sessions) already expects -- shared keys plus
that arm's block, with ``robot.arm`` and ``robot.wrist_camera_topic`` filled in. So the
dual collector resolves both arms with two calls, and an autonomous rollout resolves the
one arm it drives, without either needing to know the nested layout.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

SIDES = ("left", "right")

# Per-arm keys promoted to the TOP LEVEL of the flattened view (the rest of the stack
# reads them there, e.g. run_real's _resolve_z_floor / the pose helpers).
_ARM_TOPLEVEL_KEYS = ("z_floor_m", "begin_joints", "rest_joints")
# Per-arm keys that belong inside the `robot:` block.
_ARM_ROBOT_KEYS = ("wrist_camera_topic",)

# Named-pose plumbing: (selector key, per-arm map, resolved flat key). The selector is
# SHARED (both arms use the same pose name); the joint values are per-arm.
POSE_MAP_KEY = "poses"
_POSE_SELECTORS = (("begin_pose", "begin_joints"), ("rest_pose", "rest_joints"))


def arm_blocks(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the raw per-arm blocks, validating that both arms are present."""
    arms = cfg.get("arms") or {}
    if not isinstance(arms, dict):
        raise ValueError("robot_piper.yaml: `arms:` must be a mapping of left/right blocks")
    missing = [s for s in SIDES if not isinstance(arms.get(s), dict)]
    if missing:
        raise ValueError(
            f"robot_piper.yaml: `arms:` is missing a block for {missing}. The Piper rig is "
            "dual-arm; both `arms.left` and `arms.right` must be defined."
        )
    return {s: dict(arms[s]) for s in SIDES}


def arm_config(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    """Flatten the unified dual-arm config into the single-arm view for ``side``.

    Shared keys are kept as-is; the arm's block is merged on top (``z_floor_m``,
    ``begin_joints``, ``rest_joints`` at the top level; ``wrist_camera_topic`` into
    ``robot:``), and ``robot.arm`` is set to ``side`` (the ROS topic namespace). The
    nested ``arms:`` block is dropped from the result so nothing downstream sees it.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    blocks = arm_blocks(cfg)
    block = blocks[side]

    out = copy.deepcopy(cfg)
    out.pop("arms", None)

    robot = dict(out.get("robot") or {})
    robot["arm"] = side
    for key in _ARM_ROBOT_KEYS:
        if block.get(key) is not None:
            robot[key] = block[key]
    out["robot"] = robot

    for key in _ARM_TOPLEVEL_KEYS:
        if block.get(key) is not None:
            out[key] = block[key]

    # Any other per-arm key the user adds is promoted to the top level too, so the
    # block stays open for future per-arm overrides without touching this helper.
    for key, value in block.items():
        if key not in _ARM_ROBOT_KEYS and key not in _ARM_TOPLEVEL_KEYS:
            out[key] = value

    # Named start poses win over a plain begin_joints/rest_joints in the block: the
    # selected pose is resolved into the flat key every consumer already reads.
    for selector, flat_key in _POSE_SELECTORS:
        name = cfg.get(selector)
        if name:
            out[flat_key] = pose_joints(cfg, side, str(name), selector=selector)
    return out


def pose_names(cfg: dict[str, Any], side: str) -> list[str]:
    """The names defined under ``arms.<side>.poses`` (empty when the arm has none)."""
    poses = (arm_blocks(cfg)[side].get(POSE_MAP_KEY) or {}) if side in SIDES else {}
    if not isinstance(poses, dict):
        raise ValueError(f"robot_piper.yaml: arms.{side}.{POSE_MAP_KEY} must be a mapping of name -> joints")
    return list(poses)


def pose_joints(
    cfg: dict[str, Any], side: str, name: str, selector: Optional[str] = None
) -> list[float]:
    """The joints of the named pose ``arms.<side>.poses.<name>``.

    Raises with the available names listed, so a typo in ``begin_pose`` (or in
    ``--begin-pose``) fails immediately with a usable message instead of silently
    falling back to some other start pose.
    """
    block = arm_blocks(cfg)[side]
    poses = block.get(POSE_MAP_KEY) or {}
    if not isinstance(poses, dict) or name not in poses:
        where = f"`{selector}: {name}`" if selector else f"pose {name!r}"
        available = ", ".join(pose_names(cfg, side)) or "(none defined)"
        raise ValueError(
            f"robot_piper.yaml: {where} selects arms.{side}.{POSE_MAP_KEY}.{name}, which "
            f"does not exist. Available for the {side} arm: {available}. Capture it with: "
            f"scripts/piper/go_begin.sh --arm {side} --pose {name} --capture --write"
        )
    joints = poses[name]
    if not isinstance(joints, (list, tuple)) or not joints:
        raise ValueError(
            f"robot_piper.yaml: arms.{side}.{POSE_MAP_KEY}.{name} must be a list of joint angles"
        )
    return [float(v) for v in joints]


def both_arm_configs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{side: arm_config(cfg, side)}`` for both arms (the dual-collection view)."""
    return {side: arm_config(cfg, side) for side in SIDES}
