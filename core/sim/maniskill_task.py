"""ManiSkill task/env glue for the MVTOKEN atomic-token policy.

Builds a single-environment ManiSkill task in the
translation-only ``pd_ee_delta_pos`` control mode with an ``rgb`` observation, and exposes
the few accessors the runner needs (RGB per camera, success flag, TCP position, gripper
width). ManiSkill returns *batched* torch tensors even for ``num_envs=1``; every accessor
here collapses the batch and returns plain numpy / python so the rest of the pipeline
(``core.record.images``, ``EpisodeLogger``, the VLM client) is unchanged.

Cameras: the stock ``PickCube``/``StackCube`` tasks expose ``base_camera`` (3rd-person) and,
when the robot is ``panda_wristcam``, a gripper-mounted ``hand_camera`` -- these become the
MVTOKEN agentview + wrist inputs. ``PickCube`` defaults to a plain ``panda`` (no wrist), so
pass ``robot_uids="panda_wristcam"`` to get both.

WHICH environments exist, what they are called and how they are registered is NOT here: it
is one table in :mod:`core.sim.maniskill_scenes`. This module is the scene-agnostic layer --
construction, observation accessors, stepping -- exactly like
``core.sim.robolab_task``. Adding an environment means adding a row there, not editing this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.sim.maniskill_scenes import instruction_for, register_scene


@dataclass
class ManiskillTaskHandle:
    env: Any
    task_description: str
    env_id: str
    control_mode: str
    robot_uids: str


def maybe_set_mujoco_gl(default: str = "egl") -> None:
    # ManiSkill renders with SAPIEN (vulkan/egl), but a headless box still wants a GL
    # backend selected; harmless if already set.
    os.environ.setdefault("MUJOCO_GL", default)


def make_maniskill_task(
    env_id: str,
    control_mode: str,
    robot_uids: str,
    camera_resolution: Optional[int],
    sim_backend: str,
    max_episode_steps: int,
    task_description: Optional[str] = None,
    scene: Optional[dict] = None,
) -> ManiskillTaskHandle:
    """Register whatever ``env_id`` needs, then construct it.

    ``scene`` carries the per-scene options from the config's ``scene:`` block (tabletop
    texture, layout preset, wrist mount/resolution, camera calibration...). Which keys a
    given env understands, and their defaults, are declared in that env's
    :class:`~core.sim.maniskill_scenes.SceneSpec` row -- unknown keys are ignored, so one
    ``scene:`` block can be shared across configs targeting different envs.
    """
    import gymnasium as gym
    import mani_skill  # noqa: F401 -- registers the stock ManiSkill gym env ids

    kwargs: dict[str, Any] = dict(
        num_envs=1,
        obs_mode="rgb",
        control_mode=control_mode,
        robot_uids=robot_uids,
        sim_backend=sim_backend,
        max_episode_steps=int(max_episode_steps),
    )
    # For an RLinf rig this imports the env module (registering the gym id), applies its
    # scene globals BEFORE construction reads them, and derives the wrist-cam agent; for a
    # stock ManiSkill task it is a no-op. Either way it returns whatever extra kwargs that
    # env's constructor takes.
    kwargs.update(register_scene(env_id, **dict(scene or {})))
    # camera_resolution overrides EVERY camera's width/height. Leave it null for envs whose
    # cameras carry a calibrated intrinsic matrix (the RLinf rigs' external_cam is 640x480
    # with a real RealSense K) so a square override doesn't corrupt the intrinsics.
    if camera_resolution:
        kwargs["sensor_configs"] = dict(
            width=int(camera_resolution), height=int(camera_resolution)
        )
    env = gym.make(env_id, **kwargs)
    desc = task_description or instruction_for(env_id)
    return ManiskillTaskHandle(
        env=env,
        task_description=desc,
        env_id=env_id,
        control_mode=control_mode,
        robot_uids=robot_uids,
    )


# -- tensor/obs helpers -----------------------------------------------------
def _to_np(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 - torch always present here, but stay defensive
        pass
    return np.asarray(value)


def ms_rgb(obs: dict, camera_name: str) -> np.ndarray:
    """One camera's RGB as HWC uint8 (batch dim collapsed)."""
    try:
        rgb = obs["sensor_data"][camera_name]["rgb"]
    except (KeyError, TypeError) as exc:
        available = list(obs.get("sensor_data", {}).keys())
        raise KeyError(
            f"camera {camera_name!r} not in obs; available: {available}"
        ) from exc
    arr = _to_np(rgb)
    if arr.ndim == 4:  # [B, H, W, C] -> [H, W, C]
        arr = arr[0]
    return np.ascontiguousarray(arr.astype(np.uint8))


def _info_flag(info: dict, key: str) -> bool:
    value = info.get(key)
    if value is None:
        return False
    arr = _to_np(value).reshape(-1)
    return bool(arr[0]) if arr.size else False


def ms_success(info: dict) -> bool:
    return _info_flag(info, "success")


def ms_is_grasped(info: dict) -> bool:
    return _info_flag(info, "is_grasped")


def ms_tcp(env: Any) -> np.ndarray:
    """End-effector (tcp) position, shape (3,)."""
    p = _to_np(env.unwrapped.agent.tcp.pose.p).reshape(-1, 3)
    return p[0].astype(float)


def ms_gripper_width(env: Any) -> float:
    """Panda gripper opening width from the two finger joints (each 0..0.04)."""
    qpos = _to_np(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
    if qpos.size >= 2:
        return float(qpos[-1] + qpos[-2])
    return 0.0


def reset_maniskill(
    env: Any,
    seed: int,
    settle_steps: int,
    hold_action: np.ndarray,
    reset_qpos: Any = None,
) -> tuple[dict, dict]:
    """Reset to a seeded initial config, optionally override the robot's start qpos, then
    hold (zero delta, gripper open) to settle.

    ``reset_qpos`` (list, or ``None`` to keep the env default) sets the robot's start joint
    angles *before* the first hold step. Because ``pd_ee_delta_pos`` holds the current EE
    orientation each step, this start orientation persists through the rollout -- the knob
    for e.g. the wrist-roll joint7 (fingers' opening axis / agentview gripper pose). Pass a
    full-DoF vector, or a shorter prefix to override only the leading joints (arm-7 or
    arm-7+gripper-2); trailing joints keep their reset value.
    """
    obs, info = env.reset(seed=int(seed))
    if reset_qpos is not None:
        _apply_reset_qpos(env, reset_qpos)
    action = _batched_action(hold_action)
    for _ in range(max(0, int(settle_steps))):
        obs, _reward, _term, _trunc, info = env.step(action)
    return obs, info


def _apply_reset_qpos(env: Any, reset_qpos: Any) -> None:
    import torch

    robot = env.unwrapped.agent.robot
    current = _to_np(robot.get_qpos()).reshape(-1).astype(np.float32)
    override = np.asarray(list(reset_qpos), dtype=np.float32).reshape(-1)
    n = min(current.size, override.size)
    current[:n] = override[:n]  # prefix override; trailing joints keep the reset value
    device = env.unwrapped.device
    robot.set_qpos(torch.tensor(current[None], dtype=torch.float32, device=device))
    robot.set_qvel(torch.zeros((1, current.size), dtype=torch.float32, device=device))


def _batched_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return action[None, :] if action.ndim == 1 else action


def step_maniskill(env: Any, action: np.ndarray) -> tuple[dict, bool, bool, dict]:
    """One env step; returns (obs, terminated, truncated, info) with batch collapsed."""
    obs, _reward, terminated, truncated, info = env.step(_batched_action(action))
    return (
        obs,
        bool(_to_np(terminated).reshape(-1)[0]),
        bool(_to_np(truncated).reshape(-1)[0]),
        info,
    )


def probe_move_axes(
    env: Any, controller: Any, seed: int, repeats: int = 8
) -> dict[str, list[float]]:
    """Step each MV_* token in isolation from the reset pose and record the observed TCP
    delta. Cheap diagnostic (writes to calibration.json) documenting exactly which world
    direction each token drives -- so an obviously mirrored/swapped axis can be fixed in
    the config instead of guessed."""
    from core.action_units import MOVE_ATOMS

    result: dict[str, list[float]] = {}
    for token in MOVE_ATOMS:
        env.reset(seed=int(seed))
        p0 = ms_tcp(env)
        action = controller.action_for_atomic(token)
        for _ in range(int(repeats)):
            step_maniskill(env, action)
        delta = ms_tcp(env) - p0
        result[token] = [round(float(x), 4) for x in delta]
    controller.open_gripper()
    return result
