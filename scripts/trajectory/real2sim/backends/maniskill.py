"""ManiSkill implementation of the :class:`AtomicSimEnv` contract.

This is the ONLY ManiSkill-aware layer of the discretiser: it builds the env, applies one
``pd_ee_delta_pos`` control step, reads TCP/gripper/actor state, and renders the two
deployment views. Everything above it (``atomic_tokenizer``) is sim-agnostic, so a second
simulator needs a sibling of this file and nothing else.

Scene definitions stay in ``core/`` (``core.sim.maniskill_task`` + the ``core.sim.maniskill_scenes``
table) because they are the DEPLOYMENT contract -- the eval runner constructs its env from
exactly those modules. Generating data from a private copy of a scene would let train and
deploy drift apart silently, so this backend wraps them instead of re-declaring anything.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import AtomicSimEnv, prepared_pair, to_np

# Deployment camera contract (configs/robot_maniskill*.yaml). agentview = the calibrated
# front RealSense, raw; wrist = the centred hand camera rotated 180 deg, which puts the
# fingertips at the TOP with left/right matching the agentview -- unflipped, the two views
# disagree on left/right and the policy oscillates MV_LEFT/MV_RIGHT forever.
DEFAULT_AGENTVIEW_CAMERA = "external_cam"
DEFAULT_WRIST_CAMERA = "hand_camera"
DEFAULT_WRIST_FLIP = "both"


class ManiSkillBackend(AtomicSimEnv):
    """Drives a single ManiSkill env in ``pd_ee_delta_pos`` for the atomic discretiser."""

    def __init__(
        self,
        env: Any,
        agentview_camera: str = DEFAULT_AGENTVIEW_CAMERA,
        wrist_camera: str = DEFAULT_WRIST_CAMERA,
        agentview_flip: str = "none",
        wrist_flip: str = DEFAULT_WRIST_FLIP,
        agentview_rotation_degrees: int = 0,
        wrist_rotation_degrees: int = 0,
        agentview_crop_aspect: Optional[float] = None,
        wrist_crop_aspect: Optional[float] = None,
        agentview_square_size: Optional[int] = None,
        wrist_square_size: Optional[int] = None,
        delta_bound_m: float = 0.1,
        env_id: str = "",
        task_description: str = "",
    ) -> None:
        self.env = env
        self.agentview_camera = str(agentview_camera)
        self.wrist_camera = str(wrist_camera)
        self.agentview_flip = str(agentview_flip)
        self.wrist_flip = str(wrist_flip)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_crop_aspect = agentview_crop_aspect
        self.wrist_crop_aspect = wrist_crop_aspect
        self.agentview_square_size = (
            int(agentview_square_size) if agentview_square_size else None
        )
        self.wrist_square_size = int(wrist_square_size) if wrist_square_size else None
        self.delta_bound_m = float(delta_bound_m)
        self.env_id = str(env_id)
        self.task_description = str(task_description)
        self.last_info: dict = {}

    # -- construction --------------------------------------------------------
    @classmethod
    def make(
        cls,
        env_id: str,
        robot_uids: str,
        task_description: Optional[str] = None,
        table_tex: str = "white",
        traj_id: str = "random",
        wrist_resolution: int = 256,
        wrist_mount: str = "centered",
        sim_backend: str = "physx_cpu",
        max_episode_steps: int = 100000,
        **kwargs: Any,
    ) -> "ManiSkillBackend":
        """Build the env for ``env_id`` through the DEPLOYMENT factory.

        Deliberately the one path: :func:`core.sim.maniskill_task.make_maniskill_task` registers
        the scene and its wrist-cam agent before ``gym.make`` sees them, and it is exactly
        what the eval runner calls -- so generated data and deployed rollouts cannot drift.
        ``robot_uids`` is required (every registered scene carries its own agent);
        ``real2sim.maniskill.tasks.backend_kwargs`` fills it in from the scene table.
        """
        from core.sim.maniskill_task import make_maniskill_task

        handle = make_maniskill_task(
            env_id=env_id,
            control_mode="pd_ee_delta_pos",
            robot_uids=robot_uids,
            camera_resolution=None,
            sim_backend=sim_backend,
            max_episode_steps=max_episode_steps,
            task_description=task_description,
            scene={"table_tex": table_tex, "traj_id": traj_id,
                   "wrist_mount": wrist_mount,
                   "wrist_resolution": wrist_resolution},
        )
        return cls(
            handle.env, env_id=env_id, task_description=handle.task_description, **kwargs
        )

    # -- sim lifecycle -------------------------------------------------------
    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, seed: int) -> None:
        self.env.reset(seed=int(seed))
        self.last_info = {}

    def close(self) -> None:
        self.env.close()

    # -- state readback ------------------------------------------------------
    def tcp_pos(self) -> np.ndarray:
        return to_np(self.unwrapped.agent.tcp.pose.p).reshape(-1, 3)[0].astype(np.float64)

    def tcp_pose7(self) -> list[float]:
        p = to_np(self.unwrapped.agent.tcp.pose.p).reshape(-1, 3)[0]
        q = to_np(self.unwrapped.agent.tcp.pose.q).reshape(-1, 4)[0]
        return [round(float(v), 5) for v in (*p, *q)]

    def gripper_width(self) -> float:
        qpos = to_np(self.unwrapped.agent.robot.get_qpos()).reshape(-1)
        return float(qpos[-1] + qpos[-2])

    # -- actuation -----------------------------------------------------------
    def apply_delta(self, delta_m: np.ndarray, grip_cmd: float,
                    max_cmd_m: float) -> None:
        a = np.clip(np.asarray(delta_m) / self.delta_bound_m,
                    -max_cmd_m / self.delta_bound_m,
                    max_cmd_m / self.delta_bound_m)
        action = np.array([[a[0], a[1], a[2], float(grip_cmd)]], dtype=np.float32)
        _obs, _r, _term, _trunc, info = self.env.step(action)
        self.last_info = info

    # -- observation / evaluation -------------------------------------------
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(agentview, wrist) with the FULL deployment transform contract applied.

        Both views, all four stages, one function -- see :func:`_prepared_pair`.
        """
        u = self.unwrapped
        u.scene.update_render()
        sd = u.get_obs()["sensor_data"]
        agentview = to_np(sd[self.agentview_camera]["rgb"])
        wrist = to_np(sd[self.wrist_camera]["rgb"])
        if agentview.ndim == 4:
            agentview = agentview[0]
        if wrist.ndim == 4:
            wrist = wrist[0]
        return prepared_pair(self, agentview, wrist)

    def success(self) -> bool:
        try:
            ev = self.unwrapped.evaluate()
            s = ev.get("success")
            if s is not None:
                return bool(to_np(s).reshape(-1)[0])
        except Exception:  # noqa: BLE001 -- envs without evaluate(); fall back to info
            pass
        s = self.last_info.get("success")
        return bool(to_np(s).reshape(-1)[0]) if s is not None else False

    # -- privileged scene access (task construction) -------------------------
    def actor(self, name: str) -> Any:
        """A named actor on the env (``cube``, ``target``, ``white_block``, ...)."""
        return getattr(self.unwrapped, name)

    def actor_pos(self, name: str) -> np.ndarray:
        return to_np(self.actor(name).pose.p).reshape(-1, 3)[0].astype(np.float64)

    def set_actor_pose(self, name: str, p, q=None) -> None:
        import sapien

        pose = (sapien.Pose(p=[float(v) for v in p], q=[float(v) for v in q])
                if q is not None else sapien.Pose(p=[float(v) for v in p]))
        self.actor(name).set_pose(pose)

    def scene_const(self, name: str) -> Any:
        """A scene geometry constant (``TABLE_Z``, ``BLOCK_HALF_SIZE``, ...)."""
        return getattr(self.unwrapped, name)
