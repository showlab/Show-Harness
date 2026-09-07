"""RoboLab (NVIDIA Isaac Lab) implementation of the :class:`AtomicSimEnv` contract.

Second backend after ``backends/maniskill.py``, and the file that proves the split works:
everything above it (``atomic_tokenizer`` -- token vocabulary, closed-loop 2 cm execution,
Manhattan/RDP/chase planners, the teleop-format writer) is reused byte for byte. Only this
file knows what RoboLab is.

Like the ManiSkill backend it wraps the DEPLOYMENT modules rather than re-declaring
anything: the env is built through :func:`core.sim.robolab_task.make_robolab_task`, the one the
eval runner uses, and the camera transform chain is the same
``rotate_and_flip -> center_crop_to_aspect`` the runner applies before sending a frame. A
private copy of either would let training and deployment drift apart silently.

Three RoboLab specifics the contract has to absorb:

* **Gripper polarity.** The core's ``grip_cmd`` is +1 open / -1 close (the ManiSkill mimic
  convention). RoboLab's Robotiq binary action is the opposite: 1.0 closes, 0.0 opens.
  Translated in :meth:`apply_delta`, nowhere else.
* **``delta_bound_m`` is the action config's ``scale``** (0.5), not a controller bound:
  RoboLab's relative IK treats the action as "metres / scale" of target displacement.
* **Observations only exist after a step.** Isaac Lab returns them from ``step``/``reset``
  rather than exposing a "render now" call, so the latest observation is cached and
  :meth:`grab_frames` serves that. This is exactly the frame the contract wants -- the
  state BEFORE the token about to be recorded executes.

Isaac Sim must already be running (``core.sim.robolab_task.launch_isaac``) before this backend
is constructed; the generator scripts under ``real2sim/robolab/`` do that first.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.sim.robolab_task import (
    ee_tilt_deg,
    hold_orientation_rotvec,
    make_robolab_task,
    reset_robolab,
    rl_ee_quat,
    rl_gripper_width,
    rl_rgb,
    rl_success,
    rl_tcp,
    rl_tcp_pose7,
    step_robolab,
)
from scripts.trajectory.real2sim.atomic_tokenizer import AtomicSimEnv, prepared_pair

# Fallbacks ONLY. The camera contract that a dataset is actually built with comes from
# configs/robot_robolab.yaml, read by the generators via ``core.config.camera_contract``
# and by the deployment runner directly -- one file, so stored frames and served frames
# cannot drift. These constants exist for a bare ``RobolabBackend(...)`` in a probe script.
#
# Keep them in step with the config when it is re-measured. They were NOT, once: the
# config moved to ``front_cam`` (robot: franka) while this still said
# ``over_shoulder_left_camera``, and because the generators took their defaults from here
# rather than the config, generation died on a camera the scene no longer had.
DEFAULT_AGENTVIEW_CAMERA = "front_cam"
DEFAULT_WRIST_CAMERA = "wrist_cam"
# front_cam renders 4:3 natively (640x480, ManiSkill's own intrinsics), so no crop. Only
# RoboLab's stock 16:9 cameras (robot: droid) need 4/3 here -- see
# core.record.images.center_crop_to_aspect.
DEFAULT_CROP_ASPECT = None
# Panda-hand wrist: raw has the fingertips entering from the LEFT; the MVTOKEN contract
# wants them at the TOP, which is a 270 deg CCW turn. Measured 2026-08-04 against a
# rendered ManiSkill hand_camera frame; see the config for the full derivation.
DEFAULT_WRIST_FLIP = "none"
DEFAULT_WRIST_ROTATION_DEGREES = 270

# Episode time limit for GENERATION, in seconds of simulated time. RoboLab's own tasks
# allow 40-50 s (RubiksCubeTask: 40 s = 600 control steps at decimation 8 / dt 1/120),
# which is sized for a continuous policy -- but the atomic executor spends ~12 control
# steps per token closing the relative-IK loop, so a 50-token episode needs ~600 steps for
# the tokens ALONE and runs into the limit mid-carry.
#
# Hitting it is not a visible failure. IsaacLab resets the scene, so the gripper springs
# back to fully open and the carried object teleports to its start pose -- which reads
# exactly like "the object slipped out", and the resulting rollout keeps recording carry
# tokens over an empty gripper. This was originally mis-diagnosed as a physical drop
# (twice), until the give-away: BOTH the width and the object pose were exactly their
# initial values. 300 s leaves ~10x headroom over the longest observed episode.
DEFAULT_GEN_EPISODE_LENGTH_S = 300.0

# Layout randomisation for GENERATION, in metres of +-XY jitter around each object's
# authored pose. RoboLab's tasks are authored USD scenes whose default reset events
# restore identical poses, so without this every episode is a copy of the same
# trajectory -- measured: 10 episodes, 1 unique grasp pose, all 78 tokens long. Nothing
# flags it; every episode succeeds and the token statistics look healthy.
#
# 8 cm is chosen against the workspace rather than the scene: the whole pick-place path
# has to stay inside the region where the relative IK tracks cleanly (the axis probe was
# measured there), and both objects move independently, so the object-to-container
# distance varies by up to ~23 cm. Sampling is collision-aware, so a draw that would put
# the cube inside the bowl is rejected rather than producing a pre-solved episode.
DEFAULT_RANDOMIZE_XY_M = 0.08


class RobolabBackend(AtomicSimEnv):
    """Drives a single RoboLab env in relative differential-IK for the discretiser."""

    def __init__(
        self,
        handle: Any,
        agentview_camera: str = DEFAULT_AGENTVIEW_CAMERA,
        wrist_camera: str = DEFAULT_WRIST_CAMERA,
        agentview_flip: str = "none",
        wrist_flip: str = DEFAULT_WRIST_FLIP,
        agentview_rotation_degrees: int = 0,
        wrist_rotation_degrees: int = DEFAULT_WRIST_ROTATION_DEGREES,
        agentview_crop_aspect: Optional[float] = DEFAULT_CROP_ASPECT,
        wrist_crop_aspect: Optional[float] = DEFAULT_CROP_ASPECT,
        agentview_square_size: Optional[int] = None,
        wrist_square_size: Optional[int] = None,
        settle_steps: int = 8,
    ) -> None:
        self.handle = handle
        self.env = handle.env
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
        self.settle_steps = int(settle_steps)
        # The relative-IK action's `scale`: an action of 1.0 asks for this many metres.
        self.delta_bound_m = float(handle.ik_scale)
        self.env_id = str(handle.env_name)
        self.task_description = str(handle.task_description)
        self.targets = dict(handle.targets)
        self._obs: Optional[dict] = None
        self._terminated = False
        self._truncated = False
        # EE orientation the whole episode is held to; captured at reset (see reset()).
        self._quat_ref: Optional[np.ndarray] = None
        # Task DoneTerms lifted out of the termination manager by
        # :meth:`suspend_task_termination`; [] means "still armed" (eval behaviour).
        self._suspended_terms: list = []

    # -- construction --------------------------------------------------------
    @classmethod
    def make(
        cls,
        task: str,
        device: str = "cuda:0",
        seed: int = 0,
        instruction_type: str = "default",
        camera_preset: str = "WRIST_LEFT",
        renderer: str = "realtime",
        rendering_type: Optional[str] = None,
        output_dir: Optional[str] = None,
        episode_length_s: float = DEFAULT_GEN_EPISODE_LENGTH_S,
        randomize_xy_m: Optional[float] = DEFAULT_RANDOMIZE_XY_M,
        **kwargs: Any,
    ) -> "RobolabBackend":
        """Build the env for RoboLab task ``task`` through the deployment factory.

        ``task`` is the Task class name (``BananaInBowlTask``), the same string
        ``scripts/run_robolab_mvtoken.py --task`` takes. Isaac Sim must already be launched.

        The episode limit is raised well above the task's own (see
        :data:`DEFAULT_GEN_EPISODE_LENGTH_S`) because generation is not being scored
        against the benchmark clock -- and hitting that clock corrupts data silently.
        """
        handle = make_robolab_task(
            task=task,
            num_envs=1,
            device=device,
            seed=seed,
            instruction_type=instruction_type,
            camera_preset=camera_preset,
            renderer=renderer,
            rendering_type=rendering_type,
            output_dir=output_dir,
            # Data generation never reads the subtask score; skip the per-step predicates.
            enable_subtask=False,
            episode_length_s=episode_length_s,
            randomize_xy_m=randomize_xy_m,
        )
        return cls(handle, **kwargs)

    # -- sim lifecycle -------------------------------------------------------
    @property
    def unwrapped(self) -> Any:
        return self.env

    def reset(self, seed: int) -> None:
        """Reset the scene and settle.

        ``seed`` seeds the global RNG the reset events draw from, so a given seed
        reproduces a given layout. It canNOT re-seed ``env_cfg.seed`` -- that is baked at
        construction time -- so treat it as "advance the randomisation reproducibly",
        which is what a per-episode layout change needs.
        """
        try:
            self.env.seed(int(seed))
        except Exception:  # noqa: BLE001 -- older IsaacLab exposes no env.seed()
            pass
        try:
            self.env.reset_eval_state()
        except Exception:  # noqa: BLE001 -- plain ManagerBasedRLEnv has no eval state
            pass
        # Cleared BEFORE the reset: the hold action below goes through _action, which
        # would otherwise correct toward the PREVIOUS episode's reference while the arm is
        # being teleported home.
        self._quat_ref = None
        obs, terminated, truncated = reset_robolab(
            self.env, hold_action=self._action(np.zeros(3), 1.0), settle_steps=self.settle_steps
        )
        self._obs, self._terminated, self._truncated = obs, terminated, truncated
        # The home pose IS the contract's top-down orientation (measured: 0.000 deg from
        # vertical), so capture it as the reference every later step is pulled back to.
        self._quat_ref = rl_ee_quat(self.env)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    # -- state readback ------------------------------------------------------
    def tcp_pos(self) -> np.ndarray:
        return rl_tcp(self.env).astype(np.float64)

    def tcp_pose7(self) -> list[float]:
        return rl_tcp_pose7(self.env)

    def gripper_width(self) -> float:
        return rl_gripper_width(self.env)

    def ee_tilt_deg(self) -> float:
        """Degrees the gripper's approach axis is off straight-down.

        Recorded per episode because a tilted gripper is invisible to every other check:
        it grasps at an angle and the wrist camera stops looking down, yet the rollout
        still succeeds and every displacement statistic stays in range.
        """
        return ee_tilt_deg(rl_ee_quat(self.env))

    # -- actuation -----------------------------------------------------------
    def _action(self, delta_m: np.ndarray, grip_cmd: float) -> np.ndarray:
        """[dx, dy, dz, drx, dry, drz, gripper], the RoboLab relative-IK layout.

        ``grip_cmd`` arrives in the core's convention (+1 open / -1 close) and is mapped
        onto RoboLab's ``BinaryJointPositionZeroToOneAction`` rule (``> 0.5`` -> close).

        The rotation slots carry an ACTIVE correction back to the orientation captured at
        reset -- they used to be zeros, with a comment claiming that locked the rotation.
        It does not: in relative mode zeros mean "target the orientation you have now", so
        the IK's own orientation error becomes the next setpoint and accumulates. Measured
        with zeros: 24.5 deg off vertical by RELEASE. See
        :func:`core.sim.robolab_task.hold_orientation_rotvec`, which the deployment
        controller calls too -- this must stay ONE implementation, or generated data and
        served rollouts drift apart exactly the way the camera contract did.
        """
        scaled = np.asarray(delta_m, dtype=np.float64) / self.delta_bound_m
        gripper = 0.0 if float(grip_cmd) > 0 else 1.0
        rot = np.zeros(3)
        if self._quat_ref is not None:
            rot = hold_orientation_rotvec(
                self._quat_ref, rl_ee_quat(self.env)
            ) / self.delta_bound_m
        return np.asarray(
            [scaled[0], scaled[1], scaled[2], rot[0], rot[1], rot[2], gripper],
            dtype=np.float32,
        )

    def apply_delta(self, delta_m: np.ndarray, grip_cmd: float, max_cmd_m: float) -> None:
        delta = np.clip(np.asarray(delta_m, dtype=np.float64), -max_cmd_m, max_cmd_m)
        obs, terminated, truncated, _info = step_robolab(
            self.env, self._action(delta, grip_cmd)
        )
        self._obs = obs
        self._terminated = self._terminated or terminated
        self._truncated = self._truncated or truncated

    # -- observation / evaluation -------------------------------------------
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(agentview, wrist) with the FULL deployment transform contract applied.

        Both views, all four stages, one function -- see
        :func:`~scripts.trajectory.real2sim.atomic_tokenizer.prepared_pair`.
        """
        if self._obs is None:
            raise RuntimeError("grab_frames() before the first reset()/apply_delta()")
        return prepared_pair(
            self,
            rl_rgb(self._obs, self.agentview_camera),
            rl_rgb(self._obs, self.wrist_camera),
        )

    def frozen(self) -> bool:
        """RoboLab holds an env's final state once it terminates -- see AtomicSimEnv."""
        return bool(self._terminated or self._truncated)

    def success(self) -> bool:
        if self._suspended_terms:
            return self._evaluate_suspended()
        return bool(self._terminated or rl_success(self.env))

    # -- generation-time termination control ---------------------------------
    def suspend_task_termination(self) -> None:
        """Lift the task's success DoneTerm out of the termination manager.

        FOR GENERATION ONLY -- eval must keep RoboLab's own stopping rule.

        RoboLab freezes an env the instant it terminates (``RobolabEnv._reset_idx``:
        mark frozen, zero every future action, hold the final state). The task's success
        predicate is exactly what fires at RELEASE -- ``object_in_container`` with
        ``require_gripper_detached=True`` turns true within the settle steps the release
        itself holds -- so the env is already frozen when the generator asks for the
        post-release retreat, ``emit`` correctly refuses to record a move that cannot
        happen, and the episode ends ON the RELEASE frame.

        Measured on the last full batch: 127 of 130 rollouts ended with RELEASE as their
        final token. Downstream that frame gets used TWICE -- once labelled RELEASE and
        once as the synthesised terminal DONE (``rollout_to_llamafactory.py``) -- so the
        policy sees one image with two contradictory targets and learns to stop above the
        container while it is still holding the object.

        Suspending the term leaves ``time_out`` (the only other DoneTerm) in place, so a
        runaway episode still truncates and ``frozen()`` still means what it says; success
        is evaluated on demand instead, by calling the SAME predicate function with the
        same resolved params (:meth:`_evaluate_suspended`). Nothing about what counts as
        success changes -- only WHEN the simulator is allowed to stop.
        """
        if self._suspended_terms:
            return
        manager = getattr(self.env, "termination_manager", None)
        if manager is None:
            return
        names = list(getattr(manager, "_term_names", []))
        cfgs = list(getattr(manager, "_term_cfgs", []))
        keep_names, keep_cfgs = [], []
        for name, cfg in zip(names, cfgs):
            # ``time_out=True`` is IsaacLab's flag for "this is truncation, not the task
            # ending"; everything else IS the task's own success predicate.
            if bool(getattr(cfg, "time_out", False)):
                keep_names.append(name)
                keep_cfgs.append(cfg)
            else:
                self._suspended_terms.append((name, cfg))
        if not self._suspended_terms:
            return
        manager._term_names = keep_names
        manager._term_cfgs = keep_cfgs
        # Class-based terms are also called for reset()/__call__; drop the suspended ones
        # from that list too or they keep being invoked with no cfg entry.
        suspended_cfgs = {id(cfg) for _n, cfg in self._suspended_terms}
        manager._class_term_cfgs = [
            c for c in getattr(manager, "_class_term_cfgs", [])
            if id(c) not in suspended_cfgs
        ]
        print(
            "[robolab-backend] task termination suspended for generation: "
            f"{[n for n, _c in self._suspended_terms]} (time_out kept)",
            flush=True,
        )

    def _evaluate_suspended(self) -> bool:
        """Run the suspended DoneTerms on the CURRENT state, exactly as the manager would.

        The cfgs come out of the manager already prepared (``SceneEntityCfg`` params
        resolved against the scene), so calling ``cfg.func(env, **cfg.params)`` is the
        same computation ``TerminationManager.compute`` performs -- which is the point:
        generated episodes must be scored by RoboLab's predicate, not by a reimplementation
        of it.
        """
        for _name, cfg in self._suspended_terms:
            try:
                value = cfg.func(self.env, **cfg.params)
            except Exception:  # noqa: BLE001 -- a predicate that cannot run is not success
                return False
            if bool(np.asarray(_to_numpy(value)).reshape(-1)[0]):
                return True
        return False

    @property
    def truncated(self) -> bool:
        """RoboLab's own ``episode_length_s`` time-out fired (env is frozen)."""
        return bool(self._truncated)

    # -- privileged scene access (task construction) -------------------------
    @property
    def world(self) -> Any:
        """RoboLab's :class:`WorldState` -- poses, bounding boxes, contacts."""
        from robolab.core.world.world_state import get_world

        return get_world(self.env)

    def object_pos(self, name: str) -> np.ndarray:
        """A named scene object's root position (3,), env-local frame."""
        pos, _quat = self.world.get_pose(name, is_relative=True, env_id=0)
        return np.asarray(_to_numpy(pos), dtype=np.float64).reshape(-1)[:3]

    def object_centroid(self, name: str) -> np.ndarray:
        """Geometric centre of the object's oriented bounding box, env-local.

        Preferred over :meth:`object_pos` for grasping: a USD asset's root frame is
        wherever the author put it (often the base, sometimes off the mesh entirely),
        while the OBB centre is on the object.
        """
        return np.asarray(
            _to_numpy(self.world.get_centroid(name, env_id=0)), dtype=np.float64
        ).reshape(-1)[:3]

    def object_aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(min_corner, max_corner) of the object's bounding box in its LOCAL frame.

        WARNING -- this does NOT follow the object. ``WorldState.get_aabb`` returns cached
        *local* geometry, so for anything that has moved since the scene loaded (in
        particular, an object currently held by the gripper) it reports where the object
        STARTED. Use :meth:`object_extent` for a live answer; this method is kept only for
        querying static scene furniture.
        """
        lo, hi = self.world.get_aabb(name)
        return (
            np.asarray(_to_numpy(lo), dtype=np.float64).reshape(-1)[:3],
            np.asarray(_to_numpy(hi), dtype=np.float64).reshape(-1)[:3],
        )

    def object_extent(self, name: str) -> tuple[float, float]:
        """LIVE (bottom_z, top_z) of the object, in the env-local frame.

        Built from the live OBB centroid plus the cached size, because those are the two
        pieces RoboLab exposes that are individually correct: ``get_centroid`` transforms
        by the object's current pose, while ``get_dimensions`` is a pose-independent size.
        ``get_aabb`` looks like the obvious call and is a trap -- it is pose-independent
        too, so a held object still reports its table-top height, which silently inflates
        any "how far below the flange is the object" computation.
        """
        centre = self.object_centroid(name)
        half_h = float(self.object_dimensions(name)[2]) / 2.0
        return float(centre[2] - half_h), float(centre[2] + half_h)

    def object_dimensions(self, name: str) -> np.ndarray:
        return np.asarray(
            _to_numpy(self.world.get_dimensions(name)), dtype=np.float64
        ).reshape(-1)[:3]

    def in_contact(self, body1: str, body2: str) -> bool:
        try:
            return bool(_to_numpy(self.world.in_contact(body1, body2, env_id=0)))
        except Exception:  # noqa: BLE001 -- bodies without contact sensors
            return False


def _to_numpy(value: Any) -> Any:
    from core.sim.robolab_task import to_np

    return to_np(value)
