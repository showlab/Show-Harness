"""RoboLab (NVIDIA Isaac Lab) task/env glue for the MVTOKEN atomic-token policy.

Sibling of :mod:`core.sim.maniskill_task`: builds a single
RoboLab benchmark task in the **relative differential-IK** action mode
(``DroidRelIKActionCfg``) with the RGB image observation group, and exposes the few
accessors the runner needs (RGB per camera, success flag, TCP position, gripper width).

Why rel-IK and not RoboLab's default joint-position mode
--------------------------------------------------------
RoboLab registers its 120 benchmark tasks against ``DroidJointPositionActionCfg`` by
default (7 joint targets + gripper), which is what pi0/GR00T emit. MVTOKEN emits atomic
*Cartesian* tokens, so we register the same tasks against RoboLab's own
``DroidRelIKActionCfg`` instead -- an ``(dx, dy, dz, drx, dry, drz, gripper)`` action where
the first three are a base-frame end-effector displacement. Holding the three rotation
DOFs at zero gives exactly the "rotation locked, XYZ only" controller the atomic-token
policy assumes, the same contract as ManiSkill's ``pd_ee_delta_pos``. RoboLab ships the
registration helper for this (``auto_register_droid_rel_ik_envs``), so no fork is needed.

Import-order constraint (the reason this module imports nothing at module scope)
--------------------------------------------------------------------------------
Isaac Lab may only be imported *after* ``AppLauncher`` has started the Omniverse Kit app,
and ``cv2`` must be imported before ``isaaclab`` or the Kit runtime's own OpenCV clashes
with it (every script in the RoboLab repo carries that same comment). So:

    from core.sim.robolab_task import launch_isaac, make_robolab_task
    app = launch_isaac(headless=True)          # MUST come first
    handle = make_robolab_task(task="BananaInBowlTask", ...)

Every robolab/isaaclab import in this file therefore lives inside a function body, exactly
like ``core.sim.maniskill_task`` defers ``import mani_skill``. Merely importing this module in
an environment with no Isaac Sim is safe.

The robot is a Franka + Robotiq 2F-85 (RoboLab's "droid" embodiment), not the Panda hand
used by ManiSkill, so the gripper-width accessor converts the Robotiq
``finger_joint`` angle into metres rather than summing two prismatic finger joints.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Where the RoboLab checkout lives. Override with ROBOLAB_ROOT (the same pattern
# core.sim.maniskill_scenes uses for RLINF_ROOT).
DEFAULT_ROBOLAB_ROOT = Path.home() / "RoboLab"

# Robotiq 2F-85 stroke. RoboLab drives the gripper through a single ``finger_joint``
# whose 0 rad is fully OPEN and pi/4 rad is fully CLOSED (see
# ``BinaryJointPositionZeroToOneActionCfg`` in robolab/robots/droid.py), so the opening in
# metres is a linear map off that angle. Used only to give plugins.auto_release a width in
# the same units as every other embodiment -- the exact Robotiq linkage is slightly
# non-linear, which does not matter for "is the gripper closed on nothing".
ROBOTIQ_STROKE_M = 0.085
ROBOTIQ_CLOSED_RAD = np.pi / 4

# The camera presets RoboLab ships (robolab/registrations/droid/camera_presets.py). The
# observation key for a camera is its attribute name, so WRIST_LEFT yields
# ``over_shoulder_left_camera`` + ``wrist_cam`` -- the two views MVTOKEN needs.
CAMERA_PRESETS = ("WRIST", "WRIST_LEFT", "WRIST_RIGHT", "WRIST_LEFT_RIGHT", "WRIST_LEFT_RIGHT_HEAD", "LEFT_RIGHT")


@dataclass
class RobolabTaskHandle:
    env: Any
    env_cfg: Any
    task_description: str
    env_name: str
    task: str
    action_dim: int
    ik_scale: float
    # Objects the task's own subtask spec says to manipulate, e.g.
    # {"objects": ["banana"], "container": "bowl"}. Empty when the task declares no
    # pick_and_place subtask. Used by the real2sim oracle (Path B), not by eval.
    targets: dict = field(default_factory=dict)


# -- installation / app lifecycle -------------------------------------------
def robolab_root() -> Path:
    root = Path(os.environ.get("ROBOLAB_ROOT", DEFAULT_ROBOLAB_ROOT))
    if not (root / "robolab").is_dir():
        raise SystemExit(
            f"RoboLab checkout not found at {root} (looked for {root}/robolab). "
            "Clone https://github.com/NVLabs/RoboLab and set ROBOLAB_ROOT."
        )
    return root


def ensure_robolab_path() -> Path:
    """Put the RoboLab checkout on ``sys.path``.

    RoboLab is normally ``uv sync``-installed into its own ``.venv`` (in which case this
    is a no-op), but the eval entry point may also be run with that interpreter from this
    repo's working directory, where ``robolab``/``policies`` are not importable. Inserting
    the checkout root covers both.
    """
    root = robolab_root()
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


# Native libraries Isaac Sim dlopen()s that are NOT pip dependencies, mapped to the Ubuntu
# package that provides them. Missing ones are a nasty failure: Isaac Sim reports the
# dlopen error as a mere [Error] log line and then SEGFAULTS a few plugins later, so the
# visible symptom is a 1500-line crash dump whose backtrace points at rtx.mdltranslator --
# nowhere near the actual cause. Checking up front turns that into one readable message.
_NATIVE_DEPS: dict[str, str] = {
    # libneuray.so (the MDL material engine behind rtx.neuraylib) links against GLU.
    "libGLU.so.1": "libglu1-mesa",
}


def _check_native_deps() -> None:
    import ctypes.util

    missing = []
    for lib, package in _NATIVE_DEPS.items():
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append((lib, package))
    if not missing:
        return
    names = ", ".join(lib for lib, _ in missing)
    packages = " ".join(pkg for _, pkg in missing)
    raise SystemExit(
        f"Isaac Sim needs native libraries that are not installed: {names}.\n"
        f"Without them the RTX material stack fails to load and Isaac Sim segfaults during\n"
        f"stage creation (the crash dump blames rtx.mdltranslator, which is a red herring).\n\n"
        f"With root:      sudo apt install {packages}\n"
        f"Without root:   apt-get download {packages} \\\n"
        f"                  && dpkg -x {packages}_*.deb /tmp/deps \\\n"
        f"                  && cp -a /tmp/deps/usr/lib/x86_64-linux-gnu/lib*.so* "
        f"{robolab_root()}/.deps/lib/\n"
        f"then re-run with LD_LIBRARY_PATH={robolab_root()}/.deps/lib:$LD_LIBRARY_PATH\n"
        f"(LD_LIBRARY_PATH must be set BEFORE the interpreter starts -- the dynamic linker\n"
        f"reads it at process start, so exporting it from inside Python does not work.)"
    )


def launch_isaac(
    headless: bool = True,
    device: str = "cuda:0",
    enable_cameras: bool = True,
    renderer_kwargs: Optional[dict] = None,
) -> Any:
    """Start the Omniverse Kit app and return the ``simulation_app`` handle.

    MUST be called before anything imports ``isaaclab`` / ``robolab``; see the module
    docstring. The returned app has to be ``.close()``d at the end of the process or the
    interpreter hangs on exit.

    ``enable_cameras`` is non-optional in practice: without it Isaac Lab skips RTX sensor
    rendering and every camera observation comes back empty.
    """
    ensure_robolab_path()
    _check_native_deps()
    import argparse

    import cv2  # noqa: F401  -- must be imported before isaaclab. Do not remove.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    # Parse an EMPTY argv on purpose: this repo's entry point owns the command line, and
    # AppLauncher would otherwise choke on flags it does not know (--version, --model, ...).
    app_args = parser.parse_args([])
    app_args.headless = bool(headless)
    app_args.enable_cameras = bool(enable_cameras)
    app_args.device = str(device)
    for key, value in (renderer_kwargs or {}).items():
        setattr(app_args, key, value)
    return AppLauncher(app_args).app


# -- env construction --------------------------------------------------------
def make_robolab_task(
    task: str,
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    seed: int = 0,
    instruction_type: str = "default",
    camera_preset: str = "WRIST_LEFT",
    robot: str = "franka",
    task_dirs: Optional[list[str]] = None,
    renderer: str = "realtime",
    rendering_type: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    enable_subtask: bool = False,
    episode_length_s: Optional[float] = None,
    randomize_xy_m: Optional[float] = None,
    randomize_margin_m: float = 0.02,
    verbose: bool = False,
) -> RobolabTaskHandle:
    """Register ``task`` against the relative-IK action space and construct its env.

    ``output_dir`` redirects RoboLab's own artefacts (``env_cfg.json``, and anything else
    written through ``robolab.constants.get_output_dir``) into this repo's rollout
    directory, so one run leaves one directory rather than two.

    ``enable_subtask`` drives RoboLab's per-step subtask-progress predicates. They are
    informative (partial credit / a reason string) but cost extra physics queries per
    step, and MVTOKEN eval only consumes the binary success, so this defaults OFF.

    ``episode_length_s`` overrides the task's own time limit. Leave it None for EVAL --
    the benchmark's limit is part of the task. Raise it for DATA GENERATION: a task like
    RubiksCubeTask allows 40 s = 600 control steps, while the atomic executor spends ~12
    steps per token closing the relative-IK loop, so a ~50-token episode runs straight
    into the time-out. What that looks like is NOT an error: IsaacLab resets the scene
    mid-episode, so the gripper springs back to fully open and the carried object
    teleports back to its start pose -- indistinguishable from "the object slipped out of
    the gripper" unless you notice both values are exactly their initial ones.

    ``randomize_xy_m`` gives every reset a different LAYOUT: the task's own manipulable
    objects and its container/surface are re-sampled uniformly in +-this many metres of X
    and Y around their authored poses, with collision-aware rejection so nothing spawns
    inside anything else. Leave it None for EVAL (the benchmark's layout is part of the
    task); set it for DATA GENERATION.

    Without it, a RoboLab dataset has no layout diversity at all. RoboLab's 120 tasks are
    authored USD scenes and their default reset events restore the SAME poses every time,
    so ten episodes come back as ten copies of one trajectory -- identical token
    sequences, one grasp pose. Nothing reports this: every episode succeeds, the token
    statistics look healthy, and the count of "episodes" is what you asked for.

    Implemented through RoboLab's own ``reset_pose_uniform`` and the ``events=`` hook of
    ``create_env`` (merged with, not replacing, the task's existing events), so no RoboLab
    file is patched and any single-object pick-and-place task gets it for free. The object
    list comes from the task's OWN subtask declaration, the same source
    :func:`task_targets` reads, so adding a task needs no change here.
    """
    ensure_robolab_path()

    import robolab.constants
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_rel_ik import (
        auto_register_droid_rel_ik_envs,
    )
    from robolab.robots.droid import DroidRelIKActionCfg

    robolab.constants.VERBOSE = bool(verbose)
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = bool(enable_subtask)
    # RECORD_IMAGE_DATA would additionally stash every camera frame into the HDF5
    # recorder; this repo writes its own per-step PNGs via EpisodeLogger.
    robolab.constants.RECORD_IMAGE_DATA = False
    if output_dir is not None:
        robolab.constants.set_output_dir(str(output_dir))

    if robot == "franka":
        ik_scale = _register_franka_rel_ik(task, camera_preset, task_dirs)
    elif robot == "droid":
        kwargs: dict[str, Any] = {"task": task, "cameras": _camera_preset(camera_preset)}
        if task_dirs:
            kwargs["task_dirs"] = list(task_dirs)
        auto_register_droid_rel_ik_envs(**kwargs)
        ik_scale = float(getattr(DroidRelIKActionCfg().arm_action, "scale", 0.5))
    else:
        raise SystemExit(f"Unknown robot {robot!r}; choose 'franka' or 'droid'.")

    env_names = get_envs(task=task)
    if not env_names:
        raise SystemExit(
            f"No RoboLab environment registered for task {task!r}. Task names are the "
            "class names in <ROBOLAB_ROOT>/robolab/tasks/benchmark/*.py, e.g. "
            "'BananaInBowlTask'."
        )
    env_name = env_names[0]

    events = None
    if randomize_xy_m:
        events = _layout_randomisation_events(
            task, task_dirs, float(randomize_xy_m), float(randomize_margin_m)
        )

    env, env_cfg = create_env(
        env_name,
        device=device,
        seed=seed,
        num_envs=int(num_envs),
        use_fabric=True,
        events=events,
        instruction_type=instruction_type,
        policy="mvtoken",
        renderer=renderer,
        rendering_mode=rendering_type,
    )

    if episode_length_s is not None:
        # max_episode_length is derived from cfg.episode_length_s on every read, and the
        # time_out DoneTerm compares episode_length_buf against it, so setting it here --
        # after construction -- takes effect. env.cfg IS env_cfg, but assign to both so a
        # caller inspecting the handle sees the value it is actually running with.
        env.cfg.episode_length_s = float(episode_length_s)
        env_cfg.episode_length_s = float(episode_length_s)

    # NOTE: do NOT set env.recorder_manager = None to skip RoboLab's HDF5 episode recorder.
    # RobolabEnv's own uses are all guarded by `is not None`, but upstream
    # ManagerBasedEnv.reset() calls `self.recorder_manager.record_pre_reset(...)`
    # unconditionally, so clearing it makes the FIRST reset raise AttributeError. The
    # recorder writes <output>/data.hdf5 that this pipeline never reads; the cost is
    # tolerable, but note it takes an exclusive file lock, so two runs sharing an output
    # directory will report a corrupt file and clobber each other -- run them serially or
    # give each its own output_dir.

    action_dim = (
        getattr(getattr(env, "action_manager", None), "total_action_dim", None)
        or env.action_space.shape[-1]
    )
    return RobolabTaskHandle(
        env=env,
        env_cfg=env_cfg,
        task_description=str(env_cfg.instruction),
        env_name=env_name,
        task=task,
        action_dim=int(action_dim),
        # The action->metres scale of the registered arm action (0.5 on both
        # embodiments): an action of 1.0 commands a 0.5 m IK target delta, and the
        # controller divides by this so its config speaks plain metres. Read off the cfg
        # rather than hardcoded, so retuning it cannot silently halve our step size.
        ik_scale=ik_scale,
        targets=task_targets(env_cfg),
    )


def _layout_randomisation_events(
    task: str, task_dirs: Optional[list[str]], xy_m: float, margin_m: float
) -> dict:
    """``{name: EventTerm}`` re-sampling the task's own objects on every reset.

    Passed to ``create_env(events=...)``, which MERGES rather than replaces, so the
    task's existing reset events (``reset_scene_to_default`` and friends) survive.

    Which assets to move is read from the task's OWN subtask declaration -- the same
    source :func:`task_targets` uses -- so this works for any pick-and-place task in
    RoboLab-120 without a per-task table. The container is randomised too: moving only
    the object would still leave every episode placing into the same spot.

    Sampling is collision-aware (RoboLab's ``reset_pose_uniform`` rejects and re-samples,
    falling back to the authored pose after ``max_retries``), which matters because the
    naive alternative -- independent uniform draws -- happily spawns the cube inside the
    bowl and produces episodes that are already solved at t=0.

    Z is deliberately NOT randomised: the authored poses have each object resting on its
    surface, and an offset there would either float it or bury it in the table.
    """
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
    from robolab.core.events.reset_pose import reset_pose_uniform
    from robolab.core.task.task_utils import load_task_from_file, resolve_task_path
    from isaaclab.managers import EventTermCfg as EventTerm

    subdirs = list(task_dirs) if task_dirs else DEFAULT_TASK_SUBFOLDERS
    task_class = None
    for subdir in [*subdirs, ""]:
        try:
            root = Path(TASK_DIR) / subdir if subdir else Path(TASK_DIR)
            path, _ = resolve_task_path(task, str(root))
            task_class = load_task_from_file(path)
            break
        except Exception:  # noqa: BLE001 -- try the next subfolder
            continue
    if task_class is None:
        raise SystemExit(
            f"randomize_xy_m was requested but task {task!r} could not be loaded to find "
            "its objects; pass task_dirs or drop the randomisation."
        )

    targets = task_targets(task_class)
    assets = list(targets.get("objects") or [])
    for key in ("container", "surface"):
        if targets.get(key):
            assets.append(str(targets[key]))
    if not assets:
        raise SystemExit(
            f"randomize_xy_m was requested but task {task!r} declares no objects in its "
            "subtasks, so there is nothing to randomise."
        )

    # Deliberately named "reset" so that ``merge_events_cfg`` OVERRIDES RoboLab's default
    # reset term instead of adding a second one. Adding is what a natural name does, and
    # it silently does nothing: both terms have mode="reset", the manager ran them as
    # ['randomize_layout', 'reset'], and ``reset_scene_to_default`` -- running second --
    # put every object straight back on its authored pose. The layout came out
    # byte-identical across seeds with no error and no warning anywhere.
    #
    # Overriding is safe because ``reset_to_default_otherwise=True`` makes
    # ``reset_pose_uniform`` a full replacement: it resets every asset NOT in the list
    # (including the robot articulation and its joint state) to default first, then
    # samples the listed ones. This is also why RoboLab's own randomised task variants
    # REPLACE the whole events config rather than merging into it.
    return {
        "reset": EventTerm(
            func=reset_pose_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-xy_m, xy_m), "y": (-xy_m, xy_m), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": assets,
                "use_collision_check": True,
                "collision_margin": float(margin_m),
                "max_retries": 100,
                "reset_to_default_otherwise": True,
            },
        )
    }


def _register_franka_rel_ik(
    task: str, camera_preset: str, task_dirs: Optional[list[str]]
) -> float:
    """Register ``task`` against the Panda-hand embodiment; returns the action scale.

    Mirrors ``robolab.registrations.droid.auto_env_registrations_rel_ik`` step for step --
    build the image + proprio observation groups, then hand the whole bundle to
    ``auto_discover_and_create_cfgs`` -- but with our robot/action/camera configs instead
    of the Robotiq ones. The physics timing (dt, decimation, render_interval) is copied
    verbatim, because those numbers are what the step-size calibration was measured
    against.
    """
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    from core.sim.robolab_franka import (
        FrankaFrontCameraCfg,
        FrankaPandaCfg,
        FrankaProprioCfg,
        FrankaRelIKActionCfg,
        FrankaWristCameraCfg,
        contact_gripper,
    )

    # Swap RoboLab's cameras for the ManiSkill-matched pair:
    #   WristCameraCfg            -> FrankaWristCameraCfg  (gripper-mounted, top-down)
    #   OverShoulderLeftCameraCfg -> FrankaFrontCameraCfg  (RLinf's calibrated front view)
    # The over-shoulder camera is DROID's placement (off to the left, different lens); the
    # front camera is the one ManiSkill uses as its agentview, so the third-person view is
    # the same viewpoint in both simulators. `front_cam` is world-fixed, so unlike the
    # wrist it stays in the scene camera list.
    _SWAP = {
        "WristCameraCfg": FrankaWristCameraCfg,
        "OverShoulderLeftCameraCfg": FrankaFrontCameraCfg,
    }
    cameras = [_SWAP.get(c.__name__, c) for c in _camera_preset(camera_preset)]
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": FrankaProprioCfg(),
        "viewport_cam": ViewportCfg(),
    })
    # The wrist camera is robot-mounted (already on FrankaPandaCfg). Listing it as a scene
    # camera too would order it before `robot` in dataclass field order, spawning the
    # camera before its parent prim exists.
    scene_cameras = [c for c in cameras if c is not FrankaWristCameraCfg]

    actions = FrankaRelIKActionCfg()
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=list(task_dirs) if task_dirs else DEFAULT_TASK_SUBFOLDERS,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=actions,
        robot_cfg=FrankaPandaCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )
    return float(getattr(actions.arm_action, "scale", 0.5))


def _camera_preset(name: str) -> list:
    from robolab.registrations.droid import camera_presets

    key = str(name).upper()
    preset = getattr(camera_presets, key, None)
    if preset is None:
        raise SystemExit(f"Unknown camera preset {name!r}; choices: {list(CAMERA_PRESETS)}")
    return preset


def task_targets(env_cfg: Any) -> dict:
    """What the task's own subtask spec says to pick and where to put it.

    RoboLab tasks declare success twice: once as a termination predicate and once as a
    ``subtasks`` list built from composites like
    ``pick_and_place(object=["banana"], container="bowl")``. The composite stores its
    arguments as ``functools.partial`` keywords, so the manipulated object(s) and the
    target container can be recovered without parsing the task source. Returns
    ``{"objects": [...], "container": str}``, or ``{}`` when the task uses some other
    subtask shape (stacking, ordering, ...).

    Eval does not need this -- it is the hook the Path-B oracle uses to know what to grasp
    (see ``scripts/trajectory/real2sim/robolab/oracle.py``).
    """
    subtasks = getattr(env_cfg, "subtasks", None)
    if not subtasks:
        return {}
    objects: list[str] = []
    container: Optional[str] = None
    surface: Optional[str] = None
    for subtask in subtasks:
        conditions = getattr(subtask, "conditions", None)
        if not isinstance(conditions, dict):
            continue
        for obj_name, entries in conditions.items():
            for entry in entries if isinstance(entries, (list, tuple, set)) else [entries]:
                func = entry[0] if isinstance(entry, tuple) else entry
                keywords = getattr(func, "keywords", None) or {}
                if "container" in keywords:
                    container = container or str(keywords["container"])
                if "reference_object" in keywords:
                    surface = surface or str(keywords["reference_object"])
            if isinstance(obj_name, str) and obj_name not in objects:
                objects.append(obj_name)
    targets: dict = {}
    if objects:
        targets["objects"] = objects
    if container:
        targets["container"] = container
    if surface:
        targets["surface"] = surface
    return targets


# -- tensor/obs helpers ------------------------------------------------------
def to_np(value: Any) -> np.ndarray:
    """Tensor (torch or warp) -> numpy, batch dim kept."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 -- torch is always present under Isaac Sim
        pass
    try:
        import warp as wp

        if isinstance(value, wp.array):
            return wp.to_torch(value).detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(value)


def rl_rgb(obs: dict, camera_name: str, env_id: int = 0) -> np.ndarray:
    """One camera's RGB as HWC uint8.

    RoboLab image observations live under the ``image_obs`` group, keyed by the camera
    config's attribute name (``over_shoulder_left_camera``, ``wrist_cam``, ...) and shaped
    ``[num_envs, H, W, 3]``.
    """
    group = obs.get("image_obs")
    if group is None:
        raise KeyError(f"obs has no 'image_obs' group; keys: {sorted(obs)}")
    try:
        rgb = group[camera_name]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"camera {camera_name!r} not in image_obs; available: {sorted(group)}"
        ) from exc
    arr = to_np(rgb)
    if arr.ndim == 4:
        arr = arr[env_id]
    return np.ascontiguousarray(arr.astype(np.uint8))


# The body the relative-IK action drives, per embodiment. Order matters: the Panda hand is
# checked first because that is this repo's default (core.sim.robolab_franka); base_link is
# the Robotiq flange RoboLab's stock DroidCfg uses. Hardcoding either one makes the other
# embodiment raise ValueError deep inside the axis probe -- which Isaac Sim's
# SimulationApp.close() then swallows into a silent exit 0.
EE_BODY_CANDIDATES = ("panda_hand", "base_link")


def _ee_body_index(robot: Any) -> int:
    names = list(robot.data.body_names)
    for candidate in EE_BODY_CANDIDATES:
        if candidate in names:
            return names.index(candidate)
    raise SystemExit(
        f"No end-effector body found on the robot. Looked for {list(EE_BODY_CANDIDATES)}; "
        f"the articulation has {names}."
    )


def rl_tcp(env: Any, env_id: int = 0) -> np.ndarray:
    """End-effector position (3,) in the env-local frame.

    Reads the controlled body's pose straight off the articulation -- the same body the
    relative-IK action drives -- but without the observation group's noise terms, so the
    axis probe and the step-size calibration measure the true displacement.
    """
    robot = env.scene["robot"]
    body_idx = _ee_body_index(robot)
    pos = to_np(robot.data.body_pos_w)[:, body_idx, :]
    origins = to_np(env.scene.env_origins)[:, 0:3]
    return (pos - origins)[env_id].astype(float)


def rl_tcp_pose7(env: Any, env_id: int = 0) -> list[float]:
    """``[x, y, z, qw, qx, qy, qz]`` -- the real recorder's ``ee_pose`` field."""
    robot = env.scene["robot"]
    body_idx = _ee_body_index(robot)
    quat = to_np(robot.data.body_quat_w)[:, body_idx, :][env_id]
    return [round(float(v), 5) for v in (*rl_tcp(env, env_id), *quat)]


def rl_ee_quat(env: Any, env_id: int = 0) -> np.ndarray:
    """The EE body's world orientation as ``[qw, qx, qy, qz]`` (full precision)."""
    robot = env.scene["robot"]
    return to_np(robot.data.body_quat_w)[:, _ee_body_index(robot), :][env_id].astype(
        np.float64
    )


def ee_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the hand's approach axis (local +Z) and straight down.

    The MVTOKEN contract is translation-only, so this should stay near zero for a whole
    episode. It is the single number that tells you whether "rotation locked" is actually
    holding -- worth asserting on, because a tilted gripper is not an error anywhere: it
    grasps at an angle, the wrist camera stops looking down, and the data still passes
    every displacement statistic.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(axis @ np.array([0.0, 0.0, -1.0])),
                                              -1.0, 1.0))))


# Proportional gain and per-step clamp for :func:`hold_orientation_rotvec`.
ORIENT_HOLD_GAIN = 1.0
ORIENT_HOLD_MAX_RAD = 0.15


def hold_orientation_rotvec(
    quat_ref: np.ndarray,
    quat_cur: np.ndarray,
    gain: float = ORIENT_HOLD_GAIN,
    max_rad: float = ORIENT_HOLD_MAX_RAD,
) -> np.ndarray:
    """World-frame axis-angle that pulls ``quat_cur`` back to ``quat_ref``.

    Feeds the ``(drx, dry, drz)`` slots of RoboLab's relative-IK action. **This is what
    makes "rotation locked" true.** Commanding zeros there does NOT lock the orientation:
    in relative mode a zero rotation delta means "target = the orientation you have right
    now", an INTEGRATING reference, so any orientation error the IK introduces silently
    becomes the new setpoint and is never corrected.

    And the IK does introduce it. The position command is deliberately overdriven (RoboLab's
    relative IK only achieves ~28% of what it is asked for), which leaves the DLS solver
    saturated, and a saturated least-squares solution buys position progress with
    orientation error. Measured on RubiksCubeTask with zeros in these slots: the gripper
    left reset perfectly vertical and was **24.5 deg off** by the time it released -- with
    no error, no failed episode, and no displacement statistic out of range.

    Measured effect of this correction (same token sequence, world frame, gain 1.0):
    worst tilt 24.5 -> 16.9 deg, final 24.5 -> 5.5 deg, at a 2.5% cost in travel. Note the
    correction alone does NOT fix the transient -- during motion the saturated IK keeps
    taking orientation back, and the correction only catches up once the arm stops. Cutting
    the per-step position command (same total, more control steps) is what fixes that:
    0.018 m x 32 steps instead of 0.072 x 8 gives worst 2.5 / final 1.5 deg. Use both.

    The frame is not documented; it was determined by experiment. The body-frame
    alternative diverges loudly (worst tilt 112 deg), which is the useful property of
    testing both: a wrong guess grows the error instead of shrinking it.
    """
    rw, rx, ry, rz = (float(v) for v in np.asarray(quat_ref, dtype=np.float64))
    cw, cx, cy, cz = (float(v) for v in np.asarray(quat_cur, dtype=np.float64))
    # cur^-1 (unit quaternion -> conjugate)
    iw, ix, iy, iz = cw, -cx, -cy, -cz
    # q_err = ref (x) cur^-1: the WORLD-frame rotation taking cur onto ref.
    q_err = np.array([
        rw * iw - rx * ix - ry * iy - rz * iz,
        rw * ix + rx * iw + ry * iz - rz * iy,
        rw * iy - rx * iz + ry * iw + rz * ix,
        rw * iz + rx * iy - ry * ix + rz * iw,
    ])
    if q_err[0] < 0.0:                      # shortest arc
        q_err = -q_err
    vec = q_err[1:]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(norm, float(np.clip(q_err[0], -1.0, 1.0))))
    rotvec = (vec / norm) * angle * float(gain)
    mag = float(np.linalg.norm(rotvec))
    return rotvec * (float(max_rad) / mag) if mag > float(max_rad) else rotvec


def rl_gripper_width(env: Any, env_id: int = 0) -> float:
    """Gripper opening in metres, read straight off the articulation.

    Handles both embodiments, because they report opening in different units:

    * **Panda hand** (``core.sim.robolab_franka``, the default) -- two prismatic
      ``panda_finger_joint*`` in metres; the opening is their sum.
    * **Robotiq 2F-85** (RoboLab's stock ``DroidCfg``) -- one revolute ``finger_joint``
      where 0 rad is fully open and pi/4 fully closed, mapped onto the 85 mm stroke.

    Deliberately NOT read from ``obs["proprio_obs"]["gripper_pos"]``: that observation
    carries a Gaussian noise term (std 0.05 on a 0..1 scale, i.e. ~4 mm), which would make
    an empty-grasp threshold flap.
    """
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    qpos = to_np(robot.data.joint_pos)

    finger_idx = [i for i, n in enumerate(names) if n.startswith("panda_finger_joint")]
    if finger_idx:  # Panda hand: prismatic, already metres
        return float(qpos[env_id, finger_idx].sum())

    if "finger_joint" in names:  # Robotiq 2F-85: revolute, map onto the stroke
        angle = float(qpos[env_id, names.index("finger_joint")])
        frac = np.clip(angle / ROBOTIQ_CLOSED_RAD, 0.0, 1.0)
        return float((1.0 - frac) * ROBOTIQ_STROKE_M)

    return 0.0


def rl_success(env: Any, env_id: int = 0) -> bool:
    """The task's success predicate for one env.

    RoboLab splits its two DoneTerms: ``time_out`` is flagged ``time_out=True`` so it
    lands in *truncated*, while the task's own predicate lands in *terminated*. So
    ``terminated`` IS success -- exactly how ``RobolabEnv._reset_idx`` records its per-env
    result. Once an env terminates it is frozen (held at its final state, actions zeroed),
    and its stored result stays available through ``get_env_results()``.
    """
    stored = getattr(env, "_env_results", {}).get(env_id)
    if stored is not None:
        return bool(stored)
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return False
    return bool(to_np(manager.terminated).reshape(-1)[env_id])


def rl_instruction(env_cfg: Any) -> str:
    instruction = getattr(env_cfg, "instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("default", ""))
    return str(instruction)


# -- stepping ----------------------------------------------------------------
def batched_action(action: np.ndarray, num_envs: int, device: Any) -> Any:
    """(action_dim,) numpy -> (num_envs, action_dim) torch tensor on the sim device."""
    import torch

    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (num_envs, 1))
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def ensure_timeline_playing(env: Any = None, max_updates: int = 2000) -> bool:
    """Pump the Kit app until the timeline is playing. Returns whether it is.

    NOT optional, and the failure mode is silent: Isaac Lab advances physics only while
    the Omniverse timeline plays, and after ``env.reset()`` in a headless app it can still
    be stopped. Stepping then returns valid-looking observations from a frozen scene --
    actions apply to nothing, the arm never moves, and an axis probe reads exactly 0.0000 m
    on every token (which is how this was found).

    RoboLab's own ``robolab/eval/episode.py`` does the same pump at the top of every step;
    :func:`step_robolab` therefore calls it too rather than only once after reset.
    """
    try:
        import omni.kit.app
        import omni.timeline
    except Exception:  # noqa: BLE001 -- not under a Kit app (unit tests)
        return False
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        return True
    app = omni.kit.app.get_app()
    for _ in range(int(max_updates)):
        app.update()
        if timeline.is_playing():
            return True
    return timeline.is_playing()


def step_robolab(env: Any, action: np.ndarray) -> tuple[dict, bool, bool, dict]:
    """One env step; returns ``(obs, terminated, truncated, info)`` for env 0.

    One step advances ``decimation`` (8) physics substeps at ``dt`` 1/120 s, i.e. one
    1/15 s control period -- that is the granularity the relative-IK target is refreshed
    at, and what ``sim_steps_per_decision`` counts.
    """
    ensure_timeline_playing(env)
    obs, _reward, terminated, truncated, info = env.step(
        batched_action(action, env.num_envs, env.device)
    )
    return (
        obs,
        bool(to_np(terminated).reshape(-1)[0]),
        bool(to_np(truncated).reshape(-1)[0]),
        info,
    )


def reset_robolab(
    env: Any,
    hold_action: np.ndarray,
    settle_steps: int = 0,
) -> tuple[dict, bool, bool]:
    """Reset the scene, then hold (zero delta, gripper open) for ``settle_steps``.

    Reset is issued TWICE on purpose: RoboLab's own policy eval does the same
    (``robolab/eval/episode.py``). The first reset re-spawns the scene but the RTX sensors
    still carry the previous stage's frame, so the images from the first reset can be
    stale; the second one returns observations rendered from the new scene.

    Unlike ManiSkill there is no seed argument: a RoboLab env's layout randomisation is
    driven by ``env_cfg.seed``, fixed at construction time (see ``create_env(seed=...)``),
    so a per-episode layout change means constructing with a different seed.
    """
    # MUST come before reset(): RobolabEnv._reset_idx FREEZES any env that is reset after
    # it has been stepped (more than 2 steps in), because in RoboLab's own eval loop a
    # mid-run reset means "this episode terminated, hold its final state". A frozen env has
    # its actions zeroed in step(), so everything afterwards silently does nothing -- an
    # axis probe reads exactly 0.0000 m on every token, with no error anywhere. Clearing
    # the eval state first is what RoboLab's run_evaluation does between runs.
    reset_state = getattr(env, "reset_eval_state", None)
    if callable(reset_state):
        reset_state()
    env.reset()
    obs, _ = env.reset()
    # Physics only advances while the timeline plays -- see ensure_timeline_playing().
    ensure_timeline_playing(env)
    terminated = truncated = False
    for _ in range(max(0, int(settle_steps))):
        obs, terminated, truncated, _info = step_robolab(env, hold_action)
    return obs, terminated, truncated


def probe_move_axes(
    env: Any,
    controller: Any,
    repeats: int = 8,
    settle_steps: int = 0,
    reset_between: bool = True,
) -> dict[str, list[float]]:
    """Step each ``MV_*`` token in isolation and record the observed TCP delta.

    Cheap diagnostic (written to ``calibration.json``) documenting which world direction
    each token actually drives AND how many metres one decision travels -- the two numbers
    that have to match the training data. Mirrors
    ``core.sim.maniskill_task.probe_move_axes``.

    ``repeats`` + ``settle_steps`` must be the runner's ``sim_steps_per_decision`` +
    ``settle_steps_per_decision``: the relative-IK controller lags its target, so measuring
    without the settle steps reports the mid-flight displacement and always reads short of
    ``step_m``. ``reset_between`` re-resets the scene before each token so every probe
    starts from the same pose.
    """
    from core.action_units import MOVE_ATOMS

    result: dict[str, list[float]] = {}
    for token in MOVE_ATOMS:
        if reset_between:
            reset_robolab(env, controller.open_gripper(), settle_steps=2)
        p0 = rl_tcp(env)
        action = controller.action_for_atomic(token)
        for _ in range(int(repeats)):
            step_robolab(env, action)
        hold = controller.hold_action()
        for _ in range(max(0, int(settle_steps))):
            step_robolab(env, hold)
        delta = rl_tcp(env) - p0
        result[token] = [round(float(x), 4) for x in delta]
    controller.open_gripper()
    return result
