"""ManiSkill scene table -- one row per environment, one generic registrar.

This replaces the former one-module-per-scene layout (``maniskill_blockpap.py`` /
``maniskill_blockstack.py`` / ``maniskill_bpcam.py``). Those files were ~90-130 lines each
and *line-for-line isomorphic*: check ``RLINF_ROOT`` -> import an RLinf env module -> poke a couple of
module globals -> derive a wrist-cam agent from that module's base agent -> flip an
idempotency flag. Only five values actually differed between them, so adding a scene meant
copying a file and editing five lines, with the same task text and robot uid then repeated
again in ``core.sim.maniskill_task`` and in ``real2sim/maniskill/tasks.py``.

Here a scene is DATA (:data:`SCENES`) and the procedure is code (:func:`register_scene`).

**Adding a scene = adding one row.** No new module, no ``elif`` branch, no second copy of
the instruction or the robot uid -- ``maniskill_task`` and the real2sim generators both
read this table.

Two kinds of row:

* **RLinf real2sim rigs** (``module`` set) -- importing the module registers the gym id;
  the module's base agent has no wrist camera (the real rig's URDF variants lack a
  ``camera_link``), so a ``<Base>WristCam`` subclass is derived that keeps everything the
  env's obs plumbing depends on (notably ``ee_pose_at_robot_base``) and adds one
  ``hand_camera``. Requires the RLinf checkout (``RLINF_ROOT``).
* **Stock ManiSkill tasks** (``module`` None) -- ``import mani_skill`` already registered
  them; the row only supplies the instruction and the robot uid.

The wrist camera itself is deliberately NOT per scene: the mount pose, resolution and FOV
are the same measured rig on every scene (see :data:`WRIST_MOUNTS`), and they are a
training contract -- one copy, so a change cannot land on some scenes and not others.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

RLINF_ROOT = os.environ.get("RLINF_ROOT", str(Path.home() / "RLinf"))

# Wrist camera placement, relative to the gripper. Both keep the D415 rig's ORIENTATION
# (realsense_joint rpy = 0, -1.5707, 3.1415 in panda_v3.urdf -> _Q_D415), which is what aims
# the camera down past the fingers; only the mount offset differs.
#
#   "camera_link" -- the stock rig: realsense_joint xyz=[0.035,0,0.036] PLUS an extra
#                    camera_link_joint xyz=[0,0.02,0.0115]. That 0.02 of Y is pure lateral
#                    offset and throws the finger pair 115px (45% of the frame) off centre.
#   "centered"    -- the realsense_joint pose WITHOUT the extra hop: same +X side, same
#                    orientation, lateral Y zeroed. Finger pair lands 1px off centre.
#
# Measured with flip=both (the runner's transform), by projecting the finger links:
#   p=[0.035,-0.02,0.036] -> -115.3px | p=[0.035,0,0.036] -> -1.0px | p=[0.035,+0.02,0.036] -> +113.3px
# so Y is exactly the centring knob and 0 is the answer.
#
# Two traps, both hit while getting here -- re-measure if you touch this:
#  * p=[0,0,0.036] (zeroing X) buries the camera inside the hand mesh: it renders the
#    gripper's own interior. X must stay on the +0.035 side, outside the body.
#  * Moving to the -X side (p=[-0.03,...]) mirrors the vertical sense, so fingers enter from
#    the BOTTOM and no rigid camera pose can restore "fingertips top AND left==agentview"
#    (only an image mirror could, which is not something a real camera produces). Staying on
#    +X keeps flip=both -- a proper 180 deg rotation -- correct on both axes.
_Q_D415 = [0.0, 0.7071068, 0.0, 0.7071068]  # wxyz, = rpy(0, -1.5707, 3.1415)
WRIST_MOUNTS: dict[str, tuple[str, list[float], list[float]]] = {
    "camera_link": ("camera_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    "centered": ("panda_hand", [0.035, 0.0, 0.036], _Q_D415),
}

# Options every scene understands. A row's own ``defaults`` override these; the caller's
# kwargs override both.
COMMON_DEFAULTS: dict[str, Any] = {
    "wrist_resolution": 256,
    "wrist_mount": "centered",
}


@dataclass(frozen=True)
class SceneSpec:
    """One ManiSkill environment: how to register it and what it means.

    ``scene_globals`` maps a MODULE-LEVEL global in ``module`` to the option key that sets
    it. RLinf's real2sim envs read these inside ``_load_scene`` / ``_initialize_episode``,
    so they must be set before the env is constructed. A value may be either the option key
    alone, or ``(option_key, coerce)`` when the module expects a specific type.

    ``make_kwargs`` maps a ``gym.make`` keyword to the option key that fills it -- for
    per-scene knobs the env takes as a constructor argument rather than a global.

    ``urdf_path`` (relative to ManiSkill's ``PACKAGE_ASSET_DIR``) overrides the base
    agent's URDF on the derived wrist-cam agent. Only set it when the base agent's own URDF
    cannot carry the camera; overriding it throws away whatever that variant customised.
    """

    env_id: str
    instruction: str
    robot_uids: str = "panda_wristcam"
    module: Optional[str] = None
    agent_base: Optional[str] = None
    urdf_path: Optional[str] = None
    scene_globals: Mapping[str, Any] = field(default_factory=dict)
    make_kwargs: Mapping[str, str] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def needs_registration(self) -> bool:
        return self.module is not None


SCENES: dict[str, SceneSpec] = {
    # -- RLinf real2sim rigs: 1:1 replicas of the real Franka rig (table geometry, pedestal,
    # ground, lighting, and the calibrated front RealSense as `external_cam`), which is why
    # they are the default eval scenes -- the smallest domain gap available.
    "BlockPAP-v1": SceneSpec(
        env_id="BlockPAP-v1",
        # The block renders orange -- calling it "red" describes something the model does
        # not see.
        instruction="pick up the orange block and place it on the coaster",
        robot_uids="panda_high_friction_wristcam",
        module="real_franka.real2sim_env.pick_and_place",
        # panda_high_friction inherits Panda's panda_v2.urdf, which has no camera_link;
        # panda_v3 does, and the high-friction agent's own customisation is its pad
        # material + ee_pose_at_robot_base, both preserved by subclassing.
        agent_base="PandaHighFriction",
        urdf_path="robots/panda/panda_v3.urdf",
        scene_globals={
            "TABLE_TEX_ID": "table_tex",
            "TRAJ_ID": ("traj_id", str),
            "CAM_JITTER_RAD": "cam_jitter",
        },
        make_kwargs={"cam_t": "cam_t"},
        defaults={"table_tex": "white", "traj_id": "0", "cam_jitter": None, "cam_t": "og"},
    ),
    "BlockStack-v1": SceneSpec(
        env_id="BlockStack-v1",
        instruction="pick up the white block and stack it on the gray block",
        robot_uids="panda_extended_gripper_wristcam",
        module="real_franka.real2sim_env.block_stack",
        # panda_extended_gripper (2 cm longer fingers, high-friction pads) rides
        # panda_v2_extended.urdf. Do NOT override the URDF here: the centred mount hangs off
        # `panda_hand`, which every panda variant has, whereas swapping in panda_v3 would
        # throw away the finger extension this task depends on.
        agent_base="PandaExtendedGripper",
        scene_globals={"TRAJ_ID": ("traj_id", str)},
        defaults={"traj_id": "random"},
    ),
    # -- Stock ManiSkill tabletop tasks. Already registered by `import mani_skill`; the row
    # exists so instruction + robot uid live in ONE place for every env. `panda_wristcam` is
    # required for a wrist view (plain `panda` has none, so MVTOKEN cannot run on it).
    "PickCube-v1": SceneSpec("PickCube-v1", "pick up the red cube"),
    "StackCube-v1": SceneSpec("StackCube-v1", "stack the red cube on top of the green cube"),
    "PushCube-v1": SceneSpec("PushCube-v1", "push the cube to the goal marker"),
    "PullCube-v1": SceneSpec("PullCube-v1", "pull the cube to the goal marker"),
    "PokeCube-v1": SceneSpec("PokeCube-v1", "poke the cube to the goal marker"),
    "LiftPegUpright-v1": SceneSpec("LiftPegUpright-v1", "lift the peg upright"),
}

# Wrist-cam agents already registered in this process, by uid. ManiSkill's registry raises
# on a duplicate uid, and one process may build several scenes (batch eval, data gen).
_REGISTERED_AGENTS: set[str] = set()


def scene_spec(env_id: str) -> Optional[SceneSpec]:
    """The row for ``env_id``, or None for an env this table does not know."""
    return SCENES.get(env_id)


def instruction_for(env_id: str) -> str:
    """Default task text for ``env_id`` (falls back to the id itself)."""
    spec = SCENES.get(env_id)
    return spec.instruction if spec is not None else env_id


def robot_uids_for(env_id: str, default: str = "panda_wristcam") -> str:
    spec = SCENES.get(env_id)
    return spec.robot_uids if spec is not None else default


def scene_options(env_id: str, **overrides: Any) -> dict[str, Any]:
    """Merge COMMON_DEFAULTS <- the row's defaults <- caller overrides."""
    spec = SCENES.get(env_id)
    merged = dict(COMMON_DEFAULTS)
    if spec is not None:
        merged.update(spec.defaults)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def register_scene(env_id: str, **overrides: Any) -> dict[str, Any]:
    """Register whatever ``env_id`` needs and return extra ``gym.make`` kwargs.

    Idempotent. For a stock ManiSkill env this is a no-op returning ``{}``; for an RLinf
    rig it imports the env module (which registers the gym id), applies the scene globals,
    derives the wrist-cam agent, and returns the row's ``make_kwargs``.
    """
    spec = SCENES.get(env_id)
    if spec is None or not spec.needs_registration:
        return {}

    opts = scene_options(env_id, **overrides)
    module = _import_scene_module(spec)

    # Scene globals are read inside _load_scene / _initialize_episode, so set them BEFORE
    # the env is constructed.
    for global_name, source in spec.scene_globals.items():
        opt_key, coerce = _split_source(source)
        if opt_key in opts:
            value = opts[opt_key]
            setattr(module, global_name, coerce(value) if value is not None else None)

    _register_wristcam_agent(spec, module, opts)

    return {gym_key: opts[opt_key] for gym_key, opt_key in spec.make_kwargs.items()
            if opt_key in opts}


# -- internals ---------------------------------------------------------------
def _identity(value: Any) -> Any:
    return value


def _split_source(source: Any) -> tuple[str, Callable[[Any], Any]]:
    if isinstance(source, tuple):
        return source[0], (source[1] if len(source) > 1 else _identity)
    return str(source), _identity


def _ensure_rlinf_path(env_id: str) -> None:
    if not os.path.isdir(RLINF_ROOT):
        raise SystemExit(
            f"RLinf checkout not found at {RLINF_ROOT!r} (needed for {env_id} and its "
            "scene assets). Set RLINF_ROOT to override."
        )
    if RLINF_ROOT not in sys.path:
        sys.path.insert(0, RLINF_ROOT)


def _import_scene_module(spec: SceneSpec) -> Any:
    _ensure_rlinf_path(spec.env_id)
    try:
        return importlib.import_module(spec.module)
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import {spec.module!r} for scene {spec.env_id} from RLINF_ROOT="
            f"{RLINF_ROOT!r}: {exc}"
        ) from exc


def _register_wristcam_agent(spec: SceneSpec, module: Any, opts: dict[str, Any]) -> None:
    """Derive ``<agent_base>`` + one hand camera, registered under ``spec.robot_uids``.

    Subclassing (rather than swapping in ManiSkill's own ``PandaWristCam``) is what keeps
    the rig's customisations: the friction pads, the extended fingers, and the
    ``ee_pose_at_robot_base`` property the RLinf envs' observation plumbing calls.
    """
    if spec.robot_uids in _REGISTERED_AGENTS:
        return

    import sapien
    from mani_skill import PACKAGE_ASSET_DIR
    from mani_skill.agents.registration import register_agent
    from mani_skill.sensors.camera import CameraConfig

    mount = str(opts["wrist_mount"])
    if mount not in WRIST_MOUNTS:
        raise ValueError(f"wrist_mount {mount!r} not in {list(WRIST_MOUNTS)}")
    mount_link, mount_p, mount_q = WRIST_MOUNTS[mount]
    resolution = int(opts["wrist_resolution"])

    base_agent = getattr(module, spec.agent_base, None)
    if base_agent is None:
        raise SystemExit(
            f"Scene {spec.env_id}: {spec.module}.{spec.agent_base} not found -- the RLinf "
            "checkout may be a different revision than this table expects."
        )

    namespace: dict[str, Any] = {
        "uid": spec.robot_uids,
        "_sensor_configs": property(
            lambda self: [
                CameraConfig(
                    uid="hand_camera",
                    pose=sapien.Pose(p=mount_p, q=mount_q),
                    width=resolution,
                    height=resolution,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                    mount=self.robot.links_map[mount_link],
                )
            ]
        ),
        "__doc__": f"{spec.agent_base} + a gripper-mounted wrist camera ({mount} mount).",
    }
    if spec.urdf_path:
        namespace["urdf_path"] = f"{PACKAGE_ASSET_DIR}/{spec.urdf_path}"

    agent_cls = type(f"{spec.agent_base}WristCam", (base_agent,), namespace)
    register_agent()(agent_cls)
    _REGISTERED_AGENTS.add(spec.robot_uids)
