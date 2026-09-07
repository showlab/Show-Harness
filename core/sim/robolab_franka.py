"""Franka Panda (panda_hand) embodiment for RoboLab, matching the ManiSkill/real rigs.

RoboLab ships one embodiment: ``DroidCfg`` -- a Franka arm wearing a **Robotiq 2F-85**,
with its wrist camera bolted to the Robotiq base_link at DROID's calibrated D415 pose
(1280x720, focal 2.8, off to one side). That is faithful to DROID, but it is not what
MVTOKEN was trained on, and it differs from the ManiSkill rigs in two ways the policy sees
directly:

* **The gripper.** MVTOKEN's real Franka and every ManiSkill scene use the **Panda hand**
  (two parallel fingers). A Robotiq 2F-85 is a visibly different mechanism -- different
  silhouette, different finger motion -- so the wrist view is off-distribution before the
  policy reasons about anything.
* **The wrist camera pose.** RoboLab's ``wrist_cam`` sits beside the Robotiq base and looks
  past the fingers at an angle. ManiSkill mounts it centred between the fingers looking
  straight **down the grasp axis** (``core.sim.maniskill_scenes.WRIST_MOUNTS["centered"]``),
  which is the geometry the training wrist frames have.

This module supplies both. It does NOT patch RoboLab: it declares configs in RoboLab's own
vocabulary and hands them to ``auto_discover_and_create_cfgs``, exactly as
``robolab/registrations/droid/*.py`` does -- the "bring your own robot" path RoboLab
advertises.

IMPORT ORDER: this module imports ``isaaclab`` at module scope, like RoboLab's own
``robolab/robots/droid.py``, so it may only be imported AFTER
``core.sim.robolab_task.launch_isaac`` has started the Kit app. ``make_robolab_task``
imports it inside the function body for exactly that reason. (The configs must be real
``configclass`` fields -- RoboLab uses ``robot_cfg`` as a BASE CLASS of the generated scene
cfg, so an attribute attached after class creation is not a dataclass field and its camera
never gets spawned.)
"""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass, noise
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg, _to_torch

# -- wrist camera -------------------------------------------------------------
# Geometry COPIED from core.sim.maniskill_scenes.WRIST_MOUNTS["centered"] so the two
# simulators present the same view:
#
#   mount     : panda_hand -- the camera rides the gripper itself
#   position  : 3.5 cm along the hand's +X, 3.6 cm along +Z
#   direction : straight down the grasp axis (panda_hand's +Z, toward the fingertips)
#
# Why +X 3.5 cm rather than dead zero: ManiSkill measured that zeroing X buries the camera
# inside the hand mesh, so it renders the gripper's own interior. What "centered" fixes is
# the LATERAL (Y) offset -- the stock DROID/RealSense rig sits 2 cm to one side, throwing
# the finger pair 115 px (45% of the frame) off centre. Y = 0 lands the fingers symmetric
# in frame (measured 1 px off centre on ManiSkill), which is the training geometry.
#
# KNOWN GAP (measured 2026-07-29, not yet resolved): the view IS top-down and centred, but
# the FINGERTIPS ARE NOT IN FRAME, whereas ManiSkill's training wrist shows them entering
# from the top of the image. The two simulators define ``panda_hand`` differently:
#
#   IsaacLab panda_instanceable.usd : fingers at local [0, +-0.040, 0.0584]
#   -> at this camera pose the fingers sit 2.24 cm ahead, where a 90 deg FOV spans only
#      +-2.24 cm, so the +-4 cm finger pair falls just outside the frame.
#
# ManiSkill's origin is nearer the fingertips, so its (0.035, 0, 0.036) puts the camera
# ~6.7 cm from the fingers and they occupy the top ~25% of the image. Reproducing that here
# needs either a wider FOV (~156 deg at this stand-off -- heavy distortion) or a mount pose
# re-derived for this URDF; simply pulling the camera back was measured NOT to work (the
# view widens until the fingers are a few pixels and the frame fills with the arm and the
# floor -- candidates at z = -0.06 / -0.08 / -0.10 all failed this way).
#
# Consequence for the policy: the wrist still shows what is under the gripper, but not the
# gripper itself, so the "fingertips at the TOP" half of the MVTOKEN wrist contract is
# unmet and the policy cannot read finger-vs-object alignment from this view.
WRIST_CAM_POS = (0.035, 0.0, 0.036)
WRIST_CAM_RESOLUTION = 256
# 90 degree horizontal FOV, matching ManiSkill's ``fov=np.pi/2``. Isaac Lab describes a
# pinhole by aperture + focal length rather than an FOV; for a 90 deg horizontal FOV the
# focal length is half the horizontal aperture (tan(45 deg) == 1).
WRIST_CAM_APERTURE = 20.955
WRIST_CAM_FOCAL = WRIST_CAM_APERTURE / 2.0

# ``convention="ros"`` means the camera looks along its own +Z, so an identity rotation on
# panda_hand aims it exactly where the fingers close -- straight down in the top-down
# posture. (ManiSkill encodes the same thing as a quaternion because SAPIEN cameras look
# along +X.)
_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/panda_hand/wrist_cam",
    height=WRIST_CAM_RESOLUTION,
    width=WRIST_CAM_RESOLUTION,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=WRIST_CAM_FOCAL,
        focus_distance=0.4,
        horizontal_aperture=WRIST_CAM_APERTURE,
        vertical_aperture=WRIST_CAM_APERTURE,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=WRIST_CAM_POS, rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
    ),
)

# -- front (agentview) camera -------------------------------------------------
# The ManiSkill agentview, reproduced exactly. On ManiSkill the third-person view is
# BlockPAP's ``external_cam``: RLinf's calibration of the REAL RealSense D435 that filmed
# the Franka rig (``real_franka/real2sim_env/pick_and_place.py``, calibrated 2026-02-15).
# Reusing the same extrinsics AND intrinsics is what makes the two simulators' front views
# interchangeable to the policy. RoboLab's own ``over_shoulder_left_camera`` is DROID's
# placement -- off to the LEFT looking across the workspace, focal 2.1, 1280x720 -- i.e. a
# different viewpoint through a different lens.
#
# ``_R`` is camera-to-world in the OpenCV convention (+Z forward, +X right, +Y down), which
# is exactly Isaac Lab's ``convention="ros"`` -- so the extrinsics transfer with no axis
# surgery. (ManiSkill has to post-multiply a CV->SAPIEN matrix because SAPIEN cameras look
# along +X.) The intrinsics transfer verbatim through
# ``PinholeCameraCfg.from_intrinsic_matrix``.
FRONT_CAM_R = (
    (0.02816316, 0.21788680, -0.97556762),
    (0.99959024, -0.00114196, 0.02860160),
    (0.00511786, -0.97597338, -0.21782968),
)
# 1.10 m in front of the robot base, 0.26 m up. ("og" preset; RLinf also carries 0302/0303
# recalibrations, which ManiSkill exposes as its `cam_t` knob.)
FRONT_CAM_POS = (1.1002696, -0.00701879, 0.2589829)
# RealSense D435 intrinsics, row-major [fx, 0, cx, 0, fy, cy, 0, 0, 1].
FRONT_CAM_K = (607.875, 0.0, 348.961, 0.0, 607.719, 270.486, 0.0, 0.0, 1.0)
FRONT_CAM_W, FRONT_CAM_H = 640, 480


def _quat_wxyz_from_matrix(matrix) -> tuple[float, float, float, float]:
    """Rotation matrix -> (w, x, y, z), the quaternion order Isaac Lab cfgs expect."""
    import numpy as _np

    m = _np.asarray(matrix, dtype=float)
    trace = m.trace()
    if trace > 0.0:
        s = 0.5 / _np.sqrt(trace + 1.0)
        q = (0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
             (m[1, 0] - m[0, 1]) * s)
    else:
        i = int(_np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * _np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k]))
        out = [0.0, 0.0, 0.0, 0.0]
        out[0] = (m[k, j] - m[j, k]) / s
        out[i + 1] = 0.25 * s
        out[j + 1] = (m[j, i] + m[i, j]) / s
        out[k + 1] = (m[k, i] + m[i, k]) / s
        q = tuple(out)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


FRONT_CAM_QUAT = _quat_wxyz_from_matrix(FRONT_CAM_R)

_FRONT_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/front_cam",
    height=FRONT_CAM_H,
    width=FRONT_CAM_W,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=list(FRONT_CAM_K),
        width=FRONT_CAM_W,
        height=FRONT_CAM_H,
        clipping_range=(0.01, 10.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=FRONT_CAM_POS, rot=FRONT_CAM_QUAT, convention="ros"
    ),
)


@configclass
class FrankaFrontCameraCfg:
    """ManiSkill-identical front view. A SCENE camera (world-fixed), not robot-mounted."""

    front_cam = _FRONT_CAM


# -- robot asset --------------------------------------------------------------
# The real Franka this data is generated for does NOT wear Franka's stock black fingers:
# it wears metal L-brackets carrying a yellow printed fingertip
# (``assets/panda_short_finger.stl`` -- a mounting plate on a single screw plus a ~1 mm
# blade that sweeps inward, so the two blades close toward each other). Simulating the
# stock finger puts a differently shaped and differently coloured gripper in the middle of
# every training frame, which for a vision policy is a distribution shift, not cosmetics.
#
# ``assets/robolab_franka/panda_short_finger.usda`` is IsaacLab's own Panda with those two
# fingers swapped -- visuals AND collision meshes, so what the policy sees and what the
# physics does remain the same object. Built by
# ``scripts/trajectory/real2sim/robolab/make_short_finger_asset.py``; the arm links still
# resolve from IsaacLab's CDN, only the fingers are local.
#
# Set ``ROBOLAB_PANDA_USD`` to go back to the stock robot
# (``{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd``) or to try another
# tip. Changing it changes the grasp geometry -- ``real2sim/robolab/tasks.py``'s
# ``FLANGE_TO_FINGERTIP_M`` is MEASURED against whatever is spawned here, so re-run
# ``calibrate_fingertip.py`` after a swap rather than assuming the old number carries.
_REPO_ROOT = Path(__file__).resolve().parents[2]
PANDA_USD = os.environ.get(
    "ROBOLAB_PANDA_USD",
    str(_REPO_ROOT / "assets" / "robolab_franka" / "panda_short_finger.usda"),
)

# -- gripper ------------------------------------------------------------------
# Panda finger travel: 0 (closed) .. 0.04 m per finger; both fingers driven together.
PANDA_FINGER_OPEN_M = 0.04
PANDA_FINGER_CLOSED_M = 0.0
# Full opening in metres, for gripper-width accessors (two fingers).
PANDA_MAX_WIDTH_M = 2 * PANDA_FINGER_OPEN_M

# Arm start pose: Isaac Lab's own Franka home, which holds the hand above the table
# pointing straight down. This IS the grasp posture for the whole episode -- the
# relative-IK controller keeps whatever orientation it is reset into, because MVTOKEN
# never emits a rotation.
# SOLVED, not hand-picked: the previous pose started the EE at z = 0.590, high enough that
# a whole approach was spent shuffling sideways before descending. These angles were found
# by servoing the rel-IK straight down to z = 0.436 (the height reached at step 17 of the
# first preview) with the orientation hold active, then reading the joints back --
# measured z = 0.4367, tilt 0.002 deg from vertical.
#
# j2 + j4 + j6 = -91.2 deg keeps the hand pointing straight down, which is the whole point
# of this pose: the wrist camera looks along the grasp axis and the MVTOKEN contract is
# translation-only. Re-solve (scripts probe, not by hand) if the target height changes.
#
# Both sides reset to this pose -- the generator through RobolabBackend.reset and the
# deployment runner through reset_robolab -- so the starting geometry cannot drift apart.
FRANKA_HOME_QPOS: dict[str, float] = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.79613,   # -45.61 deg
    "panda_joint3": 0.0,
    "panda_joint4": -2.75597,   # -157.91 deg
    "panda_joint5": 0.0,
    "panda_joint6": 1.95980,    # +112.29 deg
    "panda_joint7": 0.785,      # +45 deg -- wrist roll; sets how the fingers line up
    "panda_finger_joint.*": PANDA_FINGER_OPEN_M,
}

# RoboLab's success predicates take a contact sensor named "gripper"; IsaacLab needs
# exactly one prim per env for filtered contact, so name a single finger.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/panda_leftfinger"}


@configclass
class FrankaPandaCfg:
    """Franka + Panda hand, with the centre-mounted wrist camera attached."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PANDA_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=dict(FRANKA_HOME_QPOS)
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0, velocity_limit=2.175,
                stiffness=400.0, damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0, velocity_limit=2.61,
                stiffness=400.0, damping=80.0,
            ),
            # Stiff, high-effort fingers: the grasp has to survive the discrete 2 cm jumps
            # the atomic executor makes, which a compliant grip does not.
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0, velocity_limit=0.2,
                stiffness=2e3, damping=1e2,
            ),
        },
    )

    wrist_cam = _WRIST_CAM


# RoboLab requires every robot cfg to name its EE recorder channels explicitly (no default
# body); the check lives in robolab/core/environments/config.py. Same value RoboLab's own
# FrankaCfg uses -- this rig is a Franka + Panda hand, only the fingertip visuals differ.
FrankaPandaCfg.ee_recorder_bodies = {"ee_pose": "panda_hand"}


@configclass
class FrankaWristCameraCfg:
    """Introspection wrapper so the wrist camera can be passed to the obs generator.

    The scene's ``wrist_cam`` already comes from :class:`FrankaPandaCfg`; this only exposes
    the name to ``generate_image_obs_from_cameras``. (Mirrors RoboLab's ``WristCameraCfg``,
    and like it must be excluded from the scene mixins -- a robot-mounted camera listed as
    a scene camera spawns before its parent prim exists.)
    """

    wrist_cam = _WRIST_CAM


@configclass
class FrankaRelIKActionCfg:
    """Relative EE-pose control -- RoboLab's ``DroidRelIKActionCfg`` shape, Panda hand.

    7 dims: ``(dx, dy, dz, drx, dry, drz, gripper)``. ``body_name="panda_hand"`` puts the
    controlled frame at the hand, which is also what ``rl_tcp`` reports, so the planner's
    "flange" and the IK's target are the same body.
    """

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )
    # RoboLab's 0=open / 1=close action, NOT Isaac Lab's stock BinaryJointPositionActionCfg.
    # The two have OPPOSITE conventions: upstream closes on ``action < 0``, RoboLab's
    # ZeroToOne subclass closes on ``action > 0.5``. The atomic controller and the real2sim
    # backend both emit 1.0 for "close" (RoboLab's convention), so using the upstream class
    # here silently leaves the gripper OPEN for every grasp -- the calibration sweep showed
    # the width pinned at 0.080 m (fully open) at every height, and every recorded demo
    # failed with no error.
    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": PANDA_FINGER_OPEN_M},
        close_command_expr={"panda_finger_joint.*": PANDA_FINGER_CLOSED_M},
    )


# -- observations -------------------------------------------------------------
def arm_joint_pos(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    idx = [i for i, n in enumerate(robot.data.joint_names) if n.startswith("panda_joint")]
    return _to_torch(robot.data.joint_pos)[:, idx]


def gripper_pos(env, asset_cfg=SceneEntityCfg("robot")):
    """0 = open, 1 = closed -- the normalisation RoboLab's droid.py also publishes."""
    robot = env.scene[asset_cfg.name]
    idx = [i for i, n in enumerate(robot.data.joint_names)
           if n.startswith("panda_finger_joint")]
    width = _to_torch(robot.data.joint_pos)[:, idx].sum(dim=-1, keepdim=True)
    return 1.0 - width / PANDA_MAX_WIDTH_M


def ee_pos(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    i = robot.data.body_names.index("panda_hand")
    return _to_torch(robot.data.body_pos_w)[:, i, :] - env.scene.env_origins[:, 0:3]


def ee_quat(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    i = robot.data.body_names.index("panda_hand")
    return _to_torch(robot.data.body_quat_w)[:, i, :]


@configclass
class FrankaProprioCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05),
                          clip=(0, 1))
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
