"""Synthetic tabletop pick-and-place world for the web teleop (no hardware needed).

Provides the same object shapes the real Piper wiring uses -- a robot with the
MockPiperRobot interface and a session with the PiperSession.get_observation()
contract -- but grounded in a tiny simulated scene:

  * an orange target cube and a gray distractor cube on a table,
  * a blue target plate,
  * a gripper that actually has to be NEAR a cube (xy) and LOW enough (z) for a
    GRASP to catch it; a grasp settling on nothing reads width 0.0 and therefore
    trips the controller's normal empty-grasp auto-reopen,
  * a held cube follows the gripper and is dropped where RELEASE happens,
  * task success = the target cube resting inside the plate.

Both camera views are rendered with cv2 so a browser (or a GUI agent) can drive a
genuine pick-and-place loop end-to-end. Geometry note: both views map base +X
(MV_FWD) to image-up and base +Y (MV_LEFT) to image-left, matching the ego prompt
convention used on the Piper rig (is_ego: image top -> MV_FWD).

All positions are randomized per reset() ON THE 2 cm STEP LATTICE anchored at the
home EEF xy, so exact alignment over a cube is always reachable with step_m=0.02.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import cv2
import numpy as np

import core.franka.camera_utils as camera_utils
from core.piper.piper_interface import MockPiperRobot

# World geometry (meters, Piper base frame: +X forward/far, +Y left, +Z up).
Z_TABLE = 0.145
HOME_XYZ = (0.30, 0.0, 0.25)
WORK_X = (0.24, 0.56)
WORK_Y = (-0.20, 0.20)
Z_MAX = 0.40

CUBE_HALF_M = 0.016          # cube half-extent -> held width 0.032 m (reads CLOSED, non-empty)
GRASP_XY_RADIUS_M = 0.035    # max gripper-to-cube xy distance for a catch
GRASP_MAX_HEIGHT_M = 0.06    # gripper must be within this height above the table
PLATE_RADIUS_M = 0.06
PLACE_RADIUS_M = 0.045       # cube center inside this radius counts as "on the plate"
STEP_LATTICE_M = 0.02        # matches primitives_piper.yaml step_m

VIEW_PX = 480                # native render size (streamed); observations are resized


def _snap(v: float, origin: float) -> float:
    """Snap a coordinate onto the step lattice anchored at the home pose."""
    return origin + round((v - origin) / STEP_LATTICE_M) * STEP_LATTICE_M


class SimScene:
    """Object/plate state + grasp, release and success rules."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.z_table = Z_TABLE
        self.plate_xy = np.zeros(2)
        self.cubes: dict[str, np.ndarray] = {}   # name -> xy (resting cubes)
        self.held: Optional[str] = None
        self.held_xy = np.zeros(2)               # follows the gripper while held
        self.reset()

    # -- layout --------------------------------------------------------------
    def _rand_cell(self, x_rng: tuple, y_rng: tuple) -> np.ndarray:
        x = _snap(self._rng.uniform(*x_rng), HOME_XYZ[0])
        y = _snap(self._rng.uniform(*y_rng), HOME_XYZ[1])
        return np.array([x, y])

    def reset(self) -> None:
        """New episode layout: target cube on the left field, plate on the right."""
        self.held = None
        for _ in range(100):
            target = self._rand_cell((0.38, 0.50), (0.02, 0.14))
            plate = self._rand_cell((0.36, 0.48), (-0.16, -0.04))
            distractor = self._rand_cell((0.32, 0.52), (-0.02, 0.18))
            far_enough = (
                np.linalg.norm(target - plate) > PLATE_RADIUS_M + 0.06
                and np.linalg.norm(distractor - target) > 0.07
                and np.linalg.norm(distractor - plate) > PLATE_RADIUS_M + 0.05
            )
            if far_enough:
                break
        self.plate_xy = plate
        self.cubes = {"orange_cube": target, "gray_cube": distractor}

    def to_meta(self) -> dict[str, Any]:
        return {
            "sim_scene": {
                "z_table_m": self.z_table,
                "plate_xy_m": self.plate_xy.round(4).tolist(),
                "cubes_xy_m": {k: v.round(4).tolist() for k, v in self.cubes.items()},
                "target": "orange_cube",
                "place_radius_m": PLACE_RADIUS_M,
            }
        }

    # -- physics-ish ----------------------------------------------------------
    def on_ee_moved(self, ee_pose: np.ndarray) -> None:
        if self.held is not None:
            self.held_xy = np.asarray(ee_pose[:2], dtype=float).copy()

    def try_grasp(self, ee_pose: np.ndarray) -> float:
        """A close lands on the nearest catchable cube; returns the settled width (m)."""
        pos = np.asarray(ee_pose, dtype=float)
        if self.held is not None:
            return 2 * CUBE_HALF_M
        if (pos[2] - self.z_table) > GRASP_MAX_HEIGHT_M:
            return 0.0
        best, best_d = None, GRASP_XY_RADIUS_M
        for name, xy in self.cubes.items():
            d = float(np.linalg.norm(pos[:2] - xy))
            if d <= best_d:
                best, best_d = name, d
        if best is None:
            return 0.0
        self.held = best
        self.held_xy = pos[:2].copy()
        del self.cubes[best]
        return 2 * CUBE_HALF_M

    def release(self, ee_pose: np.ndarray) -> Optional[str]:
        """Drop any held cube at the gripper's xy; returns the dropped cube's name."""
        if self.held is None:
            return None
        name = self.held
        self.cubes[name] = np.asarray(ee_pose[:2], dtype=float).copy()
        self.held = None
        return name

    def task_success(self) -> bool:
        """The orange target cube rests inside the plate (and is not being held)."""
        xy = self.cubes.get("orange_cube")
        if xy is None:
            return False
        return float(np.linalg.norm(xy - self.plate_xy)) <= PLACE_RADIUS_M

    def task_text(self) -> str:
        return "Pick up the orange cube and place it on the blue plate"


class SimPiperRobot(MockPiperRobot):
    """MockPiperRobot whose gripper interacts with a SimScene."""

    def __init__(self, scene: SimScene, arm: str = "left", open_width_m: float = 0.07) -> None:
        super().__init__(arm=arm, open_width_m=open_width_m)
        self.scene = scene
        self._pose = np.array([*HOME_XYZ, 0.0, 0.0, 0.0, 1.0])
        self._home_pose = self._pose.copy()

    def update_desired_ee_pose(self, pose: np.ndarray) -> None:
        p = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()
        p[0] = float(np.clip(p[0], *WORK_X))
        p[1] = float(np.clip(p[1], *WORK_Y))
        p[2] = float(np.clip(p[2], self.scene.z_table + 0.005, Z_MAX))
        super().update_desired_ee_pose(p)
        self.scene.on_ee_moved(self._pose)

    def control_gripper(self, gripper_action: bool) -> None:
        if gripper_action:  # close
            self._gripper_m = self.scene.try_grasp(self._pose)
        else:  # open
            self.scene.release(self._pose)
            self._gripper_m = self.open_width_m

    def set_gripper_position(self, pos: float) -> None:
        super().set_gripper_position(pos)
        if pos >= 0.5 * self.open_width_m:
            self.scene.release(self._pose)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float) -> None:
        """Homing: drop anything held in place, then jump back to the home EEF pose."""
        super().move_to_joint_positions(positions, time_to_go)
        self.scene.release(self._pose)
        self._gripper_m = self.open_width_m
        self._pose = self._home_pose.copy()


# ---------------------------------------------------------------------------
# Rendering (cv2, RGB uint8)
# ---------------------------------------------------------------------------
_C_BG = (24, 24, 28)
_C_TABLE = (206, 206, 196)
_C_GRID = (188, 188, 178)
_C_PLATE = (168, 205, 235)
_C_PLATE_RIM = (120, 165, 210)
_C_CUBES = {"orange_cube": (235, 140, 40), "gray_cube": (125, 125, 130)}
_C_CUBE_TOP = {"orange_cube": (250, 170, 80), "gray_cube": (150, 150, 155)}
_C_ARM = (235, 235, 240)
_C_JAW = (35, 105, 200)
_C_SHADOW = (172, 172, 162)


def _yaw_of(pose: np.ndarray) -> float:
    qx, qy, qz, qw = (float(v) for v in np.asarray(pose, dtype=float)[3:7])
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def render_agentview(scene: SimScene, pose: np.ndarray, gripper_m: float) -> np.ndarray:
    """Oblique third-person view: +X (FWD) toward image-top, +Y (LEFT) to image-left."""
    img = np.full((VIEW_PX, VIEW_PX, 3), _C_BG, dtype=np.uint8)
    S, KX, KZ = 880.0, 640.0, 500.0

    def fdepth(x: float) -> float:  # fake perspective: nearer -> wider
        return 1.14 - 0.52 * (x - WORK_X[0]) / (WORK_X[1] - WORK_X[0])

    def proj(x: float, y: float, z: float) -> tuple:
        u = VIEW_PX / 2 - y * S * fdepth(x)
        v = 452.0 - (x - WORK_X[0]) * KX - (z - scene.z_table) * KZ
        return int(round(u)), int(round(v))

    # Table top.
    corners = [(WORK_X[0], WORK_Y[1]), (WORK_X[0], WORK_Y[0]), (WORK_X[1], WORK_Y[0]), (WORK_X[1], WORK_Y[1])]
    cv2.fillPoly(img, [np.array([proj(x, y, scene.z_table) for x, y in corners])], _C_TABLE)
    for gx in np.arange(WORK_X[0], WORK_X[1] + 1e-9, 0.08):
        cv2.line(img, proj(gx, WORK_Y[0], scene.z_table), proj(gx, WORK_Y[1], scene.z_table), _C_GRID, 1)
    for gy in np.arange(WORK_Y[0], WORK_Y[1] + 1e-9, 0.08):
        cv2.line(img, proj(WORK_X[0], gy, scene.z_table), proj(WORK_X[1], gy, scene.z_table), _C_GRID, 1)

    # Plate (flattened ellipse).
    pu, pv = proj(scene.plate_xy[0], scene.plate_xy[1], scene.z_table)
    f = fdepth(scene.plate_xy[0])
    axes = (int(PLATE_RADIUS_M * S * f), int(PLATE_RADIUS_M * KZ * 0.42))
    cv2.ellipse(img, (pu, pv), axes, 0, 0, 360, _C_PLATE, -1)
    cv2.ellipse(img, (pu, pv), axes, 0, 0, 360, _C_PLATE_RIM, 2)

    # Cubes: front face + top face, far-to-near so nearer cubes overdraw.
    h = CUBE_HALF_M
    for name, xy in sorted(scene.cubes.items(), key=lambda kv: -kv[1][0]):
        x, y = float(xy[0]), float(xy[1])
        bl = proj(x, y + h, scene.z_table)
        br = proj(x, y - h, scene.z_table)
        tl = proj(x, y + h, scene.z_table + 2 * h)
        tr = proj(x, y - h, scene.z_table + 2 * h)
        cv2.fillPoly(img, [np.array([bl, br, tr, tl])], _C_CUBES[name])
        top = [proj(x - h, y + h, scene.z_table + 2 * h), proj(x - h, y - h, scene.z_table + 2 * h),
               proj(x + h, y - h, scene.z_table + 2 * h), proj(x + h, y + h, scene.z_table + 2 * h)]
        cv2.fillPoly(img, [np.array(top)], _C_CUBE_TOP[name])

    # Gripper: shadow on the table, arm link, two jaws (+ the held cube).
    ex, ey, ez = (float(v) for v in np.asarray(pose, dtype=float)[:3])
    cv2.ellipse(img, proj(ex, ey, scene.z_table), (14, 6), 0, 0, 360, _C_SHADOW, -1)
    cv2.line(img, proj(0.55, ey * 0.4, 0.55), proj(ex, ey, ez + 0.03), (70, 70, 78), 6)
    gu, gv = proj(ex, ey, ez)
    yaw = _yaw_of(pose)
    half_px = max(4, int(0.5 * gripper_m * S * fdepth(ex)))
    dx, dy = np.cos(yaw), np.sin(yaw)
    for sgn in (-1, 1):
        ju, jv = int(gu + sgn * half_px * dx), int(gv + sgn * half_px * dy * 0.5)
        cv2.rectangle(img, (ju - 3, jv - 12), (ju + 3, jv + 12), _C_JAW, -1)
    cv2.line(img, (gu - half_px, gv - 16), (gu + half_px, gv - 16), (60, 60, 66), 4)
    if scene.held is not None:
        cube = scene.held
        cv2.rectangle(img, (gu - 9, gv - 9), (gu + 9, gv + 9), _C_CUBES.get(cube, (200, 60, 60)), -1)
    return img


def render_wristview(scene: SimScene, pose: np.ndarray, gripper_m: float) -> np.ndarray:
    """Ego top-down view centered on the gripper; zooms in as the gripper lowers."""
    img = np.full((VIEW_PX, VIEW_PX, 3), _C_BG, dtype=np.uint8)
    ex, ey, ez = (float(v) for v in np.asarray(pose, dtype=float)[:3])
    span = float(np.clip(2.6 * (ez - scene.z_table) + 0.12, 0.18, 0.55))
    ppm = VIEW_PX / span

    def proj(x: float, y: float) -> tuple:
        return int(round(VIEW_PX / 2 - (y - ey) * ppm)), int(round(VIEW_PX / 2 - (x - ex) * ppm))

    # Table region + grid.
    cv2.rectangle(img, proj(WORK_X[1], WORK_Y[1]), proj(WORK_X[0], WORK_Y[0]), _C_TABLE, -1)
    for gx in np.arange(WORK_X[0], WORK_X[1] + 1e-9, 0.08):
        cv2.line(img, proj(gx, WORK_Y[0]), proj(gx, WORK_Y[1]), _C_GRID, 1)
    for gy in np.arange(WORK_Y[0], WORK_Y[1] + 1e-9, 0.08):
        cv2.line(img, proj(WORK_X[0], gy), proj(WORK_X[1], gy), _C_GRID, 1)

    pu, pv = proj(scene.plate_xy[0], scene.plate_xy[1])
    cv2.circle(img, (pu, pv), int(PLATE_RADIUS_M * ppm), _C_PLATE, -1)
    cv2.circle(img, (pu, pv), int(PLATE_RADIUS_M * ppm), _C_PLATE_RIM, 2)

    hpx = max(3, int(CUBE_HALF_M * ppm))
    for name, xy in scene.cubes.items():
        cu, cvv = proj(float(xy[0]), float(xy[1]))
        cv2.rectangle(img, (cu - hpx, cvv - hpx), (cu + hpx, cvv + hpx), _C_CUBES[name], -1)
        cv2.rectangle(img, (cu - hpx, cvv - hpx), (cu + hpx, cvv + hpx), _C_CUBE_TOP[name], 2)

    # Gripper jaws overlay (fixed at image center; separation tracks the width).
    yaw = _yaw_of(pose)
    c, s = np.cos(yaw), np.sin(yaw)
    half = max(10.0, 0.5 * gripper_m * ppm)
    jaw_l, jaw_w = 46, 12
    center = np.array([VIEW_PX / 2, VIEW_PX / 2])
    for sgn in (-1, 1):
        jc = center + np.array([sgn * half * c, sgn * half * s])
        along = np.array([s, -c])  # jaw long axis, perpendicular to the separation
        p1 = jc + along * jaw_l / 2 + np.array([c, s]) * jaw_w / 2 * sgn
        p2 = jc + along * jaw_l / 2 - np.array([c, s]) * jaw_w / 2 * sgn
        p3 = jc - along * jaw_l / 2 - np.array([c, s]) * jaw_w / 2 * sgn
        p4 = jc - along * jaw_l / 2 + np.array([c, s]) * jaw_w / 2 * sgn
        cv2.fillPoly(img, [np.array([p1, p2, p3, p4], dtype=int)], _C_JAW)
    if scene.held is not None:
        cu, cvv = int(center[0]), int(center[1])
        cv2.rectangle(img, (cu - hpx, cvv - hpx), (cu + hpx, cvv + hpx),
                      _C_CUBES.get(scene.held, (200, 60, 60)), -1)
    else:
        cv2.drawMarker(img, (int(center[0]), int(center[1])), (80, 80, 90),
                       cv2.MARKER_CROSS, 16, 2)
    return img


class SimSession:
    """PiperSession-shaped session over the synthetic world (get_observation contract)."""

    def __init__(self, scene: SimScene, robot: SimPiperRobot, observation_resolution: int = 256) -> None:
        self.scene = scene
        self.robot = robot
        self.observation_resolution = int(observation_resolution)
        self._display: dict[str, np.ndarray] = {}

    def connect(self) -> "SimSession":
        return self

    def close(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        pose = self.robot.get_ee_pose()
        width = float(self.robot.get_gripper_position()[0])
        agent = render_agentview(self.scene, pose, width)
        wrist = render_wristview(self.scene, pose, width)
        self._display = {"agentview": agent, "wrist": wrist}
        res = self.observation_resolution
        return {
            "agentview": np.ascontiguousarray(camera_utils.resize_with_pad(agent, res, res)),
            "wrist": np.ascontiguousarray(camera_utils.resize_with_pad(wrist, res, res)),
            "ee_pose": pose,
            "gripper_width": width,
        }

    def get_display_frames(self) -> dict[str, np.ndarray]:
        """Full-resolution renders of the last observation, for the browser streams."""
        return self._display
