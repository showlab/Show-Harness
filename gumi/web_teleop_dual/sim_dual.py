"""Synthetic DUAL-ARM tabletop world for the web teleop (no hardware needed).

Dual counterpart of ``web_teleop/sim.py`` (which stays single-arm and untouched):
the same fake physics and rendering conventions, extended to two Piper-shaped arms
sharing one table. Provides the same object shapes the real dual wiring uses -- two
robots with the MockPiperRobot interface and a session with the
``DualPiperSession.get_observation()`` contract (``agentview`` / ``wrist_left`` /
``wrist_right`` + per-side ``{"ee_pose", "gripper_width"}``) -- so the dual web
backend runs identically on this world and on the real rig.

World: an ORANGE cube on the left half, a GREEN cube on the right half, a gray
distractor, and one shared BLUE plate near the table's center line. Task success =
BOTH target cubes resting on the plate. The shared destination is deliberate: it
forces the coordination the dual GUI exists to exercise (one arm places while the
other waits clear of it), and a mid-table set-down/pick-up handover works with the
same physics (a released cube simply rests where it was dropped).

Geometry matches the single-arm sim: base +X (MV_FWD) -> image-up and +Y (MV_LEFT)
-> image-left in ALL views (the is_ego convention), the front view does NOT mirror
arm left/right (the left arm works on the image's left, same as the real rig), and
every random position lands ON the 2 cm step lattice anchored at x=0.30, y=0.0 so
both arms can align exactly with step_m=0.02. There is NO collision model and no
workspace partition -- exactly like the real rig, where keeping the arms apart is
the operator's (or the GUI agent's) reasoning job, not a gate. A scene lock makes
the fake physics safe under the backend's one-thread-per-arm execution.
"""
from __future__ import annotations

import random
import threading
from typing import Dict, Optional

import cv2
import numpy as np

import core.franka.camera_utils as camera_utils
from core.piper.config import SIDES
from core.piper.piper_interface import MockPiperRobot
from gumi.web_teleop.sim import (
    _C_ARM,
    _C_BG,
    _C_GRID,
    _C_JAW,
    _C_PLATE,
    _C_PLATE_RIM,
    _C_SHADOW,
    _C_TABLE,
    _yaw_of,
    CUBE_HALF_M,
    GRASP_MAX_HEIGHT_M,
    GRASP_XY_RADIUS_M,
    PLACE_RADIUS_M,
    PLATE_RADIUS_M,
    VIEW_PX,
    WORK_X,
    WORK_Y,
    Z_MAX,
    Z_TABLE,
    _snap,
)

# Shared lattice anchor: BOTH arms' homes and every object sit on the same 2 cm grid,
# so either arm can center exactly over any cube. (The single-arm sim anchors at its
# one home; here the anchor is the table center line and the homes are +/-0.14 = 7
# steps off it.)
LATTICE_ANCHOR_XY = (0.30, 0.0)
HOME_XYZ = {
    "left": (0.30, 0.14, 0.25),
    "right": (0.30, -0.14, 0.25),
}

# Per-arm colors are NOT drawn (no labels/markers -- the real rig has none either):
# arms are told apart by which side their link enters from, as on hardware.
_C_CUBES = {
    "orange_cube": (235, 140, 40),
    "green_cube": (70, 175, 90),
    "gray_cube": (125, 125, 130),
}
_C_CUBE_TOP = {
    "orange_cube": (250, 170, 80),
    "green_cube": (100, 205, 120),
    "gray_cube": (150, 150, 155),
}
TARGET_CUBES = ("orange_cube", "green_cube")


class DualSimScene:
    """Two-gripper object/plate state + grasp, release and success rules.

    All mutating calls take the acting ``side``; a lock serializes them because the
    dual backend executes the two arms' tokens on concurrent threads.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.z_table = Z_TABLE
        self.plate_xy = np.zeros(2)
        self.cubes: Dict[str, np.ndarray] = {}      # name -> xy (resting cubes)
        self.held: Dict[str, Optional[str]] = {s: None for s in SIDES}
        self.held_xy: Dict[str, np.ndarray] = {s: np.zeros(2) for s in SIDES}
        self.reset()

    # -- layout --------------------------------------------------------------
    def _rand_cell(self, x_rng: tuple, y_rng: tuple) -> np.ndarray:
        x = _snap(self._rng.uniform(*x_rng), LATTICE_ANCHOR_XY[0])
        y = _snap(self._rng.uniform(*y_rng), LATTICE_ANCHOR_XY[1])
        return np.array([x, y])

    def reset(self) -> None:
        """New episode: orange cube on the LEFT half, green on the RIGHT half, the
        shared plate near the center line, a gray distractor anywhere."""
        with self._lock:
            self.held = {s: None for s in SIDES}
            for _ in range(100):
                orange = self._rand_cell((0.38, 0.50), (0.06, 0.16))
                green = self._rand_cell((0.38, 0.50), (-0.16, -0.06))
                plate = self._rand_cell((0.34, 0.46), (-0.04, 0.04))
                distractor = self._rand_cell((0.32, 0.52), (-0.18, 0.18))
                spread = (
                    np.linalg.norm(orange - plate) > PLATE_RADIUS_M + 0.06
                    and np.linalg.norm(green - plate) > PLATE_RADIUS_M + 0.06
                    and np.linalg.norm(distractor - orange) > 0.07
                    and np.linalg.norm(distractor - green) > 0.07
                    and np.linalg.norm(distractor - plate) > PLATE_RADIUS_M + 0.05
                )
                if spread:
                    break
            self.plate_xy = plate
            self.cubes = {"orange_cube": orange, "green_cube": green, "gray_cube": distractor}

    def snapshot(self) -> dict:
        """Locked copy of the mutable scene state, safe to read from ANY thread.

        /api/state is answered on HTTP handler threads while a controller thread may
        be inside try_grasp/release resizing ``cubes`` -- iterating the live dict
        there would intermittently raise "dictionary changed size during iteration".
        """
        with self._lock:
            return {
                "plate_xy_m": [round(float(v), 4) for v in self.plate_xy],
                "cubes_xy_m": {k: [round(float(v), 4) for v in xy] for k, xy in self.cubes.items()},
                "held": dict(self.held),
            }

    def to_meta(self) -> dict:
        snap = self.snapshot()
        return {
            "sim_scene": {
                "z_table_m": self.z_table,
                "plate_xy_m": snap["plate_xy_m"],
                "cubes_xy_m": snap["cubes_xy_m"],
                "targets": list(TARGET_CUBES),
                "place_radius_m": PLACE_RADIUS_M,
            }
        }

    # -- physics-ish ----------------------------------------------------------
    def on_ee_moved(self, side: str, ee_pose: np.ndarray) -> None:
        with self._lock:
            if self.held[side] is not None:
                self.held_xy[side] = np.asarray(ee_pose[:2], dtype=float).copy()

    def try_grasp(self, side: str, ee_pose: np.ndarray) -> float:
        """A close lands on the nearest catchable RESTING cube; returns the settled
        width (m). A cube held by the other arm is not in ``cubes``, so it cannot be
        stolen mid-air -- a handover goes through a set-down (matches the fake
        physics, and is the safe choreography to teach anyway)."""
        pos = np.asarray(ee_pose, dtype=float)
        with self._lock:
            if self.held[side] is not None:
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
            self.held[side] = best
            self.held_xy[side] = pos[:2].copy()
            del self.cubes[best]
            return 2 * CUBE_HALF_M

    def release(self, side: str, ee_pose: np.ndarray) -> Optional[str]:
        """Drop this arm's held cube at its gripper xy; returns the cube's name."""
        with self._lock:
            name = self.held[side]
            if name is None:
                return None
            self.cubes[name] = np.asarray(ee_pose[:2], dtype=float).copy()
            self.held[side] = None
            return name

    def task_success(self) -> bool:
        """BOTH target cubes rest inside the plate (neither is being held)."""
        with self._lock:
            for name in TARGET_CUBES:
                xy = self.cubes.get(name)
                if xy is None or float(np.linalg.norm(xy - self.plate_xy)) > PLACE_RADIUS_M:
                    return False
            return True

    def task_text(self) -> str:
        return "Place the orange cube and the green cube on the blue plate"


class SimDualPiperRobot(MockPiperRobot):
    """MockPiperRobot for ONE side whose gripper interacts with a shared DualSimScene."""

    def __init__(self, scene: DualSimScene, side: str, open_width_m: float = 0.07) -> None:
        super().__init__(arm=side, open_width_m=open_width_m)
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.scene = scene
        self.side = side
        self._pose = np.array([*HOME_XYZ[side], 0.0, 0.0, 0.0, 1.0])
        self._home_pose = self._pose.copy()

    def update_desired_ee_pose(self, pose: np.ndarray) -> None:
        p = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()
        p[0] = float(np.clip(p[0], *WORK_X))
        p[1] = float(np.clip(p[1], *WORK_Y))
        p[2] = float(np.clip(p[2], self.scene.z_table + 0.005, Z_MAX))
        super().update_desired_ee_pose(p)
        self.scene.on_ee_moved(self.side, self._pose)

    def control_gripper(self, gripper_action: bool) -> None:
        if gripper_action:  # close
            self._gripper_m = self.scene.try_grasp(self.side, self._pose)
        else:  # open
            self.scene.release(self.side, self._pose)
            self._gripper_m = self.open_width_m

    def set_gripper_position(self, pos: float) -> None:
        super().set_gripper_position(pos)
        if pos >= 0.5 * self.open_width_m:
            self.scene.release(self.side, self._pose)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float) -> None:
        """Homing: drop anything held in place, then jump back to this arm's home."""
        super().move_to_joint_positions(positions, time_to_go)
        self.scene.release(self.side, self._pose)
        self._gripper_m = self.open_width_m
        self._pose = self._home_pose.copy()


# ---------------------------------------------------------------------------
# Rendering (cv2, RGB uint8). The projections mirror web_teleop/sim.py exactly so
# the dual views keep the exact direction conventions the single-arm agent learned.
# ---------------------------------------------------------------------------
def render_agentview_dual(
    scene: DualSimScene, poses: Dict[str, np.ndarray], widths: Dict[str, float]
) -> np.ndarray:
    """Oblique third-person view of the whole table with BOTH grippers.

    +X (FWD) toward image-top, +Y (LEFT) to image-left: the LEFT arm therefore
    appears on the image's LEFT, as on the real rig (the front camera does not
    mirror the arms).
    """
    img = np.full((VIEW_PX, VIEW_PX, 3), _C_BG, dtype=np.uint8)
    S, KX, KZ = 880.0, 640.0, 500.0

    def fdepth(x: float) -> float:  # fake perspective: nearer -> wider
        return 1.14 - 0.52 * (x - WORK_X[0]) / (WORK_X[1] - WORK_X[0])

    def proj(x: float, y: float, z: float) -> tuple:
        u = VIEW_PX / 2 - y * S * fdepth(x)
        v = 452.0 - (x - WORK_X[0]) * KX - (z - scene.z_table) * KZ
        return int(round(u)), int(round(v))

    # Table top + grid.
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

    # Cubes: far-to-near so nearer cubes overdraw.
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

    # Grippers, farther arm first so the nearer one overdraws. Each arm's link enters
    # from its own side of the frame -- that (not a label) is how they are told apart.
    base_y = {"left": 0.30, "right": -0.30}
    for side in sorted(SIDES, key=lambda s: -float(poses[s][0])):
        pose, width = poses[side], widths[side]
        ex, ey, ez = (float(v) for v in np.asarray(pose, dtype=float)[:3])
        cv2.ellipse(img, proj(ex, ey, scene.z_table), (14, 6), 0, 0, 360, _C_SHADOW, -1)
        cv2.line(img, proj(0.55, base_y[side], 0.55), proj(ex, ey, ez + 0.03), (70, 70, 78), 6)
        gu, gv = proj(ex, ey, ez)
        yaw = _yaw_of(pose)
        half_px = max(4, int(0.5 * width * S * fdepth(ex)))
        dx, dy = np.cos(yaw), np.sin(yaw)
        for sgn in (-1, 1):
            ju, jv = int(gu + sgn * half_px * dx), int(gv + sgn * half_px * dy * 0.5)
            cv2.rectangle(img, (ju - 3, jv - 12), (ju + 3, jv + 12), _C_JAW, -1)
        cv2.line(img, (gu - half_px, gv - 16), (gu + half_px, gv - 16), (60, 60, 66), 4)
        held = scene.held[side]
        if held is not None:
            cv2.rectangle(img, (gu - 9, gv - 9), (gu + 9, gv + 9), _C_CUBES.get(held, (200, 60, 60)), -1)
    return img


def render_wristview_dual(
    scene: DualSimScene,
    side: str,
    poses: Dict[str, np.ndarray],
    widths: Dict[str, float],
) -> np.ndarray:
    """Ego top-down view centered on ONE arm's gripper (jaws fixed at image center,
    zooms in as it lowers). The OTHER arm's gripper is drawn too when it is inside
    this view's span -- without it, a handover partner or an approaching collision
    would be invisible exactly where the agent looks most."""
    img = np.full((VIEW_PX, VIEW_PX, 3), _C_BG, dtype=np.uint8)
    pose = poses[side]
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

    # The OTHER arm, if visible from here: its two jaws + whatever it is holding.
    other = "right" if side == "left" else "left"
    opose, owidth = poses[other], widths[other]
    ox, oy = float(opose[0]), float(opose[1])
    ou, ov = proj(ox, oy)
    if -60 <= ou < VIEW_PX + 60 and -60 <= ov < VIEW_PX + 60:
        oyaw = _yaw_of(opose)
        oc, os_ = np.cos(oyaw), np.sin(oyaw)
        ohalf = max(6.0, 0.5 * owidth * ppm)
        for sgn in (-1, 1):
            ju, jv = int(ou + sgn * ohalf * oc), int(ov + sgn * ohalf * os_)
            cv2.rectangle(img, (ju - 4, jv - 14), (ju + 4, jv + 14), _C_JAW, -1)
        oheld = scene.held[other]
        if oheld is not None:
            cv2.rectangle(img, (ou - hpx, ov - hpx), (ou + hpx, ov + hpx),
                          _C_CUBES.get(oheld, (200, 60, 60)), -1)
        cv2.line(img, (ou - int(ohalf), ov - 18), (ou + int(ohalf), ov - 18), _C_ARM, 3)

    # This arm's jaws overlay (fixed at image center; separation tracks the width).
    yaw = _yaw_of(pose)
    c, s = np.cos(yaw), np.sin(yaw)
    half = max(10.0, 0.5 * widths[side] * ppm)
    jaw_l, jaw_w = 46, 12
    center = np.array([VIEW_PX / 2, VIEW_PX / 2])
    for sgn in (-1, 1):
        jc = center + np.array([sgn * half * c, sgn * half * s])
        along = np.array([s, -c])
        p1 = jc + along * jaw_l / 2 + np.array([c, s]) * jaw_w / 2 * sgn
        p2 = jc + along * jaw_l / 2 - np.array([c, s]) * jaw_w / 2 * sgn
        p3 = jc - along * jaw_l / 2 - np.array([c, s]) * jaw_w / 2 * sgn
        p4 = jc - along * jaw_l / 2 + np.array([c, s]) * jaw_w / 2 * sgn
        cv2.fillPoly(img, [np.array([p1, p2, p3, p4], dtype=int)], _C_JAW)
    held = scene.held[side]
    if held is not None:
        cu, cvv = int(center[0]), int(center[1])
        cv2.rectangle(img, (cu - hpx, cvv - hpx), (cu + hpx, cvv + hpx),
                      _C_CUBES.get(held, (200, 60, 60)), -1)
    else:
        cv2.drawMarker(img, (int(center[0]), int(center[1])), (80, 80, 90),
                       cv2.MARKER_CROSS, 16, 2)
    return img


class DualSimSession:
    """DualPiperSession-shaped session over the synthetic world.

    ``get_observation()`` returns the dual contract (``agentview`` / ``wrist_left``
    / ``wrist_right`` + per-side state); ``get_display_frames()`` returns the
    full-resolution renders for the browser streams (the real DualPiperSession has
    no display path yet, so the backend falls back to observation frames there).
    """

    def __init__(
        self,
        scene: DualSimScene,
        robots: Dict[str, SimDualPiperRobot],
        observation_resolution: int = 256,
    ) -> None:
        self.scene = scene
        self.robots = robots
        self.observation_resolution = int(observation_resolution)
        self._display: Dict[str, np.ndarray] = {}

    def connect(self) -> "DualSimSession":
        return self

    def close(self) -> None:
        pass

    def get_observation(self) -> dict:
        poses = {s: self.robots[s].get_ee_pose() for s in SIDES}
        widths = {s: float(self.robots[s].get_gripper_position()[0]) for s in SIDES}
        agent = render_agentview_dual(self.scene, poses, widths)
        wrists = {s: render_wristview_dual(self.scene, s, poses, widths) for s in SIDES}
        self._display = {
            "agentview": agent,
            "wrist_left": wrists["left"],
            "wrist_right": wrists["right"],
        }
        res = self.observation_resolution
        obs: dict = {
            name: np.ascontiguousarray(camera_utils.resize_with_pad(frame, res, res))
            for name, frame in self._display.items()
        }
        for s in SIDES:
            obs[s] = {"ee_pose": poses[s], "gripper_width": widths[s]}
        return obs

    def get_display_frames(self) -> Dict[str, np.ndarray]:
        """Full-resolution renders of the last observation, for the browser streams."""
        return self._display
