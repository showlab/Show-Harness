"""The real SO-101 as a :class:`So101ArmController`: the same motion core that runs on the
Workshop sim (scripts/trajectory/real2sim/backends/so101_workshop.py), with LeRobot I/O.

Only I/O lives here: joints in, joint targets out at a fixed tick rate, the world<-base
rotation and the cameras. Feasibility, refusal and the servo loop are the interpreter's.

- Task frame: as in the sim (world orientation, origin at the robot base, +X toward the work
  area). The base link is yawed +90 deg in that frame, as in the Workshop scene.
- One tick = 1/control_hz (30 Hz) and one control step = one tick, the same 30 Hz control
  rate as the sim's 60 Hz x 2 ticks.
- No force sensing: contact is inferred from the joint gap plus "no progress"
  (``contact_is_measured = False``).

The thresholds that work in the sim stop the real arm on every token; :data:`REAL_EXEC_KW`
and :data:`REAL_BACKEND_KW` hold the values measured on hardware (docs/so101.md).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from core.so101.units import JOINTS, rad_to_raw, raw_to_rad
from interpreters.so101_atomic_controller import So101ArmController

R_WORLD_BASE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # base +90 deg yaw

# So101AtomicExec keyword arguments for the real arm (sim defaults in the comments).
REAL_EXEC_KW = dict(
    tol_m=0.004,                            # 0.001: gravity sag and P=16 leave it a few mm short
    rot_tol_rad=float(np.radians(1.0)),     # 0.2 deg
    settle_tol_m=0.001,                     # 0.0003: one encoder tick (0.088 deg) of jitter
    settle_tol_rad=float(np.radians(0.3)),  # 0.05 deg  moves the fingertip ~0.4 mm
)
# So101ArmController keyword arguments for the real arm (sim defaults in the comments).
REAL_BACKEND_KW = dict(
    contact_tol_rad=0.15,       # 0.08: reached 0.087 rad with the hand 11 cm above the mat
    min_joint_margin_rad=0.10,  # 0.02: the servos stop tracking ~5 deg before a joint limit
)


class So101RealBackend(So101ArmController):
    # Camera contract, as in the sim backend: real cameras mounted to give the Workshop
    # cameras' raw view get the same transforms. Override per rig with keyword arguments.
    agentview_rotation_degrees = 90
    agentview_flip = "none"
    agentview_square_size = 256
    wrist_rotation_degrees = 180
    wrist_flip = "none"
    wrist_square_size = 256

    def __init__(self, robot, cameras: Optional[dict] = None, control_hz: float = 30.0, **kw) -> None:
        for key in ("agentview_rotation_degrees", "agentview_flip", "agentview_square_size",
                    "wrist_rotation_degrees", "wrist_flip", "wrist_square_size"):
            if key in kw:
                setattr(self, key, kw.pop(key))
        kw.setdefault("ticks_per_ctrl", 1)
        kw.setdefault("tick_dt", 1.0 / control_hz)
        super().__init__(**kw)
        self.robot = robot
        self.cameras = cameras or {}
        self._next_tick: Optional[float] = None
        self.contact_is_measured = False

    # ------------------------------------------------------------ I/O
    def _read_q(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return raw_to_rad([obs[f"{j}.pos"] for j in JOINTS])

    def _send(self, q_cmd: np.ndarray) -> None:
        raw = rad_to_raw(q_cmd)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, raw)})
        now = time.monotonic()  # hold the tick rate
        if self._next_tick is None:
            self._next_tick = now
        self._next_tick += self.tick_dt
        delay = self._next_tick - now
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = now  # fell behind: do not try to catch up

    def _r_world_base(self) -> np.ndarray:
        return R_WORLD_BASE

    # ------------------------------------------------------------ task / cameras
    def success(self) -> bool:
        return False  # judged by a person on the real arm

    def raw_frames(self):
        return self.cameras["external"].read(), self.cameras["wrist"].read()

    def grab_frames(self):
        from scripts.trajectory.real2sim.atomic_tokenizer import prepared_pair  # noqa: PLC0415

        return prepared_pair(self, *self.raw_frames())

    def reset(self, seed: int = 0, settle_ctrl_steps: int = 30) -> None:
        self._next_tick = None
        self._begin_episode(settle_ctrl_steps)
