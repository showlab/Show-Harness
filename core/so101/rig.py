"""Build the real SO-101 stack from configs/robot_so101_real.yaml.

    rig = open_rig("configs/robot_so101_real.yaml", port="COM3")
    rig.arm.move_to(rig.start_q)          # SafeArm, radians
    rig.backend.reset()                   # So101RealBackend (So101ArmController)
    rig.ex.run("MV_DOWN")                 # So101AtomicExec with the real-arm thresholds
    rig.close()

``mock=True`` replaces the follower with :class:`MockFollower` (and the cameras with video
files, when given), so the scripts and tests run without hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from core.so101.lerobot_io import MockFollower, SafeArm, connect_follower, load_joint_offset
from core.so101.real_backend import REAL_BACKEND_KW, REAL_EXEC_KW, So101RealBackend
from interpreters.so101_atomic_controller import So101AtomicExec

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "robot_so101_real.yaml"


@dataclass
class Rig:
    cfg: dict
    robot: Any            # SO101Follower / ExactScale / MockFollower
    arm: SafeArm
    backend: So101RealBackend
    ex: So101AtomicExec
    cameras: dict = field(default_factory=dict)

    @property
    def start_q(self) -> np.ndarray:
        return np.asarray(self.cfg["start_q_rad"], dtype=np.float64)

    @property
    def table_top(self) -> float:
        return float(self.cfg["table_top_task_z_m"])

    def go_begin(self) -> bool:
        """Start pose (joint move), then the begin hand pose (servo, not a token)."""
        from interpreters.so101_atomic_controller import ROLL_CLOSING_HORIZONTAL

        self.arm.move_to(self.start_q, duration_s=3.0)
        self.backend.reset()
        self.ex.grip_cmd = 1.0
        return self.ex.goto_pose(np.asarray(self.cfg["begin_task_m"], dtype=np.float64),
                                 np.radians(float(self.cfg["begin_pitch_deg"])), roll=ROLL_CLOSING_HORIZONTAL)

    def close(self) -> None:
        for c in self.cameras.values():
            c.close()
        self.arm.disconnect()


def load_config(path=DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def open_rig(config=DEFAULT_CONFIG, port: Optional[str] = None, mock: bool = False, dry_run: bool = False,
             cameras: Optional[dict] = None, joint_offset: Optional[str] = None, control_hz: float = 30.0,
             **exec_overrides) -> Rig:
    """``cameras``: {"external": ..., "wrist": ...} from core.so101.cameras.open_cameras, or None.
    ``control_hz`` is the real tick rate; tests raise it so the mock does not sleep."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    rb = cfg["robot"]
    offset_path = joint_offset or rb.get("joint_offset")
    if mock:
        robot = MockFollower(cfg["start_q_rad"])
    else:
        robot = connect_follower(
            port or rb["port"], rb["id"],
            max_relative_target=rb.get("max_relative_target", 10),
            exact_scale=bool(rb.get("exact_scale", False)),
            offset_deg=load_joint_offset(offset_path) if offset_path else None,
            strong_grip=bool(rb.get("strong_grip", False)),
            stiff_arm=bool(rb.get("stiff_arm", False)),
        )
    arm = SafeArm(robot, max_step_rad=float(rb.get("max_step_rad", 0.08)), dry_run=dry_run)
    table_top = float(cfg["table_top_task_z_m"])
    backend = So101RealBackend(arm, cameras=cameras or {}, control_hz=control_hz, **REAL_BACKEND_KW)
    backend.hand_floor_task_z = table_top + float(cfg.get("hand_floor_above_table_m", 0.003))
    if cfg.get("workspace"):
        backend.workspace = cfg["workspace"]
    ex = So101AtomicExec(backend, **{**REAL_EXEC_KW, **exec_overrides})
    return Rig(cfg=cfg, robot=robot, arm=arm, backend=backend, ex=ex, cameras=cameras or {})
