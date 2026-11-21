"""Joint units: Workshop radians (sim, interpreter) <-> LeRobot follower values.

LeRobot's SO-101 follower (default ``RANGE_M100_100`` mode) reports and accepts each arm
joint as -100..100 across the range recorded by ``lerobot-calibrate`` and the gripper as
0..100. The Workshop maps those to radians with fixed per-joint degree ranges
(``SO101_USD_MAPPING`` in its utils/lerobot_interface.py); this module is that mapping, so
the sim, the interpreter and the real arm share one formula.

The mapping is exact only when the calibration sweep covered exactly those degree ranges.
:class:`core.so101.lerobot_io.ExactScale` removes that dependence (see there).
"""
from __future__ import annotations

import numpy as np

# LeRobot motor names, in Workshop joint order (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw).
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Workshop SO101_USD_MAPPING (degrees).
MINS_DEG = np.array([-110, -100, -100, -95, -160, -10.0])
MAXS_DEG = np.array([110, 100, 90, 95, 160, 100.0])
Q_MIN = np.radians(MINS_DEG)
Q_MAX = np.radians(MAXS_DEG)


def rad_to_raw(q) -> np.ndarray:
    """Workshop radians -> LeRobot values (arm -100..100, gripper 0..100)."""
    n = (np.degrees(np.asarray(q, dtype=np.float64)) - MINS_DEG) / (MAXS_DEG - MINS_DEG)
    raw = np.empty(6)
    raw[:5] = n[:5] * 200.0 - 100.0
    raw[5] = n[5] * 100.0
    return raw


def raw_to_rad(raw) -> np.ndarray:
    """LeRobot values -> Workshop radians."""
    raw = np.asarray(raw, dtype=np.float64)
    n = np.empty(6)
    n[:5] = (raw[:5] + 100.0) / 200.0
    n[5] = raw[5] / 100.0
    return np.radians(MINS_DEG + n * (MAXS_DEG - MINS_DEG))


def raw_dict(raw) -> dict:
    return {f"{j}.pos": float(v) for j, v in zip(JOINTS, raw)}


def from_obs(obs: dict) -> np.ndarray:
    return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float64)
