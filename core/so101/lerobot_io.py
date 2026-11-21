"""LeRobot SO-101 follower I/O: connection, angle correction, command limits.

Everything here speaks the LeRobot Robot API (``get_observation() -> {"<joint>.pos": v}``,
``send_action({"<joint>.pos": v})``, lerobot >= 0.4), so the wrappers stack:

    SO101Follower -> ExactScale (optional) -> SafeArm -> So101RealBackend

- :func:`connect_follower` connects the follower, optionally overrides motor settings
  (``strong_grip`` / ``stiff_arm``) and retries joint reads that get no status packet.
- :class:`ExactScale` re-derives angles from encoder ticks, so they no longer depend on how
  far each joint was swept during calibration, and adds a per-joint zero offset.
- :class:`SafeArm` clips every command to the joint ranges and to ``max_step_rad`` per
  command, relative to the measured joints.
- :class:`MockFollower` stands in for the arm (first-order lag), for dry runs and tests.

Measured on one arm (2026-10, docs/so101.md): the calibration sweep covered 227 deg of
wrist_roll where the Workshop mapping assumes 320 deg, which put the start pose 27 deg off.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.so101.units import JOINTS, MAXS_DEG, MINS_DEG, Q_MAX, Q_MIN, from_obs, rad_to_raw, raw_dict, raw_to_rad

TICKS_PER_TURN = 4096  # STS3215 encoder

# LeRobot's SO101Follower.configure() sets every motor to P=16 and the gripper to 50 % torque,
# 50 % current and 25 % torque when overloaded. A grasp that stalls on a vial counts as an
# overload, so the grip drops to 25 % and the vial slipped while held. strong_grip raises the
# gripper only (P back to the motor default 32, 70 % limits, 50 % when overloaded).
STRONG_GRIP = {"P_Coefficient": 32, "Max_Torque_Limit": 700, "Protection_Current": 350, "Overload_Torque": 50}
# With P=16 the arm sagged ~1 cm holding a weighted vial with the shoulder back and the elbow
# folded, and the executor read the shortfall as a push. stiff_arm sets the 5 arm joints to 32.
STIFF_ARM_P = 32


def lerobot_lite() -> None:
    """Make LeRobot's SO-101 follower importable without torch / accelerate / datasets.

    lerobot 0.4.1 imports them from the motor-bus path (lerobot.utils.utils) but the arm only
    uses helpers that do not need them. Real modules are used when installed; missing ones are
    replaced by name-only stand-ins that raise if anything actually calls them.
    """
    import importlib.util
    import sys
    import types

    class _Absent:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, attr):
            return _Absent(f"{self._name}.{attr}")

        def __call__(self, *a, **k):
            raise RuntimeError(f"{self._name} is not available (lite install without torch)")

        def __or__(self, other):  # type annotations of the form X | Y
            return self

        __ror__ = __or__

    def stub(name: str, **attrs):
        top = name.split(".")[0]
        if name in sys.modules or (top not in sys.modules and importlib.util.find_spec(top) is not None):
            return
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr, _n=name: _Absent(f"{_n}.{attr}")  # type: ignore[attr-defined]
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    stub("torch")
    stub("accelerate", Accelerator=_Absent("accelerate.Accelerator"))
    if importlib.util.find_spec("datasets") is None:
        for n in ("datasets", "datasets.utils"):
            stub(n)
        stub("datasets.utils.logging", disable_progress_bar=lambda: None, enable_progress_bar=lambda: None)


def load_joint_offset(path) -> list:
    """``offset_deg`` from a joint_offset.json written by scripts/so101/tune_start.py."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["offset_deg"]


def connect_follower(
    port: str,
    robot_id: str,
    max_relative_target: Optional[float] = 10.0,
    exact_scale: bool = False,
    offset_deg: Optional[Sequence[float]] = None,
    strong_grip: bool = False,
    stiff_arm: bool = False,
    read_retries: int = 3,
):
    """Connect a LeRobot SO101Follower (calibrated beforehand with lerobot-calibrate).

    ``max_relative_target`` is LeRobot's own per-command limit (LeRobot units) and stacks with
    SafeArm. ``offset_deg`` implies ``exact_scale``. Motor overrides are written right after
    LeRobot's configure(), before torque comes back on.
    """
    lerobot_lite()
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={},
                                              max_relative_target=max_relative_target))
    if strong_grip or stiff_arm:
        base_configure = robot.configure

        def configure() -> None:
            base_configure()
            with robot.bus.torque_disabled():
                for reg, v in (STRONG_GRIP.items() if strong_grip else ()):
                    robot.bus.write(reg, "gripper", v, num_retry=3)
                for j in (JOINTS[:5] if stiff_arm else ()):
                    robot.bus.write("P_Coefficient", j, STIFF_ARM_P, num_retry=3)

        robot.configure = configure
    robot.connect()

    read_once = robot.get_observation

    def get_observation() -> dict:
        # A sync read occasionally gets no status packet; one miss should not end an episode.
        for i in range(read_retries + 1):
            try:
                return read_once()
            except ConnectionError:
                if i == read_retries:
                    raise
                time.sleep(0.02)

    robot.get_observation = get_observation
    if offset_deg is not None or exact_scale:
        return ExactScale(robot, offset_deg)
    return robot


def gripper_temperature(robot) -> Optional[int]:
    """Gripper motor temperature (deg C), or None when it cannot be read (mock, wrapper)."""
    bus = getattr(robot, "bus", None)
    try:
        return int(bus.read("Present_Temperature", "gripper", normalize=False))
    except Exception:  # noqa: BLE001
        return None


class ExactScale:
    """Re-read LeRobot values as encoder angles (1 tick = 360/4096 deg).

    LeRobot normalises each joint over the range swept during calibration and units.py maps
    that onto fixed Workshop degree ranges, so a sweep wider or narrower than the Workshop
    range scales every angle. Here the angle is ``(tick - zero) * 360 / 4096 + offset_deg``,
    with ``zero`` at the same fraction of the swept range as units.py's zero, so the two agree
    when the sweep was symmetric. ``offset_deg`` (scripts/so101/tune_start.py) removes what
    remains: sim angle = tick angle + offset_deg.

    Input and output stay in LeRobot values (Workshop-nominal ones), so it drops in between
    the follower and anything that uses units.py.
    """

    def __init__(self, robot, offset_deg: Optional[Sequence[float]] = None, calibration=None) -> None:
        self.robot = robot
        cal = calibration if calibration is not None else robot.calibration
        self.lo = np.array([cal[j].range_min for j in JOINTS], dtype=np.float64)
        self.span = np.array([cal[j].range_max - cal[j].range_min for j in JOINTS], dtype=np.float64)
        self.zero = self.lo + (-MINS_DEG / (MAXS_DEG - MINS_DEG)) * self.span
        self.offset_deg = np.zeros(6) if offset_deg is None else np.asarray(offset_deg, dtype=np.float64)

    @staticmethod
    def _frac(raw: np.ndarray) -> np.ndarray:
        f = (raw + 100.0) / 200.0
        f[5] = raw[5] / 100.0
        return f

    @staticmethod
    def _unfrac(f: np.ndarray) -> np.ndarray:
        raw = f * 200.0 - 100.0
        raw[5] = f[5] * 100.0
        return raw

    def to_nominal(self, raw) -> np.ndarray:
        tick = self.lo + self._frac(np.asarray(raw, dtype=np.float64)) * self.span
        return rad_to_raw(np.radians((tick - self.zero) * 360.0 / TICKS_PER_TURN + self.offset_deg))

    def to_real(self, raw) -> np.ndarray:
        deg = np.degrees(raw_to_rad(raw)) - self.offset_deg
        tick = self.zero + deg * TICKS_PER_TURN / 360.0
        return self._unfrac((tick - self.lo) / self.span)

    def get_observation(self) -> dict:
        obs = dict(self.robot.get_observation())
        obs.update(raw_dict(self.to_nominal(from_obs(obs))))
        return obs

    def send_action(self, action: dict) -> dict:
        self.robot.send_action(raw_dict(self.to_real(from_obs(action))))
        return action

    def disconnect(self) -> None:
        self.robot.disconnect()

    @property
    def bus(self):
        return self.robot.bus


class SafeArm:
    """Every command clipped to the Workshop joint ranges and to ``max_step_rad`` per joint per
    command, measured from the current joints (from the last command in a dry run, where the
    arm does not move). Same Robot API as what it wraps, plus radian helpers."""

    def __init__(self, robot, max_step_rad: float = 0.08, dry_run: bool = False) -> None:
        self.robot = robot
        self.max_step_rad = float(max_step_rad)
        self.dry_run = bool(dry_run)
        self.clipped = 0  # commands that hit a limit (for the log)
        self._last_cmd: Optional[np.ndarray] = None

    def get_observation(self) -> dict:
        return self.robot.get_observation()

    def send_action(self, action: dict) -> dict:
        q_safe = self.limit(raw_to_rad(from_obs(action)))
        self._last_cmd = q_safe
        out = raw_dict(rad_to_raw(q_safe))
        if not self.dry_run:
            self.robot.send_action(out)
        return out

    def q(self) -> np.ndarray:
        return raw_to_rad(from_obs(self.robot.get_observation()))

    def send_q(self, q_cmd) -> np.ndarray:
        self.send_action(raw_dict(rad_to_raw(np.asarray(q_cmd, dtype=np.float64))))
        return self._last_cmd

    def limit(self, q_cmd) -> np.ndarray:
        q_cmd = np.clip(np.asarray(q_cmd, dtype=np.float64), Q_MIN, Q_MAX)
        base = self._last_cmd if (self.dry_run and self._last_cmd is not None) else self.q()
        if np.any(np.abs(q_cmd - base) > self.max_step_rad + 1e-9):
            self.clipped += 1
        return base + np.clip(q_cmd - base, -self.max_step_rad, self.max_step_rad)

    def move_to(self, q_target, duration_s: float = 3.0, hz: float = 30.0) -> None:
        """Cosine-interpolated move from the current joints (the Workshop's move_to_pose shape)."""
        start = self.q()
        n = max(1, int(duration_s * hz))
        for i in range(1, n + 1):
            a = (1 - np.cos(np.pi * i / n)) / 2
            self.send_q(start + a * (np.asarray(q_target) - start))
            time.sleep(1.0 / hz)

    def hold(self) -> None:
        if not self.dry_run:
            self.send_q(self.q())

    def disconnect(self) -> None:
        self.robot.disconnect()


class MockFollower:
    """Stand-in follower: each read moves the joints half way to the last command."""

    def __init__(self, q0) -> None:
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.target = self.q.copy()

    def get_observation(self) -> dict:
        self.q += 0.5 * (self.target - self.q)
        return raw_dict(rad_to_raw(self.q))

    def send_action(self, action: dict) -> dict:
        self.target = raw_to_rad(from_obs(action))
        return action

    def disconnect(self) -> None:
        pass
