"""Real SO-101 I/O (core/so101/) without hardware: unit mapping, ExactScale, SafeArm, and the
real backend driving So101AtomicExec through a stand-in follower.

The stand-in follower speaks the LeRobot Robot API in LeRobot units and moves each read a
fraction of the way to the last command minus a gravity sag, like the arm with P=16.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from core.so101.lerobot_io import ExactScale, MockFollower, SafeArm
from core.so101.real_backend import REAL_BACKEND_KW, REAL_EXEC_KW, So101RealBackend
from core.so101.rig import open_rig
from core.so101.units import JOINTS, Q_MAX, Q_MIN, from_obs, rad_to_raw, raw_dict, raw_to_rad
from interpreters.so101_atomic_controller import ROLL_CLOSING_HORIZONTAL, So101AtomicExec

START_Q = np.array([-0.251174, -0.612255, -0.064738, 1.511172, -1.600151, 0.4498])
# Calibration recorded on one real arm (2026-10): the wrist_roll sweep covered 227 deg where the
# Workshop mapping assumes 320 deg.
CAL = {j: SimpleNamespace(range_min=lo, range_max=hi) for j, (lo, hi) in zip(JOINTS, [
    (847, 3593), (820, 3263), (979, 3122), (873, 3212), (1173, 3758), (1526, 3059)])}


class FakeFollower:
    """Records the last action; reads back what it was sent (an ideal servo)."""

    def __init__(self, raw0=None):
        self.last = raw_dict(np.zeros(6) if raw0 is None else raw0)

    def send_action(self, a):
        self.last = dict(a)
        return a

    def get_observation(self):
        return dict(self.last)

    def disconnect(self):
        pass


class LaggyFollower(MockFollower):
    """MockFollower plus a constant sag on the arm joints."""

    def __init__(self, q0, sag=(0.0, 0.01, 0.015, 0.008, 0.0, 0.0)):
        super().__init__(q0)
        self.sag = np.asarray(sag)

    def get_observation(self):
        self.q += 0.5 * (self.target - self.sag - self.q)
        return raw_dict(rad_to_raw(self.q))


def test_unit_mapping_round_trip():
    q = np.radians([-30.0, 20.0, -50.0, 80.0, -120.0, 40.0])
    np.testing.assert_allclose(raw_to_rad(rad_to_raw(q)), q, atol=1e-12)
    np.testing.assert_allclose(rad_to_raw(Q_MIN), [-100] * 5 + [0], atol=1e-9)
    np.testing.assert_allclose(rad_to_raw(Q_MAX), [100] * 5 + [100], atol=1e-9)


def test_exact_scale_round_trip_and_measured_wrist_roll_error():
    f = FakeFollower()
    es = ExactScale(f, calibration=CAL)
    es.send_action(raw_dict(rad_to_raw(START_Q)))
    np.testing.assert_allclose(raw_to_rad(from_obs(es.get_observation())), START_Q, atol=1e-9)
    # Where the start pose physically ends up WITHOUT the correction, in encoder degrees.
    old = from_obs(raw_dict(rad_to_raw(START_Q)))
    frac = (old[4] + 100) / 200
    lo, hi = CAL["wrist_roll"].range_min, CAL["wrist_roll"].range_max
    tick = lo + frac * (hi - lo)
    zero = lo + 0.5 * (hi - lo)
    phys_deg = (tick - zero) * 360 / 4096
    assert phys_deg - np.degrees(START_Q[4]) == pytest.approx(26.6, abs=0.2)


def test_exact_scale_offset_sign():
    """Commanding start+d with offset o equals commanding start with offset o-d (what
    tune_start.py saves after nudging the arm by d until it matches the sim)."""
    o0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    d = np.array([0.0, 0.0, 0.0, -3.0, 12.0, 0.0])
    f = FakeFollower()
    ExactScale(f, o0, calibration=CAL).send_action(raw_dict(rad_to_raw(START_Q + np.radians(d))))
    a = from_obs(f.last)
    ExactScale(f, o0 - d, calibration=CAL).send_action(raw_dict(rad_to_raw(START_Q)))
    np.testing.assert_allclose(from_obs(f.last), a, atol=1e-9)


def test_safearm_clips_step_and_range():
    f = FakeFollower(rad_to_raw(START_Q))
    arm = SafeArm(f, max_step_rad=0.08)
    sent = arm.send_q(START_Q + np.array([0.5, 0, 0, 0, 0, 0]))
    assert sent[0] == pytest.approx(START_Q[0] + 0.08)
    assert arm.clipped == 1
    arm = SafeArm(FakeFollower(rad_to_raw(Q_MAX - 0.01)), max_step_rad=1.0)
    assert np.all(arm.send_q(Q_MAX + 0.5) <= Q_MAX + 1e-12)


def test_safearm_dry_run_never_sends_and_advances_from_last_command():
    f = FakeFollower(rad_to_raw(START_Q))
    arm = SafeArm(f, max_step_rad=0.08, dry_run=True)
    before = dict(f.last)
    target = START_Q + np.array([0.5, 0, 0, 0, 0, 0])
    arm.send_q(target)
    second = arm.send_q(target)
    assert f.last == before  # nothing reached the follower
    assert second[0] == pytest.approx(START_Q[0] + 0.16)  # two 0.08 steps from the last command


def _real_exec(follower):
    be = So101RealBackend(SafeArm(follower), control_hz=1e4, **REAL_BACKEND_KW)
    be.hand_floor_task_z = 0.038
    ex = So101AtomicExec(be, **REAL_EXEC_KW)
    be.reset()
    assert ex.goto_pose(np.array([0.25, -0.02, 0.17]), np.radians(60.0), roll=ROLL_CLOSING_HORIZONTAL)
    return be, ex


def test_real_backend_executes_tokens_on_a_sagging_follower():
    be, ex = _real_exec(LaggyFollower(START_Q))
    for tok, axis, sign in (("MV_DOWN", 2, -1), ("MV_FWD", 0, 1), ("MV_UP", 2, 1)):
        p0 = be.tcp_pos()
        ex.last_blocked = None
        ex.run(tok)
        assert ex.last_blocked is None, tok
        assert sign * (be.tcp_pos() - p0)[axis] == pytest.approx(0.02, abs=0.004), tok


def test_real_thresholds_are_the_measured_ones():
    assert REAL_EXEC_KW["tol_m"] == 0.004
    assert REAL_BACKEND_KW["contact_tol_rad"] == 0.15
    assert REAL_BACKEND_KW["min_joint_margin_rad"] == 0.10


def test_open_rig_mock_runs_a_token():
    rig = open_rig(mock=True, control_hz=1e4)
    assert rig.go_begin()
    ok, why = rig.ex.check("MV_DOWN")
    assert ok, why
    rig.ex.run("MV_DOWN")
    assert rig.ex.last_blocked is None
    rig.close()
