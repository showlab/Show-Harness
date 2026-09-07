"""Shared joint-pose moves for the Piper arms (``go_begin`` / ``go_rest``).

Moving an arm to a saved joint configuration is used in several places, so it lives
here rather than being duplicated:
  * ``scripts/piper/go_begin.py`` -- run standalone (also backs go_rest.sh),
  * the dual teleop collector -- resets BOTH arms when a recording stops (P),
  * ``scripts/run_real.py`` -- moves to begin on init and resets when a rollout finishes.

The poses are ``begin_joints`` (auto-entered on init) and ``rest_joints`` (idle/park,
on demand), 6 joint angles in radians, resolved from ``arms.<side>`` in
configs/robot_piper.yaml -- ``begin_joints`` comes from the NAMED start pose the shared
``begin_pose`` key selects (see :mod:`core.piper.config`), so a rig can keep several.
An unset pose is a graceful no-op (warn + skip) so the auto-triggers never crash a run
before a pose has been captured. Capture one (and, by default, the other arm's mirror of
it -- see :func:`mirror_joints`) with
``python scripts/piper/go_begin.py --arm left --pose <name> --capture --write``.

The rig is DUAL-ARM: :func:`go_begin_dual` drives both arms **simultaneously** (one
thread per arm), which is what every reset path uses -- the arms are independent ROS
nodes on separate CAN buses and separate command topics, so they move together rather
than one-then-the-other.
"""
from __future__ import annotations

import threading
from typing import Any, Mapping, Optional, Sequence

import numpy as np

PIPER_DOF = 6

# The two arms face each other across the table, so a pose captured on one arm maps to
# the other by negating its YAW joints (1, 4, 6 -- the rotations about the vertical /
# roll axes); the pitch joints (2, 3, 5) keep the same shape. This is the same relation
# the shipped left/right begin+rest poses already encode. It is a geometric convention
# of the rig, so it is overridable per config (`mirror_signs`) -- and a mirrored pose is
# a STARTING POINT: verify it on hardware, and re-capture directly if the arms are not
# mounted symmetrically.
MIRROR_SIGNS: tuple[float, ...] = (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0)


def mirror_joints(
    joints: Sequence[float], signs: Optional[Sequence[float]] = None
) -> list[float]:
    """Mirror a captured joint pose from one arm to the other (see :data:`MIRROR_SIGNS`)."""
    q = np.asarray(joints, dtype=float).reshape(-1)[:PIPER_DOF]
    if q.size < PIPER_DOF:
        raise ValueError(
            f"mirror_joints: expected {PIPER_DOF} joint values, got {q.size}: "
            f"{np.round(q, 4).tolist()}"
        )
    s = np.asarray(signs if signs is not None else MIRROR_SIGNS, dtype=float).reshape(-1)
    if s.size != PIPER_DOF:
        raise ValueError(
            f"mirror_signs must have {PIPER_DOF} entries, got {s.size}: {s.tolist()}"
        )
    # +0.0 normalizes the negative zero a sign flip produces on a 0.0 joint (-0.0 is
    # valid YAML but reads as a typo in a config a human maintains).
    return [round(float(v) + 0.0, 5) for v in (q * s)]


def go_begin(
    robot: Any,
    joints: Optional[Sequence[float]],
    time_to_go: float = 3.0,
    verbose: bool = True,
    label: str = "begin",
    open_gripper: bool = False,
) -> bool:
    """Move ONE ``robot`` to a saved joint config (6 rad). True if the move ran.

    ``label`` names the target in the log ("begin" / "rest", optionally prefixed with the
    arm). Empty ``joints`` is a no-op (returns False) -- the auto-triggers rely on this so
    a run never fails just because no pose is configured yet.

    ``open_gripper`` opens the gripper AFTER the arm reaches the pose, so a reset leaves
    it empty (any held object is released at the target pose). It runs after the move on
    purpose: ``move_to_joint_positions`` clears the cached Cartesian setpoint, so the open
    command falls back to the measured (arrived) pose and cannot lunge to a stale one.
    """
    if not joints:
        if verbose:
            print(
                f"[go-{label}] joints are not set in the config; skipping. Capture with: "
                "python scripts/piper/go_begin.py --arm <left|right> --capture --write"
            )
        return False
    q = np.asarray(joints, dtype=float).reshape(-1)
    if q.size < PIPER_DOF:
        raise ValueError(
            f"{label}: expected {PIPER_DOF} joint values, got {q.size}: "
            f"{np.round(q, 4).tolist()}"
        )
    q = q[:PIPER_DOF]
    if verbose:
        print(f"[go-{label}] moving to {np.round(q, 4).tolist()} over {float(time_to_go):.1f}s ...")
    robot.move_to_joint_positions(q, float(time_to_go))
    if open_gripper:
        opener = getattr(robot, "control_gripper", None)
        if callable(opener):
            if verbose:
                print(f"[go-{label}] opening gripper.")
            opener(False)  # control_gripper: False = OPEN
        elif verbose:
            print(f"[go-{label}] robot has no control_gripper(); skipping gripper open.")
    if verbose:
        print(f"[go-{label}] reached{' (gripper open)' if open_gripper else ''}.")
    return True


def go_begin_dual(
    robots: Mapping[str, Any],
    joints: Mapping[str, Optional[Sequence[float]]],
    time_to_go: float = 3.0,
    verbose: bool = True,
    label: str = "begin",
    open_gripper: bool = False,
) -> bool:
    """Move BOTH arms to their poses **simultaneously** -- one thread per arm.

    ``robots``/``joints`` are keyed by side ("left"/"right"). Each arm is an independent
    ROS node on its own CAN bus and command topic, so the per-arm
    ``move_to_joint_positions`` calls (each of which blocks while it streams its
    trajectory) run concurrently and the arms travel together.

    Returns True only if EVERY arm actually moved. An exception in one arm does not
    silently strand the other: both threads are joined, then the first error is re-raised.
    """
    moved: dict[str, bool] = {}
    errors: dict[str, BaseException] = {}

    def _run(side: str) -> None:
        try:
            moved[side] = go_begin(
                robots[side],
                joints.get(side),
                time_to_go=time_to_go,
                verbose=verbose,
                label=f"{side} {label}",
                open_gripper=open_gripper,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised after both arms are joined
            errors[side] = exc

    threads = [threading.Thread(target=_run, args=(s,), name=f"go-{label}-{s}") for s in robots]
    for t in threads:
        t.start()
    for t in threads:  # always join both, even if one already failed
        t.join()

    if errors:
        side, exc = next(iter(errors.items()))
        raise RuntimeError(f"{side} arm failed to reach {label}: {exc}") from exc
    return bool(moved) and all(moved.values())
