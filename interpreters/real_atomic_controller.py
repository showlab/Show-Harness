"""Arm-agnostic real-robot atomic-action controller.

This is the real-robot counterpart of :class:`interpreters.atomic_controller.AtomicController`.
A sim interpreter turns the VLM's atomic tokens (``MV_FWD`` ... ``GRASP`` ...)
into a normalized action consumed by the simulator's ``env.step``. On a real
arm there is no normalization layer: we command **absolute end-effector
pose setpoints** to the robot's Cartesian controller. This class reads the
primitive vectors / step sizes from a primitives yaml and applies them as metric
end-effector deltas in the robot **base frame**.

Token semantics (unit vectors come from the primitives yaml -- the single
source of truth for per-robot axis signs):

    MV_FWD / MV_BACK / MV_LEFT / MV_RIGHT / MV_UP / MV_DOWN -> translations
    ROTATE_CW / ROTATE_CCW -> yaw about base +Z
    GRASP -> close gripper   RELEASE -> open gripper   DONE -> terminate

Motion frame (``motion_frame``): ``"base"`` (default) executes MV_* directly as
base-frame vectors. ``"wrist"`` rotates the horizontal component of every MV_* by the
gripper's current HEADING -- the horizontal projection of the tool axis (:attr:`TOOL_AXIS`
in the EEF frame) -- so MV_FWD moves along wherever the gripper points (at constant
height, MV_UP/DOWN stay world-vertical) and the wrist-camera view's directions are exact
by construction, at any yaw and on either of two mirrored arms. The heading comes from
the commanded setpoint orientation, so it is noise-free and constant unless yaw changes.
:meth:`step` also accepts a per-command frame override (``motion_frame=...``), used by
the view-select plugin to execute each move in the frame of the view that guided it.

The controller keeps an internal target pose (the commanded *setpoint*) so motion
is exact and does not accumulate sensor noise; it re-syncs from the measured pose
on :meth:`sync_from_robot`.

Safety: an optional **Z floor** (``z_floor_m`` / :meth:`set_z_floor`) locks the
minimum end-effector height. Once set -- e.g. captured with the gripper resting on
the tabletop -- the commanded setpoint is never allowed below it, so ``MV_DOWN`` (or
any negative-Z command) cannot drive the arm down past that height.

Hardware coupling lives entirely behind the duck-typed ``robot`` object
(``get_ee_pose`` / ``get_gripper_position`` / ``control_gripper`` /
``update_desired_ee_pose``) plus the per-robot subclass:

    interpreters.franka_atomic_controller.FrankaAtomicController  (Franka 7-DoF)
    interpreters.piper_atomic_controller.PiperAtomicController    (AgileX Piper 6-DoF)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# The shared vocabulary: the SAME VLM tokens drive every embodiment.
from core.action_units import MOVE_ATOMS, ROTATE_ATOMS, STOP_ATOM
from core.action_units import (  # noqa: F401  (re-exported)
    DONE_ATOM,
    GRASP_ATOM,
    GRIPPER_ATOMS,
    RELEASE_ATOM,
)

# Safety clamps per command (mirrors FrankaPolicyRunner in main_pi05.py).
DEFAULT_MAX_POSITION_DELTA_M = 0.05   # 5 cm / command
DEFAULT_MAX_ROTATION_DELTA_RAD = 0.2  # ~11 deg / command

# Gripper width below this (meters) counts as "closed". Sized for the Franka Hand
# (~0.08 m stroke); per-robot subclasses / configs override it.
GRIPPER_CLOSE_THRESHOLD_M = 0.07

# Gripper-settle detection defaults. Class attributes on RealAtomicController (so a
# per-robot subclass can retune them for its gripper's sensor dynamics); the module
# constants remain the canonical defaults.
#
# Two consecutive width reads within this (meters) count as "the gripper stopped moving".
GRIPPER_STABLE_EPS_M = 0.002

# A width change of more than this (meters) from the pre-command width counts as "the
# fingers have actually started actuating". Used to avoid mistaking the not-yet-moving
# gripper (two equal early reads) for a settled one and returning the stale start width.
GRIPPER_MOTION_EPS_M = 0.005

# The width must stay stable for at least this long (seconds) before it counts as settled,
# so a brief mid-close plateau (the lagging sensor pausing on its way to ~0) is not
# mistaken for the final width and does not trigger an early empty-check return.
GRIPPER_SUSTAIN_S = 0.3


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] -> Euler [roll, pitch, yaw] (radians, extrinsic xyz)."""
    return R.from_quat(np.asarray(quat, dtype=float)).as_euler("xyz", degrees=False)


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Euler [roll, pitch, yaw] (radians, extrinsic xyz) -> quaternion [qx, qy, qz, qw]."""
    return R.from_euler("xyz", np.asarray(euler, dtype=float), degrees=False).as_quat()


@dataclass
class AtomicStepResult:
    """Per-command record, returned by :meth:`RealAtomicController.step`."""

    token: str
    kind: str  # move | rotate | stop | gripper | done | unknown
    intended_delta_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    intended_yaw_rad: float = 0.0
    pre_pose: Optional[np.ndarray] = None     # measured EEF pose before [x,y,z,qx,qy,qz,qw]
    target_pose: Optional[np.ndarray] = None  # commanded setpoint
    post_pose: Optional[np.ndarray] = None    # measured EEF pose after
    gripper_closed: Optional[bool] = None
    done: bool = False
    grasp_empty: bool = False  # a GRASP closed below the empty-grasp width and was reopened
    note: str = ""
    # Which per-command translation magnitude a MOVE used: "fine" | "coarse" (the
    # variable-step plugin's choice), "up" (the dedicated MV_UP distance), or "" when
    # no choice was made (fixed step, or not a translation). step_m is that magnitude.
    step_kind: str = ""
    step_m: float = 0.0


class RealAtomicController:
    """Maps atomic VLM tokens to absolute Cartesian setpoints on a real arm."""

    # Log prefix for the per-command console lines; per-robot subclasses override it.
    LOG_TAG = "atomic"

    # Tool axis in the EEF frame whose horizontal projection is the gripper's HEADING
    # (used by motion_frame="wrist"). +Z verified on the Piper: at the calibrated begin
    # poses its projection tracks joint 1 within ~1.5 deg and points down-forward at the
    # workspace. Override per robot if the flange convention differs.
    TOOL_AXIS = (0.0, 0.0, 1.0)

    # Gripper-settle tuning (see the module constants above for semantics). Class
    # attributes so a per-robot subclass can retune them without forking
    # :meth:`_await_gripper_settled`.
    GRIPPER_STABLE_EPS_M = GRIPPER_STABLE_EPS_M
    GRIPPER_MOTION_EPS_M = GRIPPER_MOTION_EPS_M
    GRIPPER_SUSTAIN_S = GRIPPER_SUSTAIN_S

    def __init__(
        self,
        robot: Any,
        atomic_primitives: dict[str, Any],
        step_m: float,
        yaw_step_rad: float,
        max_position_delta_m: float = DEFAULT_MAX_POSITION_DELTA_M,
        max_rotation_delta_rad: float = DEFAULT_MAX_ROTATION_DELTA_RAD,
        settle_steps: int = 6,
        settle_dt_s: float = 0.1,
        level_orientation: bool = True,
        gripper_close_threshold_m: float = GRIPPER_CLOSE_THRESHOLD_M,
        grasp_min_width_m: Optional[float] = None,
        grasp_open_width_m: float = 0.06,
        gripper_settle_s: float = 2.5,
        gripper_min_settle_s: float = 0.5,
        gripper_poll_dt_s: float = 0.05,
        z_floor_m: Optional[float] = None,
        capture_z_floor_on_sync: bool = False,
        ensure_controller: Optional[Callable[[], None]] = None,
        variable_step_plugin: Any = None,
        table_height_m: Optional[float] = None,
        up_step_m: Optional[float] = None,
        smooth_plugin: Any = None,
        rotation_plugin: Any = None,
        motion_frame: str = "base",
        verbose: bool = True,
    ) -> None:
        """
        Args:
            robot: A duck-typed robot interface providing ``get_ee_pose()`` ->
                [x,y,z,qx,qy,qz,qw] (meters, base frame), ``get_gripper_position()``
                -> [width_m], ``control_gripper(close: bool)`` and
                ``update_desired_ee_pose(pose7)`` (e.g.
                core.franka.franka_interface.FrankaInterface or
                core.piper.piper_interface.PiperInterface, or their mocks).
            atomic_primitives: The ``atomic_primitives`` mapping from the primitives
                yaml (MV_* unit vectors and ROTATE_* signs).
            step_m: Per-step translation magnitude in meters (primitives yaml: step_m).
            yaw_step_rad: Per-step yaw magnitude in radians (primitives yaml: yaw_step_rad).
            max_position_delta_m: Safety clamp on the per-command translation.
            max_rotation_delta_rad: Safety clamp on the per-command yaw.
            settle_steps: How many times the setpoint is (re)commanded per atomic step,
                giving the robot's controller time to converge. Mirrors
                ``sim_steps_per_decision`` in the sim loops.
            settle_dt_s: Delay between setpoint re-commands (10 Hz -> 0.1 s).
            level_orientation: Keep roll/pitch fixed at the captured orientation and
                only change yaw (MV_* never tilts the gripper).
            gripper_close_threshold_m: Width below which the gripper is treated as closed.
            grasp_min_width_m: Optional empty-grasp width (meters). After a GRASP
                closes, if the measured gripper width is at/below this value the fingers
                caught nothing, so the gripper is reopened and the step is flagged
                ``grasp_empty``. ``None`` disables the check.
            grasp_open_width_m: Upper bound (meters) of the "holding" width band, just
                below the fully-open width. A measured width at/above this is treated as
                still-open / not-yet-settled rather than a hold -- used by the runner to
                tell a real grasp apart from the open width the sensor still reports for
                ~1-2 steps after an (asynchronous) close.
            gripper_settle_s: Max time (s) to wait for the physical gripper to finish
                actuating after a GRASP/RELEASE before reading its width. Grippers whose
                width sensor lags (e.g. the Franka Hand, ~1 s) would otherwise return the
                stale pre-command value -- which would make the empty-grasp check miss and
                leave the next camera frame showing the old gripper pose. Set 0 to skip
                the wait (mock / sim).
            gripper_min_settle_s: Min time (s) to wait before a "stopped moving" reading is
                trusted, so the poll cannot return the stale width before the fingers have
                even begun to move. Must be < ``gripper_settle_s``.
            gripper_poll_dt_s: Poll interval (s) while waiting for the gripper to settle.
            z_floor_m: Optional minimum EEF height (base-frame Z, meters). When set,
                the commanded setpoint is never driven below it, so downward motion
                cannot push the arm past this height.
            capture_z_floor_on_sync: If True and ``z_floor_m`` is None, the first
                :meth:`sync_from_robot` locks the floor at the robot's current
                (tabletop-contact) height.
            ensure_controller: Optional recovery hook, run (then the command retried once)
                when a setpoint command raises an error the subclass recognizes as
                recoverable via :meth:`_is_recoverable_command_error`.
            verbose: Print a one-line summary per command.
        """
        self.robot = robot
        self.move_vectors = {
            name: np.asarray(atomic_primitives[name], dtype=float) for name in MOVE_ATOMS
        }
        self.yaw_signs = {name: float(atomic_primitives[name]) for name in ROTATE_ATOMS}
        self.step_m = float(step_m)
        self.yaw_step_rad = float(yaw_step_rad)
        self.max_position_delta_m = float(max_position_delta_m)
        self.max_rotation_delta_rad = float(max_rotation_delta_rad)
        self.settle_steps = max(1, int(settle_steps))
        self.settle_dt_s = max(0.0, float(settle_dt_s))
        self.level_orientation = bool(level_orientation)
        self.gripper_close_threshold_m = float(gripper_close_threshold_m)
        self.grasp_min_width_m: Optional[float] = (
            None if grasp_min_width_m is None else float(grasp_min_width_m)
        )
        self.grasp_open_width_m = float(grasp_open_width_m)
        self.gripper_settle_s = max(0.0, float(gripper_settle_s))
        self.gripper_min_settle_s = max(0.0, float(gripper_min_settle_s))
        self.gripper_poll_dt_s = max(0.0, float(gripper_poll_dt_s))
        self.z_floor_m: Optional[float] = None if z_floor_m is None else float(z_floor_m)
        self.capture_z_floor_on_sync = bool(capture_z_floor_on_sync)
        # Optional recovery hook: called to restore the robot-side controller when a
        # setpoint command fails with an error the subclass recognizes as recoverable
        # (so the rollout/teleop self-heals instead of crashing). Wired to
        # FrankaSession.start_impedance by the Franka runners.
        self.ensure_controller: Optional[Callable[[], None]] = ensure_controller
        # Optional variable-step plugin: coarsens the per-command translation when high
        # above the table or lifting (MV_UP). ``table_height_m`` is the table-contact
        # reference it compares the EEF height against. None -> always the fixed step_m.
        self.variable_step_plugin = variable_step_plugin
        self.table_height_m: Optional[float] = (
            None if table_height_m is None else float(table_height_m)
        )
        # Dedicated MV_UP lift distance (m): when set, MV_UP uses this magnitude so a lift /
        # retreat clears the table in fewer steps. None -> MV_UP uses the normal step logic.
        self.up_step_m: Optional[float] = (
            None if up_step_m is None else max(0.0, float(up_step_m))
        )
        # The per-command safety clamp must never silently cap a step the system is configured
        # to command. Raise it to cover the variable-step coarse magnitude AND the MV_UP
        # distance, else e.g. up_step_m=0.08 is clipped by the 0.05 default and lifts only 5 cm.
        coarse_step_m = float(getattr(self.variable_step_plugin, "coarse_step_m", 0.0) or 0.0)
        self.max_position_delta_m = max(
            self.max_position_delta_m, self.step_m, coarse_step_m, self.up_step_m or 0.0
        )
        # Optional smooth-motion plugin: ramps the setpoint start->target along an eased
        # profile instead of stepping it, and (with blend on) chains aligned consecutive
        # moves at cruise speed. None -> the plain re-command settle.
        self.smooth_plugin = smooth_plugin
        # Optional rotation plugin: offers ROTATE_CW/CCW and, once the gripper has yawed,
        # rotates a wrist-judged MV_* by the accumulated yaw so the VLM can keep reasoning
        # in the wrist frame (see step()). None / disabled -> yaw compensation is off and
        # ROTATE is never commanded, so motion is unchanged.
        self.rotation_plugin = rotation_plugin
        # Motion frame for MV_*: "base" (primitives vectors as-is) or "wrist" (rotated to
        # the gripper's current heading -- see the class docstring). The rotation plugin's
        # yaw compensation IS a partial wrist frame, so composing them would rotate
        # wrist-judged moves twice; refuse the combination outright.
        self.motion_frame = str(motion_frame or "base").strip().lower()
        if self.motion_frame not in ("base", "wrist"):
            raise ValueError(f"motion_frame must be 'base' or 'wrist', got {motion_frame!r}")
        if self.motion_frame == "wrist" and bool(getattr(rotation_plugin, "enabled", False)):
            raise ValueError(
                "motion_frame='wrist' already executes MV_* in the gripper-heading frame; "
                "the rotation plugin's yaw compensation would rotate wrist-judged moves "
                "twice. Disable plugins.rotation (or use motion_frame='base')."
            )
        # One-time warning latch for a degenerate heading (tool axis ~vertical).
        self._warned_degenerate_heading = False
        self.verbose = bool(verbose)

        # Commanded setpoint (the "target" the arm is driven toward). Populated by
        # sync_from_robot() so the first command moves relative to the real pose.
        self._target_pos: Optional[np.ndarray] = None
        self._target_euler: Optional[np.ndarray] = None
        # Yaw at which the base-frame MV_* mapping is calibrated (captured on the first
        # sync). The rotation plugin measures accumulated yaw against this.
        self._reference_yaw: Optional[float] = None
        self.gripper_closed: Optional[bool] = None
        # Motion streaming (smooth plugin, blend on). A move that a caller flags
        # ``continuous`` ends at CRUISE speed with no settle dwell, so the next aligned
        # move starts from that speed and the arm flows through instead of stopping once
        # per token. _stream_dir is the unit direction it is flowing in; a move that is
        # not aligned with it (or any non-move token) flushes the stream to rest first.
        self._stream_hot = False
        self._stream_dir: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_primitives_config(
        cls,
        robot: Any,
        primitives_cfg: dict[str, Any],
        *,
        step_m: Optional[float] = None,
        yaw_step_rad: Optional[float] = None,
        **kwargs: Any,
    ) -> "RealAtomicController":
        """Build from a loaded primitives yaml mapping (atomic_primitives/step_m/yaw_step_rad).

        ``step_m`` / ``yaw_step_rad`` default to the values in the primitives yaml but
        may be overridden by the caller (e.g. ``step_m`` from the robot yaml so the
        per-step move distance is user-configurable without editing the primitives).
        """
        try:
            atomic = primitives_cfg["atomic_primitives"]
        except KeyError as exc:
            raise ValueError(
                f"primitives config missing required key: {exc}. Expected "
                "'atomic_primitives'."
            ) from exc
        resolved_step = primitives_cfg.get("step_m") if step_m is None else step_m
        resolved_yaw = (
            primitives_cfg.get("yaw_step_rad") if yaw_step_rad is None else yaw_step_rad
        )
        if resolved_step is None or resolved_yaw is None:
            raise ValueError(
                "step_m and yaw_step_rad must be defined in the primitives yaml or passed "
                "explicitly to from_primitives_config."
            )
        return cls(
            robot=robot,
            atomic_primitives=atomic,
            step_m=resolved_step,
            yaw_step_rad=resolved_yaw,
            **kwargs,
        )

    # -- state -------------------------------------------------------------
    def sync_from_robot(self) -> np.ndarray:
        """Reset the internal setpoint to the robot's measured pose + gripper state.

        Call this once after the robot-side controller is ready and before issuing the
        first atomic command, and again after any motion this controller did not command
        (e.g. a joint-space homing move), else the next command jumps relative to the
        stale setpoint. Any in-flight motion stream is definitionally over: the flags
        are cleared so the next move starts from rest.
        """
        self._stream_hot = False
        self._stream_dir = None
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        self._target_pos = pose[:3].copy()
        self._target_euler = quat_to_euler(pose[3:])
        # Lock the yaw reference once, at the orientation the MV_* mapping is calibrated for,
        # so the rotation plugin can measure accumulated yaw (a later re-sync must not move it).
        if self._reference_yaw is None:
            self._reference_yaw = float(self._target_euler[2])
        width = float(self.robot.get_gripper_position()[0])
        self.gripper_closed = width < self.gripper_close_threshold_m
        # Lock the Z safety floor at the current (tabletop-contact) height on the
        # first sync, unless an explicit floor was already provided.
        if self.capture_z_floor_on_sync and self.z_floor_m is None:
            self.z_floor_m = float(pose[2])
        if self.verbose:
            floor_txt = (
                f" z_floor={self.z_floor_m:.4f}m" if self.z_floor_m is not None else ""
            )
            print(
                f"[{self.LOG_TAG}] synced setpoint pos={np.round(self._target_pos, 4).tolist()} "
                f"euler_deg={np.round(np.rad2deg(self._target_euler), 1).tolist()} "
                f"gripper={'CLOSED' if self.gripper_closed else 'OPEN'} ({width*1000:.1f}mm)"
                f"{floor_txt}"
            )
        return pose

    def _ensure_synced(self) -> None:
        if self._target_pos is None or self._target_euler is None:
            self.sync_from_robot()

    def set_z_floor(self, z_floor_m: Optional[float] = None) -> float:
        """Lock the minimum allowed EEF height (base-frame Z, meters).

        The commanded setpoint is never allowed below this height, so the arm
        cannot be driven downward past it. If ``z_floor_m`` is ``None`` the
        robot's *current* measured height is used -- call this with the gripper
        resting on the tabletop to forbid any further descent. Returns the floor.
        """
        if z_floor_m is None:
            z_floor_m = float(np.asarray(self.robot.get_ee_pose(), dtype=float)[2])
        self.z_floor_m = float(z_floor_m)
        if self.verbose:
            print(
                f"[{self.LOG_TAG}] z-floor locked at {self.z_floor_m:.4f} m "
                "(downward motion below this height is blocked)"
            )
        return self.z_floor_m

    def clear_z_floor(self) -> None:
        """Remove the Z safety floor (allow unrestricted vertical motion)."""
        self.z_floor_m = None

    def measured_gripper_width(self) -> float:
        """Current measured gripper jaw opening (meters), read live from the robot.

        Same source the session uses for ``obs["gripper_width"]``; read AFTER a
        GRASP/RELEASE has settled it reflects the physical finger width, so callers can
        tell an empty grasp (width collapsed) from a real hold."""
        return float(self.robot.get_gripper_position()[0])

    @property
    def target_pose(self) -> np.ndarray:
        """Current commanded setpoint as a 7-D pose [x,y,z,qx,qy,qz,qw]."""
        self._ensure_synced()
        return np.concatenate([self._target_pos, euler_to_quat(self._target_euler)])

    # -- wrist/heading motion frame ------------------------------------------------
    def heading_yaw_rad(self) -> Optional[float]:
        """Yaw (rad, about base +Z) of the gripper's heading, from the SETPOINT orientation.

        The heading is the horizontal projection of :attr:`TOOL_AXIS` (EEF frame) --
        i.e. which way the gripper points across the table. Uses the commanded setpoint
        (not the measured pose) so it is noise-free; with ``level_orientation`` it only
        changes when yaw is commanded. Returns ``None`` when the projection is degenerate
        (tool pointing near-vertical, e.g. a straight-down Franka gripper) -- callers
        fall back to the base frame.
        """
        self._ensure_synced()
        tool = R.from_euler("xyz", self._target_euler).apply(np.asarray(self.TOOL_AXIS, float))
        horiz = np.hypot(float(tool[0]), float(tool[1]))
        if horiz < 0.1:  # near-vertical tool: no meaningful heading
            return None
        return float(np.arctan2(tool[1], tool[0]))

    def _to_motion_frame(
        self, delta_pos: np.ndarray, motion_frame: Optional[str] = None
    ) -> np.ndarray:
        """Resolve an MV_* delta into the effective motion frame.

        ``motion_frame`` is an optional per-command override (the view-select plugin's
        guiding-view frame); ``None`` uses the configured :attr:`motion_frame`.
        ``base``: unchanged. ``wrist``: rotate about +Z by the gripper heading, so the
        primitive +X becomes "along the heading" and +Y its left; the Z component (and
        so MV_UP/DOWN, and every move's height) is untouched. A degenerate heading falls
        back to the base frame with a one-time warning.
        """
        frame = str(motion_frame or self.motion_frame).strip().lower()
        if frame != "wrist":
            return delta_pos
        heading = self.heading_yaw_rad()
        if heading is None:
            if self.verbose and not self._warned_degenerate_heading:
                self._warned_degenerate_heading = True
                print(
                    f"[{self.LOG_TAG}] WARNING: motion_frame=wrist but the tool axis is "
                    "near-vertical (no heading); executing MV_* in the base frame."
                )
            return delta_pos
        c, s = float(np.cos(heading)), float(np.sin(heading))
        x, y, z = (float(v) for v in delta_pos)
        return np.asarray([c * x - s * y, s * x + c * y, z], dtype=float)

    # -- token -> intended motion (mirrors AtomicController.action_for_atomic) ----
    def intended_motion(
        self, token: str, step_m: Optional[float] = None
    ) -> tuple[np.ndarray, float]:
        """Return (delta_pos_m[3], yaw_rad) for a motion token, before safety clamps.

        ``step_m`` overrides the default per-step translation magnitude (used by the
        variable-step plugin for coarse moves); ``None`` uses ``self.step_m``."""
        token = token.strip().upper()
        step = self.step_m if step_m is None else float(step_m)
        if token == STOP_ATOM:
            return np.zeros(3, dtype=float), 0.0
        if token in MOVE_ATOMS:
            return self.move_vectors[token] * step, 0.0
        if token in ROTATE_ATOMS:
            return np.zeros(3, dtype=float), self.yaw_signs[token] * self.yaw_step_rad
        raise ValueError(f"{token!r} is not a motion token (MV_*/ROTATE_*/STOP)")

    def _effective_step_m(
        self, token: str, pre_pose: np.ndarray, target_in_wrist: Optional[bool] = None
    ) -> tuple[float, str]:
        """``(magnitude_m, kind)`` for one command's translation, optionally coarsened
        by the variable-step plugin based on the EEF height + token (MV_UP / high above
        table) and the VLM's wrist-visibility judgment (``target_in_wrist`` False ->
        far -> coarse). ``kind`` labels the choice for the step record / terminal
        ("up" / "coarse" / "fine"); "" when there was no choice to make (fixed step)."""
        # A dedicated MV_UP distance (lift/retreat) takes priority over the step logic.
        if self.up_step_m is not None and str(token).strip().upper() == "MV_UP":
            return self.up_step_m, "up"
        plugin = self.variable_step_plugin
        if plugin is None or not getattr(plugin, "enabled", False):
            return self.step_m, ""
        pose = np.asarray(pre_pose, dtype=float).reshape(-1)
        height = float(pose[2]) if pose.size >= 3 else None
        step = float(
            plugin.step_m_for(
                token,
                self.step_m,
                eef_height_m=height,
                table_height_m=self.table_height_m,
                target_in_wrist=target_in_wrist,
            )
        )
        return step, ("fine" if abs(step - self.step_m) < 1e-12 else "coarse")

    # -- execution ---------------------------------------------------------
    def step(
        self,
        token: str,
        target_in_wrist: Optional[bool] = None,
        continuous: bool = False,
        motion_frame: Optional[str] = None,
        step_override_m: Optional[float] = None,
    ) -> AtomicStepResult:
        """Execute one atomic token on the robot and return a record of the step.

        ``target_in_wrist`` is the controller VLM's wrist-visibility judgment forwarded by
        the runner; the variable-step plugin uses it to pick a coarse (TARGET not in wrist ->
        far) vs fine (in wrist -> close) step. ``None`` (callers without a VLM, or the plugin
        off) falls back to the height-only step logic.

        ``motion_frame`` overrides the configured motion frame for THIS command only
        ("base" / "wrist"; ``None`` -> the configured default). Forwarded by the dual
        runner under the view-select plugin, so each move executes in the frame of the
        view that guided it.

        ``continuous`` tells the smooth plugin that ANOTHER aligned move follows immediately
        (a held teleop key, the next move of an action chunk), so this one ends at cruise
        speed instead of decelerating to a stop -- the arm flows through the run. The
        caller must close the run (key released / chunk done) with :meth:`end_stream`.
        Default False -> classic rest-to-rest behaviour, unchanged.
        """
        raw = token
        token = token.strip().upper()
        self._ensure_synced()
        pre_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)

        if token == DONE_ATOM:
            self.end_stream()  # the episode is over; bring a hot stream to rest
            result = AtomicStepResult(
                token=token, kind="done", pre_pose=pre_pose,
                target_pose=self.target_pose, post_pose=pre_pose,
                gripper_closed=self.gripper_closed, done=True, note="episode done",
            )
            if self.verbose:
                print(f"[{self.LOG_TAG}] DONE")
            return result

        if token == GRASP_ATOM:
            # Bring any in-flight streamed motion to rest before the fingers act, so the
            # gripper never closes while the arm is still coasting.
            self.end_stream()
            return self._apply_gripper(close=True, pre_pose=pre_pose, token=token)
        if token == RELEASE_ATOM:
            self.end_stream()
            return self._apply_gripper(close=False, pre_pose=pre_pose, token=token)

        # Rotation plugin: return to neutral BEFORE lifting a held object. Rotation is only a
        # grasp-alignment move, so an MV_UP while the gripper is closed (holding) and still
        # yawed from that alignment un-rotates toward the reference orientation first
        # (clamped, so a large turn undoes over a step or two); only once ~neutral does MV_UP
        # actually lift. This keeps the carry/place phase in the clean un-rotated frame and
        # returns MV_* compensation to the identity. Empty gripper (e.g. RETREAT after a
        # RELEASE) or plugin off -> realign_delta_yaw is 0.0, so MV_UP lifts as usual.
        if (
            token == "MV_UP"
            and self.gripper_closed
            and self.rotation_plugin is not None
            and self._reference_yaw is not None
        ):
            accum_yaw = float(self._target_euler[2]) - float(self._reference_yaw)
            realign_yaw = self.rotation_plugin.realign_delta_yaw(accum_yaw)
            if realign_yaw != 0.0:
                return self._apply_motion(
                    np.zeros(3, dtype=float),
                    realign_yaw,
                    kind="realign",
                    pre_pose=pre_pose,
                    token=token,
                )

        if token in MOVE_ATOMS or token in ROTATE_ATOMS or token == STOP_ATOM:
            eff_step_m, step_kind = self._effective_step_m(token, pre_pose, target_in_wrist)
            if step_override_m is not None and token in MOVE_ATOMS:
                # Caller-supplied exact translation for THIS command (a scripted grid
                # move whose unit is a measured board pitch, not the generic step).
                eff_step_m, step_kind = float(step_override_m), "grid"
            delta_pos, yaw = self.intended_motion(token, step_m=eff_step_m)
            # Wrist motion frame: resolve the primitive vector along the gripper's
            # current heading (identity in the default base frame / for MV_UP/DOWN).
            if token in MOVE_ATOMS:
                delta_pos = self._to_motion_frame(delta_pos, motion_frame)
            # Rotation plugin: keep the VLM reasoning in the wrist frame. A wrist-judged MV_*
            # (target_in_wrist not False) is rotated by the accumulated gripper yaw so it
            # lands on the correct base-frame direction; agentview-judged moves (False) are
            # already base-frame and left alone. A ROTATE that would push the accumulated yaw
            # past the soft joint-travel guard is dropped to a hold. All identity when the
            # plugin is off or the gripper is still at the reference yaw.
            if self.rotation_plugin is not None and getattr(self.rotation_plugin, "enabled", False):
                accum_yaw = float(self._target_euler[2]) - float(self._reference_yaw or 0.0)
                if token in MOVE_ATOMS and target_in_wrist is not False:
                    delta_pos = self.rotation_plugin.compensate_move(delta_pos, accum_yaw)
                elif token in ROTATE_ATOMS and not self.rotation_plugin.yaw_within_limit(
                    accum_yaw + yaw
                ):
                    yaw = 0.0
            kind = "move" if token in MOVE_ATOMS else ("rotate" if token in ROTATE_ATOMS else "stop")
            result = self._apply_motion(
                delta_pos, yaw, kind=kind, pre_pose=pre_pose, token=token,
                continuous=continuous,
            )
            # Stamp which step magnitude this MOVE used (variable-step choice / the
            # dedicated MV_UP distance), for the step record and the terminal line.
            # Stamped post-hoc so per-robot _apply_motion overrides stay untouched.
            if token in MOVE_ATOMS:
                result.step_kind = step_kind
                result.step_m = float(eff_step_m)
            return result

        # Unknown token: hold position (do not move the arm on garbage input).
        if self.verbose:
            print(f"[{self.LOG_TAG}] WARNING: unknown token {raw!r}; holding position")
        self._command_setpoint()
        post_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        return AtomicStepResult(
            token=token, kind="unknown", pre_pose=pre_pose, target_pose=self.target_pose,
            post_pose=post_pose, gripper_closed=self.gripper_closed,
            note="unknown token -> hold",
        )

    def _apply_motion(
        self,
        delta_pos: np.ndarray,
        yaw: float,
        kind: str,
        pre_pose: np.ndarray,
        token: str,
        continuous: bool = False,
    ) -> AtomicStepResult:
        # Safety clamps (per command).
        clamped_pos = np.clip(delta_pos, -self.max_position_delta_m, self.max_position_delta_m)
        clamped_yaw = float(np.clip(yaw, -self.max_rotation_delta_rad, self.max_rotation_delta_rad))

        # Capture the pre-move setpoint so the smooth plugin can interpolate start->target.
        start_pos = self._target_pos.copy()
        start_euler = self._target_euler.copy()
        # Integrate into the setpoint. MV_* never tilts; only yaw changes.
        self._target_pos = self._target_pos + clamped_pos
        # Safety floor: never command the EEF below the locked height (captured
        # with the gripper on the tabletop). Limits downward motion only.
        floor_note = ""
        if self.z_floor_m is not None and self._target_pos[2] < self.z_floor_m:
            blocked = float(self.z_floor_m - self._target_pos[2])
            self._target_pos[2] = self.z_floor_m
            floor_note = (
                f"z-floor: blocked {blocked:.4f}m of descent (floor={self.z_floor_m:.4f}m)"
            )
        self._target_euler = self._target_euler.copy()
        self._target_euler[2] += clamped_yaw  # yaw about base +Z

        # A realign step un-rotates a held object back to neutral instead of lifting; note it
        # so the MV_UP that produced no lift is explainable in the logs.
        note = floor_note
        if kind == "realign":
            realign_note = "realign to neutral before lift"
            note = f"{floor_note}; {realign_note}" if floor_note else realign_note

        self._drive_to_target(start_pos, start_euler, continuous=continuous)
        post_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)

        if self.verbose:
            msg = (
                f"[{self.LOG_TAG}] {token:<9} d_pos(m)={np.round(clamped_pos, 4).tolist()} "
                f"d_yaw(deg)={np.rad2deg(clamped_yaw):.1f} -> "
                f"target_pos={np.round(self._target_pos, 4).tolist()}"
            )
            if note:
                msg += f"  [{note}]"
            print(msg)
        return AtomicStepResult(
            token=token, kind=kind, intended_delta_m=clamped_pos, intended_yaw_rad=clamped_yaw,
            pre_pose=pre_pose, target_pose=self.target_pose, post_pose=post_pose,
            gripper_closed=self.gripper_closed, note=note,
        )

    def _apply_gripper(self, close: bool, pre_pose: np.ndarray, token: str) -> AtomicStepResult:
        changed = self.gripper_closed is None or bool(self.gripper_closed) != close
        settled_width: Optional[float] = None
        if changed:
            pre_width = float(self.robot.get_gripper_position()[0])
            # Robot interface contract: control_gripper(True=close, False=open).
            self.robot.control_gripper(bool(close))
            self.gripper_closed = bool(close)
            # A physical gripper may actuate ASYNCHRONOUSLY with a lagging width sensor
            # (the Franka Hand lags ~1 s): control_gripper() returns immediately while
            # the fingers are still moving. Block until they actually stop before reading
            # the width, otherwise (a) the empty-grasp check below samples the stale
            # pre-close width and never fires, and (b) the next observation/camera frame
            # still shows the old gripper pose (which confuses the controller VLM). For a
            # close, wait until the width is below the open band (clearly closed) so a
            # stale OPEN read is not accepted as the settled width.
            settled_width = self._await_gripper_settled(
                decisive_max_m=self.grasp_open_width_m if close else None
            )
            # Per-robot hook to catch a gripper command that never actuated (default
            # no-op, so the Franka path is unchanged). Useful where a command can be
            # silently dropped and the width would otherwise be mistaken for a settled
            # grasp -- see PiperAtomicController.
            self._on_gripper_settled(close, pre_width, settled_width)

        # Empty-grasp check: a close that ends at/below the empty-grasp width caught
        # nothing (gripper too high or off-centre), so reopen and flag the step. The
        # real runner's recovery plugin then rewinds to the grasp stage; the VLM chooses
        # the next corrective move from the new observation.
        grasp_empty = False
        empty_note = ""
        if close and changed and self.grasp_min_width_m is not None:
            width = (
                settled_width
                if settled_width is not None
                else float(self.robot.get_gripper_position()[0])
            )
            if width <= self.grasp_min_width_m:
                self.robot.control_gripper(False)
                self.gripper_closed = False
                self._await_gripper_settled()
                grasp_empty = True
                empty_note = (
                    f"empty-close: width {width:.4f}m <= "
                    f"{self.grasp_min_width_m:.4f}m -> reopened"
                )

        post_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        if self.verbose:
            action = "CLOSE" if close else "OPEN"
            state = "(changed)" if changed else "(already in state)"
            print(f"[{self.LOG_TAG}] {token:<9} gripper -> {action} {state}")
            if grasp_empty:
                print(f"[{self.LOG_TAG}] {token:<9} {empty_note}")
        note = ("gripper " + ("close" if close else "open")) + ("" if changed else " (noop)")
        if empty_note:
            note = empty_note
        return AtomicStepResult(
            token=token, kind="gripper", pre_pose=pre_pose, target_pose=self.target_pose,
            post_pose=post_pose, gripper_closed=self.gripper_closed,
            grasp_empty=grasp_empty, note=note,
        )

    def hold(self) -> AtomicStepResult:
        """Re-command the current setpoint (STOP)."""
        return self.step(STOP_ATOM)

    def open_gripper(self) -> AtomicStepResult:
        return self.step(RELEASE_ATOM)

    def close_gripper(self) -> AtomicStepResult:
        return self.step(GRASP_ATOM)

    # -- low-level ---------------------------------------------------------
    def _is_recoverable_command_error(self, exc: Exception) -> bool:
        """True when a failed setpoint command can be fixed by running the
        ``ensure_controller`` hook and retrying (per-robot subclasses override this;
        e.g. Franka matches polymetis "no controller running" errors)."""
        return False

    def _on_gripper_settled(
        self, close: bool, pre_width: float, settled_width: float
    ) -> None:
        """Per-robot hook run after a changed gripper command settles. Default no-op
        (Franka path unchanged); a subclass can flag a command that never actuated."""


    def _command_setpoint(self) -> None:
        """Drive the arm toward the current target setpoint for ``settle_steps`` ticks."""
        self._settle(re_command=True)

    def end_stream(self) -> None:
        """Bring a chained (``continuous``) motion to rest at the current setpoint.

        A continuous move deliberately ends at cruise speed with no settle dwell, so the
        arm is still converging when it returns. Callers signal the end of a run of moves
        (teleop key released, action chunk finished) with this; it re-commands and settles
        so the arm comes to rest cleanly. A no-op when nothing is streaming.
        """
        if not self._stream_hot:
            return
        self._stream_hot = False
        self._stream_dir = None
        self._settle(re_command=True)

    def _drive_to_target(
        self,
        start_pos: np.ndarray,
        start_euler: np.ndarray,
        continuous: bool = False,
    ) -> None:
        """Command the integrated target.

        With the smooth plugin enabled AND an actual displacement, walk the setpoint
        start->target along the plugin's eased profile (gentle accel/decel) and settle at
        the target; otherwise (or for a no-op hold) use the plain re-command settle.

        When the plugin blends and the caller flags ``continuous`` (another aligned move is
        coming: a held teleop key, the next move of an action chunk), the profile ENDS at
        cruise speed and the settle dwell is skipped, and a following aligned move STARTS
        at that speed -- so the concatenated setpoint path is velocity-continuous and the
        arm flows through the run instead of stopping once per token. Anything that is not
        an aligned continuation (a different direction, a rotation, a gripper action, the
        end of the run) first flushes the stream to rest via :meth:`end_stream`.
        """
        # "Moved" if the position OR any orientation axis changed. Checking the full
        # euler delta (not just yaw) is what lets a pitch-only change take the smooth
        # ramp; it is a no-op for arms that only ever change yaw (e.g. Franka), whose
        # roll/pitch deltas stay 0.
        dpos = self._target_pos - start_pos
        deuler = self._target_euler - start_euler
        dist = float(np.linalg.norm(dpos))
        moved = dist > 1e-5 or float(np.linalg.norm(deuler)) > 1e-5

        smooth = self.smooth_plugin is not None and getattr(self.smooth_plugin, "enabled", False)
        if not (smooth and moved):
            self.end_stream()  # never leave a stream hanging behind a hold / plain command
            self._command_setpoint()
            return

        blend = bool(getattr(self.smooth_plugin, "blend", False))
        cruise = float(getattr(self.smooth_plugin, "cruise", 1.0))
        # Only pure translations chain: a rotation mid-stream would swing the setpoint's
        # direction, so it starts from rest.
        pure_translation = dist > 1e-9 and float(np.linalg.norm(deuler)) <= 1e-9
        unit = (dpos / dist) if dist > 1e-9 else None

        # Start speed: only carry the cruise if we are already streaming IN THIS
        # DIRECTION. Otherwise flush the in-flight motion to rest first.
        v0 = 0.0
        if blend and self._stream_hot and pure_translation and self._stream_dir is not None:
            if float(np.dot(unit, self._stream_dir)) >= 0.9:
                v0 = cruise
        if v0 == 0.0:
            self.end_stream()

        v1 = cruise if (blend and continuous and pure_translation) else 0.0

        fractions, dt_s = self.smooth_plugin.plan(dist, v0, v1)
        # An eased ramp's slow end still asks for advances below the hardware's command
        # resolution. Re-issuing a point-to-point target the arm cannot act on just makes
        # it buzz, so hold the last setpoint through those and only command once the
        # advance is actionable -- KEEPING the schedule (we still sleep), so the move takes
        # exactly as long as planned. The final waypoint is always commanded.
        min_adv = float(getattr(self.smooth_plugin, "min_waypoint_m", 0.0) or 0.0)
        last_frac = 0.0
        n = len(fractions)
        for i, frac in enumerate(fractions):
            actionable = (
                min_adv <= 0.0
                or i == n - 1
                # The floor is a TRANSLATION-resolution guard; a (near-)pure rotation
                # advances no distance per waypoint, so its ramp must not be swallowed.
                or dist <= min_adv
                or (frac - last_frac) * dist >= min_adv
            )
            if actionable:
                pos = start_pos + dpos * frac
                euler = start_euler + deuler * frac
                self._update_desired_ee_pose(np.concatenate([pos, euler_to_quat(euler)]))
                last_frac = frac
            if dt_s > 0.0:
                time.sleep(dt_s)

        if v1 > 0.0:
            # Flow into the next move: no settle dwell, remember what we are riding.
            self._stream_hot = True
            self._stream_dir = unit
        else:
            # Final convergence hold at the exact target.
            self._stream_hot = False
            self._stream_dir = None
            self._settle(re_command=True)

    def _settle(self, re_command: bool) -> None:
        pose = self.target_pose if re_command else None
        for _ in range(self.settle_steps):
            if re_command and pose is not None:
                self._update_desired_ee_pose(pose)
            if self.settle_dt_s > 0.0:
                time.sleep(self.settle_dt_s)

    def _update_desired_ee_pose(self, pose: np.ndarray) -> None:
        """Command one setpoint, self-healing if the robot-side controller was lost.

        Some robot stacks can lose the controller that consumes setpoints between
        commands -- e.g. on the Franka a joint move (``reset_to_home`` / ``go_home``)
        preempts the Cartesian-impedance controller, or a server-side reflex terminates
        it; the next ``update_desired_ee_pose`` then raises. If an ``ensure_controller``
        hook is set and the subclass recognizes the error as recoverable, we run the hook
        and retry the SAME absolute setpoint once (re-sending the target is safe: the arm
        just resumes toward it from wherever it is). A second failure propagates.
        """
        try:
            self.robot.update_desired_ee_pose(pose)
            return
        except Exception as exc:  # noqa: BLE001
            if self.ensure_controller is None or not self._is_recoverable_command_error(exc):
                raise
            if self.verbose:
                print(
                    f"[{self.LOG_TAG}] setpoint command failed with a recoverable error; "
                    "running the recovery hook and retrying ..."
                )
            self.ensure_controller()
        self.robot.update_desired_ee_pose(pose)  # retry once; a repeat failure propagates

    def _await_gripper_settled(self, decisive_max_m: Optional[float] = None) -> float:
        """Block until the (asynchronous) physical gripper stops moving; return its width.

        Polls ``get_gripper_position()`` until the fingers have ACTUALLY started actuating
        (the width moved more than ``GRIPPER_MOTION_EPS_M`` from the pre-command width) AND
        the width has then held within ``GRIPPER_STABLE_EPS_M`` for a SUSTAINED window
        (``GRIPPER_SUSTAIN_S``), after at least ``gripper_min_settle_s`` -- or until
        ``gripper_settle_s`` elapses.

        Why motion + sustained stability: right after an async close the gripper has not
        begun to move, so the first reads are equal and look "settled" -- returning that
        stale pre-close width made the in-step empty-grasp check miss (it saw the open
        width). A lagging width sensor also pauses on brief plateaus on its way to ~0;
        requiring stability to hold for ``GRIPPER_SUSTAIN_S`` stops a momentary plateau
        from triggering an early return.

        ``decisive_max_m``: for a CLOSE, the settle will not accept a "stable" reading
        until the width is at/below this (the gripper has clearly closed, below the
        open band), so a stale OPEN read is never mistaken for the settled close width.
        ``None`` (e.g. for an open/RELEASE) disables this gate.

        With ``gripper_settle_s == 0`` it returns the first read immediately (mock / sim,
        where the width updates synchronously).
        """
        start = time.monotonic()
        w0 = float(self.robot.get_gripper_position()[0])
        last = w0
        moved = False
        stable_since: Optional[float] = None
        while True:
            if time.monotonic() - start >= self.gripper_settle_s:
                return last
            if self.gripper_poll_dt_s > 0.0:
                time.sleep(self.gripper_poll_dt_s)
            now = time.monotonic()
            cur = float(self.robot.get_gripper_position()[0])
            if abs(cur - w0) > self.GRIPPER_MOTION_EPS_M:
                moved = True
            if abs(cur - last) <= self.GRIPPER_STABLE_EPS_M:
                if stable_since is None:
                    stable_since = now
            else:
                stable_since = None
            last = cur
            sustained = stable_since is not None and (now - stable_since) >= self.GRIPPER_SUSTAIN_S
            decisive = decisive_max_m is None or cur <= float(decisive_max_m)
            if moved and sustained and decisive and (now - start) >= self.gripper_min_settle_s:
                return cur
