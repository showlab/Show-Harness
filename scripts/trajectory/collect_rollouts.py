#!/usr/bin/env python3
"""Keyboard teleoperation + rollout recorder for the real Franka.

Thin Franka wiring around the shared teleop core (core/teleop/single.py): builds the
FrankaSession + FrankaAtomicController and supplies the Franka-specific homing
hook (joint move over ZeroRPC + impedance restart). The AgileX Piper counterpart
is scripts/trajectory/collect_rollouts_piper.py.

Controls (unified with DAGGER and the web teleop; source of truth:
core/teleop/dual.py:build_single_keymaps)
--------
W / S             MV_FWD / MV_BACK        (W = AWAY from the robot base)
A / D             MV_LEFT / MV_RIGHT
R / F             MV_UP / MV_DOWN
Up / Down         MV_UP / MV_DOWN         (arrow aliases)
Left / Right      MV_LEFT / MV_RIGHT      (arrow aliases)
Z / X             ROTATE_CCW / ROTATE_CW
Space / LShift    toggle gripper: GRASP (1st press) <-> RELEASE (2nd press)
P                 start / stop recording (each recording = one numbered rollout)
Q / Esc           quit

Each recording is saved under ``<save_path>/rollout_NNN/``:
    agentview/0000.png, 0001.png, ...   external-camera frames (one per step)
    wrist/0000.png, ...                 wrist-camera frames
    actions.jsonl                       per-step token + ee_pose + gripper
    metadata.json                       rollout summary (tokens, counts, config)
    visualization.mp4                   both views + action overlay per step

Examples
--------
# Real robot + real cameras (uses the verified NUC IP / serials by default):

python scripts/trajectory/collect_rollouts.py rollouts/franka


Notes
-----
* Requires a display (``$DISPLAY``) and ``pygame`` for the live window/keyboard.
* Movement keys repeat while held (rate set by ``--move-interval``).
* Frames saved are the 256x256 RGB observations the policy/VLM consume.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# File is scripts/trajectory/collect_rollouts.py, so repo root is two levels above.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreters.franka_atomic_controller import (
    EMPTY_GRASP_WIDTH_M,
    TABLE_CONTACT_Z_M,
    FrankaAtomicController,
)
from core.config import load_yaml
from core.franka.franka_session import FrankaSession, FrankaSessionConfig
from core.teleop.single import RolloutCollector, RolloutRecorder

# Home joint configuration the arm is reset to after a recording stops. Mirrors
# franka_server/go_home_client.py (move_to_joint_positions over the same ZeroRPC server),
# so pressing P to stop auto-homes the arm instead of having to run that script by hand.
HOME_JOINTS = [0.0, -0.5058451, 0.0, -2.6068573, 0.0, 2.0711833, 0.86116207]
DEFAULT_RESET_TIME_S = 3.0  # min-jerk trajectory duration, matching go_home_client.py


def make_home_fn(session: FrankaSession, controller: FrankaAtomicController, reset_time_s: float):
    """Franka homing hook: joint move to HOME_JOINTS, then restart impedance.

    ``move_to_joint_positions`` preempts the running Cartesian-impedance controller
    (same as go_home_client.py over the shared ZeroRPC server), so afterwards the
    impedance controller is restarted. The shared collector then re-syncs the atomic
    controller's setpoint to the new (home) pose.
    """

    def home() -> None:
        print(f"[reset] homing the arm (move_to_joint_positions, {reset_time_s:.1f}s) ...")
        started = time.monotonic()
        controller.robot.move_to_joint_positions(np.asarray(HOME_JOINTS, dtype=float), reset_time_s)
        # Wait out the trajectory whether or not the call blocked, so impedance is not
        # restarted mid-move.
        remaining = (reset_time_s + 0.5) - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        if session.config.start_impedance:
            # Robust restart (terminate-then-start) -- the joint move may have left the
            # joint controller running or none at all.
            session.start_impedance()

    return home


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keyboard teleoperation + rollout recorder for the real Franka.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("save_path", help="Root directory; rollouts are auto-numbered inside it.")
    p.add_argument("--rollout-prefix", default="rollout_", help="Rollout folder prefix.")

    # Hardware.
    p.add_argument("--mock-robot", action="store_true", help="Use a simulated robot (no NUC).")
    p.add_argument("--mock-cameras", action="store_true", help="Use random mock cameras.")
    p.add_argument("--nuc-ip", default=None, help="Franka NUC IP (default: robot.nuc_ip from --robot-config).")
    p.add_argument("--nuc-port", type=int, default=4242, help="Franka NUC ZeroRPC port.")
    p.add_argument("--external-serial", default=None,
                   help="External RealSense serial (default: robot.external_camera_serial from --robot-config).")
    p.add_argument("--wrist-serial", default=None,
                   help="Wrist RealSense serial (default: robot.wrist_camera_serial from --robot-config).")
    p.add_argument("--no-impedance", action="store_true", help="Do not start the impedance controller.")
    p.add_argument(
        "--no-reset-on-stop",
        action="store_true",
        help="Do not auto-home the arm when a recording stops (P). By default, stopping "
        "moves the arm to the home joint configuration, ready for the next demo.",
    )
    p.add_argument(
        "--reset-time-s",
        type=float,
        default=DEFAULT_RESET_TIME_S,
        help=f"Seconds for the home move (default {DEFAULT_RESET_TIME_S}).",
    )

    # Motion.
    p.add_argument("--primitives", default=str(ROOT / "configs" / "primitives_franka.yaml"))
    p.add_argument(
        "--robot-config",
        default=str(ROOT / "configs" / "robot_franka.yaml"),
        help="robot_franka.yaml to read the empty-grasp width (empty_width_m) from.",
    )
    p.add_argument(
        "--empty-width-m",
        type=float,
        default=None,
        help="Empty-grasp width threshold (m): a GRASP settling at/below this auto-reopens. "
        "Overrides empty_width_m from --robot-config (which falls back to 0.005 m).",
    )
    p.add_argument("--step-m", type=float, default=None, help="Override primitives step_m (meters).")
    p.add_argument("--yaw-step-rad", type=float, default=None, help="Override primitives yaw_step_rad.")
    p.add_argument("--settle-steps", type=int, default=2, help="Setpoint re-commands per token (default 2).")
    p.add_argument("--settle-dt-s", type=float, default=0.02, help="Delay between re-commands (default 0.02s).")
    p.add_argument("--move-interval", type=float, default=0.12, help="Min seconds between repeats while a key is held.")
    p.add_argument(
        "--z-floor-m",
        type=float,
        default=TABLE_CONTACT_Z_M,
        help="Minimum EEF height (m); MV_DOWN is blocked below it so the arm cannot be "
        f"driven into the table. Default: {TABLE_CONTACT_Z_M} m (calibrated table contact), "
        "the same floor used by scripts/run_real.py rollouts.",
    )
    p.add_argument(
        "--capture-z-floor",
        action="store_true",
        help="Lock the Z floor at the start (tabletop-contact) height instead of the "
        "calibrated default. Begin with the gripper resting on the table.",
    )
    p.add_argument(
        "--no-z-floor",
        action="store_true",
        help="Disable the Z safety floor entirely (NOT recommended on hardware).",
    )

    # Display / video.
    p.add_argument("--display-scale", type=int, default=2, help="Upscale factor for the live views.")
    p.add_argument("--target-fps", type=int, default=30, help="UI refresh rate cap.")
    p.add_argument("--video-fps", type=float, default=10.0, help="FPS of the saved visualization video.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    primitives_cfg = load_yaml(args.primitives)
    if args.step_m is not None:
        primitives_cfg["step_m"] = args.step_m
    if args.yaw_step_rad is not None:
        primitives_cfg["yaw_step_rad"] = args.yaw_step_rad

    # Empty-grasp width: a GRASP that settles at/below this auto-reopens (the controller's
    # empty-close rule in _apply_gripper). Resolve from --empty-width-m, else robot_franka.yaml's
    # empty_width_m, else the calibrated default -- so editing empty_width_m in robot_franka.yaml
    # actually takes effect here. It must sit ABOVE the empty-close floor (~0.0002 m) yet
    # BELOW the thinnest object you grasp, or thin grasps get wrongly reopened.
    if args.empty_width_m is not None:
        empty_width_m = float(args.empty_width_m)
    else:
        robot_cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
        rb = robot_cfg.get("robot", {}) or {}
        if args.external_serial is None:
            args.external_serial = rb.get("external_camera_serial")
        if args.wrist_serial is None:
            args.wrist_serial = rb.get("wrist_camera_serial")
        if args.nuc_ip is None:
            args.nuc_ip = rb.get("nuc_ip")
        if not args.nuc_ip and not args.mock_robot:
            raise SystemExit(
                "[collect] robot.nuc_ip is not configured: copy "
                "configs/site/franka.yaml.example to configs/site/franka.yaml "
                "(or pass --nuc-ip)."
            )
        if not args.nuc_ip:
            args.nuc_ip = "127.0.0.1"  # mock robot: never dialed
        empty_width_m = float(robot_cfg.get("empty_width_m", EMPTY_GRASP_WIDTH_M))
    print(
        f"[collect] empty-grasp width threshold = {empty_width_m:.4f} m "
        "(a GRASP settling at/below this auto-reopens)"
    )

    session_cfg = FrankaSessionConfig(
        nuc_ip=args.nuc_ip,
        nuc_port=args.nuc_port,
        use_mock_robot=args.mock_robot,
        start_impedance=not args.no_impedance,
        connect_cameras=True,  # live dual-view is the whole point
        use_mock_cameras=args.mock_cameras,
        external_camera_serial=args.external_serial,
        wrist_camera_serial=args.wrist_serial,
        verbose=True,
    )

    recorder = RolloutRecorder(args.save_path, prefix=args.rollout_prefix, video_fps=args.video_fps)
    recorder.session_meta = {
        "nuc_ip": args.nuc_ip,
        "step_m": primitives_cfg["step_m"],
        "yaw_step_rad": primitives_cfg["yaw_step_rad"],
        "external_serial": args.external_serial,
        "wrist_serial": args.wrist_serial,
        "mock_robot": args.mock_robot,
        "mock_cameras": args.mock_cameras,
        "grasp_min_width_m": empty_width_m,
    }
    print(f"[collect] saving rollouts under: {Path(args.save_path).resolve()}")

    session = FrankaSession(session_cfg)
    exit_code = 0
    try:
        session.connect()
        # Z floor resolution (same precedence as scripts/run_real.py): off / capture-at-start /
        # explicit fixed height (the calibrated table contact by default).
        if args.no_z_floor:
            z_floor_m, capture_floor = None, False
        elif args.capture_z_floor:
            z_floor_m, capture_floor = None, True
        else:
            z_floor_m, capture_floor = float(args.z_floor_m), False
        controller = FrankaAtomicController.from_primitives_config(
            session.robot,
            primitives_cfg,
            settle_steps=args.settle_steps,
            settle_dt_s=args.settle_dt_s,
            # Same empty-grasp rule as rollouts: a GRASP that catches nothing auto-reopens,
            # so teleop demos never record a "closed on nothing" hold. Threshold resolved
            # from robot_franka.yaml's empty_width_m (or --empty-width-m), not a hardcoded constant.
            grasp_min_width_m=empty_width_m,
            z_floor_m=z_floor_m,
            capture_z_floor_on_sync=capture_floor,
            # Self-heal if a setpoint command finds no controller running (e.g. right after
            # the home move preempts impedance): restart impedance and retry.
            ensure_controller=(None if args.no_impedance else session.start_impedance),
            verbose=False,
        )
        controller.sync_from_robot()
        if controller.z_floor_m is not None:
            print(
                f"[collect] z-floor (min EEF height) = {controller.z_floor_m:.4f} m; "
                "downward motion below this height is blocked"
            )
        recorder.session_meta["z_floor_m"] = controller.z_floor_m

        collector = RolloutCollector(
            session,
            controller,
            recorder,
            move_interval=args.move_interval,
            display_scale=args.display_scale,
            target_fps=args.target_fps,
            home_fn=(
                None
                if args.no_reset_on_stop
                else make_home_fn(session, controller, args.reset_time_s)
            ),
        )
        collector.run()
        print("[collect] done.")
    except KeyboardInterrupt:
        print("\n[collect] interrupted by user")
        if recorder.active:
            recorder.stop()
    except Exception as exc:  # noqa: BLE001 - surface the failure clearly
        print(f"[collect] ERROR: {exc}")
        exit_code = 1
    finally:
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
