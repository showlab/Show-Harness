#!/usr/bin/env python3
"""Dual-arm keyboard teleop + rollout recording for the AgileX Piper rig.

The Piper rig is DUAL-ARM: this drives BOTH arms from one keyboard/window and records
demonstrations in one of two storage modes (see core/teleop/dual.py for the layouts):

    Mode A (independent):  <save>/left/rollout_NNN + <save>/right/rollout_NNN --
                           each tree identical to a single-arm dataset; a step is
                           stored only for the arm that acted.
    Mode B (synchronous):  <save>/rollout_NNN with ONE record per timestamp holding
                           both arms' tokens (inactive arm = STILL) + front and both
                           wrist views.  DEFAULT.

Keys: LEFT arm W/A/S/D + Q/E (up/down) + LShift (gripper) + Z/X (rotate);
RIGHT arm I/J/K/L + U/O (up/down) + RShift (gripper) + N/M (rotate);
P start/stop recording; Esc quit (Q is left MV_UP, not quit).

All settings come from the single unified configs/robot_piper.yaml (shared keys +
per-arm `arms.left` / `arms.right`). Both arms auto-move to their BEGIN poses on start
and BOTH reset there SIMULTANEOUSLY whenever a recording stops (P).

Prerequisites: cameras (scripts/piper/run_cameras.sh) and BOTH arm nodes in mode 1
(scripts/piper/run_arm.sh).

    python scripts/trajectory/collect_rollouts_piper.py rollouts/dual
    python scripts/trajectory/collect_rollouts_piper.py rollouts/dual --mode A
    # no hardware: --mock-robots --mock-cameras
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpreters.piper_atomic_controller import (
    PIPER_EMPTY_GRASP_WIDTH_M,
    PiperAtomicController,
)
from core.config import load_yaml
from core.piper.config import SIDES, both_arm_configs
from core.piper.dual_session import DualPiperSession, DualPiperSessionConfig
from core.piper.poses import go_begin_dual
from core.teleop.single import RolloutRecorder
from core.teleop.dual import DualRolloutCollector, DualRolloutRecorder
from plugins.config import PluginsConfig
from plugins.smooth import SmoothPlugin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-Piper keyboard teleop + rollout recorder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("save_path", nargs="?", default=str(REPO_ROOT / "rollouts" / "dual"))
    p.add_argument(
        "--mode",
        choices=["A", "B"],
        default="B",
        help="A = independent per-arm datasets; B = one synchronized record per "
        "timestamp with STILL for the inactive arm.",
    )
    p.add_argument(
        "--robot-config",
        default=str(REPO_ROOT / "configs" / "robot_piper.yaml"),
        help="The unified dual-arm Piper config (shared keys + arms.left / arms.right).",
    )
    p.add_argument(
        "--primitives",
        default=str(REPO_ROOT / "configs" / "primitives_piper.yaml"),
        help="Atomic primitives, shared by both arms (their base frames are parallel).",
    )
    p.add_argument("--step-m", type=float, default=None, help="Override primitives step_m (meters).")
    p.add_argument("--rollout-prefix", default="rollout_")
    p.add_argument("--reset-time-s", type=float, default=3.0, help="Begin-pose move duration (s).")
    p.add_argument("--no-reset-on-stop", action="store_true", help="Do not auto-reset when a recording stops.")
    p.add_argument("--mock-robots", action="store_true", help="Simulated arms (no ROS/CAN).")
    p.add_argument("--mock-cameras", action="store_true", help="Random mock camera frames.")
    p.add_argument("--settle-steps", type=int, default=2, help="Setpoint re-commands per token.")
    p.add_argument("--settle-dt-s", type=float, default=0.02, help="Delay between re-commands (s).")
    p.add_argument("--move-interval", type=float, default=0.12, help="Min seconds between repeats while a key is held.")
    p.add_argument(
        "--no-sync-steps",
        action="store_true",
        help="Mode B: take a step on ANY single keypress (the idle arm is auto-filled "
        "with STILL). Default: a step waits until BOTH arms have chosen -- an idle arm "
        "confirms with its STILL key (Alt / Enter) -- so records are not flooded with STILL.",
    )
    p.add_argument(
        "--no-z-floor",
        action="store_true",
        help="Disable BOTH Z safety floors (NOT recommended: stiff MOVE P presses into the table).",
    )
    p.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smooth motion (step the setpoint straight to each target). Smoothing "
        "is otherwise driven by plugins.smooth in the robot config.",
    )
    p.add_argument("--display-scale", type=int, default=2, help="Upscale factor for the live views.")
    p.add_argument("--target-fps", type=int, default=30, help="UI refresh rate cap.")
    p.add_argument("--video-fps", type=float, default=10.0, help="FPS of the saved visualization video.")
    return p.parse_args()


def _arm_settings(cfg: dict, args: argparse.Namespace, side: str) -> dict:
    """Resolve one arm's knobs from its FLATTENED view of the unified config."""
    rb = cfg.get("robot", {}) or {}
    if args.no_z_floor:
        z_floor: Optional[float] = None
        print(f"[collect-dual] WARNING: {side} Z floor disabled -- MOVE P is stiff; mind the table.")
    elif cfg.get("enable_z_floor", True):
        raw = cfg.get("z_floor_m")
        if raw is None:
            raise SystemExit(
                f"[collect-dual] {side}: enable_z_floor is on but arms.{side}.z_floor_m is unset. "
                f"Calibrate it first: scripts/piper/capture_z_floor.sh --arm {side} --write "
                "(or pass --no-z-floor)."
            )
        z_floor = float(raw)
    else:
        z_floor = None
    return {
        "wrist_topic": str(rb.get("wrist_camera_topic", f"/camera_{side[0]}/color/image_raw")),
        "front_topic": str(rb.get("front_camera_topic", "/camera_f/color/image_raw")),
        "open_width_m": float(rb.get("open_width_m", 0.07)),
        "gripper_settle_s": float(rb.get("gripper_settle_s", 1.5)),
        "gripper_min_settle_s": float(rb.get("gripper_min_settle_s", 0.3)),
        "empty_width_m": float(cfg.get("empty_width_m", PIPER_EMPTY_GRASP_WIDTH_M)),
        "grasp_open_width_m": float(cfg.get("open_width_m", 0.055)),
        "z_floor_m": z_floor,
        "begin_joints": cfg.get("begin_joints"),
        "move_to_begin_on_init": bool(cfg.get("move_to_begin_on_init", True)),
    }


def main() -> int:
    args = parse_args()

    primitives_cfg = load_yaml(args.primitives)
    if args.step_m is not None:
        primitives_cfg["step_m"] = args.step_m

    # ONE config file -> a flattened view per arm (shared keys + that arm's block).
    unified = load_yaml(args.robot_config)
    arm_cfgs = both_arm_configs(unified)
    settings = {side: _arm_settings(arm_cfgs[side], args, side) for side in SIDES}

    # Smooth motion, from the SAME plugins.smooth config the autonomous rollout uses: ramp
    # each move's setpoint and chain a held key's repeats at cruise speed, so teleop does
    # not step-and-stop the arm once per token. --no-smooth forces it off.
    smooth_on = (not args.no_smooth) and PluginsConfig.from_config(unified).enabled("smooth", default=False)
    smooth_plugin = SmoothPlugin(
        enabled=smooth_on,
        substeps=int(unified.get("smooth_substeps", 20)),
        dt_s=float(unified.get("smooth_dt_s", 0.05)),
        min_waypoint_m=float(unified.get("smooth_min_waypoint_m", 0.0)),
        blend=bool(unified.get("smooth_blend", True)),
    )
    print(
        f"[collect-dual] smooth motion: {'ON' if smooth_on else 'OFF'}"
        + (
            f" (ramp {smooth_plugin.duration_s:.2f}s/move, >={smooth_plugin.min_waypoint_m * 1000:.0f}mm per "
            f"waypoint, blend={'on' if smooth_plugin.blend else 'off'})"
            if smooth_on
            else ""
        )
    )

    session_cfg = DualPiperSessionConfig(
        use_mock_robots=args.mock_robots,
        use_mock_cameras=args.mock_cameras,
        open_width_left_m=settings["left"]["open_width_m"],
        open_width_right_m=settings["right"]["open_width_m"],
        front_camera_topic=settings["left"]["front_topic"],  # shared
        wrist_left_camera_topic=settings["left"]["wrist_topic"],
        wrist_right_camera_topic=settings["right"]["wrist_topic"],
        verbose=True,
    )

    shared_meta = {
        "robot": "piper-dual",
        "mode": args.mode,
        "step_m": primitives_cfg["step_m"],
        "yaw_step_rad": primitives_cfg["yaw_step_rad"],
        "front_topic": settings["left"]["front_topic"],
        "wrist_left_topic": settings["left"]["wrist_topic"],
        "wrist_right_topic": settings["right"]["wrist_topic"],
        "mock_robots": args.mock_robots,
        "mock_cameras": args.mock_cameras,
    }
    save_root = Path(args.save_path)
    if args.mode == "B":
        recorders: object = DualRolloutRecorder(
            save_root, prefix=args.rollout_prefix, video_fps=args.video_fps
        )
        recorders.session_meta = dict(shared_meta)
        print(f"[collect-dual] mode B (synchronized + STILL) -> {save_root.resolve()}")
    else:
        recorders = {
            side: RolloutRecorder(
                save_root / side, prefix=args.rollout_prefix, video_fps=args.video_fps
            )
            for side in SIDES
        }
        for side in SIDES:
            recorders[side].session_meta = {**shared_meta, "arm": side}
        print(f"[collect-dual] mode A (independent per-arm) -> {save_root.resolve()}/{{left,right}}")

    session = DualPiperSession(session_cfg)
    exit_code = 0
    try:
        session.connect()

        controllers = {}
        for side in SIDES:
            s = settings[side]
            controllers[side] = PiperAtomicController.from_primitives_config(
                session.robots[side],
                primitives_cfg,
                settle_steps=args.settle_steps,
                settle_dt_s=args.settle_dt_s,
                # A GRASP that catches nothing auto-reopens, so demos never record a
                # closed-on-nothing hold.
                grasp_min_width_m=s["empty_width_m"],
                grasp_open_width_m=s["grasp_open_width_m"],
                gripper_settle_s=s["gripper_settle_s"],
                gripper_min_settle_s=s["gripper_min_settle_s"],
                z_floor_m=s["z_floor_m"],
                smooth_plugin=smooth_plugin,
                # Same motion frame as the autonomous rollouts, so a recorded MV_FWD
                # means the same motion a rollout executes (wrist = along that arm's
                # gripper heading, at constant height).
                motion_frame=str(unified.get("motion_frame", "base")),
                # AgileX-specific: joint_stream (smooth MOVE J streaming) by default.
                motion_backend=str(unified.get("motion_backend", "joint_stream")),
                joint_stream_hz=float(unified.get("joint_stream_hz", 50.0)),
                ori_flex_rad=math.radians(float(unified.get("ori_flex_deg", 15.0))),
                verbose=False,
            )

        # The reset: BOTH arms travel to their BEGIN poses AT THE SAME TIME. Two gates,
        # matching the rest of the stack: move_to_begin_on_init only skips the ONE-TIME
        # init move; the reset when a recording stops runs for every arm that has a pose.
        def go_begin_both(respect_init_flag: bool = False) -> None:
            targets = {
                side: settings[side]["begin_joints"]
                for side in SIDES
                if settings[side]["begin_joints"]
                and not (respect_init_flag and not settings[side]["move_to_begin_on_init"])
            }
            if not targets:
                return
            go_begin_dual(
                {side: session.robots[side] for side in targets},
                targets,
                time_to_go=args.reset_time_s,
                label="begin",
                open_gripper=True,  # a reset always leaves the hand empty
            )

        try:
            go_begin_both(respect_init_flag=True)
        except Exception as exc:  # noqa: BLE001 - a begin move must not kill the session
            print(f"[collect-dual] move-to-begin on init failed: {exc}")
        for side in SIDES:
            controllers[side].sync_from_robot()
            floor = controllers[side].z_floor_m
            print(
                f"[collect-dual] {side}: z-floor="
                + (f"{floor:.4f} m" if floor is not None else "OFF")
            )

        collector = DualRolloutCollector(
            session,
            controllers,
            recorders,
            mode=args.mode,
            move_interval=args.move_interval,
            display_scale=args.display_scale,
            target_fps=args.target_fps,
            home_fn=None if args.no_reset_on_stop else go_begin_both,
            include_rotate_keys=True,
            # Mode B: every recorded step carries a deliberate choice from BOTH arms
            # (an idle arm presses its STILL key), so the dataset is not padded with
            # auto-STILL. --no-sync-steps restores the any-keypress-steps behaviour.
            sync_steps=(not args.no_sync_steps) if args.mode == "B" else False,
            window_title="Show-Harness - Dual Piper Rollout Collection",
        )
        collector.run()
        print("[collect-dual] done.")
    except KeyboardInterrupt:
        print("\n[collect-dual] interrupted; shutting down.")
    except Exception as exc:  # noqa: BLE001 - surface clearly, then clean up
        print(f"[collect-dual] ERROR: {exc}")
        exit_code = 1
    finally:
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
