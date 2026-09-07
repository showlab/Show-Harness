#!/usr/bin/env python3
"""DUAL-ARM web teleop + Mode-B rollout recorder (for humans and GUI agents).

Dual counterpart of collect_rollouts_web.py: serves the compose-and-commit UI
(gumi/web_teleop_dual/) where every executed action is one
SYNCHRONIZED (a_left, a_right) pair -- the STILL no-op filling whichever side did
not act -- recorded with core.teleop.dual.DualRolloutRecorder in the Mode-B layout
(agentview/ + wrist_left/ + wrist_right/ + paired actions.jsonl), byte-compatible
with what the pygame dual collector (core/teleop/dual.py) writes.

The prompt handed to the browser agent lives in prompts/web_operator_dual.txt
(usage doc: gumi/README.md). The
agent operates by CLICKING: per-arm button panels queue each arm's next action
(click again = repeat) and a central COMMIT executes the pair as one synchronized
step -- self-verifying (no page-focus dance, no command grammar). The keyboard
and the small command box drive the same queue for humans/scripts. The prompt is
deliberately MINIMAL: only the non-discoverable rules (verify the "#N" result
counter, direction convention, empty-grasp auto-reopen, coordination). Change it
together with this UI.

Modes
-----
--sim               synthetic dual-arm tabletop (web_teleop_dual/sim_dual.py): two
                    Piper-shaped arms, an orange + a green target cube, one shared
                    plate. No CAN/ROS/cameras needed -- the full GUI-agent loop runs
                    on a dev box.
(default)           the real dual-Piper rig, wired from configs/robot_piper.yaml's
                    unified ``arms:`` block exactly like scripts/run_real_dual.py (same
                    prerequisites: both CAN buses up, both arm nodes mode 1, astra
                    multi_camera launch publishing camera_f / camera_l / camera_r).
                    --mock-robot / --mock-cameras swap in mocks for a dry run.

Training-side note: the STILL token is recorded in every pair. If the LlamaFactory
converter (rollout_to_llamafactory.py, external repo) is used on this data, STILL
must be added to its ACTION_TOKENS whitelist or it will silently drop those labels
(see gumi/README.md, "Real-rig cautions").

Examples
--------
# Dev box / GUI-agent test drive (no hardware):
.venv/bin/python gumi/collect_rollouts_web_dual.py data/rollouts_dual --sim

# Real dual-Piper rig:
.venv/bin/python gumi/collect_rollouts_web_dual.py data/rollouts_dual --port 8620
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

# The file lives one level below the repo root; make the root importable so the
# gumi package and core/ resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.piper.config import SIDES, arm_config
from core.teleop.dual import DualRolloutRecorder
from interpreters.piper_atomic_controller import (
    PIPER_EMPTY_GRASP_WIDTH_M,
    PiperAtomicController,
)
from gumi.web_teleop_dual.dual_backend import DualTeleopBackend
from gumi.web_teleop_dual.server import serve

DEFAULT_PORT = 8620  # the single-arm collector keeps 8600
DEFAULT_RESET_TIME_S = 3.0


def _require_ros_master(uri: str | None = None, timeout_s: float = 1.0) -> None:
    """Fail before ``rospy.init_node`` when the hardware ROS master is absent.

    rospy retries master registration forever, which made a missing hardware
    prerequisite look like a frozen collector. A TCP preflight is sufficient here:
    the arm/camera launchers perform the detailed ROS package and node checks.
    """
    master_uri = uri or os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    parsed = urlparse(master_uri)
    host = parsed.hostname
    port = parsed.port or 11311
    if not host:
        raise SystemExit(
            f"Invalid ROS_MASTER_URI={master_uri!r}; expected http://HOST:PORT"
        )
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return
    except OSError as exc:
        raise SystemExit(
            f"Cannot reach ROS master at {master_uri} ({exc}).\n"
            "The real dual-arm collector requires the camera and arm ROS nodes.\n"
            "Start them in separate terminals, then retry:\n"
            "  scripts/piper/run_cameras.sh\n"
            "  scripts/piper/run_arm.sh 1 true\n"
            "The arm launcher enables both arms and closes both grippers; clear the "
            "workspace and keep the emergency stop accessible."
        ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-arm web teleoperation + Mode-B rollout recorder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("save_path", help="Root directory; rollouts are auto-numbered inside it.")
    p.add_argument("--rollout-prefix", default="rollout_", help="Rollout folder prefix.")
    p.add_argument(
        "--task",
        default=None,
        help="Task prompt shown in the UI and saved to metadata. Overrides the robot "
        "YAML's task (and the --sim scene's built-in task).",
    )

    # Server.
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default {DEFAULT_PORT}).")
    p.add_argument("--capture-fps", type=float, default=15.0, help="Camera capture/stream rate.")
    p.add_argument(
        "--allow-stop-anytime",
        action="store_true",
        help="Do not gate Stop on task completion (default: Stop needs the task done -- "
        "in --sim both cubes on the plate; on hardware one pick-and-place by any arm).",
    )

    # Hardware / world.
    p.add_argument("--sim", action="store_true", help="Synthetic dual-arm world (no hardware).")
    p.add_argument("--seed", type=int, default=None, help="--sim scene layout seed.")
    p.add_argument("--mock-robot", action="store_true", help="Real path with MOCK arms (no CAN/ROS).")
    p.add_argument("--mock-cameras", action="store_true", help="Real path with random mock cameras.")
    p.add_argument("--no-reset-on-stop", action="store_true", help="Do not auto-home on Stop.")
    p.add_argument("--reset-time-s", type=float, default=DEFAULT_RESET_TIME_S)
    p.add_argument(
        "--begin-pose",
        default=None,
        help="Which NAMED start pose (arms.<side>.poses.<name>) both arms home to. "
        "Default: the robot config's begin_pose.",
    )

    # Motion / safety.
    p.add_argument("--primitives", default=None, help="Primitives YAML (default: piper).")
    p.add_argument("--robot-config", default=None, help="Robot YAML (default: robot_piper.yaml).")
    p.add_argument("--step-m", type=float, default=None, help="Override primitives step_m.")
    p.add_argument("--yaw-step-rad", type=float, default=None, help="Override yaw_step_rad.")
    p.add_argument("--settle-steps", type=int, default=4)
    p.add_argument("--settle-dt-s", type=float, default=0.05)
    p.add_argument("--no-z-floor", action="store_true", help="Disable BOTH arms' Z floors (NOT recommended).")
    p.add_argument("--video-fps", type=float, default=10.0, help="Saved visualization video FPS.")
    return p.parse_args()


def build_sim_rig(args, primitives_cfg, robot_cfg):
    """Synthetic dual world: DualSimScene + two SimDualPiperRobot + DualSimSession."""
    import numpy as np

    from gumi.web_teleop_dual import sim_dual

    scene = sim_dual.DualSimScene(seed=args.seed)
    robots = {side: sim_dual.SimDualPiperRobot(scene, side) for side in SIDES}
    session = sim_dual.DualSimSession(scene, robots)
    controllers = {}
    for side in SIDES:
        controller = PiperAtomicController.from_primitives_config(
            robots[side],
            primitives_cfg,
            settle_steps=1,
            settle_dt_s=0.0,
            grasp_min_width_m=PIPER_EMPTY_GRASP_WIDTH_M,
            grasp_open_width_m=0.055,
            gripper_settle_s=0.0,   # the sim gripper settles synchronously
            gripper_min_settle_s=0.0,
            z_floor_m=scene.z_table + 0.02,
            verbose=False,
        )
        controller.LOG_TAG = f"sim-{side[0].upper()}"
        controller.sync_from_robot()
        controllers[side] = controller

    def home_fn() -> None:
        for side in SIDES:  # instant in sim; order is irrelevant
            robots[side].move_to_joint_positions(np.zeros(6), args.reset_time_s)

    meta = {"robot": "piper_dual_sim", "sim": True, "arms": list(SIDES), "task": scene.task_text()}
    home = None if args.no_reset_on_stop else home_fn
    return session, controllers, home, scene, meta


def build_real_dual_rig(args, primitives_cfg, robot_cfg):
    """Real dual-Piper rig from the unified ``arms:`` config, like scripts/run_real_dual.py."""
    # Only the all-mock path is independent of ROS. Real cameras and real robots
    # both use ROS topics and otherwise block forever while rospy retries a missing
    # master connection.
    if not (args.mock_robot and args.mock_cameras):
        _require_ros_master()

    from core.piper.dual_session import DualPiperSession, DualPiperSessionConfig
    from core.piper.poses import go_begin_dual

    if args.begin_pose:
        robot_cfg["begin_pose"] = args.begin_pose
    arm_cfgs = {side: arm_config(robot_cfg, side) for side in SIDES}
    open_width_m = float(robot_cfg.get("open_width_m", 0.07))
    empty_width_m = float(robot_cfg.get("empty_width_m", PIPER_EMPTY_GRASP_WIDTH_M))
    rb = robot_cfg.get("robot", {}) or {}

    session = DualPiperSession(
        DualPiperSessionConfig(
            use_mock_robots=bool(args.mock_robot),
            open_width_left_m=open_width_m,
            open_width_right_m=open_width_m,
            connect_cameras=True,
            use_mock_cameras=bool(args.mock_cameras),
            front_camera_topic=str(rb.get("front_camera_topic", "/camera_f/color/image_raw")),
            wrist_left_camera_topic=str(
                arm_cfgs["left"]["robot"].get("wrist_camera_topic", "/camera_l/color/image_raw")
            ),
            wrist_right_camera_topic=str(
                arm_cfgs["right"]["robot"].get("wrist_camera_topic", "/camera_r/color/image_raw")
            ),
            observation_resolution=int(robot_cfg.get("camera_resolution", 256)),
            verbose=True,
        )
    )
    session.connect()

    controllers = {}
    for side in SIDES:
        cfg = arm_cfgs[side]
        # Per-arm calibrated Z floor, REQUIRED (MOVE P is stiff position control):
        # each arm's base height differs, so a shared value is not acceptable.
        if args.no_z_floor:
            z_floor_m = None
            print(f"[web-teleop-dual] WARNING: {side} Z floor disabled -- mind the table.")
        else:
            z_floor_m = cfg.get("z_floor_m") if robot_cfg.get("enable_z_floor", True) else None
            if z_floor_m is None:
                raise SystemExit(
                    f"No calibrated z_floor_m for the {side} arm (arms.{side}.z_floor_m). "
                    f"Calibrate it first: scripts/piper/capture_z_floor.sh --arm {side} --write "
                    "(or pass --no-z-floor, NOT recommended)."
                )
        controller = PiperAtomicController.from_primitives_config(
            session.robots[side],
            primitives_cfg,
            settle_steps=args.settle_steps,
            settle_dt_s=args.settle_dt_s,
            grasp_min_width_m=empty_width_m,
            grasp_open_width_m=float(robot_cfg.get("open_width_m", 0.055)),
            gripper_settle_s=float(rb.get("gripper_settle_s", 1.5)),
            gripper_min_settle_s=float(rb.get("gripper_min_settle_s", 0.3)),
            z_floor_m=(float(z_floor_m) if z_floor_m is not None else None),
            verbose=False,
        )
        controller.LOG_TAG = f"piper-{side[0].upper()}"
        controllers[side] = controller

    begin = {side: arm_cfgs[side].get("begin_joints") for side in SIDES}
    if robot_cfg.get("move_to_begin_on_init", True) and all(begin.values()):
        try:
            go_begin_dual(session.robots, begin, time_to_go=args.reset_time_s,
                          label="begin", open_gripper=True, verbose=False)
        except Exception as exc:  # noqa: BLE001 - a home move must not kill the session
            print(f"[web-teleop-dual] move-to-begin on init failed: {exc}")
    for side in SIDES:
        controllers[side].sync_from_robot()
        print(f"[web-teleop-dual] {side.upper():<5} Z floor = "
              f"{controllers[side].z_floor_m if controllers[side].z_floor_m is not None else 'DISABLED'}")

    home_fn = None
    if not args.no_reset_on_stop and all(begin.values()):
        def home_fn() -> None:
            go_begin_dual(session.robots, begin, time_to_go=args.reset_time_s,
                          open_gripper=True, verbose=False)
    elif not all(begin.values()):
        print("[web-teleop-dual] auto-home on stop disabled: capture begin poses for both arms.")

    meta = {
        "robot": "piper_dual",
        "arms": list(SIDES),
        "motion_frame": "cartesian",
        "open_width_m": open_width_m,
        "grasp_min_width_m": empty_width_m,
        "z_floor_m": {side: controllers[side].z_floor_m for side in SIDES},
        "task": str(robot_cfg.get("task", "")),
    }
    return session, controllers, home_fn, None, meta


def main() -> int:
    args = parse_args()
    if args.primitives is None:
        args.primitives = str(ROOT / "configs" / "primitives_piper.yaml")
    if args.robot_config is None:
        args.robot_config = str(ROOT / "configs" / "robot_piper.yaml")

    primitives_cfg = load_yaml(args.primitives)
    robot_cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    if args.step_m is not None:
        primitives_cfg["step_m"] = args.step_m
    if args.yaw_step_rad is not None:
        primitives_cfg["yaw_step_rad"] = args.yaw_step_rad

    build = build_sim_rig if args.sim else build_real_dual_rig
    session, controllers, home_fn, scene, meta = build(args, primitives_cfg, robot_cfg)

    if args.task is not None:
        meta["task"] = args.task
    print(f"[web-teleop-dual] task: {meta.get('task', '') or '(none)'}")

    recorder = DualRolloutRecorder(args.save_path, prefix=args.rollout_prefix, video_fps=args.video_fps)
    recorder.session_meta = {
        **meta,
        "step_m": primitives_cfg["step_m"],
        "yaw_step_rad": primitives_cfg["yaw_step_rad"],
    }
    print(f"[web-teleop-dual] saving rollouts under: {Path(args.save_path).resolve()}")

    backend = DualTeleopBackend(
        session,
        controllers,
        recorder,
        home_fn=home_fn,
        scene=scene,
        task_text=str(meta.get("task", "")),
        require_task_done=not args.allow_stop_anytime,
        capture_fps=args.capture_fps,
    )
    exit_code = 0
    try:
        backend.start()
        httpd = serve(backend, host=args.host, port=args.port)
        print(
            f"[web-teleop-dual] UI ready:  http://{socket.gethostname()}:{args.port}/  "
            f"(bound {args.host}:{args.port}; Ctrl-C to quit)"
        )
        print(f"[web-teleop-dual] agent API: POST http://localhost:{args.port}/api/step "
              '-d \'{"left": "MV_FWD", "right": "STILL"}\'')
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web-teleop-dual] interrupted by user")
    except Exception as exc:  # noqa: BLE001 - surface the failure clearly
        print(f"[web-teleop-dual] ERROR: {exc}")
        exit_code = 1
    finally:
        backend.shutdown()
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
