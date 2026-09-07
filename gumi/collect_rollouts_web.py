#!/usr/bin/env python3
"""Browser-based teleoperation + rollout recorder (web UI counterpart of
collect_rollouts_piper.py; see .ai/agent_todo/uidesign.png for the layout).

Serves an interactive control page (WASD / Up / Down / rotate / gripper keys,
Start / Stop / Cancel) with live MJPEG streams of both cameras. Start opens a
numbered rollout (same dataset layout as core/teleop/single.py); every key press
records (obs_t, a_t) then executes the atomic token; Stop is REFUSED until the
task is completed (pick up the object and set it down -- with --sim, the orange
cube must physically rest on the blue plate) and then saves + auto-homes the
arm. Cancel discards the in-progress rollout and homes.

Modes
-----
--sim               synthetic tabletop world (no CAN/ROS/cameras needed) -- full
                    pipeline test on a dev box, e.g. for driving via a browser agent.
--robot piper       real Piper rig, wired exactly like collect_rollouts_piper.py
(default)           (same prerequisites: CAN up, arm node mode 1, astra cameras; run
                    with ROS + the Piper workspace sourced).
--robot franka      real Franka rig, wired exactly like collect_rollouts.py
                    (prereq: the NUC single_arm_server.py is up on nuc_ip:nuc_port,
                    two RealSense cameras connected).

Examples
--------
# Dev box / GUI-agent test drive (no hardware):
.venv/bin/python gumi/collect_rollouts_web.py data/rollouts_web --sim

# Real Piper rig:
.venv/bin/python gumi/collect_rollouts_web.py data/rollouts_piper --port 8600

# Real Franka rig:
.venv/bin/python gumi/collect_rollouts_web.py data/rollouts --robot franka --port 8600
"""
from __future__ import annotations

import sys


def _franka_selected(argv: list) -> bool:
    """True iff this is a real-Franka run (``--robot franka`` and not ``--sim``)."""
    if "--sim" in argv:
        return False
    for i, a in enumerate(argv):
        if a == "--robot" and i + 1 < len(argv):
            return argv[i + 1] == "franka"
        if a == "--robot=franka":
            return True
    return False


# The real Franka backend talks to the NUC over zerorpc, which is gevent-based and whose
# event loop (Hub) is thread-local. TeleopBackend funnels all robot access onto a dedicated
# worker thread, so a client connected on the main thread and called from the worker raises
# ``gevent LoopExit: This operation would block forever``. Monkey-patching turns the threads
# into greenlets that share ONE hub, which the zerorpc client tolerates. Gate it to the real-
# Franka path (Piper/sim use no gevent) and patch BEFORE importing socket/threading/zerorpc.
if _franka_selected(sys.argv):
    from gevent import monkey

    monkey.patch_all()

import argparse
from pathlib import Path
from typing import Optional

# The file lives one level below the repo root; make the root importable so the
# gumi package and core/ resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreters.piper_atomic_controller import (
    PIPER_EMPTY_GRASP_WIDTH_M,
    PiperAtomicController,
)
from core.config import load_yaml
from core.teleop.single import RolloutRecorder
from gumi.web_teleop.backend import TeleopBackend
from gumi.web_teleop.server import serve

DEFAULT_RESET_TIME_S = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Web (browser) teleoperation + rollout recorder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("save_path", help="Root directory; rollouts are auto-numbered inside it.")
    p.add_argument("--rollout-prefix", default="rollout_", help="Rollout folder prefix.")
    p.add_argument(
        "--task",
        default=None,
        help="Task prompt shown in the UI and saved to each rollout's metadata. Overrides "
        "the robot YAML's task (and the --sim scene's built-in task).",
    )

    # Server.
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=8600, help="HTTP port (default 8600).")
    p.add_argument("--capture-fps", type=float, default=15.0, help="Camera capture/stream rate.")
    p.add_argument(
        "--allow-stop-anytime",
        action="store_true",
        help="Do not gate Stop on task completion (default: a rollout can only be "
        "stopped after a successful pick-and-place).",
    )

    # Hardware / world.
    p.add_argument("--sim", action="store_true", help="Synthetic tabletop world (no hardware).")
    p.add_argument(
        "--robot",
        default="piper",
        choices=["piper", "franka"],
        help="Real-hardware backend (ignored with --sim). Default: piper.",
    )
    p.add_argument("--seed", type=int, default=None, help="--sim scene layout seed.")
    p.add_argument("--open-width-m", type=float, default=None, help="RELEASE width (m).")
    p.add_argument("--empty-width-m", type=float, default=None, help="Empty-grasp threshold (m).")
    p.add_argument("--no-reset-on-stop", action="store_true", help="Do not auto-home on Stop.")
    p.add_argument("--reset-time-s", type=float, default=DEFAULT_RESET_TIME_S)

    # Piper hardware (--robot piper).
    p.add_argument("--arm", default=None, choices=["left", "right"], help="Piper arm to drive.")
    p.add_argument("--front-topic", default=None, help="Front (AgentView) color image topic.")
    p.add_argument("--wrist-topic", default=None, help="Wrist color image topic.")

    # Franka hardware (--robot franka).
    p.add_argument("--nuc-ip", default=None, help="Franka NUC IP (default: robot.nuc_ip from --robot-config).")
    p.add_argument("--nuc-port", type=int, default=4242, help="Franka NUC ZeroRPC port.")
    p.add_argument("--external-serial", default=None,
                   help="External RealSense serial (default: robot.external_camera_serial from --robot-config).")
    p.add_argument("--wrist-serial", default=None,
                   help="Wrist RealSense serial (default: robot.wrist_camera_serial from --robot-config).")
    p.add_argument("--mock-robot", action="store_true", help="Franka: simulated robot (no NUC).")
    p.add_argument("--mock-cameras", action="store_true", help="Franka: random mock cameras.")
    p.add_argument("--no-impedance", action="store_true", help="Franka: do not start the impedance controller.")
    p.add_argument(
        "--capture-z-floor",
        action="store_true",
        help="Franka: lock the Z floor at the start (tabletop-contact) height instead of the "
        "calibrated default. Begin with the gripper resting on the table.",
    )

    # Motion. Defaults resolved from --robot / --sim in main() (None here).
    p.add_argument("--primitives", default=None, help="Primitives YAML (default depends on --robot).")
    p.add_argument("--robot-config", default=None, help="Robot YAML (default depends on --robot).")
    p.add_argument("--step-m", type=float, default=None, help="Override primitives step_m.")
    p.add_argument("--yaw-step-rad", type=float, default=None, help="Override yaw_step_rad.")
    p.add_argument("--settle-steps", type=int, default=2)
    p.add_argument("--settle-dt-s", type=float, default=0.02)
    p.add_argument("--z-floor-m", type=float, default=None, help="Explicit min EEF height (m).")
    p.add_argument("--no-z-floor", action="store_true", help="Disable the Z safety floor.")
    p.add_argument("--video-fps", type=float, default=10.0, help="Saved visualization video FPS.")
    return p.parse_args()


def build_sim_rig(args, primitives_cfg, robot_cfg):
    """Synthetic world: SimScene + SimPiperRobot + SimSession + controller + home_fn."""
    from gumi.web_teleop import sim

    scene = sim.SimScene(seed=args.seed)
    robot = sim.SimPiperRobot(scene)
    session = sim.SimSession(scene, robot)
    controller = PiperAtomicController.from_primitives_config(
        robot,
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
    controller.sync_from_robot()

    def home_fn() -> None:
        import numpy as np

        robot.move_to_joint_positions(np.zeros(6), args.reset_time_s)

    meta = {"robot": "piper_sim", "sim": True, "task": scene.task_text()}
    return session, controller, home_fn if not args.no_reset_on_stop else None, scene, meta


def build_franka_rig(args, primitives_cfg, robot_cfg):
    """Real Franka rig, wired exactly like collect_rollouts.py (the pygame collector)."""
    import time

    import numpy as np

    from interpreters.franka_atomic_controller import (
        EMPTY_GRASP_WIDTH_M,
        TABLE_CONTACT_Z_M,
        FrankaAtomicController,
    )
    from core.franka.franka_session import FrankaSession, FrankaSessionConfig

    # Home joint configuration (mirrors collect_rollouts.py / go_home_client.py).
    home_joints = [0.0, -0.5058451, 0.0, -2.6068573, 0.0, 2.0711833, 0.86116207]

    # Empty-grasp width: a GRASP settling at/below this auto-reopens. Resolve from
    # --empty-width-m, else robot_franka.yaml's empty_width_m, else the calibrated default.
    empty_width_m = float(
        args.empty_width_m
        if args.empty_width_m is not None
        else robot_cfg.get("empty_width_m", EMPTY_GRASP_WIDTH_M)
    )

    # Rig identity (NUC address, camera serials) comes from the robot config's
    # site layer unless overridden on the command line.
    rb = robot_cfg.get("robot", {}) or {}
    nuc_ip = args.nuc_ip or rb.get("nuc_ip")
    if not nuc_ip and not args.mock_robot:
        raise SystemExit(
            "[web-teleop] robot.nuc_ip is not configured: copy "
            "configs/site/franka.yaml.example to configs/site/franka.yaml "
            "(or pass --nuc-ip)."
        )
    external_serial = args.external_serial or rb.get("external_camera_serial")
    wrist_serial = args.wrist_serial or rb.get("wrist_camera_serial")

    session = FrankaSession(
        FrankaSessionConfig(
            nuc_ip=str(nuc_ip or "127.0.0.1"),
            nuc_port=args.nuc_port,
            use_mock_robot=args.mock_robot,
            start_impedance=not args.no_impedance,
            connect_cameras=True,  # live dual-view is the whole point
            use_mock_cameras=args.mock_cameras,
            external_camera_serial=external_serial,
            wrist_camera_serial=wrist_serial,
            verbose=True,
        )
    )
    session.connect()

    # Z floor precedence (same as collect_rollouts.py): off > capture-at-start >
    # explicit fixed height (the calibrated table contact by default).
    if args.no_z_floor:
        z_floor_m: Optional[float] = None
        capture_floor = False
        print("[web-teleop] WARNING: Z floor disabled -- mind the table on MV_DOWN.")
    elif args.capture_z_floor:
        z_floor_m, capture_floor = None, True
    else:
        z_floor_m = float(args.z_floor_m if args.z_floor_m is not None else TABLE_CONTACT_Z_M)
        capture_floor = False

    controller = FrankaAtomicController.from_primitives_config(
        session.robot,
        primitives_cfg,
        settle_steps=args.settle_steps,
        settle_dt_s=args.settle_dt_s,
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
        print(f"[web-teleop] Z floor (min EEF height) = {controller.z_floor_m:.4f} m.")

    home_fn = None
    if not args.no_reset_on_stop:
        def home_fn() -> None:
            """Joint move to home, then restart impedance (mirrors collect_rollouts.make_home_fn)."""
            started = time.monotonic()
            controller.robot.move_to_joint_positions(
                np.asarray(home_joints, dtype=float), args.reset_time_s
            )
            # Wait out the trajectory so impedance is not restarted mid-move.
            remaining = (args.reset_time_s + 0.5) - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            if session.config.start_impedance:
                session.start_impedance()

    meta = {
        "robot": "franka",
        "motion_frame": "cartesian",
        "nuc_ip": args.nuc_ip,
        "external_serial": args.external_serial,
        "wrist_serial": args.wrist_serial,
        "mock_robot": args.mock_robot,
        "mock_cameras": args.mock_cameras,
        "grasp_min_width_m": empty_width_m,
        "z_floor_m": controller.z_floor_m,
        "task": str(robot_cfg.get("task", "")),
    }
    return session, controller, home_fn, None, meta


def build_piper_rig(args, primitives_cfg, robot_cfg):
    """Real Piper rig, wired exactly like collect_rollouts_piper.py."""
    from core.piper.poses import go_begin
    from core.piper.piper_session import PiperSession, PiperSessionConfig

    rb = robot_cfg.get("robot", {}) or {}
    arm = args.arm or str(rb.get("arm", "left"))
    front_topic = args.front_topic or str(rb.get("front_camera_topic", "/camera_f/color/image_raw"))
    wrist_topic = args.wrist_topic or str(rb.get("wrist_camera_topic", "/camera_l/color/image_raw"))
    open_width_m = float(
        args.open_width_m if args.open_width_m is not None else rb.get("open_width_m", 0.07)
    )
    empty_width_m = float(
        args.empty_width_m
        if args.empty_width_m is not None
        else robot_cfg.get("empty_width_m", PIPER_EMPTY_GRASP_WIDTH_M)
    )

    session = PiperSession(
        PiperSessionConfig(
            arm=arm,
            open_width_m=open_width_m,
            connect_cameras=True,
            front_camera_topic=front_topic,
            wrist_camera_topic=wrist_topic,
            verbose=True,
        )
    )
    session.connect()

    # Z floor precedence: --no-z-floor > --z-floor-m > calibrated config value.
    cfg_floor = robot_cfg.get("z_floor_m") if robot_cfg.get("enable_z_floor", True) else None
    if args.no_z_floor:
        z_floor_m: Optional[float] = None
        print("[web-teleop] WARNING: Z floor disabled -- MOVE P is stiff; mind the table.")
    elif args.z_floor_m is not None:
        z_floor_m = float(args.z_floor_m)
    elif cfg_floor is not None:
        z_floor_m = float(cfg_floor)
        print(f"[web-teleop] Z floor = {z_floor_m:.4f} m (from {Path(args.robot_config).name}).")
    else:
        raise SystemExit(
            "No z_floor_m in the robot config and none passed via --z-floor-m. The web "
            "collector has no capture-at-start mode; calibrate one first with "
            "scripts/piper/capture_z_floor.sh --write (or pass --no-z-floor, NOT recommended)."
        )

    controller = PiperAtomicController.from_primitives_config(
        session.robot,
        primitives_cfg,
        settle_steps=args.settle_steps,
        settle_dt_s=args.settle_dt_s,
        grasp_min_width_m=empty_width_m,
        grasp_open_width_m=float(robot_cfg.get("open_width_m", 0.055)),
        gripper_settle_s=float(rb.get("gripper_settle_s", 1.5)),
        gripper_min_settle_s=float(rb.get("gripper_min_settle_s", 0.3)),
        z_floor_m=z_floor_m,
        verbose=False,
    )

    home_joints = robot_cfg.get("home_joints")
    if robot_cfg.get("move_to_home_on_init", True) and home_joints:
        try:
            go_begin(session.robot, list(home_joints), time_to_go=args.reset_time_s, label="home")
        except Exception as exc:  # noqa: BLE001 - a home move must not kill the session
            print(f"[web-teleop] move-to-home on init failed: {exc}")
    controller.sync_from_robot()

    home_fn = None
    if not args.no_reset_on_stop and home_joints:
        def home_fn() -> None:
            go_begin(session.robot, list(home_joints), time_to_go=args.reset_time_s, label="home")
    elif not home_joints:
        print("[web-teleop] auto-home on stop disabled: set home_joints in the robot config.")

    meta = {
        "robot": "piper",
        "arm": arm,
        "motion_frame": "cartesian",
        "front_topic": front_topic,
        "wrist_topic": wrist_topic,
        "open_width_m": open_width_m,
        "grasp_min_width_m": empty_width_m,
        "z_floor_m": controller.z_floor_m,
        "task": str(robot_cfg.get("task", "")),
    }
    return session, controller, home_fn, None, meta


def main() -> int:
    args = parse_args()

    # Config defaults depend on the backend: Franka uses the Franka configs; Piper and
    # the sim world (a Piper-shaped rig) use the Piper configs. --primitives/--robot-config
    # override either.
    use_franka = (not args.sim) and args.robot == "franka"
    if args.primitives is None:
        args.primitives = str(
            ROOT / "configs" / ("primitives_franka.yaml" if use_franka else "primitives_piper.yaml")
        )
    if args.robot_config is None:
        args.robot_config = str(
            ROOT / "configs" / ("robot_franka.yaml" if use_franka else "robot_piper.yaml")
        )

    primitives_cfg = load_yaml(args.primitives)
    robot_cfg = load_yaml(args.robot_config) if Path(args.robot_config).is_file() else {}
    if args.step_m is not None:
        primitives_cfg["step_m"] = args.step_m
    if args.yaw_step_rad is not None:
        primitives_cfg["yaw_step_rad"] = args.yaw_step_rad

    if args.sim:
        build = build_sim_rig
    elif args.robot == "franka":
        build = build_franka_rig
    else:
        build = build_piper_rig
    session, controller, home_fn, scene, meta = build(args, primitives_cfg, robot_cfg)

    # --task overrides whatever the rig reported (robot YAML task, or the --sim scene's
    # built-in task): single point so it applies to sim / piper / franka alike.
    if args.task is not None:
        meta["task"] = args.task
    print(f"[web-teleop] task: {meta.get('task', '') or '(none)'}")

    recorder = RolloutRecorder(args.save_path, prefix=args.rollout_prefix, video_fps=args.video_fps)
    recorder.session_meta = {
        **meta,
        "step_m": primitives_cfg["step_m"],
        "yaw_step_rad": primitives_cfg["yaw_step_rad"],
    }
    print(f"[web-teleop] saving rollouts under: {Path(args.save_path).resolve()}")

    backend = TeleopBackend(
        session,
        controller,
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
        import socket

        print(
            f"[web-teleop] UI ready:  http://{socket.gethostname()}:{args.port}/  "
            f"(bound {args.host}:{args.port}; Ctrl-C to quit)"
        )
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web-teleop] interrupted by user")
    except Exception as exc:  # noqa: BLE001 - surface the failure clearly
        print(f"[web-teleop] ERROR: {exc}")
        exit_code = 1
    finally:
        backend.shutdown()
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
