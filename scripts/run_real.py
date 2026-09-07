#!/usr/bin/env python3
"""Show-Harness real-robot deployment: pure-vision closed-loop manipulation.

The real-robot zero-shot entry point. Instead
of a MuJoCo task suite this drives a physical Franka through a ``FrankaSession``
(RealSense cameras + NUC pose) and a ``FrankaAtomicController`` (atomic VLM tokens
-> Cartesian-impedance setpoints, with a Z safety floor). The task is standalone
and defined entirely by the ``task`` prompt in ``configs/robot_franka.yaml`` -- there is
no task suite / task id / episode index.

Pipeline (identical control contract to the simulator):
    plan subgoals (VLM) -> per step: observe -> controller VLM picks one atomic
    token -> execute on the robot -> log frame; DONE advances the subgoal, DONE on
    the final subgoal completes the task.

Safety + interruption:
  * An optional Z floor can lock the end-effector at or above its startup
    (tabletop-contact) height so MV_DOWN can never press the arm into the table.
    It is **off by default**; enable it per run with ``--z-floor`` (lock at the
    start height) or ``--z-floor-m <m>`` (explicit height), or via the
    ``enable_z_floor`` / ``z_floor_m`` config keys.
  * Ctrl+C (KeyboardInterrupt) triggers a graceful teardown that still compiles the
    visualization video, one shared directory layout / naming:
    ``<log_dir>/<variant>/<MMDD>/task_0/<HH-MM-SS>/rollout_{success,failure}.mp4``.

Examples
--------
    # Real robot + real cameras (Terminal 2; the VLM server runs in Terminal 1):
    python scripts/run_real.py --vlm-url http://localhost:8000/v1

    # Dry run with mock hardware (no NUC / cameras needed):
    python scripts/run_real.py --mock-robot --mock-cameras --vlm-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Any

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.ui.console as console
from core.config import load_secrets_env, load_yaml
from core.record.episode_logger import EpisodeLogger
from core.ui.live_view import LiveView
from core.prompting.prompt_loader import load_prompt_dir
from plugins.assembly import build_video_ref, install_dagger
from plugins.config import PluginsConfig


from core.launch import (  # noqa: F401  (re-exported for compatibility)
    DEFAULTS,
    REAL_TASK_ID,
    action_chunk_step_num,
    auto_home_piper,
    build_config,
    coarse_step_m,
    high_above_table_m,
    make_controller,
    make_runner,
    make_session,
    make_vlm_client,
    mem_text_len,
    move_to_begin_franka,
    move_to_begin_on_init,
    piper_session_config,
    recovery_empty_width_m,
    recovery_open_width_m,
    resolve_franka_begin,
    resolve_hardware,
    resolve_primitives_path,
    resolve_z_floor,
    resolve_z_floor_name,
    rotate_max_accum_deg,
    rotate_step_deg,
    run_variant,
    session_config,
    smooth_dt_s,
    smooth_min_waypoint_m,
    smooth_substeps,
    table_height_m,
    up_step_m,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show-Harness real-robot: pure-vision closed-loop manipulation")
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_franka.yaml"))
    parser.add_argument(
        "--arm",
        default="left",
        choices=["left", "right"],
        help="Piper only: which arm the autonomous rollout drives. The rig is dual-arm "
        "and its config is unified, but the VLM controller commands ONE arm.",
    )
    parser.add_argument(
        "--begin-pose",
        default=None,
        help="Which NAMED start pose the arm homes to before the rollout. Piper: "
        "arms.<side>.poses.<name> (list them with go_begin.sh --list). Franka: "
        "poses.<name> in robot_franka.yaml -- passing this flag also enables the "
        "begin move for the run (the config default is the go_home_client.py home "
        "pose). Default: the config's begin_pose.",
    )
    parser.add_argument(
        "--video-ref",
        default=None,
        help="Reference demo video to replicate. Enables plugins.video_ref for this "
        "run and overrides the config's video_ref_path.",
    )
    parser.add_argument(
        "--primitives-config",
        default=None,
        help="Atomic-motion primitives yaml. Default: the robot config's "
        "primitives_config key, else configs/primitives_franka.yaml (franka) / "
        "configs/primitives_piper.yaml (piper).",
    )
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--task", default=None, help="Override the task prompt.")
    parser.add_argument("--gripper-color", default=None, help="Override the gripper color.")
    parser.add_argument(
        "--fine-m",
        type=float,
        default=None,
        dest="fine_step_m",
        help="Override the fine per-step Cartesian move distance (meters).",
    )
    parser.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument(
        "--vlm-backend",
        default=os.environ.get("VLM_BACKEND"),
        help="Which vlm_backends profile to use (e.g. gemini, local). " "For `local`, must match the model launched by scripts/serve_vlm.sh.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)

    # Hardware (override config / use mock hardware for a dry run).
    parser.add_argument("--mock-robot", action="store_true", help="Use a simulated robot (no NUC).")
    parser.add_argument("--mock-cameras", action="store_true", help="Use random mock cameras.")
    parser.add_argument("--nuc-ip", default=None, help="Override the Franka NUC IP.")
    parser.add_argument("--nuc-port", type=int, default=None, help="Override the NUC port.")
    parser.add_argument(
        "--no-impedance",
        action="store_true",
        help="Do not start the Cartesian impedance controller.",
    )

    # Safety floor (ON by default at the calibrated table-contact height).
    parser.add_argument(
        "--z-floor",
        action="store_true",
        help="Re-capture the Z floor at the start (tabletop-contact) height instead of "
        "using the calibrated default. Begin with the gripper resting on the table.",
    )
    parser.add_argument(
        "--z-floor-m",
        type=float,
        default=None,
        help="Override the Z safety floor with an explicit minimum EEF height (meters).",
    )
    parser.add_argument(
        "--z-floor-name",
        default=None,
        help="Which NAMED Z floor (z_floors.<name> in the robot config) this run uses. "
        "Default: the config's z_floor_name. Capture new ones with "
        "scripts/franka/capture_z_floor.sh --name <name> --write.",
    )
    parser.add_argument(
        "--no-z-floor",
        action="store_true",
        help="Disable the Z safety floor entirely (NOT recommended on hardware).",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the live dual-camera window during the rollout.",
    )
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DEBUG") == "1")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    # Populate os.environ from configs/secrets.env (api keys) before resolving the VLM
    # config, so a hosted backend's key (api_key_env) is found without any shell setup.
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    hardware = resolve_hardware(args, robot_cfg)
    # The Piper rig is dual-arm and its config is unified (shared keys + arms.left /
    # arms.right). An autonomous rollout drives ONE arm (the VLM controller is
    # single-arm), so flatten the unified config down to the selected arm's view --
    # everything downstream then reads the same keys it always has.
    if hardware == "piper":
        from core.piper.config import arm_config

        # --begin-pose selects a NAMED start pose for this run; arm_config resolves it
        # into begin_joints, so every consumer below reads the key it always has.
        if args.begin_pose:
            robot_cfg["begin_pose"] = args.begin_pose
        robot_cfg = arm_config(robot_cfg, args.arm)
        print(f"Piper arm: {args.arm} (begin pose: {robot_cfg.get('begin_pose', 'begin_joints')})")
    elif args.begin_pose:
        # Franka: an explicit --begin-pose both selects the named pose (poses.<name>
        # in robot_franka.yaml) and enables the begin move for this run; without the
        # flag the config decides (move_to_begin_on_init, default off -- the operator
        # homes externally with franka_server/go_home_client.py).
        robot_cfg["begin_pose"] = args.begin_pose
        robot_cfg["move_to_begin_on_init"] = True
    primitives_cfg = load_yaml(resolve_primitives_path(args, robot_cfg, hardware))
    prompts_cfg = load_prompt_dir(args.prompts_dir)
    cfg = build_config(args, robot_cfg)
    cfg["hardware"] = hardware
    if hardware == "franka":
        resolve_franka_begin(cfg)
    resolve_z_floor_name(cfg, args)
    client = make_vlm_client(args, cfg)

    plugins = PluginsConfig.from_config(cfg)
    # Reference-video replication: distill a demo video into an ordered operation
    # brief the planner replicates (single-arm mode: no arm attribution). An explicit
    # --video-ref is itself the opt-in (no yaml edit needed for a one-off run).
    # Config errors fail HERE, before any VLM cost.
    video_ref_plugin = build_video_ref(plugins, cfg, args.video_ref, single=True)

    vlm_cfg = cfg["vlm"]
    # Run header: everything the operator needs to know WHICH run this is, in one block.
    print(f"\n{console.rule(f'rollout  ·  {hardware} (single arm)')}")
    print(f"  task     {cfg['task']}")
    print(
        f"  model    {vlm_cfg.get('backend', '?')} / {vlm_cfg['model']}"
        f"  ({'CoT' if vlm_cfg.get('reasoning_cot') else 'no-CoT'})"
    )
    print(
        f"  motion   {cfg.get('motion_frame', 'base')} frame  ·  step "
        f"{float(cfg.get('fine_step_m', 0.02)) * 100:g} cm  ·  max {int(cfg['max_steps'])} steps"
    )
    if cfg.get("gripper_color"):
        print(f"  gripper  {cfg['gripper_color']}")
    if hardware == "franka" and cfg.get("move_to_begin_on_init", False):
        print(f"  begin    {cfg.get('begin_pose', 'default')}")
    if cfg.get("_z_floor_name"):
        print(f"  z-floor  {cfg['_z_floor_name']} ({float(cfg['z_floor_m']):g} m)")
    _print_plugins(plugins, cfg, video_ref_plugin)
    # The /models readiness poll exists to wait for the LOCAL vLLM to cold-start. Hosted
    # providers (openai/gemini) are always-on and some do not expose /models at all
    # (some answer 404), so polling it would block for the full startup_wait_s. Skip the
    # poll for hosted backends; the planner's first request surfaces any auth/endpoint issue.
    if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
        print(
            f"Waiting for VLM endpoint {vlm_cfg['base_url']}/models "
            f"for up to {float(vlm_cfg.get('startup_wait_s', 0)):.0f}s..."
        )
        client.health_check(
            wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
            poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
        )
        print("VLM endpoint is ready.")
    else:
        print(f"VLM backend {vlm_cfg.get('backend')} is hosted ({vlm_cfg['base_url']}); " "skipping the /models readiness poll.")

    # Analyze the reference demo BEFORE the arm moves: an unusable video aborts here,
    # and the operator can eyeball the extracted operations against the actual demo.
    if video_ref_plugin.enabled:
        print(f"  demo     analyzing {video_ref_plugin.video_path.name} ...")
        video_ref_plugin.extract_brief(client, debug=args.debug)
        print(
            f"  demo     {video_ref_plugin.brief['task']}  "
            f"({len(video_ref_plugin.sampled_indices)} frames)"
        )
        for line in video_ref_plugin.summary_lines():
            print(f"           {line}")

    session = make_session(cfg, args, hardware)
    viewer = LiveView(
        enabled=not args.no_show,
        title=f"Show-Harness | {cfg['task']}",
    )
    # DAGGER: real-time human keyboard override during the rollout, with the teleop
    # bindings (single-arm layout). Keys arrive through the live-view window's
    # STREAM thread, so it needs both the window and a session that can feed it
    # (get_camera_frames -- the Franka session; the single-Piper session cannot yet).
    dagger_plugin = install_dagger(
        plugins,
        viewer,
        single=True,
        session=session,
        hardware=hardware,
        notify=lambda msg: print(f"[run-real] {msg}"),
    )
    exit_code = 0
    try:
        session.connect()
        controller = make_controller(cfg, primitives_cfg, session, args, hardware)
        # Move to the BEGIN pose BEFORE the first sync, so the run starts from a known
        # pose and the controller setpoint is captured there.
        if hardware == "piper":
            move_to_begin_on_init(cfg, session)
        else:
            move_to_begin_franka(cfg, session)
        controller.sync_from_robot()
        if controller.z_floor_m is not None:
            print(f"Z floor (min EEF height) = {controller.z_floor_m:.4f} m; " "downward motion below this height is blocked.")
        else:
            print("Z floor disabled: the arm may descend without a software limit.")
        # The RUNNER owns the console from here: one readable block per step (the
        # dual path's convention). Silence the controller's per-token coordinate
        # chatter; warnings (reach clamp, dropped gripper, ...) print regardless.
        controller.verbose = False

        logger = EpisodeLogger(
            Path(ROOT) / cfg["log_dir"],
            REAL_TASK_ID,
            variant=run_variant(vlm_cfg),
            video_fps=float((cfg.get("v0") or {}).get("video_fps", 2.0)),
        )
        logger.write_metadata(
            {
                "task": cfg["task"],
                "gripper_color": cfg.get("gripper_color", "black"),
                "robot_config": cfg,
                "primitives_config": primitives_cfg,
                "prompts_dir": str(Path(args.prompts_dir).resolve()),
                "control_mode": "real",
                "z_floor_m": controller.z_floor_m,
                "debug": args.debug,
                **({"video_ref": video_ref_plugin.metadata()} if video_ref_plugin.enabled else {}),
            }
        )

        runner = make_runner(
            cfg=cfg,
            prompts_cfg=prompts_cfg,
            client=client,
            session=session,
            controller=controller,
            logger=logger,
            debug=args.debug,
            viewer=viewer,
            video_ref_plugin=video_ref_plugin,
            dagger_plugin=dagger_plugin,
        )
        if viewer.enabled and hasattr(session, "get_camera_frames"):
            # Continuous live feed: a render thread streams fresh camera frames
            # (~12 Hz) for the whole rollout -- the window stays live while the VLM
            # thinks and while the arm moves; the runner only posts status overlays.
            # It also pumps the keyboard, which is what makes DAGGER keys land in
            # near-real time. The affordance adapter overlays the active dots on the
            # frames (pass-through when the plugin is off / has no dots).
            runner_affordance = runner.affordance_plugin
            viewer.start_stream(
                lambda: (
                    runner_affordance.annotate_frames(session.get_camera_frames())
                    if runner_affordance is not None
                    else session.get_camera_frames()
                )
            )
        result = runner.run()
        # Reset to BEGIN when the rollout finishes (Piper only; no-op if begin_joints
        # is unset). Best-effort: a reset failure must not mask the rollout result.
        if hardware == "piper":
            auto_home_piper(cfg, session)
        print(f"\n{console.rule('result')}")
        print(f"  {'SUCCESS' if result.success else 'FAILED'}  ({result.end_reason})")
        print(f"  steps    {result.steps}/{int(cfg['max_steps'])}")
        print(f"  run dir  {result.run_dir}")
        print(f"  video    {result.video_path}\n")
        exit_code = 0 if result.success else 2
    except KeyboardInterrupt:
        print("\n[run-real] interrupted before the rollout loop started.")
        exit_code = 130
    finally:
        viewer.close()
        session.close()
    return exit_code



def _print_plugins(plugins: PluginsConfig, cfg: dict[str, Any], video_ref_plugin: Any) -> None:
    """The run header's plugins block: one line per ENABLED plugin with its live
    parameters (the dual header's convention). Disabled plugins are hidden entirely."""
    def cm(key: str, default: float) -> str:
        return f"{float(cfg.get(key, default)) * 100:g} cm"

    details = {
        "subgoal": "planner-expanded stage track",
        "proprioception": f"high-above {float(high_above_table_m(cfg)):.2f} m",
        "recovery": f"empty close < {float(recovery_empty_width_m(cfg)) * 1000:g} mm",
        "variable_step": (
            f"fine {cm('fine_step_m', 0.02)} · coarse {cm('coarse_step_m', 0.05)}"
            f" · up {cm('up_step_m', 0.08)}"
        ),
        "action_chunk": f"{int(cfg.get('action_chunk_step_num', 3))} moves per VLM call",
        "mem_text": f"last {mem_text_len(cfg)} moves in the prompt",
        "smooth": (
            f"{int(cfg.get('smooth_substeps', 20))} substeps @ "
            f"{float(cfg.get('smooth_dt_s', 0.05)) * 1000:g} ms"
        ),
        "rotation": (
            f"ROTATE_CW/CCW {float(cfg.get('rotate_step_deg', 30.0)):g} deg"
            f" · max {float(cfg.get('rotate_max_accum_deg', 150.0)):g} deg"
        ),
        "deepplan": "<REASON> pivot for conditional branches",
        "affordance": (
            f"front dot per stage + wrist tracking · "
            f"{int(cfg.get('affordance_verify_rounds', 1))} verify round(s)"
        ),
        "video_ref": (
            f"{video_ref_plugin.video_path.name} ({video_ref_plugin.num_frames} frames)"
            if video_ref_plugin.enabled and video_ref_plugin.video_path is not None
            else ""
        ),
        "dagger": (
            "teleop keys override the model live · CLICK the live-view window to arm"
        ),
    }
    known = (
        "subgoal", "proprioception", "recovery", "variable_step", "action_chunk",
        "mem_text", "smooth", "coords", "mcq", "deepplan", "rotation",
        "affordance", "video_ref", "dagger",
    )
    label = "plugins"
    for name in known:
        enabled = plugins.enabled(name, default=False) or (
            name == "video_ref" and video_ref_plugin.enabled
        )
        if not enabled:
            continue
        detail = details.get(name, "")
        print(f"  {label:<8} {name:<15}{console.dim(detail)}")
        label = ""  # only the first line carries the section label

if __name__ == "__main__":
    raise SystemExit(main())
