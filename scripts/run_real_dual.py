#!/usr/bin/env python3
"""Show-Harness dual-arm deployment: pure-vision closed-loop manipulation on BOTH Pipers.

Dual counterpart of ``scripts/run_real.py`` (which drives ONE arm). The rig, config
(``configs/robot_piper.yaml``, unified ``arms:`` block), cameras, controllers, and
plugins are all shared with the single-arm path; this entry point only adds the
dual-arm wiring. Two control modes:

  * ``--mode A`` (independent): TWO single-arm VLM stacks -- each arm gets its own
    planner + controller agent + logger (the battle-tested ``RealEpisodeRunner``,
    unchanged) -- running concurrently over one shared ``DualPiperSession``. Each arm
    has its own task (``--task-left`` / ``--task-right``, or ``arms.<side>.task`` in
    the yaml; both default to the shared ``task``).
  * ``--mode B`` (unified): ONE model, THREE images (front + both wrists), one token
    PER ARM per step (``MV_*``/``GRASP``/``RELEASE``/``DONE``/``STILL``). The dual
    planner expands the shared task into one CONCURRENT subgoal track per arm, and
    both tokens execute simultaneously (``core.runners.dual.DualEpisodeRunner``).

Safety + interruption match the single-arm path: per-arm calibrated Z floors are
required (``scripts/piper/capture_z_floor.sh --arm <side> --write``), and Ctrl+C
compiles the video(s) and homes BOTH arms.

Examples
--------
    # Unified dual-arm rollout (one model commands both arms):
    python scripts/run_real_dual.py --mode B

    # Independent per-arm rollouts with per-arm tasks:
    python scripts/run_real_dual.py --mode A \
        --task-left "put the banana on the left plate" \
        --task-right "put the cup on the right plate"

    # Dry run with mock hardware:
    python scripts/run_real_dual.py --mode B --mock-robot --mock-cameras
"""
from __future__ import annotations

import argparse
import sys
import os
import threading
from pathlib import Path
from typing import Any

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.ui.console as console
from interpreters.franka_atomic_controller import TABLE_CONTACT_Z_M
from core.config import load_secrets_env, load_yaml
from core.runners.dual import DualEpisodeRunner
from core.record.episode_logger import EpisodeLogger
from core.ui.live_view import LiveView
from core.piper.config import SIDES, arm_config
from core.piper.dual_session import ArmSessionView, DualPiperSession
from core.piper.poses import go_begin_dual
from core.prompting.prompt_loader import load_prompt_dir
from core.v0_types import EpisodeResult, V0Config
from core.launch import (
    arm_summary,
    auto_home_dual,
    dual_session_config,
    fmt_z,
    make_home_arm,
    REAL_TASK_ID,
    build_config,
    high_above_table_m,
    make_controller,
    make_runner,
    make_vlm_client,
    mem_text_len,
    recovery_empty_width_m,
    recovery_open_width_m,
    resolve_primitives_path,
    run_variant,
)
from plugins.affordance import AffordancePlugin
from plugins.assembly import build_video_ref, install_dagger
from plugins.config import PluginsConfig
from plugins.ego import EgoPlugin
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.subgoal import DualSubgoalPlanner, DualSubgoalPlannerAgent
from plugins.view_select import ViewSelectPlugin
from plugins.wrist_frame import WristFramePlugin
from core.vlm.dual_roles import DualControllerAgent


# Single-arm capabilities the unified (Mode B) controller does not offer yet. They are
# per-arm answer-protocol / decision-cadence plugins whose dual composition is future
# work; a config that enables one still runs, minus that capability.
MODE_B_UNSUPPORTED_PLUGINS = ("variable_step", "action_chunk", "mcq", "coords", "deepplan", "rotation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness dual-arm real-robot: pure-vision closed-loop manipulation"
    )
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_piper.yaml"))
    parser.add_argument(
        "--mode",
        default="B",
        choices=["A", "B"],
        help="A: two independent single-arm VLM stacks (one per arm). "
        "B: one unified model commands both arms with STILL for a waiting arm.",
    )
    parser.add_argument(
        "--begin-pose",
        default=None,
        help="Which NAMED start pose (arms.<side>.poses.<name>) both arms home to. "
        "Default: the config's begin_pose. List them with go_begin.sh --list.",
    )
    parser.add_argument("--primitives-config", default=None)
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--task", default=None, help="Override the (shared) task prompt.")
    parser.add_argument(
        "--task-left", default=None, help="Mode A: the LEFT arm's own task (default: task)."
    )
    parser.add_argument(
        "--task-right", default=None, help="Mode A: the RIGHT arm's own task (default: task)."
    )
    parser.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"))
    parser.add_argument(
        "--video-ref",
        default=None,
        help="Reference demo video to replicate. Enables plugins.video_ref for this "
        "run and overrides the config's video_ref_path.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)

    parser.add_argument("--mock-robot", action="store_true", help="Use simulated arms.")
    parser.add_argument("--mock-cameras", action="store_true", help="Use random mock cameras.")

    # Z floor: per-arm heights come from arms.<side>.z_floor_m in the config; these
    # flags apply to BOTH arms (an explicit --z-floor-m is rarely right for two arms).
    parser.add_argument("--z-floor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--z-floor-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-z-floor",
        action="store_true",
        help="Disable the Z safety floor on BOTH arms (NOT recommended on hardware).",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the live camera window (Mode B only; Mode A runs headless).",
    )
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DEBUG") == "1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    hardware = str(robot_cfg.get("hardware", "")).strip().lower()
    if hardware != "piper":
        raise ValueError(
            f"scripts/run_real_dual.py drives the dual-Piper rig; {Path(args.robot_config).name} "
            f"declares hardware {hardware!r}. Use scripts/run_real.py for single-arm hardware."
        )
    primitives_cfg = load_yaml(resolve_primitives_path(args, robot_cfg, hardware))
    prompts_cfg = load_prompt_dir(args.prompts_dir)

    # --begin-pose selects a NAMED start pose for this run (both arms use the same name;
    # arm_config resolves it into each arm's begin_joints).
    if args.begin_pose:
        robot_cfg["begin_pose"] = args.begin_pose

    # Per-arm flattened views (shared keys + that arm's block) drive the per-arm
    # controller builds in BOTH modes; Mode B additionally keeps a shared cfg.
    arm_cfgs = {side: build_config(args, arm_config(robot_cfg, side)) for side in SIDES}
    for side in SIDES:
        per_arm_task = getattr(args, f"task_{side}", None)
        if per_arm_task:
            arm_cfgs[side]["task"] = per_arm_task
        arm_cfgs[side]["hardware"] = hardware

    session = DualPiperSession(dual_session_config(arm_cfgs, args))
    exit_code = 0
    viewer = None
    try:
        session.connect()
        # BEGIN pose for BOTH arms simultaneously, BEFORE the first controller sync,
        # so both runs start from the known configuration with empty hands.
        if bool(arm_cfgs["left"].get("move_to_begin_on_init", True)):
            pose = robot_cfg.get("begin_pose", "begin_joints")
            print(f"  homing both arms to BEGIN ({pose}) ...")
            go_begin_dual(
                session.robots,
                {side: arm_cfgs[side].get("begin_joints") for side in SIDES},
                time_to_go=float(arm_cfgs["left"].get("begin_time_s", 3.0)),
                label="begin",
                open_gripper=True,
                verbose=False,  # the joint vectors are noise; the runner reports the state
            )

        if args.mode == "A":
            exit_code = _run_mode_a(args, arm_cfgs, primitives_cfg, prompts_cfg, session)
        else:
            viewer = LiveView(
                enabled=not args.no_show,
                title=f"Show-Harness dual | {arm_cfgs['left'].get('task', '')}",
            )
            exit_code = _run_mode_b(
                args, robot_cfg, arm_cfgs, primitives_cfg, prompts_cfg, session, viewer
            )

        # Reset BOTH arms to BEGIN when the rollout(s) finish. Best-effort: a reset
        # failure must not mask the rollout result.
        auto_home_dual(arm_cfgs, session)
    except KeyboardInterrupt:
        print("\n[run-dual] interrupted before the rollout loop started.")
        exit_code = 130
    finally:
        if viewer is not None:
            viewer.close()
        session.close()
    return exit_code


# -- Mode A: two independent single-arm stacks ---------------------------------------
def _run_mode_a(
    args: argparse.Namespace,
    arm_cfgs: dict[str, dict[str, Any]],
    primitives_cfg: dict[str, Any],
    prompts_cfg: dict[str, Any],
    session: DualPiperSession,
) -> int:
    """Two unchanged single-arm runners over one shared session, one thread per arm.

    Each arm's stack is byte-identical to a ``scripts/run_real.py --arm <side>`` run (its own
    VLM client, planner, controller agent, plugins, logger) -- only the session is the
    shared dual one, sliced per arm by :class:`ArmSessionView`. Ctrl+C sets the shared
    stop event, which each runner sees as a KeyboardInterrupt at its next observation,
    so both compile their videos and write summaries exactly like a single-arm interrupt.
    """
    stop_event = threading.Event()
    mode_a_plugins = PluginsConfig.from_config(arm_cfgs["left"])
    for name, what, forced in (
        ("view_select", "the guiding-view motion frame", False),
        ("video_ref", "reference-video replication", bool(args.video_ref)),
        ("dagger", "real-time human override", False),
    ):
        if mode_a_plugins.enabled(name, default=False) or forced:
            print(
                f"[run-dual] mode A ignores plugins.{name} -- {what} drives the "
                "unified dual stack (mode B) only."
            )
    runners: dict[str, Any] = {}
    for side in SIDES:
        cfg = arm_cfgs[side]
        client = make_vlm_client(args, cfg)
        vlm_cfg = cfg["vlm"]
        print(
            f"[{side}] task: {cfg['task']!r} | VLM {vlm_cfg.get('backend', '?')} "
            f"(model={vlm_cfg['model']})"
        )
        view = ArmSessionView(session, side, stop_event=stop_event)
        controller = make_controller(cfg, primitives_cfg, view, args, "piper")
        # Both controllers print interleaved from their own threads; tag them per arm
        # so the terminal shows which arm each motion line belongs to.
        controller.LOG_TAG = f"piper-{side[0].upper()}"
        controller.sync_from_robot()
        print(f"[{side}] Z floor = {fmt_z(controller.z_floor_m)}")
        logger = EpisodeLogger(
            Path(ROOT) / cfg["log_dir"],
            REAL_TASK_ID,
            variant=f"{run_variant(vlm_cfg)}-{side}",
            video_fps=float((cfg.get("v0") or {}).get("video_fps", 2.0)),
        )
        logger.write_metadata(
            {
                "task": cfg["task"],
                "arm": side,
                "robot_config": cfg,
                "primitives_config": primitives_cfg,
                "control_mode": "real_dual_independent",
                "z_floor_m": controller.z_floor_m,
                "debug": args.debug,
            }
        )
        # The Mode A live window is skipped: two runners rendering into one pygame
        # display from separate threads is not safe. Watch the per-arm videos instead.
        runners[side] = make_runner(
            cfg=cfg,
            prompts_cfg=prompts_cfg,
            client=client,
            session=view,
            controller=controller,
            logger=logger,
            debug=args.debug,
            viewer=None,
        )

    results: dict[str, EpisodeResult] = {}
    errors: dict[str, BaseException] = {}
    home = make_home_arm(arm_cfgs, session)

    def _run(side: str) -> None:
        try:
            results[side] = runners[side].run()
        except BaseException as exc:  # noqa: BLE001 - reported after join
            errors[side] = exc
        finally:
            # This arm's episode is over, but the other may still be working: park it at
            # BEGIN so it stops blocking the workspace / occluding the shared front view.
            # Best-effort -- never mask the rollout result (or a crash) above.
            try:
                home(side)
            except Exception as exc:  # noqa: BLE001
                print(f"[{side}] go-begin after the rollout failed: {exc}")

    threads = [
        threading.Thread(target=_run, args=(side,), name=f"rollout-{side}")
        for side in SIDES
    ]
    for t in threads:
        t.start()
    try:
        # Poll-join so the MAIN thread stays interruptible (Ctrl+C only lands here).
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[run-dual] Ctrl+C -- stopping both arm rollouts...")
        stop_event.set()
        for t in threads:
            t.join()

    for side, exc in errors.items():
        print(f"[{side}] rollout crashed: {exc}")
    for side, result in results.items():
        print(
            f"[{side}] success={result.success} steps={result.steps} "
            f"end={result.end_reason}\n[{side}] video: {result.video_path}"
        )
    all_ok = len(results) == len(SIDES) and all(r.success for r in results.values())
    return 0 if all_ok else 2


# -- Mode B: one unified model commands both arms -------------------------------------
def _run_mode_b(
    args: argparse.Namespace,
    robot_cfg: dict[str, Any],
    arm_cfgs: dict[str, dict[str, Any]],
    primitives_cfg: dict[str, Any],
    prompts_cfg: dict[str, Any],
    session: DualPiperSession,
    viewer: Any,
) -> int:
    cfg = build_config(args, robot_cfg)  # shared view (task, vlm, plugins, steps)
    cfg["hardware"] = "piper"
    plugins = PluginsConfig.from_config(cfg)
    # Reference-video replication: distill a demo video into an ordered operation
    # brief the planner replicates. An explicit --video-ref is itself the opt-in
    # (no yaml edit needed for a one-off run). Config errors fail HERE, before any
    # VLM cost.
    video_ref_plugin = build_video_ref(plugins, cfg, args.video_ref)

    # Multi-view action selection: the controller reports each arm's guiding view and
    # that view picks the move's motion frame per step (WRIST -> wrist, FRONT -> base).
    view_select_plugin = ViewSelectPlugin(plugins.enabled("view_select", default=False))
    if view_select_plugin.enabled and (
        str(cfg.get("motion_frame", "base")).strip().lower() == "wrist"
    ):
        # The static wrist frame rewrites the prompt's front-view guidance to
        # gripper-heading directions, but under view select a FRONT-guided move
        # executes in the base frame -- the rewritten text would misdirect it.
        raise ValueError(
            "plugins.view_select picks each move's frame from the guiding view, so the "
            "prompt must stay in the base convention; it cannot combine with "
            "motion_frame: wrist. Use motion_frame: base (or disable one of them)."
        )

    # DAGGER: real-time human keyboard override during the rollout, with the teleop
    # bindings. Keys arrive through the live-view window, so it needs the window.
    dagger_plugin = install_dagger(plugins, viewer)

    client = make_vlm_client(args, cfg)
    # Affordance dots: on stage entry the pointing role grounds each arm's contact
    # point on the front view; the runner premarks it (LEFT red / RIGHT blue) and the
    # controller prompt tells each arm to steer to its dot.
    affordance_plugin = AffordancePlugin(
        enabled=plugins.enabled("affordance", default=False),
        client=client,
        verify_rounds=int(cfg.get("affordance_verify_rounds", 1)),
        view_name="Front View",
    )
    vlm_cfg = cfg["vlm"]
    # Run header: everything the operator needs to know WHICH run this is, in one block.
    print(f"\n{_rule('dual rollout  ·  mode B (unified)')}")
    print(f"  task     {cfg['task']}")
    print(
        f"  model    {vlm_cfg.get('backend', '?')} / {vlm_cfg['model']}"
        f"  ({'CoT' if vlm_cfg.get('reasoning_cot') else 'no-CoT'})"
    )
    motion_desc = (
        "view-select (WRIST->wrist, FRONT->base)"
        if view_select_plugin.enabled
        else f"{cfg.get('motion_frame', 'base')} frame"
    )
    print(
        f"  motion   {motion_desc}  ·  step "
        f"{float(cfg.get('fine_step_m', 0.02)) * 100:g} cm  ·  max {int(cfg['max_steps'])} steps"
    )
    _print_plugins(plugins, cfg, video_ref_plugin)
    if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
        client.health_check(
            wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
            poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
        )

    # Analyze the reference demo BEFORE the arms move: an unusable video aborts here,
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

    controllers: dict[str, Any] = {}
    for side in SIDES:
        controller = make_controller(arm_cfgs[side], primitives_cfg, ArmSessionView(session, side), args, "piper")
        # Both arms execute simultaneously and print interleaved; per-arm tags keep
        # any warning readable ([piper-L] / [piper-R]).
        controller.LOG_TAG = f"piper-{side[0].upper()}"
        # The RUNNER owns the console from here: one readable block per step. Silence the
        # controller's own chatter -- the sync/backend coordinate dumps, and two arms
        # interleaving "MV_DOWN d_pos=... target_pos=..." from their own threads, which
        # buried the reasoning. Warnings (reach clamp, divergence, dropped gripper, a
        # start pose near a joint limit) print regardless of this flag.
        controller.verbose = False
        controller.sync_from_robot()
        print(f"  {side.upper():<5}    {arm_summary(controller)}")
        controllers[side] = controller

    common_context = prompts_cfg["common_context"]
    cot_mode = bool(vlm_cfg.get("reasoning_cot"))
    # The dual controller prompt is written in the neutral base (AgentView-centric)
    # convention. Two prompt-only transforms adapt it to this rig, in order: the
    # optional wrist motion frame rewrites the front-view directions to follow each
    # gripper's heading (motion_frame: wrist; default base leaves it untouched), then
    # the egocentric FWD/BACK inversion -- which applies to the WRIST-guidance section
    # only (the front scene camera images the arms from the top, matching the base
    # depth convention; the eye-in-hand wrist inverts it). Under view_select the prompt
    # stays in that base convention (the guard above pins motion_frame: base): base-frame
    # execution realizes rule B exactly, and the per-step wrist frame realizes the
    # (ego-adapted) rule A exactly -- no further prompt rewrites are needed.
    dual_prompt = WristFramePlugin(
        str(cfg.get("motion_frame", "base")).strip().lower() == "wrist"
    ).apply(prompts_cfg["controller_dual_prompt"])
    dual_prompt = EgoPlugin(bool(cfg.get("is_ego", False))).apply(dual_prompt)
    planner = DualSubgoalPlanner(
        DualSubgoalPlannerAgent(
            client=client,
            common_context=common_context,
            # The reference-demo brief rides on the agent, so the initial plan AND
            # every completion-check replan replicate the same demonstration.
            video_ref_block=video_ref_plugin.render_prompt(),
        )
    )
    mem_text_plugin = MemTextPlugin(plugins.enabled("mem_text"), max_recent=mem_text_len(cfg))
    proprio_plugin = ProprioceptionPlugin(
        plugins.enabled("proprioception"),
        high_above_table_m=high_above_table_m(cfg),
        fine_step_m=controllers["left"].step_m,
        coarse_step_m=None,
    )
    recovery_tools = {
        side: RecoveryPlugin(
            enabled=plugins.enabled("recovery"),
            empty_width_m=recovery_empty_width_m(arm_cfgs[side]),
            open_width_m=recovery_open_width_m(arm_cfgs[side]),
        )
        for side in SIDES
    }
    agent = DualControllerAgent(
        client=client,
        prompt_template=dual_prompt,
        common_context=common_context,
        cot_mode=cot_mode,
        proprio_plugin=proprio_plugin,
        mem_text_plugin=mem_text_plugin,
        view_select_plugin=view_select_plugin,
        affordance_plugin=affordance_plugin,
        table_heights={
            side: (
                controllers[side].z_floor_m
                if controllers[side].z_floor_m is not None
                else TABLE_CONTACT_Z_M
            )
            for side in SIDES
        },
        # Rig fact for the CLOSED gripper line: widths under this mean an empty close.
        empty_width_m=recovery_empty_width_m(cfg),
    )

    logger = EpisodeLogger(
        Path(ROOT) / cfg["log_dir"],
        REAL_TASK_ID,
        variant=f"{run_variant(vlm_cfg)}-dual",
        video_fps=float((cfg.get("v0") or {}).get("video_fps", 2.0)),
    )
    logger.write_metadata(
        {
            "task": cfg["task"],
            "robot_config": cfg,
            "primitives_config": primitives_cfg,
            "prompts_dir": str(Path(args.prompts_dir).resolve()),
            "control_mode": "real_dual_unified",
            "z_floor_m": {side: controllers[side].z_floor_m for side in SIDES},
            "debug": args.debug,
            **({"video_ref": video_ref_plugin.metadata()} if video_ref_plugin.enabled else {}),
            **({"affordance": affordance_plugin.metadata()} if affordance_plugin.enabled else {}),
        }
    )
    runner = DualEpisodeRunner(
        session=session,
        controllers=controllers,
        planner=planner,
        controller_agent=agent,
        logger=logger,
        config=V0Config.from_dict(cfg.get("v0", {})),
        task=str(cfg["task"]),
        max_steps=int(cfg["max_steps"]),
        loop_period_s=float(cfg.get("loop_period_s", 0.0)),
        debug=args.debug,
        recovery_tools=recovery_tools,
        recent_moves_max=mem_text_plugin.max_recent,
        viewer=viewer,
        # An arm that finishes its track while the other still works is parked back at
        # BEGIN, so it stops blocking the workspace and occluding the shared front view.
        home_arm=make_home_arm(arm_cfgs, session),
        view_select_plugin=view_select_plugin,
        affordance_plugin=affordance_plugin,
        dagger_plugin=dagger_plugin,
    )
    if viewer is not None:
        # Continuous live feed: a render thread streams fresh camera frames (~12 Hz)
        # for the whole rollout -- the window stays live while the VLM thinks and
        # while the arms move; the runner only posts status overlays. The affordance
        # adapter overlays the active dots on the front frame (pass-through when the
        # plugin is off / has no dots), so the live window shows exactly the visual
        # prompt the controller is being steered by.
        viewer.start_stream(
            lambda: affordance_plugin.annotate_frames(session.get_camera_frames())
        )
    result = runner.run()
    print(f"\n{_rule('result')}")
    print(f"  {'SUCCESS' if result.success else 'FAILED'}  ({result.end_reason})")
    print(f"  steps    {result.steps}/{int(cfg['max_steps'])}")
    print(f"  video    {result.video_path}\n")
    return 0 if result.success else 2


# -- shared helpers --------------------------------------------------------------------
def _print_plugins(plugins: PluginsConfig, cfg: dict[str, Any], video_ref_plugin: Any) -> None:
    """The run header's plugins block: one line per ENABLED plugin with its live
    parameters. Disabled plugins are hidden entirely; an enabled plugin mode B cannot
    compose yet stays visible (the operator should know it was asked for) with a
    dim "ignored" note instead of a separate warning wall."""
    def cm(key: str, default: float) -> str:
        return f"{float(cfg.get(key, default)) * 100:g} cm"

    details = {
        "subgoal": "planner-expanded per-arm stage tracks",
        "proprioception": f"high-above {float(high_above_table_m(cfg)):.2f} m",
        "recovery": f"empty close < {float(recovery_empty_width_m(cfg)) * 1000:g} mm",
        "variable_step": (
            f"fine {cm('fine_step_m', 0.02)} · coarse {cm('coarse_step_m', 0.04)}"
            f" · up {cm('up_step_m', 0.05)}"
        ),
        "action_chunk": f"{int(cfg.get('action_chunk_step_num', 3))} moves per VLM call",
        "mem_text": f"last {mem_text_len(cfg)} moves in the prompt",
        "smooth": (
            f"{int(cfg.get('smooth_substeps', 20))} substeps @ "
            f"{float(cfg.get('smooth_dt_s', 0.02)) * 1000:g} ms"
        ),
        "view_select": "WRIST-guided -> wrist frame · FRONT-guided -> base frame",
        "affordance": (
            f"front dot per stage + per-step wrist tracking (LEFT red · RIGHT blue) · "
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
        "view_select", "affordance", "video_ref", "dagger",
    )
    label = "plugins"
    for name in known:
        enabled = plugins.enabled(name, default=False) or (
            name == "video_ref" and video_ref_plugin.enabled
        )
        if not enabled:
            continue
        note = _dim("  (ignored in mode B)") if name in MODE_B_UNSUPPORTED_PLUGINS else ""
        detail = details.get(name, "")
        print(f"  {label:<8} {name:<15}{_dim(detail)}{note}")
        label = ""  # only the first line carries the section label


def _dim(text: str) -> str:
    return console.dim(text)


def _rule(title: str, width: int = 68) -> str:
    """A dim section rule for the run header / footer (plain text when piped)."""
    return console.rule(title, width)


if __name__ == "__main__":
    raise SystemExit(main())
