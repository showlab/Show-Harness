"""Dual-arm MVTOKEN real-robot entrypoint (planner-free / stage-free, three schemes).

The dual-arm counterpart of ``scripts/run_real_mvtoken.py``. It reuses ``scripts/run_real_dual.py``'s hardware
wiring verbatim (dual session, per-arm controllers behind ``ArmSessionView``, Z floors, BEGIN
pose, auto-home) and swaps the subgoal stack for the bare-token dual policy: three views in,
one atomic token per arm out, executed simultaneously.

ONE scheme flag selects BOTH the prompt and the request shape -- they are two halves of the
same contract, and a LoRA trained on one cannot be served under another:

  --twice  prompts/<v>/dual_mvtoken_twice.txt   2 calls/step; the right call does NOT see the
                                                left token. 2 image encodings.
  --once   prompts/<v>/dual_mvtoken_once.txt    1 call/step; the model answers "<left> <right>".
  --chain  prompts/<v>/dual_mvtoken_chain.txt   1 image encoding, 2 answers (the right one sees
           + dual_mvtoken_chain_right.txt       the left). vLLM's prefix cache serves turn 2.

The served LoRA must be the one trained on the SAME scheme:
  --twice -> dual_cloth_v4_twice   --once -> dual_cloth_v4_once   --chain -> dual_cloth_v4_chain

Interactive collection (same contract as scripts/run_real_mvtoken.py): on a TTY the operator stages the
scene and presses Enter to start each rollout -- both arms are parked at BEGIN while that
happens, so the workspace is clear. During a rollout the restart key ('t' by default) ends it
and returns to that gate, 'q' finishes. A natural end (DONE / max_steps) also returns to the
gate, and every rollout gets its own timestamped run dir. Non-TTY runs (mock / CI) or
--single-rollout do exactly one rollout with no gate.

Prereq (Terminal 1): a vLLM server with the dual LoRA(s) attached
(scripts/serve_vlm.sh; give LORA a comma-separated name=path per adapter).

Usage (Terminal 2):
  python scripts/run_real_dual_mvtoken.py --version v4 --once \\
      --model <your-dual-adapter> --task "fold the black t-shirt"
"""
from __future__ import annotations

import argparse
import os
import sys
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
from core.piper.config import SIDES, arm_config
from core.piper.dual_session import ArmSessionView, DualPiperSession
from core.piper.poses import go_begin_dual
from core.runners.dual_mvtoken import DualMvTokenRunner
from core.teleop.keys import RolloutKeyWatcher
from plugins.assembly import build_auto_release, install_dagger
from plugins.config import PluginsConfig
from core.vlm.dual_mvtoken_roles import SCHEMES, DualMvTokenController

# Reuse run_real / run_real_dual wiring verbatim so safety (Z floor), mock support, camera
# handling and the BEGIN/home behaviour stay identical across the dual entrypoints.
from core.launch import (
    REAL_TASK_ID,
    build_config,
    make_controller,
    make_vlm_client,
    resolve_primitives_path,
    run_variant,
)
from core.launch import arm_summary, auto_home_dual, dual_session_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness dual-arm MVTOKEN: bare-token closed-loop, three schemes"
    )
    parser.add_argument(
        "--robot-config",
        default=str(ROOT / "configs" / "robot_piper_ft.yaml"),
        help="The MVTOKEN Piper config -- the same one the single-arm runner uses; its `arms:` "
        "block is what the dual runner flattens per side. NOT robot_piper.yaml: that is the "
        "subgoal/Gemini stack with no fine-tuned backends.",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Prompt version subfolder under prompts/ (e.g. v4). MUST match the version the "
        "served LoRA was trained on.",
    )
    scheme = parser.add_mutually_exclusive_group(required=True)
    scheme.add_argument(
        "--twice", action="store_const", const="twice", dest="scheme",
        help="TWO VLM calls per step (the right call does not see the left token).",
    )
    scheme.add_argument(
        "--once", action="store_const", const="once", dest="scheme",
        help="ONE call per step; the model answers '<left> <right>'.",
    )
    scheme.add_argument(
        "--chain", action="store_const", const="chain", dest="scheme",
        help="ONE image encoding, TWO answers (the right one sees the left).",
    )
    parser.add_argument(
        "--begin-pose",
        default=None,
        help="Which NAMED start pose (arms.<side>.poses.<name>) both arms home to. "
        "Default: the config's begin_pose.",
    )
    parser.add_argument("--primitives-config", default=None)
    parser.add_argument("--task", default=None, help="Override the (shared) task prompt.")
    parser.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--prompt-log-every", type=int, default=20,
        help="Save the exact VLM prompt every N steps (0 disables).",
    )

    # Interactive multi-rollout collection, identical to scripts/run_real_mvtoken.py: between rollouts
    # the operator sets up the scene and presses Enter; during one, the restart key ends it and
    # returns to that gate, 'q' finishes. Auto-disabled when stdin is not a TTY (mock / CI run
    # exactly one rollout); force a single rollout on a TTY with --single-rollout.
    parser.add_argument(
        "--restart-key", default="t",
        help="Key to end the current rollout and start a new one (interactive TTY only).",
    )
    parser.add_argument(
        "--single-rollout", action="store_true",
        help="Run exactly one rollout and exit (no interactive Enter gate / restart watcher).",
    )

    parser.add_argument("--mock-robot", action="store_true", help="Use simulated arms.")
    parser.add_argument("--mock-cameras", action="store_true", help="Use random mock cameras.")

    # Z floor: per-arm heights come from arms.<side>.z_floor_m; these flags apply to BOTH arms.
    parser.add_argument("--z-floor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--z-floor-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-z-floor", action="store_true",
        help="Disable the Z safety floor on BOTH arms (NOT recommended on hardware).",
    )

    parser.add_argument("--no-show", action="store_true", help="Do not open the live window.")
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DEBUG") == "1")
    return parser.parse_args()


def _load_prompt(version: str, filename: str) -> str:
    path = ROOT / "prompts" / version / filename
    if not path.is_file():
        raise SystemExit(
            f"Prompt template not found: {path}\n"
            f"  The dual prompts live in prompts/<version>/dual_mvtoken_<scheme>.txt "
            f"(v4 and up)."
        )
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    hardware = str(robot_cfg.get("hardware", "")).strip().lower()
    if hardware != "piper":
        raise SystemExit(
            f"scripts/run_real_dual_mvtoken.py drives the dual-Piper rig; "
            f"{Path(args.robot_config).name} declares hardware {hardware!r}."
        )
    if args.scheme not in SCHEMES:  # argparse already guarantees this; keep the invariant local
        raise SystemExit(f"--scheme must be one of {SCHEMES}")

    primitives_cfg = load_yaml(resolve_primitives_path(args, robot_cfg, hardware))

    if args.begin_pose:
        robot_cfg["begin_pose"] = args.begin_pose

    # Per-arm flattened views (shared keys + that arm's block) drive the per-arm controllers;
    # the shared view carries task / vlm / steps.
    arm_cfgs = {side: build_config(args, arm_config(robot_cfg, side)) for side in SIDES}
    for side in SIDES:
        arm_cfgs[side]["hardware"] = hardware
    cfg = build_config(args, robot_cfg)
    cfg["hardware"] = hardware

    # The scheme's prompt(s). --chain additionally needs the text-only follow-up turn.
    prompt_template = _load_prompt(args.version, f"dual_mvtoken_{args.scheme}.txt")
    followup_template = (
        _load_prompt(args.version, "dual_mvtoken_chain_right.txt")
        if args.scheme == "chain"
        else None
    )

    client = make_vlm_client(args, cfg)
    vlm_cfg = cfg["vlm"]
    # Run header: everything the operator needs to know WHICH run this is, in one block --
    # the same shape scripts/run_real.py / scripts/run_real_dual.py print, so all four entrypoints read alike.
    print(f"\n{console.rule(f'dual rollout  ·  mvtoken [{args.scheme}] (piper, stage-free)')}")
    print(f"  task     {cfg['task']}")
    print(
        f"  model    {vlm_cfg.get('backend', '?')} / {vlm_cfg['model']}"
        f"  @ {vlm_cfg['base_url']}"
    )
    print(f"  prompt   {args.version} / dual_mvtoken_{args.scheme}.txt")
    print(
        f"  motion   step {float(cfg.get('fine_step_m', 0.02)) * 100:g} cm  ·  "
        f"max {int(cfg['max_steps'])} steps"
    )
    print(f"  begin    {robot_cfg.get('begin_pose', 'begin_joints (unnamed)')}")
    # The /models readiness poll exists to wait for a LOCAL vLLM to cold-start; hosted
    # providers are always-on and some do not expose /models at all, so polling would block
    # for the full startup_wait_s. Same split as scripts/run_real.py.
    if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
        print(
            console.dim(
                f"  waiting  VLM endpoint {vlm_cfg['base_url']}/models "
                f"(up to {float(vlm_cfg.get('startup_wait_s', 0)):.0f}s) ..."
            )
        )
        client.health_check(
            wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
            poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
        )
        print("  ready    VLM endpoint is up.")
    else:
        print(
            f"  ready    backend {vlm_cfg.get('backend')} is hosted "
            + console.dim("(skipping the /models readiness poll)")
        )
    agent = DualMvTokenController(
        client=client,
        prompt_template=prompt_template,
        scheme=args.scheme,
        followup_template=followup_template,
    )

    session = DualPiperSession(dual_session_config(arm_cfgs, args))
    viewer = LiveView(
        enabled=not args.no_show,
        title=f"Show-Harness dual-mvtoken [{args.scheme}] | {cfg.get('task', '')}",
    )
    exit_code = 0
    try:
        session.connect()
        # BEGIN pose for BOTH arms BEFORE the first controller sync, so the rollout starts
        # from the known configuration with empty hands.
        if bool(arm_cfgs["left"].get("move_to_begin_on_init", True)):
            go_begin_dual(
                session.robots,
                {side: arm_cfgs[side].get("begin_joints") for side in SIDES},
                time_to_go=float(arm_cfgs["left"].get("begin_time_s", 3.0)),
                label="begin",
                open_gripper=True,
            )

        controllers: dict[str, Any] = {}
        for side in SIDES:
            controller = make_controller(
                arm_cfgs[side], primitives_cfg, ArmSessionView(session, side), args, "piper"
            )
            # Both arms execute simultaneously and print interleaved; per-arm tags keep the
            # motion log readable ([piper-L] / [piper-R]).
            controller.LOG_TAG = f"piper-{side[0].upper()}"
            # The RUNNER owns the console from here: one readable block per step. Silence the
            # controller's per-token coordinate chatter (two arms interleaving it from their
            # own threads buried the reasoning); warnings still print regardless.
            controller.verbose = False
            controller.sync_from_robot()
            print(f"  {side.upper():<5}    {arm_summary(controller)}")
            controllers[side] = controller

        # Auto-release safety rule (plugins.auto_release): reopen a closed gripper whose
        # measured width collapses below that arm's empty_width_m (it is holding nothing).
        # Same rule as scripts/run_real_mvtoken.py, one plugin per arm.
        plugins_cfg = PluginsConfig.from_config(cfg)
        auto_release = {
            side: build_auto_release(plugins_cfg, arm_cfgs[side]) for side in SIDES
        }
        if any(tool.enabled for tool in auto_release.values()):
            thresholds = ", ".join(
                f"{side} < {auto_release[side].empty_width_m * 1000:g} mm" for side in SIDES
            )
            print(
                "  plugins  auto_release   "
                + console.dim(f"reopen the gripper when its closed width is {thresholds}")
            )

        # DAGGER (plugins.dagger): live human keyboard override during the rollout, using the
        # SAME dual key layout as teleop collection. Keys arrive on the live-view window's
        # STREAM thread, which DualPiperSession can feed (it has get_camera_frames), so the
        # only prerequisite is the window itself. Same wiring as scripts/run_real_dual.py.
        dagger_plugin = install_dagger(plugins_cfg, viewer)
        if dagger_plugin.enabled:
            print(
                "  plugins  dagger         "
                + console.dim(
                    "teleop keys override the model live · CLICK the live-view window to arm"
                )
            )

        # The window must actually stream for DAGGER keys to be pumped (and it keeps the feed
        # live while the VLM thinks). scripts/run_real_mvtoken.py does the same.
        if viewer.enabled and hasattr(session, "get_camera_frames"):
            viewer.start_stream(session.get_camera_frames)

        # Interactive multi-rollout collection (same contract as scripts/run_real_mvtoken.py): each
        # rollout is a fresh timestamped run dir, the operator presses Enter between them, the
        # restart key ends the current one, 'q' finishes. The watcher raises KeyboardInterrupt
        # in this thread, which DualMvTokenRunner already handles, so the runner is untouched.
        interactive = sys.stdin.isatty() and not args.single_rollout
        watcher = RolloutKeyWatcher(restart_key=args.restart_key) if interactive else None
        if watcher is not None:
            watcher.start()
            print(
                "  keys     "
                + console.dim(
                    f"Enter = start rollout · '{watcher.restart_key}' = restart · 'q' = quit"
                )
            )

        result = None
        try:
            while True:
                # Wait for the operator to set up the scene and press Enter before EVERY
                # rollout (including the first); 'q' finishes. Both arms are already parked at
                # BEGIN at this point -- on the first pass by the go_begin_dual above, later by
                # the reset at the end of the previous rollout -- so the workspace is clear
                # while the scene is being staged. No-op for non-TTY / --single-rollout.
                if watcher is not None and not watcher.wait_for_enter(
                    "[rollout] set up the scene, then press Enter to start the rollout "
                    "(or 'q' to quit) ..."
                ):
                    print("[rollout] 'q' -> finishing.")
                    break

                # Re-sync both controllers to the live pose before each rollout.
                for side in SIDES:
                    controllers[side].sync_from_robot()

                logger = EpisodeLogger(
                    Path(ROOT) / cfg["log_dir"],
                    REAL_TASK_ID,
                    variant=f"{run_variant(vlm_cfg)}-dual_mvtoken_{args.scheme}",
                )
                logger.write_metadata(
                    {
                        "task": cfg["task"],
                        "robot_config": cfg,
                        "primitives_config": primitives_cfg,
                        "control_mode": "real_dual_mvtoken",
                        "scheme": args.scheme,
                        "prompt_version": args.version,
                        "prompt_file": str(
                            (
                                ROOT / "prompts" / args.version
                                / f"dual_mvtoken_{args.scheme}.txt"
                            ).resolve()
                        ),
                        "z_floor_m": {side: controllers[side].z_floor_m for side in SIDES},
                        "debug": args.debug,
                    }
                )

                print(f"\n{console.rule(f'rollout {logger.run_dir.name}')}")
                print(console.dim(f"  dir      {logger.run_dir}"))
                runner = DualMvTokenRunner(
                    session=session,
                    controllers=controllers,
                    agent=agent,
                    logger=logger,
                    task=str(cfg["task"]),
                    max_steps=int(cfg["max_steps"]),
                    loop_period_s=float(cfg.get("loop_period_s", 0.0)),
                    video_fps=float(cfg.get("video_fps", 10.0)),
                    debug=args.debug,
                    viewer=viewer,
                    prompt_log_every=int(args.prompt_log_every),
                    auto_release=auto_release,
                    dagger_plugin=dagger_plugin,
                )
                if watcher is not None:
                    watcher.arm()
                try:
                    result = runner.run()
                finally:
                    if watcher is not None:
                        watcher.disarm()

                print(f"\n{console.rule('result')}")
                verdict = (
                    console.c(console.GREEN, "DONE")
                    if result.end_reason == "done"
                    else "ENDED"
                )
                print(f"  {verdict}  ({result.end_reason})")
                print(f"  steps    {result.steps}/{int(cfg['max_steps'])}")
                print(f"  run dir  {result.run_dir}")
                print(f"  video    {result.video_path}\n")

                # Reset BOTH arms to BEGIN so the next rollout starts from the same known
                # configuration with empty hands (and the workspace is clear while the operator
                # re-stages the scene). Best-effort: a reset failure must not mask the result.
                auto_home_dual(arm_cfgs, session)

                if watcher is None:
                    break  # single-rollout / non-TTY
                if watcher.quit.is_set():
                    print("[rollout] 'q' -> finishing.")
                    break
                if result.end_reason == "interrupted" and not watcher.restart.is_set():
                    break  # Ctrl+C (not our restart/quit key) -> stop
                if watcher.restart.is_set():
                    print("[rollout] restart key -> ending this rollout.")
                # Loop back to the Enter-confirm gate at the top before the next rollout.
        finally:
            if watcher is not None:
                watcher.shutdown()
                watcher.join(timeout=1.0)  # let it restore the terminal (cbreak -> normal)

        # Exit status reflects the LAST rollout (unchanged single-rollout semantics).
        exit_code = 0 if (result is not None and result.end_reason == "done") else 2
    except KeyboardInterrupt:
        print("\n[run-real-dual-mvtoken] interrupted before the rollout loop started.")
        exit_code = 130
    finally:
        viewer.close()
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
