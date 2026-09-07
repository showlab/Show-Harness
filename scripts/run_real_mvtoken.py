#!/usr/bin/env python3
"""Show-Harness real-robot deployment for the MVTOKEN_v0 Qwen LoRA (planner-free / stage-free).

Sibling of ``scripts/run_real.py``: same physical stack (a ``FrankaSession`` for RealSense cameras
+ NUC pose, a ``FrankaAtomicController`` mapping atomic tokens -> Cartesian-impedance
setpoints with a Z safety floor, an ``EpisodeLogger`` + ``LiveView``), but the brain is the
Qwen ``MVTOKEN_v0`` LoRA served by vLLM instead of the multi-role subgoal pipeline.

Every step the model is given the task + current gripper state + recent MV_* moves and returns
ONE of nine atomic tokens (MV_*, GRASP, RELEASE, DONE). DONE is the terminal token: the rollout
ends when it is emitted (otherwise it runs to ``max_steps`` or Ctrl+C).

The prompt is stage-free (lite): task / gripper / recent moves only.

Start the server first (Terminal 1) -- download and serve a released adapter on :8000.
The default backend (`qwen3_5_2b`) asks the server for the adapter registered under
the name qwen3_5_2b_showharness_ft, so keep the LORA name below as it is:
    ADAPTER=qwen3_5_2b WITH_BASE=1 bash scripts/model/download_vlm_model.sh
    MODEL=Qwen/Qwen3.5-2B \\
      LORA=qwen3_5_2b_showharness_ft=models/Show-Harness-VLMs/qwen3_5_2b \\
      FAMILY=qwen3_5 bash scripts/serve_vlm.sh

--version (e.g. v1) selects the prompt folder prompts/<version>/ — keep it aligned with the
version the served LoRA was trained on (same convention as the training-data converter).

Interactive collection: on a TTY the operator stages the scene and presses Enter to start each
rollout, then during it presses the restart key ('t' by default) to end the current rollout and
go back to that gate (a fresh timestamped rollout dir each time), or 'q' to finish. A natural
end (DONE / max_steps) also returns to the gate; Ctrl+C or 'q' stops. The same gate and keys
drive scripts/run_real_dual_mvtoken.py -- both share core.teleop.keys.RolloutKeyWatcher.
Every 20 steps (``--prompt-log-every``) the exact VLM prompt is dumped to the rollout's
controller_prompts/. Non-TTY runs (mock / CI) or --single-rollout do exactly one rollout.

Then (Terminal 2):
    # Real robot + real cameras
    python scripts/run_real_mvtoken.py --version v1

    # Dry run with mock hardware (no NUC / cameras needed), 5 steps
    python scripts/run_real_mvtoken.py --version v1 --mock-robot --mock-cameras --max-steps 5 --no-show
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.ui.console as console
from core.config import load_secrets_env, load_yaml
from core.record.episode_logger import EpisodeLogger
from core.ui.live_view import LiveView
from core.runners.mvtoken import MvTokenRunner
from core.teleop.keys import RolloutKeyWatcher
from plugins.assembly import build_auto_release, install_dagger
from plugins.config import PluginsConfig
from core.vlm.mvtoken_roles import MvTokenController

# Reuse run_real's config + hardware wiring verbatim so safety (Z floor), mock support, and
# camera/NUC handling stay identical between the two entrypoints.
from core.launch import (
    REAL_TASK_ID,
    build_config,
    make_controller,
    make_session,
    make_vlm_client,
    move_to_begin_on_init,
    resolve_hardware,
    resolve_primitives_path,
    resolve_z_floor_name,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness real-robot: MVTOKEN_v0 atomic-token policy (no planner/stage)"
    )
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_franka_ft.yaml"))
    parser.add_argument(
        "--primitives-config", default=None,
        help="Override the primitives yaml. Default: the robot config's primitives_config, "
        "else the hardware default (primitives_franka.yaml for franka, primitives_piper.yaml for piper).",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Prompt version subfolder under prompts/ (v3 unified, v4 per-embodiment), matching the "
        "training-data converter (rollout_to_llamafactory.py --version). The template is "
        "read from prompts/<version>/ (mvtoken_generator_lite.txt, or the --franka/--piper "
        "embodiment variant). Keep it aligned with the served LoRA.",
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--franka", action="store_const", const="franka", dest="embodiment_view",
        help="Lite mode: use the Franka (exocentric) prompt prompts/<version>/franka_mvtoken_lite.txt.",
    )
    view_group.add_argument(
        "--piper", action="store_const", const="piper", dest="embodiment_view",
        help="Lite mode: use the Piper (egocentric) prompt prompts/<version>/piper_mvtoken_lite.txt. "
        "MUST match the prompt the served LoRA was trained on.",
    )
    parser.add_argument("--task", default=None, help="Override the task prompt.")
    parser.add_argument("--gripper-color", default=None)
    parser.add_argument("--step-m", type=float, default=None, dest="fine_step_m")
    parser.add_argument(
        "--vlm-backend",
        default=os.environ.get("VLM_BACKEND"),
        help="vlm_backends profile for the action policy "
        "(default: the config's vlm_backend).",
    )
    parser.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"),
        help="Override the OpenAI-compatible vLLM base URL (default: the profile's).",
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)

    # Interactive multi-rollout collection: while a rollout runs, press the restart key
    # (default 't') to end it and start a fresh rollout, or 'q' to finish. Auto-disabled when
    # stdin is not a TTY (mock / CI do a single rollout); force one with --single-rollout.
    parser.add_argument(
        "--restart-key", default="t",
        help="Key to end the current rollout and start a new one (interactive TTY only).",
    )
    parser.add_argument(
        "--single-rollout", action="store_true",
        help="Run exactly one rollout and exit (no interactive restart watcher).",
    )
    parser.add_argument(
        "--prompt-log-every", type=int, default=20,
        help="Save the exact VLM prompt every N steps (0 disables).",
    )

    # Hardware (override config / use mock hardware for a dry run).
    parser.add_argument("--mock-robot", action="store_true")
    parser.add_argument("--mock-cameras", action="store_true")
    parser.add_argument("--nuc-ip", default=None)
    parser.add_argument("--nuc-port", type=int, default=None)
    parser.add_argument("--no-impedance", action="store_true")

    # Z safety floor (ON by default at the calibrated table-contact height).
    parser.add_argument("--z-floor", action="store_true")
    parser.add_argument("--z-floor-m", type=float, default=None)
    parser.add_argument("--no-z-floor", action="store_true")
    parser.add_argument(
        "--z-floor-name",
        default=None,
        help="Which NAMED Z floor (z_floors.<name> in the robot config) this run uses. "
        "Default: the config's z_floor_name. Same semantics as scripts/run_real.py.",
    )

    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG") == "1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    hardware = resolve_hardware(args, robot_cfg)
    primitives_cfg = load_yaml(resolve_primitives_path(args, robot_cfg, hardware))
    cfg = build_config(args, robot_cfg)
    cfg["hardware"] = hardware
    # Named Z floors (z_floors: + z_floor_name / --z-floor-name), same resolution
    # as scripts/run_real.py: the selected floor lands in cfg["z_floor_m"] before the
    # controller build. Configs without a z_floors block are untouched.
    resolve_z_floor_name(cfg, args)
    if args.embodiment_view:
        # --franka / --piper select the embodiment-specific lite prompt under prompts/<version>/.
        prompt_name = f"{args.embodiment_view}_mvtoken_lite.txt"
    else:
        prompt_name = "mvtoken_generator_lite.txt"
    prompt_path = ROOT / "prompts" / args.version / prompt_name
    if not prompt_path.is_file():
        # v4 has split franka/piper prompts (needs --franka/--piper); v0-v3 has one unified
        # prompt (must NOT get an embodiment flag). Point at whichever the caller missed.
        hint = (
            f"drop --{args.embodiment_view} ({args.version} has one unified prompt)"
            if args.embodiment_view
            else f"add --franka or --piper ({args.version} has split prompts)"
        )
        raise SystemExit(f"Prompt template not found: {prompt_path}  ({hint})")
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    client = make_vlm_client(args, cfg)

    vlm_cfg = cfg["vlm"]
    # Run header: everything the operator needs to know WHICH run this is, in one
    # block (the subgoal entrypoints' convention).
    print(f"\n{console.rule(f'rollout  ·  mvtoken ({hardware}, stage-free)')}")
    print(f"  task     {cfg['task']}")
    print(
        f"  model    {vlm_cfg.get('backend', '?')} / {vlm_cfg['model']}"
        f"  @ {vlm_cfg['base_url']}"
    )
    print(f"  prompt   {args.version} / {prompt_path.name}")
    print(
        f"  motion   step {float(cfg.get('fine_step_m', 0.02)) * 100:g} cm  ·  "
        f"max {int(cfg['max_steps'])} steps"
    )
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

    session = make_session(cfg, args, hardware)
    viewer = LiveView(enabled=not args.no_show, title=f"Show-Harness MVTOKEN | {cfg['task']}")
    exit_code = 0
    try:
        session.connect()
        controller = make_controller(cfg, primitives_cfg, session, args, hardware)
        controller.sync_from_robot()
        if controller.z_floor_m is not None:
            floor_name = f" [{cfg['_z_floor_name']}]" if cfg.get("_z_floor_name") else ""
            print(
                f"  safety   z-floor {controller.z_floor_m:.4f} m{floor_name} "
                f"{console.dim('(downward motion below this height is blocked)')}"
            )
        else:
            print(
                "  safety   "
                + console.c(console.YELLOW, "z-floor DISABLED -- no software descent limit")
            )
        # The RUNNER owns the console from here: one readable block per step (the
        # subgoal runners' convention). Silence the controller's per-token coordinate
        # chatter; warnings (reach clamp, dropped gripper, ...) print regardless.
        controller.verbose = False

        # Auto-release safety rule (plugins.auto_release): reopen a closed gripper whose
        # measured width collapses below empty_width_m (it is holding nothing). On by
        # default; disable with `plugins: {auto_release: false}` in robot_franka_ft.yaml.
        # Built once and shared by every rollout.
        plugins_cfg = PluginsConfig.from_config(cfg)
        auto_release = build_auto_release(plugins_cfg, cfg)
        if auto_release.enabled:
            print(
                "  plugins  auto_release   "
                + console.dim(
                    f"reopen the gripper when its closed width < "
                    f"{auto_release.empty_width_m * 1000:g} mm"
                )
            )

        # DAGGER (plugins.dagger): live human keyboard override during the rollout, using the
        # teleop bindings. Keys arrive on the live-view window's STREAM thread, so it needs
        # BOTH the window and a session that can feed it -- the Franka session can, the
        # single-Piper one cannot (no get_camera_frames). Same guards as scripts/run_real.py.
        dagger_plugin = install_dagger(
            plugins_cfg,
            viewer,
            single=True,
            session=session,
            hardware=hardware,
            notify=lambda msg: print("  plugins  " + console.c(console.YELLOW, msg)),
        )
        if dagger_plugin.enabled:
            print(
                "  plugins  dagger         "
                + console.dim(
                    "teleop keys override the model live · CLICK the live-view window to arm"
                )
            )

        # Continuous live feed (when the session can supply frames): the render
        # thread streams all views at ~12 Hz across every rollout -- the window stays
        # live while the VLM thinks and while the arm moves; the runner only posts
        # status overlays via show_single.
        if viewer.enabled and hasattr(session, "get_camera_frames"):
            viewer.start_stream(session.get_camera_frames)

        # Interactive multi-rollout collection: each rollout is a fresh timestamped run dir
        # (standard EpisodeLogger layout). A background watcher lets the operator press the
        # restart key to end the current rollout and start a new one, or 'q' to finish; it
        # works by raising KeyboardInterrupt in this thread, which MvTokenRunner already
        # handles, so the runner itself needs no changes. Disabled for non-TTY / --single.
        interactive = sys.stdin.isatty() and not args.single_rollout
        watcher = RolloutKeyWatcher(restart_key=args.restart_key) if interactive else None
        if watcher is not None:
            watcher.start()
            print(
                "  keys     "
                + console.dim(
                    f"Enter = start rollout · '{watcher.restart_key}' = restart · "
                    "'q' = quit"
                )
            )
        try:
            while True:
                # Wait for the operator to set up the scene and press Enter before EVERY
                # rollout (including the first); 'q' finishes. No-op for non-TTY / --single.
                if watcher is not None and not watcher.wait_for_enter(
                    "[rollout] set up the scene, then press Enter to start the rollout "
                    "(or 'q' to quit) ..."
                ):
                    print("[rollout] 'q' -> finishing.")
                    break
                logger = EpisodeLogger(
                    Path(ROOT) / cfg["log_dir"], REAL_TASK_ID, variant="MVTOKEN"
                )

                # Piper returns to BEGIN before each rollout so every episode starts from the
                # same known pose as the training data; then re-sync to it.
                if hardware == "piper":
                    move_to_begin_on_init(cfg, session)
                # Re-sync to the live robot pose before each rollout.
                controller.sync_from_robot()
                extra_fields: dict = {}

                logger.write_metadata(
                    {
                        "task": cfg["task"],
                        "gripper_color": cfg.get("gripper_color", "black"),
                        "robot_config": cfg,
                        "primitives_config": primitives_cfg,
                        "prompt_file": str(prompt_path.resolve()),
                        "prompt_version": args.version,
                        "mode": "lite",
                        "control_mode": "real_mvtoken",
                        "z_floor_m": controller.z_floor_m,
                        "debug": args.debug,
                    }
                )

                print(f"\n{console.rule(f'rollout {logger.run_dir.name}')}")
                print(console.dim(f"  dir      {logger.run_dir}"))
                runner = MvTokenRunner(
                    session=session,
                    controller=controller,
                    agent=MvTokenController(
                        client=client, prompt_template=prompt_template,
                        extra_fields=extra_fields,
                    ),
                    logger=logger,
                    task=str(cfg["task"]),
                    gripper_color=str(cfg.get("gripper_color", "black")),
                    max_steps=int(cfg["max_steps"]),
                    loop_period_s=float(cfg.get("loop_period_s", 0.0)),
                    use_wrist_image=bool(cfg.get("use_wrist_image", True)),
                    video_fps=float(cfg.get("v0", {}).get("video_fps", 2.0)),
                    debug=args.debug,
                    viewer=viewer,
                    auto_release=auto_release,
                    prompt_log_every=int(args.prompt_log_every),
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

                if watcher is None:
                    break  # single-rollout / non-TTY
                if watcher.quit.is_set():
                    print("[rollout] 'q' -> finishing.")
                    break
                if result.end_reason == "interrupted" and not watcher.restart.is_set():
                    # Ctrl+C (not our restart/quit key) -> stop.
                    break
                if watcher.restart.is_set():
                    print("[rollout] restart key -> ending this rollout.")
                # Loop back to the Enter-confirm gate at the top before the next rollout.
        finally:
            if watcher is not None:
                watcher.shutdown()
                watcher.join(timeout=1.0)  # let it restore the terminal (cbreak -> normal)
        exit_code = 0
    except KeyboardInterrupt:
        print("\n[run-real-mvtoken] interrupted before the rollout loop started.")
        exit_code = 130
    finally:
        viewer.close()
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
