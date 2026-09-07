#!/usr/bin/env python3
"""Show-Harness RoboLab experiment for the MVTOKEN LoRA (planner-free / stage-free).

Simulator sibling of ``scripts/run_maniskill_mvtoken.py``: the
same flat MVTOKEN control loop, driving a RoboLab (NVIDIA Isaac Lab) Franka + Robotiq 2F-85
through the relative differential-IK action space. It reuses ``run.py``'s config + VLM
client wiring, builds a 7-dim :class:`RobolabAtomicController`, and runs
:class:`MvTokenRobolabRunner`.

Two structural differences from the ManiSkill entry point, both forced by Isaac Sim:

* **Isaac Sim is launched before the env is built, and only once per process.** Nothing may
  import ``isaaclab``/``robolab`` before ``AppLauncher`` starts the Kit app, which is why
  the whole RoboLab side lives behind ``core.sim.robolab_task.launch_isaac``. The VLM client is
  built and health-checked FIRST, so a down vLLM server fails in a second instead of after
  Isaac Sim's cold start.
* **``--episodes N`` runs N episodes in ONE process**, reusing the app and the env
  (``env.reset_eval_state()`` between them, exactly as RoboLab's own ``run_evaluation``
  does). Isaac Sim start-up plus env construction costs tens of seconds; one process per
  episode -- fine on ManiSkill -- would dominate the wall clock here.

CROSS-DOMAIN experiment: the LoRA was trained on REAL Franka/Piper rollouts, so behaviour
in RoboLab's photoreal scenes is exploratory. Before reading anything into a success rate,
verify the two image contracts with ``--dump-views`` and the step size with
``--probe-axes``; see docs/simulators.md.

Start the vLLM server first (Terminal 1, on the GPU box) -- serve the adapter the
config's vlm_backend names, on :8000 (scripts/serve_vlm.sh):
    MODEL=<base> FAMILY=qwen3_5 \\
      LORA=qwen3_5_2b_showharness_sim=<path> bash scripts/serve_vlm.sh

Then (Terminal 2), with RoboLab's interpreter:
    <robolab-venv>/bin/python scripts/run_robolab_mvtoken.py \
        --version v3 --task BananaInBowlTask --dump-views
"""
from __future__ import annotations

import argparse
import sys
import os
import traceback
from pathlib import Path

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml
from core.record.episode_logger import EpisodeLogger
from core.v0_types import V0Config
from plugins.auto_release import AutoReleasePlugin
from plugins.config import PluginsConfig
from core.vlm.mvtoken_roles import MvTokenController

# Shared sim config/client wiring (no startup calibration -- the RoboLab atomic
# controller maps MV_* -> base-frame XYZ directly).
from core.sim.launch import build_config, make_vlm_client



def _prompt_path(prompts_dir: Path, version: str) -> Path:
    """prompts/<version>/mvtoken_generator_lite.txt, with the v4 per-embodiment fallback
    (v4 has no shared lite prompt -> use the franka overhead one, matching base_camera).
    """
    base = prompts_dir / version
    names = ("mvtoken_generator_lite.txt", "franka_mvtoken_lite.txt")
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No lite prompt under {base} (looked for {' / '.join(names)}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness RoboLab: MVTOKEN atomic-token policy (no planner/stage)"
    )
    parser.add_argument(
        "--robot-config", default=str(ROOT / "configs" / "robot_robolab.yaml")
    )
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument(
        "--version",
        default="v3",
        help="Prompt version subfolder under prompts/ (e.g. v3, v4), matching the served "
        "LoRA's training data. Reads prompts/<version>/mvtoken_generator_lite.txt.",
    )
    # -- RoboLab env selection -------------------------------------------------
    parser.add_argument(
        "--task", default=None,
        help="RoboLab Task class name, e.g. BananaInBowlTask (see --list-tasks).",
    )
    parser.add_argument("--instruction-type", default=None, help="default | vague | specific")
    parser.add_argument("--camera-preset", default=None, help="WRIST_LEFT | WRIST_LEFT_RIGHT | ...")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Episodes to run in this process (app + env reused between them).",
    )
    parser.add_argument("--renderer", default=None, choices=(None, "realtime", "pathtracing"))
    parser.add_argument(
        "--rendering-type", default=None, choices=(None, "performance", "balanced", "quality")
    )
    parser.add_argument(
        "--enable-subtask", action="store_true", default=None,
        help="Turn on RoboLab's per-step subtask-progress predicates (off by default).",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Run WITH the Isaac Sim viewport (default is headless).",
    )
    # -- diagnostics -----------------------------------------------------------
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="Print every registered RoboLab task name and exit (needs Isaac Sim).",
    )
    parser.add_argument(
        "--dump-views", action="store_true",
        help="After reset, save the exact agentview/wrist PNGs the policy would be sent "
        "(plus the raw renders) into the run dir, then continue. THE way to verify "
        "wrist_flip and the crop/letterbox geometry -- do this before trusting any rollout.",
    )
    parser.add_argument(
        "--probe-axes", action="store_true",
        help="Before the rollout, step each MV_* token in isolation and record the observed "
        "TCP delta into calibration.json (diagnostic for the axis/sign mapping AND for "
        "whether one decision really travels step_m).",
    )
    parser.add_argument(
        "--no-rollout", action="store_true",
        help="Skip the policy rollout (use with --dump-views / --probe-axes for a smoke "
        "test that needs no VLM server).",
    )
    # -- VLM -------------------------------------------------------------------
    parser.add_argument(
        "--vlm-backend", default=os.environ.get("VLM_BACKEND")
    )
    parser.add_argument(
        "--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL")
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--prompt-log-every", type=int, default=20,
        help="Save the exact VLM prompt + returned token every N steps (0 disables).",
    )
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG") == "1"
    )
    # build_config reads these; RoboLab does not use them.
    parser.set_defaults(task_suite_name=None, task_id=None)
    return parser.parse_args()


def _fold_overrides(args: argparse.Namespace, robot_cfg: dict) -> dict:
    """CLI -> raw config, before build_config merges in the shared keys."""
    for arg_name, cfg_key in [
        ("task", "task"),
        ("instruction_type", "instruction_type"),
        ("camera_preset", "camera_preset"),
        ("device", "device"),
        ("seed", "seed"),
        ("episode_index", "episode_index"),
        ("episodes", "episodes"),
        ("renderer", "renderer"),
        ("rendering_type", "rendering_type"),
        ("enable_subtask", "enable_subtask"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            robot_cfg[cfg_key] = value
    if args.gui:
        robot_cfg["headless"] = False
    return robot_cfg


def _dump_views(runner, obs, run_dir: Path) -> None:
    """Save the raw renders AND the transformed views the policy would actually see.

    The pair is the point: comparing ``*_raw.png`` with ``*_sent.png`` shows exactly what
    the rotation/flip/crop/letterbox chain did, which is the only reliable way to settle
    ``wrist_flip`` (fingertips at the TOP, wrist left == agentview left) and to judge
    whether the 4:3 crop framed the scene the way the training rigs do.
    """
    from core.record.images import save_png
    from core.sim.robolab_task import rl_rgb

    out = Path(run_dir) / "views"
    save_png(out / "agentview_raw.png", rl_rgb(obs, runner.agentview_camera))
    save_png(out / "wrist_raw.png", rl_rgb(obs, runner.wrist_camera))
    agentview, wrist = runner._images(obs)
    save_png(out / "agentview_sent.png", agentview)
    if wrist is not None:
        save_png(out / "wrist_sent.png", wrist)
    print(f"[robolab] wrote view dump to {out}")
    print(
        "[robolab] CHECK: wrist_sent.png must show the fingertips at the TOP, and left/right "
        "must agree with agentview_sent.png. If not, change wrist_flip in the config "
        "(none | vertical | horizontal | both)."
    )


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = _fold_overrides(args, load_yaml(args.robot_config))
    cfg = build_config(args, robot_cfg)

    # --list-tasks only reads the task directory; do not pay Isaac Sim's cold start for it.
    if args.list_tasks:
        return _list_tasks()

    need_vlm = not args.no_rollout
    client = None
    if need_vlm:
        prompt_path = _prompt_path(
            Path(args.prompts_dir), args.version
        )
        prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        # Build + health-check the VLM BEFORE Isaac Sim boots: a down server should fail in
        # a second, not after a minute of Kit start-up.
        client = make_vlm_client(args, cfg)
        vlm_cfg = cfg["vlm"]
        print(
            f"VLM backend: {vlm_cfg.get('backend', '?')} "
            f"(model={vlm_cfg['model']}, base_url={vlm_cfg['base_url']})"
        )
        if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
            print(f"Waiting for VLM endpoint {vlm_cfg['base_url']}/models ...")
            client.health_check(
                wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
                poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
            )
            print("VLM endpoint is ready.")
    else:
        prompt_path = None
        prompt_template = ""

    # ---- Isaac Sim: everything below this line may import isaaclab/robolab ----
    from core.sim.robolab_task import launch_isaac

    headless = bool(cfg.get("headless", True))
    simulation_app = launch_isaac(headless=headless, device=str(cfg.get("device", "cuda:0")))
    try:
        return _run(args, cfg, client, prompt_path, prompt_template)
    except Exception:
        # Isaac Sim's SimulationApp.close() can terminate the process outright, which
        # swallows BOTH the traceback and the exit code -- a crash then looks like a clean
        # exit 0 that quietly skipped half the run. Print before closing.
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


def _run(args, cfg, client, prompt_path, prompt_template) -> int:
    from interpreters.robolab_atomic_controller import RobolabAtomicController
    from core.sim.mvtoken_robolab_runner import MvTokenRobolabRunner
    from core.sim.robolab_task import make_robolab_task, probe_move_axes, reset_robolab

    episodes = max(1, int(cfg.get("episodes", 1)))
    seed = int(cfg["seed"]) + int(cfg.get("episode_index", 0))

    # The logger is built BEFORE the env so RoboLab's own artefacts (env_cfg.json) land in
    # the same run directory as this repo's steps.jsonl / videos, instead of RoboLab's
    # separate output/ tree.
    logger = EpisodeLogger(Path(ROOT) / cfg["log_dir"], 0, variant=f"RL-{cfg['task']}")
    handle = make_robolab_task(
        task=str(cfg["task"]),
        num_envs=1,
        device=str(cfg.get("device", "cuda:0")),
        seed=seed,
        instruction_type=str(cfg.get("instruction_type", "default")),
        camera_preset=str(cfg.get("camera_preset", "WRIST_LEFT")),
        renderer=str(cfg.get("renderer", "realtime")),
        rendering_type=cfg.get("rendering_type"),
        output_dir=logger.run_dir,
        enable_subtask=bool(cfg.get("enable_subtask", False)),
        verbose=args.debug,
    )
    print(
        f"Env: {handle.env_name} | task: {handle.task} | action_dim: {handle.action_dim} "
        f"| ik_scale: {handle.ik_scale}"
    )
    print(f"Instruction: {handle.task_description}")
    if handle.action_dim != 7:
        raise SystemExit(
            f"Expected the 7-dim relative-IK action space, got action_dim={handle.action_dim}. "
            "The task was registered against the wrong action config."
        )

    controller = RobolabAtomicController(
        move_vectors=cfg["move_vectors"],
        step_m=float(cfg["step_m"]),
        ik_scale=handle.ik_scale,
        sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
        max_delta_m=float(cfg.get("max_delta_m", 0.05)),
    )

    # Auto-release safety rule (plugins.auto_release): reopen a closed gripper whose measured
    # width collapses below empty_width_m (it is holding nothing).
    plugins_cfg = PluginsConfig.from_config(cfg)
    auto_release = AutoReleasePlugin(
        enabled=plugins_cfg.enabled("auto_release", default=True),
        empty_width_m=float(cfg.get("empty_width_m", 0.005)),
    )
    if auto_release.enabled:
        print(
            "Auto-release: ON (reopen a closed gripper narrower than "
            f"{auto_release.empty_width_m:.4f} m)."
        )

    def make_agent():
        """The MVTOKEN controller, or None on the VLM-free calibration path.

        ``--no-rollout`` (used with --dump-views / --probe-axes) deliberately never loads a
        prompt, and MvTokenController rejects an empty template -- it derives the legal
        action set from the prompt text, so an empty one would mean "no legal action".
        The runner only touches ``agent`` inside ``run()``, which that path never reaches.
        """
        if client is None:
            return None
        return MvTokenController(
            client=client,
            prompt_template=prompt_template,
        )

    def make_runner(ep_logger: EpisodeLogger) -> MvTokenRobolabRunner:
        return MvTokenRobolabRunner(
            env=handle.env,
            task_description=handle.task_description,
            controller=controller,
            agent=make_agent(),
            logger=ep_logger,
            config=V0Config.from_dict(cfg.get("v0", {})),
            max_steps=int(cfg["max_steps"]),
            num_steps_wait=int(cfg["num_steps_wait"]),
            loop_period_s=float(cfg["loop_period_s"]),
            sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
            settle_steps_per_decision=int(cfg.get("settle_steps_per_decision", 0)),
            agentview_camera=str(cfg["agentview_camera"]),
            wrist_camera=str(cfg["wrist_camera"]),
            agentview_rotation_degrees=int(cfg["agentview_rotation_degrees"]),
            wrist_rotation_degrees=int(cfg["wrist_rotation_degrees"]),
            agentview_flip=str(cfg.get("agentview_flip", "none")),
            wrist_flip=str(cfg.get("wrist_flip", "none")),
            use_wrist_image=bool(cfg["use_wrist_image"]),
            auto_release=auto_release,
            debug=args.debug,
            prompt_log_every=int(args.prompt_log_every),
            agentview_square_size=cfg.get("agentview_square_size"),
            agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
            wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
            wrist_square_size=cfg.get("wrist_square_size"),
            gripper_hold_steps=int(cfg.get("gripper_hold_steps", 0)),
            close_env=False,  # the app owns the env; it is reused across episodes
        )

    metadata = {
        "task": handle.task,
        "env_name": handle.env_name,
        "task_description": handle.task_description,
        "targets": handle.targets,
        "action_dim": handle.action_dim,
        "ik_scale": handle.ik_scale,
        "episode_index": int(cfg.get("episode_index", 0)),
        "seed": seed,
        "robot_config": cfg,
        "prompt_file": str(prompt_path.resolve()) if prompt_path else None,
        "prompt_version": args.version,
        "control_loop": "robolab_mvtoken",
        "debug": args.debug,
    }
    logger.write_metadata(metadata)

    if args.dump_views or args.probe_axes:
        probe_runner = make_runner(logger)
        obs, _term, _trunc = reset_robolab(
            handle.env, controller.open_gripper(), settle_steps=int(cfg["num_steps_wait"])
        )
        if args.dump_views:
            _dump_views(probe_runner, obs, logger.run_dir)
        if args.probe_axes:
            # Same step budget a real decision gets, settle steps included -- measuring
            # without them reports the mid-flight displacement and always reads short.
            axes = probe_move_axes(
                handle.env,
                controller,
                repeats=int(cfg["sim_steps_per_decision"]),
                settle_steps=int(cfg.get("settle_steps_per_decision", 0)),
            )
            logger.write_calibration({"move_axis_tcp_delta": axes, "step_m": cfg["step_m"]})
            print("Axis probe (TCP delta per MV_* token, world XYZ metres):")
            for token, delta in axes.items():
                magnitude = sum(x * x for x in delta) ** 0.5
                print(f"  {token:8s} -> {delta}   |d| = {magnitude:.4f} m")
            print(
                f"  (target |d| = step_m = {float(cfg['step_m']):.3f} m; if it reads short, "
                "raise settle_steps_per_decision.)"
            )
        handle.env.reset_eval_state()

    if args.no_rollout:
        print("--no-rollout: skipping the policy rollout.")
        return 0

    successes = 0
    for episode in range(episodes):
        ep_logger = (
            logger
            if episode == 0
            else EpisodeLogger(Path(ROOT) / cfg["log_dir"], episode, variant=f"RL-{cfg['task']}")
        )
        if episode > 0:
            ep_logger.write_metadata({**metadata, "episode_index": episode})
            handle.env.reset_eval_state()
        result = make_runner(ep_logger).run()
        successes += int(result.success)
        print(
            f"[episode {episode}] success={result.success} steps={result.steps} "
            f"end_reason={result.end_reason}"
        )
        print(f"  run dir: {result.run_dir}")
        print(f"  video:   {result.video_path}")

    print(f"Success rate: {successes}/{episodes}")
    try:
        handle.env.close()
    except Exception:
        pass
    return 0 if successes else 2


def _list_tasks() -> int:
    """Print the RoboLab Task CLASS names ``--task`` accepts. Needs the checkout, NOT Isaac Sim.

    The class names are parsed out of the task sources with :mod:`ast` rather than
    imported, because importing a task module pulls in ``isaaclab`` and would cost the
    full Kit cold start just to print a list.

    Parsing (not deriving) is the point: the filename does NOT determine the class name.
    ``bagel_on_plate_task.py`` declares ``BagelsOnPlateTask`` (plural) and
    ``bbq_sauce_in_bin_task.py`` declares ``BBQSauceInBinTask`` (initialism), so any
    snake_case -> CamelCase rule produces names RoboLab rejects.
    """
    import ast

    from core.sim.robolab_task import ensure_robolab_path

    ensure_robolab_path()
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

    task_dir = Path(TASK_DIR)
    names: list[tuple[str, str]] = []
    for subdir in DEFAULT_TASK_SUBFOLDERS:
        for path in sorted((task_dir / subdir).glob("*.py")):
            if path.stem.startswith("_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in tree.body:
                # A task is a class whose name ends in "Task" and which subclasses Task.
                if not isinstance(node, ast.ClassDef) or not node.name.endswith("Task"):
                    continue
                bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
                if "Task" in bases:
                    names.append((node.name, f"{subdir}/{path.name}"))

    print(f"RoboLab tasks under {task_dir} ({len(names)}):")
    for name, source in sorted(names):
        print(f"  {name:42s}  {source}")
    print("\nPass one of these to --task. Full table: <ROBOLAB_ROOT>/robolab/tasks/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
