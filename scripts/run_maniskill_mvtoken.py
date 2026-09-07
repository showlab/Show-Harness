#!/usr/bin/env python3
"""Show-Harness ManiSkill experiment for the MVTOKEN Qwen LoRA (planner-free / stage-free).

Flat MVTOKEN control loop driving
a ManiSkill Panda in the translation-only ``pd_ee_delta_pos`` control mode instead of the
a ManiSkill env. It reuses the shared sim config + VLM-client wiring, builds a 4-dim
:class:`ManiskillAtomicController`, and runs :class:`MvTokenManiskillRunner`.

CROSS-DOMAIN experiment: the LoRA was trained on REAL Franka/Piper rollouts, so its
behaviour in the visually-different ManiSkill simulator is exploratory, not expected to
match real quality. The camera orientation and the base-frame move_vectors in
``configs/robot_maniskill.yaml`` are the knobs to tune from the rollout video.

Start the vLLM server first (Terminal 1, on the GPU box) -- serve the adapter the
config's vlm_backend names, on :8000 (scripts/serve_vlm.sh):
    MODEL=<base> FAMILY=qwen3_5 \\
      LORA=qwen3_5_2b_showharness_sim=<path> bash scripts/serve_vlm.sh

Then (Terminal 2), inside a ManiSkill-capable env (e.g. conda `mimicgen`):
    python scripts/run_maniskill_mvtoken.py --version v3 --env-id PickCube-v1
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreters.maniskill_atomic_controller import ManiskillAtomicController
from core.config import load_secrets_env, load_yaml
from core.record.episode_logger import EpisodeLogger
from core.sim.maniskill_task import make_maniskill_task, maybe_set_mujoco_gl, probe_move_axes
from core.sim.mvtoken_maniskill_runner import MvTokenManiskillRunner
from core.v0_types import V0Config
from plugins.auto_release import AutoReleasePlugin
from plugins.config import PluginsConfig
from core.vlm.mvtoken_roles import MvTokenController

# Shared sim config/client wiring (no startup calibration -- the ManiSkill
# atomic controller maps MV_* -> base-frame XYZ directly, no OSC axis probe needed).
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
        description="Show-Harness ManiSkill: MVTOKEN atomic-token policy (no planner/stage)"
    )
    parser.add_argument(
        "--robot-config", default=str(ROOT / "configs" / "robot_maniskill.yaml")
    )
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument(
        "--version",
        default="v3",
        help="Prompt version subfolder under prompts/ (e.g. v3, v4), matching the served "
        "LoRA's training data. Reads prompts/<version>/mvtoken_generator_lite.txt.",
    )
    parser.add_argument("--env-id", default=None, help="ManiSkill env id (e.g. PickCube-v1).")
    parser.add_argument("--robot-uids", default=None)
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument(
        "--traj-id",
        default=None,
        help="Scene layout preset (scene.traj_id): 0 | 15 | 25 | 40 | 45 | random.",
    )
    parser.add_argument(
        "--layout",
        default=None,
        help="Object layout at reset: `wide` re-randomises block + coaster like the "
             "training data generator. With --traj-id random this is the evaluation "
             "protocol (formerly configs/robot_maniskill_blockpap_eval.yaml).",
    )
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
        "--probe-axes", action="store_true",
        help="Before the rollout, step each MV_* token in isolation and record the observed "
        "TCP delta into calibration.json (diagnostic for the axis/sign mapping).",
    )
    parser.add_argument(
        "--prompt-log-every", type=int, default=20,
        help="Save the exact VLM prompt + returned token every N steps (0 disables).",
    )
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG") == "1"
    )
    # build_config reads these; ManiSkill does not use them.
    parser.set_defaults(task_suite_name=None, task_id=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    # Fold ManiSkill-only CLI overrides into the raw config before build_config runs.
    for arg_name, cfg_key in [
        ("env_id", "env_id"),
        ("robot_uids", "robot_uids"),
        ("control_mode", "control_mode"),
        ("sim_backend", "sim_backend"),
        ("task_description", "task_description"),
        ("episode_index", "episode_index"),
        ("layout", "layout"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            robot_cfg[cfg_key] = value
    # traj_id lives under the per-scene block, not at the top level.
    if args.traj_id is not None:
        robot_cfg.setdefault("scene", {})["traj_id"] = args.traj_id

    cfg = build_config(args, robot_cfg)
    prompt_path = _prompt_path(
        Path(args.prompts_dir), args.version
    )
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()

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

    maybe_set_mujoco_gl()
    max_steps = int(cfg["max_steps"])
    sim_steps = int(cfg["sim_steps_per_decision"])
    num_wait = int(cfg["num_steps_wait"])
    # Give the env enough budget so its own max_episode_steps truncation never cuts a
    # rollout short before our max_steps decisions run.
    max_episode_steps = max_steps * sim_steps + num_wait + sim_steps + 8

    handle = make_maniskill_task(
        env_id=str(cfg["env_id"]),
        control_mode=str(cfg["control_mode"]),
        robot_uids=str(cfg["robot_uids"]),
        camera_resolution=(
            int(cfg["camera_resolution"]) if cfg.get("camera_resolution") else None
        ),
        sim_backend=str(cfg["sim_backend"]),
        max_episode_steps=max_episode_steps,
        task_description=cfg.get("task_description") or None,
        # `scene:` is the per-scene option block (core.sim.maniskill_scenes.SceneSpec declares
        # which keys each env understands). `blockpap:` is the old name for the same block,
        # still accepted so existing configs keep working.
        scene=cfg.get("scene") or cfg.get("blockpap"),
    )
    print(f"Env: {handle.env_id} | robot: {handle.robot_uids} | task: {handle.task_description}")

    seed = int(cfg["seed"]) + int(cfg["episode_index"])
    controller = ManiskillAtomicController(
        move_vectors=cfg["move_vectors"],
        step_m=float(cfg["step_m"]),
        delta_bound_m=float(cfg["delta_bound_m"]),
        open_gripper_action=float(cfg["open_gripper_action"]),
        close_gripper_action=float(cfg["close_gripper_action"]),
    )

    # Auto-release safety rule (plugins.auto_release): reopen a closed gripper whose measured
    # width collapses below empty_width_m (it is holding nothing). On by default; disable
    # with `plugins: {auto_release: false}` in configs/robot_maniskill.yaml.
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

    logger = EpisodeLogger(
        Path(ROOT) / cfg["log_dir"], 0, variant=f"MS-{handle.env_id}"
    )
    logger.write_metadata(
        {
            "env_id": handle.env_id,
            "robot_uids": handle.robot_uids,
            "control_mode": handle.control_mode,
            "task_description": handle.task_description,
            "episode_index": int(cfg["episode_index"]),
            "seed": seed,
            "robot_config": cfg,
            "prompt_file": str(prompt_path.resolve()),
            "prompt_version": args.version,
            "control_loop": "maniskill_mvtoken",
            "debug": args.debug,
        }
    )

    if args.probe_axes:
        axes = probe_move_axes(handle.env, controller, seed=seed)
        logger.write_calibration({"move_axis_tcp_delta": axes})
        print("Axis probe (TCP delta per MV_* token, world XYZ metres):")
        for token, delta in axes.items():
            print(f"  {token:8s} -> {delta}")

    runner = MvTokenManiskillRunner(
        env=handle.env,
        task_description=handle.task_description,
        controller=controller,
        agent=MvTokenController(
            client=client,
            prompt_template=prompt_template,
        ),
        logger=logger,
        config=V0Config.from_dict(cfg.get("v0", {})),
        max_steps=max_steps,
        num_steps_wait=num_wait,
        loop_period_s=float(cfg["loop_period_s"]),
        sim_steps_per_decision=sim_steps,
        seed=seed,
        agentview_camera=str(cfg["agentview_camera"]),
        wrist_camera=str(cfg["wrist_camera"]),
        agentview_rotation_degrees=int(cfg["agentview_rotation_degrees"]),
        wrist_rotation_degrees=int(cfg["wrist_rotation_degrees"]),
        agentview_flip=str(cfg.get("agentview_flip", "none")),
        wrist_flip=str(cfg.get("wrist_flip", "none")),
        use_wrist_image=bool(cfg["use_wrist_image"]),
        reset_qpos=cfg.get("reset_qpos"),
        layout=cfg.get("layout"),
        auto_release=auto_release,
        debug=args.debug,
        prompt_log_every=int(args.prompt_log_every),
        agentview_square_size=cfg.get("agentview_square_size"),
        agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
        wrist_square_size=cfg.get("wrist_square_size"),
        wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
    )
    result = runner.run()
    print(f"Episode success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"End reason: {result.end_reason}")
    print(f"Run dir: {result.run_dir}")
    print(f"Video: {result.video_path}")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
