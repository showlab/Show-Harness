#!/usr/bin/env python3
"""Call SubgoalPlanner on the first frame of each rollout and write task_config.json.

Uses the same vlm_backends config as the runtime. Select a backend with
--vlm-backend (e.g. gemma, qwen35); override URL or model with --vlm-url / --model.

Every subgoal field (id/target/affordance/motion/description/completion) is produced by
the VLM, then run through Show-Harness's own ``plugins.subgoal.SubgoalPlanner`` so the stored
plan is byte-for-byte what the Show-Harness runtime would consume: ``Subgoal.from_dict``
validation plus the pre-grasp merge heuristic. No keyword-based stage bucketing is applied
here — the ordered subgoal list is written verbatim.

The written task_config.json is read by rollouts_to_alpaca.py, which aligns each
recorded action step to one of these subgoals (anchored on the recorded GRASP/RELEASE) and
fills the per-step prompt with that subgoal's target/affordance/description/completion.

Schema:
    {"task": "...", "subgoals": [{"id","target","affordance","motion","description",
                                  "completion"}, ...]}

Examples
--------
# Qwen3.5-27B via LlamaFactory API (start server first):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli api examples/inference/qwen3_5_27b.yaml
python train/data_preparation/generate_subgoals.py \\
    train/data/overfit_test/rollout_000 \\
    --task "pick up the red cube and place it on the blue shelf" \\
    --vlm-backend qwen35

# Qwen3.5-27B via vLLM (start server first):
#   MODEL=<base model> bash scripts/serve_vlm.sh
python train/data_preparation/generate_subgoals.py \\
    train/data/0622/banana/rollout_030 \\
    --task "pick up the banana and place it on the blue plate" \\
    --vlm-backend qwen35 \\
    --vlm-url http://localhost:8101/v1

# Override model (e.g. local fine-tuned checkpoint):
    --model <repo>/train/saves/qwen3.5-27b/robot/merged

# Dry-run — print plans without saving task_config.json:
    --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Repo roots ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]   # train/data_preparation/ -> train/ -> <repo>/
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "robot_franka.yaml"


def _add_to_path(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ── LlamaFactory provider wrapper: skip guided_json ──────────────────────────

class _NoGuidedJsonPlanner:
    """Wraps SubgoalPlannerAgent and always calls complete_json with schema=None.

    LlamaFactory's API server does not support vLLM's guided_json parameter.
    The agent's fallback already handles this, but we enter that path directly
    to avoid one unnecessary failed request.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def plan(self, task: str, agentview_image, wrist_image=None, debug: bool = False):
        from plugins.subgoal.agent import _join_prompt_parts  # noqa: PLC0415

        prompt = _join_prompt_parts(
            self._inner.common_context,
            self._inner.prompt_template.format(task=task),
        )
        prompt += (
            "\n\nReturn only the JSON object in the format shown above. "
            "Do not include thought, reasoning, markdown, or prose."
        )
        return self._inner.client.complete_json(
            prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=None,
            max_tokens=2048,
            temperature=0.0,
            chat_template_kwargs={},
            debug=debug,
        )


# ── Target resolution ─────────────────────────────────────────────────────────

def _is_rollout(path: Path) -> bool:
    return (path / "agentview" / "0000.png").exists() or (path / "actions.jsonl").exists()


def _resolve_target(path: Path) -> tuple[Path | None, Path | None]:
    """Return ``(image_rollout_dir, out_dir)`` for ``path``.

    Each task folder holds one task, so the plan is reusable across its rollouts:
      * a rollout dir  -> infer from it, write task_config.json INTO it;
      * a task folder  -> infer once from its FIRST rollout, write task_config.json into
        the TASK FOLDER so every rollout under it reuses the single inference.
    """
    if _is_rollout(path):
        return path, path
    if path.is_dir():
        rollouts = sorted(c for c in path.iterdir() if c.is_dir() and _is_rollout(c))
        if rollouts:
            return rollouts[0], path
    return None, None


# ── Per-rollout processing ────────────────────────────────────────────────────

def generate_for_rollout(
    image_dir: Path,
    out_dir: Path,
    task: str,
    planner,
    dry_run: bool = False,
    debug: bool = False,
) -> dict | None:
    """Run the planner on ``image_dir``'s first frame; write task_config.json to ``out_dir``."""
    tag = out_dir.name
    agentview_path = image_dir / "agentview" / "0000.png"
    wrist_path     = image_dir / "wrist"     / "0000.png"

    if not agentview_path.exists():
        print(f"[skip] {tag}: missing {agentview_path}", file=sys.stderr)
        return None

    agentview = _load_image(agentview_path)
    wrist = _load_image(wrist_path) if wrist_path.exists() else None

    src = "" if image_dir == out_dir else f" (from {image_dir.name})"
    print(f"[{tag}] calling SubgoalPlanner{src} ...", flush=True)
    # SubgoalPlanner.plan returns (list[core.v0_types.Subgoal], raw_text): the SAME
    # validated + pre-grasp-merged list Show-Harness's runtime advances through.
    try:
        subgoals, raw_plan = planner.plan(task, agentview, wrist=wrist, debug=debug)
    except (RuntimeError, ValueError) as exc:
        print(f"[{tag}] WARNING: planner failed: {exc}", file=sys.stderr)
        return None

    if not subgoals:
        print(f"[{tag}] WARNING: planner returned no subgoals", file=sys.stderr)
        return None

    print(f"[{tag}] {len(subgoals)} subgoals:")
    for sg in subgoals:
        print(f"  [{sg.motion:>14s}] {sg.id} — {sg.target} / {sg.affordance}")

    task_config = {
        "task":     task,
        "subgoals": [sg.to_prompt_dict() for sg in subgoals],
    }

    out_path = out_dir / "task_config.json"
    if dry_run:
        print(f"[{tag}] [dry-run] would write {out_path}:")
        print(json.dumps(task_config, indent=2, ensure_ascii=False))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(task_config, f, indent=2, ensure_ascii=False)
        print(f"[{tag}] wrote {out_path}")

    return task_config


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task_config.json via SubgoalPlanner")
    parser.add_argument("rollout_dirs", nargs="+", type=Path)
    parser.add_argument("--task", required=True, help="Natural language task description")
    parser.add_argument(
        "--vlm-backend", default=None,
        help="Backend from the robot config's vlm_backends (e.g. gemma, qwen35). "
             "Defaults to vlm_backend key in config.",
    )
    parser.add_argument("--vlm-url", default=None, help="Override VLM base URL")
    parser.add_argument("--model",   default=None, help="Override model name or local path")
    parser.add_argument("--config",  type=Path, default=_DEFAULT_CONFIG,
                        help=f"Show-Harness robot config yaml (default: {_DEFAULT_CONFIG})")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory to write task_config.json into. Overrides the location derived from "
             "the input path (the image source is still taken from the input path).",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    _add_to_path(args.repo_root)

    from core.config import load_yaml, resolve_vlm_config  # noqa: PLC0415
    from core.vlm.vlm_client import VLMClient           # noqa: PLC0415
    from plugins.subgoal import SubgoalPlanner               # noqa: PLC0415
    from plugins.subgoal.agent import SubgoalPlannerAgent    # noqa: PLC0415

    # robot_franka.yaml pulls in a site layer that is gitignored; without it fall back to
    # an empty config and rely on --vlm-backend / --vlm-url.
    try:
        cfg = load_yaml(str(args.config)) if args.config.exists() else {}
    except FileNotFoundError as e:
        print(f"note: {e}", file=sys.stderr)
        cfg = {}
    vlm_cfg = resolve_vlm_config(cfg, backend=args.vlm_backend)

    if args.vlm_url: vlm_cfg["base_url"] = args.vlm_url
    if args.model:   vlm_cfg["model"]    = args.model
    if args.api_key: vlm_cfg["api_key"]  = args.api_key

    provider = vlm_cfg.get("provider", "vllm")
    print(
        f"backend : {vlm_cfg['backend']}  provider={provider}\n"
        f"model   : {vlm_cfg['model']}\n"
        f"url     : {vlm_cfg.get('base_url', '?')}"
    )

    client = VLMClient(
        base_url=vlm_cfg.get("base_url", "http://localhost:8000/v1"),
        model=vlm_cfg["model"],
        api_key=vlm_cfg.get("api_key", "EMPTY"),
        timeout_s=60,
        max_tokens=vlm_cfg.get("max_tokens", 2048),
        temperature=0.0,
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs"),
    )
    # SubgoalPlanner (the Show-Harness tool) wraps the VLM agent and applies the same
    # Subgoal.from_dict validation + pre-grasp merge the runtime uses. For the
    # LlamaFactory backend the agent is the no-guided-json shim (still a drop-in agent,
    # since SubgoalPlanner only calls agent.plan()).
    inner = SubgoalPlannerAgent(client=client, common_context="")
    agent = _NoGuidedJsonPlanner(inner) if provider == "llamafactory" else inner
    planner = SubgoalPlanner(agent)

    if provider == "llamafactory":
        print("note: LlamaFactory backend — using prompt-only JSON (no guided_json)")

    for d in args.rollout_dirs:
        image_dir, out_dir = _resolve_target(d)
        if image_dir is None:
            print(f"[skip] {d}: not a rollout and no rollout subdirs", file=sys.stderr)
            continue
        if args.out_dir is not None:
            out_dir = args.out_dir
        generate_for_rollout(
            image_dir, out_dir, args.task, planner, dry_run=args.dry_run, debug=args.debug
        )


if __name__ == "__main__":
    main()
