#!/usr/bin/env python3
"""Call a focused grasp-affordance VLM role on the first frame and write affordance_config.json.

A lightweight cousin of generate_subgoals.py: instead of a full ordered plan it asks the VLM
for ONE thing -- which object to grasp first and the best visible grasp point on it -- so the
training prompt can carry a stable "where to grasp" hint without the subgoal machinery. The
``affordance`` semantics match Show-Harness's SubgoalPlanner (a visible graspable part / contact
region; for hollow objects the left/right side wall).

Uses the same vlm_backends config as the runtime. Select a backend with --vlm-backend
(e.g. gemma, qwen35); override URL or model with --vlm-url / --model.

The written affordance_config.json is read by rollouts_to_alpaca.py --use-affordance.

Schema:
    {"task": "...", "target": "...", "affordance": "..."}

Examples
--------
# Qwen3.5-27B via vLLM (start server first):
#   MODEL=<base model> bash scripts/serve_vlm.sh
python train/data_preparation/generate_affordance.py \\
    train/data/0622/banana/rollout_030 \\
    --task "pick up the banana and place it on the blue plate" \\
    --vlm-backend qwen35 \\
    --vlm-url http://localhost:8101/v1

# Dry-run — print the grasp point without saving:
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

# No-think regime: emit the JSON directly (mirrors tools/subgoal/agent.py).
_NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}

# Output contract: the single object to grasp and one visible grasp point on it.
_AFFORDANCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "affordance": {"type": "string"},
    },
    "required": ["target", "affordance"],
    "additionalProperties": False,
}

# Co-located prompt. Mirrors the affordance rules in tools/subgoal/subgoal_planner.txt so the
# grasp-point definition is consistent with the full subgoal planner.
_AFFORDANCE_PROMPT = """ROLE: GraspAffordance

Task: {task}

Identify the single object to grasp first for this task, and the best visible grasp point on
it for a parallel-jaw gripper.

Return JSON only:
{{
  "target": "the object to grasp, with a distinguishing relation if similar objects exist",
  "affordance": "the visible part or contact region to put between the gripper fingers"
}}

Rules:
- Pick a grasp point clearly visible in AgentView and reachable by the open fingers
- For simple solid objects (blocks, fruit), the affordance is the object body itself
- For hollow/container objects (bowls, cups), pick a visible side/rim contact region,
  preferably the left or right wall
- Return JSON only, no thought, no markdown, no prose
"""


def _add_to_path(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ── VLM call ──────────────────────────────────────────────────────────────────

def _plan_affordance(client, task: str, agentview, wrist, provider: str, debug: bool):
    """Return ``(target, affordance, raw_text)`` for the grasp point of ``task``.

    LlamaFactory's API server has no guided_json, so for that provider we skip the schema
    and rely on the prompt's explicit JSON format (same strategy as generate_subgoals.py).
    For vLLM we use guided_json and fall back to prompt-only JSON if the decoder returns
    empty content.
    """
    base_prompt = _AFFORDANCE_PROMPT.format(task=task)
    use_schema = provider != "llamafactory"

    def _call(prompt: str, schema):
        response = client.complete_json(
            prompt,
            agentview,
            wrist_image=wrist,
            schema=schema,
            max_tokens=512,
            temperature=0.0,
            chat_template_kwargs=_NO_THINK_CHAT_TEMPLATE_KWARGS,
            debug=debug,
        )
        data = response.payload.get("json") or {}
        return (
            str(data.get("target", "")).strip(),
            str(data.get("affordance", "")).strip(),
            response.raw_text,
        )

    if not use_schema:
        prompt = base_prompt + "\n\nReturn only the JSON object in the format shown above."
        return _call(prompt, None)

    try:
        return _call(base_prompt, _AFFORDANCE_SCHEMA)
    except RuntimeError as exc:
        if "VLM returned" not in str(exc):
            raise
        prompt = base_prompt + "\n\nReturn only the JSON object in the format shown above."
        return _call(prompt, None)


# ── Target resolution ─────────────────────────────────────────────────────────

def _is_rollout(path: Path) -> bool:
    return (path / "agentview" / "0000.png").exists() or (path / "actions.jsonl").exists()


def _resolve_target(path: Path) -> tuple[Path | None, Path | None]:
    """Return ``(image_rollout_dir, out_dir)`` for ``path``.

    Each task folder holds one task, so the grasp point is reusable across its rollouts:
      * a rollout dir  -> infer from it, write affordance_config.json INTO it;
      * a task folder  -> infer once from its FIRST rollout, write affordance_config.json
        into the TASK FOLDER so every rollout under it reuses the single inference.
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
    client,
    provider: str,
    dry_run: bool = False,
    debug: bool = False,
) -> dict | None:
    """Plan the grasp point from ``image_dir``; write affordance_config.json to ``out_dir``."""
    tag = out_dir.name
    agentview_path = image_dir / "agentview" / "0000.png"
    wrist_path     = image_dir / "wrist"     / "0000.png"

    if not agentview_path.exists():
        print(f"[skip] {tag}: missing {agentview_path}", file=sys.stderr)
        return None

    agentview = _load_image(agentview_path)
    wrist = _load_image(wrist_path) if wrist_path.exists() else None

    src = "" if image_dir == out_dir else f" (from {image_dir.name})"
    print(f"[{tag}] calling GraspAffordance{src} ...", flush=True)
    target, affordance, raw_text = _plan_affordance(
        client, task, agentview, wrist, provider, debug
    )

    if not target or not affordance:
        print(f"[{tag}] WARNING: empty target/affordance", file=sys.stderr)
        print(f"  raw: {raw_text[:300]}", file=sys.stderr)
        return None

    print(f"[{tag}] target={target!r}  affordance={affordance!r}")

    affordance_config = {"task": task, "target": target, "affordance": affordance}

    out_path = out_dir / "affordance_config.json"
    if dry_run:
        print(f"[{tag}] [dry-run] would write {out_path}:")
        print(json.dumps(affordance_config, indent=2, ensure_ascii=False))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(affordance_config, f, indent=2, ensure_ascii=False)
        print(f"[{tag}] wrote {out_path}")

    return affordance_config


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate affordance_config.json via GraspAffordance")
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
        help="Directory to write affordance_config.json into. Overrides the location derived "
             "from the input path (the image source is still taken from the input path).",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    _add_to_path(args.repo_root)

    from core.config import load_yaml, resolve_vlm_config  # noqa: PLC0415
    from core.vlm.vlm_client import VLMClient           # noqa: PLC0415

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
            image_dir, out_dir, args.task, client, provider,
            dry_run=args.dry_run, debug=args.debug,
        )


if __name__ == "__main__":
    main()
