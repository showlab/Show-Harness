#!/usr/bin/env python3
"""Convert rollout directories to the Alpaca multimodal format (what LLaMA Factory consumes).

Each action step becomes one training sample. The two camera images are placed at the TOP
of the user turn -- the prompt template itself owns the camera-description header, and the
converter only prepends the "<image><image>" markers -- so every sample looks like:

  instruction : "<image><image>" + the rendered prompt (camera header + per-step context)
  input       : "" (everything lives in instruction)
  output      : the action token (MV_FWD/BACK/LEFT/RIGHT/UP/DOWN, GRASP, RELEASE) -- plus one
                synthesized DONE sample per episode on the final frame (all modes)
  images      : [agentview_abs_path, wrist_abs_path]

Prompt templates live under <repo>/prompts/<version>/ (selected by the required --version
arg, e.g. v3/v4). Within a version folder the per-mode filenames are fixed, all starting
with the camera header:
  mvtoken_generator_lite.txt        — default (lite). STAGE-FREE: only {task} /
                                {recent_moves}; no gripper state, no subgoal/affordance info.
  mvtoken_generator.txt             — used with --use-subgoal. Adds per-step
                                {stage}/{target}/{affordance}/{description}/{completion}
                                (+ {gripper_color}), taken VERBATIM from the VLM subgoals in
                                task_config.json (generate_subgoals.py). Each recorded step
                                is aligned to the subgoal active at that step: the recorded
                                GRASP/RELEASE anchor the approach/transport/retract phases
                                and the planner's subgoals advance within each phase, exactly
                                as Show-Harness's runtime steps through them.
  mvtoken_generator_affordance.txt  — used with --use-affordance. Lite + a single grasp-point
                                hint ({target}/{affordance}) reused on every step, from
                                affordance_config.json (generate_affordance.py). Lightweight
                                fix for "where to grasp" without the full subgoal machinery.

Recent moves are the last RECENT_WINDOW MV_* tokens, newest first (GRASP/RELEASE excluded),
matching inference.

Usage:
    # Lite (stage-free) — single rollout (--version picks prompts/<version>/)
    python train/data_preparation/rollouts_to_alpaca.py \\
        train/data/0622/banana/rollout_030 \\
        --version v1 \\
        --task "pick up the banana and place it on the blue plate"

    # Lite — parent dir auto-expands into rollout_* subdirs; multiple tasks via --task-map
    python train/data_preparation/rollouts_to_alpaca.py \\
        train/data/0622/banana \\
        train/data/0622/mango \\
        --version v1 \\
        --task-map "banana=pick up the banana and place it on the blue plate" \\
                   "mango=pick up the mango and place it on the blue plate" \\
        --output train/data/0622/rollout.json

    # Subgoal mode — full prompt with per-step VLM subgoal info (task_config.json auto-gen)
    python train/data_preparation/rollouts_to_alpaca.py \\
        train/data/0622/banana \\
        --version v1 \\
        --use-subgoal \\
        --task "pick up the banana and place it on the blue plate" \\
        --vlm-backend qwen3_5_2b \\
        --vlm-url http://localhost:8000/v1

    # Affordance mode — lite + a single grasp-point hint (affordance_config.json auto-gen)
    python train/data_preparation/rollouts_to_alpaca.py \\
        train/data/0622/banana \\
        --version v1 \\
        --use-affordance \\
        --task "pick up the banana and place it on the blue plate" \\
        --vlm-backend qwen3_5_2b \\
        --vlm-url http://localhost:8000/v1

    # Dual-arm (piper-dual) — three cameras, one token per arm per step
    python train/data_preparation/rollouts_to_alpaca.py \\
        train/data/dual_cloth \\
        --version v4 --dual --once \\
        --task "fold the black t-shirt"

DUAL-ARM (--dual)
-----------------
Dual recordings (dual_cloth) carry THREE views (agentview / wrist_left / wrist_right) and TWO
tokens per step (actions.jsonl rows have "left" and "right" objects). Because the arms are
teleoperated independently, a step where only one arm moved records STILL for the other, so
STILL joins the vocabulary. --dual is a hardware flag exactly like --franka / --piper; on top of
it, ONE scheme flag picks both the prompt file and the sample shape (see convert_dual_rollout):

  --twice  2 VLM calls/step -> 2 Alpaca samples/step (both carry all 3 views; {arm} says which
           arm to answer). The right call does NOT see the left token -> the two calls stay
           conditionally independent and can be batched in parallel at inference.
  --once   1 VLM call/step  -> 1 Alpaca sample/step, output "<left> <right>" (LEFT first).
  --chain  1 image forward, 2 answers -> 1 ShareGPT sample/step with two assistant turns
           (left token, follow-up user turn, right token).

--chain output is ShareGPT, so register it with formatting=sharegpt + columns.messages; --twice
and --once stay Alpaca (instruction/input/output/images). --dual is lite-only and rejects
--video-slot (3 views do not pair into Qwen's 2-frame temporal patches).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]   # train/data_preparation/ -> train/ -> <repo>/
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_GENERATE_SUBGOALS = _HERE.parent / "generate_subgoals.py"
_GENERATE_AFFORDANCE = _HERE.parent / "generate_affordance.py"

# ── Constants ─────────────────────────────────────────────────────────────────
RECENT_WINDOW = 5

MV_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"}
GRIPPER_TOKENS = {"GRASP", "RELEASE"}
ACTION_TOKENS = MV_TOKENS | GRIPPER_TOKENS

# Terminal token. The recordings contain no DONE, so one DONE sample is synthesized per
# episode on the final observed frame to teach "task complete" (all modes).
DONE_TOKEN = "DONE"

# The two image markers (agentview, wrist) are prepended so the images sit at the TOP of
# the prompt. The camera-description header now lives inside each prompt template (lite and
# full), so there is no separate instruction header here.
IMAGE_TOKENS = "<image><image>"

# --video-slot: the two views ride in the video slot as two "frames" instead of two images.
# Qwen's patch embed is a 3D conv with temporal_patch_size=2, so the two frames are fused into
# a single set of visual tokens: 64 instead of 128 per sample (at 256x256). The two viewpoints
# end up mixed at every spatial position -- an experiment, not the default. See ShowRobot-VLM_HANDOFF.md.
VIDEO_TOKEN = "<video>"

# ── Dual-arm (--dual) ─────────────────────────────────────────────────────────
# dual_cloth-style recordings: THREE cameras (agentview + one wrist per arm) and one token per
# arm per step, stored under the "left" / "right" keys of each actions.jsonl row. The two arms
# are teleoperated independently, so a step where only one arm moved records STILL for the other
# -- STILL is a first-class token here, and is NOT part of the single-arm vocabulary above.
STILL_TOKEN = "STILL"
DUAL_IMAGE_TOKENS = "<image><image><image>"  # agentview, wrist_left, wrist_right
DUAL_ACTION_TOKENS = ACTION_TOKENS | {STILL_TOKEN}
# Per-arm "recent moves": MV_* plus STILL (STILL is exactly what tells the model this arm is
# waiting on the other one). GRASP/RELEASE stay out, matching the single-arm convention.
DUAL_HISTORY_TOKENS = MV_TOKENS | {STILL_TOKEN}
DUAL_SCHEMES = ("twice", "once", "chain")

# --use-subgoal motion keywords: used ONLY to locate the grasp and release subgoals in the
# VLM plan (so the recorded GRASP/RELEASE tokens can anchor the phase boundaries). Every
# field shown to the model is taken verbatim from the VLM subgoals -- no placeholders.
GRASP_MOTION_KW = ("grasp", "pick", "grip", "grab", "clamp", "secure", "close")
RELEASE_MOTION_KW = ("release", "open", "drop", "ungrip", "let_go", "letgo")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prompt(version: str, filename: str) -> str:
    path = _PROMPTS_DIR / version / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_config(search_dirs: list[Path], filename: str) -> dict:
    for d in search_dirs:
        cfg_path = d / filename
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
    return {}


def _load_subgoals(task_cfg: dict) -> list[dict]:
    """The ordered VLM subgoal list from task_config.json (generate_subgoals.py).

    Prefers the current ``subgoals`` key (validated + pre-grasp-merged, Show-Harness-
    consistent); falls back to the legacy ``subgoals_raw`` so older configs still convert.
    """
    sgs = task_cfg.get("subgoals")
    if isinstance(sgs, list) and sgs:
        return sgs
    raw = task_cfg.get("subgoals_raw")
    if isinstance(raw, list) and raw:
        return raw
    return []


def _load_affordance(task_cfg: dict) -> dict[str, str]:
    """The single grasp-point hint from affordance_config.json (generate_affordance.py).

    Returns ``{"target": ..., "affordance": ...}`` or ``{}`` when absent/incomplete.
    """
    target = str(task_cfg.get("target", "")).strip()
    affordance = str(task_cfg.get("affordance", "")).strip()
    if target and affordance:
        return {"target": target, "affordance": affordance}
    return {}


def _find_motion(subgoals: list[dict], keywords: tuple[str, ...], start: int = 0) -> int | None:
    for j in range(start, len(subgoals)):
        motion = str(subgoals[j].get("motion", "")).lower()
        if any(kw in motion for kw in keywords):
            return j
    return None


def _distribute(step_idxs: list[int], sgs: list[dict]) -> dict[int, dict]:
    """Assign each step (in order) to a subgoal (in plan order) by an even split."""
    n_steps, n_sg = len(step_idxs), len(sgs)
    if n_steps == 0 or n_sg == 0:
        return {}
    return {si: sgs[min(n_sg - 1, (k * n_sg) // n_steps)] for k, si in enumerate(step_idxs)}


def _assign_subgoals_to_steps(all_tokens: list[str], subgoals: list[dict]) -> dict[int, dict]:
    """Map each step index -> the VLM subgoal active at that step.

    The recorded GRASP / RELEASE tokens are the only observable phase boundaries, so they
    anchor the three macro-phases while the planner's subgoals advance within each phase,
    mirroring Show-Harness's runtime (which steps through every subgoal in order):

      approach+grasp : steps [0 .. grasp]        -> subgoals [0 .. grasp_sg]
      transport      : steps (grasp .. release)  -> subgoals (grasp_sg .. release_sg)
      retract        : steps [release .. end]    -> subgoals [release_sg .. end]

    Within each phase the steps are split evenly across that phase's subgoals.
    """
    n = len(all_tokens)
    if not subgoals:
        return {}

    grasp_step = next((i for i, t in enumerate(all_tokens) if t == "GRASP"), None)
    release_step = next(
        (
            i for i, t in enumerate(all_tokens)
            if t == "RELEASE" and (grasp_step is None or i > grasp_step)
        ),
        None,
    )

    # No grasp event recorded -> nothing to anchor on; spread the whole plan over all steps.
    if grasp_step is None:
        return _distribute(list(range(n)), subgoals)

    grasp_sg = _find_motion(subgoals, GRASP_MOTION_KW) or 0
    release_sg = _find_motion(subgoals, RELEASE_MOTION_KW, start=grasp_sg + 1)
    if release_sg is None:
        release_sg = len(subgoals) - 1

    mapping: dict[int, dict] = {}

    # Phase A: approach + grasp.
    a_sgs = subgoals[: grasp_sg + 1] or subgoals[:1]
    mapping.update(_distribute(list(range(0, grasp_step + 1)), a_sgs))

    if release_step is None:
        # No release recorded: treat everything after the grasp as the post-grasp plan.
        b_sgs = subgoals[grasp_sg + 1:] or [subgoals[grasp_sg]]
        mapping.update(_distribute(list(range(grasp_step + 1, n)), b_sgs))
        return mapping

    # Phase B: transport (strictly between grasp and release).
    b_sgs = subgoals[grasp_sg + 1: release_sg] or [subgoals[min(release_sg, len(subgoals) - 1)]]
    mapping.update(_distribute(list(range(grasp_step + 1, release_step)), b_sgs))

    # Phase C: release + retract.
    c_sgs = subgoals[release_sg:] or [subgoals[-1]]
    mapping.update(_distribute(list(range(release_step, n)), c_sgs))

    return mapping


def _ensure_task_config(
    rollout_dir: Path, config_dir: Path, src_path: Path, task: str | None, vlm_args: list[str]
) -> None:
    """Ensure task_config.json exists in ``config_dir`` for this rollout.

    Images are read from ``src_path`` (the rollout itself for a single conversion, or the task
    folder for a multi-folder run — generated ONCE from its first rollout and reused), while the
    config is written into ``config_dir`` (next to the output json for single, a per-task
    subfolder of the output dir for multi). convert_rollout searches [rollout_dir, config_dir].
    """
    if (rollout_dir / "task_config.json").exists() or (config_dir / "task_config.json").exists():
        return
    if not task:
        raise ValueError(
            f"No task_config.json found in {rollout_dir} (or {config_dir}) "
            "and --task was not provided. Cannot auto-generate subgoals."
        )
    print(f"[subgoal] task_config.json missing for {rollout_dir.name}, generating once ...")
    config_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_GENERATE_SUBGOALS), str(src_path),
        "--task", task, "--out-dir", str(config_dir),
    ] + vlm_args
    subprocess.run(cmd, check=True)


def _ensure_affordance_config(
    rollout_dir: Path, config_dir: Path, src_path: Path, task: str | None, vlm_args: list[str]
) -> None:
    """Ensure affordance_config.json exists in ``config_dir`` for this rollout.

    Same placement rules as :func:`_ensure_task_config`: images from ``src_path``, config written
    into ``config_dir``.
    """
    if (rollout_dir / "affordance_config.json").exists() or (
        config_dir / "affordance_config.json"
    ).exists():
        return
    if not task:
        raise ValueError(
            f"No affordance_config.json found in {rollout_dir} (or {config_dir}) "
            "and --task was not provided. Cannot auto-generate the affordance hint."
        )
    print(f"[affordance] affordance_config.json missing for {rollout_dir.name}, generating once ...")
    config_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_GENERATE_AFFORDANCE), str(src_path),
        "--task", task, "--out-dir", str(config_dir),
    ] + vlm_args
    subprocess.run(cmd, check=True)


def _expand_dirs(paths: list[Path]) -> list[Path]:
    """Expand parent directories into their rollout subdirectories."""
    result: list[Path] = []
    for p in paths:
        if (p / "actions.jsonl").exists():
            result.append(p)
        else:
            children = sorted(c for c in p.iterdir() if c.is_dir() and (c / "actions.jsonl").exists())
            if children:
                print(f"[expand] {p.name}: found {len(children)} rollout(s)")
                result.extend(children)
            else:
                print(f"[skip] {p}: no actions.jsonl and no rollout subdirs", file=sys.stderr)
    return result


# ── Core conversion ───────────────────────────────────────────────────────────

def convert_rollout(
    rollout_dir: Path,
    prompt_template: str,
    mode: str = "lite",
    task_override: str | None = None,
    gripper_color: str = "black",
    config_dir: Path | None = None,
    video_slot: bool = False,
) -> list[dict]:
    """Convert one rollout to LLaMA Factory samples.

    ``mode`` selects the prompt body: ``lite`` (stage-free), ``subgoal`` (per-step VLM
    subgoal), or ``affordance`` (a single grasp-point hint reused for every step).

    ``config_dir`` is where the matching task/affordance config lives (next to the output json
    for a single rollout, a per-task subfolder of the output dir for a multi-folder run); it is
    searched after ``rollout_dir``.

    ``video_slot`` emits the two views as video frames (``<video>`` + ``videos``) instead of
    two images (``<image><image>`` + ``images``).
    """
    actions_path = rollout_dir / "actions.jsonl"
    if not actions_path.exists():
        print(f"[skip] no actions.jsonl in {rollout_dir}", file=sys.stderr)
        return []

    cfg_name = {
        "subgoal": "task_config.json",
        "affordance": "affordance_config.json",
    }.get(mode)
    search_dirs = [rollout_dir, config_dir or rollout_dir.parent]
    cfg = _load_config(search_dirs, cfg_name) if cfg_name else {}
    task_name = task_override or cfg.get("task", rollout_dir.parent.name)

    steps = []
    with open(actions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    all_tokens = [s["token"] for s in steps]

    # Resolve the per-mode context that does not vary inside the loop.
    step_subgoal: dict[int, dict] = {}
    afford: dict[str, str] = {}
    if mode == "subgoal":
        subgoals = _load_subgoals(cfg)
        if not subgoals:
            raise ValueError(
                f"No subgoals in task_config.json for {rollout_dir}. "
                "Run generate_subgoals.py first (or use --vlm-* to auto-generate)."
            )
        step_subgoal = _assign_subgoals_to_steps(all_tokens, subgoals)
        fallback_sg = subgoals[0]
    elif mode == "affordance":
        afford = _load_affordance(cfg)
        if not afford:
            raise ValueError(
                f"No affordance in affordance_config.json for {rollout_dir}. "
                "Run generate_affordance.py first (or use --vlm-* to auto-generate)."
            )

    def _render(step: dict, recent_str: str, sg: dict) -> str:
        gripper_state = "closed" if step.get("gripper_closed") else "open"
        if mode == "subgoal":
            return prompt_template.format(
                task=task_name,
                stage=sg.get("motion", ""),
                target=sg.get("target", ""),
                affordance=sg.get("affordance", ""),
                description=sg.get("description", ""),
                completion=sg.get("completion", ""),
                gripper_state=gripper_state,
                recent_moves=recent_str,
                gripper_color=gripper_color,
            )
        if mode == "affordance":
            return prompt_template.format(
                task=task_name,
                target=afford["target"],
                affordance=afford["affordance"],
                gripper_state=gripper_state,
                recent_moves=recent_str,
            )
        return prompt_template.format(
            task=task_name, gripper_state=gripper_state, recent_moves=recent_str
        )

    def _sample(step: dict, input_text: str, token: str) -> dict:
        views = [
            str((rollout_dir / step["agentview"]).resolve()),
            str((rollout_dir / step["wrist"]).resolve()),
        ]
        if video_slot:
            # "videos" is nested: one frame list per <video> placeholder. Paths must stay
            # absolute -- the converter only prepends media_dir to flat string lists.
            return {
                "instruction": VIDEO_TOKEN + input_text,
                "input": "",
                "output": token,
                "videos": [views],
            }

        return {
            "instruction": IMAGE_TOKENS + input_text,
            "input": "",
            "output": token,
            "images": views,
        }

    samples = []
    mv_history: list[str] = []
    last_action_idx = -1

    for i, step in enumerate(steps):
        token = step["token"]
        if token not in ACTION_TOKENS:
            continue
        last_action_idx = i

        recent_str = ", ".join(mv_history[-RECENT_WINDOW:][::-1]) or "none"
        sg = step_subgoal.get(i, fallback_sg) if mode == "subgoal" else {}
        samples.append(_sample(step, _render(step, recent_str, sg), token))

        if token in MV_TOKENS:
            mv_history.append(token)

    # Terminal DONE: one synthesized sample per episode (no DONE in the recordings). It reuses
    # the last observed frame and the final context (last subgoal in subgoal mode); recent
    # moves now include every move, so the newest is the episode's last MV_*.
    if last_action_idx >= 0:
        last_step = steps[last_action_idx]
        recent_str = ", ".join(mv_history[-RECENT_WINDOW:][::-1]) or "none"
        sg = step_subgoal.get(last_action_idx, fallback_sg) if mode == "subgoal" else {}
        samples.append(_sample(last_step, _render(last_step, recent_str, sg), DONE_TOKEN))

    return samples


# ── Dual-arm conversion ───────────────────────────────────────────────────────

def convert_dual_rollout(
    rollout_dir: Path,
    prompt_template: str,
    scheme: str,
    followup_template: str | None = None,
    task_override: str | None = None,
) -> list[dict]:
    """Convert one dual-arm rollout (three cameras, one action token per arm per step).

    ``scheme`` is the train/inference contract, and each one produces a different sample
    shape from the SAME recording:

      twice — two VLM calls per step. Two Alpaca samples, both carrying all three views; the
              prompt's ``{arm}`` says which arm to answer for. The RIGHT sample does NOT see the
              LEFT token, so the two calls stay conditionally independent given the images and
              can be batched in parallel at inference.
      once  — one VLM call per step. One Alpaca sample whose output is ``"<left> <right>"``. The
              right token is still conditioned on the left one, through the decoder's own
              autoregression, at the cost of a single image forward.
      chain — one image forward, two answers. One ShareGPT sample with two assistant turns:
              the LEFT token, a short follow-up user turn, then the RIGHT token. Same sequential
              conditioning as ``twice`` but the three views are encoded only once.

    Terminal DONE follows the single-arm convention: the recordings contain no DONE, so one
    sample is synthesized per episode on the last observed frame, with BOTH arms DONE (``once``
    emits ``"DONE DONE"``), teaching "the whole task is complete".
    """
    actions_path = rollout_dir / "actions.jsonl"
    if not actions_path.exists():
        print(f"[skip] no actions.jsonl in {rollout_dir}", file=sys.stderr)
        return []

    task_name = task_override or rollout_dir.parent.name

    steps = []
    with open(actions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    def _views(step: dict) -> list[str]:
        return [
            str((rollout_dir / step["agentview"]).resolve()),
            str((rollout_dir / step["wrist_left"]).resolve()),
            str((rollout_dir / step["wrist_right"]).resolve()),
        ]

    def _render(history: dict[str, list[str]], arm: str) -> str:
        # once/chain templates have no {arm} field; the extra kwarg is simply ignored by format().
        return prompt_template.format(
            task=task_name,
            recent_left=", ".join(history["left"][-RECENT_WINDOW:][::-1]) or "none",
            recent_right=", ".join(history["right"][-RECENT_WINDOW:][::-1]) or "none",
            arm=arm.upper(),
        )

    def _emit(step: dict, history: dict[str, list[str]], tok_l: str, tok_r: str) -> list[dict]:
        views = _views(step)
        if scheme == "twice":
            return [
                {
                    "instruction": DUAL_IMAGE_TOKENS + _render(history, arm),
                    "input": "",
                    "output": token,
                    "images": views,
                }
                for arm, token in (("left", tok_l), ("right", tok_r))
            ]
        if scheme == "once":
            return [{
                "instruction": DUAL_IMAGE_TOKENS + _render(history, ""),
                "input": "",
                "output": f"{tok_l} {tok_r}",
                "images": views,
            }]
        return [{
            "messages": [
                {"role": "user", "content": DUAL_IMAGE_TOKENS + _render(history, "")},
                {"role": "assistant", "content": tok_l},
                {"role": "user", "content": followup_template},
                {"role": "assistant", "content": tok_r},
            ],
            "images": views,
        }]

    samples: list[dict] = []
    history: dict[str, list[str]] = {"left": [], "right": []}
    last_action_idx = -1

    for i, step in enumerate(steps):
        tok_l = step["left"]["token"]
        tok_r = step["right"]["token"]
        if tok_l not in DUAL_ACTION_TOKENS or tok_r not in DUAL_ACTION_TOKENS:
            continue
        last_action_idx = i

        # Render against the history BEFORE this step, then fold this step into it.
        samples.extend(_emit(step, history, tok_l, tok_r))
        for arm, token in (("left", tok_l), ("right", tok_r)):
            if token in DUAL_HISTORY_TOKENS:
                history[arm].append(token)

    if last_action_idx >= 0:
        samples.extend(_emit(steps[last_action_idx], history, DONE_TOKEN, DONE_TOKEN))

    return samples


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert rollouts to LLaMA Factory Alpaca format")
    parser.add_argument(
        "rollout_dirs", nargs="+", type=Path,
        help="Rollout directories (or parent directories containing rollout subdirs)",
    )
    parser.add_argument(
        "--version", required=True,
        help="Prompt version subfolder under <repo>/prompts/ (v3 unified, v4 per-embodiment). "
             "The per-mode template is read from prompts/<version>/<mode_file>: "
             "lite -> mvtoken_generator_lite.txt, affordance -> mvtoken_generator_affordance.txt, "
             "subgoal -> mvtoken_generator.txt.",
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--franka", action="store_const", const="franka", dest="embodiment_view",
        help="Lite mode: use the Franka (exocentric) prompt prompts/<version>/franka_mvtoken_lite.txt.",
    )
    view_group.add_argument(
        "--piper", action="store_const", const="piper", dest="embodiment_view",
        help="Lite mode: use the Piper (egocentric) prompt prompts/<version>/piper_mvtoken_lite.txt.",
    )
    view_group.add_argument(
        "--dual", action="store_const", const="dual", dest="embodiment_view",
        help="Dual-arm (piper-dual): three cameras (agentview + wrist_left + wrist_right) and one "
             "token per arm per step. Requires one of --twice / --once / --chain, which selects "
             "prompts/<version>/dual_mvtoken_<scheme>.txt AND the sample shape.",
    )
    scheme_group = parser.add_mutually_exclusive_group()
    scheme_group.add_argument(
        "--twice", action="store_const", const="twice", dest="dual_scheme",
        help="--dual: TWO VLM calls per step (two Alpaca samples, both with all three views; the "
             "prompt says which arm to answer for). The right call does not see the left token, "
             "so the two calls can be batched in parallel at inference.",
    )
    scheme_group.add_argument(
        "--once", action="store_const", const="once", dest="dual_scheme",
        help="--dual: ONE VLM call per step emitting both tokens as '<left> <right>' (one Alpaca "
             "sample). Cheapest; the right token is conditioned on the left via autoregression.",
    )
    scheme_group.add_argument(
        "--chain", action="store_const", const="chain", dest="dual_scheme",
        help="--dual: ONE image forward, TWO answers -- a ShareGPT sample with two assistant "
             "turns (left token, follow-up user turn, right token). Register the output with "
             "formatting=sharegpt + columns.messages, NOT the Alpaca columns.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Defaults to rollout[_subgoal].json inside the rollout dir "
             "(single rollout) or a 'process_out' folder under the task folders' parent dir "
             "(multiple rollouts). The config (task/affordance) is written under this json's "
             "folder: directly beside it for a single rollout, or in a per-task subfolder "
             "(named after the source task folder) for a multi-folder run.",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description string applied to all rollouts (overridden per-folder by --task-map).",
    )
    parser.add_argument(
        "--task-map", nargs="+", metavar="DIRNAME=TASK", default=None,
        help="Map folder names to task descriptions, e.g. "
             "'banana=pick up the banana' 'mango=pick up the mango'. "
             "Takes priority over --task for matching folders.",
    )
    parser.add_argument(
        "--jsonl", action="store_true", default=False,
        help="Write output as JSON Lines (.jsonl) instead of a JSON array.",
    )
    parser.add_argument(
        "--video-slot", action="store_true", default=False,
        help="Emit the two views as video frames ('<video>' + 'videos') instead of two images. "
             "Qwen fuses every 2 frames into one temporal patch, so the sample costs half the "
             "visual tokens (64 vs 128 at 256x256) but the two viewpoints get mixed. "
             "Training yaml must set video_fps (it decides the '<0.2 seconds>' timestamp text) "
             "and video_max_pixels; eval must use the same values.",
    )
    parser.add_argument(
        "--use-subgoal", action="store_true", default=False,
        help="Use the full prompt (mvtoken_generator.txt) with per-step VLM subgoal info. "
             "Reads task_config.json; auto-generates it via generate_subgoals.py if absent. "
             "Default (off) uses the stage-free mvtoken_generator_lite.txt prompt.",
    )
    parser.add_argument(
        "--use-affordance", action="store_true", default=False,
        help="Use the lite+affordance prompt (mvtoken_generator_affordance.txt) with a single "
             "grasp-point hint (target + affordance) reused for every step. Reads "
             "affordance_config.json; auto-generates it via generate_affordance.py if absent. "
             "Mutually exclusive with --use-subgoal.",
    )
    parser.add_argument(
        "--gripper-color", default="black",
        help="Gripper color hint used by the full prompt (--use-subgoal only; default: black).",
    )
    # VLM forwarding args — only needed when auto-generating task/affordance config
    parser.add_argument("--vlm-backend", default=None, help="VLM backend for subgoal generation")
    parser.add_argument("--vlm-url", default=None, help="Override VLM base URL")
    parser.add_argument("--model", default=None, help="Override model name or path")
    parser.add_argument("--api-key", default=None, help="API key for VLM server")
    args = parser.parse_args()

    if args.use_subgoal and args.use_affordance:
        parser.error("--use-subgoal and --use-affordance are mutually exclusive.")
    mode = "subgoal" if args.use_subgoal else "affordance" if args.use_affordance else "lite"

    dual = args.embodiment_view == "dual"
    if dual:
        if not args.dual_scheme:
            parser.error("--dual requires exactly one of --twice / --once / --chain.")
        if mode != "lite":
            parser.error("--dual supports the lite prompt only (no --use-subgoal / --use-affordance).")
        if args.video_slot:
            parser.error(
                "--dual is incompatible with --video-slot: three views cannot be paired into "
                "Qwen's 2-frame temporal patches."
            )
    elif args.dual_scheme:
        parser.error("--twice / --once / --chain are only meaningful with --dual.")

    rollout_dirs = _expand_dirs(args.rollout_dirs)
    if not rollout_dirs:
        print("No valid rollout directories found.", file=sys.stderr)
        sys.exit(1)

    ext = ".jsonl" if args.jsonl else ".json"
    stem = {"subgoal": "rollout_subgoal", "affordance": "rollout_affordance"}.get(
        mode, "rollout"
    )
    if dual:
        stem = f"rollout_dual_{args.dual_scheme}"
    default_filename = stem + ext
    single = len(rollout_dirs) == 1
    if args.output is not None:
        output_path = args.output
    elif single:
        # Single rollout: write next to the rollout (its own folder).
        output_path = rollout_dirs[0] / default_filename
    else:
        # Multiple: a 'process_out' folder under the task folders' shared parent.
        output_path = rollout_dirs[0].parent.parent / "process_out" / default_filename

    # Build task map: dirname -> task description
    task_map: dict[str, str] = {}
    for entry in (args.task_map or []):
        if "=" not in entry:
            parser.error(f"--task-map entry must be 'dirname=task description', got: {entry!r}")
        dirname, _, task_desc = entry.partition("=")
        task_map[dirname.strip()] = task_desc.strip()

    followup_template: str | None = None
    if dual:
        # --twice / --once / --chain select BOTH the prompt and the sample shape.
        prompt_template = _load_prompt(args.version, f"dual_mvtoken_{args.dual_scheme}.txt")
        if args.dual_scheme == "chain":
            followup_template = _load_prompt(args.version, "dual_mvtoken_chain_right.txt").strip()
    elif mode == "lite" and args.embodiment_view:
        # --franka / --piper select the embodiment-specific lite prompt under prompts/<version>/.
        prompt_template = _load_prompt(args.version, f"{args.embodiment_view}_mvtoken_lite.txt")
    else:
        prompt_filename = {
            "subgoal": "mvtoken_generator.txt",
            "affordance": "mvtoken_generator_affordance.txt",
        }.get(mode, "mvtoken_generator_lite.txt")
        prompt_template = _load_prompt(args.version, prompt_filename)

    # Build VLM forwarding args for the subgoal-generation subprocess.
    vlm_args: list[str] = []
    for flag, value in (
        ("--vlm-backend", args.vlm_backend),
        ("--vlm-url", args.vlm_url),
        ("--model", args.model),
        ("--api-key", args.api_key),
    ):
        if value:
            vlm_args += [flag, value]

    def _resolve_task(rollout_dir: Path) -> str | None:
        """Return task string for this rollout: task_map > --task > None."""
        return task_map.get(rollout_dir.parent.name) or args.task or None

    def _config_dir(rollout_dir: Path) -> Path:
        """Where this rollout's config lives: next to the output json (single) or in a per-task
        subfolder of the output dir, named after the source task folder (multi)."""
        return output_path.parent if single else output_path.parent / rollout_dir.parent.name

    def _src_path(rollout_dir: Path) -> Path:
        """Image source for config generation: the rollout itself (single) or its task folder,
        which generates once from its first rollout and is reused (multi)."""
        return rollout_dir if single else rollout_dir.parent

    # Ensure the per-rollout config exists (auto-generates via the matching VLM script).
    if mode == "subgoal":
        for d in rollout_dirs:
            _ensure_task_config(d, _config_dir(d), _src_path(d), _resolve_task(d), vlm_args)
    elif mode == "affordance":
        for d in rollout_dirs:
            _ensure_affordance_config(d, _config_dir(d), _src_path(d), _resolve_task(d), vlm_args)

    all_samples: list[dict] = []
    for d in rollout_dirs:
        if dual:
            samples = convert_dual_rollout(
                d,
                prompt_template=prompt_template,
                scheme=args.dual_scheme,
                followup_template=followup_template,
                task_override=_resolve_task(d),
            )
        else:
            samples = convert_rollout(
                d,
                prompt_template=prompt_template,
                mode=mode,
                task_override=_resolve_task(d),
                gripper_color=args.gripper_color,
                config_dir=_config_dir(d),
                video_slot=args.video_slot,
            )
        print(f"{d.name}: {len(samples)} samples")
        all_samples.extend(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if args.jsonl:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(all_samples)} samples -> {output_path}")


if __name__ == "__main__":
    main()
