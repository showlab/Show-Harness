#!/usr/bin/env python3
"""Offline check for the affordance capability: one frame + a task -> annotated image(s).

Runs the exact pointing role the rollout uses (point call + draw-and-verify loop) on a
saved front-view frame, draws the returned dot(s), and writes the annotated image(s) --
the fastest way to eyeball whether affordance grounding is precise BEFORE spending a
robot episode on it. No robot, no cameras; only a VLM endpoint.

Three modes:
  * --plan (the real pipeline): runs the ACTUAL subgoal planner on the frame (dual
    tracks by default, --single for the single-arm planner), then grounds EVERY spatial
    stage of every track through the same AffordancePlugin the runner mounts -- one
    annotated image per stage plus the step-0 composite (both arms' first dots on one
    frame) and the exact rewritten AFFORD fields. Wrist frames are auto-found
    next to a rollout's ``images/agentview/NNNN.png`` (the concatenated
    ``images/wrist/NNNN.png``) or passed with --wrist. NOTE: every stage is grounded
    on the INITIAL frame -- destinations (plates, boxes) are static so their dots are
    truthful; a later stage whose target has moved by then only checks the pointer's
    part choice, not the live pixel.
  * task mode (default): the pointer marks the contact point(s) the task's NEXT
    manipulation needs (1-2 points, e.g. two arms acting at once = one each).
  * stage mode (--target, plus optional --afford/--motion/--arm): simulates one
    per-stage call by hand, exactly one point with abstain allowed.

Usage:
  python -m plugins.affordance.test_affordance --image frame.png \
      --task "Left arm hands the banana over to the right arm" --plan
  python -m plugins.affordance.test_affordance --image frame.png --task "..." \
      --target banana --afford "left end" --arm left

The VLM backend comes from the robot config (default configs/robot_piper.yaml,
override with --backend); keys resolve from configs/secrets.env like a real run.

Exit code: --plan -> 0 once a plan was produced; task mode -> 0 when at least one
point was grounded; stage mode -> 0 on a point (or a deliberate abstain with
--allow-abstain); else 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import (
    load_secrets_env,
    load_yaml,
    make_api_key_refresher,
    resolve_vlm_config,
)
from core.record.images import save_png, to_uint8_hwc
from plugins.affordance.agent import AffordancePointerAgent, draw_point
from plugins.affordance.plugin import DOT_COLORS, AffordancePlugin

TASK_INSTRUCTION = "the contact point(s) for the task's NEXT manipulation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="front-view frame (png/jpg)")
    parser.add_argument("--task", required=True, help="the task prompt")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="run the REAL pipeline: subgoal planner -> per-stage affordance "
        "grounding of every track (one image per spatial stage + step-0 composite)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="--plan with the single-arm planner instead of the dual one",
    )
    parser.add_argument(
        "--wrist",
        default=None,
        help="--plan: wrist frame for the planner (a rollout's concatenated "
        "left|right image is split; auto-discovered from images/wrist/ when omitted)",
    )
    parser.add_argument(
        "--target", default="", help="stage mode: the stage's target object/destination"
    )
    parser.add_argument(
        "--afford", default="", help="stage mode: the stage's affordance text"
    )
    parser.add_argument("--motion", default="GRASP", help="stage mode: stage label")
    parser.add_argument(
        "--arm",
        default="arm",
        choices=sorted(DOT_COLORS),
        help="stage mode: which arm's dot color/wording to use",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "robot_piper.yaml"),
        help="robot config supplying the vlm_backends block",
    )
    parser.add_argument("--backend", default=None, help="override cfg vlm_backend")
    parser.add_argument(
        "--verify-rounds",
        type=int,
        default=1,
        help="self-verification rounds per point (0 = trust the first answer)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output file OR directory (default: <image>_affordance.png beside the "
        "input; a directory gets default filenames inside it; --plan always "
        "treats it as a directory)",
    )
    parser.add_argument(
        "--allow-abstain",
        action="store_true",
        help="stage mode: exit 0 when the pointer deliberately returns no point",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _make_client(cfg_path: str, backend: str | None):
    load_secrets_env()
    cfg = load_yaml(cfg_path)
    vlm_cfg = resolve_vlm_config(cfg, backend=backend)
    from core.vlm.vlm_client import VLMClient

    print(
        f"backend  {vlm_cfg.get('backend')} / {vlm_cfg['model']} "
        f"({vlm_cfg.get('base_url')})"
    )
    return VLMClient(
        base_url=vlm_cfg["base_url"],
        model=vlm_cfg["model"],
        api_key=vlm_cfg.get("api_key", "EMPTY"),
        timeout_s=float(vlm_cfg.get("timeout_s", 120)),
        max_tokens=int(vlm_cfg.get("max_tokens", 2048)),
        temperature=float(vlm_cfg.get("temperature", 0.0)),
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
        provider=vlm_cfg.get("provider", "vllm"),
        api_dialect=vlm_cfg.get("api_dialect"),
        reasoning_effort=vlm_cfg.get("reasoning_effort"),
        max_retries=vlm_cfg.get("max_retries"),
        api_key_refresh=make_api_key_refresher(vlm_cfg),
    )


def _load_frame(path: Path) -> np.ndarray:
    return to_uint8_hwc(np.asarray(imageio.imread(path))[..., :3])


def _resolve_out_path(out_arg: str | None, image_path: Path) -> Path:
    """--out as a file or a directory; default beside the input image.

    A path that exists as a directory, or has no image suffix, is treated as a
    directory and receives the default ``<stem>_affordance.png`` name inside it
    (PIL needs a real extension to pick the save format).
    """
    default_name = f"{image_path.stem}_affordance.png"
    if not out_arg:
        return image_path.with_name(default_name)
    out = Path(out_arg)
    if out.is_dir() or not out.suffix:
        return out / default_name
    return out


def _resolve_out_dir(out_arg: str | None, image_path: Path) -> Path:
    """--plan writes several files, so --out is always a directory here (a file-like
    path contributes its parent)."""
    if not out_arg:
        return image_path.parent
    out = Path(out_arg)
    return out.parent if out.suffix else out


# -- plan mode -------------------------------------------------------------------
def _find_wrists(
    image_path: Path, wrist_arg: str | None, single: bool
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The planner's wrist view(s): --wrist, or the rollout layout's concatenated
    ``images/wrist/NNNN.png`` sibling of ``images/agentview/NNNN.png``.

    Dual rollouts store [left|right] side by side (episode_logger), so the frame is
    split at the horizontal midpoint; --single uses it whole. Returns (left, right)
    with right=None for single; (None, None) when nothing is found.
    """
    wrist_path: Optional[Path] = Path(wrist_arg) if wrist_arg else None
    if wrist_path is None and image_path.parent.name == "agentview":
        candidate = image_path.parent.parent / "wrist" / image_path.name
        wrist_path = candidate if candidate.is_file() else None
    if wrist_path is None:
        return None, None
    frame = _load_frame(wrist_path)
    if single:
        print(f"wrist    {wrist_path.name}  {frame.shape[1]}x{frame.shape[0]}")
        return frame, None
    half = frame.shape[1] // 2
    print(f"wrist    {wrist_path.name}  split at x={half} into left|right")
    return frame[:, :half], frame[:, half:]


def _plan_tracks(
    args: argparse.Namespace, client: Any, frame: np.ndarray, image_path: Path
) -> dict[str, list]:
    """The REAL planner call: dual tracks {'left','right'} or single {'arm'}."""
    common_path = ROOT / "prompts" / "common_context.txt"
    common_context = (
        common_path.read_text(encoding="utf-8").strip() if common_path.is_file() else ""
    )
    wrist_left, wrist_right = _find_wrists(image_path, args.wrist, args.single)
    if wrist_left is None:
        print("wrist    none found -- the planner sees only the front view "
              "(arm assignment may be weaker than a real run)")
    if args.single:
        from plugins.subgoal import SubgoalPlanner
        from plugins.subgoal.agent import SubgoalPlannerAgent

        planner = SubgoalPlanner(
            SubgoalPlannerAgent(client=client, common_context=common_context)
        )
        subgoals, _raw = planner.plan(args.task, frame, wrist=wrist_left, debug=args.debug)
        return {"arm": subgoals}
    from plugins.subgoal.dual_agent import DualSubgoalPlannerAgent
    from plugins.subgoal.dual_plugin import DualSubgoalPlanner

    planner = DualSubgoalPlanner(
        DualSubgoalPlannerAgent(client=client, common_context=common_context)
    )
    tracks, _raw = planner.plan(
        args.task, frame, wrist_left=wrist_left, wrist_right=wrist_right, debug=args.debug
    )
    return dict(tracks)


def _print_plan(tracks: dict[str, list]) -> None:
    print("plan")
    for slot, subgoals in tracks.items():
        if not subgoals:
            print(f"  {slot:<5} (empty track)")
            continue
        for i, sg in enumerate(subgoals):
            print(
                f"  {slot:<5} {i + 1}/{len(subgoals)}  {sg.motion:<8} "
                f"{sg.target}  ·  {sg.affordance}"
            )


def _run_plan_mode(
    args: argparse.Namespace, client: Any, frame: np.ndarray, image_path: Path
) -> int:
    tracks = _plan_tracks(args, client, frame, image_path)
    _print_plan(tracks)
    tool = AffordancePlugin(
        enabled=True,
        client=client,
        verify_rounds=args.verify_rounds,
        view_name="AgentView" if args.single else "Front View",
    )
    spatial = sum(
        1 for sgs in tracks.values() for sg in sgs if tool.wants(sg.to_prompt_dict())
    )
    print(f"grounding {spatial} spatial stage(s), ~2 VLM calls each\n")

    out_dir = _resolve_out_dir(args.out, image_path)
    stem = image_path.stem
    points: dict[tuple[str, int], Optional[dict[str, Any]]] = {}

    def ground(slot: str, i: int) -> None:
        subgoal = tracks[slot][i].to_prompt_dict()
        if not tool.wants(subgoal):
            print(
                f"  [affordance] {slot} · {subgoal.get('motion')} "
                f"{subgoal.get('target')} -> skipped (non-spatial stage)"
            )
            points[(slot, i)] = None
            return
        tool.ensure(slot, f"{slot}:{i}", args.task, subgoal, frame, debug=args.debug)
        point = tool.point_of(slot)
        points[(slot, i)] = point
        if point is not None:
            y, x = point["point"]
            stage_img = draw_point(frame, x, y, DOT_COLORS.get(slot, DOT_COLORS["arm"])[1])
            stage_path = out_dir / f"{stem}_{slot}{i}_{subgoal.get('motion', '?')}.png"
            save_png(stage_path, stage_img)
            print(f"           saved {stage_path}")

    # Phase 1 -- stage 0 of every track, then freeze the step-0 composite: exactly
    # the dots (and rewritten AFFORD fields) the controller sees on the first step;
    # an arm whose track STARTS with WAIT correctly has none.
    for slot in tracks:
        if tracks[slot]:
            ground(slot, 0)
    composite = tool.annotate(frame)
    if not np.array_equal(composite, frame):
        step0_path = out_dir / f"{stem}_step0.png"
        save_png(step0_path, composite)
        print(f"\nstep0    {step0_path}")
    for slot in tracks:
        if tracks[slot] and tool.point_of(slot) is not None:
            afford = tracks[slot][0].affordance
            print(f"AFFORD   [{slot}] {tool.afford_field(slot, afford)}")
    print()

    # Phase 2 -- sweep the remaining stages (all grounded on the INITIAL frame; see
    # the module docstring for what that does and does not verify).
    for slot in tracks:
        for i in range(1, len(tracks[slot])):
            ground(slot, i)

    grounded = sum(1 for p in points.values() if p)
    print(f"\nresult   {grounded}/{spatial} spatial stage(s) grounded")
    return 0


# -- task / stage modes ------------------------------------------------------------
def _run_point_mode(
    args: argparse.Namespace, client: Any, frame: np.ndarray, image_path: Path
) -> int:
    agent = AffordancePointerAgent(client, verify_rounds=args.verify_rounds)
    stage_mode = bool(args.target)
    if stage_mode:
        arm = f"the {args.arm.upper()} arm's " if args.arm in ("left", "right") else "the arm's "
        instruction = f"{arm}{args.motion} stage -- target: {args.target}"
        if args.afford:
            instruction += f"; contact part: {args.afford}"
        color_name, color = DOT_COLORS[args.arm]
        max_points = 1
    else:
        instruction = TASK_INSTRUCTION
        color_name, color = DOT_COLORS["arm"]
        max_points = 2
    print(f"mode     {'stage' if stage_mode else 'task'}  ·  {instruction}")

    points = agent.locate(
        task=args.task,
        instruction=instruction,
        agentview=frame,
        max_points=max_points,
        color=color,
        color_name=color_name,
        debug=args.debug,
    )
    if not points:
        print("result   NO POINT (the pointer abstained or every call failed)")
        return 0 if (stage_mode and args.allow_abstain) else 2

    annotated = frame
    for i, point in enumerate(points):
        # Task mode with two points: second dot in blue so both stay distinguishable.
        _, dot_color = (color_name, color) if i == 0 else DOT_COLORS["right"]
        annotated = draw_point(annotated, point.x, point.y, dot_color)
        print(f"point    {json.dumps(point.to_dict(), ensure_ascii=False)}")

    out_path = _resolve_out_path(args.out, image_path)
    save_png(out_path, annotated)
    print(f"saved    {out_path}")
    return 0


def main() -> int:
    args = _parse_args()
    image_path = Path(args.image)
    frame = _load_frame(image_path)
    print(f"frame    {image_path.name}  {frame.shape[1]}x{frame.shape[0]}")
    print(f"task     {args.task}")

    client = _make_client(args.config, args.backend)
    if args.plan or args.single:
        return _run_plan_mode(args, client, frame, image_path)
    return _run_point_mode(args, client, frame, image_path)


if __name__ == "__main__":
    sys.exit(main())
