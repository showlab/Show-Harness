#!/usr/bin/env python3
"""Replay atomic actions from a recorded real-Franka rollout.

The positional argument is a rollout directory containing ``steps.jsonl`` (the
real-runner format) or ``actions.jsonl`` (the teleoperation format).  Records are
executed in file order through the same :class:`FrankaAtomicController` used by
live rollouts.  The front and wrist cameras remain live in the existing pygame
dashboard while actions execute and while the replay waits between actions.

Examples
--------
# Inspect the resolved source and actions without connecting to hardware.
python scripts/trajectory/replay_rollout.py /path/to/run --dry-run

# Replay on the real Franka.  The default delay between records is uniform in [3, 5] s.
python scripts/trajectory/replay_rollout.py /path/to/run

# End-to-end software check without real hardware or cameras.
python scripts/trajectory/replay_rollout.py /path/to/run \
  --mock-robot --mock-cameras --no-show --pause-min-s 0 --pause-max-s 0

Important: atomic movements are relative to the robot pose at replay startup.
Place the arm, scene, and gripper in the same initial state as the recording.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Optional, Sequence

# File is scripts/trajectory/replay_rollout.py, so repo root is two levels above.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.action_units import MOVE_ATOMS, ROTATE_ATOMS, STOP_ATOM  # noqa: E402
from interpreters.franka_atomic_controller import (  # noqa: E402
    EMPTY_GRASP_WIDTH_M,
    FrankaAtomicController,
)
from interpreters.real_atomic_controller import (  # noqa: E402
    DEFAULT_MAX_POSITION_DELTA_M,
    DEFAULT_MAX_ROTATION_DELTA_RAD,
    DONE_ATOM,
    GRIPPER_ATOMS,
)
from core.config import load_yaml  # noqa: E402
from core.franka.franka_session import (  # noqa: E402
    FrankaSession,
    FrankaSessionConfig,
)
from core.ui.live_view import LiveView  # noqa: E402


DEFAULT_PRIMITIVES = ROOT / "configs" / "primitives_franka.yaml"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot_franka.yaml"
SUPPORTED_TOKENS = frozenset(
    MOVE_ATOMS + ROTATE_ATOMS + GRIPPER_ATOMS + (STOP_ATOM, DONE_ATOM)
)


@dataclass(frozen=True)
class ReplayStep:
    """One validated JSONL record in replay order."""

    ordinal: int
    source_line: int
    recorded_index: Optional[int]
    token: str
    stage: str = "-"
    step_override_m: Optional[float] = None


@dataclass(frozen=True)
class ReplayPlan:
    rollout_dir: Path
    source_path: Path
    source_kind: str
    steps: tuple[ReplayStep, ...]
    metadata: dict[str, Any]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay steps.jsonl/actions.jsonl atomic actions on the real Franka."
    )
    parser.add_argument(
        "rollout_dir",
        help="Directory containing steps.jsonl or actions.jsonl.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "steps", "actions"),
        default="auto",
        help="JSONL source. auto prefers steps.jsonl when both exist (default: auto).",
    )
    parser.add_argument(
        "--pause-min-s",
        type=float,
        default=3.0,
        help="Minimum random pause between adjacent JSONL records (default: 3).",
    )
    parser.add_argument(
        "--pause-max-s",
        type=float,
        default=5.0,
        help="Maximum random pause between adjacent JSONL records (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible pause durations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate the replay without connecting to robot or cameras.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Disable the live camera window."
    )

    # Hardware/session settings mirror collect_rollouts.py and scripts/run_real.py.
    parser.add_argument("--mock-robot", action="store_true", help="Use MockRobot.")
    parser.add_argument("--mock-cameras", action="store_true", help="Use mock cameras.")
    parser.add_argument("--nuc-ip", default=None, help="Override robot config NUC IP.")
    parser.add_argument(
        "--nuc-port", type=int, default=None, help="Override NUC ZeroRPC port."
    )
    parser.add_argument(
        "--no-impedance",
        action="store_true",
        help="Do not start/recover Cartesian impedance (normally required for movement).",
    )
    parser.add_argument(
        "--robot-config",
        default=str(DEFAULT_ROBOT_CONFIG),
        help="Current Franka hardware/camera/safety config.",
    )
    parser.add_argument(
        "--primitives-config",
        default=None,
        help="Explicit primitives YAML. By default use embedded rollout primitives, "
        "falling back to configs/primitives_franka.yaml.",
    )

    # Reproduction overrides.  Per-record steps.jsonl step_cm wins unless --step-m is set.
    parser.add_argument(
        "--step-m",
        type=float,
        default=None,
        help="Override every MV_* distance; otherwise preserve recorded step_cm when present.",
    )
    parser.add_argument(
        "--yaw-step-rad",
        type=float,
        default=None,
        help="Override the recorded/configured ROTATE_* increment.",
    )
    parser.add_argument(
        "--max-translation-m",
        type=float,
        default=None,
        help="Per-command translation safety limit. Defaults to the primitives config limit.",
    )
    parser.add_argument(
        "--z-floor-m",
        type=float,
        default=None,
        help="Override the current robot config's minimum EEF Z height.",
    )
    parser.add_argument(
        "--no-z-floor",
        action="store_true",
        help="Disable the Z safety floor (not recommended on hardware).",
    )
    return parser.parse_args(argv)


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


def validate_pause_range(min_s: float, max_s: float) -> tuple[float, float]:
    min_s = _finite_nonnegative(min_s, "pause-min-s")
    max_s = _finite_nonnegative(max_s, "pause-max-s")
    if min_s > max_s:
        raise ValueError(f"pause-min-s ({min_s}) must be <= pause-max-s ({max_s})")
    return min_s, max_s


def resolve_source_path(rollout_dir: Path, source: str = "auto") -> tuple[Path, str]:
    """Resolve a directory to one JSONL source without searching child folders."""
    rollout_dir = Path(rollout_dir).expanduser().resolve()
    if not rollout_dir.is_dir():
        raise ValueError(f"rollout directory does not exist: {rollout_dir}")

    candidates = {
        "steps": rollout_dir / "steps.jsonl",
        "actions": rollout_dir / "actions.jsonl",
    }
    if source == "auto":
        # Real-runner output is richer (per-record step_cm and stage), so it is the
        # deterministic choice if a directory happens to contain both formats.
        source = "steps" if candidates["steps"].is_file() else "actions"
    path = candidates[source]
    if not path.is_file():
        available = [
            name for name, candidate in candidates.items() if candidate.is_file()
        ]
        suffix = f"; available sources: {available}" if available else ""
        raise ValueError(f"missing {path.name} in {rollout_dir}{suffix}")
    return path, source


def _optional_recorded_index(
    record: dict[str, Any], key: str, path: Path, line_no: int
) -> Optional[int]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}:{line_no}: {key!r} must be an integer")
    return int(value)


def load_replay_plan(rollout_dir: str | Path, source: str = "auto") -> ReplayPlan:
    """Load and strictly validate an atomic-action replay plan.

    JSONL file order is authoritative.  If every record supplies ``i``/``step``, the
    indices must be strictly increasing; they are validated but never used to reorder
    the file.  This makes malformed or concatenated logs fail before hardware connects.
    """
    source_path, source_kind = resolve_source_path(Path(rollout_dir), source)
    token_key = "act" if source_kind == "steps" else "token"
    index_key = "i" if source_kind == "steps" else "step"
    steps: list[ReplayStep] = []

    with source_path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source_path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source_path}:{line_no}: expected a JSON object")

            raw_token = record.get(token_key)
            if not isinstance(raw_token, str) or not raw_token.strip():
                raise ValueError(
                    f"{source_path}:{line_no}: missing non-empty {token_key!r} action"
                )
            token = raw_token.strip().upper()
            if token not in SUPPORTED_TOKENS:
                raise ValueError(
                    f"{source_path}:{line_no}: unsupported action {token!r}; "
                    f"expected one of {sorted(SUPPORTED_TOKENS)}"
                )

            recorded_index = _optional_recorded_index(
                record, index_key, source_path, line_no
            )
            step_override_m: Optional[float] = None
            if (
                source_kind == "steps"
                and token in MOVE_ATOMS
                and record.get("step_cm") is not None
            ):
                try:
                    step_cm = float(record["step_cm"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{source_path}:{line_no}: step_cm must be numeric"
                    ) from exc
                if not math.isfinite(step_cm) or step_cm <= 0.0:
                    raise ValueError(
                        f"{source_path}:{line_no}: step_cm must be finite and > 0"
                    )
                step_override_m = step_cm / 100.0

            steps.append(
                ReplayStep(
                    ordinal=len(steps),
                    source_line=line_no,
                    recorded_index=recorded_index,
                    token=token,
                    stage=str(record.get("stage") or "-"),
                    step_override_m=step_override_m,
                )
            )

    if not steps:
        raise ValueError(f"no actions found in {source_path}")

    present_indices = [
        step.recorded_index for step in steps if step.recorded_index is not None
    ]
    if present_indices and len(present_indices) != len(steps):
        raise ValueError(
            f"{source_path}: {index_key!r} must be present on every record or none"
        )
    if any(right <= left for left, right in zip(present_indices, present_indices[1:])):
        raise ValueError(
            f"{source_path}: {index_key!r} values must be strictly increasing"
        )

    metadata_path = source_path.parent / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{metadata_path}: invalid JSON: {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{metadata_path}: expected a JSON object")
        metadata = loaded

    return ReplayPlan(
        rollout_dir=source_path.parent,
        source_path=source_path,
        source_kind=source_kind,
        steps=tuple(steps),
        metadata=metadata,
    )


def resolve_primitives_config(
    plan: ReplayPlan,
    explicit_path: Optional[str] = None,
    *,
    step_m: Optional[float] = None,
    yaw_step_rad: Optional[float] = None,
) -> dict[str, Any]:
    """Resolve action axes and magnitudes, preferring the recording's snapshot."""
    embedded = plan.metadata.get("primitives_config")
    if explicit_path is not None:
        primitives = load_yaml(explicit_path)
        origin = str(Path(explicit_path).expanduser().resolve())
    elif isinstance(embedded, dict):
        # Round-trip through JSON to detach nested dictionaries from metadata without
        # adding a deepcopy dependency or mutating the loaded metadata object.
        primitives = json.loads(json.dumps(embedded))
        origin = "metadata.json:primitives_config"
    else:
        primitives = load_yaml(DEFAULT_PRIMITIVES)
        origin = str(DEFAULT_PRIMITIVES)

    # Older teleop metadata stores the magnitudes at top level rather than embedding
    # its primitives snapshot.  Preserve those values when there is no CLI override.
    if explicit_path is None and not isinstance(embedded, dict):
        if plan.metadata.get("step_m") is not None:
            primitives["step_m"] = float(plan.metadata["step_m"])
        if plan.metadata.get("yaw_step_rad") is not None:
            primitives["yaw_step_rad"] = float(plan.metadata["yaw_step_rad"])
    if step_m is not None:
        if not math.isfinite(step_m) or step_m <= 0.0:
            raise ValueError("step-m must be finite and > 0")
        primitives["step_m"] = float(step_m)
    if yaw_step_rad is not None:
        if not math.isfinite(yaw_step_rad) or yaw_step_rad <= 0.0:
            raise ValueError("yaw-step-rad must be finite and > 0")
        primitives["yaw_step_rad"] = float(yaw_step_rad)
    primitives["_replay_origin"] = origin
    return primitives


def resolve_z_floor(
    robot_cfg: dict[str, Any], args: argparse.Namespace
) -> tuple[Optional[float], bool, str]:
    """Resolve the active safety floor from current hardware config, never old metadata."""
    if args.no_z_floor:
        return None, False, "disabled by --no-z-floor"
    if args.z_floor_m is not None:
        value = float(args.z_floor_m)
        if not math.isfinite(value):
            raise ValueError("z-floor-m must be finite")
        return value, False, "--z-floor-m"
    if not bool(robot_cfg.get("enable_z_floor", False)):
        return None, False, "disabled by robot config"

    floors = robot_cfg.get("z_floors") or {}
    name = robot_cfg.get("z_floor_name")
    if name is not None and isinstance(floors, dict):
        if name not in floors:
            raise ValueError(
                f"robot config z_floor_name {name!r} is missing from z_floors"
            )
        value = float(floors[name])
        if not math.isfinite(value):
            raise ValueError(f"z_floors.{name} must be finite")
        return value, False, f"robot config z_floors.{name}"
    if robot_cfg.get("z_floor_m") is not None:
        value = float(robot_cfg["z_floor_m"])
        if not math.isfinite(value):
            raise ValueError("robot config z_floor_m must be finite")
        return value, False, "robot config z_floor_m"
    # Matches the existing Franka runner: an enabled floor without a calibrated value
    # captures the startup height.  This is explicit in the header before execution.
    return None, True, "captured at startup"


def make_session_config(
    robot_cfg: dict[str, Any], args: argparse.Namespace
) -> FrankaSessionConfig:
    rb = robot_cfg.get("robot", {}) or {}
    return FrankaSessionConfig(
        nuc_ip=str(args.nuc_ip or rb.get("nuc_ip", "")),
        nuc_port=int(
            args.nuc_port if args.nuc_port is not None else rb.get("nuc_port", 4242)
        ),
        use_mock_robot=bool(args.mock_robot or rb.get("use_mock_robot", False)),
        start_impedance=bool(rb.get("start_impedance", True)) and not args.no_impedance,
        connect_cameras=True,
        use_mock_cameras=bool(args.mock_cameras or rb.get("use_mock_cameras", False)),
        external_camera_serial=rb.get("external_camera_serial"),
        wrist_camera_serial=rb.get("wrist_camera_serial"),
        camera_width=int(rb.get("camera_width", 640)),
        camera_height=int(rb.get("camera_height", 480)),
        camera_fps=int(rb.get("camera_fps", 30)),
        camera_read_timeout_ms=int(rb.get("camera_read_timeout_ms", 3000)),
        camera_read_retries=int(rb.get("camera_read_retries", 2)),
        camera_read_retry_delay_s=float(rb.get("camera_read_retry_delay_s", 0.05)),
        camera_restart_on_read_failure=bool(
            rb.get("camera_restart_on_read_failure", True)
        ),
        observation_resolution=int(robot_cfg.get("camera_resolution", 256)),
        verbose=True,
    )


def _task_label(plan: ReplayPlan) -> str:
    recorded_cfg = plan.metadata.get("robot_config")
    if isinstance(recorded_cfg, dict) and str(recorded_cfg.get("task") or "").strip():
        return str(recorded_cfg["task"])
    return f"Replay {plan.rollout_dir.name}"


def _print_plan_summary(plan: ReplayPlan, primitives_cfg: dict[str, Any]) -> None:
    counts = Counter(step.token for step in plan.steps)
    per_record = sum(step.step_override_m is not None for step in plan.steps)
    print("\n[replay] validated replay plan")
    print(f"  directory   {plan.rollout_dir}")
    print(f"  source      {plan.source_path.name} ({plan.source_kind})")
    print(f"  records     {len(plan.steps)}")
    print(f"  primitives  {primitives_cfg.get('_replay_origin')}")
    print(
        f"  fallback    step={float(primitives_cfg['step_m']) * 100:g} cm  "
        f"yaw={float(primitives_cfg['yaw_step_rad']):g} rad"
    )
    if per_record:
        print(f"  exact steps {per_record} MV_* records carry their own step_cm")
    print(
        "  actions     "
        + "  ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    )
    print(
        "  IMPORTANT   movements are relative; use the recording's original arm, "
        "gripper, and scene start state."
    )


def _viewer_status(
    viewer: Optional[LiveView],
    *,
    step: ReplayStep,
    total: int,
    grip: str,
    task: str,
    phase: str,
    telemetry: str,
) -> None:
    if viewer is None:
        return
    viewer.show_single(
        step=step.ordinal,
        arm={"stage": step.stage, "token": step.token, "grip": grip},
        task=task,
        phase=f"REPLAY {step.ordinal + 1}/{total} · {phase}",
        telemetry=telemetry,
    )


def execute_replay(
    plan: ReplayPlan,
    controller: FrankaAtomicController,
    *,
    pause_min_s: float,
    pause_max_s: float,
    rng: random.Random,
    viewer: Optional[LiveView] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    force_step_m: Optional[float] = None,
) -> int:
    """Execute validated steps.  ``sleep_fn`` makes delay behavior unit-testable."""
    pause_min_s, pause_max_s = validate_pause_range(pause_min_s, pause_max_s)
    task = _task_label(plan)
    total = len(plan.steps)

    for position, step in enumerate(plan.steps):
        grip_before = "CLOSED" if controller.gripper_closed else "OPEN"
        step_override = (
            float(force_step_m)
            if force_step_m is not None and step.token in MOVE_ATOMS
            else step.step_override_m
        )
        step_text = (
            f"{step_override * 100:g}cm"
            if step_override is not None
            else f"{controller.step_m * 100:g}cm"
        )
        telemetry = f"source line {step.source_line} · move {step_text}"
        print(
            f"[replay] {position + 1:03d}/{total:03d}  line={step.source_line:<4d} "
            f"stage={step.stage:<10.10s} action={step.token:<11s} "
            f"step={step_text if step.token in MOVE_ATOMS else '-'}"
        )
        _viewer_status(
            viewer,
            step=step,
            total=total,
            grip=grip_before,
            task=task,
            phase="EXECUTING",
            telemetry=telemetry,
        )
        result = controller.step(step.token, step_override_m=step_override)

        if position == total - 1:
            continue
        delay_s = rng.uniform(pause_min_s, pause_max_s)
        grip_after = "CLOSED" if result.gripper_closed else "OPEN"
        print(f"[replay]            waiting {delay_s:.2f}s before next record")
        _viewer_status(
            viewer,
            step=step,
            total=total,
            grip=grip_after,
            task=task,
            phase=f"WAIT {delay_s:.2f}s",
            telemetry=telemetry,
        )
        sleep_fn(delay_s)
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        pause_min_s, pause_max_s = validate_pause_range(
            args.pause_min_s, args.pause_max_s
        )
        plan = load_replay_plan(args.rollout_dir, args.source)
        primitives_cfg = resolve_primitives_config(
            plan,
            args.primitives_config,
            step_m=args.step_m,
            yaw_step_rad=args.yaw_step_rad,
        )
        _print_plan_summary(plan, primitives_cfg)
        if args.dry_run:
            print("[replay] dry-run complete; no hardware connection was opened.")
            return 0

        robot_cfg = load_yaml(args.robot_config)
        hardware = str(robot_cfg.get("hardware", "franka")).strip().lower()
        if hardware != "franka":
            raise ValueError(
                f"replay_rollout.py controls Franka only, but {args.robot_config} "
                f"declares hardware={hardware!r}"
            )
        z_floor_m, capture_floor, floor_origin = resolve_z_floor(robot_cfg, args)
        max_translation_m = (
            float(args.max_translation_m)
            if args.max_translation_m is not None
            else float(
                primitives_cfg.get(
                    "osc_translation_output_max_m", DEFAULT_MAX_POSITION_DELTA_M
                )
            )
        )
        if not math.isfinite(max_translation_m) or max_translation_m <= 0.0:
            raise ValueError("max-translation-m must be finite and > 0")
        if args.step_m is not None and float(args.step_m) > max_translation_m + 1e-12:
            raise ValueError(
                f"--step-m {float(args.step_m):g} exceeds the active "
                f"--max-translation-m safety limit {max_translation_m:g}"
            )
        max_recorded = max(
            (step.step_override_m or 0.0 for step in plan.steps), default=0.0
        )
        if args.step_m is None and max_recorded > max_translation_m + 1e-12:
            raise ValueError(
                f"recorded step_cm reaches {max_recorded * 100:g} cm, above the active "
                f"{max_translation_m * 100:g} cm translation safety limit. Pass "
                f"--max-translation-m {max_recorded:g} only after confirming that "
                "distance is safe for the current setup."
            )

        print(f"  pauses      uniform [{pause_min_s:g}, {pause_max_s:g}] seconds")
        floor_text = (
            "off"
            if z_floor_m is None and not capture_floor
            else ("startup height" if capture_floor else f"{z_floor_m:g} m")
        )
        print(f"  z-floor     {floor_text} ({floor_origin})")
        print(f"  safety max  {max_translation_m * 100:g} cm per translation command\n")

        session = FrankaSession(make_session_config(robot_cfg, args))
        viewer = LiveView(
            enabled=not args.no_show,
            title=f"Show-Harness replay | {plan.rollout_dir.name}",
        )
        try:
            session.connect()
            rb = robot_cfg.get("robot", {}) or {}
            mock_robot = bool(args.mock_robot or rb.get("use_mock_robot", False))
            controller = FrankaAtomicController.from_primitives_config(
                session.robot,
                primitives_cfg,
                step_m=args.step_m,
                max_position_delta_m=max_translation_m,
                max_rotation_delta_rad=float(
                    primitives_cfg.get(
                        "osc_rotation_output_max_rad", DEFAULT_MAX_ROTATION_DELTA_RAD
                    )
                ),
                settle_steps=int(rb.get("settle_steps", 4)),
                settle_dt_s=float(rb.get("settle_dt_s", 0.05)),
                grasp_min_width_m=float(
                    robot_cfg.get("empty_width_m", EMPTY_GRASP_WIDTH_M)
                ),
                grasp_open_width_m=float(robot_cfg.get("open_width_m", 0.07)),
                gripper_settle_s=(
                    0.0 if mock_robot else float(rb.get("gripper_settle_s", 2.5))
                ),
                gripper_min_settle_s=(
                    0.0 if mock_robot else float(rb.get("gripper_min_settle_s", 0.5))
                ),
                z_floor_m=z_floor_m,
                capture_z_floor_on_sync=capture_floor,
                ensure_controller=(
                    session.start_impedance if session.config.start_impedance else None
                ),
                verbose=True,
            )
            controller.sync_from_robot()
            if viewer.enabled:
                viewer.start_stream(session.get_camera_frames, fps=12.0)
            completed = execute_replay(
                plan,
                controller,
                pause_min_s=pause_min_s,
                pause_max_s=pause_max_s,
                rng=random.Random(args.seed),
                viewer=viewer,
                force_step_m=args.step_m,
            )
            print(
                f"\n[replay] complete: executed {completed} records from {plan.source_path.name}"
            )
            return 0
        finally:
            viewer.close()
            session.close()
    except KeyboardInterrupt:
        print("\n[replay] interrupted by operator")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI must surface preflight/hardware errors
        print(f"[replay] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
