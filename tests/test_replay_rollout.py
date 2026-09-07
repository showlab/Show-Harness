from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from scripts.trajectory.replay_rollout import (
    ReplayPlan,
    ReplayStep,
    execute_replay,
    load_replay_plan,
    resolve_primitives_config,
    validate_pause_range,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class ReplayPlanLoadingTests(unittest.TestCase):
    def test_auto_prefers_steps_and_preserves_per_record_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_jsonl(
                run / "steps.jsonl",
                [
                    {"i": 4, "act": "mv_left", "stage": "MOVE", "step_cm": 2.2},
                    {"i": 5, "act": "DONE", "stage": "MOVE"},
                    {"i": 6, "act": "GRASP", "stage": "GRASP"},
                ],
            )
            _write_jsonl(run / "actions.jsonl", [{"step": 0, "token": "MV_RIGHT"}])
            (run / "metadata.json").write_text(
                json.dumps({"primitives_config": {"step_m": 0.03}}), encoding="utf-8"
            )

            plan = load_replay_plan(run)

            self.assertEqual(plan.source_kind, "steps")
            self.assertEqual(
                [step.token for step in plan.steps], ["MV_LEFT", "DONE", "GRASP"]
            )
            self.assertAlmostEqual(plan.steps[0].step_override_m or 0.0, 0.022)
            self.assertIsNone(plan.steps[1].step_override_m)
            self.assertEqual(plan.steps[0].recorded_index, 4)
            self.assertEqual(plan.metadata["primitives_config"]["step_m"], 0.03)

    def test_actions_format_uses_token_and_has_no_per_record_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_jsonl(
                run / "actions.jsonl",
                [
                    {"step": 0, "token": "MV_FWD"},
                    {"step": 1, "token": "RELEASE"},
                ],
            )
            (run / "metadata.json").write_text(
                json.dumps({"step_m": 0.017, "yaw_step_rad": 0.25}), encoding="utf-8"
            )

            plan = load_replay_plan(run)
            primitives = resolve_primitives_config(plan)

            self.assertEqual(plan.source_kind, "actions")
            self.assertEqual([step.token for step in plan.steps], ["MV_FWD", "RELEASE"])
            self.assertTrue(all(step.step_override_m is None for step in plan.steps))
            self.assertAlmostEqual(primitives["step_m"], 0.017)
            self.assertAlmostEqual(primitives["yaw_step_rad"], 0.25)

    def test_invalid_token_fails_before_hardware_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_jsonl(run / "steps.jsonl", [{"i": 0, "act": "TELEPORT"}])

            with self.assertRaisesRegex(ValueError, "unsupported action"):
                load_replay_plan(run)

    def test_non_increasing_indices_are_rejected_without_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_jsonl(
                run / "actions.jsonl",
                [
                    {"step": 1, "token": "MV_FWD"},
                    {"step": 1, "token": "MV_LEFT"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_replay_plan(run)

    def test_partial_indices_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_jsonl(
                run / "steps.jsonl",
                [{"i": 0, "act": "MV_FWD"}, {"act": "MV_LEFT"}],
            )

            with self.assertRaisesRegex(ValueError, "every record or none"):
                load_replay_plan(run)


class _FakeResult:
    def __init__(self, closed: bool) -> None:
        self.gripper_closed = closed


class _FakeController:
    def __init__(self) -> None:
        self.step_m = 0.02
        self.gripper_closed = False
        self.calls: list[tuple[str, float | None]] = []

    def step(self, token: str, step_override_m: float | None = None) -> _FakeResult:
        self.calls.append((token, step_override_m))
        if token == "GRASP":
            self.gripper_closed = True
        elif token == "RELEASE":
            self.gripper_closed = False
        return _FakeResult(self.gripper_closed)


class ReplayExecutionTests(unittest.TestCase):
    def _plan(self) -> ReplayPlan:
        return ReplayPlan(
            rollout_dir=Path("/tmp/example"),
            source_path=Path("/tmp/example/steps.jsonl"),
            source_kind="steps",
            steps=(
                ReplayStep(0, 1, 0, "MV_FWD", "MOVE", 0.03),
                ReplayStep(1, 2, 1, "DONE", "MOVE", None),
                ReplayStep(2, 3, 2, "GRASP", "GRASP", None),
            ),
            metadata={},
        )

    def test_executes_done_as_no_motion_record_and_waits_only_between_records(
        self,
    ) -> None:
        controller = _FakeController()
        delays: list[float] = []

        completed = execute_replay(
            self._plan(),
            controller,  # type: ignore[arg-type]
            pause_min_s=3.0,
            pause_max_s=5.0,
            rng=random.Random(7),
            sleep_fn=delays.append,
        )

        self.assertEqual(completed, 3)
        self.assertEqual(
            controller.calls,
            [("MV_FWD", 0.03), ("DONE", None), ("GRASP", None)],
        )
        self.assertEqual(len(delays), 2)
        self.assertTrue(all(3.0 <= delay <= 5.0 for delay in delays))

    def test_global_step_override_replaces_recorded_distance(self) -> None:
        controller = _FakeController()

        execute_replay(
            self._plan(),
            controller,  # type: ignore[arg-type]
            pause_min_s=0.0,
            pause_max_s=0.0,
            rng=random.Random(0),
            sleep_fn=lambda _: None,
            force_step_m=0.01,
        )

        self.assertEqual(controller.calls[0], ("MV_FWD", 0.01))

    def test_pause_range_validation(self) -> None:
        self.assertEqual(validate_pause_range(3, 5), (3.0, 5.0))
        with self.assertRaisesRegex(ValueError, "must be <="):
            validate_pause_range(5, 3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_pause_range(-1, 3)


if __name__ == "__main__":
    unittest.main()
