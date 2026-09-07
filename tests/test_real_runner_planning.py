"""The runner's planning path, offline: the planner receives both live frames
with labeled roles, the plan and diagnostics are logged, and the rollout starts
from an open gripper."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.runners.real import RealEpisodeRunner
from core.v0_types import Subgoal, V0Config


class _Session:
    def __init__(self) -> None:
        self.front = np.zeros((12, 16, 3), dtype=np.uint8)
        self.wrist = np.ones((8, 10, 3), dtype=np.uint8)

    def get_observation(self):
        return {
            "agentview": self.front,
            "wrist": self.wrist,
            "agentview_hd": self.front,
            "wrist_hd": self.wrist,
        }


class _Controller:
    z_floor_m = 0.0

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def step(self, token):
        self.tokens.append(token)


class _Logger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.plans: list[dict] = []
        self.diagnostics: list[dict] = []
        self.prompts: list[tuple[int, str]] = []
        self.summary: dict = {}

    def write_plan(self, value):
        self.plans.append(value)

    def write_planner_diagnostics(self, value):
        self.diagnostics.append(value)

    def save_planner_prompt(self, attempt, prompt):
        self.prompts.append((attempt, prompt))

    def close(self, **kwargs):
        return self.run_dir / "rollout_failure.mp4"

    def write_summary(self, value):
        self.summary = value


class _Planner:
    def __init__(self, stages: list[Subgoal]) -> None:
        self.stages = stages
        self.calls: list[dict] = []

    def plan(self, task, agentview, **kwargs):
        self.calls.append({"task": task, "agentview": agentview, **kwargs})
        return list(self.stages), json.dumps(
            {"subgoals": [stage.to_prompt_dict() for stage in self.stages]}
        )

    def diagnostics(self):
        return {"route": "guided_json", "image_roles": self.calls[-1]["image_roles"]}

    def last_prompt(self):
        return f"prompt attempt {len(self.calls)}"


def _runner(temp_dir: str, planner) -> RealEpisodeRunner:
    return RealEpisodeRunner(
        session=_Session(),
        controller=_Controller(),
        planner=planner,
        controls=None,
        logger=_Logger(Path(temp_dir)),
        config=V0Config(max_subgoal_steps=4, max_replans=0, video_fps=1.0),
        task="put the mug in the bowl",
        gripper_color="black",
        max_steps=0,
        loop_period_s=0.0,
        use_wrist_image=True,
        debug=False,
    )


class RealRunnerPlanningTests(unittest.TestCase):
    def test_planner_gets_both_live_views_and_the_plan_is_logged(self) -> None:
        stage = Subgoal(
            id="place",
            target="bowl",
            affordance="visible bowl interior",
            motion="PLACE",
            description="Place the object in the bowl.",
            completion="The object is visibly inside the bowl.",
        )
        planner = _Planner([stage])

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = _runner(temp_dir, planner)
            result = runner.run()

        self.assertEqual(result.end_reason, "max_steps_exceeded")
        self.assertEqual(len(planner.calls), 1)
        call = planner.calls[0]
        self.assertEqual(len(call["image_roles"]), 2)
        self.assertIn("AgentView", call["image_roles"][0])
        self.assertIn("Wrist", call["image_roles"][1])
        self.assertIs(call["wrist"], runner.session.wrist)
        # The rollout always begins from a commanded-open gripper.
        self.assertEqual(runner.controller.tokens, ["RELEASE"])
        self.assertEqual(runner.logger.plans[-1]["task"], "put the mug in the bowl")
        self.assertEqual(len(runner.logger.plans[-1]["subgoals"]), 1)
        attempts = runner.logger.diagnostics[-1]["attempts"]
        self.assertEqual(attempts[0]["subgoal_count"], 1)
        self.assertEqual(runner.logger.prompts, [(1, "prompt attempt 1")])


if __name__ == "__main__":
    unittest.main()
