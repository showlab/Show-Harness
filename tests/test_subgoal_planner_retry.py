from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field

import numpy as np

from plugins.subgoal.agent import SubgoalPlannerAgent
from plugins.subgoal.plugin import SubgoalPlanner


@dataclass
class _Response:
    raw_text: str
    payload: dict = field(default_factory=dict)
    token: str = ""


class _RetryClient:
    def __init__(self) -> None:
        self.json_prompts: list[str] = []
        self.text_prompts: list[str] = []

    def complete_json(self, prompt, image, **kwargs):
        self.json_prompts.append(prompt)
        raise RuntimeError("VLM returned empty content")

    def complete_text(self, prompt, image, **kwargs):
        self.text_prompts.append(prompt)
        plan = {
            "subgoals": [
                {
                    "id": "inspect",
                    "target": "object",
                    "affordance": "visible object surface",
                    "motion": "MOVE",
                    "description": "Inspect and move the object.",
                    "completion": "The object is visibly at its destination.",
                }
            ]
        }
        return _Response(json.dumps(plan), payload={"latency_s": 1.25})


class SubgoalPlannerRetryTests(unittest.TestCase):
    def test_pregrasp_merge_folds_a_pure_reach_into_the_grasp_stage(self) -> None:
        class _StaticAgent:
            def plan(self, task, image, **kwargs):
                stages = {
                    "subgoals": [
                        {
                            "id": "travel",
                            "target": "cup",
                            "affordance": "cup rim",
                            "motion": "TRAVEL",
                            "description": "Move over the cup.",
                            "completion": "The gripper is above the cup.",
                        },
                        {
                            "id": "grasp",
                            "target": "cup",
                            "affordance": "cup rim",
                            "motion": "GRASP",
                            "description": "Close on the cup.",
                            "completion": "The cup is held.",
                        },
                    ]
                }
                return _Response(json.dumps(stages), payload={"json": stages})

        planner = SubgoalPlanner(_StaticAgent())
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        merged, _ = planner.plan("task", image)

        self.assertEqual([stage.motion for stage in merged], ["GRASP"])
        self.assertIn("Move over the cup", merged[0].description)

    def test_text_retry_preserves_the_full_rendered_prompt(self) -> None:
        client = _RetryClient()
        agent = SubgoalPlannerAgent(
            client=client,
            common_context="GENERAL_CONTEXT_SENTINEL",
            prompt_template=(
                "TASK={task}\nSCENE_CONTRACT_SENTINEL\nREFERENCE={video_ref}"
            ),
            video_ref_block="DEMO_SENTINEL",
        )

        response = agent.plan(
            "move every piece",
            np.zeros((2, 2, 3), dtype=np.uint8),
            image_roles=["LIVE FRONT", "LIVE WRIST"],
        )

        self.assertEqual(len(client.json_prompts), 2)
        self.assertEqual(len(client.text_prompts), 1)
        retry_prompt = client.text_prompts[0]
        for sentinel in (
            "GENERAL_CONTEXT_SENTINEL",
            "SCENE_CONTRACT_SENTINEL",
            "DEMO_SENTINEL",
            "LIVE FRONT",
            "LIVE WRIST",
        ):
            self.assertIn(sentinel, retry_prompt)
        self.assertEqual(response.payload["planner_retry"], "text_no_think")
        diagnostics = agent.diagnostics()
        self.assertEqual(diagnostics["route"], "text_no_think")
        self.assertEqual(len(diagnostics["errors"]), 2)
        self.assertGreaterEqual(diagnostics["total_latency_s"], 0.0)

    def test_no_optional_context_keeps_the_generic_render_unchanged(self) -> None:
        class _GuidedClient:
            def __init__(self) -> None:
                self.prompt = ""

            def complete_json(self, prompt, image, **kwargs):
                self.prompt = prompt
                return _Response(
                    "ok",
                    payload={
                        "json": {
                            "subgoals": [
                                {
                                    "id": "task",
                                    "target": "task",
                                    "affordance": "visible region",
                                    "motion": "TASK",
                                    "description": "Complete it.",
                                    "completion": "It is visibly complete.",
                                }
                            ]
                        }
                    },
                )

        client = _GuidedClient()
        agent = SubgoalPlannerAgent(
            client=client,
            common_context="GENERAL",
            prompt_template="DO {task}; DEMO={video_ref}",
        )

        agent.plan("anything", np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(client.prompt, "GENERAL\n\nDO anything; DEMO=")


if __name__ == "__main__":
    unittest.main()
