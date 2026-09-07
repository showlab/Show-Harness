#!/usr/bin/env python3
"""Unit and local HTTP tests for the GPT web operator (no API key or robot)."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import requests

from gumi.gpt_operator.operator import (
    DecisionError,
    GPTWebOperator,
    OperatorConfig,
    compact_model_state,
    normalize_decision,
)
from gumi.gpt_operator.server import serve as serve_dashboard
from gumi.gpt_web_operator import parse_args
from gumi.collect_rollouts_web_dual import _require_ros_master
from gumi.web_teleop.server import TeleopServer, build_handler
from core.vlm.vlm_client import VLMParseError, VLMResponse, _message_content


def decision(**overrides):
    value = {
        "phase": "align",
        "evidence": "The cube is left of the gripper center.",
        "next_goal": "center the gripper over the cube",
        "left": {"action": "MV_LEFT", "repeat": 1},
        "right": {"action": "STILL", "repeat": 1},
        "confidence": 0.9,
        "finish": False,
        "pause": False,
    }
    value.update(overrides)
    return value


class LauncherArgsTests(unittest.TestCase):
    def test_whitespace_only_shell_arguments_are_ignored(self):
        args = parse_args(
            [
                " ",
                "--target-url",
                "http://127.0.0.1:8620",
                "\t",
                "--max-repeat",
                "1",
                "  ",
                "--no-auto-save",
            ]
        )

        self.assertEqual(args.target_url, "http://127.0.0.1:8620")
        self.assertEqual(args.max_repeat, 1)
        self.assertTrue(args.no_auto_save)

    def test_real_collector_fails_fast_when_ros_master_is_absent(self):
        with patch(
            "gumi.collect_rollouts_web_dual.socket.create_connection",
            side_effect=ConnectionRefusedError("connection refused"),
        ):
            with self.assertRaisesRegex(SystemExit, "scripts/piper/run_cameras.sh"):
                _require_ros_master("http://localhost:11311")


class FakeVLM:
    model = "fake-gpt"
    provider = "openai"

    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return VLMResponse(
            token="",
            raw_text=json.dumps(self.output),
            payload={"json": copy.deepcopy(self.output), "latency_s": 0.123},
        )


class RetryVLM(FakeVLM):
    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise VLMParseError("VLM returned non-JSON text: ''", "")
        return VLMResponse(
            token="",
            raw_text=json.dumps(self.output),
            payload={"json": copy.deepcopy(self.output), "latency_s": 0.456},
        )


class FakeTarget:
    base_url = "http://fake-target"

    def __init__(self, dual=False):
        self.posts = []
        self.current = {
            "dual_arm": dual,
            "recording": False,
            "steps": 0,
            "total_pairs": 0,
            "task_text": "put the cube on the plate",
            "gripper_closed": ({"left": False, "right": False} if dual else False),
            "holding": ({"left": False, "right": False} if dual else False),
            "task_done": False,
            "can_stop": False,
            "gap_warn": False,
        }

    def state(self):
        return copy.deepcopy(self.current)

    def snapshots(self, dual_arm):
        names = (
            ("agentview", "wrist_left", "wrist_right")
            if dual_arm
            else ("agentview", "wrist")
        )
        images = {name: np.zeros((64, 64, 3), dtype=np.uint8) for name in names}
        frames = {}
        for name, image in images.items():
            ok, buf = cv2.imencode(".jpg", image)
            assert ok
            frames[name] = buf.tobytes()
        return frames, images

    def snapshot_bytes(self, name):
        return b"live-" + name.encode("ascii")

    def post(self, path, body=None):
        self.posts.append((path, copy.deepcopy(body)))
        if path == "/api/start":
            self.current["recording"] = True
            return {"ok": True, "message": "recording", "state": self.state()}
        if path == "/api/step":
            if self.current["dual_arm"]:
                self.current["total_pairs"] += 1
            self.current["steps"] += 1
            return {
                "ok": True,
                "message": "executed",
                "executed": 1,
                "state": self.state(),
            }
        return {"ok": True, "message": path, "state": self.state()}


class DecisionTests(unittest.TestCase):
    def test_model_state_preserves_sim_completion_authority(self):
        compact = compact_model_state(
            {"sim": True, "task_done": True, "sim_scene": {"secret": "ground truth"}}
        )
        self.assertEqual(compact, {"sim": True, "task_done": True})

    def test_single_arm_requires_right_still(self):
        raw = decision(right={"action": "MV_RIGHT", "repeat": 1})
        with self.assertRaisesRegex(DecisionError, "right.action=STILL"):
            normalize_decision(
                raw,
                dual_arm=False,
                state={},
                confidence_threshold=0.5,
                max_repeat=3,
            )

    def test_low_confidence_becomes_pause(self):
        normalized = normalize_decision(
            decision(confidence=0.2),
            dual_arm=False,
            state={},
            confidence_threshold=0.6,
            max_repeat=3,
        )
        self.assertTrue(normalized["pause"])
        self.assertEqual(normalized["left"]["action"], "STILL")

    def test_unsafe_dual_action_is_rejected_when_arms_close(self):
        raw = decision(
            right={"action": "MV_RIGHT", "repeat": 1},
        )
        with self.assertRaisesRegex(DecisionError, "gap_warn"):
            normalize_decision(
                raw,
                dual_arm=True,
                state={"gap_warn": True},
                confidence_threshold=0.5,
                max_repeat=3,
            )

    def test_descend_and_grasp_repeats_are_forced_to_one(self):
        for action in ("MV_DOWN", "GRASP", "ROTATE_CW"):
            normalized = normalize_decision(
                decision(left={"action": action, "repeat": 3}),
                dual_arm=False,
                state={},
                confidence_threshold=0.5,
                max_repeat=3,
            )
            self.assertEqual(normalized["left"]["repeat"], 1)

    def test_image_detail_is_forwarded_to_every_image(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        content = _message_content("JSON please", image, [image, image], image_detail="high")
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 3)
        self.assertTrue(all(part["image_url"]["detail"] == "high" for part in image_parts))


class OperatorCycleTests(unittest.TestCase):
    def test_single_cycle_starts_recording_then_executes(self):
        target = FakeTarget(dual=False)
        vlm = FakeVLM(decision(left={"action": "MV_FWD", "repeat": 2}))
        with tempfile.TemporaryDirectory() as temp:
            config = OperatorConfig(
                target_url=target.base_url,
                trace_root=Path(temp),
                auto_record=True,
                dry_run=False,
            )
            operator = GPTWebOperator(target, vlm, "test prompt", config)
            event = operator.run_cycle()
        self.assertEqual(event["outcome"], "executed")
        self.assertEqual(target.posts[0][0], "/api/start")
        self.assertEqual(target.posts[1], ("/api/step", {"tokens": ["MV_FWD", "MV_FWD"]}))
        self.assertEqual(vlm.calls[0]["image_detail"], "high")

    def test_dry_run_never_starts_recording_or_acts(self):
        target = FakeTarget(dual=False)
        with tempfile.TemporaryDirectory() as temp:
            operator = GPTWebOperator(
                target,
                FakeVLM(decision()),
                "test prompt",
                OperatorConfig(
                    target_url=target.base_url,
                    trace_root=Path(temp),
                    auto_record=True,
                    dry_run=True,
                ),
            )
            event = operator.run_cycle()
        self.assertEqual(event["outcome"], "dry_run")
        self.assertEqual(target.posts, [])

    def test_incomplete_json_retries_once_with_larger_budget(self):
        target = FakeTarget(dual=False)
        vlm = RetryVLM(decision())
        with tempfile.TemporaryDirectory() as temp:
            operator = GPTWebOperator(
                target,
                vlm,
                "test prompt",
                OperatorConfig(
                    target_url=target.base_url,
                    trace_root=Path(temp),
                    auto_record=False,
                    dry_run=True,
                    max_output_tokens=512,
                ),
            )
            event = operator.run_cycle()
        self.assertEqual(event["decision_attempts"], 2)
        self.assertIn("non-JSON", event["retry_reason"])
        self.assertEqual([call["max_tokens"] for call in vlm.calls], [512, 1024])


class SnapshotEndpointTests(unittest.TestCase):
    class Backend:
        def __init__(self):
            self.frame_cond = threading.Condition()
            self.frames = {"agentview": b"jpeg-bytes", "wrist_right": b"right-jpeg"}

        def state(self):
            return {"ok": True}

    def test_snapshot_uses_backend_frame_names_for_single_or_dual(self):
        backend = self.Backend()
        server = TeleopServer(("127.0.0.1", 0), build_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            response = requests.get(
                f"http://127.0.0.1:{port}/snapshot/wrist_right", timeout=2
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"right-jpeg")
            missing = requests.get(
                f"http://127.0.0.1:{port}/snapshot/missing", timeout=2
            )
            self.assertEqual(missing.status_code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class DashboardEndpointTests(unittest.TestCase):
    def test_live_view_is_proxied_through_dashboard_host(self):
        target = FakeTarget(dual=True)
        with tempfile.TemporaryDirectory() as temp:
            operator = GPTWebOperator(
                target,
                FakeVLM(decision()),
                "test prompt",
                OperatorConfig(target_url=target.base_url, trace_root=Path(temp)),
            )
            server = serve_dashboard(operator, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                response = requests.get(
                    f"http://127.0.0.1:{port}/api/live/wrist_right", timeout=2
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"live-wrist_right")
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
