"""The config overlay mechanism and the secrets-file precedence rules."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.config import load_secrets_env, load_yaml


class OverlayTests(unittest.TestCase):
    def _write(self, root: Path, name: str, text: str) -> Path:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def test_defaults_merge_under_and_overlays_over_the_body(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write(
                root, "site/rig.yaml",
                "robot:\n  nuc_ip: '10.0.0.9'\nfine_step_m: 0.05\n",
            )
            self._write(root, "exp.yaml", "vlm_backend: lab\nplugins:\n  mcq: true\n")
            main = self._write(
                root,
                "main.yaml",
                "vlm_backend: public\nfine_step_m: 0.02\n"
                "plugins:\n  subgoal: true\n  mcq: false\n"
                "defaults:\n  - site/rig.yaml\n"
                "overlays:\n  - exp.yaml\n",
            )
            cfg = load_yaml(main)
        self.assertEqual(cfg["vlm_backend"], "lab")          # overlay beats body
        self.assertEqual(cfg["fine_step_m"], 0.02)           # body beats defaults
        self.assertEqual(cfg["robot"]["nuc_ip"], "10.0.0.9")  # defaults still supply
        # deep merge: sibling plugin keys survive, overlaid one wins
        self.assertEqual(cfg["plugins"], {"subgoal": True, "mcq": True})
        self.assertNotIn("overlays", cfg)
        self.assertNotIn("defaults", cfg)

    def test_optional_overlay_is_skipped_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            main = self._write(
                Path(d),
                "main.yaml",
                "a: 1\noverlays:\n  - {path: missing.yaml, optional: true}\n",
            )
            cfg = load_yaml(main)
        self.assertEqual(cfg, {"a": 1})

    def test_required_overlay_missing_names_the_example(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            main = self._write(
                Path(d), "main.yaml", "a: 1\noverlays:\n  - site/rig.yaml\n"
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                load_yaml(main)
        self.assertIn("site/rig.yaml.example", str(ctx.exception))

    def test_nested_overlays_and_cycle_detection(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write(root, "b.yaml", "x: 2\noverlays:\n  - c.yaml\n")
            self._write(root, "c.yaml", "y: 3\n")
            main = self._write(root, "a.yaml", "x: 1\noverlays:\n  - b.yaml\n")
            cfg = load_yaml(main)
            self.assertEqual(cfg, {"x": 2, "y": 3})

            self._write(root, "loop1.yaml", "overlays:\n  - loop2.yaml\n")
            self._write(root, "loop2.yaml", "overlays:\n  - loop1.yaml\n")
            with self.assertRaises(ValueError):
                load_yaml(root / "loop1.yaml")


class SecretsPrecedenceTests(unittest.TestCase):
    def test_explicit_path_and_shell_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env_file = Path(d) / "s.env"
            env_file.write_text("A_TEST_KEY=file\nB_TEST_KEY='quoted'\n")
            os.environ["A_TEST_KEY"] = "shell"
            try:
                parsed = load_secrets_env(env_file)
                self.assertEqual(parsed, {"A_TEST_KEY": "file", "B_TEST_KEY": "quoted"})
                self.assertEqual(os.environ["A_TEST_KEY"], "shell")  # shell wins
                self.assertEqual(os.environ["B_TEST_KEY"], "quoted")
            finally:
                os.environ.pop("A_TEST_KEY", None)
                os.environ.pop("B_TEST_KEY", None)


if __name__ == "__main__":
    unittest.main()
