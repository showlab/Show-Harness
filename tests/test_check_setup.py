"""The preflight checker: passes on a complete config, fails with guidance
when site identity is missing. Offline modes only (--no-vlm --no-hardware)."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.check_setup import main

GOOD = """\
hardware: franka
robot:
  nuc_ip: "10.0.0.9"
  external_camera_serial: "000000000001"
  wrist_camera_serial: "000000000002"
enable_z_floor: true
z_floor_name: default
z_floors: {default: 0.14}
vlm_backend: local
vlm: {base_url: http://localhost:8000/v1}
vlm_backends:
  local: {provider: vllm, base_url: http://localhost:8000/v1, model: m}
"""


class CheckSetupTests(unittest.TestCase):
    def _run(self, cfg_text: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "robot.yaml"
            p.write_text(cfg_text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--robot-config", str(p), "--no-vlm", "--no-hardware"])
        return code, out.getvalue()

    def test_complete_config_passes(self) -> None:
        code, out = self._run(GOOD)
        self.assertEqual(code, 0, out)
        self.assertIn("z floor", out)
        self.assertIn("vlm backend", out)

    def test_missing_identity_fails_with_site_hint(self) -> None:
        code, out = self._run(GOOD.replace('  nuc_ip: "10.0.0.9"\n', ""))
        self.assertEqual(code, 1, out)
        self.assertIn("site/franka.yaml.example", out)

    def test_placeholder_z_floor_fails(self) -> None:
        code, out = self._run(GOOD.replace("z_floors: {default: 0.14}",
                                           "z_floors: {default: 0.0}"))
        self.assertEqual(code, 1, out)
        self.assertIn("placeholder", out)


if __name__ == "__main__":
    unittest.main()
