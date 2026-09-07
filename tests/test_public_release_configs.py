"""The public release resolves cleanly and leaks nothing internal.

Simulates the exported tree: configs are copied with the layers listed in
release/public-exclude.txt removed and the site examples copied into place,
then every real-robot config must resolve to public defaults with no internal
endpoint, alias, serial, or personal path in the result.
"""
from __future__ import annotations

import fnmatch
import shutil
import tempfile
import unittest
from pathlib import Path

from core.config import load_yaml, resolve_vlm_config

REPO = Path(__file__).resolve().parents[1]

# Strings that must never appear in a publicly resolved config. The generic
# markers live inline; the concrete internal identifiers (serials, hosts,
# personal paths) come from release/leakcheck.internal-patterns, which exists
# only in the internal repo -- the shipped test must not carry them itself.
_GENERIC_MARKERS = (
    "trapi",
    "aiberm",
    "showrobot",
    "showlab",
    "xiaomimi",
    "letters_blind",
    "/workspace1",
)


def _internal_markers() -> tuple[str, ...]:
    markers = list(_GENERIC_MARKERS)
    pattern_file = REPO / "release/leakcheck.internal-patterns"
    if pattern_file.exists():
        for line in pattern_file.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            for alt in line.split("|"):
                alt = alt.replace("\\.", ".").strip()
                # Only substring-safe alternatives (no residual regex syntax).
                if alt and not any(c in alt for c in "[](){}$^*+?\\"):
                    markers.append(alt)
    return tuple(dict.fromkeys(markers))


INTERNAL_MARKERS = _internal_markers()

REAL_CONFIGS = (
    "robot_franka.yaml",
    "robot_franka_ft.yaml",
    "robot_piper.yaml",
    "robot_piper_ft.yaml",
)
SIM_CONFIGS = (
    "robot_maniskill.yaml",
    "robot_robolab.yaml",
)


# A public default must be reachable by anyone who cloned the repo: the
# rename-me example, one of the released checkpoints, or -- for a real robot
# only -- a hosted baseline. Sim configs must not default to a paid hosted
# model. Released adapters are named <model>_showharness_<split>: `_ft` for the real
# corpus, `_sim` for the simulated one (RoboLab and ManiSkill share a single policy).
RELEASED_MARKER = "_showharness_"


def _is_public_default(vlm: dict, *, allow_hosted: bool) -> bool:
    """The marker is checked on the MODEL, not the backend key: profile keys are
    short names (`qwen3_5_2b`) and only the adapter they resolve to carries it."""
    if vlm["backend"] == "finetuned_local" or RELEASED_MARKER in str(vlm.get("model", "")):
        return True
    return allow_hosted and vlm["backend"] == "gemini"


MANIFEST = REPO / "release/public-exclude.txt"


def _excluded_config_paths() -> list[str]:
    """configs/ entries from release/public-exclude.txt, relative to configs/.

    The manifest exists only in the internal repo; in the exported public tree
    the internal layers are already gone and there is nothing to exclude.
    """
    if not MANIFEST.exists():
        return []
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("configs/"):
            out.append(line[len("configs/"):])
    return out


class PublicReleaseConfigTests(unittest.TestCase):
    def _public_configs(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pub = tmp / "configs"
        shutil.copytree(
            REPO / "configs",
            pub,
            ignore=shutil.ignore_patterns("secrets.env", "secrets.local.env", "xiaomimi.env"),
        )
        for rel in _excluded_config_paths():
            target = pub / rel.rstrip("/")
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        # A fresh install copies the shipped examples.
        for example in pub.glob("site/*.example"):
            target = example.with_suffix("")
            if not target.exists():
                shutil.copy(example, target)
        return pub

    def test_manifest_excludes_the_internal_layers(self) -> None:
        if not MANIFEST.exists():
            self.skipTest("public tree: the exclusion manifest is internal-only")
        rels = _excluded_config_paths()
        for required in (
            "site/franka.yaml",
            "site/piper_arms.yaml",
            "backends/internal.yaml",
            "backends/lab_lora_servers.yaml",
            "experiments/",
        ):
            self.assertTrue(
                any(fnmatch.fnmatch(required.rstrip("/"), r.rstrip("/")) or r.rstrip("/") == required.rstrip("/") for r in rels),
                f"release/public-exclude.txt must exclude configs/{required}",
            )

    def test_public_configs_resolve_clean(self) -> None:
        pub = self._public_configs()
        for name in REAL_CONFIGS:
            cfg = load_yaml(pub / name)
            vlm = resolve_vlm_config(cfg)
            blob = repr(cfg).lower()
            for marker in INTERNAL_MARKERS:
                self.assertNotIn(marker.lower(), blob, f"{name} leaks {marker!r}")
            self.assertTrue(
                _is_public_default(vlm, allow_hosted=True),
                f"{name} default backend {vlm['backend']!r} "
                f"({vlm.get('model')!r}) is not a public default",
            )
        for name in SIM_CONFIGS:
            cfg = load_yaml(pub / name)
            vlm = resolve_vlm_config(cfg)
            blob = repr(cfg).lower()
            for marker in INTERNAL_MARKERS + ("mix_22", "mvtoken_0622", ":8109", ":8202", ":8101"):
                self.assertNotIn(str(marker).lower(), blob, f"{name} leaks {marker!r}")
            self.assertTrue(
                _is_public_default(vlm, allow_hosted=False),
                f"{name} default backend {vlm['backend']!r} "
                f"({vlm.get('model')!r}) is not a public offline default",
            )

    def test_missing_site_layer_error_is_actionable(self) -> None:
        pub = self._public_configs()
        (pub / "site/franka.yaml").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            load_yaml(pub / "robot_franka.yaml")
        self.assertIn("site/franka.yaml.example", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
