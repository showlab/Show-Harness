"""The execution-boundary token swap: the mixed-corpus deployment convention.

The released single-arm adapters were co-trained on Franka + AgileX demonstrations under
ONE prompt, which only reads consistently because the AgileX episodes were rendered with
MV_FWD/MV_BACK exchanged. Deployed on the AgileX rig such a checkpoint therefore emits
swapped-convention tokens, and the swap has to be undone where a token becomes motion --
and NOWHERE else, because the move history the adapter saw was swapped too.

These tests pin the three properties that make that correct: the swap reaches the
controller, it does not reach anything else, and a config typo fails loudly instead of
silently leaving the arm reversed.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from core.config import load_yaml, resolve_vlm_config
from core.launch import execution_token_swap_pairs, install_execution_token_swap

REPO = Path(__file__).resolve().parents[1]

ALL_UNITS = (
    "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
    "GRASP", "RELEASE", "DONE",
)
SWAP_CFG = {"vlm": {"backend": "test", "execution_token_swap": ["MV_FWD", "MV_BACK"]}}


class RecordingController:
    """Stands in for RealAtomicController: records what would reach the arm."""

    def __init__(self) -> None:
        self.executed: list = []
        self.calls: list = []

    def step(self, token, *args, **kwargs):
        self.executed.append(token)
        self.calls.append((token, args, kwargs))
        return f"moved:{token}"


class ParsingTests(unittest.TestCase):
    def test_absent_means_no_pairs(self) -> None:
        self.assertEqual(execution_token_swap_pairs({}), ())
        self.assertEqual(execution_token_swap_pairs({"vlm": {}}), ())

    def test_pairs_are_read_two_by_two_and_normalized(self) -> None:
        self.assertEqual(
            execution_token_swap_pairs({"vlm": {"execution_token_swap": ["mv_fwd", " MV_BACK "]}}),
            (("MV_FWD", "MV_BACK"),),
        )

    def test_a_typo_is_refused_rather_than_silently_ignored(self) -> None:
        # The failure mode this guards: a misspelled unit would disable the swap, and a
        # reversed-depth rollout reads as a bad policy, not as a config error.
        for bad in (["MV_FWD", "MV_FOWARD"], ["MV_FWD"], ["MV_FWD", "MV_FWD"], "MV_FWD"):
            with self.assertRaises(ValueError):
                execution_token_swap_pairs({"vlm": {"execution_token_swap": bad}})


class WrappingTests(unittest.TestCase):
    def test_only_the_named_pair_is_exchanged(self) -> None:
        c = RecordingController()
        install_execution_token_swap(c, SWAP_CFG)
        for token in ALL_UNITS:
            c.step(token)
        self.assertEqual(
            c.executed,
            ["MV_BACK", "MV_FWD", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
             "GRASP", "RELEASE", "DONE"],
        )

    def test_disabled_is_token_identical(self) -> None:
        c = RecordingController()
        install_execution_token_swap(c, {"vlm": {"backend": "finetuned_local"}})
        for token in ALL_UNITS:
            c.step(token)
        self.assertEqual(c.executed, list(ALL_UNITS))

    def test_wraps_the_instance_not_the_class(self) -> None:
        # A dual rig builds one controller per arm; swapping the class would hit both.
        swapped = RecordingController()
        install_execution_token_swap(swapped, SWAP_CFG)
        untouched = RecordingController()
        swapped.step("MV_FWD")
        untouched.step("MV_FWD")
        self.assertEqual(swapped.executed, ["MV_BACK"])
        self.assertEqual(untouched.executed, ["MV_FWD"])

    def test_step_arguments_pass_through_untouched(self) -> None:
        c = RecordingController()
        install_execution_token_swap(c, SWAP_CFG)
        c.step("MV_FWD", True, motion_frame="wrist", step_override_m=0.01)
        self.assertEqual(
            c.calls,
            [("MV_BACK", (True,), {"motion_frame": "wrist", "step_override_m": 0.01})],
        )


class ShippedConfigTests(unittest.TestCase):
    """The released checkpoints must carry the convention on AgileX and not on Franka."""

    RELEASED = ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b", "internvl3_5_2b", "gemma4_e4b")

    def _configs(self) -> Path:
        """A fresh-install view of configs/: the shipped examples become the site layer."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pub = tmp / "configs"
        shutil.copytree(REPO / "configs", pub, ignore=shutil.ignore_patterns("secrets*.env"))
        for example in pub.glob("site/*.example"):
            target = example.with_suffix("")
            if not target.exists():
                shutil.copy(example, target)
        return pub

    def test_agilex_ft_config_swaps_for_every_released_adapter(self) -> None:
        cfg = load_yaml(self._configs() / "robot_piper_ft.yaml")
        for backend in self.RELEASED:
            resolved = {"vlm": resolve_vlm_config(cfg, backend=backend)}
            self.assertEqual(
                execution_token_swap_pairs(resolved),
                (("MV_FWD", "MV_BACK"),),
                f"{backend} on the AgileX rig must undo the training-time exchange",
            )

    def test_franka_ft_config_never_swaps(self) -> None:
        cfg = load_yaml(self._configs() / "robot_franka_ft.yaml")
        for backend in self.RELEASED:
            resolved = {"vlm": resolve_vlm_config(cfg, backend=backend)}
            self.assertEqual(
                execution_token_swap_pairs(resolved),
                (),
                f"{backend} reads straight on Franka; a swap there would reverse it",
            )

    def test_bring_your_own_adapter_defaults_to_no_swap(self) -> None:
        # finetuned_local is for an adapter trained from this repo's converter, which is
        # already in the rig's own convention.
        cfg = load_yaml(self._configs() / "robot_piper_ft.yaml")
        resolved = {"vlm": resolve_vlm_config(cfg, backend="finetuned_local")}
        self.assertEqual(execution_token_swap_pairs(resolved), ())


if __name__ == "__main__":
    unittest.main()
