"""DAGGER: real-time human keyboard override during an autonomous rollout.

DAgger-style intervention: while the model drives the arm(s), the operator can
take over at any moment from the live-view window, using the SAME key bindings as
data collection (:func:`core.teleop.dual.build_dual_keymaps` is the single source
of truth, so the muscle memory transfers 1:1). Two modes share one tool:

  * **dual** (default; Piper Mode B): per-arm intents under the dual key layout.
  * **single** (``single=True``; the Franka / ``scripts/run_real.py`` path): one intent
    slot under the side ``"arm"``, driven by the dual layout's LEFT-hand key
    cluster (:func:`core.teleop.dual.build_single_keymaps`) plus SPACE for the
    gripper. ROTATE keys are offered only when the rig's rotation plugin is on
    (``include_rotate_keys``), so a human can never inject a token the
    controller would refuse.

Correctness contract (the reason this is a tool and not a bigger teleop):

  * Key events arrive on the live view's STREAM thread, which pumps pygame
    continuously (~12 Hz) -- including while the VLM API call is in flight.
  * Every captured input bumps :attr:`generation`. The runner snapshots the
    generation when it starts a decision and DROPS the decision if it changed by
    the time the result lands: that decision was made on a pre-intervention
    observation, so executing it would fight the human (the stated conflict).
  * Intents are LATEST-WINS slots per arm (one pending move + one pending gripper
    toggle, mirroring the teleop collector): a burst of taps during a blocked step
    never replays as a backlog. STILL cancels that arm's pending move.
  * A human step executes through the runner's normal per-arm execution path, so
    Z floors, reach clamps, and grasp-width verification all still apply.

The tool only captures and hands over intents; WHEN they execute (and how they
preempt the in-flight VLM call) is the runner's logic. Disabled -> inert: no key
handler is installed and every query returns its empty value.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from core.teleop.dual import build_dual_keymaps, build_single_keymaps

_DUAL_SIDES = ("left", "right")
# The single-arm mode's one intent slot (the single-arm runner's affordance slot
# name, so records and prompts stay consistent across plugins).
SINGLE_SIDE = "arm"

# Intent kinds handed to the runner. "gripper" is resolved into GRASP/RELEASE at
# execution time from the controller's actual state (teleop's flush-time contract).
KIND_MOVE = "move"
KIND_GRIPPER = "gripper"
KIND_STILL = "still"


class DaggerPlugin:
    """Capture teleop-keymap interventions and expose them as per-arm intents."""

    def __init__(
        self,
        enabled: bool = False,
        include_rotate_keys: bool = True,
        single: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.include_rotate_keys = bool(include_rotate_keys)
        self.single = bool(single)
        self.sides = (SINGLE_SIDE,) if self.single else _DUAL_SIDES
        self._lock = threading.Lock()
        # Monotonic input counter: bumped on EVERY captured key, consumed or not.
        # The runner compares snapshots of this to detect "human input arrived
        # after the observation my in-flight decision is based on".
        self.generation = 0
        self._pending_move: dict[str, Optional[str]] = {s: None for s in self.sides}
        self._pending_gripper: dict[str, bool] = {s: False for s in self.sides}
        # key -> (side, token|None, kind); built lazily on the first event, when
        # pygame is guaranteed initialized by the live view that forwards keys.
        self._keymap: Optional[dict[int, tuple[str, Optional[str], str]]] = None

    # -- wiring -----------------------------------------------------------------
    def install(self, viewer: Any) -> None:
        """Attach to a :class:`core.ui.live_view.LiveView` (its stream thread pumps keys)."""
        if self.enabled and viewer is not None:
            viewer.key_handler = self.on_key

    def _ensure_keymap(self) -> dict[int, tuple[str, Optional[str], str]]:
        if self._keymap is None:
            import pygame  # noqa: PLC0415 - initialized by the live view by now

            keymap: dict[int, tuple[str, Optional[str], str]] = {}
            if self.single:
                moves, gripper_keys, still_keys = build_single_keymaps(
                    pygame, self.include_rotate_keys
                )
                for key, token in moves.items():
                    keymap[key] = (SINGLE_SIDE, token, KIND_MOVE)
                for key in gripper_keys:
                    keymap[key] = (SINGLE_SIDE, None, KIND_GRIPPER)
                for key in still_keys:
                    keymap[key] = (SINGLE_SIDE, None, KIND_STILL)
            else:
                left, right, grippers, still = build_dual_keymaps(
                    pygame, self.include_rotate_keys
                )
                for key, token in left.items():
                    keymap[key] = ("left", token, KIND_MOVE)
                for key, token in right.items():
                    keymap[key] = ("right", token, KIND_MOVE)
                for key, side in grippers.items():
                    keymap[key] = (side, None, KIND_GRIPPER)
                for key, side in still.items():
                    keymap[key] = (side, None, KIND_STILL)
            self._keymap = keymap
        return self._keymap

    # -- event capture (live-view stream thread) ---------------------------------
    def on_key(self, key: int) -> None:
        """Record one KEYDOWN. Unknown keys (P, Esc, ...) are ignored."""
        if not self.enabled:
            return
        entry = self._ensure_keymap().get(int(key))
        if entry is None:
            return
        side, token, kind = entry
        with self._lock:
            self.generation += 1
            if kind == KIND_GRIPPER:
                # Separate slot: a deliberate, rare toggle must not be eaten by a
                # later movement tap.
                self._pending_gripper[side] = True
            elif kind == KIND_STILL:
                # STILL both confirms "hold this arm" and CANCELS a pending move.
                self._pending_move[side] = "STILL"
            else:
                self._pending_move[side] = token  # newest wins
        # Instant capture receipt (from the live-view thread): the operator must
        # never wonder whether a press registered -- silence here is what made an
        # unfocused window look like a dead feature on hardware.
        label = {KIND_GRIPPER: "gripper toggle", KIND_STILL: "STILL"}.get(kind, token)
        prefix = "" if self.single else f"{side.upper()} "
        print(f"[dagger] {prefix}{label}")

    # -- runner-facing queries -----------------------------------------------------
    def has_intent(self) -> bool:
        """Whether any arm has an unconsumed human intent."""
        if not self.enabled:
            return False
        with self._lock:
            return any(self._pending_gripper.values()) or any(
                move is not None for move in self._pending_move.values()
            )

    def drain(self) -> dict[str, tuple[str, str]]:
        """Consume all pending intents: ``{side: (token_or_placeholder, kind)}``.

        The gripper slot wins over a move for the same arm (teleop's priority);
        its token placeholder is ``"GRIPPER"`` -- the runner resolves it into
        GRASP/RELEASE from that controller's actual state at execution time.
        """
        if not self.enabled:
            return {}
        intents: dict[str, tuple[str, str]] = {}
        with self._lock:
            for side in self.sides:
                if self._pending_gripper[side]:
                    intents[side] = ("GRIPPER", KIND_GRIPPER)
                elif self._pending_move[side] == "STILL":
                    intents[side] = ("STILL", KIND_STILL)
                elif self._pending_move[side] is not None:
                    intents[side] = (self._pending_move[side], KIND_MOVE)
                self._pending_gripper[side] = False
                self._pending_move[side] = None
        return intents
