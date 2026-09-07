"""MCQ tool: a controller *answer protocol*.

VLMs see a lot of multiple-choice VQA in pretraining, so picking a labeled option may
suit them better than emitting a bare action token. When enabled this tool *is* the
controller's answer protocol: it owns the output-contract text the prompt shows, the
alphabet the VLM may answer in (letters ``A``-``I``), the degraded-output fallback in
that alphabet, and the mapping that turns the chosen letter back into an atomic action.
Everything downstream of the controller therefore still sees ``MV_DOWN`` / ``GRASP`` /
``DONE`` as before.

When disabled the controller keeps its default atomic-token protocol (owned by
``core.vlm.roles``), so the output is byte-identical to today's behaviour.

The answer-protocol surface the controller depends on (duck-typed in ``core.vlm.roles``):
  * ``answer_tokens``        -- the allowed decode alphabet (option letters)
  * ``output_contract()``    -- the prompt tail asking for one letter
  * ``fallback_answer(act)`` -- an atomic action expressed as its option letter
  * ``map_response(resp)``   -- rewrite a letter decision into its atomic action
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

from plugins.prompt_text import fragment
from core.vlm.vlm_client import VLMResponse

# The atomic action vocabulary, in the canonical order shown in prompts/controller.txt.
ATOMIC_ACTIONS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
    "GRASP",
    "RELEASE",
    "DONE",
)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class McqPlugin:
    """Bidirectional letter <-> atomic-action map + the MCQ answer-protocol behaviour."""

    def __init__(
        self, enabled: bool = False, actions: Sequence[str] = ATOMIC_ACTIONS
    ) -> None:
        self.enabled = bool(enabled)
        self.actions = tuple(actions)
        if len(self.actions) > len(_LETTERS):
            raise ValueError("too many actions to label with single letters")
        self.letters = tuple(_LETTERS[: len(self.actions)])
        self._to_token = dict(zip(self.letters, self.actions))
        self._to_letter = dict(zip(self.actions, self.letters))

    # -- option labelling --------------------------------------------------
    @property
    def answer_tokens(self) -> tuple[str, ...]:
        """The allowed answer tokens the VLM may emit (the option letters)."""
        return self.letters

    def options_block(self) -> str:
        """The labelled option list, e.g. ``A) MV_FWD  B) MV_BACK  ... I) DONE``."""
        return "  ".join(f"{letter}) {token}" for letter, token in self._to_token.items())

    def to_token(self, letter: str) -> Optional[str]:
        """Map an answer letter to its atomic action, or ``None`` if unknown."""
        return self._to_token.get(str(letter).strip().upper())

    def to_letter(self, token: str) -> Optional[str]:
        """Map an atomic action to its answer letter, or ``None`` if unknown."""
        return self._to_letter.get(str(token).strip().upper())

    # -- answer protocol ---------------------------------------------------
    def output_contract(self) -> str:
        """The prompt tail asking the VLM to answer with a single option letter.
        The text lives in the co-located mcq.txt ({options} substituted here; replace,
        not format, so its literal JSON braces are safe)."""
        return fragment(__file__, "mcq.txt", "output_contract").replace(
            "{options}", self.options_block()
        )

    def fallback_answer(self, action_token: str) -> str:
        """An atomic action expressed in the answer alphabet, for the degraded-output
        safeguard. Unknown actions fall back to MV_DOWN's letter."""
        return self.to_letter(action_token) or self.to_letter("MV_DOWN")

    def map_response(self, response: VLMResponse) -> VLMResponse:
        """Rewrite an MCQ decision (an option letter) into its atomic action.

        Keeps the model's reasoning (tagging the chosen letter for provenance) and
        rebuilds the normalized decision JSON so everything downstream of the controller
        still sees a plain atomic action token (``MV_DOWN`` / ``GRASP`` / ``DONE`` ...)."""
        letter = response.token
        token = self.to_token(letter) or letter
        payload = dict(response.payload)
        json_obj = payload.get("json")
        raw_text = response.raw_text
        if isinstance(json_obj, dict):
            normalized = dict(json_obj)
            reasoning = str(normalized.get("reasoning") or "").strip()
            normalized["decision"] = token
            normalized["reasoning"] = (
                f"[{letter}] {reasoning}" if reasoning else f"chose {letter}"
            )
            payload["json"] = normalized
            raw_text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return VLMResponse(token=token, raw_text=raw_text, payload=payload)
