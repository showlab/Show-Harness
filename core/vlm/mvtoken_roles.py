"""Fine-tuned controller role: planner-free, stage-free atomic-token policy.

The controller for the single-arm mvtoken LoRAs (served by vLLM, see
``scripts/serve_vlm.sh``). Unlike the multi-role subgoal pipeline in ``core/vlm/roles.py``
it is a single role with no subgoal planner: every step it sends one image-grounded
request and gets back exactly one atomic action token.

The request shape reproduces the training samples the adapter was fitted on: the two
images FIRST (agentview, then wrist), the prompt text appended AFTER them, no captions,
thinking off, temperature 0. Wire order is built in one place --
``vlm_client._message_content`` -- and mirrors the converter, which prepends the
``<image><image>`` markers to the rendered prompt (see
``train/data_preparation/rollouts_to_alpaca.py``).

Vocabulary is enforced at PARSE time, not at decode time: the request carries NO
``guided_choice``, so the model generates freely and ``_parse_single_token`` accepts only
``allowed_tokens`` (via a recovery ladder, raising when nothing maps -- the runner then
falls back). Constraining the decoder instead would hide whether the adapter actually
learned the vocabulary; the zero-shot path in ``vlm_client.complete_token`` does constrain
it, because there the model was never trained on this vocabulary at all.

The prompt is the lite training prompt: stage-free, ``{task}`` and ``{recent_moves}`` only.
The model still triggers GRASP/RELEASE from the images.
"""
from __future__ import annotations

import re
from typing import Any

from core.record.images import image_manifest

from .vlm_client import VLMResponse

# The atomic tokens the CONTROLLER can physically execute -- the vocabulary, not the policy.
# Which of these the model is actually allowed to emit is decided by the prompt (see
# _actions_from_prompt): the deployment is prompt-driven, so dropping a token from the prompt
# template drops it from the parsed/recovered set. DONE is the terminal token (the v1 data
# carries one synthesized DONE per episode): the runner ends the rollout when it is emitted.
MVTOKEN_ACTIONS = (
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


# The views sent ahead of the prompt text, in WIRE ORDER. A TRAINING CONTRACT: the converter
# emitted the ``<image>`` slots in this order, so swapping them silently feeds the model the
# wrong view. Both the request (``_ordered_images``) and the log manifest read this one tuple.
CAMERA_ORDER = ("agentview", "wrist")


def _actions_from_prompt(prompt_template: str) -> tuple[str, ...]:
    """The tokens the model may emit = exactly those the prompt template offers.

    Behavior is prompt-driven: deleting a token from the template removes it from the
    allowed set passed to the parser/recovery, so an un-finetuned model can no longer have
    a stray "MV_FWD" recovered into a legal action. ``MVTOKEN_ACTIONS`` is only the physical
    vocabulary; the prompt SELECTS from it. Matched on the RAW template (before
    ``{recent_moves}`` / ``{task}`` are filled) so runtime move-history never re-introduces a
    token the prompt omitted.
    """
    present = tuple(
        tok
        for tok in MVTOKEN_ACTIONS
        if re.search(rf"\b{re.escape(tok)}\b", prompt_template)
    )
    if not present:
        raise ValueError(
            "Prompt template offers none of the known action tokens "
            f"{list(MVTOKEN_ACTIONS)}; the model would have no legal action to emit."
        )
    return present


class MvTokenController:
    """One VLM call per step -> one atomic action token, in the infer.py request shape.

    ``extra_fields`` carries per-episode prompt fields the chosen template needs but the runner
    does not vary step to step -- e.g. the affordance prompt's ``{target}`` / ``{affordance}``
    grasp-point hint, planned once by the BASE model at episode start. Lite mode passes ``{}``;
    extra keys are simply ignored by templates that don't reference them.
    """

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        extra_fields: dict | None = None,
    ) -> None:
        self.client = client
        # The full user-message template (camera header + task/gripper/recent-moves body),
        # already in the `instruction + "\n\n" + input` form, minus the Stage line.
        self.prompt_template = prompt_template
        # Allowed action tokens are whatever THIS prompt offers, not a fixed constant -- so the
        # policy is fully prompt-driven (drop a token from the template -> it can't be emitted).
        self.allowed_tokens = _actions_from_prompt(prompt_template)
        # Per-episode template fields (e.g. affordance target/affordance); see class docstring.
        self.extra_fields = dict(extra_fields or {})
        # The most recent fully-rendered prompt, for periodic logging (matches the
        # `last_prompt` attribute the runner/logger look for on the controller).
        self.last_prompt = ""
        # The media parts of the most recent request, in WIRE ORDER, for the same periodic
        # logging (see `_describe_*`). Parallel to `last_prompt`: the runner reads it with
        # getattr(), so roles that never set it simply log no media section.
        self.last_media: list[dict] = []

    def decide(
        self,
        task: str,
        gripper_state: str,
        recent_moves: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> VLMResponse:
        """One step -> one atomic token. See :meth:`_ordered_images` for the wire order."""
        prompt = self.prompt_template.format(
            task=task,
            gripper_state=gripper_state,
            recent_moves=recent_moves or "none",
            **self.extra_fields,
        )
        self.last_prompt = prompt

        images = self._ordered_images(agentview_image, wrist_image)
        self.last_media = image_manifest(CAMERA_ORDER, images)
        # The client puts the first image first and then every "extra" view, all ahead of the
        # prompt text -- so handing it images[0] + images[1:] reproduces our order verbatim.
        return self.client.complete_action_token(
            prompt,
            self.allowed_tokens,
            images[0],
            wrist_image=images[1:] or None,
            debug=debug,
        )

    @staticmethod
    def _ordered_images(agentview_image, wrist_image) -> list:
        """The image sequence sent before the prompt text, in ``CAMERA_ORDER``."""
        images = [agentview_image]
        if wrist_image is not None:
            images.append(wrist_image)
        return images
