"""Dual-arm MVTOKEN controller role: planner-free, stage-free, one atomic token per arm.

The dual-arm counterpart of :mod:`core.vlm.mvtoken_roles`. Where ``DualControllerAgent``
(:mod:`core.vlm.dual_roles`) drives the subgoal / JSON / CoT stack, this role is the bare-token
policy for the ``dual_cloth`` LoRAs: three views in, one atomic token per arm out. No planner,
no stage, no reasoning.

The three schemes are the SAME policy packaged three ways. They must match the training data
exactly -- ``train/data_preparation/rollouts_to_alpaca.py --dual --twice/--once/--chain`` and
the templates in ``prompts/<version>/dual_mvtoken_<scheme>.txt`` -- because a LoRA trained on
one contract cannot be served under another:

  once  — ONE call. The model answers ``"<left> <right>"``; the right token is conditioned on
          the left through the decoder's own autoregression. Cheapest, and the scheme our
          dual-arm comparison settled on -- prefer it unless you have measured otherwise.
  twice — TWO VLM calls per step. Both carry all three views; the prompt's ``{arm}`` says which
          arm to answer for, and the RIGHT call does NOT see the LEFT token (the two calls are
          conditionally independent given the images, so they can be batched in parallel).
          Costs two image encodings.
  chain — ONE image encoding, TWO answers: the left token, then a text-only follow-up asks for
          the right one (which sees the left). On paper this is ``twice``'s conditioning at
          ``once``'s cost; kept as an alternative, not as the recommended default.

STILL is a first-class token here and is NOT in the single-arm vocabulary: the arms were
teleoperated independently, so a step where only one arm moved recorded STILL for the other.
"""
from __future__ import annotations

from typing import Any

from core.record.images import image_manifest

from .dual_roles import DualDecision

# One of these per arm, per step. STILL = "hold position, the other arm is catching up".
# DONE = "the whole task is complete" (one synthesized DONE per episode in the training data).
DUAL_MVTOKEN_ACTIONS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
    "GRASP",
    "RELEASE",
    "STILL",
    "DONE",
)

SCHEMES = ("twice", "once", "chain")
SIDES = ("left", "right")
# The three views every scheme sends ahead of the prompt text, in WIRE ORDER -- the order the
# converter wrote into `images`, so the order the LoRA was trained on.
CAMERA_ORDER = ("agentview", "wrist_left", "wrist_right")


class DualMvTokenController:
    """One step -> one atomic token per arm, in the scheme the served LoRA was trained on.

    ``prompt_template`` is the per-step template for the scheme
    (``dual_mvtoken_<scheme>.txt``). ``followup_template`` is required by -- and only by --
    ``chain``: the short, text-only second user turn (``dual_mvtoken_chain_right.txt``).

    ``client`` is a :class:`core.vlm.vlm_client.VLMClient`; every scheme's request shape
    (``complete_action_token`` / ``_pair`` / ``_chain``) is a method on it.
    """

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        scheme: str,
        followup_template: str | None = None,
        extra_fields: dict | None = None,
    ) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if scheme == "chain" and not followup_template:
            raise ValueError(
                "scheme 'chain' needs followup_template "
                "(prompts/<version>/dual_mvtoken_chain_right.txt)"
            )
        self.client = client
        self.prompt_template = prompt_template
        self.scheme = scheme
        self.followup_template = followup_template or ""
        self.extra_fields = dict(extra_fields or {})
        # The most recent fully-rendered prompt, for the runner's periodic prompt dump.
        self.last_prompt = ""
        # Media parts of the most recent request, in wire order (parallel to `last_prompt`;
        # the runner passes it to EpisodeLogger.save_controller_prompt). Always the same three
        # views for every scheme -- see decide().
        self.last_media: list[dict] = []

    def _render(self, task: str, recent_left: str, recent_right: str, arm: str = "") -> str:
        # once/chain templates carry no {arm} field; format() ignores the extra kwarg.
        return self.prompt_template.format(
            task=task,
            recent_left=recent_left or "none",
            recent_right=recent_right or "none",
            arm=arm.upper(),
            **self.extra_fields,
        )

    def decide(
        self,
        task: str,
        recent_left: str,
        recent_right: str,
        agentview_image,
        wrist_left_image,
        wrist_right_image,
        debug: bool = False,
    ) -> DualDecision:
        # Three views in a fixed order: agentview, wrist_left, wrist_right. This is the order
        # the converter wrote into `images`, so it is the order the LoRA was trained on.
        wrists = [wrist_left_image, wrist_right_image]
        # Record what actually goes on the wire, for the controller-prompt log. Every scheme
        # ("once" / "chain" / the two "twice" calls) sends exactly these three images ahead of
        # the text, so one manifest describes them all.
        self.last_media = image_manifest(
            CAMERA_ORDER, (agentview_image, wrist_left_image, wrist_right_image)
        )

        if self.scheme == "twice":
            tokens: dict[str, str] = {}
            raws: list[str] = []
            latency_s = 0.0
            for side in SIDES:
                # Each call is rendered for ITS arm. The right call is deliberately NOT told
                # what the left one just chose -- that independence IS the scheme.
                prompt = self._render(task, recent_left, recent_right, arm=side)
                self.last_prompt = prompt
                response = self.client.complete_action_token(
                    prompt,
                    DUAL_MVTOKEN_ACTIONS,
                    agentview_image,
                    wrist_image=wrists,
                    debug=debug,
                )
                tokens[side] = response.token
                raws.append(f"{side}={response.token}")
                latency_s += float((response.payload or {}).get("latency_s") or 0.0)
            return DualDecision(
                tokens=tokens,
                reasoning="",
                raw_text=" ".join(raws),
                payload={"latency_s": latency_s, "scheme": "twice"},
            )

        prompt = self._render(task, recent_left, recent_right)
        self.last_prompt = prompt

        if self.scheme == "once":
            response = self.client.complete_action_token_pair(
                prompt,
                DUAL_MVTOKEN_ACTIONS,
                agentview_image,
                wrist_image=wrists,
                debug=debug,
            )
        else:  # chain
            response = self.client.complete_action_token_chain(
                prompt,
                self.followup_template,
                DUAL_MVTOKEN_ACTIONS,
                agentview_image,
                wrist_image=wrists,
                debug=debug,
            )

        left, right = response.token.split()
        return DualDecision(
            tokens={"left": left, "right": right},
            reasoning="",
            raw_text=response.raw_text,
            payload={**(response.payload or {}), "scheme": self.scheme},
        )
