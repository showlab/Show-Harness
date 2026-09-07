from __future__ import annotations

import json
from typing import Any, Sequence

from core.prompting.wrist_marker import parse_wrist_marker, wrist_marker_prompt

from .vlm_client import VLMParseError, VLMResponse, recover_allowed_token


DIRECTION_TOKENS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
)
CONTROLLER_TOKENS = DIRECTION_TOKENS + ("GRASP", "RELEASE", "DONE")
TOKEN_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}


def _join_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _default_output_contract(extra_tokens: Sequence[str] = ()) -> str:
    """The controller's default answer protocol: one atomic-action token as JSON.

    This is the output contract that used to live inside ``prompts/controller.txt``; it
    now lives in the role so an answer-protocol tool (e.g. ``plugins.mcq``) can swap it
    without editing the prompt body. Protocol plugins own their own contract text.

    ``extra_tokens`` appends optional action tokens contributed by a tool (e.g.
    ``plugins.rotation``'s ROTATE_CW/CCW), so the offered set matches ``allowed_tokens``.
    """
    tokens = tuple(CONTROLLER_TOKENS) + tuple(extra_tokens)
    return (
        "Choose exactly one action:\n"
        + ", ".join(tokens)
        + '\nReturn JSON only: {"decision":"ONE_ACTION","reasoning":"one visual sentence"}'
    )


def _without_json_output_contract(prompt: str) -> str:
    lines = []
    skip_fields_line = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Return JSON only"):
            skip_fields_line = True
            continue
        if skip_fields_line and stripped:
            skip_fields_line = False
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _token_only_prompt(prompt: str, allowed_tokens: Sequence[str]) -> str:
    clean_prompt = _without_json_output_contract(prompt)
    return (
        "Answer with exactly one token from this list and nothing else:\n"
        + " ".join(allowed_tokens)
        + "\n\nUse the task, stage, image, and rules below only to choose that token.\n\n"
        + clean_prompt
        + "\n\nFinal answer: exactly one allowed token, no JSON, no markdown, no prose."
    )


def _decision_schema(allowed_tokens: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(allowed_tokens)},
            "reasoning": {"type": "string"},
        },
        "required": ["decision", "reasoning"],
        "additionalProperties": False,
    }


def _complete_decision_json(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    agentview_image,
    wrist_image=None,
    fallback_token: str | None = None,
    fallback_reason: str = "fallback after invalid decision JSON and token retry",
    debug: bool = False,
) -> VLMResponse:
    json_prompt = (
        prompt
        + '\n\nReturn JSON only. Put "decision" first, then "reasoning". '
        + 'Use one short sentence for "reasoning".'
    )
    try:
        response = client.complete_json(
            json_prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=_decision_schema(allowed_tokens),
            max_tokens=None,
            temperature=0.0,
            chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
            debug=debug,
        )
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Decision JSON missing object payload: {response.raw_text!r}")
        decision = str(payload.get("decision", "")).strip()
        if decision not in allowed_tokens:
            raise RuntimeError(
                f"Decision JSON has invalid decision {decision!r}; "
                f"allowed tokens are {list(allowed_tokens)}"
            )
        reasoning = str(payload.get("reasoning") or "").strip()
        normalized = {"decision": decision, "reasoning": reasoning}
        merged_payload = dict(response.payload)
        merged_payload["json"] = normalized
        return VLMResponse(
            token=decision,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=merged_payload,
        )
    except RuntimeError as exc:
        recovered = _recover_decision_from_malformed_json(exc, allowed_tokens)
        if recovered:
            decision, raw_text = recovered
            normalized = {
                "decision": decision,
                "reasoning": (
                    f"Recovered {decision} from malformed VLM JSON output. "
                    f"Raw output: {_one_line(raw_text)}"
                ),
            }
            payload = {
                "json": normalized,
                "recovered_from_malformed_json": True,
                "json_error": str(exc),
                "latency_s": 0.0,
            }
            return VLMResponse(
                token=decision,
                raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                payload=payload,
            )
        return _token_retry_or_fallback(
            client=client,
            prompt=prompt,
            allowed_tokens=allowed_tokens,
            fallback_token=fallback_token,
            fallback_reason=fallback_reason,
            agentview_image=agentview_image,
            wrist_image=wrist_image,
            debug=debug,
            json_error=exc,
            primary_reasoning=_reasoning_from_exception(exc),
        )


def _complete_cot_decision(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    agentview_image,
    wrist_image=None,
    fallback_token: str | None = None,
    fallback_reason: str = "cot fallback after no recoverable token",
    debug: bool = False,
) -> VLMResponse:
    """Reasoner path: a reasoner only grounds spatially when it reasons BEFORE
    answering, so this does NOT use guided JSON (which forces the answer at token
    0). The prompt makes the model write a localisation paragraph and end with
    'FINAL: <TOKEN>'; we recover the token from that prose. On failure, fall back
    to the strict token retry."""
    # Chain-of-thought needs headroom, but an unbounded budget lets a verbose thinker
    # (Gemma) decode ~1024 tokens every step -- the dominant latency cost. A backend can
    # cap it via `cot_max_tokens`; otherwise keep the old 1024 floor. chat_template_kwargs={}
    # keeps the backend's own thinking setting (Gemma: enable_thinking=true -> <think>).
    # strip_reasoning=False preserves the CoT for analysis.
    cot_budget = int(getattr(client, "cot_max_tokens", 0) or 0) or max(
        1024, int(getattr(client, "max_tokens", 0) or 0)
    )
    # Per-backend reasoning directive controls how much the model thinks -- length is
    # prompt-driven, not a token cap. Brief for a verbose thinker (Gemma), more thorough
    # for a concise one. Empty -> use the backend prompt as-is.
    directive = str(getattr(client, "reasoning_directive", "") or "").strip()
    cot_prompt = f"{prompt}\n\n{directive}" if directive else prompt
    response = client.complete_text(
        cot_prompt,
        agentview_image,
        wrist_image=wrist_image,
        max_tokens=cot_budget,
        temperature=0.0,
        chat_template_kwargs={},
        strip_reasoning=False,
        debug=debug,
    )
    raw_text = response.raw_text or ""
    token = recover_allowed_token(raw_text, allowed_tokens)
    if token in allowed_tokens:
        # Keep the complete chain-of-thought; steps.json logs it in full and
        # steps.jsonl truncates its own copy for compactness.
        reasoning = _one_line(raw_text)
        normalized = {"decision": token, "reasoning": reasoning}
        payload = dict(response.payload)
        payload["json"] = normalized
        payload["cot"] = True
        return VLMResponse(
            token=token,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )
    return _token_retry_or_fallback(
        client=client,
        prompt=prompt,
        allowed_tokens=allowed_tokens,
        fallback_token=fallback_token,
        fallback_reason=fallback_reason,
        agentview_image=agentview_image,
        wrist_image=wrist_image,
        debug=debug,
        primary_reasoning=_one_line(raw_text),
    )


def _token_retry_or_fallback(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    fallback_token: str | None,
    fallback_reason: str,
    agentview_image,
    wrist_image=None,
    debug: bool = False,
    json_error: RuntimeError | None = None,
    extra_fields: dict[str, str] | None = None,
    primary_reasoning: str = "",
) -> VLMResponse:
    tokens = tuple(allowed_tokens)
    if not tokens:
        raise RuntimeError("No allowed tokens available for VLM fallback")
    fallback = fallback_token if fallback_token in tokens else None
    try:
        retry = client.complete_token(
            _token_only_prompt(prompt, tokens),
            tokens,
            agentview_image,
            wrist_image=wrist_image,
            chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
            debug=debug,
        )
        # Keep the model's own reasoning from the first pass; the token just came from
        # a strict-retry call. Only fall back to the bare label when no reasoning exists.
        normalized = {
            "decision": retry.token,
            "reasoning": _retry_reasoning(
                primary_reasoning, "strict token retry after invalid JSON"
            ),
        }
        if extra_fields:
            normalized.update(extra_fields)
        payload: dict[str, Any] = {"json": normalized, "retry": retry.payload}
        if "latency_s" in retry.payload:
            payload["latency_s"] = retry.payload["latency_s"]
        if json_error is not None:
            payload["json_error"] = str(json_error)
        return VLMResponse(
            token=retry.token,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )
    except RuntimeError as retry_exc:
        if fallback is None:
            raise RuntimeError(
                "Invalid decision JSON and strict token retry failed; "
                f"json_error={json_error}; retry_error={retry_exc}"
            ) from retry_exc
        normalized = {
            "decision": fallback,
            "reasoning": _retry_reasoning(primary_reasoning, fallback_reason),
        }
        if extra_fields:
            normalized.update(extra_fields)
        payload = {
            "json": normalized,
            "fallback": True,
            "json_error": "" if json_error is None else str(json_error),
            "retry_error": str(retry_exc),
            "latency_s": 0.0,
        }
        return VLMResponse(
            token=fallback,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )


def _retry_reasoning(primary_reasoning: str, label: str) -> str:
    """The reasoning to log when the token came from a strict retry / commit fallback.
    Prefer the model's own first-pass reasoning, tagging how the token was recovered so
    the provenance is still visible; fall back to the bare label when none exists."""
    primary = (primary_reasoning or "").strip()
    if primary:
        return f"{primary} [{label}]"
    return label


def _reasoning_from_exception(exc: RuntimeError | None) -> str:
    """Best-effort recovery of the model's raw output text from a parse failure, so it
    can still be logged as the reasoning instead of a generic label."""
    raw = getattr(exc, "raw_text", "") or ""
    return _one_line(raw) if raw else ""


def _recover_decision_from_malformed_json(
    exc: RuntimeError, allowed_tokens: Sequence[str]
) -> tuple[str, str] | None:
    raw_text = str(getattr(exc, "raw_text", "") or "")
    if not raw_text and isinstance(exc, VLMParseError):
        raw_text = str(exc.raw_text or "")
    if not raw_text:
        return None
    token = recover_allowed_token(raw_text, allowed_tokens)
    if not token:
        return None
    return token, raw_text


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


class ControllerAgent:
    """Single per-step role: returns the executed base/grasp token directly."""

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        common_context: str,
        cot_mode: bool = False,
        gripper_color: str = "black",
        proprio_plugin: Any = None,
        mcq_plugin: Any = None,
        mem_text_plugin: Any = None,
        variable_step_plugin: Any = None,
        action_chunk_plugin: Any = None,
        rotation_plugin: Any = None,
        affordance_plugin: Any = None,
        action_ablation_plugin: Any = None,
        table_height_m: float | None = None,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.common_context = common_context
        self.cot_mode = cot_mode
        # Physical color of the gripper as seen in the camera views, substituted for
        # the prompt's {gripper_color} placeholder (default: "black").
        self.gripper_color = str(gripper_color or "black")
        # Optional controller plugins. proprio_plugin is a context provider (renders the
        # {proprio} block); mcq_plugin is an answer protocol (swaps {output_contract} and
        # the allowed answer tokens). Both default to off -> today's behaviour, so callers
        # that don't pass them are unaffected. table_height_m is
        # the constant the proprio tool needs and is fixed for the episode.
        self.proprio_plugin = proprio_plugin
        self.mcq_plugin = mcq_plugin
        # mem_text_plugin (context provider) owns the move-memory text: the {mem_text}
        # "Recent moves" line and the {mem_text_rules} history bullets. Injected like the
        # others; None -> those placeholders render empty (no move history in the prompt).
        self.mem_text_plugin = mem_text_plugin
        # Wrist-visibility consumers. variable_step_plugin (shared with the controller) sizes
        # the step from the TARGET's wrist visibility; action_chunk_plugin repeats a move while
        # the TARGET is far. Both read the shared "WRIST: YES/NO" judgment, so whenever EITHER
        # is enabled we render the marker (core.prompting.wrist_marker) and parse the VLM's reply into
        # response.payload["target_in_wrist"] for the runner to forward.
        self.variable_step_plugin = variable_step_plugin
        self.action_chunk_plugin = action_chunk_plugin
        # Optional rotation plugin: offers ROTATE_CW/CCW as extra controller tokens and, once
        # the gripper has yawed, compensates wrist-judged MV_* moves for that yaw (the
        # compensation lives in the controller; here we only surface the tokens + prompt).
        # It also needs the shared WRIST: YES/NO judgment, so it joins _wants_wrist().
        self.rotation_plugin = rotation_plugin
        # Optional affordance tool (context provider): the runner grounds each stage's
        # contact point and annotates the AgentView; here it only mediates the AFFORD
        # field ("<part> = RED dot in AgentView" while a dot is active). None/disabled
        # -> the planner's affordance text passes through unchanged.
        self.affordance_plugin = affordance_plugin
        # Action-type ablation (plugins.action_ablation). Its letters modes are an answer
        # protocol on the same duck interface as mcq (and win over mcq when both are
        # on); every enabled mode also funnels the FULLY ASSEMBLED prompt through
        # filter_final so run-time text obeys the setting.
        self.action_ablation_plugin = action_ablation_plugin
        self.table_height_m = table_height_m
        # The most recent fully-rendered controller prompt, for periodic logging.
        self.last_prompt = ""

    def _active_protocol(self) -> Any:
        """The active answer-protocol tool (ablation letters > mcq > None)."""
        ablation = self.action_ablation_plugin
        if (
            ablation is not None
            and getattr(ablation, "enabled", False)
            and getattr(ablation, "answer_protocol", False)
        ):
            return ablation
        return (
            self.mcq_plugin
            if (self.mcq_plugin is not None and self.mcq_plugin.enabled)
            else None
        )

    def _wants_wrist(self) -> bool:
        """Whether any consumer needs the shared WRIST: YES/NO wrist-visibility judgment."""
        return bool(
            getattr(self.variable_step_plugin, "enabled", False)
            or getattr(self.action_chunk_plugin, "enabled", False)
            or getattr(self.rotation_plugin, "enabled", False)
        )

    def decide(
        self,
        task: str,
        subgoal: dict[str, Any],
        recent_moves: str,
        previous_direction: str,
        gripper_state: str,
        agentview_image,
        wrist_image=None,
        prev_agentview_image=None,
        proprio: dict[str, Any] | None = None,
        recovery_context: str = "",
        debug: bool = False,
    ) -> VLMResponse:
        # Tools choose the prompt's context block and answer protocol. proprio (context
        # provider) is additive; mcq (answer protocol) is one-or-the-other with the default.
        proprio_block = (
            self.proprio_plugin.render(
                proprio, self.table_height_m, holding=(gripper_state == "CLOSED")
            )
            if self.proprio_plugin is not None
            else ""
        )
        gripper_proprio = (
            self.proprio_plugin.render_gripper(proprio)
            if self.proprio_plugin is not None
            else ""
        )
        mem_text = (
            self.mem_text_plugin.render_recent(recent_moves)
            if self.mem_text_plugin is not None
            else ""
        )
        mem_text_rules = (
            self.mem_text_plugin.render_rules() if self.mem_text_plugin is not None else ""
        )
        wrist_block = wrist_marker_prompt() if self._wants_wrist() else ""
        action_chunk_block = (
            self.action_chunk_plugin.render_prompt()
            if self.action_chunk_plugin is not None
            else ""
        )
        rotation_block = (
            self.rotation_plugin.render_prompt() if self.rotation_plugin is not None else ""
        )
        mcq = self._active_protocol()
        if mcq is not None:
            # An answer-protocol tool owns the whole alphabet; rotation (an extra action in
            # the default protocol) does not compose with it, so mcq wins if both are on.
            allowed_tokens: Sequence[str] = list(mcq.answer_tokens)
            output_contract = mcq.output_contract()
        else:
            rotation_tokens = (
                tuple(self.rotation_plugin.action_tokens())
                if self.rotation_plugin is not None
                else ()
            )
            allowed_tokens = tuple(CONTROLLER_TOKENS) + rotation_tokens
            output_contract = _default_output_contract(rotation_tokens)

        prompt = _join_prompt_parts(
            self.common_context,
            self.prompt_template.format(
                task=task,
                subgoal_json=json.dumps(subgoal, sort_keys=True),
                stage=str(subgoal.get("motion", "")),
                target=str(subgoal.get("target", "")),
                affordance=(
                    self.affordance_plugin.afford_field(
                        "arm", str(subgoal.get("affordance", ""))
                    )
                    if self.affordance_plugin is not None
                    else str(subgoal.get("affordance", ""))
                ),
                description=str(subgoal.get("description", "")),
                completion=str(subgoal.get("completion", "")),
                gripper_state=gripper_state,
                gripper_color=self.gripper_color,
                recent_moves=recent_moves or "none",
                mem_text=mem_text,
                mem_text_rules=mem_text_rules,
                variable_step=wrist_block,
                action_chunk=action_chunk_block,
                rotation=rotation_block,
                recovery=str(recovery_context or ""),
                proprio=proprio_block,
                gripper_proprio=gripper_proprio,
                output_contract=output_contract,
            ),
        )
        ablation = self.action_ablation_plugin
        ablation_on = ablation is not None and getattr(ablation, "enabled", False)
        # Blind mode's two-frame review: when the runner supplies the frame captured
        # BEFORE the previous direction token, the prompt asks the model to judge
        # that action from the before/after pair and update its table (notes are
        # never written blind, at decision time).
        review_symbol = (
            ablation.review_symbol(previous_direction)
            if (ablation_on and prev_agentview_image is not None)
            else None
        )
        if ablation_on:
            # Arm the blind harvest gate: only NOTE[<reviewed symbol>] is recorded
            # this step (None -> no review requested -> all NOTE lines ignored).
            ablation.begin_step(review_symbol)
            # One funnel for the whole setting: symbolize/strip the assembled prompt
            # (covers run-time injections) and substitute the blind table/review.
            prompt = ablation.filter_final(prompt, review_symbol=review_symbol)
        self.last_prompt = prompt
        if review_symbol is not None:
            # The review text promises the BEFORE frame as the LAST attached image.
            wrist_image = ([wrist_image] if wrist_image is not None else []) + [
                prev_agentview_image
            ]
        # On a rare double parse failure, commit to the last movement direction
        # instead of crashing the episode (degraded-output safeguard, not control).
        # Under MCQ the fallback must be expressed in the answer alphabet (a letter).
        fallback_token = (
            previous_direction if previous_direction in DIRECTION_TOKENS else "MV_DOWN"
        )
        if mcq is not None:
            fallback_token = mcq.fallback_answer(fallback_token)

        if self.cot_mode:
            response = _complete_cot_decision(
                self.client,
                prompt,
                allowed_tokens,
                agentview_image,
                wrist_image=wrist_image,
                fallback_token=fallback_token,
                fallback_reason="controller cot commit-fallback after no recoverable token",
                debug=debug,
            )
        else:
            response = _complete_decision_json(
                self.client,
                prompt,
                allowed_tokens,
                agentview_image,
                wrist_image=wrist_image,
                fallback_token=fallback_token,
                fallback_reason="controller commit-fallback after invalid JSON and token retry",
                debug=debug,
            )
        if mcq is not None:
            response = mcq.map_response(response)
        # Shared wrist-visibility judgment: recover the WRIST: YES/NO marker from the model's
        # reasoning/output and stash it on the payload for the runner to forward to its
        # consumers (variable_step step size, action_chunk step count). No marker -> None.
        if self._wants_wrist() and isinstance(response.payload, dict):
            json_obj = response.payload.get("json")
            reasoning = json_obj.get("reasoning") if isinstance(json_obj, dict) else ""
            text = str(reasoning or "") or (response.raw_text or "")
            if ablation is not None and getattr(ablation, "answer_protocol", False):
                # Letters modes: the model plans in ACT_* symbols (the prompt was
                # symbolized), but parse_plan reads atomic tokens.
                text = ablation.decode_symbols(text)
            target_in_wrist = parse_wrist_marker(text)
            response.payload["target_in_wrist"] = target_in_wrist
            # action_chunk: when far, recover the model's planned move sequence (PLAN: ...)
            # so the runner can execute it open-loop. [] -> single step.
            if getattr(self.action_chunk_plugin, "enabled", False):
                response.payload["chunk_plan"] = self.action_chunk_plugin.parse_plan(
                    text, target_in_wrist
                )
        return response
