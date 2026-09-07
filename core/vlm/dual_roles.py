"""Dual-arm controller role: ONE VLM call per step decides BOTH arms' tokens.

The dual counterpart of :class:`core.vlm.roles.ControllerAgent` (unified control, "Mode B").
The model receives THREE images (front view + left wrist + right wrist) and the two
arms' current stages side by side, and answers with one action token PER ARM. The
per-arm alphabet is the single-arm one plus ``STILL`` -- the explicit "this arm does
not move this step" token that makes waiting (handover, collision avoidance, a
finished track) a first-class decision instead of an implicit gap.

Kept separate from ``core.vlm.roles`` on purpose: the single-arm decision helpers there are
shaped around one enum-constrained token, while the dual contract is a two-field JSON
(``{"left": ..., "right": ...}``) with its own CoT recovery (``FINAL: LEFT=... RIGHT=...``).
Nothing in the single-arm path changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.vlm.roles import CONTROLLER_TOKENS, DIRECTION_TOKENS, TOKEN_CHAT_TEMPLATE_KWARGS

STILL_TOKEN = "STILL"
DUAL_ARM_TOKENS = tuple(CONTROLLER_TOKENS) + (STILL_TOKEN,)
SIDES = ("left", "right")

IMAGE_LABELS = {"agentview": "Image A", "left": "Image B", "right": "Image C"}

# Rendered for an arm whose subgoal track is exhausted; the runner also force-overrides
# that arm's token to STILL (and parks the arm back at its BEGIN pose), so this only
# needs to keep the model consistent about why that gripper sits idle and out of the way.
_FINISHED_BLOCK = (
    "All stages complete. This arm has returned to its start pose, clear of the "
    "workspace. Keep it STILL."
)


def _vs_enabled(view_select: Any) -> bool:
    """Whether an (optional, duck-typed) view-select plugin is present and enabled."""
    return view_select is not None and bool(getattr(view_select, "enabled", False))


def _dual_output_contract(view_select: Any = None) -> str:
    vs = view_select if _vs_enabled(view_select) else None
    contract = (
        "Choose exactly one action PER ARM:\n"
        + ", ".join(DUAL_ARM_TOKENS)
        + '\nReturn JSON only: {"left":"ONE_ACTION",'
        + (vs.json_example_field("left") if vs else "")
        + '"right":"ONE_ACTION",'
        + (vs.json_example_field("right") if vs else "")
        + '"reasoning":"one visual sentence"}'
    )
    if vs:
        contract += "\n" + vs.contract_note("left_view/right_view")
    return contract


def _dual_cot_contract(view_select: Any = None) -> str:
    vs = view_select if _vs_enabled(view_select) else None
    final_line = vs.cot_final_line() if vs else "FINAL: LEFT=<ACTION> RIGHT=<ACTION>"
    contract = (
        "Allowed actions PER ARM:\n"
        + ", ".join(DUAL_ARM_TOKENS)
        + "\nEnd your answer with exactly one line:\n"
        + final_line
    )
    if vs:
        contract += "\n" + vs.contract_note("<VIEW>")
    return contract


def _dual_decision_schema(view_select: Any = None) -> dict[str, Any]:
    # Field order mirrors the contract example (view right after its arm's action);
    # the view fields exist only when the view-select plugin is enabled.
    extra = view_select.schema_fields() if _vs_enabled(view_select) else {}
    fields: dict[str, Any] = {}
    for side in SIDES:
        fields[side] = {"type": "string", "enum": list(DUAL_ARM_TOKENS)}
        if f"{side}_view" in extra:
            fields[f"{side}_view"] = extra[f"{side}_view"]
    fields["reasoning"] = {"type": "string"}
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


# Final-check contract: one strict visual judgment of whole-task completion, made
# AFTER both arms have finished and retreated (so nothing occludes the scene).
_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["complete", "reason"],
    "additionalProperties": False,
}


@dataclass
class DualDecision:
    """One per-step dual decision: the executed token for each arm."""

    tokens: dict[str, str]  # {"left": token, "right": token}
    reasoning: str
    raw_text: str
    payload: dict[str, Any] = field(default_factory=dict)
    # Per-arm guiding view ("WRIST"/"FRONT"), reported only under the view-select
    # tool; a missing side / empty dict means "no view -> no frame override".
    views: dict[str, Optional[str]] = field(default_factory=dict)


def recover_dual_tokens(raw_text: str) -> dict[str, str] | None:
    """Recover ``LEFT=<TOKEN> RIGHT=<TOKEN>`` (or ``"left": "<TOKEN>"``) from free text.

    Used on the CoT path and as the malformed-JSON salvage. Returns None unless BOTH
    arms' tokens are found (a single recovered arm is not a safe dual command).
    """
    text = str(raw_text or "")
    token_pattern = "|".join(re.escape(t) for t in DUAL_ARM_TOKENS)
    tokens: dict[str, str] = {}
    for side in SIDES:
        # Take the LAST match so the FINAL line wins over tokens quoted mid-reasoning.
        matches = re.findall(
            rf'(?:"?{side}"?|{side.upper()})\s*[=:]\s*"?\s*({token_pattern})\b',
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            return None
        tokens[side] = matches[-1]
    return tokens


class DualControllerAgent:
    """Single per-step dual role: three images in, one token per arm out."""

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        common_context: str,
        cot_mode: bool = False,
        proprio_plugin: Any = None,
        mem_text_plugin: Any = None,
        view_select_plugin: Any = None,
        affordance_plugin: Any = None,
        table_heights: dict[str, float] | None = None,
        empty_width_m: float | None = None,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.common_context = common_context
        self.cot_mode = cot_mode
        # Context providers, shared across arms (they are stateless renderers); the
        # per-arm data (proprio dict, recent moves, table height) arrives per decide().
        self.proprio_plugin = proprio_plugin
        self.mem_text_plugin = mem_text_plugin
        # Multi-view action selection: extends the output contract with each arm's
        # guiding view and recovers it into DualDecision.views (None/disabled -> the
        # contract, schema, and decisions are byte-identical to the tool-less path).
        self.view_select_plugin = view_select_plugin
        # Affordance dots: the runner grounds each arm's stage contact point and
        # annotates the front image; here it only mediates each arm's AFFORD field
        # ("<part> = RED dot in Front View" while that arm's dot is active).
        # None/disabled -> the planner's affordance text passes through unchanged.
        self.affordance_plugin = affordance_plugin
        # Per-arm table-contact reference heights (each arm has its own Z floor).
        self.table_heights = dict(table_heights or {})
        # Rig fact for the CLOSED gripper line: below this measured width a close
        # caught nothing. Lets the VLM tell a thin real hold (a pinched fabric fold
        # can hide below the wrist camera's frame) from an empty close by WIDTH
        # instead of guessing from a white-on-white image.
        self.empty_width_m = empty_width_m
        # The most recent fully-rendered dual prompt, for periodic logging.
        self.last_prompt = ""

    # -- prompt assembly -----------------------------------------------------
    def _gripper_line(
        self, gripper_state: str, proprio: dict[str, Any] | None
    ) -> str:
        """The ``Gripper now:`` value; a CLOSED gripper carries its measured width.

        Rendered only when the proprio tool is mounted (it owns measured context).
        """
        if (
            gripper_state != "CLOSED"
            or self.proprio_plugin is None
            or not isinstance(proprio, dict)
        ):
            return gripper_state
        try:
            width_cm = float(proprio.get("gripper_width")) * 100.0
        except (TypeError, ValueError):
            return gripper_state
        line = f"CLOSED, fingers {width_cm:.1f} cm apart"
        if self.empty_width_m:
            line += f" (an empty close reads under {self.empty_width_m * 100.0:.1f} cm)"
        return line

    def _arm_fields(
        self,
        side: str,
        subgoal: dict[str, Any] | None,
        gripper_state: str,
        recent_moves: str,
        recovery_context: str,
        proprio: dict[str, Any] | None,
    ) -> dict[str, str]:
        suffix = "l" if side == "left" else "r"
        if subgoal is None:
            # Track exhausted: collapse the whole block to the finished notice.
            return {
                f"stage_{suffix}": "FINISHED",
                f"target_{suffix}": "-",
                f"afford_{suffix}": "-",
                f"description_{suffix}": _FINISHED_BLOCK,
                f"completion_{suffix}": "already complete",
                f"gripper_{suffix}": gripper_state,
                f"mem_{suffix}": "",
                f"recovery_{suffix}": "",
                f"proprio_{suffix}": "",
            }
        mem = (
            self.mem_text_plugin.render_recent(recent_moves)
            if self.mem_text_plugin is not None
            else ""
        )
        proprio_block = (
            self.proprio_plugin.render(
                proprio,
                self.table_heights.get(side),
                holding=(gripper_state == "CLOSED"),
            )
            if self.proprio_plugin is not None
            else ""
        )
        afford = str(subgoal.get("affordance", ""))
        if self.affordance_plugin is not None:
            afford = self.affordance_plugin.afford_field(side, afford)
        return {
            f"stage_{suffix}": str(subgoal.get("motion", "")),
            f"target_{suffix}": str(subgoal.get("target", "")),
            f"afford_{suffix}": afford,
            f"description_{suffix}": str(subgoal.get("description", "")),
            f"completion_{suffix}": str(subgoal.get("completion", "")),
            f"gripper_{suffix}": self._gripper_line(gripper_state, proprio),
            f"mem_{suffix}": mem,
            f"recovery_{suffix}": str(recovery_context or ""),
            f"proprio_{suffix}": proprio_block,
        }

    def decide(
        self,
        task: str,
        subgoals: dict[str, dict[str, Any] | None],
        gripper_states: dict[str, str],
        recent_moves: dict[str, str],
        previous_directions: dict[str, str],
        recovery_contexts: dict[str, str],
        proprios: dict[str, dict[str, Any] | None],
        agentview_image,
        wrist_left_image,
        wrist_right_image,
        debug: bool = False,
    ) -> DualDecision:
        fields: dict[str, str] = {"task": str(task)}
        for side in SIDES:
            fields.update(
                self._arm_fields(
                    side,
                    subgoals.get(side),
                    gripper_states.get(side, "OPEN"),
                    recent_moves.get(side, "none"),
                    recovery_contexts.get(side, ""),
                    proprios.get(side),
                )
            )
        fields["mem_rules"] = (
            self.mem_text_plugin.render_rules() if self.mem_text_plugin is not None else ""
        )
        fields["output_contract"] = (
            _dual_cot_contract(self.view_select_plugin)
            if self.cot_mode
            else _dual_output_contract(self.view_select_plugin)
        )
        prompt = "\n\n".join(
            part.strip()
            for part in (self.common_context, self.prompt_template.format(**fields))
            if part and part.strip()
        )
        self.last_prompt = prompt
        # Degraded-output safeguard (mirrors the single-arm commit-fallback): continue
        # an arm's last movement direction, else keep it STILL -- never invent a grasp.
        fallback = {
            side: (
                previous_directions.get(side)
                if previous_directions.get(side) in DIRECTION_TOKENS
                else STILL_TOKEN
            )
            for side in SIDES
        }
        wrists = [wrist_left_image, wrist_right_image]
        if self.cot_mode:
            return self._decide_cot(prompt, agentview_image, wrists, fallback, debug)
        return self._decide_json(prompt, agentview_image, wrists, fallback, debug)

    # -- decision paths --------------------------------------------------------
    def _views_from_payload(self, payload: dict[str, Any]) -> dict[str, Optional[str]]:
        if self.view_select_plugin is None:
            return {}
        return self.view_select_plugin.views_from_payload(payload)

    def _parse_views(self, text: str) -> dict[str, Optional[str]]:
        if self.view_select_plugin is None:
            return {}
        return self.view_select_plugin.parse_views(text)

    def _decide_json(
        self, prompt: str, agentview, wrists, fallback: dict[str, str], debug: bool
    ) -> DualDecision:
        field_order = ["left"]
        if _vs_enabled(self.view_select_plugin):
            field_order.append("left_view")
        field_order.append("right")
        if _vs_enabled(self.view_select_plugin):
            field_order.append("right_view")
        field_order.append("reasoning")
        json_prompt = (
            prompt
            + '\n\nReturn JSON only. Put "left" first, then '
            + ", then ".join(f'"{name}"' for name in field_order[1:])
            + '. Use one short sentence for "reasoning".'
        )
        try:
            response = self.client.complete_json(
                json_prompt,
                agentview,
                wrist_image=wrists,
                schema=_dual_decision_schema(self.view_select_plugin),
                max_tokens=None,
                temperature=0.0,
                chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError(f"Dual decision JSON missing object: {response.raw_text!r}")
            tokens = {side: str(payload.get(side, "")).strip() for side in SIDES}
            invalid = [s for s, t in tokens.items() if t not in DUAL_ARM_TOKENS]
            if invalid:
                raise RuntimeError(
                    f"Dual decision has invalid tokens {tokens!r}; "
                    f"allowed: {list(DUAL_ARM_TOKENS)}"
                )
            reasoning = str(payload.get("reasoning") or "").strip()
            views = self._views_from_payload(payload)
            merged = dict(response.payload)
            merged["json"] = {
                **tokens,
                **{f"{s}_view": v for s, v in views.items() if v},
                "reasoning": reasoning,
            }
            return DualDecision(
                tokens=tokens,
                reasoning=reasoning,
                raw_text=response.raw_text,
                payload=merged,
                views=views,
            )
        except RuntimeError as exc:
            raw = str(getattr(exc, "raw_text", "") or exc)
            recovered = recover_dual_tokens(raw)
            if recovered:
                return DualDecision(
                    tokens=recovered,
                    reasoning=f"Recovered from malformed dual JSON: {_one_line(raw)}",
                    raw_text=raw,
                    payload={"recovered_from_malformed_json": True, "latency_s": 0.0},
                    views=self._parse_views(raw),
                )
            return self._fallback(fallback, f"invalid dual decision JSON: {_one_line(raw)}")

    def _decide_cot(
        self, prompt: str, agentview, wrists, fallback: dict[str, str], debug: bool
    ) -> DualDecision:
        # Same rationale as the single-arm CoT path: a reasoner grounds spatially only
        # when it reasons BEFORE answering, so no guided JSON; the token pair is
        # recovered from the trailing "FINAL: LEFT=... RIGHT=..." line.
        cot_budget = int(getattr(self.client, "cot_max_tokens", 0) or 0) or max(
            1024, int(getattr(self.client, "max_tokens", 0) or 0)
        )
        directive = str(getattr(self.client, "reasoning_directive", "") or "").strip()
        cot_prompt = f"{prompt}\n\n{directive}" if directive else prompt
        response = self.client.complete_text(
            cot_prompt,
            agentview,
            wrist_image=wrists,
            max_tokens=cot_budget,
            temperature=0.0,
            chat_template_kwargs={},
            strip_reasoning=False,
            debug=debug,
        )
        raw_text = response.raw_text or ""
        tokens = recover_dual_tokens(raw_text)
        if tokens:
            payload = dict(response.payload)
            payload["cot"] = True
            return DualDecision(
                tokens=tokens,
                reasoning=_one_line(raw_text),
                raw_text=raw_text,
                payload=payload,
                views=self._parse_views(raw_text),
            )
        # Strict retry: re-ask for the FINAL line only, thinking off.
        final_line = (
            self.view_select_plugin.cot_final_line()
            if _vs_enabled(self.view_select_plugin)
            else "FINAL: LEFT=<ACTION> RIGHT=<ACTION>"
        )
        try:
            retry = self.client.complete_text(
                prompt
                + "\n\nCritical output format: answer with exactly one line and nothing "
                + "else:\n"
                + final_line
                + "\nAllowed actions: "
                + ", ".join(DUAL_ARM_TOKENS),
                agentview,
                wrist_image=wrists,
                max_tokens=None,
                temperature=0.0,
                chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            tokens = recover_dual_tokens(retry.raw_text or "")
            if tokens:
                return DualDecision(
                    tokens=tokens,
                    reasoning=_retry_reasoning(raw_text, "strict dual-token retry"),
                    raw_text=retry.raw_text or "",
                    payload={"retry": retry.payload},
                    views=self._parse_views(retry.raw_text or ""),
                )
        except RuntimeError:
            pass
        return self._fallback(fallback, _retry_reasoning(raw_text, "dual cot fallback"))

    @staticmethod
    def _fallback(fallback: dict[str, str], reason: str) -> DualDecision:
        return DualDecision(
            tokens=dict(fallback),
            reasoning=reason,
            raw_text=json.dumps({**fallback, "reasoning": reason}, ensure_ascii=False),
            payload={"fallback": True, "latency_s": 0.0},
        )

    # -- final task verification ------------------------------------------------
    def verify_task(
        self,
        task: str,
        agentview_image,
        wrist_left_image,
        wrist_right_image,
        debug: bool = False,
    ) -> tuple[bool, str]:
        """One strict visual check of whole-task completion, after both arms finish.

        The per-stage DONE judgments happen while a gripper often occludes the drop
        point, so an episode can reach "plan complete" with an object dropped beside
        its destination (observed on hardware). This re-judges the TASK from a fresh,
        retreated observation. Returns ``(complete, reason)``; any VLM/parse failure
        returns ``(True, ...)`` -- verification must never turn a finished episode
        into a crash or a spurious replan.
        """
        prompt = (
            f"TASK: {task}\n\n"
            "Image A = front view. Image B = LEFT wrist view. Image C = RIGHT wrist view.\n"
            "Both robot arms have finished their plans and retreated clear.\n"
            "Judge STRICTLY from the images whether the task is fully complete: every "
            "object at its required destination, nothing dropped beside or outside it.\n"
            'Return JSON only: {"complete":true|false,"reason":"one visual sentence"}'
        )
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=[wrist_left_image, wrist_right_image],
                schema=_VERIFY_SCHEMA,
                max_tokens=None,
                temperature=0.0,
                chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            payload = response.payload.get("json")
            if isinstance(payload, dict) and isinstance(payload.get("complete"), bool):
                return bool(payload["complete"]), str(payload.get("reason") or "")
            raw = str(response.raw_text or "")
        except RuntimeError as exc:
            raw = str(getattr(exc, "raw_text", "") or exc)
        match = re.search(r'"complete"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true", _one_line(raw)[:200]
        return True, f"verification unavailable, accepting completion ({_one_line(raw)[:120]})"


def _retry_reasoning(primary: str, label: str) -> str:
    primary = _one_line(primary)
    return f"{primary} [{label}]" if primary else label


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())
