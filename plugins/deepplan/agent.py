"""DeepPlanResolver VLM role: resolve a <REASON> checkpoint into concrete subgoals.

The single VLM call for the DeepPlan capability, and the mirror image of
:class:`plugins.subgoal.agent.SubgoalPlannerAgent`. Where the planner turns a task +
image into the INITIAL plan, the resolver is fired mid-episode when execution reaches a
``motion == "REASON"`` pivot: it looks at the LIVE images, decides which branch of the
plan's conditional is now true, and returns the concrete stages to run next.

It is self-contained: it owns its prompt file (``deepplan_resolver.txt``, loaded by
default) and its output schema, and talks to a duck-typed VLM ``client``
(``complete_json`` / ``complete_text``) rather than a concrete client class.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Co-located prompt file: the capability owns its prompt rather than relying on the
# top-level prompts/ directory.
RESOLVER_PROMPT_PATH = Path(__file__).with_name("deepplan_resolver.txt")

# No-think regime: the resolver emits its decision JSON directly (no chain-of-thought),
# matching the planner. Thinking sends the model into a monologue that overruns the
# budget before any JSON appears.
NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}

# One subgoal's shape, identical to plugins.subgoal.agent.SUBGOAL_PLAN_SCHEMA's item, so a
# resolved stage parses through core.v0_types.Subgoal.from_dict exactly like a planned
# one. Kept local (not imported) to keep the capability self-contained.
_SUBGOAL_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "target": {"type": "string"},
        "affordance": {"type": "string"},
        "motion": {"type": "string"},
        "description": {"type": "string"},
        "completion": {"type": "string"},
    },
    "required": ["id", "target", "affordance", "motion", "description", "completion"],
    "additionalProperties": False,
}

# Output contract for one resolution. ``branch`` is a free-text label for logging only --
# the splice is driven ENTIRELY by ``subgoals``. It is intentionally NOT enum-constrained:
# the resolved case is whatever the planner's per-task rule defined, so a fixed vocabulary
# (e.g. FOUND/NOT_FOUND) would tie the capability to one task shape (search) and hurt both
# generalization and backend portability of guided decoding.
DEEPPLAN_RESOLVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "branch": {"type": "string"},
        "reasoning": {"type": "string"},
        "subgoals": {"type": "array", "minItems": 1, "items": _SUBGOAL_ITEM_SCHEMA},
    },
    "required": ["branch", "reasoning", "subgoals"],
    "additionalProperties": False,
}


def load_resolver_prompt() -> str:
    """Return the co-located resolver prompt template (with the ``{...}`` placeholders)."""
    return RESOLVER_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _join_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


class DeepPlanResolverAgent:
    """One VLM call that turns a reached <REASON> checkpoint + live image(s) into the
    concrete subgoal JSON to run next."""

    def __init__(
        self,
        client: Any,
        common_context: str,
        prompt_template: str | None = None,
    ) -> None:
        self.client = client
        self.common_context = common_context
        self.prompt_template = (
            prompt_template if prompt_template is not None else load_resolver_prompt()
        )
        # Decode budget. A reasoning/"thinking" backend (gemini/CoT) spends hidden tokens
        # BEFORE the JSON, and those count against max_tokens; resolving a conditional from a
        # cluttered live scene is a harder question than the initial plan, so the planner's
        # 2048 starves the JSON and the model returns empty content. Give the resolver a
        # generous floor (this is a one-shot-per-pivot call, latency is not critical) while
        # still honouring a larger configured budget. Same max(floor, max_tokens) idiom the
        # CoT controller path uses in core.vlm.roles.
        self.max_tokens = max(4096, int(getattr(client, "max_tokens", 0) or 0))

    def resolve(
        self,
        *,
        task: str,
        branch_rule: str,
        observe_condition: str,
        remaining_goal: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Return the parsed ``{branch, reasoning, subgoals}`` object, or raise.

        Raising (rather than returning a synthetic fallback like the planner) is
        deliberate: the runner must NOT splice a guessed plan at a pivot, so the owning
        tool catches the failure and ends the episode cleanly instead.
        """
        prompt = self._render(
            task=task,
            branch_rule=branch_rule,
            observe_condition=observe_condition,
            remaining_goal=remaining_goal,
        )
        errors: list[str] = []
        # 1) Guided JSON: strongest contract on backends that support guided decoding.
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=DEEPPLAN_RESOLVE_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            return _require_resolution(response.payload.get("json"))
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"guided_json: {_one_line(exc)}")
        # 2) Free JSON: drop guided decoding (some hosted decoders choke on the nested
        # schema and return empty); the explicit JSON shape in the prompt still parses.
        strict_prompt = (
            prompt
            + "\n\nReturn only the JSON object in the shape shown above. "
            "Do not include thought, reasoning, markdown, or prose."
        )
        try:
            response = self.client.complete_json(
                strict_prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=None,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                debug=debug,
            )
            return _require_resolution(response.payload.get("json"))
        except RuntimeError as exc:
            if "VLM returned" not in str(exc):
                raise
            errors.append(f"free_json: {_one_line(exc)}")
        # 3) Free text: last resort for a verbose backend; recover the first JSON object.
        if hasattr(self.client, "complete_text"):
            try:
                response = self.client.complete_text(
                    strict_prompt,
                    agentview_image,
                    wrist_image=wrist_image,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    chat_template_kwargs=NO_THINK_CHAT_TEMPLATE_KWARGS,
                    debug=debug,
                    strip_reasoning=False,
                )
                return _require_resolution(_parse_json_object(response.raw_text))
            except RuntimeError as exc:
                errors.append(f"text: {_one_line(exc)}")
        raise RuntimeError("DeepPlan resolver returned no usable JSON: " + "; ".join(errors))

    def _render(
        self, *, task: str, branch_rule: str, observe_condition: str, remaining_goal: str
    ) -> str:
        # Explicit substitution (not str.format): the template ends with a literal JSON
        # skeleton whose braces would break .format(), so we replace the named slots only.
        body = (
            self.prompt_template.replace("{task}", task)
            .replace("{branch_rule}", branch_rule)
            .replace("{observe_condition}", observe_condition or "(judge from the images)")
            .replace("{remaining_goal}", remaining_goal)
        )
        return _join_prompt_parts(self.common_context, body)


def _require_resolution(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        items = parsed.get("subgoals")
        if isinstance(items, list) and any(isinstance(it, dict) for it in items):
            return parsed
    raise RuntimeError(
        "VLM returned JSON without non-empty subgoals: "
        f"{json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed!r}"
    )


def _parse_json_object(raw_text: Any) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise RuntimeError(f"VLM returned non-JSON text: {text!r}")


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())
