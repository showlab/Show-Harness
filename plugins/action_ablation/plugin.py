"""Action-type ablation: how much does the ACTION REPRESENTATION itself carry?

Four settings over two axes -- token semantics (meaningful ``MV_LEFT`` names vs
opaque ``ACT_C`` symbols) x direction explanations (present vs absent):

    off           setting 1: MV_* + explanations           (baseline, byte-identical)
    bare          setting 2: MV_* only, explanations removed
    letters       setting 3: opaque symbols + explanations
    letters_blind setting 4: opaque symbols; TOLD the range (the six symbols ARE the
                  six directions up/down/left/right/fwd/back) but NOT the mapping.
                  The model blind-picks; on the NEXT step the frame captured BEFORE
                  that action rides along and the model judges the effect from the
                  before/after pair (``NOTE[ACT_X]: ...`` -> a self-written
                  symbol->effect table fed back every step, and persisted into the
                  rollout as action_table.json by the runner). Recorded symbols can
                  be chosen directly; unknowns keep being blind-picked.

``letters``/``letters_blind`` are ANSWER PROTOCOLS (the same duck-typed slot as
:class:`plugins.mcq.McqPlugin`): the VLM answers in symbols, :meth:`map_response`
rewrites them back to atomic tokens, so the runner/controller/logs never change.
``bare`` only transforms the prompt. Everything the model could read leaks through
one funnel -- :meth:`filter_final` runs on the FULLY ASSEMBLED prompt each step, so
run-time text (recent moves, memory rules, proprio hints) is symbolized too, and
the blind mode's record table + last-action review are substituted there each step.

Symbols are ``ACT_A`` style, not bare letters: the CoT token-recovery regex is
case-insensitive and a bare ``A`` collides with the English article.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from plugins.prompt_text import fragment
from core.vlm.vlm_client import VLMResponse

MODES = ("off", "bare", "letters", "letters_blind")

# Fixed token -> opaque-symbol map: ONLY the six axis directions are ever blinded
# (user decision). GRASP/RELEASE/DONE -- and ROTATE_CW/CCW when the rotation plugin
# offers them -- keep their names in every mode: the ablation targets DIRECTION
# semantics, not task semantics.
SYMBOLS = {
    "MV_FWD": "ACT_A",
    "MV_BACK": "ACT_B",
    "MV_LEFT": "ACT_C",
    "MV_RIGHT": "ACT_D",
    "MV_UP": "ACT_E",
    "MV_DOWN": "ACT_F",
}
_TOKENS_OF = {sym: tok for tok, sym in SYMBOLS.items()}
_PASSTHROUGH = ("GRASP", "RELEASE", "DONE")

# The blind mode's record line ("NOTE[ACT_F]: lowered the gripper"), written only
# when the prompt requests a two-frame REVIEW of the previous action.
_NOTE_RE = re.compile(r"NOTE\[\s*(ACT_[A-F])\s*\]\s*:\s*([^\n\"]+)", re.IGNORECASE)
_NOTE_MAX_CHARS = 90
# Direction-pair leak in the memory rules ("(Pairs: MV_LEFT/MV_RIGHT, ...)"): telling
# the model which tokens are opposites IS an explanation, so the no-explanation
# modes strip the parenthetical (the oscillation rule itself survives).
_PAIRS_RE = re.compile(r"\s*\(Pairs?:[^)]*\)")
# Substituted into the assembled prompt each step (the template is static; the
# table and the previous-action review are not).
_TABLE_SENTINEL = "<<ACTION_TABLE>>"
_REVIEW_SENTINEL = "<<BLIND_REVIEW>>"

# Blind-mode leak scrubbing. Other plugins' fragments pair direction tokens with
# advice; after symbolization those become explanations (observed on hardware:
# proprio's "If height > 8 cm, ACT_F first" plus mem_text's "prioritize ACT_E,
# ACT_B, ACT_F, or ACT_A" let the model name symbols without ever acting).
# Tier 1: targeted rewrites that keep the neutral info / the protocol line.
_BLIND_REWRITES = (
    # plugins/proprioception: keep the height/step facts, drop the descend directive.
    (re.compile(r"\s*If height > [^\n]*? cm, MV_DOWN first\.?"), ""),
    # plugins/mem_text: keep the no-regrasp rule, drop the escape-direction list.
    (re.compile(r"prioritize MV_UP, MV_BACK, MV_DOWN, or MV_FWD"), "move away first"),
    # plugins/mem_text: pure token advice -- drop the whole line.
    (
        re.compile(
            r"(?m)^.*When opposite directions appear in recent moves, "
            r"prioritize MV_DOWN or MV_UP\.?[ \t]*\n?"
        ),
        "",
    ),
    # plugins/action_chunk: keep the PLAN protocol, neutralize the descent example.
    (
        re.compile(r"\(e\.g\. never plan more MV_DOWN than the height above the table allows\)"),
        "(mind the height above the table and the step size)",
    ),
)
# Tier 2 safety net: any line still naming a blinded token is a leak from a
# fragment this tool does not know yet. The action-history line is exempt
# (symbolized history, not semantics); labeled task/subgoal field lines keep
# their prose with the token deleted; every other such line is dropped whole.
_DIRECTION_TOKEN_RE = re.compile(r"\b(?:MV_FWD|MV_BACK|MV_LEFT|MV_RIGHT|MV_UP|MV_DOWN)\b")
_HISTORY_LINE_RE = re.compile(r"^\s*Recent moves", re.IGNORECASE)
_FIELD_LINE_RE = re.compile(r"^\s*(?:TASK|STAGE|TARGET|AFFORD|Stage goal|DONE WHEN):", re.IGNORECASE)


def _fragment(section: str) -> str:
    return fragment(__file__, "action_ablation.txt", section)


def _replace_section(prompt: str, start: str, end: str, replacement: str) -> str:
    """CoordsPlugin's in-memory section swap (the source prompt file is untouched)."""
    i = prompt.find(start)
    if i < 0:
        return prompt
    j = prompt.find(end, i + len(start))
    if j < 0:
        return prompt
    return prompt[:i].rstrip() + "\n\n" + replacement.strip() + "\n\n" + prompt[j:].lstrip()


class ActionAblationPlugin:
    """Prompt/protocol transforms for the four action-representation settings."""

    def __init__(self, mode: str = "off", include_rotate: bool = False) -> None:
        mode = str(mode or "off").strip().lower()
        if mode not in MODES:
            raise ValueError(f"action_ablation_mode {mode!r}; choices: {MODES}")
        self.mode = mode
        self.include_rotate = bool(include_rotate)
        # Blind mode's per-episode table: symbol -> the model's own latest note,
        # plus the full sequence of note events for the rollout record.
        self.notes: dict[str, str] = {}
        self.note_history: list[dict[str, str]] = []
        # The symbol under two-frame review THIS step (armed by begin_step); only
        # its NOTE may be harvested -- a note with no before/after evidence is a
        # guess, not a record.
        self._review_symbol: Optional[str] = None

    # -- identity ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def answer_protocol(self) -> bool:
        """Whether this tool owns the answer alphabet (the mcq-style slot)."""
        return self.mode in ("letters", "letters_blind")

    def _symbols(self) -> list[str]:
        return ["ACT_A", "ACT_B", "ACT_C", "ACT_D", "ACT_E", "ACT_F"]

    # -- template transform (build time) ---------------------------------------
    def apply(self, prompt_template: str) -> str:
        """The controller template for this setting (in memory; files untouched).

        ``bare``/``letters_blind`` replace the whole DIRECTION section (the per-token
        direction rules ARE the explanations); ``letters`` keeps it -- the running
        symbolization in :meth:`filter_final` turns its rules into "-> ACT_C" form,
        which is exactly setting 3 (symbols WITH explanations)."""
        if self.mode == "bare":
            return _replace_section(
                prompt_template, "DIRECTION:", "ATTENTION:", _fragment("bare_direction")
            )
        if self.mode == "letters_blind":
            return _replace_section(
                prompt_template, "DIRECTION:", "ATTENTION:", _fragment("blind_direction")
            )
        return prompt_template

    # -- answer protocol (mcq-slot duck interface; letters modes only) ----------
    @property
    def answer_tokens(self) -> tuple[str, ...]:
        # Rotation rides along BY NAME when the rotation plugin is on: only the six
        # direction tokens are part of the symbol alphabet.
        rotate = ("ROTATE_CCW", "ROTATE_CW") if self.include_rotate else ()
        return tuple(self._symbols()) + rotate + _PASSTHROUGH

    def output_contract(self) -> str:
        section = "blind_contract" if self.mode == "letters_blind" else "letters_contract"
        actions = ", ".join(self.answer_tokens)
        return _fragment(section).replace("{actions}", actions)

    def fallback_answer(self, action_token: str) -> str:
        """The degraded-output fallback, expressed in the answer alphabet."""
        return SYMBOLS.get(str(action_token).strip().upper(), SYMBOLS["MV_DOWN"])

    def map_response(self, response: VLMResponse) -> VLMResponse:
        """Symbol decision -> atomic token; blind mode also harvests the NOTE line."""
        symbol = str(response.token).strip().upper()
        token = _TOKENS_OF.get(symbol, symbol)
        if self.mode == "letters_blind":
            self._harvest_note(response)
        payload = dict(response.payload)
        json_obj = payload.get("json")
        raw_text = response.raw_text
        if isinstance(json_obj, dict):
            normalized = dict(json_obj)
            reasoning = str(normalized.get("reasoning") or "").strip()
            normalized["decision"] = token
            normalized["reasoning"] = (
                f"[{symbol}] {reasoning}" if reasoning else f"chose {symbol}"
            )
            payload["json"] = normalized
            raw_text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return VLMResponse(token=token, raw_text=raw_text, payload=payload)

    # -- blind mode's two-frame review ------------------------------------------
    def review_symbol(self, previous_direction: str) -> Optional[str]:
        """The symbol to review this step, or None. Non-None only in blind mode and
        only when the previous executed action was one of the six blinded tokens --
        the caller must additionally hold the frame captured BEFORE that action."""
        if self.mode != "letters_blind":
            return None
        return SYMBOLS.get(str(previous_direction or "").strip().upper())

    def begin_step(self, review_symbol: Optional[str]) -> None:
        """Arm this step's review gate: harvest accepts only NOTE[<review_symbol>].
        None (no before-frame attached) -> every NOTE line is ignored."""
        self._review_symbol = review_symbol

    def decode_symbols(self, text: str) -> str:
        """Symbols -> atomic tokens, for parsers that read the model's prose
        (e.g. the action_chunk PLAN line, written in ACT_* under letters modes)."""
        if not self.answer_protocol:
            return text
        for symbol, token in _TOKENS_OF.items():
            text = re.sub(rf"\b{symbol}\b", token, text)
        return text

    # -- final-prompt funnel (every step) ---------------------------------------
    def filter_final(self, prompt: str, review_symbol: Optional[str] = None) -> str:
        """Transform the FULLY ASSEMBLED prompt, so run-time injections (recent
        moves, memory rules, proprio/recovery text) obey the setting too. In blind
        mode this also substitutes the current record table and, when
        ``review_symbol`` is set (the before-frame is attached), the two-frame
        review instruction for the previous action."""
        if not self.enabled:
            return prompt
        if self.mode in ("bare", "letters_blind"):
            prompt = _PAIRS_RE.sub("", prompt)
        if self.mode == "letters_blind":
            prompt = self._scrub_blind(prompt)
        if self.answer_protocol:
            # The chunk protocol names the answer alphabet; keep it mode-correct.
            prompt = prompt.replace("(MV_ tokens only)", "(action symbols only)")
            for token, symbol in SYMBOLS.items():
                prompt = re.sub(rf"\b{token}\b", symbol, prompt)
        if self.mode == "letters_blind":
            prompt = prompt.replace(_TABLE_SENTINEL, self._render_table())
            review = (
                _fragment("blind_review").replace("{symbol}", review_symbol)
                if review_symbol
                else ""
            )
            prompt = prompt.replace(_REVIEW_SENTINEL, review)
            # An empty review leaves a blank line behind; collapse it.
            prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        return prompt

    # -- the blind mode's self-maintained table ----------------------------------
    def _scrub_blind(self, prompt: str) -> str:
        """Remove direction-token directives leaking in from other plugins' text
        (tier 1 exact rewrites, then the tier-2 net) BEFORE symbolization, so no
        assembled line ever pairs an ACT_* symbol with its meaning."""
        for pattern, repl in _BLIND_REWRITES:
            prompt = pattern.sub(repl, prompt)
        kept = []
        for line in prompt.split("\n"):
            if _DIRECTION_TOKEN_RE.search(line) and not _HISTORY_LINE_RE.match(line):
                if not _FIELD_LINE_RE.match(line):
                    continue
                # Planner-authored prose: keep the sentence, delete the token.
                line = re.sub(r"[ \t]{2,}", " ", _DIRECTION_TOKEN_RE.sub("", line))
            kept.append(line)
        return "\n".join(kept)

    def _harvest_note(self, response: VLMResponse) -> None:
        if self._review_symbol is None:
            return
        json_obj = (
            response.payload.get("json") if isinstance(response.payload, dict) else None
        )
        reasoning = (
            str(json_obj.get("reasoning") or "") if isinstance(json_obj, dict) else ""
        )
        text = reasoning or str(response.raw_text or "")
        for symbol, note in _NOTE_RE.findall(text):
            symbol = symbol.upper()
            if symbol != self._review_symbol:
                # One action, one reviewed note: bulk NOTE dumps for symbols the
                # model never reviewed are guesses and must not fill the table.
                continue
            note = " ".join(note.split())
            # "one short clause": keep the first sentence only -- inside a JSON
            # reasoning string the model's prose continues on the SAME line after
            # the NOTE, and must not leak into the table.
            note = re.split(r"(?<=[.;!?])\s", note, maxsplit=1)[0][:_NOTE_MAX_CHARS]
            self.notes[symbol] = note
            self.note_history.append({"symbol": symbol, "note": note})

    def _render_table(self) -> str:
        lines = [_fragment("table_header")]
        for symbol in self._symbols():
            lines.append(f"{symbol}: {self.notes.get(symbol, '(unknown)')}")
        return "\n".join(lines)

    def table_record(self) -> Optional[dict[str, Any]]:
        """The rollout-persisted record (action_table.json): all six symbols, null
        while undiscovered, plus every note event in order. Deliberately NO ground
        truth mapping -- the file must reflect only what the model has learned.
        None outside blind mode -> nothing is written."""
        if self.mode != "letters_blind":
            return None
        return {
            "mode": self.mode,
            "notes": {symbol: self.notes.get(symbol) for symbol in self._symbols()},
            "note_history": list(self.note_history),
        }

    def metadata(self) -> dict[str, Any]:
        """Run-log record: which setting ran, and (blind) the final learned table."""
        if not self.enabled:
            return {}
        out: dict[str, Any] = {"mode": self.mode, "symbols": dict(SYMBOLS)}
        if self.mode == "letters_blind":
            out["learned_notes"] = dict(self.notes)
        return out
