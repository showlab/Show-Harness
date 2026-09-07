"""Move-memory text tool: the controller's recent-move history + history-based rules.

A controller prompt *context provider* (like proprioception). It owns the move-memory
text: the "Recent moves, newest first" line and the rules that reason over it (avoid
oscillation, what to do after an empty grasp). Enabled -> these are injected into the
prompt; disabled -> the controller prompt carries NO move history at all, so the VLM
decides purely from the current images and the other context.
"""
from __future__ import annotations

from plugins.prompt_text import fragment


def _fragment(section: str) -> str:
    """This tool's prompt text lives in the co-located mem_text.txt (the recent-moves
    line and the move-history rules, with their hardware rationale)."""
    return fragment(__file__, "mem_text.txt", section)


class MemTextPlugin:
    """Render the controller prompt's move-memory line and its history rules."""

    def __init__(self, enabled: bool = True, max_recent: int = 3) -> None:
        self.enabled = bool(enabled)
        # How many recent moves the "Recent moves" line shows (and the runner retains).
        self.max_recent = max(1, int(max_recent))

    def render_recent(self, recent_moves: str) -> str:
        """The 'Recent moves, newest first: ...' context line, or '' when disabled."""
        if not self.enabled:
            return ""
        moves = str(recent_moves or "").strip() or "none"
        return _fragment("recent").replace("{moves}", moves)

    def render_rules(self) -> str:
        """The move-history rule bullets (newline-joined), or '' when disabled."""
        if not self.enabled:
            return ""
        return _fragment("rules")
