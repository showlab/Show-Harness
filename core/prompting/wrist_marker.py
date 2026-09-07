"""Shared wrist-visibility signal: the controller-prompt marker and its parser.

Several controller plugins key off the same question -- *is the TARGET visible in the wrist
view?* ``variable_step`` uses it to pick the per-command step magnitude; ``action_chunk``
uses it to pick how many steps to commit per VLM call. Because it is one VLM judgment, the
prompt line and its parser live here (shared infrastructure), not inside any single tool.

The :class:`ControllerAgent` renders the marker whenever ANY consumer is enabled and parses
the VLM's reply into ``response.payload["target_in_wrist"]`` (True / False / None when no
explicit marker was emitted). Tools then read that one field; they never render or parse it.
"""
from __future__ import annotations

import re
from typing import Optional

from plugins.prompt_text import fragment


# Explicit marker the VLM writes so we never mistake casual prose for the judgment. The
# trailing \b stops partial-word matches (e.g. "WRIST: NOPE" / "WRIST: YESTERDAY").
_WRIST_MARKER = re.compile(r"WRIST\s*[:=]\s*(YES|NO)\b", re.IGNORECASE)


def wrist_marker_prompt() -> str:
    """The controller-prompt line asking the VLM to report wrist visibility as a marker.
    The text lives in the co-located wrist_marker.txt."""
    return fragment(__file__, "wrist_marker.txt", "marker")


def parse_wrist_marker(text: str) -> Optional[bool]:
    """Recover the VLM's wrist-visibility judgment from its reasoning/output text.

    Returns True (TARGET in the wrist view), False (not in view), or None when no explicit
    ``WRIST: YES/NO`` marker is present. Uses the LAST marker, i.e. the VLM's final word.
    """
    if not text:
        return None
    matches = _WRIST_MARKER.findall(str(text))
    if not matches:
        return None
    return matches[-1].upper() == "YES"
