"""Shared terminal formatting for the rollout entry points and runners.

One place for the ANSI styling used by the run headers, plan dumps, and per-step
blocks, so the single-arm (Franka) and dual-arm (Piper) paths read as the same
interface. Styling is applied only on a real terminal -- piped logs / CI stay
plain text. On truecolor terminals the accents use the same Morandi tones as the
live window and the saved analysis video (dusty teal LEFT / violet RIGHT / sage /
ochre / terracotta); otherwise they degrade to the nearest classic ANSI color.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import Any

_TTY = sys.stdout.isatty()
_TRUECOLOR = _TTY and os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def _rgb(r: int, g: int, b: int, fallback: str) -> str:
    return f"38;2;{r};{g};{b}" if _TRUECOLOR else fallback


DIM, BOLD = "2", "1"
GREEN = _rgb(22, 163, 74, "32")
YELLOW = _rgb(217, 119, 6, "33")
RED = _rgb(220, 38, 38, "31")
SIDE_COLOR = {
    "left": _rgb(13, 148, 136, "36"),
    "right": _rgb(124, 58, 237, "35"),  # violet, deliberately NOT red/pink
}
# Single-arm accent (the Franka path has no side identity): the LEFT teal, so the
# terminal, live window, and analysis video keep one accent per arm everywhere.
ARM_COLOR = SIDE_COLOR["left"]


def c(code: str, text: str) -> str:
    """Wrap ``text`` in one ANSI style ``code`` (no-op when piped)."""
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def dim(text: str) -> str:
    return c(DIM, text) if text else ""


def rule(title: str, width: int = 68) -> str:
    """A dim section rule for run headers / footers (plain text when piped)."""
    line = f"─── {title} " + "─" * max(0, width - len(title) - 5)
    return c(DIM, line)


def short(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def wrap_reason(reasoning: Any, strip: str | None = None, width: int = 94) -> list[str]:
    """The model's own words, wrapped -- never truncated (this is the one thing the
    operator actually needs from the terminal). ``strip`` removes a trailing
    machine-readable pattern (e.g. the dual ``FINAL: LEFT=... RIGHT=...`` line)
    whose content is already shown elsewhere."""
    text = " ".join(str(reasoning or "").split())
    if strip:
        text = re.sub(strip, "", text, flags=re.I)
    if not text:
        return ["(no reasoning returned)"]
    return textwrap.wrap(text, width=width) or [text]
