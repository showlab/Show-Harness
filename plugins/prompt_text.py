"""Loader for a plugin's co-located prompt-fragment file.

Convention: every plugin that contributes text to the CONTROLLER prompt keeps that
text in a ``<name>.txt`` file next to its ``plugin.py``, split into named sections:

    [section_name]
    the fragment text...

A section runs to the next ``[header]`` line (or EOF); bodies are stripped of
leading/trailing whitespace, inner newlines kept verbatim. Placeholders like
``{gap_cm}`` are substituted by the owning plugin with ``str.replace`` (never
``str.format``), so fragment text may safely contain literal braces (e.g. the
JSON example in the MCQ output contract).

Rationale: the exact sentences a plugin injects into the prompt are
operator-inspectable (and editable) without reading Python -- the same
co-location the VLM-role plugins already use for their full prompts
(``subgoal_planner.txt``, ``rotation.txt``, ``video_ref.txt``, ...).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_HEADER = re.compile(r"^\[([A-Za-z0-9_.-]+)\]\s*$")


@lru_cache(maxsize=None)
def _load(path: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _HEADER.match(line)
        if match:
            current = sections.setdefault(match.group(1), [])
            continue
        if current is not None:
            current.append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def fragment(module_file: str, filename: str, section: str) -> str:
    """One named fragment from ``<module dir>/<filename>``.

    Raises ``KeyError`` naming the available sections on a miss, so a renamed or
    deleted section fails loudly at first render instead of silently blanking a
    prompt line."""
    path = Path(module_file).with_name(filename)
    sections = _load(str(path))
    if section not in sections:
        raise KeyError(
            f"{path.name}: no [{section}] section; available: {sorted(sections)}"
        )
    return sections[section]
