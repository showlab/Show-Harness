"""Coordinate-system prompt tool.

When disabled, this tool returns the controller prompt unchanged. When enabled, it
overrides only the controller prompt's ``DIRECTION:`` and ``REMARK:`` sections in
memory before the prompt template is passed to the controller agent. The source
``prompts/controller.txt`` file is not modified.

The active prompt keeps atomic actions axis-only, then adds a minimal camera-to-axis
calibration so the VLM can translate raw image evidence into robot-frame displacement
without reusing the base prompt's full image-direction policy.
"""
from __future__ import annotations

from plugins.prompt_text import fragment


class CoordsPlugin:
    """Prompt-only coordinate-system override for the controller role."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def apply(self, prompt_template: str) -> str:
        """Return ``prompt_template`` unchanged or with coordinate-mode sections.
        The replacement blocks live in the co-located coords.txt."""
        if not self.enabled:
            return prompt_template
        direction = fragment(__file__, "coords.txt", "direction")
        remark = fragment(__file__, "coords.txt", "remark")
        prompt = _replace_section(prompt_template, "DIRECTION:", "REMARK:", direction)
        return _replace_section(prompt, "REMARK:", "GRIPPER:", remark)


def _replace_section(
    prompt: str, start_marker: str, end_marker: str, replacement: str
) -> str:
    start = prompt.find(start_marker)
    if start < 0:
        return prompt
    end = prompt.find(end_marker, start + len(start_marker))
    if end < 0:
        return prompt
    before = prompt[:start].rstrip()
    after = prompt[end:].lstrip()
    return before + "\n\n" + replacement.strip() + "\n\n" + after

