"""Plugin enable/disable resolution for the ``plugins:`` section of a robot config.

Every capability the harness can mount is declared under a single ``plugins:`` block so
it can be turned on or off without touching code. Entries are booleans only; physical
calibrations and other parameters live in their owning config sections, not under
``plugins:``::

    plugins:
      subgoal: true
      proprioception: true
      recovery: true
      coords: false
      mcq: false

Resolution is deliberately permissive so a disabled or omitted plugin never crashes the
build: an omitted plugin falls back to the caller's ``default``. For compatibility with
older configs, ``{"enabled": bool}`` is still accepted, but new configs should use only
plain booleans.
"""
from __future__ import annotations

from typing import Any, Dict


def plugin_block(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """The effective ``plugins:`` mapping of a resolved robot config.

    Older configs named the block ``tools:``; that spelling is still honored (with a
    one-line notice) so existing setups keep working. When both keys are present the
    newer ``plugins:`` entries win key-by-key.
    """
    cfg = cfg or {}
    plugins = dict(cfg.get("plugins") or {})
    legacy = cfg.get("tools")
    if isinstance(legacy, dict) and legacy:
        if not getattr(plugin_block, "_warned", False):
            print("[config] note: the 'tools:' block is now 'plugins:'; both are read for now.")
            plugin_block._warned = True  # type: ignore[attr-defined]
        merged = dict(legacy)
        merged.update(plugins)
        return merged
    return plugins


class PluginsConfig:
    """Read-only view over the resolved ``plugins:`` mapping."""

    def __init__(self, raw: Dict[str, Any] | None = None) -> None:
        self._raw: Dict[str, Any] = dict(raw or {})

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None) -> "PluginsConfig":
        """Resolve from a full robot config, honoring the legacy ``tools:`` spelling."""
        return cls(plugin_block(cfg))

    def enabled(self, name: str, default: bool = True) -> bool:
        """Whether plugin ``name`` is enabled.

        ``default`` is used only when the plugin is omitted entirely; a present entry
        with no ``enabled`` key counts as enabled (the "default yes" convention).
        """
        spec = self._raw.get(name)
        if spec is None:
            return bool(default)
        if isinstance(spec, bool):
            return spec
        if isinstance(spec, dict):
            return bool(spec.get("enabled", True))
        # A truthy non-mapping (e.g. a string) is treated as "present and on".
        return bool(spec)
