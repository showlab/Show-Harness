"""Preemptible VLM decisions for the DAGGER human-override plugin.

All runners share the same contract: one controller ``decide`` runs in a worker
thread and is abandoned the moment human keys arrive; the runner then executes
the human's intent instead. An abandoned call keeps running in the background
and its result is DROPPED when eventually collected -- it was decided on a
pre-intervention observation, so executing it would fight the operator.
Dropping is double-guarded: the abandon mark, plus a generation check (the
plugin counts every captured key), so input that arrived and was consumed while
the call was in flight still invalidates it. At most one call is ever in
flight; a fresh one starts only after the stale one is collected.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from core.ui import console


class InterruptibleDecider:
    """Runs ``decide(**kwargs)`` in a worker thread, preempted by human keys.

    ``plugin`` is the DAGGER plugin (``has_intent()`` / ``generation``);
    ``decide`` is the VLM decision callable. Callers construct one per runner
    and invoke :meth:`decide` only while the plugin is enabled.
    """

    def __init__(self, plugin: Any, decide: Callable[..., Any]) -> None:
        self._plugin = plugin
        self._decide = decide
        self._inflight: Optional[dict[str, Any]] = None

    def decide(self, decide_kwargs: dict[str, Any]) -> Optional[Any]:
        """The response, or ``None`` the moment the plugin holds human input."""
        plugin = self._plugin
        while True:
            if plugin.has_intent():
                if self._inflight is not None:
                    self._inflight["stale"] = True
                return None
            entry = self._inflight
            if entry is None:
                box: dict[str, Any] = {}

                def _work(
                    kwargs: dict[str, Any] = decide_kwargs, out: dict[str, Any] = box
                ) -> None:
                    try:
                        out["result"] = self._decide(**kwargs)
                    except BaseException as exc:  # noqa: BLE001 - re-raised on collect
                        out["error"] = exc

                entry = self._inflight = {
                    "thread": threading.Thread(
                        target=_work, name="vlm-decide", daemon=True
                    ),
                    "box": box,
                    "gen": plugin.generation,
                    "stale": False,
                }
                entry["thread"].start()
            while entry["thread"].is_alive():
                if plugin.has_intent():
                    entry["stale"] = True
                    return None
                entry["thread"].join(timeout=0.05)
            self._inflight = None
            if entry["stale"] or entry["gen"] != plugin.generation:
                print(
                    console.c(
                        console.DIM,
                        "  [dagger] dropped a stale VLM decision "
                        "(human input superseded it)",
                    )
                )
                continue  # start a fresh call for the current observation
            if "error" in entry["box"]:
                raise entry["box"]["error"]
            return entry["box"]["result"]
