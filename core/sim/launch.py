"""Simulator harness wiring shared by the sim entry points.

The ``run_*_mvtoken.py`` sim entry points are thin CLIs over these
functions: config layering and the VLM client.
"""
from __future__ import annotations

import argparse
from typing import Any

from core.config import deep_merge, make_api_key_refresher, resolve_vlm_config
from core.vlm.vlm_client import VLMClient


def build_config(args: argparse.Namespace, robot_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for arg_name, cfg_name in [
        ("task_suite_name", "task_suite_name"),
        ("task_id", "task_id"),
        ("episode_index", "episode_index"),
        ("max_steps", "max_steps"),
        ("loop_period_s", "loop_period_s"),
        ("log_dir", "log_dir"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value
    cfg = deep_merge(robot_cfg, overrides)

    if args.vlm_url and args.vlm_url.lower() == "mock":
        raise ValueError("Mock VLM mode has been removed; provide a real vLLM URL.")
    # Collapse the selected backend profile into a flat vlm config, then let
    # explicit CLI flags win over the profile.
    vlm = resolve_vlm_config(cfg, backend=args.vlm_backend)
    if args.vlm_url:
        # --vlm-url / VLM_URL / VLLM_BASE_URL target the LOCAL vLLM. A hosted backend
        # (openai/gemini) carries its own base_url; overriding it would send its model id
        # to the wrong server, so scope the override to the vllm provider.
        if vlm.get("provider", "vllm") == "vllm":
            vlm["base_url"] = args.vlm_url
        else:
            print(
                f"[run] ignoring --vlm-url/VLM_URL for hosted backend "
                f"{vlm['backend']!r}; using its own endpoint {vlm['base_url']}."
            )
    if args.model:
        vlm["model"] = args.model
    cfg["vlm"] = vlm
    cfg["vlm_backend"] = vlm["backend"]
    return cfg


def make_vlm_client(args: argparse.Namespace, cfg: dict[str, Any]):
    vlm_cfg = cfg["vlm"]
    return VLMClient(
        base_url=vlm_cfg["base_url"],
        model=vlm_cfg["model"],
        api_key=vlm_cfg.get("api_key", "EMPTY"),
        timeout_s=float(vlm_cfg["timeout_s"]),
        max_tokens=int(vlm_cfg["max_tokens"]),
        temperature=float(vlm_cfg["temperature"]),
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
        cot_max_tokens=vlm_cfg.get("cot_max_tokens"),
        reasoning_directive=vlm_cfg.get("reasoning_directive"),
        provider=vlm_cfg.get("provider", "vllm"),
        api_dialect=vlm_cfg.get("api_dialect"),
        reasoning_effort=vlm_cfg.get("reasoning_effort"),
        max_retries=vlm_cfg.get("max_retries"),
        retry_base_delay_s=vlm_cfg.get("retry_base_delay_s"),
        retry_max_delay_s=vlm_cfg.get("retry_max_delay_s"),
        api_key_refresh=make_api_key_refresher(vlm_cfg),
    )


