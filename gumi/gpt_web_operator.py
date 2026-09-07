#!/usr/bin/env python3
"""Run GPT VLM automation against an existing Show-Harness web-teleop server.

Start the teleop server first, then this supervisor.  The dashboard starts PAUSED;
no robot action is sent until the user clicks Run or Step once.

Examples
--------
# Dual-arm simulator + GPT dashboard:
.venv/bin/python gumi/collect_rollouts_web_dual.py data/rollouts_dual --sim
.venv/bin/python gumi/gpt_web_operator.py --target-url http://localhost:8620

# Single-arm keyboard teleop target:
.venv/bin/python gumi/gpt_web_operator.py --target-url http://localhost:8610

# One model decision without executing or recording anything:
.venv/bin/python gumi/gpt_web_operator.py \
  --target-url http://localhost:8620 --once --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml, make_api_key_refresher, resolve_vlm_config
from core.vlm.vlm_client import VLMClient
from gumi.gpt_operator.operator import (
    GPTWebOperator,
    OperatorConfig,
    TeleopHTTPClient,
)
from gumi.gpt_operator.server import serve

DEFAULT_PROMPT = ROOT / "prompts" / "gpt_web_operator.txt"
DEFAULT_CONFIG = ROOT / "configs" / "robot_piper.yaml"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM operator + dashboard for a GUMI web-teleop server."
    )
    parser.add_argument("--target-url", default="http://localhost:8620")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8630)
    parser.add_argument("--robot-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--vlm-backend", default="chatgpt",
                    help="A vlm_backends profile from --robot-config.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-repeat", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--image-max-side", type=int, default=768)
    parser.add_argument(
        "--image-detail", choices=("low", "high", "original", "auto"), default="high"
    )
    parser.add_argument("--trace-dir", default=str(ROOT / "data" / "gpt_operator_traces"))
    parser.add_argument("--dry-run", action="store_true", help="Decide and trace, but never act.")
    parser.add_argument("--no-auto-record", action="store_true")
    parser.add_argument("--no-auto-save", action="store_true")
    parser.add_argument("--auto-run", action="store_true", help="Begin looping immediately.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one GPT cycle, print its event JSON, then exit without a dashboard.",
    )
    # A command copied from a rendered one-line snippet can contain ``\  --flag``
    # (backslash followed by spaces) instead of a backslash immediately followed by
    # a newline. Bash turns every escaped space into a whitespace-only argument.
    # Ignore only those empty/whitespace arguments; all real unknown arguments must
    # still fail normally so misspelled safety options are never hidden.
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    return parser.parse_args([arg for arg in raw_argv if str(arg).strip()])


def _assert_token_not_expired(vlm_cfg: dict) -> None:
    """Give a useful error before a long vision call when a JWT bearer token is stale."""
    if not str(vlm_cfg.get("api_key", "")).startswith("eyJ"):  # not a JWT
        return
    token = str(vlm_cfg.get("api_key") or "")
    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        expires = float(json.loads(base64.urlsafe_b64decode(segment)).get("exp"))
    except Exception:
        return
    if time.time() >= expires:
        raise SystemExit(
            "The API bearer token has expired; refresh it and retry."
        )


def build_vlm(cfg: dict, args: argparse.Namespace) -> VLMClient:
    vlm_cfg = resolve_vlm_config(cfg, args.vlm_backend)
    if args.model:
        vlm_cfg["model"] = args.model
    if args.base_url:
        vlm_cfg["base_url"] = args.base_url
    vlm_cfg["reasoning_effort"] = args.reasoning_effort
    _assert_token_not_expired(vlm_cfg)
    key = vlm_cfg.get("api_key", "")
    if not key or key == "EMPTY":
        name = vlm_cfg.get("api_key_env") or "api_key"
        raise SystemExit(f"No GPT API credential resolved from {name}")
    return VLMClient(
        base_url=vlm_cfg["base_url"],
        model=vlm_cfg["model"],
        api_key=key,
        timeout_s=float(vlm_cfg.get("timeout_s", args.timeout_s)),
        max_tokens=args.max_output_tokens,
        temperature=float(vlm_cfg.get("temperature", 0.0)),
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
        provider=vlm_cfg.get("provider", "openai"),
        api_dialect=vlm_cfg.get("api_dialect"),
        reasoning_effort=vlm_cfg.get("reasoning_effort"),
        max_retries=vlm_cfg.get("max_retries"),
        retry_base_delay_s=vlm_cfg.get("retry_base_delay_s"),
        retry_max_delay_s=vlm_cfg.get("retry_max_delay_s"),
        api_key_refresh=make_api_key_refresher(vlm_cfg),
    )


def main() -> int:
    args = parse_args()
    load_secrets_env()
    cfg = load_yaml(args.robot_config)
    prompt_path = Path(args.prompt)
    if not prompt_path.is_file():
        raise SystemExit(f"GPT operator prompt not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    vlm = build_vlm(cfg, args)
    op_cfg = OperatorConfig(
        target_url=args.target_url,
        interval_s=args.interval_s,
        request_timeout_s=args.timeout_s,
        confidence_threshold=args.confidence_threshold,
        max_repeat=args.max_repeat,
        max_steps=args.max_steps,
        max_output_tokens=args.max_output_tokens,
        image_max_side=args.image_max_side,
        image_detail=args.image_detail,
        auto_record=not args.no_auto_record,
        auto_save=not args.no_auto_save,
        dry_run=args.dry_run,
        trace_root=Path(args.trace_dir),
    )
    target = TeleopHTTPClient(args.target_url, timeout_s=args.timeout_s)
    try:
        initial = target.state()
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach teleop server at {args.target_url}: {exc}\n"
            "Start collect_rollouts_web.py or collect_rollouts_web_dual.py first."
        ) from exc
    operator = GPTWebOperator(target, vlm, prompt, op_cfg)
    operator.target_state = initial

    if args.once:
        operator.start_worker()
        operator.step_once()
        deadline = time.monotonic() + max(120.0, args.timeout_s + 30.0)
        while time.monotonic() < deadline:
            snap = operator.snapshot()
            if snap["cycle"] >= 1 and not snap["busy"] and snap["pending_once"] == 0:
                print(json.dumps((snap.get("events") or [{}])[0], ensure_ascii=False, indent=2))
                operator.shutdown()
                return 1 if snap.get("last_error") else 0
            time.sleep(0.05)
        operator.shutdown()
        raise SystemExit("Timed out waiting for the one-shot GPT cycle")

    operator.start_worker()
    if args.auto_run:
        operator.resume()
    httpd = serve(operator, host=args.host, port=args.port)
    print(
        f"[gpt-web-operator] dashboard: http://{socket.gethostname()}:{args.port}/ "
        f"(bound {args.host}:{args.port})"
    )
    print(f"[gpt-web-operator] target:    {args.target_url}/")
    print(f"[gpt-web-operator] model:     {vlm.model} ({args.reasoning_effort} reasoning)")
    print(f"[gpt-web-operator] traces:    {operator.trace_writer.session_dir.resolve()}")
    if args.auto_run:
        print("[gpt-web-operator] AUTO-RUN enabled; use Pause to stop new actions.")
    else:
        print("[gpt-web-operator] starts PAUSED; use the dashboard Run / Step once controls.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gpt-web-operator] interrupted")
    finally:
        httpd.server_close()
        operator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
