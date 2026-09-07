#!/usr/bin/env python3
"""Preflight check for a Show-Harness rig: run this before the first rollout.

Verifies, in order: the Python environment, the resolved robot config (site
identity, safety calibration, primitives), the selected VLM backend (key
present, endpoint answering), and — for a Franka rig — that the NUC control
server and the two RealSense cameras are reachable. Read-only: nothing moves.

    python scripts/check_setup.py --robot-config configs/robot_franka.yaml

Exit code 0 = ready (warnings allowed), 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TTY = sys.stdout.isatty()


def _tag(kind: str) -> str:
    colors = {"OK": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}
    text = f"[{kind:^4}]"
    return f"{colors[kind]}{text}\033[0m" if _TTY else text


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"{_tag('OK')} {label}" + (f" — {detail}" if detail else ""))

    def warn(self, label: str, detail: str, hint: str = "") -> None:
        self.warnings += 1
        print(f"{_tag('WARN')} {label} — {detail}")
        if hint:
            print(f"       fix: {hint}")

    def fail(self, label: str, detail: str, hint: str = "") -> None:
        self.failures += 1
        print(f"{_tag('FAIL')} {label} — {detail}")
        if hint:
            print(f"       fix: {hint}")


def check_environment(r: Report) -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        r.ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        r.fail("python", f"{v.major}.{v.minor} found, 3.10+ required")
    for mod, why in [
        ("numpy", "arrays"), ("cv2", "image ops"), ("scipy", "SO(3) math"),
        ("yaml", "configs"), ("PIL", "images"), ("requests", "VLM client"),
        ("imageio", "videos"), ("zerorpc", "Franka transport"),
    ]:
        try:
            importlib.import_module(mod)
            r.ok(f"import {mod}")
        except Exception as exc:  # noqa: BLE001
            r.fail(f"import {mod}", f"{exc} ({why})",
                   "pip install -r requirements/requirements.txt")
    try:
        importlib.import_module("pyrealsense2")
        r.ok("import pyrealsense2")
    except Exception:  # noqa: BLE001
        r.warn("import pyrealsense2", "not installed (needed for real Franka cameras)",
               "pip install -r requirements/requirements-real.txt")


def check_config(r: Report, cfg_path: str):
    from core.config import load_yaml

    try:
        cfg = load_yaml(cfg_path)
    except FileNotFoundError as exc:
        r.fail("robot config", str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        r.fail("robot config", f"{cfg_path}: {exc}")
        return None
    if "hardware" not in cfg and "robot" not in cfg and "arms" not in cfg:
        r.ok("robot config", f"{cfg_path} (simulator profile — rig checks skipped)")
        cfg["_sim_profile"] = True
        return cfg
    hardware = str(cfg.get("hardware", "franka")).lower()
    r.ok("robot config", f"{cfg_path} (hardware: {hardware})")

    if hardware == "piper":
        _check_piper_identity(r, cfg)
    else:
        _check_franka_identity(r, cfg)

    args = SimpleNamespace(primitives_config=None)
    from core.launch import resolve_primitives_path
    prim = resolve_primitives_path(args, cfg, hardware)
    if Path(prim).is_file():
        r.ok("primitives", prim)
    else:
        r.fail("primitives", f"{prim} not found")
    return cfg


def _check_franka_identity(r: Report, cfg: dict) -> None:
    rb = cfg.get("robot", {}) or {}
    if rb.get("nuc_ip"):
        r.ok("robot.nuc_ip", str(rb["nuc_ip"]))
    else:
        r.fail("robot.nuc_ip", "not set",
               "cp configs/site/franka.yaml.example configs/site/franka.yaml and fill it in")
    for key in ("external_camera_serial", "wrist_camera_serial"):
        if rb.get(key):
            r.ok(f"robot.{key}", str(rb[key]))
        else:
            r.warn(f"robot.{key}", "empty", "rs-enumerate-devices | grep Serial")
    if cfg.get("enable_z_floor", True):
        name = str(cfg.get("z_floor_name", "default"))
        floor = (cfg.get("z_floors") or {}).get(name, cfg.get("z_floor_m"))
        if floor is None:
            r.fail("z floor", f"enabled but no value for {name!r}",
                   "bash scripts/franka/capture_z_floor.sh --name default --write")
        elif float(floor) <= 0.0:
            r.fail("z floor", f"{name} = {floor} (placeholder — MV_DOWN would reach the table)",
                   "recapture on YOUR table: scripts/franka/capture_z_floor.sh")
        else:
            r.ok("z floor", f"{name} = {float(floor):.3f} m")
    else:
        r.warn("z floor", "disabled — not recommended on hardware")


def _check_piper_identity(r: Report, cfg: dict) -> None:
    try:
        from core.piper.config import SIDES, arm_config
        for side in SIDES:
            view = arm_config(cfg, side)
            floor = view.get("z_floor_m")
            if floor is None or float(floor) <= 0.0:
                r.fail(f"{side} z floor", f"{floor!r} (placeholder)",
                       "capture with scripts/piper/capture_z_floor.sh --arm "
                       f"{side} --write --robot-config configs/site/piper_arms.yaml")
            else:
                r.ok(f"{side} z floor", f"{float(floor):.3f} m")
            if not view.get("begin_joints"):
                r.warn(f"{side} begin pose", "not captured",
                       f"scripts/piper/go_begin.py --arm {side} --capture --write")
    except Exception as exc:  # noqa: BLE001
        r.fail("arms calibration", str(exc),
               "cp configs/site/piper_arms.yaml.example configs/site/piper_arms.yaml and calibrate")
    if os.environ.get("ROS_MASTER_URI"):
        r.ok("ROS_MASTER_URI", os.environ["ROS_MASTER_URI"])
    else:
        r.warn("ROS_MASTER_URI", "not set in this shell",
               "source the Piper ROS workspace, then scripts/piper/run_can.sh / run_cameras.sh / run_arm.sh")


def check_vlm(r: Report, cfg: dict, backend: str | None, live: bool) -> None:
    from core.config import load_secrets_env, resolve_vlm_config

    if not (ROOT / "configs" / "secrets.env").is_file():
        r.warn("configs/secrets.env", "missing (hosted backends need it)",
               "cp configs/secrets.env.example configs/secrets.env")
    load_secrets_env()
    try:
        vlm = resolve_vlm_config(cfg, backend=backend)
    except Exception as exc:  # noqa: BLE001
        r.fail("vlm backend", str(exc))
        return
    provider = str(vlm.get("provider", "vllm")).lower()
    r.ok("vlm backend", f"{vlm['backend']} (provider {provider}, model {vlm['model']})")

    env_name = vlm.get("api_key_env")
    key = str(vlm.get("api_key") or "")
    if env_name and (not key or key == "EMPTY"):
        r.fail("api key", f"${env_name} is not set",
               f"add {env_name}=... to configs/secrets.env")
        return
    if env_name:
        r.ok("api key", f"${env_name} is set ({len(key)} chars)")

    if not live:
        r.warn("vlm reachability", "skipped (--no-vlm)")
        return
    try:
        from core.vlm.vlm_client import VLMClient
        client = VLMClient(
            base_url=vlm["base_url"], model=vlm["model"], api_key=key or "EMPTY",
            timeout_s=20, max_tokens=8, temperature=0.0,
            provider=provider, api_dialect=vlm.get("api_dialect"), max_retries=0,
        )
        if provider == "vllm":
            client.health_check(wait_s=0)
            r.ok("vlm reachability", f"{vlm['base_url']}/models answered")
        else:
            import numpy as np
            resp = client.complete_text(
                "Reply with the single word: ready", np.zeros((8, 8, 3), dtype=np.uint8),
                max_tokens=8,
            )
            text = (resp.raw_text or "").strip().replace("\n", " ")[:40]
            r.ok("vlm reachability", f"live reply: {text!r}")
    except Exception as exc:  # noqa: BLE001
        r.fail("vlm reachability", str(exc)[:200],
               "local vLLM: bash scripts/serve_vlm.sh — hosted: check the key and network")


def check_hardware(r: Report, cfg: dict) -> None:
    hardware = str(cfg.get("hardware", "franka")).lower()
    if hardware != "franka":
        return  # piper bring-up is covered by the ROS checks above
    rb = cfg.get("robot", {}) or {}
    ip, port = rb.get("nuc_ip"), int(rb.get("nuc_port", 4242))
    if not ip:
        return  # already failed in the identity check
    try:
        import zerorpc
        c = zerorpc.Client(heartbeat=None, timeout=4)
        c.connect(f"tcp://{ip}:{port}")
        pose = c.get_ee_pose()
        c.close()
        r.ok("franka NUC", f"{ip}:{port} answered; EEF z = {float(pose[2]):.3f} m")
    except Exception as exc:  # noqa: BLE001
        r.fail("franka NUC", f"{ip}:{port} not answering ({type(exc).__name__})",
               "start the Polymetis franka_server on the NUC and check the network")
    try:
        import pyrealsense2 as rs
        found = {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}
        for key in ("external_camera_serial", "wrist_camera_serial"):
            want = str(rb.get(key) or "")
            if not want:
                continue
            if want in found:
                r.ok(f"camera {key.split('_')[0]}", want)
            else:
                r.fail(f"camera {key.split('_')[0]}", f"{want} not connected "
                       f"(found: {sorted(found) or 'none'})",
                       "check the USB connection / serial in configs/site/franka.yaml")
    except Exception as exc:  # noqa: BLE001
        r.warn("cameras", f"could not enumerate ({exc})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_franka.yaml"))
    p.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"),
                   help="vlm_backends profile to check (default: the config's vlm_backend)")
    p.add_argument("--no-vlm", action="store_true", help="skip the live VLM call")
    p.add_argument("--no-hardware", action="store_true", help="skip robot/camera probes")
    args = p.parse_args(argv)

    r = Report()
    print("— environment —")
    check_environment(r)
    print("— configuration —")
    cfg = check_config(r, args.robot_config)
    if cfg is not None:
        print("— vlm —")
        check_vlm(r, cfg, args.vlm_backend, live=not args.no_vlm)
        if not args.no_hardware and not cfg.get("_sim_profile"):
            print("— hardware —")
            check_hardware(r, cfg)

    print()
    if r.failures:
        print(f"{_tag('FAIL')} {r.failures} failure(s), {r.warnings} warning(s) — fix the items above, then re-run.")
        return 1
    if r.warnings:
        print(f"{_tag('WARN')} ready with {r.warnings} warning(s).")
    else:
        print(f"{_tag('OK')} all checks passed — you are ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
