"""Real-robot harness wiring, shared by every real entry point.

This module owns the run-time assembly of the Show-Harness harness on a real arm:
config layering (code DEFAULTS under the robot yaml under CLI overrides), the
VLM client, the hardware session, the embodiment interpreter (controller), the
plugin set, and the runner. ``scripts/run_real.py`` and the other ``run_real_*.py``
entry points are thin CLIs over these functions.

The per-key helpers near the bottom read one plugin hyperparameter each from
the resolved config, giving every knob exactly one defaulting site.
"""


from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from interpreters.franka_atomic_controller import (
    EMPTY_GRASP_WIDTH_M,
    TABLE_CONTACT_Z_M,
    FrankaAtomicController,
)
from interpreters.piper_atomic_controller import PiperAtomicController
from interpreters.real_atomic_controller import RealAtomicController
from core.config import deep_merge, make_api_key_refresher, resolve_vlm_config
from core.record.episode_logger import EpisodeLogger
from core.franka.franka_session import FrankaSession, FrankaSessionConfig
from core.piper.config import SIDES
from core.piper.dual_session import DualPiperSession, DualPiperSessionConfig
from core.piper.piper_session import PiperSession, PiperSessionConfig
from core.piper.poses import go_begin, go_begin_dual
from core.runners.real import RealEpisodeRunner
from core.v0_types import V0Config
from core.agent.stage_control import Controller, StageControlSuite
from plugins.action_ablation import ActionAblationPlugin
from plugins.action_chunk import ActionChunkPlugin
from plugins.affordance import AffordancePlugin
from plugins.config import PluginsConfig
from plugins.coords import CoordsPlugin
from plugins.deepplan import DeepPlanPlugin
from plugins.ego import EgoPlugin
from plugins.mcq import McqPlugin
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.rotation import RotationPlugin
from plugins.smooth import SmoothPlugin
from plugins.subgoal import SubgoalPlanner, SubgoalPlannerAgent, load_prompt
from plugins.variable_step import VariableStepPlugin
from plugins.wrist_frame import WristFramePlugin
from core.vlm.vlm_client import VLMClient
from core.vlm.roles import ControllerAgent

ROOT = Path(__file__).resolve().parents[1]

# A real-robot task is standalone (no task suite). The EpisodeLogger still groups
# runs under a ``task_<id>`` directory to keep one rollout layout, so all real
# rollouts live under ``task_0``.
REAL_TASK_ID = 0

# Code-side defaults for every non-core setting, so configs/robot_franka.yaml can stay
# small and only carry the parameters a user actually tunes (task, gripper color,
# step distance, grasp width, robot/camera identity, VLM endpoint). Anything the
# YAML omits falls back here; anything it sets overrides these.



DEFAULTS: dict[str, Any] = {
    "gripper_color": "black",
    "fine_step_m": 0.02,  # fine per-step Cartesian move distance (meters)
    "up_step_m": 0.08,  # MV_UP lift/retreat distance (meters)
    "max_steps": 80,
    # Physical grasp verification thresholds. A closed width at/below empty_width_m
    # means empty close; a width at/above open_width_m is still open/not settled.
    "empty_width_m": EMPTY_GRASP_WIDTH_M,
    "open_width_m": 0.06,
    "loop_period_s": 0.0,
    "log_dir": "rollouts/real",
    "camera_resolution": 256,
    "use_wrist_image": True,
    # Z safety floor is ON by default at the calibrated table-contact height, so
    # MV_DOWN can never drive the arm into the table. Override the height with
    # --z-floor-m / z_floor_m, or disable it entirely with --no-z-floor /
    # enable_z_floor: false (not recommended on hardware).
    "enable_z_floor": True,
    "z_floor_m": TABLE_CONTACT_Z_M,
    "robot": {
        # nuc_ip is deliberately absent: rig identity comes from configs/site/
        # (session_config raises with guidance when it is missing).
        "nuc_port": 4242,
        "use_mock_robot": False,
        "use_mock_cameras": False,
        "start_impedance": True,
        "external_camera_serial": None,
        "wrist_camera_serial": None,
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        # RealSense frames occasionally stall for >1s on hardware. Retry and restart
        # the pipeline once so a transient camera hiccup does not abort the rollout.
        "camera_read_timeout_ms": 3000,
        "camera_read_retries": 2,
        "camera_read_retry_delay_s": 0.05,
        "camera_restart_on_read_failure": True,
        "settle_steps": 4,
        "settle_dt_s": 0.05,
        # The real Franka gripper actuates asynchronously (~1 s) and its width sensor
        # lags; the controller blocks up to gripper_settle_s (but >= gripper_min_settle_s)
        # for the fingers to actually move and then stop before reading the width, so the
        # empty-grasp check sees the true closed width IN THE SAME STEP and reopens
        # immediately. Generous because a step already takes tens of seconds; raise it if
        # the empty-grasp reopen still lands a step late on your gripper.
        "gripper_settle_s": 2.5,
        "gripper_min_settle_s": 0.5,
    },
    "v0": {
        "max_subgoal_steps": 40,
        "max_replans": 1,
        "video_fps": 2.0,
    },
    # Pluggable capabilities, each independently toggleable with a boolean only. Tool
    # parameters live in their owning config sections (for example, Safety rules).
    # A disabled tool must still let the build run:
    #   subgoal off        -> single whole-task stage (no planner VLM call)
    #   proprioception off -> no proprio prompt context (Z-floor SAFETY is separate)
    #   recovery off       -> no measured-width release/rollback intervention
    #   coords off         -> controller DIRECTION/REMARK prompt unchanged
    #   mcq off            -> default atomic-token output (byte-identical to today)
    #   auto_release off   -> MVTOKEN runner never reopens an empty closed gripper
    #                         (used only by scripts/run_real_mvtoken.py; real_runner ignores it)
    #   variable_step off  -> fixed step_m for every move (no coarse step when high/MV_UP)
    #   action_chunk off   -> one move per VLM call (no open-loop repeats when far)
    #   mem_text off       -> no move-history line or oscillation/empty-grasp rules in the prompt
    #   smooth off         -> setpoint steps straight to target (no min-jerk ramp)
    #   deepplan off       -> planner emits no <REASON> pivot; runner never resolves a branch
    #                         (linear plan, byte-identical to today). On -> conditional tasks.
    #   affordance off     -> no contact-point grounding, no dot on the AgentView, and the
    #                         AFFORD field keeps the planner's text (byte-identical to today)
    "plugins": {
        "subgoal": True,
        "proprioception": True,
        "recovery": True,
        "coords": False,
        "mcq": False,
        "auto_release": True,
        "variable_step": False,
        "action_chunk": False,
        "mem_text": True,
        "smooth": False,
        "deepplan": False,
        #   rotation off       -> ROTATE_CW/CCW never offered to the VLM and no yaw
        #                         compensation; gripper keeps its captured orientation (as today)
        "rotation": False,
        "affordance": False,
        #   video_ref off      -> no demo-brief extraction; the planner plans on its own
        #   dagger off         -> no key handler installed; the model is never overridden
        "video_ref": False,
        "dagger": False,
    },
    "vlm_backend": "gemma",
    "vlm": {
        "base_url": "http://localhost:8000/v1",
        "timeout_s": 120,
        "startup_wait_s": 600,
        "startup_poll_s": 5,
        "temperature": 0.0,
        "api_key": "EMPTY",
    },
}



def auto_home_piper(cfg: dict[str, Any], session: Any) -> None:
    """Send the driven Piper arm back to its BEGIN pose at the end of a rollout.

    ``cfg`` is the FLATTENED single-arm view (core.piper.config.arm_config), so
    ``begin_joints`` is already the selected arm's. Best-effort: never masks the result.
    """
    try:
        from core.piper.poses import go_begin

        go_begin(
            session.robot,
            cfg.get("begin_joints"),
            time_to_go=float(cfg.get("begin_time_s", 3.0)),
            label="begin",
            open_gripper=True,  # a reset always leaves the hand empty
        )
    except Exception as exc:  # noqa: BLE001 - the reset must not mask the rollout result
        print(f"[run-real] go-begin after the rollout failed: {exc}")



def move_to_begin_on_init(cfg: dict[str, Any], session: Any) -> None:
    """Move the driven Piper arm to its BEGIN pose on init (best-effort; no-op if
    disabled or not configured). Called before the controller's first sync, so the run
    starts from a known pose and the setpoint is captured there."""
    if not cfg.get("move_to_begin_on_init", True):
        return
    try:
        from core.piper.poses import go_begin

        go_begin(
            session.robot,
            cfg.get("begin_joints"),
            time_to_go=float(cfg.get("begin_time_s", 3.0)),
            label="begin",
            open_gripper=True,  # the run starts from a known pose AND an empty hand
        )
    except Exception as exc:  # noqa: BLE001 - a begin move must not abort the run setup
        print(f"[run-real] move-to-begin on init failed: {exc}")



def resolve_franka_begin(cfg: dict[str, Any]) -> None:
    """Resolve the Franka ``begin_pose`` name into ``begin_joints`` (in place).

    Named poses live under a flat ``poses:`` block in robot_franka.yaml. The
    ``default`` pose IS the ``franka_server/go_home_client.py`` home (the external
    reset command doubles as home and rest), so the base pose never drifts from the
    operator's reset; other presets build on it (e.g. wrist rotations). Explicit
    ``begin_joints`` in the config win over the named pose."""
    if cfg.get("begin_joints"):
        return
    poses = cfg.get("poses") or {}
    name = str(cfg.get("begin_pose", "default"))
    if not poses:
        if cfg.get("move_to_begin_on_init", False):
            raise ValueError(
                "move_to_begin_on_init/--begin-pose need a poses: block in "
                "robot_franka.yaml (poses.<name>: [7 joint radians])."
            )
        return
    if name not in poses:
        raise ValueError(
            f"Unknown begin_pose {name!r}; available poses: {sorted(poses)}"
        )
    cfg["begin_joints"] = [float(v) for v in poses[name]]



def move_to_begin_franka(cfg: dict[str, Any], session: Any) -> None:
    """Move the Franka to its named BEGIN pose on init (best-effort; no-op unless
    ``move_to_begin_on_init`` is set -- the default workflow keeps the external
    ``go_home_client.py`` reset). Called before the controller's first sync, so the
    run starts from a known pose and the setpoint is captured there. The joint move
    preempts the Cartesian-impedance controller; it is restarted afterwards (the
    same sequence as collect_rollouts' homing hook)."""
    if not cfg.get("move_to_begin_on_init", False):
        return
    joints = cfg.get("begin_joints")
    if not joints:
        return
    try:
        print(f"  homing the arm to BEGIN ({cfg.get('begin_pose', 'default')}) ...")
        session.robot.move_to_joint_positions(
            np.asarray(joints, dtype=float), float(cfg.get("begin_time_s", 3.0))
        )
        if session.config.start_impedance:
            session.start_impedance()
    except Exception as exc:  # noqa: BLE001 - a begin move must not abort the run setup
        print(f"[run-real] move-to-begin on init failed: {exc}")



def run_variant(vlm_cfg: dict[str, Any]) -> str:
    """Run-grouping folder name: backend + chain-of-thought mode (e.g. ``Gemma-CoT``).
    Matches run.py so real + sim rollouts share the same directory convention."""
    backend = str(vlm_cfg.get("backend") or "vlm").strip() or "vlm"
    label = backend[:1].upper() + backend[1:]
    cot = "CoT" if vlm_cfg.get("reasoning_cot") else "NCoT"
    return f"{label}-{cot}"



def build_config(args: argparse.Namespace, robot_cfg: dict[str, Any]) -> dict[str, Any]:
    # Start from the code-side DEFAULTS so robot_franka.yaml can stay minimal: the YAML
    # only needs the handful of core params, everything else falls back here.
    cfg = deep_merge(DEFAULTS, robot_cfg)
    overrides: dict[str, Any] = {}
    for arg_name, cfg_name in [
        ("task", "task"),
        ("gripper_color", "gripper_color"),
        ("fine_step_m", "fine_step_m"),
        ("max_steps", "max_steps"),
        ("loop_period_s", "loop_period_s"),
        ("log_dir", "log_dir"),
    ]:
        # getattr default None: build_config is shared by scripts/run_real.py and scripts/run_real_mvtoken.py,
        # whose parsers don't define an identical arg set, so a missing arg is "no override".
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_name] = value
    cfg = deep_merge(cfg, overrides)
    if "task" not in cfg or not str(cfg["task"]).strip():
        raise ValueError("No task defined; set `task` in robot_franka.yaml or pass --task.")

    if args.vlm_url and args.vlm_url.lower() == "mock":
        raise ValueError("Mock VLM mode is not supported; provide a real vLLM URL.")
    vlm = resolve_vlm_config(cfg, backend=args.vlm_backend)
    if args.vlm_url:
        # --vlm-url / VLM_URL / VLLM_BASE_URL target the LOCAL vLLM (the tunnel). A hosted
        # backend (openai/gemini) carries its own base_url in its profile; overriding that
        # would send its model id to the wrong server (e.g. gemini-flash-latest -> the
        # local Gemma vLLM -> HTTP 404), so the override is scoped to the vllm provider.
        if vlm.get("provider", "vllm") == "vllm":
            vlm["base_url"] = args.vlm_url
        else:
            print(
                f"[run-real] ignoring --vlm-url/VLM_URL for hosted backend "
                f"{vlm['backend']!r}; using its own endpoint {vlm['base_url']}."
            )
    if args.model:
        vlm["model"] = args.model
    cfg["vlm"] = vlm
    cfg["vlm_backend"] = vlm["backend"]
    return cfg



def make_vlm_client(args: argparse.Namespace, cfg: dict[str, Any]):
    vlm_cfg = cfg["vlm"]
    # resolve_vlm_config already resolved the key from api_key_env (configs/secrets.env),
    # so no secret lives in the git-tracked config.
    return VLMClient(
        base_url=vlm_cfg["base_url"],
        model=vlm_cfg["model"],
        api_key=vlm_cfg.get("api_key", "EMPTY"),
        timeout_s=float(vlm_cfg.get("timeout_s", 120)),
        max_tokens=int(vlm_cfg.get("max_tokens", 256)),
        temperature=float(vlm_cfg.get("temperature", 0.0)),
        chat_template_kwargs=vlm_cfg.get("chat_template_kwargs", {}),
        cot_max_tokens=vlm_cfg.get("cot_max_tokens"),
        reasoning_directive=vlm_cfg.get("reasoning_directive"),
        provider=vlm_cfg.get("provider", "vllm"),
        api_dialect=vlm_cfg.get("api_dialect"),
        reasoning_effort=vlm_cfg.get("reasoning_effort"),
        max_retries=vlm_cfg.get("max_retries"),
        retry_base_delay_s=vlm_cfg.get("retry_base_delay_s"),
        retry_max_delay_s=vlm_cfg.get("retry_max_delay_s"),
        # On a 401 mid-rollout, re-read secrets.env (kept fresh by the stay-open
        # its refresher) and retry instead of crashing the episode.
        api_key_refresh=make_api_key_refresher(vlm_cfg),
    )


SUPPORTED_HARDWARE = ("franka", "piper")



def resolve_hardware(args: argparse.Namespace, robot_cfg: dict[str, Any]) -> str:
    """Select the hardware target from the robot config's ``hardware`` key.

    Selection is by which ``--robot-config`` you pass (robot_franka.yaml -> franka,
    robot_piper.yaml -> piper), so hardware-specific constants (Z-floor, gripper
    widths, camera identity) always travel with the matching config -- there is no
    separate flag that could contradict it.
    """
    hardware = str(robot_cfg.get("hardware", "franka")).strip().lower()
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(
            f"Unknown hardware {hardware!r} in {Path(args.robot_config).name}; "
            f"choices: {SUPPORTED_HARDWARE}."
        )
    return hardware



def resolve_primitives_path(
    args: argparse.Namespace, robot_cfg: dict[str, Any], hardware: str
) -> str:
    """--primitives-config (explicit) > robot config's primitives_config > the
    hardware default (configs/primitives_franka.yaml for franka, primitives_piper.yaml for
    piper). A relative path in the config is resolved against the repo root."""
    if args.primitives_config:
        return args.primitives_config
    from_cfg = robot_cfg.get("primitives_config")
    if from_cfg:
        p = Path(from_cfg)
        return str(p if p.is_absolute() else ROOT / p)
    default_name = "primitives_piper.yaml" if hardware == "piper" else "primitives_franka.yaml"
    return str(ROOT / "configs" / default_name)



def make_session(cfg: dict[str, Any], args: argparse.Namespace, hardware: str) -> Any:
    if hardware == "piper":
        return PiperSession(piper_session_config(cfg, args))
    return FrankaSession(session_config(cfg, args))



def piper_session_config(
    cfg: dict[str, Any], args: argparse.Namespace
) -> PiperSessionConfig:
    rb = cfg.get("robot", {}) or {}
    return PiperSessionConfig(
        arm=str(rb.get("arm", "left")),
        use_mock_robot=bool(args.mock_robot or rb.get("use_mock_robot", False)),
        open_width_m=float(rb.get("open_width_m", 0.07)),
        connect_cameras=True,
        use_mock_cameras=bool(args.mock_cameras or rb.get("use_mock_cameras", False)),
        front_camera_topic=str(rb.get("front_camera_topic", "/camera_f/color/image_raw")),
        wrist_camera_topic=str(rb.get("wrist_camera_topic", "/camera_l/color/image_raw")),
        camera_max_age_s=float(rb.get("camera_max_age_s", 1.0)),
        camera_connect_timeout_s=float(rb.get("camera_connect_timeout_s", 10.0)),
        observation_resolution=int(cfg.get("camera_resolution", 256)),
        verbose=True,
    )



def session_config(cfg: dict[str, Any], args: argparse.Namespace) -> FrankaSessionConfig:
    rb = cfg.get("robot", {}) or {}
    mock = bool(args.mock_robot or rb.get("use_mock_robot", False))
    nuc_ip = args.nuc_ip or rb.get("nuc_ip")
    if not nuc_ip:
        if not mock:
            raise ValueError(
                "robot.nuc_ip is not configured. Rig identity is site-specific: "
                "copy configs/site/franka.yaml.example to configs/site/franka.yaml "
                "and fill in your NUC's address (or pass --nuc-ip)."
            )
        nuc_ip = "127.0.0.1"  # mock robot: never dialed
    nuc_ip = str(nuc_ip)
    nuc_port = int(args.nuc_port if args.nuc_port is not None else rb.get("nuc_port", 4242))
    # High-res renders (hd_res, 0 = off; legacy key affordance_hd_res honored) for
    # the consumers that need legible detail: the subgoal PLANNER and the affordance
    # pointer/verify/track calls. The controller observation pipeline, live stream,
    # and saved video all stay at camera_resolution.
    hd_res = int(cfg.get("hd_res", cfg.get("affordance_hd_res", 640)) or 0)
    return FrankaSessionConfig(
        nuc_ip=nuc_ip,
        nuc_port=nuc_port,
        hd_resolution=hd_res if hd_res > 0 else None,
        use_mock_robot=bool(args.mock_robot or rb.get("use_mock_robot", False)),
        start_impedance=bool(rb.get("start_impedance", True)) and not args.no_impedance,
        connect_cameras=True,
        use_mock_cameras=bool(args.mock_cameras or rb.get("use_mock_cameras", False)),
        external_camera_serial=rb.get("external_camera_serial"),
        wrist_camera_serial=rb.get("wrist_camera_serial"),
        camera_width=int(rb.get("camera_width", 640)),
        camera_height=int(rb.get("camera_height", 480)),
        camera_fps=int(rb.get("camera_fps", 30)),
        camera_read_timeout_ms=int(rb.get("camera_read_timeout_ms", 3000)),
        camera_read_retries=int(rb.get("camera_read_retries", 2)),
        camera_read_retry_delay_s=float(rb.get("camera_read_retry_delay_s", 0.05)),
        camera_restart_on_read_failure=bool(rb.get("camera_restart_on_read_failure", True)),
        observation_resolution=int(cfg.get("camera_resolution", 256)),
        verbose=True,
    )



def resolve_z_floor_name(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Resolve a NAMED Z floor (``z_floors.<name>``) into ``cfg["z_floor_m"]`` in place.

    Named floors let one config carry a calibrated floor per table/task setting (the
    Franka analog of robot_piper.yaml's per-pose z_floor_m variants); capture new ones
    with scripts/franka/capture_z_floor.sh. ``--z-floor-name`` beats the config's
    ``z_floor_name``. The resolved height replaces any plain ``z_floor_m`` (which the
    code-side DEFAULTS always inject, so it is not a signal of user intent); the
    existing precedence above it is untouched: --no-z-floor > --z-floor-m > --z-floor.
    No ``z_floors`` block (e.g. the Piper unified config) -> no-op unless a name was
    explicitly requested."""
    floors = cfg.get("z_floors") or {}
    name = args.z_floor_name or cfg.get("z_floor_name")
    if not name:
        return
    if not floors:
        if args.z_floor_name:
            raise ValueError(
                f"--z-floor-name {name!r} given but the robot config has no z_floors: block."
            )
        return
    if name not in floors:
        raise ValueError(
            f"Unknown z_floor_name {name!r}; available floors: {sorted(floors)}. "
            "Capture one with scripts/franka/capture_z_floor.sh --name <name> --write."
        )
    cfg["z_floor_m"] = float(floors[name])
    cfg["_z_floor_name"] = str(name)  # for the run header



def resolve_z_floor(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Any, bool]:
    """Resolve the Z floor to ``(z_floor_m, capture_at_start)``.

    ``z_floor_m`` is an explicit height (meters) or ``None``; when ``None`` and
    ``capture_at_start`` is True the controller locks the floor at the robot's
    current height on the first sync. When both are falsy the floor is disabled.
    """
    if args.no_z_floor:
        return None, False
    if args.z_floor_m is not None:
        return float(args.z_floor_m), False
    if args.z_floor:  # operator asked to re-capture at the start height
        return None, True
    if not bool(cfg.get("enable_z_floor", False)):
        return None, False
    cfg_floor = cfg.get("z_floor_m")
    if cfg_floor is not None:
        return float(cfg_floor), False
    return None, True  # enabled but no explicit height -> capture at start



def make_controller(
    cfg: dict[str, Any],
    primitives_cfg: dict[str, Any],
    session: Any,
    args: argparse.Namespace,
    hardware: str = "franka",
) -> RealAtomicController:
    rb = cfg.get("robot", {}) or {}
    # Resolve the Z floor. It is ON by default at the calibrated table-contact height
    # so MV_DOWN can never drive the arm into the table. Precedence (highest first):
    #   --no-z-floor   -> off entirely
    #   --z-floor-m X  -> explicit fixed height X
    #   --z-floor      -> capture at the current (start) height instead of the default
    #   config         -> enable_z_floor / z_floor_m (z_floor_m=None under an enabled
    #                     floor means capture-at-start)
    z_floor_m, capture = resolve_z_floor(cfg, args)
    # Piper is stiff MOVE-P position control (no impedance compliance), so a wrong or
    # missing floor is dangerous: capturing at a raised start pose forbids all descent,
    # and no floor lets the arm press into the table. Refuse an autonomous Piper run
    # until the floor is a MEASURED height -- calibrate it once per arm with
    # scripts/piper/capture_z_floor.sh --arm <side> --write. Mock runs are exempt.
    mock_robot = bool(args.mock_robot or rb.get("use_mock_robot", False))
    if hardware == "piper" and capture and not mock_robot:
        raise ValueError(
            "Piper: the Z safety floor would be captured at the start pose, which is "
            "unsafe (a raised start locks out all descent; a low start offers no "
            "protection). Calibrate it first: rest that arm's gripper on the tabletop, "
            "then run  scripts/piper/capture_z_floor.sh --arm <left|right> --write  "
            "(sets arms.<side>.z_floor_m in configs/robot_piper.yaml) -- or pass "
            "--z-floor-m <height> explicitly."
        )
    step_m = cfg.get("fine_step_m")
    plugins = PluginsConfig.from_config(cfg)
    grasp_min_width_m = recovery_empty_width_m(cfg) if plugins.enabled("recovery") else None
    # Variable-step tool: coarse step when high above the table or lifting (MV_UP). The
    # controller compares the EEF height against the table-contact reference; disabled ->
    # the controller always uses the fixed step_m.
    variable_step_plugin = VariableStepPlugin(
        enabled=plugins.enabled("variable_step", default=False),
        coarse_step_m=coarse_step_m(cfg),
        high_above_table_m=high_above_table_m(cfg),
    )
    # Smooth-motion tool: ramp each move's setpoint along a min-jerk profile so the arm
    # accelerates/decelerates gently instead of stepping abruptly to the target.
    smooth_plugin = SmoothPlugin(
        enabled=plugins.enabled("smooth", default=False),
        substeps=smooth_substeps(cfg),
        dt_s=smooth_dt_s(cfg),
        # Never crawl the setpoint below the hardware's command resolution (the Piper's
        # MOVE P is 1 mm-quantized), and chain aligned consecutive moves at cruise speed
        # so a run of steps does not stop-and-start once per token. Blending is a
        # PIPER-specific behavior: the Franka default stays rest-to-rest min-jerk with a
        # settle per move (byte-identical to the pre-blend implementation).
        min_waypoint_m=smooth_min_waypoint_m(cfg),
        blend=bool(cfg.get("smooth_blend", hardware == "piper")),
    )
    # Rotation plugin: adds ROTATE_CW/CCW (a grasp-alignment yaw) and, once yawed, compensates
    # wrist-judged MV_* for the accumulated yaw. When enabled we also raise the controller's
    # per-command yaw step to the tool's (30 deg) and widen the yaw safety clamp to match,
    # else the default 0.2 rad clamp would silently truncate a 0.524 rad step.
    # The Piper runs cartesian-only and never offers rotation: hard-gate it off there
    # so no config can enable it on that path (the Franka path is unaffected). The tool
    # object is still built (disabled) and threaded through the shared kwargs -- a
    # disabled RotationPlugin is a no-op (empty tokens/prompt, identity compensation).
    rotation_plugin = RotationPlugin(
        enabled=plugins.enabled("rotation", default=False) and hardware != "piper",
        yaw_step_rad=math.radians(rotate_step_deg(cfg)),
        compensation_sign=float(cfg.get("rotate_compensation_sign", 1.0)),
        max_accumulated_yaw_rad=math.radians(rotate_max_accum_deg(cfg)),
    )
    rotation_kwargs: dict[str, Any] = {"rotation_plugin": rotation_plugin}
    if rotation_plugin.enabled:
        rotation_kwargs["yaw_step_rad"] = rotation_plugin.yaw_step_rad
        rotation_kwargs["max_rotation_delta_rad"] = rotation_plugin.yaw_step_rad + 1e-3

    # Shared construction kwargs (both controllers derive from RealAtomicController).
    common_kwargs: dict[str, Any] = dict(
        # Motion frame for MV_*: "base" (default; Franka) or "wrist" (rotated to the
        # gripper's heading -- the Piper dual rig, where the arms start yawed +/-45 deg
        # and each wrist view defines that arm's own forward).
        motion_frame=str(cfg.get("motion_frame", "base")),
        step_m=None if step_m is None else float(step_m),
        settle_steps=int(rb.get("settle_steps", 4)),
        settle_dt_s=float(rb.get("settle_dt_s", 0.05)),
        grasp_min_width_m=grasp_min_width_m,
        grasp_open_width_m=recovery_open_width_m(cfg),
        gripper_settle_s=float(rb.get("gripper_settle_s", 1.2)),
        gripper_min_settle_s=float(rb.get("gripper_min_settle_s", 0.5)),
        z_floor_m=None if z_floor_m is None else float(z_floor_m),
        capture_z_floor_on_sync=capture,
        variable_step_plugin=variable_step_plugin,
        table_height_m=table_height_m(cfg, z_floor_m),
        up_step_m=up_step_m(cfg),
        smooth_plugin=smooth_plugin,
        verbose=True,
        **rotation_kwargs,
    )

    if hardware == "piper":
        # Cartesian-only: MV_* are base-frame translations (the primitives unit vectors)
        # and the Piper node's MOVE P does the IK -- no impedance controller to self-heal.
        # The gripper close threshold is Piper-scaled (stroke ~0.07 m, not the Franka
        # 0.08 m); only forwarded when the config overrides the Piper default.
        piper_kwargs = dict(common_kwargs)
        if cfg.get("gripper_close_threshold_m") is not None:
            piper_kwargs["gripper_close_threshold_m"] = float(cfg["gripper_close_threshold_m"])
        # AgileX-specific motion backend: joint_stream (smooth MOVE J streaming via
        # on-board IK; default) or endpose (plain firmware MOVE P per command).
        piper_kwargs["motion_backend"] = str(cfg.get("motion_backend", "joint_stream"))
        if cfg.get("joint_stream_hz") is not None:
            piper_kwargs["joint_stream_hz"] = float(cfg["joint_stream_hz"])
        if cfg.get("ori_flex_deg") is not None:
            piper_kwargs["ori_flex_rad"] = math.radians(float(cfg["ori_flex_deg"]))
        return PiperAtomicController.from_primitives_config(
            session.robot, primitives_cfg, **piper_kwargs
        )

    return FrankaAtomicController.from_primitives_config(
        session.robot,
        primitives_cfg,
        # Self-heal a lost controller (server-side reflex / termination): restart impedance
        # and retry the setpoint instead of crashing the rollout with "no controller running".
        ensure_controller=(session.start_impedance if session.config.start_impedance else None),
        **common_kwargs,
    )



def recovery_empty_width_m(cfg: dict[str, Any]) -> float:
    return float(cfg.get("empty_width_m", cfg.get("grasp_min_width_m", EMPTY_GRASP_WIDTH_M)))



def recovery_open_width_m(cfg: dict[str, Any]) -> float:
    return float(cfg.get("open_width_m", cfg.get("grasp_open_width_m", 0.06)))



def high_above_table_m(cfg: dict[str, Any]) -> float:
    """Threshold X (m): above this height the controller is told to descend first, and
    (if variable_step is on) uses the coarse step. Shared by proprioception + variable_step."""
    return float(cfg.get("high_above_table_m", 0.10))



def coarse_step_m(cfg: dict[str, Any]) -> float:
    """Coarse per-command translation (m) used by variable_step when high or lifting."""
    return float(cfg.get("coarse_step_m", 0.05))



def up_step_m(cfg: dict[str, Any]) -> Optional[float]:
    """Dedicated MV_UP lift/retreat distance (m); None -> MV_UP uses the normal step logic."""
    value = cfg.get("up_step_m")
    return None if value is None else float(value)



def action_chunk_step_num(cfg: dict[str, Any]) -> int:
    """Moves committed per VLM call by action_chunk while the TARGET is far (default 3)."""
    return int(cfg.get("action_chunk_step_num", 3))



def mem_text_len(cfg: dict[str, Any]) -> int:
    """How many recent moves the mem_text 'Recent moves' line shows / the runner keeps."""
    return int(cfg.get("mem_text_len", 3))



def table_height_m(cfg: dict[str, Any], z_floor_m: Any) -> float:
    """Table-contact reference height (m): the Z-floor when set, else the calibrated constant."""
    return float(z_floor_m) if z_floor_m is not None else TABLE_CONTACT_Z_M



def smooth_substeps(cfg: dict[str, Any]) -> int:
    """Number of min-jerk waypoints per move when the smooth plugin is on."""
    return int(cfg.get("smooth_substeps", 20))



def smooth_dt_s(cfg: dict[str, Any]) -> float:
    """Delay (s) between smooth-ramp waypoints (substeps * dt = total move duration)."""
    return float(cfg.get("smooth_dt_s", 0.05))



def smooth_min_waypoint_m(cfg: dict[str, Any]) -> float:
    """Minimum setpoint advance per smooth waypoint (m); 0 = no floor.

    Guards against commanding sub-resolution increments the arm cannot act on (the
    Piper's EndPoseCtrl MOVE P is quantized to 1 mm), which is what makes the ends of a
    min-jerk ramp stutter. The move duration is preserved -- only the command density
    drops.
    """
    return float(cfg.get("smooth_min_waypoint_m", 0.0))



def rotate_step_deg(cfg: dict[str, Any]) -> float:
    """Per-command gripper yaw for the rotation plugin's ROTATE_CW/CCW (degrees, default 30)."""
    return float(cfg.get("rotate_step_deg", 30.0))



def rotate_max_accum_deg(cfg: dict[str, Any]) -> float:
    """Soft cap on total yaw from the reference orientation (degrees), a joint-travel guard."""
    return float(cfg.get("rotate_max_accum_deg", 150.0))



def make_runner(
    cfg: dict[str, Any],
    prompts_cfg: dict[str, Any],
    client: Any,
    session: Any,
    controller: RealAtomicController,
    logger: EpisodeLogger,
    debug: bool,
    viewer: Any = None,
    video_ref_plugin: Any = None,
    dagger_plugin: Any = None,
) -> RealEpisodeRunner:
    v0_config = V0Config.from_dict(cfg.get("v0", {}))
    common_context = prompts_cfg["common_context"]
    plugins = PluginsConfig.from_config(cfg)
    # DeepPlan: when enabled, the planner prompt gains the <REASON>-pivot rules and the
    # runner resolves a reached pivot on the live scene. Disabled -> None: no augmentation
    # and an unreachable pivot path, so the planner build below is byte-identical to today.
    dp_enabled = plugins.enabled("deepplan", default=False)
    deepplan_plugin = (
        DeepPlanPlugin(enabled=True, client=client, common_context=common_context)
        if dp_enabled
        else None
    )
    # Subgoal tool: build the planner only when enabled; otherwise the runner falls back to a
    # single whole-task stage (planner=None). When DeepPlan is on, the REASON-pivot rules are
    # appended to the planner prompt via the existing prompt_template override (the file on
    # disk is never edited); off -> prompt_template=None -> the untouched co-located prompt.
    if plugins.enabled("subgoal"):
        planner_template = None
        if dp_enabled:
            planner_template = (
                load_prompt() + "\n\n" + deepplan_plugin.render_planner_addendum()
            )
        planner = SubgoalPlanner(
            SubgoalPlannerAgent(
                client=client,
                common_context=common_context,
                prompt_template=planner_template,
                # The reference-demo brief rides on the agent, so the initial plan
                # AND every replan replicate the same demonstration ("" when off).
                video_ref_block=(
                    video_ref_plugin.render_prompt() if video_ref_plugin is not None else ""
                ),
                # Long plans (many pieces x 6 stages) on pretty-printing backends
                # (gemini) overflow small budgets and silently degrade the plan.
                max_tokens=int(cfg.get("planner_max_tokens", 4096)),
            )
        )
    else:
        planner = None
    vlm_cfg = cfg.get("vlm", {})
    cot_mode = bool(vlm_cfg.get("reasoning_cot"))
    backend_prompt = prompts_cfg.get(f"controller_{vlm_cfg.get('backend')}_prompt") if cot_mode else None
    controller_prompt = backend_prompt or prompts_cfg["controller_prompt"]
    controller_prompt = CoordsPlugin(plugins.enabled("coords", default=False)).apply(controller_prompt)
    # Optional wrist motion frame (motion_frame: wrist): MV_* follow the gripper's
    # heading, so the front-view direction lines are rewritten to heading-relative
    # judgments. Default base -> untouched (the AgentView-centric convention).
    wrist_frame = str(cfg.get("motion_frame", "base")).strip().lower() == "wrist"
    if wrist_frame and plugins.enabled("coords", default=False):
        print(
            "[run-real] WARNING: plugins.coords writes base-axis directions, but "
            "motion_frame: wrist executes MV_* along the gripper heading -- the "
            "coords scheme will not match the executed motion."
        )
    controller_prompt = WristFramePlugin(wrist_frame).apply(controller_prompt)
    # Egocentric scene camera (e.g. the Piper left-arm rig): the FWD/BACK <-> image
    # depth mapping is inverted vs the Franka convention baked into controller.txt.
    # Driven by the hardware `is_ego` flag; a non-ego rig leaves the prompt untouched.
    controller_prompt = EgoPlugin(bool(cfg.get("is_ego", False))).apply(controller_prompt)
    # Action-type ablation (action_ablation_mode: off|bare|letters|letters_blind):
    # the no-explanation settings replace the DIRECTION section at template level;
    # the letters modes also swap the answer alphabet to opaque ACT_* symbols
    # (agent-side; blind adds the two-frame review + self-written table).
    action_ablation_plugin = ActionAblationPlugin(
        mode=str(cfg.get("action_ablation_mode", "off") or "off"),
        include_rotate=bool(getattr(controller.rotation_plugin, "enabled", False)),
    )
    controller_prompt = action_ablation_plugin.apply(controller_prompt)
    if action_ablation_plugin.enabled:
        print(f"[run-real] action-type ablation: {action_ablation_plugin.mode}")
        if action_ablation_plugin.answer_protocol and plugins.enabled("mcq", default=False):
            print("[run-real] note: plugins.mcq is ignored while the ablation owns the answer alphabet.")
    gripper_color = str(cfg.get("gripper_color", "black"))
    recovery_plugin = RecoveryPlugin(
        enabled=plugins.enabled("recovery"),
        empty_width_m=recovery_empty_width_m(cfg),
        open_width_m=recovery_open_width_m(cfg),
    )
    # Controller plugins. Proprioception needs the table height (the Z-floor when set, else
    # the calibrated table-contact constant); the Z-floor SAFETY limit is enforced inside
    # the controller regardless of this tool.
    table_height_m = controller.z_floor_m if controller.z_floor_m is not None else TABLE_CONTACT_Z_M
    # action_chunk (runner-level decision cadence): repeat a move step_num times while the
    # TARGET is far. Shared with the controller agent so the WRIST marker is rendered/parsed
    # whenever this OR variable_step is enabled.
    action_chunk_plugin = ActionChunkPlugin(
        enabled=plugins.enabled("action_chunk", default=False),
        step_num=action_chunk_step_num(cfg),
    )
    # mem_text owns the "Recent moves" line; its length (max_recent) is shared with the
    # runner, which retains exactly that many moves.
    mem_text_plugin = MemTextPlugin(plugins.enabled("mem_text"), max_recent=mem_text_len(cfg))
    # Affordance dots: the runner grounds each stage's contact point and premarks it on
    # the AgentView; the controller agent rewrites AFFORD to bind that dot.
    affordance_plugin = AffordancePlugin(
        enabled=plugins.enabled("affordance", default=False),
        client=client,
        verify_rounds=int(cfg.get("affordance_verify_rounds", 1)),
        # The single-arm prompt names the scene image "AgentView"; describe it to the
        # pointer as what it physically is on this rig (an external upper camera).
        view_desc="the AgentView: an external camera's upper view of the robot workspace",
        # affordance_wrist_track: false -> front-only dots (no wrist VLM calls).
        wrist_tracking=bool(cfg.get("affordance_wrist_track", True)),
    )
    vstep = controller.variable_step_plugin
    if action_chunk_plugin.enabled and vstep is not None and vstep.enabled:
        # Both far-strategies stack: while the TARGET is far, each VLM decision commits
        # step_num coarse moves open-loop (no re-observation). Surface the resulting reach.
        print(
            f"[run-real] action_chunk + variable_step both on: when the TARGET is far, each "
            f"decision commits {action_chunk_plugin.step_num} x {vstep.coarse_step_m:.3f} m = up "
            f"to {action_chunk_plugin.step_num * vstep.coarse_step_m:.3f} m open-loop before re-checking."
        )
    controls = StageControlSuite(
        controller=Controller(
            ControllerAgent(
                client=client,
                prompt_template=controller_prompt,
                common_context=common_context,
                cot_mode=cot_mode,
                gripper_color=gripper_color,
                proprio_plugin=ProprioceptionPlugin(
                    plugins.enabled("proprioception"),
                    high_above_table_m=high_above_table_m(cfg),
                    # Surface the real per-step distances: the controller's fine step, and the
                    # coarse step only when variable_step is on (else every step is fine).
                    fine_step_m=controller.step_m,
                    coarse_step_m=(vstep.coarse_step_m if (vstep is not None and vstep.enabled) else None),
                ),
                mcq_plugin=McqPlugin(plugins.enabled("mcq", default=False)),
                mem_text_plugin=mem_text_plugin,
                # Same instance the controller uses, so (when enabled) it renders the
                # {variable_step} prompt and parses the VLM's wrist-visibility judgment.
                variable_step_plugin=controller.variable_step_plugin,
                action_chunk_plugin=action_chunk_plugin,
                # Same instance the controller uses for yaw compensation, so the prompt block
                # + offered ROTATE tokens stay in lockstep with the executor.
                rotation_plugin=controller.rotation_plugin,
                # Same instance the runner grounds/annotates with, so the rewritten
                # AFFORD field always names a dot that is actually drawn on the frame.
                affordance_plugin=affordance_plugin,
                action_ablation_plugin=action_ablation_plugin,
                table_height_m=table_height_m,
            )
        ),
    )
    return RealEpisodeRunner(
        session=session,
        controller=controller,
        planner=planner,
        controls=controls,
        logger=logger,
        config=v0_config,
        task=str(cfg["task"]),
        gripper_color=gripper_color,
        max_steps=int(cfg["max_steps"]),
        loop_period_s=float(cfg.get("loop_period_s", 0.0)),
        use_wrist_image=bool(cfg.get("use_wrist_image", True)),
        debug=debug,
        recovery_plugin=recovery_plugin,
        action_chunk_plugin=action_chunk_plugin,
        deepplan_plugin=deepplan_plugin,
        affordance_plugin=affordance_plugin,
        dagger_plugin=dagger_plugin,
        # Blind-mode review frames + action_table.json persistence; off -> inert.
        action_ablation_plugin=action_ablation_plugin,
        recent_moves_max=mem_text_plugin.max_recent,
        viewer=viewer,
    )

def dual_session_config(
    arm_cfgs: dict[str, dict[str, Any]], args: argparse.Namespace
) -> DualPiperSessionConfig:
    """Both arms + three cameras from the per-arm flattened views (the wrist topic is
    the only camera key that differs per arm; the front camera is shared)."""
    rbs = {side: (arm_cfgs[side].get("robot", {}) or {}) for side in SIDES}
    left = rbs["left"]
    return DualPiperSessionConfig(
        use_mock_robots=bool(args.mock_robot or left.get("use_mock_robot", False)),
        open_width_left_m=float(left.get("open_width_m", 0.07)),
        open_width_right_m=float(rbs["right"].get("open_width_m", 0.07)),
        connect_cameras=True,
        use_mock_cameras=bool(args.mock_cameras or left.get("use_mock_cameras", False)),
        front_camera_topic=str(left.get("front_camera_topic", "/camera_f/color/image_raw")),
        wrist_left_camera_topic=str(
            left.get("wrist_camera_topic", "/camera_l/color/image_raw")
        ),
        wrist_right_camera_topic=str(
            rbs["right"].get("wrist_camera_topic", "/camera_r/color/image_raw")
        ),
        camera_max_age_s=float(left.get("camera_max_age_s", 1.0)),
        camera_connect_timeout_s=float(left.get("camera_connect_timeout_s", 10.0)),
        observation_resolution=int(arm_cfgs["left"].get("camera_resolution", 256)),
        verbose=True,
    )


def make_home_arm(
    arm_cfgs: dict[str, dict[str, Any]], session: DualPiperSession
) -> Any:
    """Return ``home(side)``: send ONE arm back to its BEGIN pose.

    Used to retire an arm that finished its work while the other is still running, in
    both modes. The gripper is NOT force-opened: a finished arm has already released,
    and opening one that unexpectedly still holds an object would drop it mid-air.
    """

    def home(side: str) -> None:
        cfg = arm_cfgs[side]
        go_begin(
            session.robots[side],
            cfg.get("begin_joints"),
            time_to_go=float(cfg.get("begin_time_s", 3.0)),
            label=f"{side} begin",
            open_gripper=False,
            verbose=False,  # the runner already narrates this; the joint vector is noise
        )

    return home


def auto_home_dual(arm_cfgs: dict[str, dict[str, Any]], session: DualPiperSession) -> None:
    try:
        go_begin_dual(
            session.robots,
            {side: arm_cfgs[side].get("begin_joints") for side in SIDES},
            time_to_go=float(arm_cfgs["left"].get("begin_time_s", 3.0)),
            label="begin",
            open_gripper=True,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - the reset must not mask the rollout result
        print(f"[run-dual] go-begin after the rollout failed: {exc}")


def arm_summary(controller: Any) -> str:
    """One line per arm: the safety floor and how its motion is actually realized."""
    parts = [f"z-floor {fmt_z(controller.z_floor_m)}"]
    if getattr(controller, "_joint_stream_ok", False):
        parts.append(
            f"joint-stream {float(controller.joint_stream_hz):.0f} Hz "
            f"(ori-flex {math.degrees(controller.ori_flex_rad):.0f} deg)"
        )
    else:
        parts.append("endpose (MOVE P)")
    return "  ·  ".join(parts)


def fmt_z(z_floor_m: Any) -> str:
    if z_floor_m is None:
        return "DISABLED (the arm may descend without a software limit)"
    return f"{float(z_floor_m):.4f} m"
