"""ManiSkill task construction: which scene, which actors, and how a layout is sampled.

One table describes every task the generators can drive (:data:`TASKS`), and one sampler
places its objects (:func:`randomize_layout`). This used to be copy-pasted three times --
once per generator -- with slightly different bookkeeping each time; keeping it here means
a scene tweak lands in the oracle, the demo recorder, the follower AND the eval runner at
the same moment. ``core.sim.mvtoken_maniskill_runner`` calls :func:`randomize_layout` for
``layout: wide`` evals precisely so eval layouts come from the training sampler.

The functions take the raw ManiSkill ``env`` (not a discretiser backend): they are scene
plumbing, shared with the deployment runner, and predate any token being emitted.

Why the layout is sampled here at all
-------------------------------------
Both RLinf rigs randomise only the MANIPULATED object and effectively pin the target
(BlockPAP's coaster moves over ~4x2 cm; BlockStack pins the gray block at a fixed pose), so
a policy trained on their native layouts would only ever see one target position. These
samplers place BOTH objects over the full reachable box, rejection-sampled for a minimum
separation so the open fingers can descend on one object without fouling the other.

Lattice snapping (``snap``)
---------------------------
Where the success tolerance is TIGHTER than half an atomic step, a freely-sampled target is
geometrically unreachable for a large fraction of episodes no matter how good the policy
(one 2 cm token cannot land closer than 1 cm per axis). ``snap`` quantises the sampled XY
onto the ``step_m`` lattice anchored at the gripper's settled start XY, which keeps the
positions random and spread over the whole box -- the snapping is invisible in the images --
while making the task exactly solvable. BlockStack needs it (1 cm tolerance vs 2 cm step);
BlockPAP's 4.3 cm coaster does not.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import to_np

# Reachable sampling box on the RLinf rig, robot-base world frame (corners verified
# reachable by a probe run before the first batch). The table spans X 0.20-0.80; top-down
# grasping stays reliable inside this box.
WIDE_BOX = {"x": (0.40, 0.62), "y": (-0.22, 0.22), "min_sep": 0.12}

# Rejection-sampling budget. Only reached if a draw keeps violating ``min_sep``, which is
# rare in this box; on exhaustion the last draw is used.
_SAMPLE_ATTEMPTS = 400

# Coaster orientation in BlockPAP (flat-lying quaternion, wxyz).
_COASTER_Q = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]


def const(env, name: str) -> Any:
    """A scene geometry constant off the env (``TABLE_Z``, ``BLOCK_HALF_SIZE``, ...)."""
    return getattr(env.unwrapped, name)


def actor_pos(env, name: str) -> np.ndarray:
    return to_np(getattr(env.unwrapped, name).pose.p).reshape(-1, 3)[0].astype(np.float64)


def _block_z(env) -> float:
    return float(const(env, "TABLE_Z") + const(env, "BLOCK_HALF_SIZE")[2])


def _coaster_z(env) -> float:
    return float(const(env, "TABLE_Z") + const(env, "COASTER_THICKNESS"))


TASKS: dict[str, dict[str, Any]] = {
    # -- RLinf real2sim rigs. ``env_id`` is the key into core.sim.maniskill_scenes.SCENES, which
    # owns the instruction and the robot uid; only the LAYOUT SAMPLING is described here --
    # that is this file's job and has no meaning at deployment.
    "blockpap": {
        "env_id": "BlockPAP-v1",
        "snap_layout": False,   # 4.3 cm coaster -- the lattice residual fits easily
        "carried": "cube",
        "target": "target",
        # (actor, z, quaternion) for the manipulated object, then the target.
        "layout": [("cube", _block_z, None), ("target", _coaster_z, _COASTER_Q)],
        "xy_keys": ("block_xy", "coaster_xy"),
        # Height the CARRIED object's centre should end at, given the target's position:
        # block seated on the coaster (+4 mm, so it is dropped rather than pressed in).
        "drop_z": lambda env, t: (t[2] + float(const(env, "COASTER_THICKNESS"))
                                  + float(const(env, "BLOCK_HALF_SIZE")[2]) + 0.004),
    },
    "blockstack": {
        "env_id": "BlockStack-v1",
        "snap_layout": True,    # 1 cm success tolerance vs 2 cm step -- see module docstring
        "carried": "white_block",
        "target": "gray_block",
        "layout": [("white_block", _block_z, None), ("gray_block", _block_z, None)],
        "xy_keys": ("white_xy", "gray_xy"),
        "drop_z": lambda env, t: t[2] + 2 * float(const(env, "BLOCK_HALF_SIZE")[2]),
    },
}


def task_spec(task: str) -> dict[str, Any]:
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; available: {sorted(TASKS)}")
    return TASKS[task]


def task_text(task: str) -> str:
    """The task's instruction, from the deployment scene table (not a second copy)."""
    from core.sim.maniskill_scenes import instruction_for

    return instruction_for(task_spec(task)["env_id"])


def task_robot_uids(task: str) -> str:
    """The robot uid the deployment runner builds this task with."""
    from core.sim.maniskill_scenes import robot_uids_for

    return robot_uids_for(task_spec(task)["env_id"])


#: Deployment config whose camera contract generated data must match. Same role as the
#: SCENES table below: one source of truth, read by both the runner and the generators.
DEFAULT_ROBOT_CONFIG = "configs/robot_maniskill.yaml"


def backend_kwargs(task: str, robot_config: str = DEFAULT_ROBOT_CONFIG,
                   **overrides: Any) -> dict[str, Any]:
    """``make_backend`` kwargs for a task.

    The instruction and the robot uid come from :data:`core.sim.maniskill_scenes.SCENES` -- the
    DEPLOYMENT table -- so generated data is labelled with exactly the task text the eval
    runner sends and driven by exactly the robot it builds.

    The camera transform contract comes from the deployment CONFIG for the same reason,
    via :func:`core.config.camera_contract`: training frames and inference frames have to
    be byte-identical, so which camera / rotation / flip / crop is used cannot be stated
    twice. (The RoboLab side learned this the expensive way -- its generators kept their
    own argparse defaults, and a re-measured wrist rotation left a whole dataset 90 deg
    off with nothing in the aggregate statistics to show for it.)

    ``overrides`` still wins, so a probe can pin one value without editing the config.
    """
    from pathlib import Path

    from core.config import camera_contract, load_yaml
    from core.sim.maniskill_scenes import instruction_for, robot_uids_for

    spec = task_spec(task)
    env_id = spec["env_id"]
    # .../scripts/trajectory/real2sim/maniskill/tasks.py -> repo root is 4 levels up.
    root = Path(__file__).resolve().parents[4]
    kwargs: dict[str, Any] = {
        "env_id": env_id,
        "task_description": instruction_for(env_id),
        "robot_uids": robot_uids_for(env_id),
        **camera_contract(load_yaml(root / robot_config)),
    }
    kwargs.update(overrides)
    return kwargs


def _sample_pair(rng: np.random.Generator, anchor_xy, step_m: float,
                 snap: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two independent random XY in :data:`WIDE_BOX`, rejection-sampled for separation."""
    def q(x: float, y: float) -> tuple[float, float]:
        if not snap:
            return float(x), float(y)
        return (float(anchor_xy[0] + round((x - anchor_xy[0]) / step_m) * step_m),
                float(anchor_xy[1] + round((y - anchor_xy[1]) / step_m) * step_m))

    ax = ay = bx = by = 0.0
    for _ in range(_SAMPLE_ATTEMPTS):
        ax, ay = q(rng.uniform(*WIDE_BOX["x"]), rng.uniform(*WIDE_BOX["y"]))
        bx, by = q(rng.uniform(*WIDE_BOX["x"]), rng.uniform(*WIDE_BOX["y"]))
        if float(np.hypot(ax - bx, ay - by)) >= WIDE_BOX["min_sep"]:
            break
    return (ax, ay), (bx, by)


def sample_layout(env, task: str, rng: np.random.Generator, anchor_xy=None,
                  step_m: float = 0.02, snap: bool = False) -> dict:
    """Sample absolute poses for the task's two objects; does NOT touch the sim.

    Returns ``{"actors": {name: {"p": [...], "q": [...]}}, ...}`` -- absolute poses, so a
    track can carry the layout and the follower can reproduce the exact same scene later
    (see :func:`apply_layout`).
    """
    spec = task_spec(task)
    if not spec.get("layout"):
        raise ValueError(f"no layout randomisation defined for task {task!r}")
    anchor = anchor_xy if anchor_xy is not None else (0.0, 0.0)
    (ax, ay), (bx, by) = _sample_pair(rng, anchor, step_m, snap)
    actors: dict[str, dict] = {}
    for (name, z_of, quat), (x, y) in zip(spec["layout"], ((ax, ay), (bx, by))):
        pose: dict[str, Any] = {"p": [float(x), float(y), z_of(env)]}
        if quat is not None:
            pose["q"] = list(quat)
        actors[name] = pose
    key_a, key_b = spec["xy_keys"]
    return {
        "actors": actors,
        "separation": round(float(np.hypot(ax - bx, ay - by)), 4),
        "snapped": bool(snap),
        key_a: [round(ax, 4), round(ay, 4)],
        key_b: [round(bx, 4), round(by, 4)],
    }


def apply_layout(env, layout: dict) -> None:
    """Set actor poses from a layout dict (``{actor: {"p": [...], "q": [...]}}``).

    Accepts either a full layout (with an ``actors`` key) or the inner mapping.
    """
    import sapien

    u = env.unwrapped
    for name, pose in layout.get("actors", layout).items():
        p = [float(v) for v in pose["p"]]
        q = pose.get("q")
        getattr(u, name).set_pose(
            sapien.Pose(p=p, q=[float(v) for v in q]) if q else sapien.Pose(p=p)
        )


def randomize_layout(env, task: str, rng: np.random.Generator, anchor_xy=None,
                     step_m: float = 0.02, snap: Optional[bool] = None) -> dict:
    """Sample a layout and apply it. ``snap=None`` uses the task's own policy.

    Call AFTER the arm has settled (the lattice is anchored at the settled gripper XY) and
    let the objects come to rest before reading their poses.
    """
    if snap is None:
        snap = bool(task_spec(task).get("snap_layout", False))
    layout = sample_layout(env, task, rng, anchor_xy, step_m, snap)
    apply_layout(env, layout)
    return layout
