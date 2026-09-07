"""RoboLab task construction: what to grasp, where to place it, and at what height.

Sibling of ``real2sim/maniskill/tasks.py``, but the job is different in one important way.
On ManiSkill the scenes are OURS (the ``core.sim.maniskill_scenes`` table), so that file
declares a hand-written table of actors, heights and a layout sampler. RoboLab ships 120
authored USD scenes whose success predicates and manipulation targets are already declared
IN the task class::

    subtasks = [pick_and_place(object=["banana"], container="bowl", ...)]

so this module *reads* that declaration (via
:func:`core.sim.robolab_task.task_targets`, surfaced as ``handle.targets``) instead of
re-stating it. One consequence worth knowing: adding a new RoboLab task needs no change
here -- if its subtask spec is a ``pick_and_place`` / ``pick_and_place_on_surface``, the
oracle can already drive it.

Geometry, and why the numbers look odd
--------------------------------------
RoboLab's "TCP" (the body its relative IK drives, and what ``rl_tcp`` reports) is
``base_link`` -- the Robotiq 2F-85's mounting flange, NOT the fingertips. The Robotiq spec
puts the fingertips :data:`FLANGE_TO_FINGERTIP_M` below the flange, so every height here
adds that offset: to close the fingers around an object at height z, the flange must sit at
``z + FLANGE_TO_FINGERTIP_M``. (RoboLab's own ``DroidRelIKActionCfg`` carries the same
constant commented out as an optional ``body_offset``; we keep the flange as the control
frame -- matching deployment -- and correct in the planner instead.)

Object heights come from the OBB centroid rather than the USD root pose: an authored
asset's root frame is wherever its author put it, often the base and sometimes off the mesh
entirely, whereas the OBB centre is always on the object.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Flange -> fingertip, i.e. how far below the frame ``rl_tcp`` reports the fingers close.
# EMBODIMENT-SPECIFIC -- the default embodiment is the Panda hand (robot: franka):
#
#   Panda hand   0.1034 m -- the standard Franka TCP offset from panda_hand, CONFIRMED
#                behaviourally on 2026-08-04 (RubiksCubeTask, 58 mm cube, centroid z
#                0.0317): flange z 0.090-0.120 closed on air (width 0.000), while 0.130,
#                0.135, 0.140, 0.150 and 0.160 all gripped (width ~0.055) and lifted the
#                cube 142 mm. 0.0317 + 0.1034 = 0.135 sits inside that window with the
#                fingertips at the cube's centre of mass.
#                NB the sweep's own "centre of the held range => 0.1113" is NOT the value
#                to use: the range was never closed from above (0.160, the last point
#                tested, still held), so that "centre" is just the midpoint of the sweep.
#                The upper heights work because the fingers still catch the top of a 58 mm
#                cube; aiming at the centroid is the robust choice for smaller objects.
#   Robotiq 2F-85 0.118 m -- MEASURED behaviourally on 2026-07-29, because RoboLab's Robotiq
#                USD is flattened (every finger body reports the same pose as base_link, so
#                the offset cannot be read off the kinematics). Sweeping flange heights on
#                the 58 mm cube: z 0.020/0.040 jammed the fingers on the TABLE, z 0.060-0.120
#                closed on air (fingertips would be below the table), z 0.150 HELD and lifted
#                the cube 126 mm -> 0.150 - 0.032 = 0.118. Note the datasheet figure (0.1628)
#                is wrong for this rig: it aims ~45 mm high and grasps thin air, silently.
#
# RE-MEASURED 2026-08-12 for the SHORT FINGER on its 35.9 mm bracket
# (core.sim.robolab_franka.PANDA_USD). Sweep in
# rollouts/robolab/CHECK/calib_bracket35.9.log (calibrate_fingertip.py, RubiksCubeTask,
# 58 mm cube): HELD from 103.7 mm to 156.4 mm above the centroid, width 55-57 mm throughout
# (i.e. the cube's own width -- see GRASPED_WIDTH_M), and 161.4 mm closed on air.
#
# WHICH LIMIT BINDS CHANGED, and that is why this is not the old value plus the 20 mm the
# tip moved down. The fingertip now reaches 132.3 mm below the flange, so on short objects
# the TIP hits the table before the palm ever reaches the object -- and the tip-on-table
# bound rises as objects get SHORTER, the opposite of the palm bound. Per object:
#
#     object        half-height   palm bound   tip bound   -> usable
#     blocks 46 mm      23.0         89.0        109.3        109.3 .. 155.3
#     rubiks 58 mm      29.0         95.0        103.3        103.3 .. 161.3
#     soup/can 63 mm    31.5         97.5        100.8        100.8 .. 163.8
#     yogurt 65 mm      32.5         98.5         99.8         99.8 .. 164.8
#
# The intersection across all twelve tasks is 109.3 .. 155.3 mm, and the blocks set the
# floor. 0.130 leaves them 20.7 mm of margin below and 25.3 mm above; the naive
# "old value + 20 mm" of 0.1234 would have left only 14.1 mm, under the ~15 mm that go_z's
# 11 mm tolerance plus the +-4 mm grasp jitter can consume on its own.
FLANGE_TO_FINGERTIP_M = 0.130

# How far above the grasp height the arm travels while carrying. Enough to clear a typical
# RoboLab container rim (bowls/bins are 6-12 cm tall) without leaving the IK's comfortable
# workspace.
CARRY_CLEARANCE_M = 0.12

# Vertical margin left between the carried object's underside and the drop target when the
# gripper opens. Too small risks colliding with the rim on the way in; too large drops the
# object from a height and it bounces out.
#
# Lowered 0.04 -> 0.02 (one whole token) on 2026-08-12. At 4 cm the release happened a
# clear two tokens above the rim, which reads in the video as the arm stopping early and
# letting go into the air -- and is the geometry the policy was being taught to reproduce.
# 2 cm keeps a token's worth of clearance over the rim (the descent is quantised to 2 cm
# and ``go_z`` tolerates 1.1 cm, so the flange can arrive up to ~1 cm low) while putting
# the object close enough that it drops in rather than bounces.
DROP_MARGIN_M = 0.02

# Approach height above the object before descending onto it.
APPROACH_CLEARANCE_M = 0.10

# A grasp is "on something" when the fingers stop short of fully closed. A closed-and-empty
# gripper drives its joint(s) to the close command and reads ~0 m on both embodiments.
#
# 0.005 works because the yellow tip's grasping face is flush with the stock finger's
# (link y = 0, see make_short_finger_asset.py), so a held object reports its OWN width:
# a 46 mm block reads 46 mm, exactly as the real rig's 38.8 mm cube reads 38.76 mm.
#
# This was briefly 0.002, during a version that mounted the tips 20 mm outboard. That
# shifts every reported width down by twice the offset -- the block tasks fell to 6.0 mm,
# one millimetre over this threshold, where a slight squeeze reads as an empty grasp and
# `_grasp_with_retries` discards a perfectly good episode. The lesson is not the threshold:
# it is that moving the grasping face silently rescales every width in the dataset.
GRASPED_WIDTH_M = 0.005

# Control steps to hold a gripper command before the width means anything. RoboLab drives
# the Robotiq through a BINARY joint command and the linkage takes a while to travel:
# measured, the width still read 0.081 m (near-open) right after a close command that
# ultimately settled at 0.000 m (empty). Sampling too early therefore reports a successful
# grasp for an empty gripper, and the mistake only surfaces later as an object that never
# left the table. The atomic executor's default of 10 is not enough here.
GRIPPER_SETTLE_STEPS = 40

# Width above which a gripper commanded CLOSED is provably holding nothing -- i.e. the
# object slipped out. Sits just under the fully-open width of BOTH embodiments (Panda hand
# 0.080 m = 2 x 0.040; Robotiq 2F-85 0.085 m), while staying clear of a held 58 mm cube. Detecting this matters for DATA QUALITY, not just
# for success accounting: a follower that drops mid-carry keeps executing the recorded
# carry tokens, so every remaining frame is labelled with a transport action while the
# gripper is empty -- textbook mislabelled training data, and invisible in the aggregate
# step-size / off-axis statistics. Only watching the video, or this check, catches it.
DROPPED_WIDTH_M = 0.075


class UnsupportedTask(RuntimeError):
    """The task's subtask spec is not a shape the scripted oracle knows how to drive."""


def resolve_plan(backend: Any) -> dict[str, Any]:
    """What to pick and where to put it, read off the task's own subtask declaration.

    Returns ``{"object": str, "target": str, "target_kind": "container"|"surface"}``.
    Raises :class:`UnsupportedTask` when the task declares something else (stacking,
    ordering, tool use, multi-object clutter) -- those need their own planner, and the
    honest failure is better than an oracle that silently grasps the wrong thing.
    """
    targets = dict(getattr(backend, "targets", {}) or {})
    objects = list(targets.get("objects") or [])
    container = targets.get("container")
    surface = targets.get("surface")
    if not objects:
        raise UnsupportedTask(
            "task declares no manipulable object in its subtasks "
            f"(targets={targets!r}); the scripted oracle only drives pick_and_place tasks."
        )
    if len(objects) > 1:
        raise UnsupportedTask(
            f"task declares {len(objects)} objects ({objects}); the scripted oracle drives "
            "single-object pick_and_place only."
        )
    if container:
        return {"object": objects[0], "target": container, "target_kind": "container"}
    if surface:
        return {"object": objects[0], "target": surface, "target_kind": "surface"}
    raise UnsupportedTask(
        f"task declares object {objects[0]!r} but no container/surface to place it in."
    )


def _extent(backend: Any, name: str) -> Optional[tuple[float, float]]:
    """LIVE (bottom_z, top_z) of an object, or None if its geometry is not queryable.

    Deliberately NOT ``backend.object_aabb``: RoboLab's ``get_aabb`` returns cached
    POSE-INDEPENDENT geometry, so a held object still reports its table-top height. Using
    it to ask "how far below the flange is the carried object" over-estimates by the whole
    lift height, which puts the release point tens of centimetres too high -- and nothing
    errors, the object is simply dropped from above. ``object_extent`` combines the live
    centroid with the pose-independent size instead.
    """
    try:
        return backend.object_extent(name)
    except Exception:  # noqa: BLE001 -- extras/prims without queryable geometry
        return None


def object_centroid(backend: Any, name: str) -> np.ndarray:
    """OBB centre of a scene object, env-local frame."""
    return backend.object_centroid(name)


def grasp_tcp(backend: Any, obj: str, z_jitter: float = 0.0,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Flange pose that puts the fingertips around the object's centre of mass.

    ``z_jitter`` adds a small uniform vertical offset so the frames preceding GRASP are not
    all identical across episodes (the ManiSkill oracle does the same).
    """
    centre = object_centroid(backend, obj)
    z = float(centre[2]) + FLANGE_TO_FINGERTIP_M
    if z_jitter and rng is not None:
        z += float(rng.uniform(-z_jitter, z_jitter))
    return np.array([centre[0], centre[1], z], dtype=np.float64)


def approach_tcp(backend: Any, obj: str) -> np.ndarray:
    tcp = grasp_tcp(backend, obj)
    return np.array([tcp[0], tcp[1], tcp[2] + APPROACH_CLEARANCE_M], dtype=np.float64)


def carry_z(backend: Any, obj: str, target: str,
            grasp_flange_z: Optional[float] = None) -> float:
    """Flange height to travel at: clear of BOTH the pick site and the target rim.

    ``grasp_flange_z`` is the flange height the grasp happened at, and the caller MUST
    pass it once the object is in the fingers.

    Without it this function is self-referential and ratchets. It used to derive the pick
    side from ``grasp_tcp(backend, obj)``, i.e. from the object's LIVE centroid -- but a
    held object moves with the hand, so that term evaluates to roughly "the flange, right
    now", and the returned height becomes "wherever you are + 10.8 cm". Every
    recomputation therefore raises the bar:

        measured on RubiksCubeTask -- flange 0.138 -> carry_z 0.246 -> carry_z 0.354,
        demo peaked at z = 0.393 against an intended 0.283, an 11 cm overshoot

    and because the recorder calls it once to lift and again to move across, the lift was
    still running when the horizontal leg started. The two blended into a diagonal (26% of
    carry control steps moved z and xy together), and the follower quantised that diagonal
    into the up-right-up-right staircase seen in the rollouts. Nothing errored: the demo
    succeeded, the tokens were single-axis, the statistics were clean.
    """
    extent = _extent(backend, target)
    top_z = extent[1] if extent is not None else float(object_centroid(backend, target)[2])
    heights = [top_z + FLANGE_TO_FINGERTIP_M]
    # The pick side comes from the caller's recorded grasp height when the object is held,
    # and only from the object's own pose while it is still on the table.
    heights.append(float(grasp_flange_z) if grasp_flange_z is not None
                   else grasp_tcp(backend, obj)[2])
    return max(heights) + CARRY_CLEARANCE_M


def place_tcp(backend: Any, obj: str, target: str, target_kind: str,
              drop_margin_m: Optional[float] = None) -> np.ndarray:
    """Flange pose to open the fingers at, so the carried object lands on/in the target.

    The XY is the target's OBB centre, corrected for how far the CARRIED object actually
    sits from the flange -- after a grasp the object can end up a centimetre off-centre in
    the fingers, and RoboLab's containment predicates check the OBJECT's position, not the
    gripper's.

    The Z is the target's top surface plus a margin, again in flange coordinates. For a
    container that is its rim -- releasing just above the rim lets the object drop in; for
    a bare surface it is the surface itself.

    Both heights come from LIVE geometry (:func:`_extent`). Using the cached
    ``get_aabb`` here is what made the first RoboLab dataset release the cube ~47 cm up:
    the carried object still reported its table-top underside, so ``carried_drop`` came out
    ~0.38 m instead of ~0.15 m, and the follower then spent the rest of the episode driving
    into its own workspace limit.
    """
    target_centre = object_centroid(backend, target)
    carried_centre = object_centroid(backend, obj)
    tcp = backend.tcp_pos()
    # Where the object is relative to the flange right now == where it will land relative
    # to wherever we put the flange.
    offset_xy = carried_centre[:2] - tcp[:2]
    xy = target_centre[:2] - offset_xy

    target_extent = _extent(backend, target)
    top_z = target_extent[1] if target_extent is not None else float(target_centre[2])
    # The carried object hangs below the flange by this much; keep its UNDERSIDE above the
    # drop surface rather than the flange, or tall objects get slammed into the rim.
    carried_extent = _extent(backend, obj)
    carried_drop = (
        float(tcp[2] - carried_extent[0]) if carried_extent is not None
        else FLANGE_TO_FINGERTIP_M
    )
    margin = DROP_MARGIN_M if drop_margin_m is None else float(drop_margin_m)
    z = top_z + margin + carried_drop
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def describe(backend: Any, plan: dict) -> str:
    return (
        f"{backend.task_description!r} | object={plan['object']} "
        f"{plan['target_kind']}={plan['target']}"
    )
