#!/usr/bin/env python3
"""Build the short-finger Franka USD: ``assets/panda_short_finger.stl`` on a Panda hand.

The real rig this data is meant to match does not wear Franka's stock black fingers. It
wears a pair of metal L-brackets carrying a **yellow printed fingertip** -- a 20 x 18 x 8.5
mm mounting plate on a single screw, with a ~1 mm blade that sweeps 9.5 mm inward, so the
two blades close toward each other like tweezers. ``panda_short_finger.stl`` IS that yellow
part, and simulating the stock finger instead puts a differently shaped, differently
coloured gripper in every training frame.

This script writes a LOCAL robot asset that keeps everything else about IsaacLab's
``panda_instanceable.usd`` and swaps only the two fingers -- both their visuals and their
collision meshes, so what the policy sees and what the physics does stay the same object.

Layout written under ``--out`` (default ``assets/robolab_franka``)::

    panda_short_finger.usda                 root: references the stock robot, overrides fingers
    Props/short_finger_left.usd             visual mesh  + yellow material
    Props/short_finger_right.usd            mirrored
    Props/short_finger_collisions.usd       both collision meshes (convexHull, as stock)

The root layer REFERENCES IsaacLab's asset over https rather than vendoring 10 MB of USD;
Kit resolves and caches it exactly as it does for the stock path.

Placement (STL mm -> panda_leftfinger link frame, metres). Both axes are pinned to
something measured, not to what looks right:

    link X (width)   = (stl_x - 10.0) / 1000        # STL spans 0..20, link frame is centred
    link Y (across)  = (9.5 - stl_z)  / 1000        # the BLADE TIP (stl_z=+9.5) lands on
                                                    # y = 0 -- the stock finger's grasping
                                                    # plane, which is what gripper_width
                                                    # already measures
    link Z (down)    = (18.0 - stl_y + 55.9) / 1000 # stl_y=+18 (plate) sits at the end of
                                                    # the bracket; the tip spans 55.9..93.9

**Y is aligned by the GRASPING face, not the mounting face.** ``gripper_width`` comes off
the finger joints, so it reports where the stock finger's grasping face would be -- and the
real rig still reads a plain 79.86 mm fully open, so its semantics are untouched by the
retrofit. Putting the tip's grasping face anywhere else silently rescales every width in
the dataset by twice the offset. Calibrated against
``MVTOKEN_RAW/0704/pap_pink_cube/rollout_004``: 79.86 mm open, 38.76 mm holding its cube.

Two earlier versions aligned the tip's MOUNTING face against the stock finger's back
instead, and neither failed loudly -- 128 episodes were generated and shipped on the first
one. Only holding a real frame next to a rendered one showed it.

**Z is set by the bracket length**, which is most of the assembly: the tip hangs well below
the palm on the real rig. ``BRACKET_LENGTH_MM`` is therefore the constant the whole height
stack keys off, and changing it invalidates ``tasks.FLANGE_TO_FINGERTIP_M``.

Other consequences worth knowing:

* PhysX approximates the collision mesh as a convex hull (the stock fingers do too), which
  fills the space under the blade: the effective grasping face is a near-vertical plane at
  link y = 0 rather than the blade alone. That is what makes a 58 mm cube graspable at all
  -- a 1 mm blade at 24.6 degrees would squeeze it out.
* ``FLANGE_TO_FINGERTIP_M`` is MEASURED. Re-run ``calibrate_fingertip.py`` after any change
  here; do not scale the old value by the geometric shift.

Run (RoboLab's interpreter; no Isaac Sim app needed, this only writes USD)::

    $ROBOLAB_ROOT/.venv/bin/python \
        scripts/trajectory/real2sim/robolab/make_short_finger_asset.py
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

# IsaacLab's stock Franka, the asset this one is a thin override of. Kept as a literal
# rather than read from ``ISAACLAB_NUCLEUS_DIR`` so the generator runs without Isaac Sim;
# it is the same string ``isaaclab.utils.assets`` composes at runtime.
STOCK_PANDA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
    "/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)

# -- placement (see module docstring) -----------------------------------------
STL_WIDTH_CENTRE_MM = 10.0   # STL spans x 0..20; the link frame is centred on the width
STL_ROOT_END_MM = 18.0       # link_z = this - stl_y; puts the plate's top at the link origin

# WHAT THE TIP MUST BE ALIGNED TO IS ITS GRASPING FACE, NOT ITS MOUNTING FACE.
#
# ``gripper_width`` is derived from the finger JOINTS, so it measures where the stock
# finger's grasping face would be — on the real rig too, which still reports 79.86 mm fully
# open, i.e. the untouched Franka stroke. For a recorded width to mean the same thing in
# sim as on the robot, the yellow tip's grasping face has to sit where the stock finger's
# did: at the link frame, y = 0.
#
# CALIBRATED against a real rollout (MVTOKEN_RAW/0704/pap_pink_cube/rollout_004): fully
# open 79.86 mm, holding the cube 38.76 mm. Both are plain stock-Franka numbers.
#
# Two earlier versions got this wrong by aligning the tip's MOUNTING face against the stock
# finger's back instead, and neither failed loudly:
#   * inner face at y = 11.85 → widths 16 mm too large, 128 episodes generated and shipped;
#   * inner face at y = 20.00 → tips 120 mm apart fully open against the rig's 79.86, and
#     a 46 mm block reporting a 6.5 mm width against the rig's 38.76 for a 38.8 mm cube.
# Only holding a real frame next to a rendered one showed it.
TIP_INNER_FACE_MM = 0.0                        # grasping face, flush with the stock finger's
STL_BLADE_TIP_MM = 9.5                         # the STL's most-inward point, in its own z
STL_MOUNT_FACE_MM = TIP_INNER_FACE_MM + STL_BLADE_TIP_MM   # link_y = this - stl_z
STL_TIP_THICKNESS_MM = 14.5                    # STL z spans -5..+9.5
TIP_OUTER_FACE_MM = TIP_INNER_FACE_MM + STL_TIP_THICKNESS_MM   # 14.5, the mounting face
# Consequences, and the numbers to check a rebuild against: the tips close to touching
# (0 mm) and open to 79.86 mm exactly like the stock fingers, and a held object reports its
# own width — a 46 mm block reads 46 mm, matching how the rig's 38.8 mm cube reads 38.76.
STOCK_SLIDER_FACE_MM = 26.35                   # stock finger's back face, on the slider

# -- mounting bracket ---------------------------------------------------------
# The yellow tip is NOT bolted to the finger slider: on the real gripper it hangs off the
# end of a metal L-bracket that replaces the stock finger, and the bracket is most of the
# assembly's length. Modelling only the tip -- 38 mm where the stock finger was 53.9 --
# does not just look wrong, it stops the gripper working: the PALM reaches 66 mm below the
# flange, so the usable grip depth falls from 46.3 mm to 30.4 mm, and closing on a 58 mm
# cube means driving the hand into the top of it. MEASURED: the flange bottomed out 98.5 mm
# above the cube's centroid with the hand 3.5 mm off its top face, and the whole sweep
# collapsed onto that one blocked height.
#
# So the bracket is modelled too: a plain plate from the finger's link origin down to where
# the tip's mounting plate begins. Its LENGTH is what matters -- it sets where the
# fingertips end up -- and the default restores the stock finger's reach (53.9 mm), which
# is the geometry every height in ``tasks.py`` was calibrated against and keeps the palm
# clearance the planner assumes.
#
# On the real rig the bracket is proportionally LONGER than this (from the photo, roughly
# 1.5x the tip's own length, putting the fingertips ~95 mm out). Lengthening it here is one
# constant -- but it moves the fingertips, so re-run ``calibrate_fingertip.py`` afterwards
# and expect the whole approach/carry/place height stack to shift with it.
STOCK_FINGER_REACH_MM = 53.9      # stock panda finger: link origin -> fingertip
SHORT_TIP_LENGTH_MM = 38.0        # the STL's own length along the finger

# How far down the bracket carries the tip, measured from the finger's link frame. The real
# bracket is most of the assembly's length -- from the photo the yellow part hangs well
# below the palm -- so the tip sits 20 mm lower than the first estimate placed it.
#
# This is the number the whole height stack keys off: the fingertip ends at
# BRACKET_LENGTH_MM + 38 mm below the link frame, i.e. that much further below the flange.
# Any change here invalidates ``tasks.FLANGE_TO_FINGERTIP_M`` -- re-run
# calibrate_fingertip.py, do not scale the old value by hand.
#
# A 40 mm drop was tried first and swept: it pushed the fingertip to 152.3 mm below the
# flange, at which point the FINGERTIP hits the table before the palm ever reaches the
# object (measured: flange bottomed out 123.8 mm above the cube's centroid, with the tip at
# z = 0.0033 against a table at 0.0027). The whole usable height band shifts up with it,
# and a hand-scaled FLANGE_TO_FINGERTIP_M of 0.1434 would have driven the tip 20 mm through
# the table. 20 mm keeps the tip clear.
BRACKET_LENGTH_MM = 35.9          # tip spans 35.9..73.9 mm; fingertip 132.3 mm below flange
BRACKET_WIDTH_MM = 20.0
BRACKET_THICKNESS_MM = 5.0        # sheet metal, per the photo

# The bracket fills the gap between the tip's mounting face (y = 14.5) and the slider
# (y = 26.35), which is where the real rig's metal L-bracket sits: the yellow part bolts to
# its INBOARD face, so the bracket is entirely outboard of the tip. No step needed now that
# the tip is inboard of the slider rather than outboard of it.
BRACKET_INNER_MM = TIP_OUTER_FACE_MM            # 14.5, flush against the tip's back
BRACKET_OUTER_MM = STOCK_SLIDER_FACE_MM         # 26.35, flush against the slider

# Brushed aluminium, for the bracket.
BRACKET_COLOR = (0.72, 0.73, 0.75)
BRACKET_ROUGHNESS = 0.30
BRACKET_METALLIC = 0.9

# Franka's yellow printed tip. Linear-space sRGB, eyeballed off the reference photo -- the
# point is that the fingers read YELLOW in the frame, not that the hue is colorimetric.
TIP_COLOR = (0.93, 0.72, 0.05)
TIP_ROUGHNESS = 0.45
TIP_METALLIC = 0.0


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(triangles (n,3,3), face normals (n,3)) from a binary STL, in the STL's own units."""
    data = path.read_bytes()
    if len(data) < 84:
        raise SystemExit(f"{path} is too short to be a binary STL")
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if len(data) != expected:
        raise SystemExit(
            f"{path}: expected {expected} bytes for {count} triangles, got {len(data)}. "
            "ASCII STL is not supported; re-export as binary."
        )
    record = np.dtype([("n", "<3f4"), ("v", "3,3f4"), ("a", "<u2")])
    tris = np.frombuffer(data[84:expected], dtype=record)
    return tris["v"].astype(np.float64), tris["n"].astype(np.float64)


def to_link_frame(verts_mm: np.ndarray, mirror: bool) -> np.ndarray:
    """STL coordinates (mm) -> panda_leftfinger / panda_rightfinger link frame (metres).

    The tip is pushed down the finger by :data:`BRACKET_LENGTH_MM`: it mounts on the end of
    the bracket, not on the slider.
    """
    x, y, z = verts_mm[..., 0], verts_mm[..., 1], verts_mm[..., 2]
    link = np.stack(
        [
            (x - STL_WIDTH_CENTRE_MM) / 1000.0,
            (STL_MOUNT_FACE_MM - z) / 1000.0,
            (STL_ROOT_END_MM - y + BRACKET_LENGTH_MM) / 1000.0,
        ],
        axis=-1,
    )
    if mirror:
        # The right finger is the left one mirrored through the hand's Y=0 plane, exactly
        # as the stock pair is (left mesh spans y 0..+26.35, right spans -26.35..0).
        link[..., 1] *= -1.0
    return link


def box_mesh(lo: tuple, hi: tuple, mirror: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, faces, per-face normals) for an axis-aligned box, in metres.

    Winding is outward-facing, and flipped for the mirrored side for the same reason the
    STL's is (see :func:`mesh_arrays`).
    """
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    if mirror:
        y0, y1 = -y1, -y0
    points = np.array(
        [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64
    )
    # index = 4*ix + 2*iy + iz
    quads = [
        (0, 1, 3, 2), (4, 6, 7, 5),   # -X, +X
        (0, 4, 5, 1), (2, 3, 7, 6),   # -Y, +Y
        (0, 2, 6, 4), (1, 5, 7, 3),   # -Z, +Z
    ]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    if mirror:
        faces = faces[:, ::-1]
    e1 = points[faces[:, 1]] - points[faces[:, 0]]
    e2 = points[faces[:, 2]] - points[faces[:, 0]]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, faces, normals


def bracket_boxes() -> list[tuple[tuple, tuple]]:
    """The bracket, as one plate spanning from the tip's back face out to the slider.

    Entirely outboard of the tip, so it never intrudes on the grasping face: the convex
    hull over bracket + tip presents y = 0 as its innermost surface, which is exactly the
    stock finger's grasping plane and therefore what ``gripper_width`` already means.
    """
    half_w = BRACKET_WIDTH_MM / 2.0
    return [
        ((-half_w, BRACKET_INNER_MM, 0.0),
         (half_w, BRACKET_OUTER_MM, BRACKET_LENGTH_MM)),
    ]


def bracket_arrays(mirror: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, faces, per-face normals) for the whole L-bracket, in metres."""
    pts_all, faces_all, norms_all = [], [], []
    offset = 0
    for lo, hi in bracket_boxes():
        p, f, n = box_mesh(tuple(v / 1000.0 for v in lo),
                           tuple(v / 1000.0 for v in hi), mirror)
        pts_all.append(p)
        faces_all.append(f + offset)
        norms_all.append(n)
        offset += len(p)
    return np.vstack(pts_all), np.vstack(faces_all), np.vstack(norms_all)


def mesh_arrays(tris: np.ndarray, mirror: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, faceVertexIndices, per-face normals) for a UsdGeom.Mesh.

    Vertices are welded on exact position so the mesh renders as a solid rather than as
    864 disconnected triangles, and the winding is flipped for the mirrored finger --
    mirroring reverses handedness, and a mesh whose faces all point inward renders black
    and gives PhysX an inside-out hull.
    """
    verts = to_link_frame(tris, mirror).reshape(-1, 3)
    points, inverse = np.unique(np.round(verts, 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    if mirror:
        faces = faces[:, ::-1]
    e1 = points[faces[:, 1]] - points[faces[:, 0]]
    e2 = points[faces[:, 2]] - points[faces[:, 0]]
    normals = np.cross(e1, e2)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    return points, faces, normals


def write_visual_layer(path: Path, parts: list) -> None:
    """Visual layer holding one Mesh per part, each with its own material.

    ``parts`` is ``[(prim_name, material_name, (color, roughness, metallic),
    points, faces, normals), ...]`` -- the yellow printed tip and the metal bracket that
    carries it.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind("component")

    for prim_name, mat_name, (color, rough, metal), points, faces, normals in parts:
        mesh = UsdGeom.Mesh.Define(stage, f"/Root/{prim_name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        # faceVarying normals from the FACE normals: these parts are all flats, chamfers
        # and a drilled hole, so shared vertex normals would round the edges away.
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in np.repeat(normals, 3, axis=0)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(
            [Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))]
        )

        material = UsdShade.Material.Define(stage, f"/Root/Looks/{mat_name}")
        shader = UsdShade.Shader.Define(stage, f"/Root/Looks/{mat_name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()


def write_collision_layer(path: Path, sides: dict) -> None:
    """One layer holding both fingers' collision meshes, mirroring the stock asset's shape.

    ``physics:approximation = convexHull`` matches the stock fingers. It is not a detail:
    the blade is 1 mm thick and swept 24.6 degrees, so the exact mesh would grip a cube on
    a knife edge. The hull closes the space under the blade into a near-flat face, which is
    the surface the grasp heights in ``tasks.py`` are calibrated against.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for prim_name, (points, faces, _normals) in sides.items():
        xform = UsdGeom.Xform.Define(stage, f"/{prim_name}")
        mesh = UsdGeom.Mesh.Define(stage, f"/{prim_name}/collisions")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = points.min(axis=0), points.max(axis=0)
        mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
        # Collision geometry must not be drawn -- without this the hull renders on top of
        # the visual mesh and the finger looks like a solid block in every frame.
        UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        _ = xform
    stage.GetRootLayer().Save()


def fetch_stock_layer(url: str, cache_dir: Path) -> Path:
    """Download IsaacLab's ``panda_instanceable.usd`` (8 KB) once, into ``cache_dir``."""
    import urllib.request

    dest = cache_dir / "panda_instanceable.stock.usd"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def write_root_layer(path: Path, stock_usd: str, cache_dir: Path) -> None:
    """The robot asset: a COPY of the stock Panda layer with the finger arcs rewritten.

    The obvious version of this -- a thin ``over`` layer that references the stock robot
    and re-declares the fingers' references -- does not work, and fails in a way that
    looks fine until you inspect the composed stage. USD composes the two reference arcs
    at DIFFERENT levels: an explicit list op in this layer does not block the one written
    inside ``panda_instanceable.usd``, so both meshes end up on the prim. The new points
    win (this layer is stronger), but the stock finger's leftovers ride along -- its
    ``xformOp:scale = 0.01`` (which shrinks the tip to a 0.4 mm speck), its two
    ``GeomSubset`` children indexing faces that no longer exist, and, on the right finger,
    an ``xformOp:transform`` that rotates the already-mirrored mesh back.

    So the layer is COPIED and edited instead. It is 8 KB of joints and link transforms;
    only the four finger reference arcs change. Every other reference (``./Props/*.usd``
    for the arm links) is rewritten to its absolute URL, so the copy resolves them from
    the same CDN the stock asset does rather than needing 10 MB vendored next to it.
    """
    from pxr import Sdf

    stock_local = fetch_stock_layer(stock_usd, cache_dir)
    source = Sdf.Layer.FindOrOpen(str(stock_local))
    if source is None:
        raise SystemExit(f"could not open the downloaded stock layer {stock_local}")

    layer = Sdf.Layer.CreateNew(str(path))
    layer.TransferContent(source)
    layer.comment = (
        "Franka Panda wearing the yellow short fingertip (assets/panda_short_finger.stl) "
        "instead of the stock black fingers -- visuals AND collision meshes. A copy of "
        f"IsaacLab's panda_instanceable.usd ({stock_usd}) with the four finger reference "
        "arcs rewritten. GENERATED by "
        "scripts/trajectory/real2sim/robolab/make_short_finger_asset.py -- edit that."
    )

    remote_base = stock_usd.rsplit("/", 1)[0]
    swap = {
        "/panda/panda_leftfinger/visuals":
            Sdf.Reference("./Props/short_finger_left.usd", "/Root"),
        "/panda/panda_rightfinger/visuals":
            Sdf.Reference("./Props/short_finger_right.usd", "/Root"),
        "/panda/panda_leftfinger/collisions":
            Sdf.Reference("./Props/short_finger_collisions.usd",
                          "/panda_leftfinger_collisions"),
        "/panda/panda_rightfinger/collisions":
            Sdf.Reference("./Props/short_finger_collisions.usd",
                          "/panda_rightfinger_collisions"),
    }
    rewritten, redirected = 0, 0

    def visit(spec) -> None:
        nonlocal rewritten, redirected
        for child in spec.nameChildren:
            path_str = str(child.path)
            items = child.referenceList.GetAddedOrExplicitItems()
            if items:
                if path_str in swap:
                    child.referenceList.explicitItems = [swap[path_str]]
                    redirected += 1
                else:
                    # Relative ``./Props/foo.usd`` would resolve next to THIS file, where
                    # nothing lives; point each one back at the CDN it came from.
                    child.referenceList.explicitItems = [
                        Sdf.Reference(
                            f"{remote_base}/{ref.assetPath.lstrip('./')}"
                            if ref.assetPath.startswith("./") else ref.assetPath,
                            ref.primPath, ref.layerOffset, ref.customData,
                        )
                        for ref in items
                    ]
                    rewritten += 1
            visit(child)

    visit(layer.pseudoRoot)
    if redirected != len(swap):
        raise SystemExit(
            f"expected to redirect {len(swap)} finger reference arcs, redirected "
            f"{redirected} -- the stock asset's prim layout changed, check {stock_usd}"
        )
    layer.Save()
    print(f"[finger] root layer: {redirected} finger arcs redirected, "
          f"{rewritten} stock arcs pointed at {remote_base}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stl", default=str(ROOT / "assets" / "panda_short_finger.stl"))
    ap.add_argument("--out", default=str(ROOT / "assets" / "robolab_franka"),
                    help="directory for the generated robot asset")
    ap.add_argument("--stock-usd", default=STOCK_PANDA_USD,
                    help="the Panda asset to override (IsaacLab's, by default)")
    args = ap.parse_args()

    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        print(
            "pxr (USD) is not importable. Run this with RoboLab's interpreter and Kit's "
            "USD libraries on the path -- the simplest way is to run it the same way the "
            "generators run, i.e. under $ROBOLAB_ROOT/.venv/bin/python with "
            "Isaac Sim's extscache on PYTHONPATH.",
            file=sys.stderr,
        )
        return 1

    stl = Path(args.stl)
    out = Path(args.out)
    (out / "Props").mkdir(parents=True, exist_ok=True)

    tris, _face_normals = read_binary_stl(stl)
    print(f"[finger] {stl.name}: {len(tris)} triangles, "
          f"bbox mm {tris.reshape(-1, 3).min(0).round(2)} .. "
          f"{tris.reshape(-1, 3).max(0).round(2)}")

    sides = {}
    for prim_name, mirror in (("panda_leftfinger", False), ("panda_rightfinger", True)):
        tip = mesh_arrays(tris, mirror)
        bracket = bracket_arrays(mirror)
        side = "left" if not mirror else "right"
        write_visual_layer(
            out / "Props" / f"short_finger_{side}.usd",
            [
                (prim_name, "YellowTip", (TIP_COLOR, TIP_ROUGHNESS, TIP_METALLIC), *tip),
                (f"{prim_name}_bracket", "Bracket",
                 (BRACKET_COLOR, BRACKET_ROUGHNESS, BRACKET_METALLIC), *bracket),
            ],
        )
        # ONE collider over bracket + tip. PhysX takes the convex hull of it anyway, so
        # handing it two prims would only cost a second hull with the same result.
        col_points = np.vstack([tip[0], bracket[0]])
        col_faces = np.vstack([tip[1], bracket[1] + len(tip[0])])
        col_normals = np.vstack([tip[2], bracket[2]])
        sides[f"{prim_name}_collisions"] = (col_points, col_faces, col_normals)
        inner = np.abs(col_points[:, 1]).min() * 1000
        print(f"[finger] {prim_name}: tip z "
              f"{tip[0][:, 2].min()*1000:.1f}..{tip[0][:, 2].max()*1000:.1f} mm | "
              f"bracket z 0.0..{BRACKET_LENGTH_MM:.1f} mm | "
              f"innermost |y| {inner:.2f} mm -> closed gap {2*inner:.2f} mm")

    write_collision_layer(out / "Props" / "short_finger_collisions.usd", sides)
    # .usda on purpose: the root layer is 40 lines of composition arcs and is the piece a
    # human reads when the fingers come out in the wrong place. The meshes stay binary.
    write_root_layer(out / "panda_short_finger.usda", args.stock_usd, out / ".cache")
    print(f"[finger] wrote {out / 'panda_short_finger.usda'}")
    print("[finger] point core.sim.robolab_franka.FrankaPandaCfg at it, then RE-MEASURE "
          "tasks.FLANGE_TO_FINGERTIP_M -- the fingertip moved ~16 mm up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
