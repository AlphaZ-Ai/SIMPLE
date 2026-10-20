# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Build MuJoCo collision boxes for HSSD house scenes (walls + furniture).

HSSD houses are only loaded in IsaacSim, which renders but does not simulate;
MuJoCo (the physics) only has a ground plane. This script turns each house into
a set of axis-aligned boxes that MuJoCo can collide with (and that a simulated
lidar can hit), placed with the same transform IsaacSim uses.

Method: sample points on every wall and furniture triangle, keep points between
`--z-min` and `--z-max` above the floor, bin them into a `--cell` grid, and
merge occupied cells of similar height into rectangles. Each rectangle becomes
a box from the floor up to its tallest point. Doors/windows (`openings`) are
left out so doorways stay passable, and the task's own table surface is skipped
because MuJoCo already models it as a box.

Output per scene uid (frame "G": scene file scaled to meters, then rotated by
the x/y part of `center_orientation`, so z is up and the floor is z=0):

    data/scenes/hssd_collision/<uid>.npz
        boxes        (N, 6) float32   center xyz, half size xyz  (frame G)
        surface_top  (3,)   float64   surface OBB top center, scene file frame
        rot_xy       (2,)   float64   x, y of center_orientation (deg) used for G
        floor        float            floor height in frame G before shifting to 0

Runtime (simple/engines/mujoco.py) places frame G with yaw = center_orientation[2]
and translation T = center_offset - R * S * surface_top, matching
IsaacSim's add_scene / calc_surface_center.

Runs with usd-core in an isolated env (IsaacSim's pxr needs Kit):

    uvx --python 3.10 --from usd-core --with numpy --with scipy --with pyyaml \\
        python scripts/build_hssd_collision.py --uids scene0
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import yaml
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/simple/resources/hssd-scenes/config.yaml"
OUT_DIR = ROOT / "data/scenes/hssd_collision"

COLLIDE_GROUPS = ("walls", "furniture")


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def surface_top(stage, prim_path):
    """Mirror IsaacSim calc_surface_center (scene untransformed): OBB centroid
    pushed to the +axis1 face, i.e. mean of corners 2, 3, 6, 7."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path))
    centroid = np.array(bound.ComputeCentroid())
    axes = np.array(bound.GetMatrix().ExtractRotationMatrix())
    half = np.array(bound.GetRange().GetSize()) * 0.5
    return centroid + axes[1] * half[1]


def mesh_triangles(prim):
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    pts = pts @ xf[:3, :3] + xf[3, :3]  # USD matrices are row-vector
    idx = np.asarray(idx)
    tris, start = [], 0
    for c in counts:  # fan triangulation
        f = idx[start:start + c]
        for k in range(1, c - 1):
            tris.append((f[0], f[k], f[k + 1]))
        start += c
    return pts[np.asarray(tris)] if tris else None


def sample_triangles(tris, spacing):
    """Points on each triangle on a barycentric lattice no coarser than `spacing`."""
    out = [tris.reshape(-1, 3)]
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    longest = np.maximum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1))
    longest = np.maximum(longest, np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1))
    n = np.clip(np.ceil(longest / spacing).astype(int), 1, 400)
    for k in np.unique(n):
        sel = n == k
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (u + v) <= k
        u, v = u[keep] / k, v[keep] / k
        p = tris[sel, 0][:, None] + e1[sel][:, None] * u[None, :, None] + e2[sel][:, None] * v[None, :, None]
        out.append(p.reshape(-1, 3))
    return np.concatenate(out)


def merge_rectangles(height, occupied, h_step):
    """Greedy merge of occupied cells with the same quantized height into rectangles.
    Returns list of (i0, j0, i1, j1, h) with inclusive-exclusive cell bounds."""
    q = np.where(occupied, np.ceil(height / h_step).astype(int), 0)
    used = ~occupied
    rects = []
    nx, ny = q.shape
    for i in range(nx):
        for j in range(ny):
            if used[i, j]:
                continue
            hq = q[i, j]
            j1 = j
            while j1 < ny and not used[i, j1] and q[i, j1] == hq:
                j1 += 1
            i1 = i + 1
            while i1 < nx and np.all(~used[i1, j:j1]) and np.all(q[i1, j:j1] == hq):
                i1 += 1
            used[i:i1, j:j1] = True
            rects.append((i, j, i1, j1, hq * h_step))
    return rects


def build(uid, conf, cell, z_min, z_max, h_step, min_area):
    usd = ROOT / "data" / conf["data_dir"] / conf["name"] / f"{conf['name']}.usd"
    stage = Usd.Stage.Open(str(usd))
    world = stage.GetDefaultPrim()
    scale = float(conf["scale"])
    rot_xy = np.array(conf["center_orientation_limit_up"][:2], dtype=np.float64)
    assert np.allclose(rot_xy, conf["center_orientation_limit_down"][:2]), "only yaw may be randomized"
    R_g = rot_y(rot_xy[1]) @ rot_x(rot_xy[0])

    surface_path = conf["surface"]["prim_path"]
    skip = [surface_path]
    if conf.get("surface2"):
        skip.append(conf["surface2"]["prim_path"])

    pts = []
    for group in COLLIDE_GROUPS:
        g = world.GetChild(group)
        if not g:
            continue
        for prim in Usd.PrimRange(g, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if any(path.startswith(s) for s in skip):
                continue
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
                tris = mesh_triangles(prim)
                if tris is not None:
                    pts.append(sample_triangles(tris * scale, cell / 2) @ R_g.T)
    pts = np.concatenate(pts)

    # floor = lowest wall point (walls start at the floor)
    wall_pts = []
    for prim in Usd.PrimRange(world.GetChild("walls"), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            t = mesh_triangles(prim)
            if t is not None:
                wall_pts.append((t.reshape(-1, 3) * scale) @ R_g.T)
    floor = float(np.concatenate(wall_pts)[:, 2].min())
    pts[:, 2] -= floor

    band = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]
    lo = band[:, :2].min(axis=0) - cell
    ij = np.floor((band[:, :2] - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 2
    height = np.zeros(shape)
    np.maximum.at(height, (ij[:, 0], ij[:, 1]), band[:, 2])
    occupied = height > 0

    # drop small isolated clutter (decor, small objects on the floor): walls and
    # furniture are connected regions far larger than this
    labels, n = ndimage.label(occupied, structure=np.ones((3, 3)))
    area = ndimage.sum(occupied, labels, index=np.arange(1, n + 1)) * cell * cell
    small = np.isin(labels, np.flatnonzero(area < min_area) + 1)
    occupied &= ~small
    height[~occupied] = 0

    rects = merge_rectangles(height, occupied, h_step)
    boxes = []
    for i0, j0, i1, j1, h in rects:
        h = min(h, z_max)
        cx, cy = lo + (np.array([i0 + i1, j0 + j1]) / 2.0) * cell
        boxes.append([cx, cy, h / 2, (i1 - i0) * cell / 2, (j1 - j0) * cell / 2, h / 2])
    boxes = np.asarray(boxes, dtype=np.float32)

    top = surface_top(stage, surface_path) * scale
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{uid}.npz"
    np.savez_compressed(out, boxes=boxes, surface_top=top, rot_xy=rot_xy, floor=floor,
                        cell=cell, scale=scale, name=conf["name"])
    print(f"{uid} ({conf['name']}): {len(boxes)} boxes from {occupied.sum()} cells ({small.sum()} clutter cells dropped), "
          f"floor={floor:.3f}, extent={shape * cell} m -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="*", help="scene uids from hssd-scenes/config.yaml (default: all)")
    ap.add_argument("--cell", type=float, default=0.05, help="grid cell size (m)")
    ap.add_argument("--z-min", type=float, default=0.05, help="ignore geometry below this height (rugs)")
    ap.add_argument("--z-max", type=float, default=2.0, help="ignore geometry above this height (lamps)")
    ap.add_argument("--h-step", type=float, default=0.1, help="box height quantization (m)")
    ap.add_argument("--min-area", type=float, default=0.05, help="drop isolated clutter smaller than this footprint (m^2)")
    args = ap.parse_args()

    scenes = {s["uid"]: s for s in yaml.safe_load(CONFIG.read_text())}
    for uid in args.uids or scenes:
        build(uid, scenes[uid], args.cell, args.z_min, args.z_max, args.h_step, args.min_area)


if __name__ == "__main__":
    main()
