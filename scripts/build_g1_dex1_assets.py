# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Build the G1 (29dof) + Unitree Dex1-1 gripper assets used by `g1_dex1_wholebody`.

SIMPLE needs two models of the same robot: an MJCF for MuJoCo (physics) and a
USD for IsaacSim (rendering), synced by joint name. Unitree only ships the Dex1
G1 as USD (HF dataset unitreerobotics/unitree_sim_isaaclab_usds), so this script:

  1. MJCF: takes SIMPLE's G1 dex3 wholebody model, removes both Dex3 hands and
     attaches Dex1 grippers built from unitree_ros `dex1_1.urdf`, mounted at the
     pose the Unitree USD uses (`*_hand_palm_joint`). Body and joint names match
     the USD (`left_hand_base_link`, `left_hand_Joint1_1`, ...).
  2. USD: writes a thin wrapper that references the Unitree USD under
     /root/<robot_ns>, which is the prim layout SIMPLE's Isaac engine expects.

Inputs:  data/robots/g1/ (SIMPLE pre-download), data/robots/g1-29dof_wholebody_dex1/
         (Unitree USD). Dex1 meshes are fetched from GitHub unless --dex1-dir is given.
Output:  data/robots/g1_dex1/

    python scripts/build_g1_dex1_assets.py
"""

import os
import shutil
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "robots"
SRC_MJCF = DATA / "g1" / "g1_29dof_wholebody_dex3.xml"
SRC_MESHES = DATA / "g1" / "meshes"
UNITREE_USD = DATA / "g1-29dof_wholebody_dex1" / "g1_29dof_with_dex1_rev_1_0.usd"
OUT = DATA / "g1_dex1"
ROBOT_NS = "g1_29dof_wholebody_dex1"

DEX1_URL = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/dexterous_hand_description/dex1_1/meshes"
DEX1_MESHES = ["base_link", "Link1_1", "Link1_2", "Link1_3", "Link2_1", "Link2_2", "Link2_3"]

# Dex1 mount on the wrist (left_/right_hand_palm_joint in the Unitree USD), wxyz.
MOUNT = {
    "left": dict(pos=[0.0415, 0.003, 0.0], quat=[0.70710677, 0, 0, -0.70710677]),
    "right": dict(pos=[0.0415, -0.003, 0.0], quat=[0.7073882, 0, 0, -0.7068252]),
}

# Dex1 links from dex1_1.urdf: parent, pos, inertial (pos, mass, full inertia
# ixx iyy izz ixy ixz iyz), joint (axis) for the two actuated prismatic fingers.
DEX1_BASE = dict(ipos=[0.0001178, 0.036886, 0.0080974], mass=0.11,
                 inertia=[4.9533e-05, 6.8944e-05, 3.4443e-05, 5.9474e-08, -3.1979e-08, -1.7064e-06])
DEX1_LINKS = {
    "Link1_1": dict(parent="base_link", pos=[0.04055, 0.0593, 0.0152], axis=[-1, 0, 0],
                    ipos=[-0.041607, 0.0039411, -0.00047344], mass=0.0084398,
                    inertia=[1.0466e-07, 3.5113e-06, 3.523e-06, 4.1386e-09, 1.086e-09, -1.1358e-09]),
    "Link1_2": dict(parent="Link1_1", pos=[-0.013, 0.013, 0.0],
                    ipos=[0.0046767, 0.026417, -0.014407], mass=0.028354,
                    inertia=[1.3423e-05, 2.249e-06, 1.1892e-05, 3.5258e-07, 5.8181e-08, 6.3947e-07]),
    "Link1_3": dict(parent="Link1_2", pos=[-0.0025165, 0.025043, -0.001],
                    ipos=[0.0018267, 0.017456, -0.0142], mass=0.004583,
                    inertia=[9.6165e-07, 2.5948e-07, 7.1226e-07, 6.207e-09, 0.0, 0.0]),
    "Link2_1": dict(parent="base_link", pos=[-0.04055, 0.0593, -0.0152], axis=[1, 0, 0],
                    ipos=[0.041607, 0.0039411, 0.00047344], mass=0.0084398,
                    inertia=[1.0466e-07, 3.5113e-06, 3.523e-06, -4.1386e-09, 1.086e-09, 1.1358e-09]),
    "Link2_2": dict(parent="Link2_1", pos=[0.013, 0.013, 0.0],
                    ipos=[-0.0046767, 0.026417, 0.014407], mass=0.028354,
                    inertia=[1.3423e-05, 2.249e-06, 1.1892e-05, -3.5258e-07, 5.8181e-08, -6.3947e-07]),
    "Link2_3": dict(parent="Link2_2", pos=[0.0025165, 0.025043, 0.0294],
                    ipos=[-0.0018267, 0.017456, -0.0142], mass=0.004583,
                    inertia=[9.6165e-07, 2.5948e-07, 7.1226e-07, -6.207e-09, 0.0, 0.0]),
}
# dex1_1.urdf limit: -0.02 (wide open) .. 0.0245 (pads touching)
FINGER_RANGE = [-0.02, 0.0245]
FINGER_FORCE = 20.0  # N, dex1_1.urdf effort
LINK_RGBA = {"base_link": [0.79, 0.82, 0.93, 1], "1": [0.79, 0.82, 0.93, 1], "2": [0.9, 0.92, 0.93, 1], "3": [0.3, 0.3, 0.3, 1]}

DEX3_HAND_BODIES = ["thumb_0_link", "middle_0_link", "index_0_link"]


def fetch_dex1_meshes(dst: Path, dex1_dir: Path | None):
    for name in DEX1_MESHES:
        out = dst / f"dex1_{name}.STL"
        if out.exists():
            continue
        if dex1_dir is not None:
            shutil.copy(dex1_dir / f"{name}.STL", out)
        else:
            print(f"downloading {name}.STL")
            urllib.request.urlretrieve(f"{DEX1_URL}/{name}.STL", out)


def full_inertia(i):
    ixx, iyy, izz, ixy, ixz, iyz = i
    return [ixx, iyy, izz, ixy, ixz, iyz]


def add_mesh_geoms(body, mesh, rgba, collide=True):
    vis = body.add_geom()
    vis.type = mujoco.mjtGeom.mjGEOM_MESH
    vis.meshname = mesh
    vis.contype, vis.conaffinity, vis.group, vis.density = 0, 0, 1, 0
    vis.rgba = rgba
    if collide:
        col = body.add_geom()
        col.type = mujoco.mjtGeom.mjGEOM_MESH
        col.meshname = mesh
        col.group, col.density = 3, 0
        col.rgba = rgba
        col.friction = [1.5, 0.01, 0.001]


def build_mjcf(spec: mujoco.MjSpec):
    spec.modelname = ROBOT_NS
    spec.meshdir = "meshes"
    spec.modelfiledir = str(OUT)  # resolve meshes against the output dir

    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        # Dex3 palm geoms live directly on the wrist link; drop them
        for g in [g for g in wrist.geoms if g.meshname == f"{side}_hand_palm_link"]:
            spec.delete(g)
        for b in DEX3_HAND_BODIES:
            spec.delete(spec.body(f"{side}_hand_{b}"))

        base = wrist.add_body()
        base.name = f"{side}_hand_base_link"
        base.pos, base.quat = MOUNT[side]["pos"], MOUNT[side]["quat"]
        base.ipos, base.mass = DEX1_BASE["ipos"], DEX1_BASE["mass"]
        base.fullinertia = full_inertia(DEX1_BASE["inertia"])
        base.explicitinertial = True
        add_mesh_geoms(base, "dex1_base_link", LINK_RGBA["base_link"])

        bodies = {"base_link": base}
        for lname, cfg in DEX1_LINKS.items():
            b = bodies[cfg["parent"]].add_body()
            b.name = f"{side}_hand_{lname}"
            b.pos = cfg["pos"]
            b.ipos, b.mass = cfg["ipos"], cfg["mass"]
            b.fullinertia = full_inertia(cfg["inertia"])
            b.explicitinertial = True
            add_mesh_geoms(b, f"dex1_{lname}", LINK_RGBA[lname[-1]])
            if "axis" in cfg:
                j = b.add_joint()
                j.name = f"{side}_hand_Joint{lname[4:]}"
                j.type = mujoco.mjtJoint.mjJNT_SLIDE
                j.axis = cfg["axis"]
                j.range = FINGER_RANGE
                j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                j.actfrcrange = [-FINGER_FORCE, FINGER_FORCE]
                j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
                j.damping = 5.0
                j.armature = 0.01
            bodies[lname] = b

        # finger segments overlap the gripper base, the wrist and each other within a
        # finger; only finger-vs-finger and gripper-vs-world contacts should remain
        wrist_name, base_name = f"{side}_wrist_yaw_link", f"{side}_hand_base_link"
        links = [f"{side}_hand_{n}" for n in DEX1_LINKS]
        pairs = [(wrist_name, base_name)]
        pairs += [(p, l) for l in links for p in (wrist_name, base_name)]
        pairs += [(a, b) for i, a in enumerate(links) for b in links[i + 1:] if a[-3] == b[-3]]
        for a, b in pairs:
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = a, b

    for name in DEX1_MESHES:
        m = spec.add_mesh()
        m.name = f"dex1_{name}"
        m.file = f"dex1_{name}.STL"

    # swap Dex3 hand motors for Dex1 finger motors (torque; PD runs in python like Dex3)
    for act in [a for a in spec.actuators if "_hand_" in a.name]:
        spec.delete(act)
    for side in ("left", "right"):
        for f in ("Joint1_1", "Joint2_1"):
            a = spec.add_actuator()
            a.name = a.target = f"{side}_hand_{f}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gainprm[0] = 1.0
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = [-FINGER_FORCE, FINGER_FORCE]

    # sensors on removed Dex3 joints/bodies would dangle
    for s in list(spec.sensors):
        if "_hand_" in (s.objname or ""):
            spec.delete(s)
    # unused Dex3 meshes
    for m in [m for m in spec.meshes if "_hand_" in (m.name or m.file) and not m.file.startswith("dex1")]:
        spec.delete(m)


def write_usd_wrapper(path: Path):
    rel = os.path.relpath(UNITREE_USD, path.parent)
    path.write_text(
        "#usda 1.0\n"
        '(\n    defaultPrim = "root"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "root"\n{\n'
        f'    def Xform "{ROBOT_NS}" (\n        prepend references = @{rel}@\n    )\n    {{\n    }}\n'
        "}\n"
    )


def main(dex1_dir: Path | None = None):
    assert SRC_MJCF.exists(), f"missing {SRC_MJCF}; run data/_download.sh --only robots"
    assert UNITREE_USD.exists(), f"missing {UNITREE_USD}; download unitreerobotics/unitree_sim_isaaclab_usds"

    meshes = OUT / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for f in SRC_MESHES.glob("*.STL"):
        if "_hand_" not in f.name and not (meshes / f.name).exists():
            shutil.copy(f, meshes / f.name)
    fetch_dex1_meshes(meshes, dex1_dir)

    spec = mujoco.MjSpec.from_file(str(SRC_MJCF))
    build_mjcf(spec)
    xml_path = OUT / f"{ROBOT_NS}.xml"
    xml_path.write_text(spec.to_xml())

    # sanity: compiles from disk and has the expected joints
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    jnames = [model.joint(i).name for i in range(model.njnt)]
    fingers = [j for j in jnames if "_hand_Joint" in j]
    assert len(fingers) == 4, fingers
    print(f"wrote {xml_path} ({model.njnt} joints, {model.nu} actuators, fingers={fingers})")

    usd_path = OUT / f"{ROBOT_NS}.usda"
    write_usd_wrapper(usd_path)
    print(f"wrote {usd_path}")


if __name__ == "__main__":
    typer.run(main)
