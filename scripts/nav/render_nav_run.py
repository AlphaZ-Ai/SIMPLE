# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Re-render a logged navigation run (scripts/nav/run_nav_sim.py --log-path ...)
in IsaacSim, plus a top-down map of the path.

IsaacSim in SIMPLE only mirrors MuJoCo's state, so replaying the logged MuJoCo
qpos step by step reproduces the run exactly; nothing is re-simulated. The
video has a third-person chase camera (the task's fixed front camera would
lose the robot once it leaves the room) next to the robot's head camera.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/nav/render_nav_run.py data/output/nav_runs/run.npz
"""

import json
import os
import random

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import matplotlib  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import transforms3d as t3d  # noqa: E402
import typer  # noqa: E402
from typing_extensions import Annotated  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import simple.envs  # noqa: E402,F401

CHASE_CAM = "front_stereo_left"
HEAD_CAM = "head_stereo_left"


def make_env(meta: dict, sim_mode: str):
    """Rebuild the scene exactly as NavSimServer did (same seed -> same placement)."""
    np.random.seed(meta["seed"])
    random.seed(meta["seed"])
    env = gym.make(meta["env_id"], robot_uid=meta["robot_uid"], render_hz=50, sim_mode=sim_mode,
                   headless=True, max_episode_steps=10**9, **meta.get("env_kwargs", {}))
    task = env.unwrapped.task
    task.scene_collision = True
    task.dr_cfgs["scene"].room_choices = [meta["scene_uid"]]
    env.reset(seed=meta["seed"])
    return env


def plot_map(env, log, out_png, title):
    sim = env.unwrapped.mujoco
    m, d = sim.mjModel, sim.mjData
    mujoco.mj_forward(m, d)
    fig, ax = plt.subplots(figsize=(8, 9))
    sb = m.body("scene_collision").id
    for i in range(m.ngeom):
        if m.geom_bodyid[i] != sb:
            continue
        c, s, R = d.geom_xpos[i], m.geom_size[i], d.geom_xmat[i].reshape(3, 3)
        corners = [c + R @ [sx * s[0], sy * s[1], 0] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        ax.add_patch(Polygon([p[:2] for p in corners], color=plt.cm.Greys(0.35 + 0.5 * min(1.0, s[2])), lw=0))
    xy = log["qpos"][:, :2]
    contact = np.array([len(json.loads(c)) > 0 for c in log["contacts"]])
    ax.plot(xy[:, 0], xy[:, 1], "-", color="tab:blue", lw=2, label="robot path")
    ax.plot(*xy[0], "o", color="tab:green", ms=10, label="start")
    ax.plot(*xy[-1], "s", color="tab:purple", ms=9, label="end")
    if contact.any():
        ax.plot(xy[contact, 0], xy[contact, 1], ".", color="tab:red", ms=4, label="touching house")
    goal = log["meta"].get("goal")
    if goal:
        ax.plot(*goal, "*", color="gold", mec="k", ms=18, label="goal")
    wps = [json.loads(e)[1] for e in log["events"]]
    wxy = np.array([[float(v) for v in w.split("=")[1].strip("()").split(",")] for w in wps if w.startswith("waypoint=")])
    if len(wxy):
        ax.plot(wxy[:, 0], wxy[:, 1], "x", color="tab:orange", ms=7, label="FAR waypoints")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.autoscale()
    plt.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


CHASE_DIST = 2.4
HOUSE_GROUP = 5  # geom group the house boxes are moved to, so rays test only the house


def chase_clearance(model, data, base_xy, heading, floor_z, height=1.9):
    """How far behind the robot the chase camera can sit before a wall is in the way."""
    start = np.array([base_xy[0], base_xy[1], floor_z + height])
    back = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    groups = np.zeros(6, dtype=np.uint8)
    groups[HOUSE_GROUP] = 1
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, start, back, groups, 1, -1, gid)
    return CHASE_DIST if dist < 0 else dist


def chase_pose(base_xy, heading, floor_z, dist=CHASE_DIST, height=1.9, look_height=0.8):
    eye = np.array([base_xy[0] - dist * np.cos(heading), base_xy[1] - dist * np.sin(heading), floor_z + height])
    target = np.array([base_xy[0], base_xy[1], floor_z + look_height])
    v = target - eye
    yaw = np.arctan2(v[1], v[0])
    pitch = np.arctan2(-v[2], np.hypot(v[0], v[1]))  # +pitch about y looks down in "world" camera axes
    return eye, t3d.euler.euler2quat(0, pitch, yaw)  # wxyz


def main(
    log_path: str,
    fps: Annotated[int, typer.Option(help="output video frame rate (log is 50 Hz)")] = 25,
    out: Annotated[str, typer.Option(help="output prefix (default: next to the log)")] = "",
    map_only: Annotated[bool, typer.Option(help="only draw the top-down map (no IsaacSim)")] = False,
):
    raw = np.load(log_path, allow_pickle=False)
    log = {k: raw[k] for k in raw.files}
    log["meta"] = json.loads(str(raw["meta"]))
    meta = log["meta"]
    prefix = out or os.path.splitext(log_path)[0]
    result = meta.get("result", "")

    env = make_env(meta, "mujoco" if map_only else "mujoco_isaac")
    plot_map(env, log, prefix + "_map.png",
             f"{meta['scene_uid']} seed={meta['seed']}  {result}".strip())
    print(f"map -> {prefix}_map.png", flush=True)
    if map_only:
        env.close()
        return

    sim, isaac = env.unwrapped.mujoco, env.unwrapped.isaac
    floor_z = float(sim.scene_floor_z)

    # the task only lights the area around its table; carry lights with the robot
    # so the other rooms it walks through are visible
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux
    stage = omni.usd.get_context().get_stage()
    follow_lights = []
    for name, intensity in (("nav_light_robot", 60000.0), ("nav_light_ahead", 30000.0)):
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(0.4)
        light.CreateTreatAsPointAttr(False)
        follow_lights.append(UsdGeom.XformCommonAPI(light.GetPrim()))
    stride = max(1, round(50 / fps))
    heading = None
    cam_dist = CHASE_DIST
    m = sim.mjModel
    house = m.body("scene_collision").id
    m.geom_group[m.geom_bodyid == house] = HOUSE_GROUP
    frames = []
    for k in range(0, len(log["t"]), stride):
        sim.mjData.qpos[:] = log["qpos"][k]
        sim.mjData.qvel[:] = 0
        mujoco.mj_forward(sim.mjModel, sim.mjData)
        yaw = t3d.euler.quat2euler(log["qpos"][k][3:7])[2]
        # smooth the chase heading so the camera does not shake with every step
        heading = yaw if heading is None else heading + 0.1 * ((yaw - heading + np.pi) % (2 * np.pi) - np.pi)
        # pull the camera in front of walls between it and the robot; ease back out
        free = chase_clearance(m, sim.mjData, log["qpos"][k][:2], heading, floor_z) - 0.3
        want = float(np.clip(free, 0.6, CHASE_DIST))
        cam_dist = want if want < cam_dist else cam_dist + 0.05 * (want - cam_dist)
        eye, quat = chase_pose(log["qpos"][k][:2], heading, floor_z, dist=cam_dist)
        isaac.cameras[CHASE_CAM].set_world_pose(eye, quat, camera_axes="world")
        bx, by = log["qpos"][k][:2]
        follow_lights[0].SetTranslate(Gf.Vec3d(float(bx), float(by), floor_z + 2.3))
        follow_lights[1].SetTranslate(Gf.Vec3d(float(bx + 1.5 * np.cos(heading)), float(by + 1.5 * np.sin(heading)), floor_z + 2.3))
        isaac.step(sim)
        imgs = isaac.render()
        frames.append(np.concatenate([imgs[CHASE_CAM], imgs[HEAD_CAM]], axis=1))
        if len(frames) % 100 == 0:
            print(f"rendered {len(frames)} frames ({log['t'][k]:.0f} s)", flush=True)

    imageio.mimwrite(prefix + "_isaac.mp4", frames, fps=fps, quality=7, macro_block_size=8)
    print(f"video -> {prefix}_isaac.mp4 ({len(frames)} frames, {len(frames) / fps:.0f} s)", flush=True)
    env.close()


if __name__ == "__main__":
    typer.run(main)
