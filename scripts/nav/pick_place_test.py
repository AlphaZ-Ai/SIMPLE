# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Dex1 pick and place at the table, standing at the task's stance (no navigation):
tuck arms, stand, then simple.nav.pick_place. MuJoCo only; writes a video.

    .venv/bin/python scripts/nav/pick_place_test.py --seed 0
"""

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import transforms3d as t3d  # noqa: E402
import typer  # noqa: E402

import simple.envs  # noqa: E402,F401
from simple.core.action import ActionCmd  # noqa: E402
from simple.nav.pick_place import PickPlace  # noqa: E402
from simple.robots.g1_dex1_wholebody import TUCK_ARM_POSE  # noqa: E402


def main(seed: int = 0, out: str = "data/output/pick_place", video: bool = True, target: str = "primitive:cube_0.05", back_off: float = 0.3):
    os.makedirs(out, exist_ok=True)
    env = gym.make("simple/G1WholebodyTabletopGraspMP-v0", robot_uid="g1_dex1_wholebody", render_hz=50,
                   sim_mode="mujoco", headless=True, max_episode_steps=10**9, target_object=target)
    task = env.unwrapped.task
    task.scene_collision = True
    np.random.seed(seed)
    env.reset(seed=seed)
    sim = env.unwrapped.mujoco
    robot = task.robot
    m, d = robot.mjmodel, robot.mjdata
    obj_body = m.body(sim.mj_objects["target"].id).name
    table = m.geom("table_geom").id

    frames, contacts_log = [], set()
    renderer = mujoco.Renderer(m, 360, 640) if video else None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation, cam.azimuth = 1.6, -25, 210

    def record(k):
        for c in d.contact[:d.ncon]:
            g = {c.geom1, c.geom2}
            if table in g:
                other = (g - {table}).pop()
                b = m.body(m.geom_bodyid[other]).name
                if b not in (obj_body,) and not b.startswith("distractor"):
                    contacts_log.add(b)
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render())

    stand = ActionCmd("loco_command", command=[0, 0, 0, 0, 0, 0, 0, 0], motion_type="stand")
    if back_off:
        # arrive like navigation does: a bit back from the task's stance, arms tucked
        yaw = t3d.euler.quat2euler(d.qpos[3:7])[2]
        d.qpos[0:2] -= back_off * np.array([np.cos(yaw), np.sin(yaw)])
        mujoco.mj_forward(m, d)
    robot.set_arm_targets(TUCK_ARM_POSE)
    k = 0
    for _ in range(75):  # 1.5 s: settle with arms tucked
        env.step(stand)
        record(k); k += 1

    pp = PickPlace(robot, obj_body)
    while not pp.done:
        env.step(pp.step())
        record(k); k += 1
        if robot.get_robot_pose()[2] < -0.3:  # pelvis dropped (world z of standing pelvis is ~0 here)
            print("FELL during", pp.phases[min(pp.i, len(pp.phases) - 1)].name)
            break

    r = pp.result
    print("\n".join(r.log))
    print(json.dumps({"object": obj_body, "lifted": r.lifted, "lift_height": round(r.lift_height, 3),
                      "placed": r.placed, "place_error": round(r.place_error, 3), "success": r.success,
                      "ik_failures": r.ik_failures, "robot_touched_table": sorted(contacts_log)}, indent=1))
    if frames:
        imageio.mimwrite(f"{out}/pick_place_seed{seed}.mp4", frames, fps=25, quality=7, macro_block_size=8)
        print("video ->", f"{out}/pick_place_seed{seed}.mp4")
    env.close()


if __name__ == "__main__":
    typer.run(main)
