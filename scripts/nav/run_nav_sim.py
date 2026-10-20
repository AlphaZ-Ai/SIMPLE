# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Run the SIMPLE navigation simulator (G1 + HSSD house with wall/furniture
collision + simulated Mid-360) and serve it to a bridge client.

    python scripts/nav/run_nav_sim.py                       # MuJoCo only, headless
    python scripts/nav/run_nav_sim.py --sim-mode mujoco_isaac --no-headless   # watch in IsaacSim

Then start the ROS2 bridge (scripts/nav/ros2_bridge_node.py) or the test client
(scripts/nav/goal_client_test.py).
"""

import typer
from typing_extensions import Annotated

from simple.nav.sim_server import NavSimServer


def main(
    env_id: str = "simple/G1WholebodyBendPickMP-v0",
    robot_uid: str = "g1_dex1_wholebody",
    scene_uid: Annotated[str, typer.Option(help="e.g. hssd:scene0 (collision built by scripts/build_hssd_collision.py)")] = "hssd:scene0",
    sim_mode: Annotated[str, typer.Option(help="mujoco or mujoco_isaac")] = "mujoco",
    headless: Annotated[bool, typer.Option()] = True,
    port: int = 5555,
    authkey: str = "simple-nav",
    seed: int = 0,
    spawn: Annotated[tuple[float, float, float], typer.Option(help="start pose x y yaw_deg (world); default: task spawn")] = (None, None, None),
    log_path: Annotated[str, typer.Option(help="save a per-step run log (.npz) for scripts/nav/render_nav_run.py")] = "",
    goal: Annotated[tuple[float, float], typer.Option(help="goal x y, only for success detection in the log")] = (None, None),
    mission: Annotated[str, typer.Option(help="'pick_place': navigate to the task's table, then pick and place")] = "",
    target_object: Annotated[str, typer.Option(help="task target object, e.g. primitive:cube_0.05")] = "",
):
    spawn = None if spawn[0] is None else spawn
    goal = None if goal[0] is None else goal
    env_kwargs = {"target_object": target_object} if target_object else {}
    NavSimServer(env_id, robot_uid, scene_uid, sim_mode, headless, port, authkey.encode(), seed, spawn,
                 log_path or None, goal, env_kwargs=env_kwargs, mission=mission or None).run()


if __name__ == "__main__":
    typer.run(main)
