# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Show the G1 with Dex1 grippers (`g1_dex1_wholebody`) standing, raising its arms
and opening/closing both grippers.

    # IsaacSim viewer
    python scripts/demo_g1_dex1.py --no-headless

    # headless, writes videos to data/output/demo_g1_dex1
    python scripts/demo_g1_dex1.py

    # MuJoCo only, live in the MuJoCo viewer (no IsaacSim needed)
    python scripts/demo_g1_dex1.py --sim-mode mujoco --viewer

    # walk instead: stand, turn left, walk ~1 m, turn back
    python scripts/demo_g1_dex1.py --routine walk
"""

import os
import time
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import typer
from typing_extensions import Annotated

import simple.envs  # noqa: F401  registers envs
from simple.core.action import ActionCmd
from simple.envs.wrappers import VideoRecorder
from simple.robots.g1_dex1_wholebody import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    WAIST_JOINTS,
)

# arms raised in front of the chest, grippers facing each other
ARMS_UP = {
    "left_shoulder_pitch_joint": -0.9, "left_shoulder_roll_joint": 0.25, "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": -0.9, "right_shoulder_roll_joint": -0.25, "right_elbow_joint": 0.6,
}


def stand_cmd():
    # vx, target_yaw, vy, d_height, roll, pitch, yaw, turning_flag
    return ActionCmd("loco_command", command=[0, 0, 0, 0, 0, 0, 0, 0], motion_type="stand")


def heading_cmd(vx: float, yaw: float):
    # turning_flag=1: command[1] is an absolute heading that AMO holds (see TurnSpec in agents/mp.py)
    return ActionCmd("loco_command", command=[vx, yaw, 0, 0, 0, 0, 0, 1], motion_type="stand")


def gripper_routine(t: int):
    """Stand, raise both arms, then alternate open / closed grippers."""
    stand_steps, raise_steps, cycle = 60, 50, 100
    if t < stand_steps:
        return stand_cmd()
    k = t - stand_steps
    alpha = min(1.0, k / raise_steps)
    if k < raise_steps:
        grip = 0.0
    else:  # alternate open / closed every half cycle
        grip = GRIPPER_OPEN if ((k - raise_steps) // (cycle // 2)) % 2 == 0 else GRIPPER_CLOSED
    return move_cmd(alpha, grip)


# (phase, steps) at 50 Hz; turns ramp the heading over 70 steps then hold, as TurnSpec does
WALK_HEADING = np.pi / 2  # robot's left, the open side of the default scene
WALK_PHASES = [("stand", 60), ("turn_left", 100), ("walk", 300), ("turn_back", 100), ("stop", 50)]


def walk_routine(t: int):
    """Stand, turn 90 deg left, walk forward ~1 m, turn back, stop."""
    for phase, n in WALK_PHASES:
        if t < n:
            break
        t -= n
    ramp = min(1.0, t / 69)
    if phase == "stand":
        return stand_cmd()
    if phase == "turn_left":
        return heading_cmd(0.0, WALK_HEADING * ramp)
    if phase == "walk":
        # AMO reaches ~40% of the commanded speed here: 0.5 -> ~0.2 m/s
        return heading_cmd(0.5, WALK_HEADING)
    if phase == "turn_back":
        return heading_cmd(0.0, WALK_HEADING * (1 - ramp))
    return heading_cmd(0.0, 0.0)


ROUTINES = {
    "grippers": (gripper_routine, 600),
    "walk": (walk_routine, sum(n for _, n in WALK_PHASES)),
}


def move_cmd(arm_alpha: float, gripper_q: float):
    # move_qpos expects the 3 waist joints first, then arm and hand joints
    target = {j: 0.0 for j in WAIST_JOINTS}
    for j in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        target[j] = arm_alpha * ARMS_UP.get(j, 0.0)
    for j in LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS:
        target[j] = gripper_q
    return ActionCmd("move_qpos", target_qpos=target)


def main(
    env_id: str = "simple/G1WholebodyBendPickMP-v0",
    sim_mode: Annotated[str, typer.Option(help="mujoco_isaac (render in IsaacSim) or mujoco")] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    routine: Annotated[str, typer.Option(help="grippers or walk")] = "grippers",
    steps: Annotated[int, typer.Option(help="control steps at 50 Hz (default: whole routine)")] = 0,
    save_dir: Annotated[str, typer.Option()] = "data/output/demo_g1_dex1",
    viewer: Annotated[bool, typer.Option(help="show the MuJoCo physics live in the MuJoCo viewer")] = False,
):
    policy, routine_steps = ROUTINES[routine]
    steps = steps or routine_steps
    env = gym.make(
        env_id,
        robot_uid="g1_dex1_wholebody",
        render_hz=50,  # wholebody tasks require render/physics parity (AMO runs at 50 Hz)
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=steps + 100,
    )
    record = headless and not viewer
    if record:
        os.makedirs(save_dir, exist_ok=True)
        env = VideoRecorder(env=env, video_folder=save_dir)

    env.reset()
    sim_app = getattr(env.unwrapped, "simulation_app", None) if not headless else None
    robot = env.unwrapped.task.robot

    mj_viewer = None
    if viewer:
        import mujoco.viewer
        # the robot steps these directly (AMO + PD loop), so the viewer shows live physics
        mj_viewer = mujoco.viewer.launch_passive(robot.mjmodel, robot.mjdata)
    control_dt = 1.0 / 50

    for t in range(steps):
        tic = time.perf_counter()
        if mj_viewer is not None and not mj_viewer.is_running():
            break
        if sim_app is not None:
            if not sim_app.is_running():
                break
            sim_app.update()

        env.step(policy(t))
        if mj_viewer is not None:
            mj_viewer.sync()
            time.sleep(max(0.0, control_dt - (time.perf_counter() - tic)))  # real time

        if t % 50 == 0:
            q = robot.get_robot_qpos()
            pose = robot.get_robot_pose()
            fingers = [round(q[j], 4) for j in RIGHT_HAND_JOINTS]
            print(f"step {t:4d}  base xyz={np.round(pose[:3], 2)}  right fingers={fingers}")

    if mj_viewer is not None:
        mj_viewer.close()
    env.close()
    if record:
        print(f"videos written to {save_dir}")


if __name__ == "__main__":
    typer.run(main)
