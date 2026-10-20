# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Walk the G1 with the SONIC controller's locomotion planner, driven from code
(no headset): stand, walk forward, turn left, walk, stop. Logs base pose so we
can see whether velocity commands turn into the expected motion.

    .venv/bin/python scripts/nav/sonic_walk_test.py
"""

import math
import os
import signal
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SONIC_AUTO_START", "1")
os.environ.setdefault("G1_ELBOW_POSE", "down")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import transforms3d as t3d  # noqa: E402
import typer  # noqa: E402

import simple.envs  # noqa: E402,F401
from simple.cli.teleop_wbc import _load_sonic_config  # noqa: E402
from simple.nav.sonic_agent import NavSonicAgent, controller_command, log_ready_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# (name, seconds, vx, vy, wz) in the robot body frame
SCRIPT = [
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("walk forward 0.4 m/s", 5.0, 0.4, 0.0, 0.0),
    ("turn left in place", 4.0, 0.0, 0.0, 0.4),
    ("walk forward 0.4 m/s", 5.0, 0.4, 0.0, 0.0),
    ("stop", 3.0, 0.0, 0.0, 0.0),
]


def pose_xy_yaw(robot):
    q = robot.mjData.qpos
    return q[0], q[1], q[2], math.degrees(t3d.euler.quat2euler(q[3:7])[2])


def main(env_id: str = "simple/G1WholebodyXMoveBendCarryBoxSonic-v0", log_dir: str = "data/output/sonic_walk",
         spawn: tuple[float, float, float] = (-2.5, 0.0, 180.0)):
    logs = ROOT / log_dir
    logs.mkdir(parents=True, exist_ok=True)
    ctrl_log = open(logs / "controller.log", "w")
    ctrl = subprocess.Popen(controller_command(ROOT, logs / "controller_runtime"), stdout=ctrl_log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    print(f"[sonic_walk] controller pid {ctrl.pid}, log {logs / 'controller.log'}", flush=True)

    def _term(signum, frame):
        raise KeyboardInterrupt  # so the finally block stops the controller
    signal.signal(signal.SIGTERM, _term)

    sonic_config = _load_sonic_config()
    env = gym.make(env_id, sim_mode="mujoco", render_hz=50, physics_dt=sonic_config["SIMULATE_DT"],
                   headless=True, max_episode_steps=10**9, sonic_config=sonic_config)
    sonic_env = env.unwrapped
    robot = sonic_env.task.robot
    agent = NavSonicAgent(robot, controller_ready=log_ready_check(logs / "controller.log"))
    try:
        observation, info = env.reset()
        if spawn:
            # move the robot clear of the task's table (it spawns facing a box on it)
            import mujoco
            x, y, yaw_deg = spawn
            robot.mjData.qpos[0:2] = [x, y]
            robot.mjData.qpos[3:7] = t3d.euler.euler2quat(0, 0, math.radians(yaw_deg))
            robot.mjData.qvel[:6] = 0
            mujoco.mj_forward(robot.mjModel, robot.mjData)
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False  # the stabilization WBC holds the robot
        agent.reset_policy()
        sonic_env.substep_callback = lambda: agent.cache_proprio(robot.prepare_obs())
        agent.start_state_heartbeat()

        dt = 0.02
        t0 = time.monotonic()
        qpos_log = []
        phase_start, phase_idx, last_print = None, 0, 0.0
        while phase_idx < len(SCRIPT):
            tick = time.monotonic()
            if ctrl.poll() is not None:
                raise RuntimeError(f"controller exited with {ctrl.returncode}; see {logs / 'controller.log'}")
            action = agent.get_action(observation, privileged_info=info)
            observation, _, _, _, info = env.step(action)
            qpos_log.append((time.monotonic() - t0, robot.mjData.qpos.copy(), agent.handed_over))

            if robot.stabilized and not agent.planner_engaged and agent.unitree_bridge.cmd_received():
                agent.engage_planner()
                print(f"[sonic_walk] t={tick - t0:.1f}s stabilized, controller up -> PLANNER engaged", flush=True)
            if agent.planner_engaged and not agent.handed_over:
                agent.walk(0.0, 0.0, 0.0, dt)  # keep the planner alive (1 s timeout) while it takes over
            if agent.handed_over and phase_start is None:
                phase_start = time.monotonic()
            if phase_start is not None:
                name, dur, vx, vy, wz = SCRIPT[phase_idx]
                agent.walk(vx, vy, wz, dt)
                if time.monotonic() - phase_start > dur:
                    x, y, z, yaw = pose_xy_yaw(robot)
                    print(f"[sonic_walk] done '{name}': base=({x:+.2f},{y:+.2f}) z={z:.2f} yaw={yaw:+.0f}deg", flush=True)
                    phase_idx, phase_start = phase_idx + 1, time.monotonic()
            if tick - last_print > float(os.environ.get("SONIC_PRINT_DT", "2.0")):
                x, y, z, yaw = pose_xy_yaw(robot)
                state = ("stabilizing" if not robot.stabilized else "waiting for controller" if not agent.planner_engaged
                         else "controller taking over" if not agent.handed_over else SCRIPT[min(phase_idx, len(SCRIPT) - 1)][0])
                print(f"[sonic_walk] t={tick - t0:5.1f}s {state:24s} base=({x:+.2f},{y:+.2f}) z={z:.2f} yaw={yaw:+.0f}deg", flush=True)
                last_print = tick
            if z_fell(robot):
                print("[sonic_walk] FELL", flush=True)
                break
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
            if time.monotonic() - t0 > 180:
                print("[sonic_walk] TIMEOUT", flush=True)
                break
    finally:
        if "qpos_log" in locals() and qpos_log:
            render_video(robot, qpos_log, logs / "sonic_walk_mujoco.mp4")
        agent.stop_controller()
        agent.close()
        os.killpg(ctrl.pid, signal.SIGINT)
        try:
            ctrl.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(ctrl.pid, signal.SIGKILL)
        env.close()


def render_video(robot, qpos_log, path, fps=25):
    """MuJoCo-rendered replay from a camera that tracks the pelvis."""
    import imageio
    import mujoco
    m, d = robot.mjModel, robot.mjData
    r = mujoco.Renderer(m, 360, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 135
    frames = []
    for t, q, handed in qpos_log[::max(1, 50 // fps)]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        img = r.render()
        frames.append(img)
    imageio.mimwrite(path, frames, fps=fps, quality=7, macro_block_size=8)
    print(f"[sonic_walk] video -> {path} ({len(frames)} frames)", flush=True)


def z_fell(robot):
    return robot.mjData.qpos[2] < float(os.environ.get("SONIC_FALL_Z", "0.45"))  # pelvis; ~0.75 standing in this env


if __name__ == "__main__":
    typer.run(main)
