# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""SIMPLE side of the navigation bridge.

Runs a wholebody G1 env in real time with HSSD walls/furniture as MuJoCo
collision, and exchanges navigation data with one client over a local socket
(stdlib `multiprocessing.connection`, so the ROS2 node can run on the system
Python that rclpy is built for, while SIMPLE runs on its own venv):

    server -> client  every scan (10 Hz):
        send_bytes(json header)   {"type": "scan", "stamp", "pose": [x y z qw qx qy qz],
                                   "twist": [vx vy vz wx wy wz] (sensor frame), "n"}
        send_bytes(points)        float32 x,y,z in the map (= MuJoCo world) frame
    client -> server  any time:
        send_bytes(json)          {"type": "cmd_vel", "vx", "vy", "wz"}   (body frame, m/s, rad/s)

The pose is the lidar ("sensor") frame, matching the CMU autonomy stack's
/state_estimation. Velocity commands are turned into AMO loco commands: AMO
holds an absolute heading (turning flag), so yaw rate becomes a heading target
just ahead of the current yaw, and speeds are scaled by the measured AMO gain.
"""

import json
import os
import random
import signal
import time
from multiprocessing.connection import Listener

import numpy as np
import transforms3d as t3d

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402

import simple.envs  # noqa: E402,F401
from simple.core.action import ActionCmd  # noqa: E402
from simple.nav.lidar import Mid360Lidar  # noqa: E402
from simple.nav.mission import PickPlaceMission  # noqa: E402
from simple.nav.run_log import NavRunLog, house_contacts  # noqa: E402

CONTROL_HZ = 50
SCAN_EVERY = 5  # control steps per lidar scan -> 10 Hz

# AMO reaches ~42% of the commanded forward speed with the turning flag set
# (measured: command 0.5 m/s -> 0.21 m/s on g1_dex1_wholebody)
AMO_SPEED_GAIN = 0.42
MAX_AMO_VX = 0.8
MAX_AMO_VY = 0.4
# yaw rate is integrated into AMO's heading target, kept within this of the actual yaw
MAX_HEADING_LEAD = 0.3  # rad
CMD_TIMEOUT_S = 0.5   # stop if no cmd_vel for this long
# below these, stand still: tiny commands turn into single shuffling steps
DEADBAND_V = 0.06     # m/s
DEADBAND_WZ = 0.05    # rad/s


def yaw_of(quat_wxyz):
    return t3d.euler.quat2euler(quat_wxyz)[2]


class NavSimServer:
    def __init__(self, env_id: str, robot_uid: str, scene_uid: str | None, sim_mode: str,
                 headless: bool, port: int, authkey: bytes, seed: int = 0,
                 spawn: tuple[float, float, float] | None = None,
                 log_path: str | None = None, goal: tuple[float, float] | None = None,
                 env_kwargs: dict | None = None, mission: str | None = None):
        # scene placement and objects are randomized with numpy's global RNG
        np.random.seed(seed)
        random.seed(seed)
        self.env = gym.make(env_id, robot_uid=robot_uid, render_hz=CONTROL_HZ, sim_mode=sim_mode,
                            headless=headless, max_episode_steps=10**9, **(env_kwargs or {}))
        task = self.env.unwrapped.task
        task.scene_collision = True
        if scene_uid:
            task.dr_cfgs["scene"].room_choices = [scene_uid]
        self.env.reset(seed=seed)
        self.task = task
        self.robot = task.robot
        self.task_spawn_pose = np.array(self.robot.get_robot_pose(), dtype=float)  # faces the task's table
        if spawn is not None:
            self._teleport(*spawn)
        self.lidar = Mid360Lidar(self.robot.mjmodel, seed=seed)
        self.sim_app = getattr(self.env.unwrapped, "simulation_app", None) if not headless else None

        self.cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.cmd_time = 0.0
        self.target_yaw = yaw_of(self.robot.get_robot_pose()[3:7])
        self._last_sensor = None

        self.mission = None
        if mission == "pick_place":
            self.mission = PickPlaceMission(self, self.task_spawn_pose)
            goal = self.mission.goal()  # the approach point in front of the table
        elif mission:
            raise ValueError(f"unknown mission {mission}")
        self.goal = self.free_point_near(goal) if goal is not None else None
        if goal is not None:
            print(f"[nav_sim] goal=({self.goal[0]:+.2f},{self.goal[1]:+.2f}) requested=({goal[0]:+.2f},{goal[1]:+.2f})", flush=True)
        self.reached = False
        self.fell = False
        self.scene_body = self.robot.mjmodel.body("scene_collision").id
        self.log = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self.log = NavRunLog(log_path, dict(
                env_id=env_id, robot_uid=robot_uid, scene_uid=task.layout.scene.uid, seed=seed,
                spawn=list(spawn) if spawn else None, goal=list(self.goal) if self.goal else None,
                mission=mission, env_kwargs=env_kwargs or {}))

        self.listener = Listener(("127.0.0.1", port), authkey=authkey)
        self.conn = None
        scene = task.layout.scene
        pose = self.robot.get_robot_pose()
        print(f"[nav_sim] scene={scene.uid} offset={np.round(scene.center_offset, 3).tolist()} "
              f"yaw={float(scene.center_orientation[2]):.2f}deg robot=({pose[0]:+.2f},{pose[1]:+.2f}) "
              f"heading={np.degrees(self.target_yaw):+.1f}deg seed={seed}", flush=True)
        print(f"[nav_sim] listening on 127.0.0.1:{port}", flush=True)

    def house_clearance(self, xy: np.ndarray) -> np.ndarray:
        """Distance (m) from points (N, 2) to the nearest house collision box footprint."""
        m, d = self.robot.mjmodel, self.robot.mjdata
        sb = m.body("scene_collision").id
        ids = np.flatnonzero(m.geom_bodyid == sb)
        R = d.xmat[sb].reshape(3, 3)[:2, :2]
        local = (xy[:, None, :] - d.geom_xpos[ids][None, :, :2]) @ R  # into the house frame
        gap = np.maximum(np.abs(local) - m.geom_size[ids][None, :, :2], 0.0)
        return np.linalg.norm(gap, axis=2).min(axis=1)

    def free_point_near(self, goal, clearance: float = 0.5, search: float = 2.0):
        """`goal`, or the nearest point at least `clearance` from walls and furniture."""
        g = np.asarray(goal, dtype=np.float64)
        if self.house_clearance(g[None])[0] >= clearance:
            return tuple(g)
        offs = np.arange(-search, search + 1e-9, 0.1)
        cand = g + np.stack(np.meshgrid(offs, offs), axis=-1).reshape(-1, 2)
        cand = cand[np.argsort(np.linalg.norm(cand - g, axis=1))]
        free = cand[self.house_clearance(cand) >= clearance]
        if not len(free):
            print(f"[nav_sim] WARNING: no free spot within {search} m of goal {tuple(g)}", flush=True)
            return tuple(g)
        return tuple(free[0])

    def _teleport(self, x: float, y: float, yaw_deg: float):
        """Move the standing robot to (x, y) facing yaw (world frame), before any stepping."""
        d, m = self.robot.mjdata, self.robot.mjmodel
        d.qpos[0:2] = [x, y]
        d.qpos[3:7] = t3d.euler.euler2quat(0, 0, np.radians(yaw_deg))
        d.qvel[:6] = 0
        mujoco.mj_forward(m, d)
        sb = m.body("scene_collision").id
        hits = {m.body(m.geom_bodyid[c.geom1 if m.geom_bodyid[c.geom2] == sb else c.geom2]).name
                for c in d.contact[:d.ncon] if sb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])}
        if hits:
            print(f"[nav_sim] WARNING: spawn ({x}, {y}) touches the house with {sorted(hits)}", flush=True)

    # ---- velocity -> AMO -------------------------------------------------
    def _amo_heading(self, yaw: float, err: float) -> float:
        """Heading target `err` ahead of the current yaw, in AMO's frame: AMO measures
        yaw relative to the pose it first saw (AMO_Policy._initial_quat), which is the
        task spawn, not a --spawn teleport."""
        ref = getattr(self.robot.amo_policy, "_initial_quat", None)
        amo_offset = yaw_of(ref) if ref is not None else 0.0
        return (yaw - amo_offset + np.pi) % (2 * np.pi) - np.pi + err

    def heading_action(self, target_yaw: float):
        """Stand and turn in place toward a world heading."""
        yaw = yaw_of(self.robot.get_robot_pose()[3:7])
        err = float(np.clip((target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi, -MAX_HEADING_LEAD, MAX_HEADING_LEAD))
        self.target_yaw = yaw + err
        return ActionCmd("loco_command", command=[0, self._amo_heading(yaw, err), 0, 0, 0, 0, 0, 1], motion_type="stand")

    def loco_action(self):
        if time.monotonic() - self.cmd_time > CMD_TIMEOUT_S:
            vx = vy = wz = 0.0
        else:
            vx, vy, wz = self.cmd["vx"], self.cmd["vy"], self.cmd["wz"]
            if abs(vx) < DEADBAND_V and abs(vy) < DEADBAND_V:
                vx = vy = 0.0
            if abs(wz) < DEADBAND_WZ:
                wz = 0.0
        yaw = yaw_of(self.robot.get_robot_pose()[3:7])
        # integrate the commanded yaw rate (a lead on the measured yaw lags and
        # overshoots on a walking robot), without winding up past the robot
        target = self.target_yaw + wz / CONTROL_HZ
        err = (target - yaw + np.pi) % (2 * np.pi) - np.pi
        err = float(np.clip(err, -MAX_HEADING_LEAD, MAX_HEADING_LEAD))
        self.target_yaw = yaw + err
        amo_target = self._amo_heading(yaw, err)
        amo_vx = float(np.clip(vx / AMO_SPEED_GAIN, -MAX_AMO_VX, MAX_AMO_VX))
        amo_vy = float(np.clip(vy / AMO_SPEED_GAIN, -MAX_AMO_VY, MAX_AMO_VY))
        # vx, target_yaw, vy, d_height, roll, pitch, yaw, turning_flag
        return ActionCmd("loco_command", command=[amo_vx, amo_target, amo_vy, 0, 0, 0, 0, 1], motion_type="stand")

    # ---- messaging -------------------------------------------------------
    def _poll_commands(self):
        while self.conn is not None and self.conn.poll():
            msg = json.loads(self.conn.recv_bytes())
            if msg.get("type") == "cmd_vel":
                self.cmd = {k: float(msg.get(k, 0.0)) for k in ("vx", "vy", "wz")}
                self.cmd_time = time.monotonic()
            elif msg.get("type") in ("waypoint", "goal") and self.log is not None:
                self.log.event(self.sim_time, f"{msg['type']}=({msg['x']:+.2f},{msg['y']:+.2f})")

    def _send_scan(self, now):
        pts, p, R = self.lidar.scan(self.robot.mjdata)
        q = t3d.quaternions.mat2quat(R)
        twist = [0.0] * 6
        if self._last_sensor is not None:
            p0, yaw0, t0 = self._last_sensor
            dt = max(now - t0, 1e-3)
            v_world = (p - p0) / dt
            v_body = R.T @ v_world
            dyaw = (yaw_of(q) - yaw0 + np.pi) % (2 * np.pi) - np.pi
            twist = [*v_body.tolist(), 0.0, 0.0, dyaw / dt]
        self._last_sensor = (p.copy(), yaw_of(q), now)
        header = {"type": "scan", "stamp": now, "pose": [*p.tolist(), *q.tolist()], "twist": twist, "n": len(pts)}
        self.conn.send_bytes(json.dumps(header).encode())
        self.conn.send_bytes(pts.tobytes())

    def _accept(self):
        print("[nav_sim] waiting for bridge client ...", flush=True)
        self.conn = self.listener.accept()
        print("[nav_sim] client connected", flush=True)

    # ---- main loop -------------------------------------------------------
    def run(self, wait_for_client: bool = True):
        if wait_for_client:
            self._accept()
        dt = 1.0 / CONTROL_HZ
        self.sim_time = 0.0

        def stop(signum, frame):
            raise KeyboardInterrupt
        # background jobs start with SIGINT ignored; make both INT and TERM end the run cleanly
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            self._loop(dt)
        except KeyboardInterrupt:
            pass
        finally:
            self._finish()

    def _check_progress(self):
        pose = self.robot.get_robot_pose()
        floor = getattr(self.env.unwrapped.mujoco, "scene_floor_z", None)
        floor = float(self.robot.mjdata.geom("ground").xpos[2]) if floor is None else floor
        if not self.fell and pose[2] - floor < 0.5:  # pelvis ~0.78 m above the floor when standing
            self.fell = True
            self._event(f"FELL at ({pose[0]:+.2f},{pose[1]:+.2f})")
        if self.goal is not None and not self.reached and not self.fell:
            if np.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1]) < 0.4:
                self.reached = True
                self._event(f"REACHED goal in {self.sim_time:.1f}s")

    def _event(self, text):
        print(f"[nav_sim] t={self.sim_time:.1f}s {text}", flush=True)
        if self.log is not None:
            self.log.event(self.sim_time, text)

    def _finish(self):
        if self.log is not None:
            self.log.meta["result"] = "REACHED" if self.reached else ("FELL" if self.fell else "NOT_REACHED")
            if self.mission is not None:
                r = self.mission.result
                self.log.meta["mission_state"] = self.mission.state
                self.log.meta["mission"] = None if r is None else dict(
                    success=r.success, lifted=r.lifted, lift_height=r.lift_height,
                    placed=r.placed, place_error=r.place_error, ik_failures=r.ik_failures, log=r.log)
            self.log.save()
        self.env.close()

    def _sim_step(self, action):
        """Physics step without rendering the task's cameras: env.step() renders
        every camera each step, which can push the sim below real time (the nav
        stack runs on wall time). IsaacSim, when present, still mirrors MuJoCo."""
        u = self.env.unwrapped
        u.mujoco.apply_action(action)
        u.mujoco.step(render=False)
        if u.isaac is not None:
            u.isaac.step(u.mujoco)

    def _loop(self, dt):
        step = 0
        next_t = time.monotonic()
        while True:
            if self.sim_app is not None:
                if not self.sim_app.is_running():
                    break
                self.sim_app.update()
            try:
                self._poll_commands()
                action = self.loco_action()
                if self.mission is not None:
                    action = self.mission.action(action)
                self._sim_step(action)
                self.sim_time += dt
                self._check_progress()
                if self.log is not None:
                    self.log.add(self.sim_time, self.robot.mjdata.qpos, self.cmd, action.parameters["command"],
                                 house_contacts(self.robot.mjmodel, self.robot.mjdata, self.scene_body))
                if step % SCAN_EVERY == 0 and self.conn is not None:
                    self._send_scan(time.time())
            except (EOFError, ConnectionResetError, BrokenPipeError):
                print("[nav_sim] client disconnected; holding position", flush=True)
                self.conn, self.cmd = None, {"vx": 0.0, "vy": 0.0, "wz": 0.0}
                self._accept()
            step += 1
            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # running slower than real time; don't try to catch up
