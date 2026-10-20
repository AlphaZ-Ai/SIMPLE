# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Scripted pick and place with one Dex1 gripper while AMO keeps the G1 balanced.

    turn -> line_up -> step_in -> calm -> reach -> approach -> close -> lift -> carry -> lower -> release -> retreat -> tuck -> settle

- The robot first steps to a stance next to the object (in front of the
  grasping arm's shoulder) and then holds it with PI station keeping: AMO
  "stand" otherwise lets the robot wander a few cm while the arm moves.
- Cartesian phases move the TCP (centre between the finger pads) along a
  straight line to their target and re-solve IK every few control steps from
  the robot's *current* pose, so base drift is corrected instead of accumulating.
- Object poses come from the simulator (ground truth); success is judged from
  where the object actually ends up.
"""

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
import transforms3d as t3d

from simple.core.action import ActionCmd
from simple.nav.arm_ik import ARM_JOINTS, ArmIK
from simple.robots.g1_dex1_wholebody import GRIPPER_CLOSED, GRIPPER_OPEN, TUCK_ARM_POSE

CONTROL_DT = 0.02
IK_EVERY = 2  # control steps between IK solves (0.04 s)
MAX_WAIT = 4.0  # s a phase may extend while waiting for its `ready` condition
ABORT_STANCE_ERROR = 0.06  # m: give up instead of reaching from further than this

# station keeping (AMO command units; AMO ignores very small commands, hence V_MIN)
STATION_DEADBAND = 0.02
STATION_KP, STATION_KI = 2.0, 1.0
STATION_V_MIN, STATION_V_MAX = 0.15, 0.3


@dataclass
class Phase:
    name: str
    duration: float
    gripper: float
    tcp: Callable[[], np.ndarray] | None = None  # target computed when the phase starts
    joints: dict | None = None  # joint-space target instead of a TCP target
    hold_stance: bool = True  # step to keep the stance; off for fine motion (stepping sways the arm)
    track: bool = False  # recompute the TCP target every solve (follow the object)
    ready: Callable[[], bool] | None = None  # extend the phase (up to max_wait) until this holds
    max_wait: float = MAX_WAIT
    station: bool = True  # with hold_stance: also step toward the stance (False: only turn)
    back: float = 0.0  # station-keep this far behind the stance (m, along the heading)


@dataclass
class PickPlaceResult:
    lifted: bool = False
    placed: bool = False
    aborted: str = ""
    lift_height: float = 0.0
    place_error: float = float("nan")
    ik_failures: list = field(default_factory=list)
    log: list = field(default_factory=list)

    @property
    def success(self):
        return self.lifted and self.placed


class PickPlace:
    def __init__(self, robot, object_body: str, place_offset=(0.0, -0.12), tilt_deg: float = 25.0,
                 pregrasp: float = 0.10, lift: float = 0.10, side: str = "right",
                 standoff: float = 0.38, lateral: float = 0.10, finger_clearance: float = 0.025,
                 yaw: float | None = None):
        """place_offset: where to put the object, relative to where it was (robot frame, m).
        standoff/lateral: stance relative to the object (robot frame): `standoff` behind it and
        `lateral` toward the other arm, so the object is in front of the grasping shoulder.
        finger_clearance: lowest TCP height above the table (keeps the pads off it).
        yaw: heading to work in (world, rad); default: the robot's current heading."""
        self.robot, self.side = robot, side
        self.m, self.d = robot.mjmodel, robot.mjdata
        self.obj = self.m.body(object_body).id
        self.ik = ArmIK(self.m, side)
        self.finger_clearance = finger_clearance
        self.result = PickPlaceResult()

        self.yaw = t3d.euler.quat2euler(robot.get_robot_pose()[3:7])[2] if yaw is None else float(yaw)
        Rz = t3d.axangles.axangle2mat([0, 0, 1], self.yaw)
        # front grasp: fingers point forward (gripper +y = robot +x), close sideways
        # (gripper x = robot -y), tilted down so the hand stays above the table edge
        R_front = Rz @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        self.R_grasp = R_front @ t3d.axangles.axangle2mat([1, 0, 0], -np.radians(tilt_deg))
        self.approach = self.R_grasp[:, 1]

        self.table_z = self.object_bottom()
        self.start_obj = self.object_center()
        g = self.grasp_point()
        self.place = g + Rz @ np.array([place_offset[0], place_offset[1], 0.0])
        away = 1.0 if side == "right" else -1.0
        self.stance = g[:2] + (Rz @ np.array([-standoff, away * lateral, 0.0]))[:2]
        self._I = np.zeros(2)

        other = "left" if side == "right" else "right"
        # idle arm tucked: from the stance its hand is well behind the table edge
        # (in the zero pose it reaches over the table and knocks objects over)
        self.other_arm = {j: v for j, v in TUCK_ARM_POSE.items() if j.startswith(other)}
        tuck = {j: v for j, v in TUCK_ARM_POSE.items() if j.startswith(side)}
        # ready pose: TCP 15 cm ahead of the pelvis, 12 cm above the table, fingers
        # forward - above the objects and short of them, so the robot can step in
        # holding it (the zero pose reaches into the objects, the tuck sits below the
        # table edge and cannot get over it on a straight line)
        ready = self._ready_pose(Rz, ahead=0.15, above_table=0.12)
        up = np.array([0.0, 0.0, 1.0])
        self.phases = [
            # turn to the table on the spot, then step into the stance (doing both at
            # once rarely converged); the arm goes to the ready (zero) pose meanwhile
            Phase("turn", 1.0, GRIPPER_OPEN, joints=ready, station=False,
                  ready=lambda: abs(self.heading_error()) < 0.08, max_wait=10.0),
            # line up sideways with room to spare (side-stepping along the table edge
            # rubs against it), then step straight in
            Phase("line_up", 2.0, GRIPPER_OPEN, joints=ready, back=0.25,
                  ready=lambda: self.stance_error(0.25) < 0.04 and abs(self.heading_error()) < 0.1, max_wait=12.0),
            Phase("step_in", 2.0, GRIPPER_OPEN, joints=ready,
                  ready=lambda: self.stance_error() < 0.03 and abs(self.heading_error()) < 0.1, max_wait=10.0),
            # come to a calm plain stand before any arm work: switching from stepping
            # straight into arm motion is what made AMO wander off
            Phase("calm", 1.5, GRIPPER_OPEN, joints=ready, hold_stance=False),
            # hover at the pregrasp until the TCP is on target
            Phase("reach", 2.0, GRIPPER_OPEN, tcp=lambda: self.grasp_point() - pregrasp * self.approach,
                  hold_stance=False, ready=lambda: self.tcp_error() < 0.015),
            Phase("approach", 2.0, GRIPPER_OPEN, tcp=lambda: self.grasp_point(), hold_stance=False, track=True),
            Phase("close", 1.0, GRIPPER_CLOSED, tcp=lambda: self.goal, hold_stance=False),
            Phase("lift", 1.5, GRIPPER_CLOSED, tcp=lambda: self.goal + lift * up, hold_stance=False),
            Phase("carry", 2.0, GRIPPER_CLOSED, tcp=lambda: self.place + lift * up, hold_stance=False),
            Phase("lower", 1.5, GRIPPER_CLOSED, tcp=lambda: self.place + 0.01 * up, hold_stance=False),
            Phase("release", 1.0, GRIPPER_OPEN, tcp=lambda: self.goal, hold_stance=False),
            # straight up first: pulling the open fingers back along the table drags the object
            Phase("clear", 1.0, GRIPPER_OPEN, tcp=lambda: self.goal + 0.05 * up, hold_stance=False),
            Phase("retreat", 1.5, GRIPPER_OPEN, tcp=lambda: self.goal - pregrasp * self.approach),
            Phase("tuck", 2.0, GRIPPER_OPEN, joints=tuck),
            Phase("settle", 1.5, GRIPPER_OPEN, joints=tuck),
        ]
        self.i, self.k, self.started = 0, 0, False
        self.goal = None
        self.done = False

    # ---- object state -----------------------------------------------------------
    def _object_points(self):
        pts = []
        for g in np.flatnonzero(self.m.geom_bodyid == self.obj):
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                pts.append(v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g])
            else:
                pts.append(self.d.geom_xpos[g][None])
        return np.concatenate(pts)

    def object_center(self):
        p = self._object_points()
        return (p.min(0) + p.max(0)) / 2

    def object_bottom(self):
        return float(self._object_points()[:, 2].min())

    def grasp_point(self):
        c = self.object_center()
        c[2] = max(c[2], self.table_z + self.finger_clearance)
        return c

    # ---- control --------------------------------------------------------------------
    def _arm_q(self):
        return {j: float(self.robot.joints[j].qpos.item()) for j in ARM_JOINTS[self.side]}

    def _start_phase(self):
        ph = self.phases[self.i]
        self.q_from = self._arm_q()
        self.q_cmd = dict(self.q_from)
        if ph.tcp is not None:
            self.tcp_from, _ = self.ik.tcp_pose(self.d)
            self.goal = ph.tcp()
        self.started = True
        self.result.log.append(f"{ph.name}: start")

    def _solve(self, target):
        tries = [self.q_cmd, self._arm_q(), {j: 0.0 for j in ARM_JOINTS[self.side]}]
        return min((self.ik.solve(self.d, target, self.R_grasp, q_init=q, iters=100) for q in tries),
                   key=lambda r: r.pos_err + 0.1 * r.rot_err)

    def station_target(self, back: float = 0.0):
        fwd = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        return self.stance - back * fwd

    def stance_error(self, back: float = 0.0):
        return float(np.linalg.norm(np.asarray(self.robot.get_robot_pose()[:2]) - self.station_target(back)))

    def _ready_pose(self, Rz, ahead: float, above_table: float) -> dict:
        """Arm joints for a TCP `ahead` of the pelvis and `above_table`, solved with the
        robot placed at the stance (only the arm moves, so the pose is reusable anywhere)."""
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        scratch.qpos[0:2] = self.stance
        scratch.qpos[3:7] = t3d.euler.euler2quat(0, 0, self.yaw)
        side = -1.0 if self.side == "right" else 1.0
        target = np.array([*self.stance, 0.0]) + Rz @ np.array([ahead, side * 0.12, 0.0])
        target[2] = self.table_z + above_table
        r = self.ik.solve(scratch, target, self.R_grasp, q_init={j: 0.0 for j in ARM_JOINTS[self.side]})
        if not r.converged:
            self.result.ik_failures.append(("ready_pose", round(r.pos_err, 3)))
        return r.q

    def heading_error(self):
        yaw_now = t3d.euler.quat2euler(self.robot.get_robot_pose()[3:7])[2]
        return float((self.yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi)

    def tcp_error(self):
        tcp, _ = self.ik.tcp_pose(self.d)
        return float(np.linalg.norm(tcp - self.goal)) if self.goal is not None else float("inf")

    def _end_phase(self):
        ph = self.phases[self.i]
        c = self.object_center()
        drift = float(np.linalg.norm(np.asarray(self.robot.get_robot_pose()[:2]) - self.stance))
        msg = f"{ph.name}: end obj=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) stance err={drift * 100:.1f}cm"
        if ph.tcp is not None:
            tcp, _ = self.ik.tcp_pose(self.d)
            err = float(np.linalg.norm(tcp - self.goal))
            msg += f" tcp_err={err * 100:.1f}cm"
            if err > 0.03:
                self.result.ik_failures.append((ph.name, round(err, 3)))
        if ph.name == "step_in" and self.stance_error() > ABORT_STANCE_ERROR:
            # reaching from the wrong spot overextends the arm and can topple the robot
            self.result.aborted = f"stance not reached ({self.stance_error() * 100:.0f} cm off)"
        if ph.name == "lift":
            self.result.lift_height = float(c[2] - self.start_obj[2])
            self.result.lifted = self.result.lift_height > 0.05
        if ph.name == "settle":
            err = float(np.linalg.norm(c[:2] - self.place[:2]))
            on_table = abs(self.object_bottom() - self.table_z) < 0.02
            self.result.place_error = err
            self.result.placed = err < 0.05 and on_table and self.result.lifted
        self.result.log.append(msg)

    def step(self) -> ActionCmd:
        """Advance one control step; returns the loco action (AMO stand + station keeping)."""
        if self.done:
            return self._stand()
        if not self.started:
            self._start_phase()
        ph = self.phases[self.i]
        s = min(1.0, (self.k + 1) * CONTROL_DT / ph.duration)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth ramp
        if ph.joints is not None:
            self.q_cmd = {j: (1 - s) * self.q_from[j] + s * ph.joints[j] for j in ph.joints}
        elif self.k % IK_EVERY == 0:
            if ph.track:
                self.goal = ph.tcp()
            self.q_cmd = self._solve((1 - s) * self.tcp_from + s * self.goal).q
        self.robot.set_arm_targets(self.q_cmd)
        self.robot.set_arm_targets(self.other_arm)
        self.robot.set_gripper(self.side, ph.gripper)
        self.k += 1
        t = self.k * CONTROL_DT
        waiting = ph.ready is not None and not ph.ready() and t < ph.duration + ph.max_wait
        if t >= ph.duration and not waiting:
            self._end_phase()
            self.i, self.k, self.started = self.i + 1, 0, False
            self.done = self.i >= len(self.phases) or bool(self.result.aborted)
        return self._stand(hold=ph.hold_stance, station=ph.station, back=ph.back)

    def _stand(self, hold: bool = True, station: bool = True, back: float = 0.0):
        """AMO stand at the stance: PI station keeping in the body frame, heading held.

        hold=False: plain AMO stand (turning flag off) - the mode that keeps still
        while the arm works. With the flag on AMO holds an absolute heading and
        wanders off under arm motion. Commands are sent whole (no keep_waist_pose):
        that path only copies the first four fields, so the flag would stick."""
        if not hold:
            self._I[:] = 0.0
            # heading slot still in AMO's frame: AMO reads it as a target here too, and a
            # bare 0 means "the heading AMO started at", which differs after a turn
            return ActionCmd("loco_command", command=[0, self.yaw_amo(), 0, 0, 0, 0, 0, 0], motion_type="stand")
        pose = self.robot.get_robot_pose()
        yaw_now = t3d.euler.quat2euler(pose[3:7])[2]
        err = self.station_target(back) - np.asarray(pose[:2])
        c, s = np.cos(yaw_now), np.sin(yaw_now)
        e = np.array([c * err[0] + s * err[1], -s * err[0] + c * err[1]])
        v = np.zeros(2)
        if not station:
            self._I[:] = 0.0
        elif np.linalg.norm(e) > STATION_DEADBAND:
            self._I = np.clip(self._I + e * CONTROL_DT, -0.2, 0.2)
            v = STATION_KP * e + STATION_KI * self._I
            n = np.linalg.norm(v)
            if 0 < n < STATION_V_MIN:
                v *= STATION_V_MIN / n
            v = np.clip(v, -STATION_V_MAX, STATION_V_MAX)
        else:
            self._I *= 0.95
        return ActionCmd("loco_command", command=[float(v[0]), self.yaw_amo(), float(v[1]), 0, 0, 0, 0, 1],
                         motion_type="stand")

    def yaw_amo(self):
        """Heading to hold, in AMO's frame (AMO measures yaw from the pose it first saw)."""
        ref = getattr(self.robot.amo_policy, "_initial_quat", None)
        off = t3d.euler.quat2euler(ref)[2] if ref is not None else 0.0
        return float((self.yaw - off + np.pi) % (2 * np.pi) - np.pi)
