# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Navigate to a table in another room, then pick and place there.

Runs inside NavSimServer. The ROS nav stack drives while the mission is in NAV;
after that the mission ignores /cmd_vel and commands the robot itself:

    NAV    arms tucked, grippers held; the nav stack walks to an approach point
           `approach_back` behind the grasp stance, on the table-facing heading
    PICK   simple.nav.pick_place.PickPlace: turn to the table and step into the
           stance (it holds position; a separate in-place turn let AMO drift
           backward), then grasp and place
    DONE

The table-facing heading and the object come from the task: its own robot
spawn faces the table, and its "target" object is what gets picked.
"""

import numpy as np
import transforms3d as t3d

from simple.nav.pick_place import PickPlace
from simple.robots.g1_dex1_wholebody import GRIPPER_OPEN, TUCK_ARM_POSE

ARRIVE_RADIUS = 0.35   # m from the approach point to stop navigating


class PickPlaceMission:
    def __init__(self, server, task_spawn_pose, object_name: str = "target", approach_back: float = 0.6):
        self.server, self.robot = server, server.robot
        m = self.robot.mjmodel
        self.object_body = m.body(server.env.unwrapped.mujoco.mj_objects[object_name].id).name
        self.yaw = t3d.euler.quat2euler(task_spawn_pose[3:7])[2]
        # stance as PickPlace would choose it, computed with the table-facing heading
        probe = PickPlace(self.robot, self.object_body, yaw=self.yaw)
        fwd = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        self.stance = probe.stance
        self.approach_point = self.stance - approach_back * fwd
        self.state = "NAV"
        self.pick = None
        self.result = None

    def goal(self):
        return tuple(self.approach_point)

    def _event(self, text):
        self.server._event(text)

    def action(self, nav_action):
        pose = self.robot.get_robot_pose()
        if self.state == "NAV":
            self.robot.set_arm_targets(TUCK_ARM_POSE)
            self.robot.set_gripper("left", GRIPPER_OPEN)
            self.robot.set_gripper("right", GRIPPER_OPEN)
            if np.hypot(*(np.asarray(pose[:2]) - self.approach_point)) < ARRIVE_RADIUS:
                self.state = "PICK"
                self.pick = PickPlace(self.robot, self.object_body, yaw=self.yaw)
                self._event("MISSION arrived at the table -> PICK")
            else:
                return nav_action
        if self.state == "PICK":
            a = self.pick.step()
            if self.pick.done:
                r = self.pick.result
                self.result = r
                self.state = "DONE"
                self._event(f"MISSION_DONE success={r.success} lifted={r.lifted} lift={r.lift_height:.3f} "
                            f"placed={r.placed} place_err={r.place_error:.3f}"
                            + (f" aborted: {r.aborted}" if r.aborted else ""))
            return a
        return self.pick.step()  # DONE: keep standing at the stance
