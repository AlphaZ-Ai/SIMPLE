# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Unitree G1 (29dof) wholebody with Unitree Dex1-1 parallel grippers.

Assets are built by `scripts/build_g1_dex1_assets.py`: the MJCF swaps the Dex3
hands of `g1_wholebody` for Dex1 grippers, the USD wraps Unitree's
`g1_29dof_with_dex1_rev_1_0.usd`. Lower body control (AMO) is unchanged since it
only reads leg, waist and arm joints.

Motion planning (CuRobo) still uses the Dex3 kinematics of `g1_wholebody`, so
MP data generation is not supported for this robot yet.
"""

import numpy as np

from simple.core.controller import ControllerCfg
from simple.robots.controllers.combo import WholeBodyEEFControllerCfg
from simple.robots.controllers.eef import DexHandEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.g1_wholebody import (
    G1Wholebody,
    LEFT_LEG_JOINTS,
    RIGHT_LEFT_JOINTS,
    WAIST_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
)
from simple.robots.policy.AMO_Policy import AMO_Policy
from simple.robots.registry import RobotRegistry

# prismatic fingers, q in [-0.02 (open), 0.0245 (closed)] meters
LEFT_HAND_JOINTS = ["left_hand_Joint1_1", "left_hand_Joint2_1"]
RIGHT_HAND_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]

WHOLE_BODY_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

GRIPPER_OPEN = -0.02
GRIPPER_CLOSED = 0.0245


@RobotRegistry.register("g1_dex1_wholebody")
class G1Dex1Wholebody(G1Wholebody):
    uid: str = "g1_dex1_wholebody"
    label: str = "Unitree G1 Wholebody with Dex1 Gripper"

    wholebody_dof: int = 33
    dof: int = 21

    mjcf_path: str = "robots/g1_dex1/g1_29dof_wholebody_dex1.xml"
    usd_path: str = "robots/g1_dex1/g1_29dof_wholebody_dex1.usda"
    robot_ns: str = "g1_29dof_wholebody_dex1"

    hand_uid: str = "dex1_right"
    hand_dof: int = 2

    init_joint_states: dict[str, float] = dict(zip(WHOLE_BODY_JOINTS, [0] * len(WHOLE_BODY_JOINTS)))
    joints_names = WHOLE_BODY_JOINTS
    hand_names = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

    controller_cfg: ControllerCfg = WholeBodyEEFControllerCfg(
        left_leg=PDJointPosControllerCfg(joint_names=LEFT_LEG_JOINTS, init_qpos=[0.0] * 6),
        right_leg=PDJointPosControllerCfg(joint_names=RIGHT_LEFT_JOINTS, init_qpos=[0.0] * 6),
        waist=PDJointPosControllerCfg(joint_names=WAIST_JOINTS, init_qpos=[0.0] * 3),
        left_arm=PDJointPosControllerCfg(joint_names=LEFT_ARM_JOINTS, init_qpos=[0.0] * 7),
        right_arm=PDJointPosControllerCfg(joint_names=RIGHT_ARM_JOINTS, init_qpos=[0.0] * 7),
        left_eef=DexHandEEFControllerCfg(
            joint_names=LEFT_HAND_JOINTS,
            init_qpos=[0.0, 0.0],
            close_qpos=[GRIPPER_CLOSED, GRIPPER_CLOSED],
        ),
        right_eef=DexHandEEFControllerCfg(
            joint_names=RIGHT_HAND_JOINTS,
            init_qpos=[0.0, 0.0],
            close_qpos=[GRIPPER_CLOSED, GRIPPER_CLOSED],
        ),
    )

    hand_prim_path = "right_hand_base_link"
    eef_prim_path: str = "right_hand_base_link"
    wrist_cam_link: str = "right_hand_base_link"

    LEFT_ARM_EE_LINK: str = "left_hand_base_link"
    RIGHT_ARM_EE_LINK: str = "right_hand_base_link"

    def __init__(self):
        super().__init__()
        self.amo_policy = AMO_Policy(robot_type="g1_dex1_wholebody", device="cuda", joint_names=self.joint_names)

        # body gains from g1_wholebody; fingers are PD-controlled in python
        # (torque motors in the MJCF), tuned in scripts/build_g1_dex1_assets.py
        n_body = len(WHOLE_BODY_JOINTS) - len(self.hand_names)
        self.stiffness = np.concatenate([self.stiffness[:n_body], [200.0] * len(self.hand_names)])
        self.damping = np.concatenate([self.damping[:n_body], [2.0] * len(self.hand_names)])
        self.torque_limits = np.concatenate([self.torque_limits[:n_body], [20.0] * len(self.hand_names)])

    def reset(self):
        super().reset()
        self.amo_policy = AMO_Policy(robot_type="g1_dex1_wholebody", device="cuda", joint_names=self.joint_names)

    def update_ee_link(self, hand_uid):
        raise NotImplementedError("motion planning is not supported for g1_dex1_wholebody yet")
