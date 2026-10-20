# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Damped-least-squares IK for one G1 arm with a Dex1 gripper.

Solves on a scratch MjData copy so the live simulation is untouched. The
controlled point is the Dex1 "TCP": the centre between the finger pads,
expressed in the gripper base frame (`<side>_hand_base_link`). In that frame
the fingers point along +y and close along x.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = {
    side: [f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint", f"{side}_shoulder_yaw_joint",
           f"{side}_elbow_joint", f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint", f"{side}_wrist_yaw_joint"]
    for side in ("left", "right")
}
# centre between the Dex1 finger pads in the gripper base frame (fingers along +y)
TCP_OFFSET = np.array([0.0, 0.11, 0.014])


@dataclass
class IKResult:
    q: dict[str, float]
    pos_err: float
    rot_err: float
    converged: bool


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str = "right"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.side = side
        self.joints = ARM_JOINTS[side]
        self.jid = [model.joint(j).id for j in self.joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jid])
        self.lo = model.jnt_range[self.jid, 0]
        self.hi = model.jnt_range[self.jid, 1]
        self.body = model.body(f"{side}_hand_base_link").id

    def tcp_pose(self, data: mujoco.MjData):
        """World TCP position and gripper-base rotation for the given data."""
        R = data.xmat[self.body].reshape(3, 3)
        return data.xpos[self.body] + R @ TCP_OFFSET, R

    def solve(self, data: mujoco.MjData, target_pos, target_rot, q_init: dict | None = None,
              iters: int = 200, damping: float = 0.05, rot_weight: float = 0.5,
              pos_tol: float = 0.005, rot_tol: float = 0.05) -> IKResult:
        """Arm joint angles putting the TCP at target_pos with gripper-base rotation target_rot.

        Everything except this arm keeps its value from `data` (current robot state).
        """
        d = self.d
        d.qpos[:] = data.qpos
        if q_init:
            for j, a in zip(self.joints, self.qadr):
                d.qpos[a] = q_init.get(j, d.qpos[a])
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_rot = np.asarray(target_rot, dtype=np.float64)
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, R = self.tcp_pose(d)
            e_pos = target_pos - p
            # rotation error as an axis-angle vector in the world frame
            R_err = target_rot @ R.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R_err.reshape(-1))
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                break
            mujoco.mj_jac(self.m, d, jacp, jacr, p, self.body)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            e = np.concatenate([e_pos, rot_weight * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, d)
        p, R = self.tcp_pose(d)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (target_rot @ R.T).reshape(-1))
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, quat, 1.0)
        pe, re = float(np.linalg.norm(target_pos - p)), float(np.linalg.norm(e_rot))
        return IKResult({j: float(d.qpos[a]) for j, a in zip(self.joints, self.qadr)}, pe, re,
                        pe < pos_tol * 2 and re < rot_tol * 2)
