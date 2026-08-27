# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Run a brief "stand" warmup inside `env.reset` so agents see a stabilized
robot from step 0, instead of every baseline duplicating the same 60-step
stand-loop boilerplate.
"""

from __future__ import annotations

import gymnasium as gym

from simple.core.action import ActionCmd


class StandStabilizationWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Run N stand-in-place actions inside `reset` for Humanoid robots.

    Eliminates the per-agent boilerplate where every Wholebody-G1 baseline
    (psi0, dp_g1, act_g1, intervla_m1_g1, dreamzero, gr00t_n16, pi05) and
    `ReplayAgent` queues 60 `motion_type="stand"` commands at the start of
    every rollout. With this wrapper applied at `gym.make` time, the inner
    env's `reset` returns a post-stand observation, so:

        - Agents can drop the `if self._global_step_idx == 0: ...` block
          and start their first VLA query immediately.
        - The recorded rollout video / dataset frame 0 is the post-warmup
          observation (no more polluted-by-warmup frames).
        - Frame 0 of a `VideoRecorder` wrapped on top of this matches the
          training-data convention.

    No-op if the robot is not Humanoid, so safe to apply unconditionally.
    """

    def __init__(self, env: gym.Env, n_stand_steps: int = 60):
        gym.utils.RecordConstructorArgs.__init__(self, n_stand_steps=n_stand_steps)
        gym.Wrapper.__init__(self, env)
        self.n_stand_steps = n_stand_steps

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.n_stand_steps <= 0:
            return observation, info

        from simple.robots.protocols import Humanoid

        robot = getattr(self.unwrapped.task, "robot", None)
        if not isinstance(robot, Humanoid):
            return observation, info

        print(
            f"Applying StandStabilizationWrapper: running {self.n_stand_steps} stand steps to stabilize the robot before returning from reset.",
        )
        stand = ActionCmd(
            "loco_command",
            command=[0, 0, 0, 0, 0, 0, 0, 0],  # vx, target_yaw, vy, d_height, roll, pitch, yaw, turning_flag
            motion_type="stand",
            keep_waist_pose=False,
        )
        for _ in range(self.n_stand_steps):
            observation, _reward, _term, _trunc, info = self.env.step(stand)

        # Reset step bookkeeping so the warmup doesn't count toward TimeLimit
        # / step_count budgets observed by downstream consumers.
        if hasattr(self.unwrapped, "step_count"):
            self.unwrapped.step_count = 0
        if hasattr(self.unwrapped, "_success"):
            self.unwrapped._success = False

        return observation, info
