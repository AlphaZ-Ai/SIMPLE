# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Simulated Livox Mid-360 on MuJoCo raycasts.

The Mid-360 scans 360 deg horizontally and -7..+52 deg vertically with a
non-repetitive pattern; we approximate it with a fresh set of uniformly random
directions per scan. Rays that hit the robot itself are dropped, as are rays
with no return within `max_range`.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Mid360Cfg:
    mount_body: str = "torso_link"
    # mid360_joint in Unitree's g1_29dof_with_dex1_rev_1_0.usd
    mount_pos: tuple = (0.00028, 0.00003, 0.41618)
    mount_quat: tuple = (0.999799, 0.0, 0.020069, 0.0)  # wxyz
    # The real G1 carries the Mid-360 upside down in its head, so it looks
    # -52..+7 deg and sees the floor from ~1 m out (the USD mount frame omits
    # this flip). The reported sensor pose stays upright; only rays flip.
    upside_down: bool = True
    fov_down_deg: float = -7.0
    fov_up_deg: float = 52.0
    rays_per_scan: int = 4000  # real sensor: ~20k points per 0.1 s scan; raycast cost grows with house size
    min_range: float = 0.1
    max_range: float = 40.0


class Mid360Lidar:
    def __init__(self, model: mujoco.MjModel, robot_root_body: str = "pelvis", cfg: Mid360Cfg | None = None, seed: int = 0):
        self.cfg = cfg or Mid360Cfg()
        self.model = model
        self.mount_body = model.body(self.cfg.mount_body).id
        self.rng = np.random.default_rng(seed)

        # geoms on the robot (everything under robot_root_body, plus anything welded to it)
        root = model.body(robot_root_body).id
        robot_bodies = {b for b in range(model.nbody) if self._is_descendant(b, root)}
        self.robot_geom = np.zeros(model.ngeom, dtype=bool)
        self.robot_geom[[g for g in range(model.ngeom) if model.geom_bodyid[g] in robot_bodies]] = True

        n = self.cfg.rays_per_scan
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)

        w, x, y, z = self.cfg.mount_quat
        self._R_mount = np.zeros(9)
        mujoco.mju_quat2Mat(self._R_mount, np.array([w, x, y, z], dtype=np.float64))
        self._R_mount = self._R_mount.reshape(3, 3)
        self._p_mount = np.asarray(self.cfg.mount_pos, dtype=np.float64)

    def _is_descendant(self, body, root):
        while body != 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return False

    def sensor_pose(self, data: mujoco.MjData):
        """World position (3,) and rotation (3, 3) of the lidar frame."""
        R_body = data.xmat[self.mount_body].reshape(3, 3)
        p = data.xpos[self.mount_body] + R_body @ self._p_mount
        return p, R_body @ self._R_mount

    def _directions(self):
        n = self.cfg.rays_per_scan
        az = self.rng.uniform(-np.pi, np.pi, n)
        # uniform on the sphere band between the two elevations
        s_lo, s_hi = np.sin(np.radians([self.cfg.fov_down_deg, self.cfg.fov_up_deg]))
        el = np.arcsin(self.rng.uniform(s_lo, s_hi, n))
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)

    def scan(self, data: mujoco.MjData):
        """Return (points_world (N, 3) float32, sensor_pos (3,), sensor_rot (3, 3)).
        sensor_rot is the upright sensor frame, even when the lidar is mounted flipped."""
        p, R = self.sensor_pose(data)
        dirs = self._directions()
        if self.cfg.upside_down:  # rotate pi about the sensor x axis
            dirs = dirs * np.array([1.0, -1.0, -1.0])
        dirs_world = dirs @ R.T
        # the sensor sits inside the torso/head collision mesh, so skip that body
        mujoco.mj_multiRay(
            self.model, data, p, dirs_world.reshape(-1), None, 1, self.mount_body,
            self._geomid, self._dist, self.cfg.rays_per_scan, self.cfg.max_range,
        )
        hit = (self._geomid >= 0) & (self._dist >= self.cfg.min_range) & (self._dist <= self.cfg.max_range)
        hit &= ~self.robot_geom[np.clip(self._geomid, 0, None)]
        pts = p + dirs_world[hit] * self._dist[hit, None]
        return pts.astype(np.float32), p, R
