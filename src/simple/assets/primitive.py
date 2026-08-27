# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import numpy as np

from .asset_manager import AssetManager
from simple.core.asset import Asset
from simple.core.actor import Actor
# import os
from simple.core.types import Pose

class Primitive(Asset, Actor):
    material: dict


class Box(Primitive):
    def __init__(self, size, position, quaternion) -> None:
        self.uid = "box"

        self.size = size
        # self.position = position
        # self.quaternion = quaternion
        self.pose = Pose(position, quaternion)

    def set_material(self, material: dict) -> None:
        self.material = material

    def __repr__(self) -> str:
        return f"Box(size={self.size}, position={self.pose.position}, quaternion={self.pose.quaternion})"


class Cube(Primitive):
    """A movable box primitive usable as a pickable target object.

    Unlike :class:`Box` (which is wired up as static table geometry), ``Cube``
    carries the semantic/spatial annotations the layout and randomizers expect
    from a target object, so it can be spawned, placed and grasped like any
    mesh asset -- but with a caller-provided size instead of a fixed mesh.
    """

    def __init__(
        self,
        size=0.15,
        position=None,
        quaternion=None,
        name: str = "box",
    ) -> None:
        self.uid = "cube"
        self.label = "cube"
        self.name = name
        self.description = "a cube-shaped box"

        # Accept a scalar (uniform cube) or an explicit [x, y, z] extent.
        if np.isscalar(size):
            self.size = [float(size)] * 3
        else:
            self.size = [float(s) for s in size]

        if position is None:
            position = [0.0, 0.0, 0.5 * self.size[2]]
        if quaternion is None:
            quaternion = [1.0, 0.0, 0.0, 0.0]
        self.pose = Pose(position, quaternion)

        # SpatialAnnotated contract. A single stable pose: resting flat on a
        # surface, the centroid sits half the cube's height above it. Each
        # stable pose is [x, y, z, qw, qx, qy, qz] (see SpatialDR placement).
        self.stable_poses = [[0.0, 0.0, 0.5 * self.size[2], 1.0, 0.0, 0.0, 0.0]]
        self.keypoints = {}
        self.axes = {}
        self.canonical_grasps = []
        self.functional_grasps = {}

    def set_material(self, material: dict) -> None:
        self.material = material

    def to_dict(self) -> dict:
        return {
            "res_id": "primitive",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "size": self.size,
        }

    def __repr__(self) -> str:
        return f"Cube(size={self.size}, position={self.pose.position})"

@AssetManager.register("primitive")
class PrimitiveAsseManager(AssetManager):
    
    def __init__(self) -> None:
        ...

    def load(self, asset_id: str, *args, **kwargs) -> Asset:
        if asset_id == "box":
            return Box(*args, **kwargs)
        elif asset_id == "cube":
            return Cube(*args, **kwargs)
        else:
            raise ValueError(f"Unknown primitive asset_id: {asset_id}")

    def sample(self, exclude: list[str] | None = None) -> Asset:
        ...

    # def box(self, size, position, height, quaternion) -> Asset:
    #     return Box(size, position, height, quaternion)
    
