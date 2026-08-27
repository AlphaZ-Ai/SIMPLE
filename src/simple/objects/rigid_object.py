# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from pathlib import Path

class RigidObject:

    usd_path: Path
    visual_mesh_path: Path
    collision_mesh_path: Path | None

