# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import gymnasium as gym

from simple.envs import *

for spec in gym.registry.values():
    if spec.id.startswith("simple"):
        print(spec.id)
