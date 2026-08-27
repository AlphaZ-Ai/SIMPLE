# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from .franka_fr3 import FrankaResearch3
from .aloha import Aloha
from .vega import Vega1
from .protocols import WristCamMountable
from .g1 import G1
from .g1_inspire import G1Inspire
from .g1_wholebody import G1Wholebody
from .g1_inspire_wholebody import G1InspireWholebody

try:
    from .g1_sonic import G1Sonic
except ModuleNotFoundError:
    # Keep the core SIMPLE eval path importable when sonic-only dependencies
    # such as pinocchio are not installed in the current environment.
    G1Sonic = None
