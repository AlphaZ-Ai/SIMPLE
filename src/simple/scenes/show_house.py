# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from simple.core.scene import TabletopScene
from simple.core.asset import Asset

class ShowHouse(TabletopScene):

    def __init__(self, table: Asset) -> None:
        self.uid = "show_house"
        self.table = table
