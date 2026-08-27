# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.sensors.config import CameraCfg
from dataclasses import dataclass
from typing import Type

class CameraDR(Randomizer):

    def __init__(self, cfg: "CameraDRCfg") -> None:
        # self.cam_id = cam_id
        # self._inner_state = {} # FIXME
        super().__init__(cfg)

    def __call__(self, split: str, cam_cfg: CameraCfg) -> CameraCfg:
        return cam_cfg

    # def apply(self, split: str, cam_cfg: CameraCfg) -> CameraCfg:
    #     # TODO
    #     return cam_cfg
    
@dataclass
class CameraDRCfg(RandomizerCfg):
    cam_id: str | None = None
    randmizer_class: Type[Randomizer] = CameraDR
