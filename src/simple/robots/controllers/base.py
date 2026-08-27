# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from simple.core.controller import Controller, ControllerCfg


class BaseController(Controller):
    def __init__(self, controller_cfg):
        super().__init__(controller_cfg)


class BaseControllerCfg(ControllerCfg):
    clazz: type[Controller] = BaseController

    
