# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from typing import Type
from simple.core.task import Task
from simple.core.registry import RegistryMixin

class TaskRegistry(RegistryMixin[Task]):

    @classmethod
    def _base_type(cls) -> Type:
        return Task
