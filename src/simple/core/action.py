# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from copy import deepcopy

@dataclass
class ActionCmd:

    type: str
    parameters: dict | None = None

    def __init__(self, type: str, **parameters):
        self.type = type
        self.parameters = parameters

    def __getitem__(self, key):
        if self.parameters is None:
            # raise KeyError(f"No parameters set for this ActionCmd.")
            return None
        return self.parameters[key] if key in self.parameters else None

    def copy(self):
        """Deep copy the ActionCmd object."""
        params = deepcopy(self.parameters) if self.parameters is not None else {}
        return ActionCmd(self.type, **params)
