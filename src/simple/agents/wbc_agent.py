# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod

class WholeBodyControlAgent(ABC):
    ...

    @abstractmethod
    def get_action(self, observation, instruction=None, **kwargs):
        raise NotImplementedError
    

    def reset(self, **kwargs):
        ...
