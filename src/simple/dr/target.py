# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.asset import Asset
from simple.assets import AssetManager
from dataclasses import dataclass
from typing import Type, Dict, Any

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simple.dr.target import TargetDr

class TargetDR(Randomizer):

    # asset_id: str
    # object: "Asset"

    def __init__(self, cfg: "TargetDRCfg") -> None:
        super().__init__(cfg)
        self.res_id, self.obj_id = cfg.asset_id.split(':')
    
    def state_dict(self) -> Dict[str, Any]:
        # state_dict = {}
        # for k, v in self._inner_state.items():
        #     state_dict[k] = v.to_dict()
        # return state_dict
        return self._inner_state.to_dict() if self._inner_state else {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Older recordings (generic Entity.to_dict, e.g. primitive Cube targets)
        # carry no res_id — fall back to this randomizer's configured asset.
        res_id = state_dict.get("res_id", self.res_id)
        uid = state_dict.get("uid", self.obj_id)
        load_kwargs = {}
        # Primitives record their extent; mesh assets don't (their loaders
        # would reject the kwarg anyway).
        if state_dict.get("size") is not None:
            load_kwargs["size"] = state_dict["size"]
        self._inner_state = AssetManager.get(res_id).load(uid, **load_kwargs)

    def __call__(self, split: str, **kwargs) -> Asset:
        
        # res_id, obj_id = self.asset_id.split(':')
        # Primitive targets (e.g. "primitive:cube") take a caller-provided size;
        # mesh loaders don't accept the kwarg, so only forward it when set.
        load_kwargs = {}
        if getattr(self.cfg, "size", None) is not None:
            load_kwargs["size"] = self.cfg.size
        asset = AssetManager.get(self.res_id).load(self.obj_id, **load_kwargs)
        # return self.object

        # if self.rand_stable_pose:
        #     ...

        return super()._transient(asset)

@dataclass
class TargetDRCfg(RandomizerCfg):
    asset_id: str | None = None  # e.g., "res_id:obj_id"
    # Size for primitive targets (e.g. "primitive:cube"): a scalar for a uniform
    # cube or an [x, y, z] extent in meters. Ignored by mesh assets.
    size: Any | None = None
    randmizer_class: Type[Randomizer] = TargetDR

    # def __init__(self, asset_id) -> None:
    #     if asset_id is None or ':' not in asset_id:
    #         raise ValueError("Invalid asset_id format. Expected 'res_id:obj_id'.")
    #     self.res_id, self.obj_id = asset_id.split(':')


    
