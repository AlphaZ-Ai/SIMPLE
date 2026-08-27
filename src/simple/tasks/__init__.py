# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# import gymnasium
# from gymnasium.envs.registration import register, registry

# def register_task(task):
#     # from simple.envs.tabletop_grasp import TabletopGraspEnv

#     # register(
#     #     id="FrankaTabletopGrasp-v0",
#     #     entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
#     # )

#     # env_spec = registry.get("FrankaTabletopGrasp-v0")

#     # register(entry_point=)
#     ...

from .franka_tabletop_grasp_mp import FrankaTabletopGraspTaskMP
from .franka_tabletop_pick_n_place_mp import FrankaTabletopPickNPlaceTaskMP
from .aloha_tabletop_grasp_mp import AlohaTabletopGraspTaskMP
from .aloha_tabletop_handover_mp import AlohaTabletopHandoverTaskMP
from .aloha_tabletop_find_n_grasp_mp import AlohaTabletopFindNGraspTaskMP
from .vega_tabletop_grasp_mp import VegaTabletopGraspMP


from .g1_tabletop_grasp_mp import G1TabletopGraspMP
from .g1_inspire_tabletop_grasp_mp import G1InspireTabletopGraspMP
from .g1_tabletop_pick_n_place_mp import G1TabletopPickNPlaceMP
from .g1_tabletop_handover_mp import G1TabletopHandoverMP
from .g1_inspire_wholebody_locomotion_mp import G1InspireWholebodyLocomotionTaskMP
from .g1_inspire_wholebody_pick_n_place_mp import G1InspireWholebodyPickNPlaceMP
from .g1_wholebody_bend_pick_mp import G1WholebodyBendPickMP
from .g1_wholebody_bend_pick_and_place_on_sofa_mp import G1WholebodyBendPickAndPlaceOnSofaMP
from .g1_wholebody_tabletop_grasp_mp import G1WholebodyTabletopGraspMP
from .g1_wholebody_pick_and_bend_place_mp import G1WholebodyPickAndBendPlaceMP  
from .g1_wholebody_tabletop_handover_mp import G1WholebodyTabletopHandoverMP


from .g1_wholebody_locomotion_pick_between_tables_teleop import G1WholebodyLocomotionPickBetweenTablesTaskTeleop
from .g1_wholebody_handover_teleop import G1WholebodyHandoverTeleop
from .g1_wholebody_pick_and_place_and_hug_container_teleop import G1WholebodyPickAndPlaceAndHugContainerTaskTeleop
from .g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from .g1_wholebody_close_door_teleop import G1WholebodyCloseDoorTaskTeleop
from .g1_wholebody_open_oven_teleop import G1WholebodyOpenOvenTaskTeleop
from .g1_wholebody_open_faucet_teleop import G1WholebodyOpenFaucetTaskTeleop
from .g1_wholebody_push_office_chair_teleop import G1WholebodyPushOfficeChairTaskTeleop
from .g1_wholebody_open_trash_can_teleop import G1WholebodyOpenTrashCanTaskTeleop
from .g1_wholebody_bend_handover_teleop import G1WholebodyBendHandoverTeleop
from .g1_wholebody_bend_pick_and_place_teleop import G1WholebodyBendPickAndPlaceTeleop
from .g1_wholebody_bend_pick_teleop import G1WholebodyBendPickTeleop
from .g1_wholebody_xmove_bend_carry_box_sonic import G1WholebodyXMoveBendCarryBoxTaskSonic
