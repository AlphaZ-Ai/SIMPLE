# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from enum import Enum
class SIM_MODE(Enum):
    MUJOCO = "mujoco"
    ISAAC = "isaac"
    MUJOCO_ISAAC = "mujoco_isaac"

from gymnasium.envs.registration import register

register(
    id="simple/FrankaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"franka_tabletop_grasp_mp"},
)
register(
    id="simple/FrankaTabletopPickNPlaceMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"franka_tabletop_pick_n_place_mp"},
)
register(
    id="simple/AlohaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_grasp_mp"},
)
register(
    id="simple/AlohaTabletopHandoverMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_handover_mp"},
)
register(
    id="simple/VegaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"vega_tabletop_grasp_mp"},
)
register(
    id="simple/AlohaTabletopFindNGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_find_n_grasp_mp"},
)
register(
    id="simple/VegaTabletopFindNGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"vega_tabletop_find_n_grasp_mp"},
)

register(
    id="simple/G1TabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_grasp_mp"},
)
register(
    id="simple/G1TabletopPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_pick_n_place_mp"},

)

register(
    id="simple/G1InspireTabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_tabletop_grasp_mp"},
)
register(
    id="simple/G1TabletopHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_handover_mp"},
)


register(
    id="simple/G1WholebodyBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_mp"},
)
register(
    id="simple/G1WholebodyBendPickAndPlaceOnSofaMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_on_sofa_mp"},
)
register(
    id="simple/G1WholebodyTabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_grasp_mp"},
)
register(
    id="simple/G1InspireWholebodyLocomotionMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_wholebody_locomotion_mp"},
)
register(
    id="simple/G1InspireWholebodyPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_wholebody_pick_n_place_mp"},
)










register(
    id="simple/G1WholebodyPickAndBendPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_pick_and_bend_place_mp"},
)



register(
    id="simple/G1WholebodyTabletopHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_handover_mp"},
)




register(
    id="simple/G1WholebodyXMovePickTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_xmove_pick_teleop"},
)




register(
    id="simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_pick_and_place_and_hug_container_teleop"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_teleop"},
)
register(
    id="simple/G1WholebodyHandoverTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_handover_teleop"},
)

register(
    id="simple/G1WholebodyCloseDoorTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_close_door_teleop"},
)
register(
    id="simple/G1WholebodyOpenOvenTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_oven_teleop"},    
)
register(
    id="simple/G1WholebodyOpenFaucetTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_faucet_teleop"},
)
register(
    id="simple/G1WholebodyPushOfficeChairTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_push_office_chair_teleop"},
)
register(
    id="simple/G1WholebodyOpenTrashCanTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_trash_can_teleop"},
)
register(
    id="simple/G1WholebodyBendHandoverTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_bend_handover_teleop"},
)
register(
    id="simple/G1WholebodyBendPickAndPlaceTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_teleop"},
)
register(
    id="simple/G1WholebodyBendPickTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_bend_pick_teleop"},
)
# def task_register(task):
#     register(
#         id="FrankaTabletopGrasp-v0",
#         entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
#     )


# task_register()
from .base_dual_env import BaseDualSim
from .tabletop_grasp import TabletopGraspEnv
from .loco_manipulation import LocoManipulationEnv

register(
    id="simple/G1WholebodyXMoveBendCarryBoxSonic-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task": "g1_wholebody_xmove_bend_carry_box_sonic"},
)
