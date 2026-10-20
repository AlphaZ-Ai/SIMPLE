# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
CMU autonomy stack (Navigation-Physical-Experiment) for the simulated G1 in
SIMPLE. Same nodes as its system_simulation_with_route_planner.launch, minus
Unity / vehicle_simulator / joystick (SIMPLE + scripts/nav/ros2_bridge_node.py
provide /registered_scan, /state_estimation and consume /cmd_vel), with
parameters for a walking humanoid instead of the wheeled T-Bot:

  - forward + turn only ("standard" config, no mecanum strafing, no reversing)
  - speeds limited to what AMO reaches (~0.35 m/s)
  - lidar ("sensor" frame) at ~1.23 m above the floor instead of ~0.75 m

Build the stack once (see scripts/nav/README.md), then:

    source ~/ros_jazzy.sh && source ~/alphaz_ws/nav_ws/install/setup.bash
    ros2 launch scripts/nav/simple_nav.launch.py

Send goals with the "Goalpoint" tool in RViz (publishes /goal_point).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SENSOR_HEIGHT = 1.23   # Mid-360 above the floor on a standing G1 (simple.nav.lidar)
ROBOT_WIDTH = 0.6      # shoulder-to-shoulder incl. arms
ROBOT_LENGTH = 0.45
MAX_SPEED = 0.35       # m/s, see AMO_SPEED_GAIN in simple/nav/sim_server.py
AUTONOMY_SPEED = 0.3


def generate_launch_description():
    lp_share = get_package_share_directory("local_planner")
    far_share = get_package_share_directory("far_planner")
    standard_cfg = os.path.join(lp_share, "config", "standard.yaml")
    rviz = LaunchConfiguration("rviz")

    local_planner = Node(
        package="local_planner", executable="localPlanner", name="localPlanner", output="screen",
        parameters=[standard_cfg, {
            "pathFolder": os.path.join(lp_share, "paths"),
            "vehicleLength": ROBOT_LENGTH, "vehicleWidth": ROBOT_WIDTH,
            "sensorOffsetX": 0.0, "sensorOffsetY": 0.0,
            "twoWayDrive": False,
            "laserVoxelSize": 0.05, "terrainVoxelSize": 0.2,
            "useTerrainAnalysis": True, "checkObstacle": True, "checkRotObstacle": False,
            "adjacentRange": 3.5,
            "obstacleHeightThre": 0.15, "groundHeightThre": 0.05,
            "costHeightThre1": 0.15, "costHeightThre2": 0.1, "useCost": False,
            "slowPathNumThre": 5, "slowGroupNumThre": 1, "pointPerPathThre": 2,
            # points relative to the sensor: from the floor up to just above the head
            "minRelZ": -(SENSOR_HEIGHT + 0.1), "maxRelZ": 0.3,
            "maxSpeed": MAX_SPEED,
            "dirWeight": 0.02, "dirThre": 90.0, "dirToVehicle": False,
            "pathScale": 0.875, "minPathScale": 0.675, "pathScaleStep": 0.1, "pathScaleBySpeed": True,
            "minPathRange": 0.8, "pathRangeStep": 0.6, "pathRangeBySpeed": True, "pathCropByGoal": True,
            "autonomyMode": True, "autonomySpeed": AUTONOMY_SPEED,
            "joyToSpeedDelay": 2.0, "joyToCheckObstacleDelay": 5.0,
            "goalClearRange": 0.35, "goalBehindRange": 0.35,
            "freezeAng": 90.0, "freezeTime": 0.0,
            "goalX": 0.0, "goalY": 0.0,
        }],
    )

    path_follower = Node(
        package="local_planner", executable="pathFollower", name="pathFollower", output="screen",
        parameters=[standard_cfg, {
            "realRobot": False,
            "sensorOffsetX": 0.0, "sensorOffsetY": 0.0, "pubSkipNum": 1,
            "twoWayDrive": False,
            "lookAheadDis": 0.5,
            "maxSpeed": MAX_SPEED, "maxAccel": 1.0,
            # AMO turns at up to ~0.6 rad/s; standard.yaml allows 150 deg/s
            "yawRateGain": 3.0, "stopYawRateGain": 3.0, "maxYawRate": 35.0,
            # standard.yaml stops to re-align beyond 0.15 rad; a walking robot's
            # heading wobbles more than that, which made it crawl
            "dirDiffThre": 0.5,
            "switchTimeThre": 1.0, "omniDirDiffThre": 1.5, "slowDwnDisThre": 0.875,
            "useInclRateToSlow": False, "inclRateThre": 120.0,
            "slowRate1": 0.25, "slowRate2": 0.5, "slowRate3": 0.75, "slowTime1": 2.0, "slowTime2": 2.0,
            "useInclToStop": False, "inclThre": 45.0, "stopTime": 5.0,
            "noRotAtStop": False, "noRotAtGoal": True,
            "autonomyMode": True, "autonomySpeed": AUTONOMY_SPEED, "joyToSpeedDelay": 2.0,
        }],
    )

    terrain_analysis = Node(
        package="terrain_analysis", executable="terrainAnalysis", name="terrainAnalysis", output="screen",
        parameters=[{
            "scanVoxelSize": 0.05, "decayTime": 1.0, "noDecayDis": 1.75, "clearingDis": 8.0,
            "useSorting": True, "quantileZ": 0.25, "considerDrop": False,
            "limitGroundLift": False, "maxGroundLift": 0.15,
            "clearDyObs": True, "minDyObsDis": 0.14, "absDyObsRelZThre": 0.2,
            # upside-down Mid-360 on the G1 head looks -52..+7 deg
            "minDyObsVFOV": -52.0, "maxDyObsVFOV": 7.0,
            "minDyObsPointNum": 1, "minOutOfFovPointNum": 10,
            "obstacleHeightThre": 0.15, "noDataObstacle": False, "noDataBlockSkipNum": 0,
            "minBlockPointNum": 10, "vehicleHeight": 1.5,
            "voxelPointUpdateThre": 100, "voxelTimeUpdateThre": 2.0,
            "minRelZ": -(SENSOR_HEIGHT + 0.3), "maxRelZ": 0.3, "disRatioZ": 0.2,
        }],
    )

    terrain_analysis_ext = Node(
        package="terrain_analysis_ext", executable="terrainAnalysisExt", name="terrainAnalysisExt", output="screen",
        parameters=[{
            "scanVoxelSize": 0.1, "decayTime": 4.0, "noDecayDis": 0.0, "clearingDis": 30.0,
            "useSorting": True, "quantileZ": 0.1, "vehicleHeight": 1.5,
            "voxelPointUpdateThre": 100, "voxelTimeUpdateThre": 2.0,
            "lowerBoundZ": -2.5, "upperBoundZ": 1.0, "disRatioZ": 0.1,
            "checkTerrainConn": True, "terrainConnThre": 0.5,
            "terrainUnderVehicle": -(SENSOR_HEIGHT - 0.1),
            "ceilingFilteringThre": 2.0, "localTerrainMapRadius": 4.0,
        }],
    )

    sensor_scan_generation = Node(
        package="sensor_scan_generation", executable="sensorScanGeneration",
        name="sensorScanGeneration", output="screen",
    )

    sensor_to_vehicle = Node(
        package="tf2_ros", executable="static_transform_publisher", name="vehicleTransPublisher",
        arguments=["0", "0", "0", "0", "0", "0", "sensor", "vehicle"],
    )

    far_planner = Node(
        package="far_planner", executable="far_planner", name="far_planner", output="screen",
        parameters=[os.path.join(far_share, "config", "indoor.yaml"), {
            "vehicle_height": SENSOR_HEIGHT, "robot_dim": ROBOT_WIDTH,
        }],
        remappings=[
            ("/odom_world", "/state_estimation"),
            ("/terrain_cloud", "/terrain_map_ext"),
            ("/scan_cloud", "/terrain_map"),
            ("/terrain_local_cloud", "/registered_scan"),
        ],
    )
    graph_decoder = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory("graph_decoder"), "launch", "decoder.launch")))

    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(far_share, "rviz", "default.rviz")],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        terrain_analysis, terrain_analysis_ext, sensor_scan_generation,
        local_planner, path_follower, sensor_to_vehicle,
        far_planner, graph_decoder, rviz_node,
    ])
