# Room-to-room navigation for the G1 (Dex1) in SIMPLE

SIMPLE simulates the robot and house; the CMU autonomy stack
(`third_party/humanoid-planner/external/nav/Navigation-Physical-Experiment`,
ROS2 Jazzy) plans and drives it.

```
SIMPLE (Python 3.10 venv)                         ROS2 Jazzy (system Python 3.12)
 run_nav_sim.py                                    ros2_bridge_node.py            simple_nav.launch.py
  G1 + Dex1, AMO walking, MuJoCo physics  --scan, pose-->  /registered_scan  ----->  terrain analysis
  HSSD house walls/furniture as collision               /state_estimation  ----->  local planner, path follower
  simulated Mid-360 (src/simple/nav/lidar.py) <--cmd_vel--  /cmd_vel  <-----------  FAR route planner (/goal_point)
```

The two sides talk over a local socket because rclpy is built for the system
Python and SIMPLE runs on its own venv.

## One-time setup

1. ROS2 Jazzy: `~/install_ros2_jazzy.sh`, then `source ~/ros_jazzy.sh` in any
   shell that uses ROS (it switches conda off; conda's Python breaks rclpy).
2. House collision for MuJoCo (all 50 HSSD scenes, ~7 min):
   ```
   uvx --python 3.10 --from usd-core --with numpy --with scipy --with pyyaml \
       python scripts/build_hssd_collision.py
   ```
3. Build the nav stack outside the submodule:
   ```
   source ~/ros_jazzy.sh && mkdir -p ~/alphaz_ws/nav_ws && cd ~/alphaz_ws/nav_ws
   colcon build --base-paths ~/alphaz_ws/SIMPLE/third_party/humanoid-planner/external/nav/Navigation-Physical-Experiment/src \
       --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
       --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2
   ```
   SLAM is skipped: the bridge publishes the simulator's ground-truth pose.

## Run it interactively

Three terminals, from the SIMPLE root:

```
# 1. simulator (add --sim-mode mujoco_isaac --no-headless to watch in IsaacSim)
.venv/bin/python scripts/nav/run_nav_sim.py --spawn -0.7 1.3 180

# 2. bridge
source ~/ros_jazzy.sh && python3 scripts/nav/ros2_bridge_node.py

# 3. nav stack + RViz; set a goal with the "Goalpoint" tool
source ~/ros_jazzy.sh && source ~/alphaz_ws/nav_ws/install/setup.bash
ros2 launch scripts/nav/simple_nav.launch.py
```

## Headless trials, logs and videos

```
scripts/nav/run_nav_trial.sh --goal 0.0 -5.5 --seed 1 --spawn "-0.7 1.3 180" --tag lower_room
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/nav/render_nav_run.py data/output/nav_runs/lower_room_seed1.npz
```

A trial prints `RESULT REACHED / FELL / TIMEOUT` and saves the full MuJoCo
state at every step. `render_nav_run.py` replays it in IsaacSim (IsaacSim only
mirrors MuJoCo, so the replay is exact) with a chase camera next to the head
camera, and draws a top-down map of the path, the FAR waypoints and any
contact with the house (`--map-only` skips IsaacSim).

## Things worth knowing

- **Heading frame.** AMO measures yaw relative to the pose it first saw, which
  is the task's spawn, not `--spawn`. `NavSimServer.loco_action` sends headings
  in AMO's frame; sending world headings after a teleport made the robot spin.
- **Speed.** AMO reaches ~42% of the commanded forward speed, so `/cmd_vel` is
  scaled up (`AMO_SPEED_GAIN`); top speed is ~0.3 m/s.
- **Lidar.** The real G1 mounts the Mid-360 upside down; the simulated one does
  too, so terrain analysis sees the floor.
- **Startup.** Until a goal arrives the local planner drives to its `goalX/goalY`
  parameter (0, 0); the bridge publishes the robot's own position as the first
  waypoint so it stands still.
- **Collision is approximate.** Furniture is solid from the floor up (5 cm grid)
  and doors count as open. Fine for navigation, not for manipulation.
- `G1WholebodyBendPickMP` (the env used here) ignores `scene_uid`; the nav sim
  sets the scene through `task.dr_cfgs["scene"].room_choices`.
