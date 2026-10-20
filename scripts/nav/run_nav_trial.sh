#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
#
# One headless navigation trial: SIMPLE nav sim + ROS2 bridge + CMU autonomy
# stack, send a goal, wait for REACHED / FELL / TIMEOUT, save a run log.
#
#   scripts/nav/run_nav_trial.sh --goal 0.0 -5.5 --seed 1 --spawn "-0.7 1.3 180" --tag lower_room
#
# Navigate to the task's table in another room, then pick and place there:
#
#   scripts/nav/run_nav_trial.sh --mission pick_place --env-id simple/G1WholebodyTabletopGraspMP-v0 \
#       --scene hssd:scene2 --target primitive:cube_0.05 --spawn "0.5 8.0 -90" --tag fetch
#
# Run log:   data/output/nav_runs/<tag>_seed<seed>.npz  (render with render_nav_run.py)
# Text logs: data/output/nav_runs/logs/<tag>_seed<seed>_{sim,bridge,stack}.log
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GX=0.0; GY=-5.5; SEED=0; SPAWN=""; TAG=run; DUR=240; SCENE=hssd:scene0
ENV_ID=simple/G1WholebodyBendPickMP-v0; MISSION=""; TARGET=""
NAV_WS="${NAV_WS:-$HOME/alphaz_ws/nav_ws}"
while [ $# -gt 0 ]; do
  case $1 in
    --goal) GX=$2; GY=$3; shift 3 ;;
    --seed) SEED=$2; shift 2 ;;
    --spawn) SPAWN=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --timeout) DUR=$2; shift 2 ;;
    --scene) SCENE=$2; shift 2 ;;
    --env-id) ENV_ID=$2; shift 2 ;;
    --mission) MISSION=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

OUT=data/output/nav_runs; LOGS=$OUT/logs; mkdir -p "$LOGS"
NAME=${TAG}_seed${SEED}
SIM_LOG=$LOGS/${NAME}_sim.log

# every component in its own process group, so it can be stopped as a whole
setsid .venv/bin/python scripts/nav/run_nav_sim.py --env-id "$ENV_ID" --scene-uid "$SCENE" --seed "$SEED" \
  ${SPAWN:+--spawn $SPAWN} --goal "$GX" "$GY" ${MISSION:+--mission $MISSION} ${TARGET:+--target-object $TARGET} \
  --log-path "$OUT/$NAME.npz" > "$SIM_LOG" 2>&1 &
SIM=$!
for _ in $(seq 1 180); do grep -q "waiting for bridge" "$SIM_LOG" && break; sleep 1; done
grep -E "\[nav_sim\] (scene|goal|WARNING)" "$SIM_LOG"

set +u  # ROS setup scripts read unset variables
source ~/ros_jazzy.sh > /dev/null
source "$NAV_WS/install/setup.bash"
set -u
setsid python3 scripts/nav/ros2_bridge_node.py > "$LOGS/${NAME}_bridge.log" 2>&1 &
BR=$!
setsid ros2 launch scripts/nav/simple_nav.launch.py rviz:=false > "$LOGS/${NAME}_stack.log" 2>&1 &
ST=$!

# the sim moves goals that sit in furniture to the nearest free spot; use that one
G=$(grep -oP "\[nav_sim\] goal=\(\K[^)]+" "$SIM_LOG" | head -1)
if [ -n "$G" ]; then GX=${G%,*}; GY=${G#*,}; fi
sleep 15  # let the stack start and the robot settle
# a single --once publish can be lost before DDS discovery completes; repeat it
ros2 topic pub -r 1 -t 5 /goal_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: map}, point: {x: $GX, y: $GY, z: 0.0}}" > /dev/null
echo "goal sent: ($GX, $GY)"

python3 - "$DUR" "$GX" "$GY" "$SIM_LOG" "$MISSION" <<'PY'
import math, sys, time
import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry, Path

rclpy.init()
n = rclpy.create_node("nav_trial_monitor")
st = {"odom": None, "cmd": None, "wp": None, "path": None}
n.create_subscription(Odometry, "/state_estimation", lambda m: st.update(odom=m), 5)
n.create_subscription(TwistStamped, "/cmd_vel", lambda m: st.update(cmd=m), 5)
n.create_subscription(PointStamped, "/way_point", lambda m: st.update(wp=m), 5)
n.create_subscription(Path, "/path", lambda m: st.update(path=m), 5)
dur, gx, gy = map(float, sys.argv[1:4])
sim_log, mission = sys.argv[4], sys.argv[5]
t0, last, result = time.time(), 0.0, "TIMEOUT"
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.1)
    if st["odom"] is None:
        continue
    p = st["odom"].pose.pose.position
    if "FELL" in open(sim_log).read():  # the sim judges falls (it knows the floor height)
        result = f"FELL at t={time.time() - t0:.0f}s pos=({p.x:+.2f},{p.y:+.2f})"
        break
    if mission:
        # the sim runs the mission; wait for its verdict
        text = open(sim_log).read()
        if "MISSION_DONE" in text:
            result = text[text.index("MISSION_DONE"):].splitlines()[0] + f" ({time.time() - t0:.0f}s)"
            break
    elif math.hypot(p.x - gx, p.y - gy) < 0.4:
        result = f"REACHED in {time.time() - t0:.0f}s"
        break
    if time.time() - last > 5:
        c, w, pa = st["cmd"], st["wp"], st["path"]
        cs = f"cmd v={c.twist.linear.x:+.2f} w={c.twist.angular.z:+.2f}" if c else "cmd none"
        ws = f"waypoint=({w.point.x:+.1f},{w.point.y:+.1f})" if w else "waypoint none"
        ps = f"path {len(pa.poses)} pts" if pa is not None and len(pa.poses) else "path empty"
        print(f"t={time.time() - t0:5.0f}s pos=({p.x:+.2f},{p.y:+.2f}) z={p.z:.2f} {cs} {ws} {ps}", flush=True)
        last = time.time()
print("RESULT", result, flush=True)
PY

kill -TERM -- -$SIM 2>/dev/null   # sim first: it saves its run log
for pid in $ST $BR; do kill -INT -- -$pid 2>/dev/null; done
for _ in $(seq 1 30); do kill -0 $SIM 2>/dev/null || break; sleep 1; done
grep -E "saved run log" "$SIM_LOG"
for pid in $ST $BR $SIM; do kill -KILL -- -$pid 2>/dev/null; done
true
