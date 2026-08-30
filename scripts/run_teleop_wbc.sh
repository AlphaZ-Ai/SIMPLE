#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# One-command launcher for the SONIC WBC teleop stack.
#
#   bash scripts/run_teleop_wbc.sh [ENV_ID] [extra teleop_wbc.py args...]
#
# Starts, in order:
#   1. SONIC controller (gear_sonic_deploy)   -> log file
#   2. pico manager (headset -> ZMQ)          -> log file
#   3. the sim (foreground, owns the viewer)
# One Ctrl+C tears the whole stack down.
#
# NOT covered here: the XRoboToolkit PC Service (GUI app the headset connects
# to) — start it separately and leave it running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ID="${1:-simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
shift || true
# Default arm pose for the Python robot models (pico manager FK calibration +
# sim-side recording FK): low | high | down.  The controller binary's standing
# pose is separate (policy_parameters.hpp default_angles — rebuild to change).
export G1_ELBOW_POSE="${G1_ELBOW_POSE:-down}"

# Auto-engage the SONIC policy after its init ramp (no ']' keypress needed);
# requires a controller binary built with the SONIC_AUTO_START patch.
# Set SONIC_AUTO_START=0 to restore manual engagement.
export SONIC_AUTO_START="${SONIC_AUTO_START:-1}"

LOGS="${TMPDIR:-/tmp}/teleop_wbc_logs"
mkdir -p "$LOGS"
cd "$ROOT_DIR"

# Supervisor state: a flag the trap raises so a deliberate Ctrl+C is never
# fought with a restart, and per-process restart caps so a genuinely broken
# controller cannot spin forever.
SHUTDOWN_FLAG="$LOGS/.shutting_down"
CONTROLLER_MAX_RESTARTS="${CONTROLLER_MAX_RESTARTS:-5}"
MANAGER_MAX_RESTARTS="${MANAGER_MAX_RESTARTS:-5}"
rm -f "$SHUTDOWN_FLAG"

# On exit/Ctrl+C: TERM the controller binary by name first (it shrugs off
# the group signal), then nuke the whole process group.  Two rules learned
# the hard way: `set +e` first (a pkill that matches nothing returns 1 and
# would abort the trap mid-cleanup), and `kill 0` last (it signals this
# script too).
trap 'trap - TERM INT EXIT; set +e; touch "$SHUTDOWN_FLAG" 2>/dev/null; echo; echo "Stopping controller + manager...";
      pkill -TERM -f "g1_deploy_onnx_ref" 2>/dev/null;
      pkill -TERM -f "pico_manager_thread_server" 2>/dev/null;
      sleep 1;
      pkill -KILL -f "g1_deploy_onnx_ref" 2>/dev/null;
      kill 0' TERM INT EXIT

# Preflight: a stack left over from an earlier run keeps publishing on the
# same DDS topics and ZMQ port as this one.  Two publishers on rt/lowcmd is
# not a warning-level problem — the sim follows whichever arrives last, so
# the robot behaves erratically and every diagnosis lies.  Clear them.
STRAYS="g1_deploy_onnx_ref|pico_manager_thread_server|cli/teleop_wbc.py|cli/teleop_decoupled_wbc.py"
# Never signal ourselves or an ancestor: a shell whose command line merely
# mentions these names matches -f, and a blind pkill then kills the very
# terminal running this script.
self_chain=" $$ "
_p="$PPID"
while [[ -n "$_p" && "$_p" != "0" && "$_p" != "1" ]]; do
  self_chain+="$_p "
  _p="$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')"
done
stray_pids=()
for pid in $(pgrep -f "$STRAYS" 2>/dev/null || true); do
  [[ "$self_chain" == *" $pid "* ]] && continue
  stray_pids+=("$pid")
done
if (( ${#stray_pids[@]} )); then
  echo "Found processes from a previous run — clearing them first:"
  ps -o pid=,args= -p "$(IFS=,; echo "${stray_pids[*]}")" 2>/dev/null | sed 's/^/    /'
  kill -TERM "${stray_pids[@]}" 2>/dev/null || true
  sleep 2
  kill -KILL "${stray_pids[@]}" 2>/dev/null || true
  sleep 1
fi
if command -v ss >/dev/null && ss -tln 2>/dev/null | grep -q ':13579 '; then
  echo "WARNING: port 13579 is still held by something — the headset video"
  echo "         connects to whatever owns it, which may not be this run."
fi

if ! pgrep -f RoboticsServiceProcess >/dev/null; then
  echo "WARNING: XRoboToolkit PC Service is not running — VR input will be dead."
  echo "         Start it separately: /opt/apps/roboticsservice/runService.sh"
fi

echo "[1/3] SONIC controller  (log: $LOGS/controller.log)"
# Self-heal: if the controller binary is not built (fresh clone, or build/ wiped),
# build it now.  build_sonic_controller.sh is idempotent and needs no sudo/network
# once the deps are in ~/tools; it also writes ~/tools/sonic_env.sh.
CONTROLLER_BIN="$ROOT_DIR/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
if [[ ! -x "$CONTROLLER_BIN" ]]; then
  echo "  controller binary missing — building it (scripts/build_sonic_controller.sh)..."
  bash "$ROOT_DIR/scripts/build_sonic_controller.sh" || { echo "  controller build FAILED"; exit 1; }
fi
[[ -f "$HOME/tools/sonic_env.sh" ]] || { echo "  ~/tools/sonic_env.sh still missing after build"; exit 1; }

if [[ -z "${CUDAToolkit_ROOT:-}" && -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  export CUDAToolkit_ROOT="$CUDA_HOME"
fi

start_controller() {
  (
    # upstream setup_env.sh/deploy.sh are not strict-mode-safe
    set +e +u +o pipefail
    source "$HOME/tools/sonic_env.sh"
    cd "$ROOT_DIR/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    source scripts/setup_env.sh
    # deploy.sh asks "Proceed? [Y/n]" — empty answer means yes
    ./deploy.sh sim --input-type zmq_manager <<< ""
  ) >> "$LOGS/controller.log" 2>&1 &
  CTRL_PID=$!
}
: > "$LOGS/controller.log"   # truncate once; restarts append so crash reasons accumulate
start_controller
sleep "${CONTROLLER_WAIT:-10}"
kill -0 "$CTRL_PID" 2>/dev/null || {
  echo "  controller DIED during startup — last log lines:"; tail -15 "$LOGS/controller.log"; exit 1; }

# Full-body teleop is POSE mode (encoder "smpl"): the operator's whole SMPL
# pose — legs included — maps onto the robot.  That is the point of WBC vs
# DWBC, so the manager auto-enters POSE on engage (same CALIB_FULL, no extra
# A+X).  Use A+X at runtime to toggle to PLANNER (stick locomotion) and back.
# --cuda runs the manager's SMPL compute on the GPU.
MGR_ARGS=(--auto_pose --cuda)

echo "[2/3] pico manager ${MGR_ARGS[*]}  (log: $LOGS/pico_manager.log)"
start_manager() {
  .venv/bin/python third_party/gear_sonic/scripts/pico_manager_thread_server.py --manager \
    "${MGR_ARGS[@]}" >> "$LOGS/pico_manager.log" 2>&1 &
  MGR_PID=$!
}
: > "$LOGS/pico_manager.log"
start_manager
sleep 3
kill -0 "$MGR_PID" 2>/dev/null || {
  echo "  pico manager DIED during startup — last log lines:"; tail -15 "$LOGS/pico_manager.log"; exit 1; }
# The XRT SDK prints "device found" once the PC service hands it the headset.
# Without it the manager runs but never sees a single button press — the sim
# then sits at "policy NOT ENGAGED" with no clue why.  Warn loudly up front.
device_found=0
for _ in $(seq 1 10); do
  if grep -q "device found" "$LOGS/pico_manager.log" 2>/dev/null; then
    device_found=1; break
  fi
  sleep 1
done
if (( ! device_found )); then
  echo "  WARNING: manager connected to the XRoboToolkit service but NO HEADSET"
  echo "           device appeared within 10 s — A+X will do nothing."
  echo "           Check the headset is connected in the service, or restart it:"
  echo "           /opt/apps/roboticsservice/runService.sh"
fi

# HEADLESS=1 skips the desktop viewer — teleop still works, the operator
# sees the robot's eye view streamed into the headset.
VIEW_FLAG="--no-headless"
[[ "${HEADLESS:-0}" == "1" ]] && VIEW_FLAG="--headless"

# The manager announces every mode change ("[Manager] ...", "[PlannerLoop]
# Mode -> ..."), which is exactly what the operator needs to see — but it
# goes to its log file.  Mirror those lines into this terminal.
( tail -n0 -F "$LOGS/pico_manager.log" 2>/dev/null \
    | grep --line-buffered -E "\[Manager\]|\[PlannerLoop\]|Mode ->" \
    | sed -u 's/^/[mgr] /' ) &

# Supervisor: the controller exits on its own when its 500 ms rt/lowstate
# watchdog trips or a safety check fails, and the manager can exit on a stop
# chord — either one leaves the robot frozen for the rest of the session.
# Watch both and bring back whichever dies (up to a cap), surfacing the reason.
# The trap raises $SHUTDOWN_FLAG first, so a deliberate Ctrl+C is never fought.
supervise() {
  set +e
  local cr=0 mr=0
  while [[ ! -f "$SHUTDOWN_FLAG" ]]; do
    if ! kill -0 "$CTRL_PID" 2>/dev/null; then
      [[ -f "$SHUTDOWN_FLAG" ]] && break
      if (( cr >= CONTROLLER_MAX_RESTARTS )); then
        echo "[supervisor] controller hit the restart cap ($CONTROLLER_MAX_RESTARTS) — leaving it down. Last log:"
        tail -8 "$LOGS/controller.log" | sed 's/^/    /'
        break
      fi
      cr=$((cr + 1))
      echo "[supervisor] controller exited — restarting ($cr/$CONTROLLER_MAX_RESTARTS). Reason:"
      tail -6 "$LOGS/controller.log" 2>/dev/null | sed 's/^/    /'
      start_controller
      echo "[supervisor] controller back up (pid $CTRL_PID) — re-engage from the headset (A+X)."
    fi
    if ! kill -0 "$MGR_PID" 2>/dev/null; then
      [[ -f "$SHUTDOWN_FLAG" ]] && break
      if (( mr >= MANAGER_MAX_RESTARTS )); then
        echo "[supervisor] pico manager hit the restart cap ($MANAGER_MAX_RESTARTS) — leaving it down."
        break
      fi
      mr=$((mr + 1))
      echo "[supervisor] pico manager exited — restarting ($mr/$MANAGER_MAX_RESTARTS)."
      start_manager
    fi
    sleep 2
  done
}
# supervise &

echo "[3/3] sim: $ENV_ID $VIEW_FLAG  (Ctrl+C here stops everything)"
# -u: the agent's crane/controller telemetry must appear live, not sit in a
# buffer that is lost when the run is killed.
# DEBUG=1: open a debugpy port and block until VS Code attaches ("Attach to
# teleop_wbc" in .vscode/launch.json).  While paused at a breakpoint the sim
# stops publishing rt/lowstate, so the controller's watchdog will trip and the
# supervisor restarts it — re-engage from the headset (A+X) after resuming.
DEBUGPY=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUGPY=(-m debugpy --listen "${DEBUG_PORT:-5678}" --wait-for-client)
  echo "[debug] debugpy ON — sim will PAUSE on port ${DEBUG_PORT:-5678} until VS Code attaches"
  echo "[debug] attach with the \"Attach to teleop_wbc\" config (F5) in .vscode/launch.json"
fi
# Recording is on by default; it goes BEFORE "$@" so a user-passed
# --no-record (last flag wins in typer) can still turn it off.
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1 DISPLAY=":1" \
  .venv/bin/python -u "${DEBUGPY[@]}" src/simple/cli/teleop_wbc.py "$ENV_ID" "$VIEW_FLAG" --record "$@"
