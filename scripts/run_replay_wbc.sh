#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# One-command launcher for the SONIC WBC *physics replay* stack.
#
#   bash scripts/run_replay_wbc.sh [ENV_ID] [extra replay_wbc.py args...]
#
# Streams each recorded 64-dim WBC token through the live SONIC controller so
# physics plays the motion out (as opposed to the kinematic teleport replay).
#
# Unlike teleop there is NO pico manager and NO headset: the replay CLI itself
# injects the recorded tokens over ZMQ (it binds the pico_manager PUB role) and
# drives the controller's engage -> token-mode handshake.  So this launches only
# two things:
#   1. SONIC controller (gear_sonic_deploy, zmq_manager)  -> log file
#   2. the replay sim (foreground)
# One Ctrl+C tears the whole stack down.
#
# Common use:
#   bash scripts/run_replay_wbc.sh simple/G1WholebodyBendPick-v1 \
#       --data-dir data/teleop_wbc/simple/G1WholebodyBendPick-v1/level-0 \
#       --replay-dir /tmp/wbc_replay
#
# The controller lives inside third_party/GR00T-WholeBodyControl/gear_sonic_deploy
# (SIMPLE-S has no standalone gear_sonic_deploy submodule); it is built on demand.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Env id: positional arg wins, then the ENV_ID environment variable, then the default.
ENV_ID="${1:-${ENV_ID:-simple/G1WholebodyXMoveBendCarryBoxSonic-v0}}"
shift || true
SIM_LOCKSTEP=0
for arg in "$@"; do
  [[ "$arg" == "--sim-lockstep" ]] && SIM_LOCKSTEP=1
done

# Default arm pose for the Python robot models (sim-side FK); low | high | down.
export G1_ELBOW_POSE="${G1_ELBOW_POSE:-down}"

# Auto-engage the SONIC policy after its init ramp (no ']' keypress needed);
# requires a controller binary built with the SONIC_AUTO_START patch.
# Set SONIC_AUTO_START=0 to restore manual engagement.
export SONIC_AUTO_START="${SONIC_AUTO_START:-1}"

LOGS="${TMPDIR:-/tmp}/replay_wbc_logs"
mkdir -p "$LOGS"
cd "$ROOT_DIR"

# Supervisor state: a flag the trap raises so a deliberate Ctrl+C is never
# fought with a restart, plus a controller restart cap.
SHUTDOWN_FLAG="$LOGS/.shutting_down"
CONTROLLER_MAX_RESTARTS="${CONTROLLER_MAX_RESTARTS:-5}"
rm -f "$SHUTDOWN_FLAG"

# Stop only this launcher's background jobs and preserve the foreground
# command's exit status.  The previous `kill 0` also signalled this script,
# turning a successful replay into exit code 143.
cleanup() {
  local status="${1:-0}"
  trap - TERM INT EXIT
  set +e +u
  touch "$SHUTDOWN_FLAG" 2>/dev/null
  echo
  echo "Stopping controller..."

  local job_pids
  job_pids="$(jobs -pr)"
  if (( SIM_LOCKSTEP )); then
    [[ -n "${CTRL_PID:-}" ]] && kill -TERM "$CTRL_PID" 2>/dev/null
    sleep 1
    [[ -n "${CTRL_PID:-}" ]] && kill -KILL "$CTRL_PID" 2>/dev/null
  else
    pkill -TERM -f "g1_deploy_onnx_ref" 2>/dev/null
    [[ -n "$job_pids" ]] && kill -TERM $job_pids 2>/dev/null
    sleep 1
    pkill -KILL -f "g1_deploy_onnx_ref" 2>/dev/null
    [[ -n "$job_pids" ]] && kill -KILL $job_pids 2>/dev/null
  fi
  wait 2>/dev/null
  exit "$status"
}
trap 'cleanup "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Preflight: a stack left over from an earlier run keeps publishing on the same
# DDS topics and ZMQ port.  Two publishers on rt/lowcmd is not a warning — the
# sim follows whichever arrives last, so the robot behaves erratically and every
# diagnosis lies.  Also, a leftover pico manager holds the ZMQ port this replay
# needs to bind for token injection.  Clear all strays.
STRAYS="g1_deploy_onnx_ref|pico_manager_thread_server|cli/replay_wbc.py|cli/teleop_wbc.py"
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
  if (( SIM_LOCKSTEP )); then
    echo "Lockstep launch refuses to terminate processes it did not start:"
    ps -o pid=,args= -p "$(IFS=,; echo "${stray_pids[*]}")" 2>/dev/null | sed 's/^/    /'
    exit 1
  fi
  echo "Found processes from a previous run — clearing them first:"
  ps -o pid=,args= -p "$(IFS=,; echo "${stray_pids[*]}")" 2>/dev/null | sed 's/^/    /'
  kill -TERM "${stray_pids[@]}" 2>/dev/null || true
  sleep 2
  kill -KILL "${stray_pids[@]}" 2>/dev/null || true
  sleep 1
fi

echo "[1/2] SONIC controller  (log: $LOGS/controller.log)"
# Self-heal: if the controller binary is not built (fresh clone, or build/ wiped),
# build it now.  build_sonic_controller.sh is idempotent and needs no sudo/network
# once the deps are in ~/tools; it also writes ~/tools/sonic_env.sh.
CONTROLLER_BIN="$ROOT_DIR/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
if [[ ! -x "$CONTROLLER_BIN" ]]; then
  if (( SIM_LOCKSTEP )); then
    echo "  lockstep controller binary missing — build it explicitly before launch"
    exit 1
  else
    echo "  controller binary missing — building it (scripts/build_sonic_controller.sh)..."
    bash "$ROOT_DIR/scripts/build_sonic_controller.sh" || { echo "  controller build FAILED"; exit 1; }
  fi
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
    if (( SIM_LOCKSTEP )); then
      exec env SONIC_FORCE_UNITREE_DDS=1 SONIC_AUTO_START=1 \
        "$CONTROLLER_BIN" lo policy/release/model_decoder.onnx reference/example/ \
        --obs-config policy/release/observation_config.yaml \
        --encoder-file policy/release/model_encoder.onnx \
        --input-type zmq_manager --output-type zmq --zmq-host localhost \
        --default-motion neutral_kick_R_001__A543 \
        --disable-crc-check --sim-lockstep
    else
      # deploy.sh asks "Proceed? [Y/n]" — empty answer means yes.  zmq_manager so
      # the controller reads the tokens the replay CLI injects on the ZMQ pose topic.
      SONIC_SKIP_DEPLOY_BUILD=1 SONIC_FORCE_UNITREE_DDS=1 ./deploy.sh sim --input-type zmq_manager <<< ""
    fi
  ) >> "$LOGS/controller.log" 2>&1 &
  CTRL_PID=$!
}
: > "$LOGS/controller.log"   # truncate once; restarts append so crash reasons accumulate
start_controller
# Wait for actual readiness, not a fixed sleep: model loading + CUDA graph
# capture takes tens of seconds (deploy.sh may even rebuild TRT engines, which
# takes minutes on a cold cache).  Starting the sim before this marker means
# the whole settle phase elapses while the controller is still constructing —
# its command SUB conflates the engage handshake away and the policy engages
# mid-playback (robot falls).  The controller prints this once all control
# threads are up.
READY_MARKER="G1Deploy object created successfully"
CONTROLLER_WAIT="${CONTROLLER_WAIT:-600}"
waited=0
until grep -q "$READY_MARKER" "$LOGS/controller.log" 2>/dev/null; do
  kill -0 "$CTRL_PID" 2>/dev/null || {
    echo "  controller DIED during startup — last log lines:"; tail -15 "$LOGS/controller.log"; exit 1; }
  if (( waited >= CONTROLLER_WAIT )); then
    echo "  controller not ready after ${CONTROLLER_WAIT}s — last log lines:"; tail -15 "$LOGS/controller.log"; exit 1
  fi
  sleep 2; waited=$((waited + 2))
done
echo "  controller ready (${waited}s)"

# Supervisor: the controller exits on its own when its 500 ms rt/lowstate
# watchdog trips or a safety check fails — that would freeze the rest of the
# replay.  Bring it back (up to a cap), surfacing the reason.  The trap raises
# $SHUTDOWN_FLAG first, so a deliberate Ctrl+C is never fought with a restart.
supervise() {
  set +e
  local cr=0
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
      echo "[supervisor] controller back up (pid $CTRL_PID)."
    fi
    sleep 2
  done
}
if (( ! SIM_LOCKSTEP )); then
  supervise &
fi

# The viewer needs a live X display.  An inherited DISPLAY like ":0" can be
# stale (tmux/SSH shells outlive X server restarts) and GLFW then dies at
# startup.  Only local ":N" displays are checked against their socket; anything
# else (e.g. an SSH-forwarded "localhost:10.0") is left alone.
if [[ "${DISPLAY:-}" =~ ^:([0-9]+) ]] && [[ ! -S "/tmp/.X11-unix/X${BASH_REMATCH[1]}" ]]; then
  live="$(ls /tmp/.X11-unix 2>/dev/null | sed -n 's/^X\([0-9]*\)$/:\1/p' | head -1)"
  if [[ -n "$live" ]]; then
    echo "WARNING: DISPLAY=$DISPLAY has no X socket — using $live instead."
    export DISPLAY="$live"
  fi
fi

# HEADLESS=1 skips the desktop viewer — the replay still runs; pass --replay-dir
# to save an MP4 of each episode for a visual check afterwards.
VIEW_FLAG="--no-headless"
[[ "${HEADLESS:-0}" == "1" ]] && VIEW_FLAG="--headless"

echo "[2/2] replay: $ENV_ID $VIEW_FLAG  (Ctrl+C here stops everything)"
# -u / PYTHONUNBUFFERED: the per-episode settle + tracking-fidelity telemetry
# must appear live, not sit in a buffer that is lost when the run is killed.
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1 DISPLAY="${DISPLAY:-:1}" \
  .venv/bin/python -u src/simple/cli/replay_wbc.py "$ENV_ID" "$VIEW_FLAG" "$@"
