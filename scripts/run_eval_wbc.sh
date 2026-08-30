#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Launch one lockstep SONIC controller and a foreground whole-body evaluator.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ID=${1:?usage: run_eval_wbc.sh ENV_ID POLICY [eval options]}
POLICY=${2:?usage: run_eval_wbc.sh ENV_ID POLICY [eval options]}
shift 2

DEPLOY="$ROOT_DIR/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
CONTROLLER_BIN="$DEPLOY/target/release/g1_deploy_onnx_ref"
LOGS="${TMPDIR:-/tmp}/eval_wbc_logs"
mkdir -p "$LOGS"
CONTROLLER_LOG="$LOGS/controller.log"
CTRL_PID=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$CTRL_PID" ]] && kill -0 "$CTRL_PID" 2>/dev/null; then
    kill -TERM -- "-$CTRL_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$CTRL_PID" 2>/dev/null || break
      sleep 0.05
    done
    if kill -0 "$CTRL_PID" 2>/dev/null; then
      kill -KILL -- "-$CTRL_PID" 2>/dev/null || true
    fi
    wait "$CTRL_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

[[ -x "$CONTROLLER_BIN" ]] || {
  echo "lockstep controller binary missing: $CONTROLLER_BIN" >&2
  exit 1
}
[[ -f "$HOME/tools/sonic_env.sh" ]] || {
  echo "runtime environment missing: $HOME/tools/sonic_env.sh" >&2
  exit 1
}

for port in 5556 5557 13579; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "required port $port is occupied; refusing to stop an unknown process" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export HEADLESS=${HEADLESS:-1}
export PYTHONUNBUFFERED=1
export SONIC_AUTO_START=1
export G1_ELBOW_POSE=${G1_ELBOW_POSE:-down}

if [[ -z "${CUDAToolkit_ROOT:-}" && -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  export CUDAToolkit_ROOT="$CUDA_HOME"
fi

: > "$CONTROLLER_LOG"
setsid env DEPLOY="$DEPLOY" CONTROLLER_BIN="$CONTROLLER_BIN" bash -c '
  set +e +u +o pipefail
  source "$HOME/tools/sonic_env.sh"
  cd "$DEPLOY"
  source scripts/setup_env.sh
  exec env SONIC_FORCE_UNITREE_DDS=1 SONIC_AUTO_START=1 \
    "$CONTROLLER_BIN" lo policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --input-type zmq_manager --output-type zmq --zmq-host localhost \
    --default-motion neutral_kick_R_001__A543 \
    --disable-crc-check --sim-lockstep
' >> "$CONTROLLER_LOG" 2>&1 &
CTRL_PID=$!

waited=0
limit=${CONTROLLER_WAIT:-600}
until grep -q "G1Deploy object created successfully" "$CONTROLLER_LOG" 2>/dev/null; do
  if ! kill -0 "$CTRL_PID" 2>/dev/null; then
    echo "controller exited during startup" >&2
    tail -n 30 "$CONTROLLER_LOG" >&2
    exit 1
  fi
  if (( waited >= limit )); then
    echo "controller did not become ready within ${limit}s" >&2
    tail -n 30 "$CONTROLLER_LOG" >&2
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
done
echo "controller ready (${waited}s); log: $CONTROLLER_LOG"

cd "$ROOT_DIR"

# The viewer needs a live X display.  An inherited DISPLAY like ":0" can be
# stale (tmux/SSH shells outlive X server restarts) and the viewer then dies at
# startup.  Only local ":N" displays are checked against their socket; anything
# else (e.g. an SSH-forwarded "localhost:10.0") is left alone.
if [[ "$HEADLESS" != "1" ]]; then
  if [[ "${DISPLAY:-}" =~ ^:([0-9]+) ]] && [[ ! -S "/tmp/.X11-unix/X${BASH_REMATCH[1]}" ]]; then
    live="$(ls /tmp/.X11-unix 2>/dev/null | sed -n 's/^X\([0-9]*\)$/:\1/p' | head -1)"
    if [[ -n "$live" ]]; then
      echo "WARNING: DISPLAY=$DISPLAY has no X socket - using $live instead."
      export DISPLAY="$live"
    fi
  fi
  export DISPLAY="${DISPLAY:-:1}"
fi

# HEADLESS=1 skips the desktop viewer; the evaluation still runs and still
# writes its MP4s.  The eval CLI defaults to --headless, so the flag is always
# passed explicitly.
VIEW_FLAG="--no-headless"
[[ "$HEADLESS" == "1" ]] && VIEW_FLAG="--headless"

echo "eval: $ENV_ID $POLICY $VIEW_FLAG  (Ctrl+C here stops everything)"
if [[ -n "${EVAL_WBC_WRAPPER:-}" ]]; then
  .venv/bin/python -u "$EVAL_WBC_WRAPPER" "$ENV_ID" "$POLICY" "$VIEW_FLAG" "$@"
else
  .venv/bin/python -u -m simple.cli.eval_wbc "$ENV_ID" "$POLICY" "$VIEW_FLAG" "$@"
fi
