#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# ONE-COMMAND fresh-machine setup for the SONIC WBC teleop path (teleop/wbc
# branch).  Mirrors the main README's flow — after this, it's just:
#   source .venv/bin/activate
#
# PLATFORM: Linux x86_64 with an NVIDIA GPU + CUDA (the SONIC controller and
# curobo are built from source against TensorRT/CUDA; macOS/ARM is unsupported).
#
# Usage, from the SIMPLE repo root:
#   bash scripts/setup_teleop_wbc.sh [GROOT_DIR]
#
# Everything here is sudo-free and idempotent (re-run skips finished steps).
# The ONLY sudo step left, and only on the machine physically connected to
# the headset:
#   sudo dpkg -i third_party/decoupled_wbc/control/teleop/device/pico/XRoboToolkit_PC_Service_*.deb
#
# Env knobs:
#   SKIP_CUROBO=1      skip the curobo build (teleop CLI won't start without
#                      curobo — only skip if you'll install it another way)
#   SKIP_CONTROLLER=1  skip the SONIC C++ controller build (e.g. on a machine
#                      that only runs the sim, not the controller)
#
# Known gotchas this script encodes (learned the hard way):
#   - openpi-client submodule must be initialized or uv sync dies resolving it
#   - xrobotoolkit-sdk C++ build needs pybind11 on CMAKE_PREFIX_PATH,
#     so it is skipped during sync and built explicitly afterwards
#   - SONIC ONNX checkpoints come from HuggingFace, NOT git-lfs
#   - GR00T-WholeBodyControl is OUR FORK (junsooki/...) pinned as a submodule,
#     so upstream NVlabs changes can never break this branch
set -euo pipefail

# --- failure reporting: name the step, say how to recover -------------------
CURRENT_STEP="startup"
step() { CURRENT_STEP="$1"; echo "== $1 =="; }
trap 'echo; echo "FAILED during step: $CURRENT_STEP"; \
echo "  - This script is idempotent: re-running skips completed steps."; \
echo "  - curl error above => that server refused/rate-limited or a proxy"; \
echo "    interfered; usually transient, just re-run."; \
echo "  - Persistent failure: paste the step name + error to whoever debugs."' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_DIR="${1:-$ROOT_DIR/third_party/GR00T-WholeBodyControl}"
cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$PATH"

step "[1/7] Submodules (needs SSH access to the songlin/* forks)"
# sync first: submodule URLs change when we repoint to our forks, and
# existing checkouts keep the old URL until synced
git submodule sync >/dev/null
git submodule update --init --depth 1 \
  third_party/gear_sonic \
  third_party/decoupled_wbc \
  third_party/unitree_sdk2_python \
  third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  third_party/openpi-client

step "[2/7] uv"
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  pip3 install --user uv
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

step "[3/7] Python environment (skipping xrobotoolkit C++ build for now)"
# --inexact: don't uninstall packages sync doesn't know about.  curobo is
# installed by step 6 OUTSIDE the lockfile (upstream keeps it commented out
# of pyproject), and a rerun of this script must not evict it.
# --extra lerobot: required by --record on both teleop paths.
"$UV" sync --inexact --group sonic --extra lerobot --no-install-package xrobotoolkit-sdk

step "[4/7] xrobotoolkit-sdk (same submodule source, built with pybind11 hint)"
# pin setuptools-scm <9: 9.x moved its core into a separate 'vcs_versioning'
# package, and its global setuptools entry point then fails to import during
# this --no-build-isolation build (ModuleNotFoundError: vcs_versioning).
"$UV" pip install pybind11 "setuptools-scm<9"
CMAKE_PREFIX_PATH="$(.venv/bin/python -m pybind11 --cmakedir)" \
  "$UV" pip install --no-build-isolation -e third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
.venv/bin/python -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk OK')"

step "[5/7] GR00T-WholeBodyControl (our fork) + SONIC checkpoints -> $GROOT_DIR"
if [[ "$GROOT_DIR" == "$ROOT_DIR/third_party/GR00T-WholeBodyControl" ]]; then
  # canonical location: submodule pinned to the junsooki fork
  GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 third_party/GR00T-WholeBodyControl
elif [[ ! -d "$GROOT_DIR" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://github.com/junsooki/GR00T-WholeBodyControl.git "$GROOT_DIR"
fi
# git-lfs without sudo, if the machine lacks it
if ! command -v git-lfs >/dev/null; then
  echo "   installing git-lfs to ~/.local/bin (no sudo)"
  curl -fsSL --retry 3 https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz | tar xz -C /tmp
  mkdir -p "$HOME/.local/bin" && cp /tmp/git-lfs-3.5.1/git-lfs "$HOME/.local/bin/"
fi
git -C "$GROOT_DIR" lfs install --local
# only the deploy stack's LFS content (meshes + reference motion); skips ~2GB
# of training data/media that teleop never touches
git -C "$GROOT_DIR" lfs pull --include="gear_sonic_deploy/**"
(cd "$GROOT_DIR" && "$ROOT_DIR/.venv/bin/python" download_from_hf.py)

step "[6/7] curobo (simple.tasks imports it eagerly; CUDA build, slow first time)"
if [[ "${SKIP_CUROBO:-0}" == "1" ]]; then
  echo "   SKIP_CUROBO=1 — skipping.  The teleop CLI will refuse to start"
  echo "   until curobo is importable (scripts/install_curobo.sh)."
elif .venv/bin/python -c "from curobo.types.base import TensorDeviceType" 2>/dev/null; then
  echo "   curobo already importable — skipping"
else
  git submodule update --init --depth 1 third_party/curobo
  # torch compiles curobo's CUDA kernels with nvcc and refuses a major-version
  # mismatch.  Ubuntu's /usr/bin/nvcc (11.x) often shadows the real toolkit in
  # /usr/local, so pick the toolkit that matches torch's CUDA version.
  TORCH_CUDA="$(.venv/bin/python -c 'import torch; print(torch.version.cuda or "")')"
  if [[ -n "$TORCH_CUDA" ]]; then
    for c in "/usr/local/cuda-$TORCH_CUDA" "/usr/local/cuda-${TORCH_CUDA%%.*}" \
             /usr/local/cuda-"${TORCH_CUDA%%.*}".* /usr/local/cuda; do
      if [[ -x "$c/bin/nvcc" ]] && "$c/bin/nvcc" --version | grep -q "release ${TORCH_CUDA%%.*}\."; then
        export CUDA_HOME="$c" PATH="$c/bin:$PATH"
        echo "   using nvcc from $c (torch built with CUDA $TORCH_CUDA)"
        break
      fi
    done
  fi
  # curobo versions itself with setuptools-scm from git tags; the fork has no
  # tags (and shallow submodules wouldn't see them anyway), so the wheel
  # filename and its metadata disagree ("0.0.0" vs "0.0.post1.dev1") and uv
  # refuses the install.  Pin a consistent version instead.
  export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.4}"
  rm -rf third_party/curobo/src/*.egg-info
  # UV_NO_CACHE: uv's cached metadata for this path package can survive
  # 'uv cache clean' and then fail the metadata-vs-wheel version check on
  # every retry.  Bypassing the cache guarantees metadata and wheel come
  # from the same build.
  UV_NO_CACHE=1 bash "$ROOT_DIR/scripts/install_curobo.sh"
fi

step "[7/7] SONIC controller (bare-metal C++ build, no sudo)"
if [[ "${SKIP_CONTROLLER:-0}" == "1" ]]; then
  echo "   SKIP_CONTROLLER=1 — skipping controller build"
else
  bash "$ROOT_DIR/scripts/build_sonic_controller.sh" "$GROOT_DIR"
fi

echo
echo "Setup complete.  Activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "Launch the full stack (controller + manager + sim) in one command:"
echo "  bash scripts/run_teleop_wbc.sh simple/G1WholebodyXMoveBendCarryBoxSonic-v0"
echo "  (add --record to capture a LeRobot dataset)"
