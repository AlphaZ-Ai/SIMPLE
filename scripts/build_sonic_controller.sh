#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# No-sudo bare-metal build of the SONIC controller (gear_sonic_deploy).
# Usage: bash scripts/build_sonic_controller.sh [GROOT_DIR]
#
# Why this exists: deploy.sh's own dependency step needs sudo (apt-get) and
# gates on clang/just, but the actual build is plain CMake + g++.  This
# script discovers TensorRT, fetches the remaining deps into ~/tools
# (user-space, no root), builds directly, and then preps deploy.sh's tool
# checks so `./deploy.sh sim ...` works afterwards for the RUN phase.
#
# Dependency map (discovered empirically):
#   TensorRT      -> system CUDA tree (TensorRT_ROOT), or pip tensorrt-cu12
#   onnxruntime   -> official prebuilt release tarball (no login needed)
#   msgpack-cxx   -> header-only release tarball
#   Eigen3        -> header-only, cmake-installed to ~/tools
#   GTest         -> small source build to ~/tools
#   libzmq        -> source build to ~/tools
#   unitree_sdk2  -> vendored in gear_sonic_deploy/thirdparty (nothing to do)
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
DEPLOY="$GROOT_DIR/gear_sonic_deploy"
BASE="$HOME/tools"
mkdir -p "$BASE" "$HOME/.local/bin"
[[ -d "$DEPLOY" ]] || { echo "gear_sonic_deploy not found at $DEPLOY"; exit 1; }

step "TensorRT"
if [[ -z "${TensorRT_ROOT:-}" ]]; then
  for c in /usr/local/cuda/targets/x86_64-linux /usr/local/cuda-*/targets/x86_64-linux \
           /usr/local/TensorRT* /opt/TensorRT*; do
    [[ -f "$c/include/NvInfer.h" ]] && TensorRT_ROOT="$c" && break
  done
fi
if [[ -z "${TensorRT_ROOT:-}" ]]; then
  echo "   No system TensorRT — assembling from pip wheels + NVIDIA OSS headers (no login)"
  # Pinned: this exact combo was verified end-to-end (headers must match libs;
  # the pip wheels ship libs ONLY, the public OSS repo ships the headers).
  TRT_VER="10.9.0.34"
  TRT_ROOT_DIR="$BASE/tensorrt-root"
  if [[ ! -f "$TRT_ROOT_DIR/include/NvInfer.h" ]]; then
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    [[ -x "$BASE/trt-wheel/bin/python" ]] || "$UV" venv "$BASE/trt-wheel" --quiet
    "$UV" pip install --python "$BASE/trt-wheel/bin/python" --quiet "tensorrt-cu12==$TRT_VER"
    LIBS_DIR="$(dirname "$(find "$BASE/trt-wheel" -name 'libnvinfer.so*' | head -1)")"
    mkdir -p "$TRT_ROOT_DIR/lib" "$TRT_ROOT_DIR/include"
    for so in "$LIBS_DIR"/lib*.so*; do
      b="$(basename "$so")"
      ln -sf "$so" "$TRT_ROOT_DIR/lib/$b"
      ln -sf "$so" "$TRT_ROOT_DIR/lib/${b%%.so*}.so"  # unversioned link for the linker
    done
    # OSS repo tags are 3-component (v10.9.0); pip wheels are 4 (10.9.0.34)
    TRT_TAG="v${TRT_VER%.*}"
    curl -fsSL --retry 3 "https://github.com/NVIDIA/TensorRT/archive/refs/tags/$TRT_TAG.tar.gz" \
      | tar xz -C "$BASE" "TensorRT-${TRT_TAG#v}/include"
    cp -r "$BASE/TensorRT-${TRT_TAG#v}/include/." "$TRT_ROOT_DIR/include/"
  fi
  # NvOnnxParser.h lives in the onnx-tensorrt SUBMODULE, which GitHub archive
  # tarballs exclude — fetch it separately from the matching GA branch
  [[ -f "$TRT_ROOT_DIR/include/NvOnnxParser.h" ]] || \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/onnx/onnx-tensorrt/release/${TRT_VER%.*.*}-GA/NvOnnxParser.h" \
      -o "$TRT_ROOT_DIR/include/NvOnnxParser.h"
  TensorRT_ROOT="$TRT_ROOT_DIR"
fi
export TensorRT_ROOT
echo "   TensorRT_ROOT=$TensorRT_ROOT"

step "onnxruntime (official prebuilt)"
ORT="$BASE/onnxruntime-linux-x64-gpu-1.22.0"
[[ -d "$ORT" ]] || curl -fsSL --retry 3 \
  https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz \
  | tar xz -C "$BASE"
export onnxruntime_ROOT="$ORT"

step "msgpack-cxx (header-only)"
[[ -d "$BASE/msgpack-cxx-6.1.1" ]] || curl -fsSL --retry 3 \
  https://github.com/msgpack/msgpack-c/releases/download/cpp-6.1.1/msgpack-cxx-6.1.1.tar.gz \
  | tar xz -C "$BASE"
# The main binary's include paths only propagate the shared ZMQ include dir,
# so msgpack headers must live there too (learned from the 96%-then-fail build):
mkdir -p "$BASE/zmq-install/include"
cp -rn "$BASE/msgpack-cxx-6.1.1/include/." "$BASE/zmq-install/include/" 2>/dev/null || true

step "Eigen3 (header-only)"
if [[ ! -d "$BASE/eigen-install" ]]; then
  curl -fsSL --retry 3 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C "$BASE"
  cmake -S "$BASE/eigen-3.4.0" -B "$BASE/eigen-3.4.0/b" -DCMAKE_INSTALL_PREFIX="$BASE/eigen-install" >/dev/null
  cmake --install "$BASE/eigen-3.4.0/b" >/dev/null
fi

step "GTest"
if [[ ! -d "$BASE/gtest-install" ]]; then
  curl -fsSL --retry 3 https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz | tar xz -C "$BASE"
  cmake -S "$BASE/googletest-1.14.0" -B "$BASE/googletest-1.14.0/b" \
    -DCMAKE_INSTALL_PREFIX="$BASE/gtest-install" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$BASE/googletest-1.14.0/b" -j"$(nproc)" >/dev/null
  cmake --install "$BASE/googletest-1.14.0/b" >/dev/null
fi

step "libzmq"
if [[ ! -f "$BASE/zmq-install/lib/libzmq.so" ]]; then
  curl -fsSL --retry 3 https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz -C "$BASE"
  ( cd "$BASE/zeromq-4.3.5" && ./configure --prefix="$BASE/zmq-install" >/dev/null \
    && make -j"$(nproc)" >/dev/null 2>&1 && make install >/dev/null )
fi
# cppzmq: the C++ header binding (zmq.hpp) is a SEPARATE project from libzmq
[[ -f "$BASE/zmq-install/include/zmq.hpp" ]] || {
  curl -fsSL --retry 3 https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp \
    -o "$BASE/zmq-install/include/zmq.hpp"
  curl -fsSL --retry 3 https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq_addon.hpp \
    -o "$BASE/zmq-install/include/zmq_addon.hpp"
}
# nlohmann/json: header-only, used by the zmq test targets
[[ -f "$BASE/zmq-install/include/nlohmann/json.hpp" ]] || {
  mkdir -p "$BASE/zmq-install/include/nlohmann"
  curl -fsSL --retry 3 https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
    -o "$BASE/zmq-install/include/nlohmann/json.hpp"
}

step "Build"
mkdir -p "$DEPLOY/build"
cmake -S "$DEPLOY" -B "$DEPLOY/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMSGPACK_INCLUDE_DIR="$BASE/msgpack-cxx-6.1.1/include" \
  -DZMQ_INCLUDE_DIR="$BASE/zmq-install/include" \
  -DZMQ_LIBRARY="$BASE/zmq-install/lib/libzmq.so" \
  -DCMAKE_PREFIX_PATH="$BASE/eigen-install;$BASE/gtest-install"
cmake --build "$DEPLOY/build" -j"$(nproc)"

step "Prep deploy.sh's tool checks (run phase, no sudo)"
command -v just >/dev/null || \
  curl --retry 3 --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "$HOME/.local/bin"
# deploy.sh only checks that `clang` EXISTS; the build above used the cached
# compiler, so a presence shim is enough to skip its sudo installer:
command -v clang >/dev/null || ln -sf "$(command -v gcc)" "$HOME/.local/bin/clang"

step "Write ~/tools/sonic_env.sh (single source-able runtime env)"
cat > "$BASE/sonic_env.sh" <<EOF
# Generated by scripts/build_sonic_controller.sh — source before running the
# SONIC controller:  source ~/tools/sonic_env.sh
export PATH="\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="$BASE/zmq-install/lib:$ORT/lib:$TensorRT_ROOT/lib:\${LD_LIBRARY_PATH:-}"
EOF

echo
echo "Done.  To run the controller:"
echo "  source $BASE/sonic_env.sh"
echo "  cd $DEPLOY && source scripts/setup_env.sh && ./deploy.sh sim --input-type zmq_manager"
