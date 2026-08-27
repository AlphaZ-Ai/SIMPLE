#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] phase 1/2: python environment"
"$ROOT_DIR/scripts/nix/bootstrap-python.sh"

echo "[bootstrap] phase 2/2: gpu extensions"
"$ROOT_DIR/scripts/nix/bootstrap-gpu.sh"

echo "[bootstrap] done"
