#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSETS_ROOT="${REPO_ROOT}/data/assets/articulated"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${PYTHON}}"
URDF_TO_MJCF="${URDF_TO_MJCF:-urdf-to-mjcf}"
SKIP_USD=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/process_articulated_asset.sh ASSET_ID SCALE [options]

Examples:
  bash scripts/process_articulated_asset.sh 006 0.5
  bash scripts/process_articulated_asset.sh 6 0.25 --skip-usd

Options:
  --assets-root PATH  Articulated asset root directory.
  --skip-usd          Skip the Isaac Sim URDF-to-USD step.
  -h, --help          Show this help.

Environment overrides:
  PYTHON, ISAAC_PYTHON, URDF_TO_MJCF
EOF
}

POSITIONAL=()
while (($#)); do
    case "$1" in
        --assets-root)
            ASSETS_ROOT="$2"
            shift 2
            ;;
        --skip-usd)
            SKIP_USD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if ((${#POSITIONAL[@]} > 2)); then
    echo "Expected ASSET_ID and SCALE only." >&2
    usage >&2
    exit 2
fi

ASSET_INPUT="${POSITIONAL[0]:-}"
SCALE="${POSITIONAL[1]:-}"

if [[ -z "${ASSET_INPUT}" ]]; then
    read -r -p "Asset ID (for example 006): " ASSET_INPUT
fi
if [[ -z "${SCALE}" ]]; then
    read -r -p "Scale (for example 0.5): " SCALE
fi

if [[ ! "${ASSET_INPUT}" =~ ^[0-9]+$ ]]; then
    echo "Asset ID must contain digits only: ${ASSET_INPUT}" >&2
    exit 2
fi
ASSET_ID="$(printf '%03d' "$((10#${ASSET_INPUT}))")"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Isaac Python executable not found: ${ISAAC_PYTHON}" >&2
    exit 1
fi
if ! command -v "${URDF_TO_MJCF}" >/dev/null 2>&1; then
    echo "urdf-to-mjcf executable not found: ${URDF_TO_MJCF}" >&2
    exit 1
fi

"${PYTHON}" -c \
    'import sys; value=float(sys.argv[1]); assert value > 0' \
    "${SCALE}" \
    || {
        echo "Scale must be a positive number: ${SCALE}" >&2
        exit 2
    }

if [[ ! -d "${ASSETS_ROOT}" ]]; then
    echo "Assets root not found: ${ASSETS_ROOT}" >&2
    exit 1
fi
ASSET_DIR="$(cd -- "${ASSETS_ROOT}" && pwd)/${ASSET_ID}"
URDF_PATH="${ASSET_DIR}/mobility.urdf"
OUTPUT_MJCF_DIR="${ASSET_DIR}/output_mjcf"
OUTPUT_USD_DIR="${ASSET_DIR}/output_usd"
SCALE_TAG="${SCALE//\//_}"
SCALED_URDF="${ASSET_DIR}/${ASSET_ID}_scaled_${SCALE_TAG}.urdf"

if [[ ! -d "${ASSET_DIR}" ]]; then
    echo "Asset directory not found: ${ASSET_DIR}" >&2
    exit 1
fi
if [[ ! -f "${URDF_PATH}" ]]; then
    echo "URDF not found: ${URDF_PATH}" >&2
    exit 1
fi

STAGING_DIR="$(mktemp -d "${ASSET_DIR}/.asset_pipeline.XXXXXX")"
cleanup() {
    rm -rf -- "${STAGING_DIR}"
}
trap cleanup EXIT

CONVERT_DIR="${STAGING_DIR}/converted_mjcf"
FINAL_MJCF_DIR="${STAGING_DIR}/final_mjcf"
STAGED_USD_DIR="${STAGING_DIR}/output_usd"
mkdir -p "${CONVERT_DIR}" "${FINAL_MJCF_DIR}"

RAW_MJCF="${CONVERT_DIR}/${ASSET_ID}.xml"
UNSCALED_MJCF="${FINAL_MJCF_DIR}/${ASSET_ID}_unscaled.xml"
FINAL_MJCF="${FINAL_MJCF_DIR}/${ASSET_ID}.xml"

echo "[1/5] URDF -> raw MJCF"
"${URDF_TO_MJCF}" "${URDF_PATH}" --output "${RAW_MJCF}"

echo "[2/5] Clean MJCF"
"${PYTHON}" "${SCRIPT_DIR}/process_articulated_mjcf.py" clean \
    --input "${RAW_MJCF}" \
    --output "${UNSCALED_MJCF}" \
    --asset-dir "${ASSET_DIR}"

echo "[3/5] Scale MJCF by ${SCALE}"
"${PYTHON}" "${SCRIPT_DIR}/process_articulated_mjcf.py" scale \
    --input "${UNSCALED_MJCF}" \
    --output "${FINAL_MJCF}" \
    --scale "${SCALE}"

mkdir -p "${OUTPUT_MJCF_DIR}"
for generated_path in "${CONVERT_DIR}"/*; do
    if [[ -e "${generated_path}" && "${generated_path}" != "${RAW_MJCF}" ]]; then
        cp -a -- "${generated_path}" "${OUTPUT_MJCF_DIR}/"
    fi
done
cp -a -- "${UNSCALED_MJCF}" "${OUTPUT_MJCF_DIR}/${ASSET_ID}_unscaled.xml"
cp -a -- "${FINAL_MJCF}" "${OUTPUT_MJCF_DIR}/${ASSET_ID}.xml"

"${PYTHON}" "${SCRIPT_DIR}/process_articulated_mjcf.py" validate \
    --input "${OUTPUT_MJCF_DIR}/${ASSET_ID}.xml" \
    --compile

echo "[4/5] Sanitize and scale URDF"
"${PYTHON}" "${SCRIPT_DIR}/change_urdf_and_mesh_dir.py" \
    --input "${URDF_PATH}" \
    --output "${SCALED_URDF}" \
    --mesh-dir "${ASSET_DIR}" \
    --scale "${SCALE}"

if ((SKIP_USD)); then
    echo "[5/5] USD export skipped"
else
    echo "[5/5] URDF -> USD"
    mkdir -p "${STAGED_USD_DIR}"
    if ! "${ISAAC_PYTHON}" "${SCRIPT_DIR}/export_urdf_to_usd.py" \
        --urdf "${SCALED_URDF}" \
        --output "${STAGED_USD_DIR}/${ASSET_ID}.usd" \
        --headless; then
        echo "USD export failed. MJCF and scaled URDF were generated successfully." >&2
        echo "Check Isaac Sim, GPU/Vulkan, and inotify resource availability." >&2
        exit 1
    fi

    mkdir -p "${OUTPUT_USD_DIR}"
    cp -a -- "${STAGED_USD_DIR}/." "${OUTPUT_USD_DIR}/"
fi

echo "Asset processing complete:"
echo "  asset:       ${ASSET_ID}"
echo "  scale:       ${SCALE}"
echo "  MJCF:        ${OUTPUT_MJCF_DIR}/${ASSET_ID}.xml"
echo "  MJCF source: ${OUTPUT_MJCF_DIR}/${ASSET_ID}_unscaled.xml"
echo "  URDF:        ${SCALED_URDF}"
if ((!SKIP_USD)); then
    echo "  USD:         ${OUTPUT_USD_DIR}/${ASSET_ID}.usd"
fi
