#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

cd data
echo "📥 Downloading data from Hugging Face..."
python _download.py

unzip assets.zip
rm -rf assets.zip

unzip robots.zip
rm -rf robots.zip

unzip scenes.zip
rm -rf scenes.zip

unzip vMaterials_2.zip
rm -rf vMaterials_2.zip

cd ..
echo "✅ All data downloaded successfully."
