#!/usr/bin/env python3
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Sanitize mesh names and uniformly scale an articulated-object URDF.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _format_number(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.12g}"


def _scale_values(value: str, factor: float) -> str:
    return " ".join(
        _format_number(float(item) * factor) for item in value.split()
    )


def _safe_name(name: str) -> str:
    new_name = name.replace("-", "_")
    new_name = re.sub(r"[^a-zA-Z0-9_]", "_", new_name)
    if new_name and new_name[0].isdigit():
        new_name = f"m_{new_name}"
    return new_name


def process_and_scale_urdf(
    input_urdf_path: str | Path,
    output_urdf_path: str | Path,
    mesh_dir: str | Path,
    scale_factor: float = 0.5,
) -> None:
    input_path = Path(input_urdf_path).expanduser().resolve()
    output_path = Path(output_urdf_path).expanduser().resolve()
    mesh_root = Path(mesh_dir).expanduser().resolve()

    if scale_factor <= 0:
        raise ValueError(f"Scale must be positive, got {scale_factor}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    print(f"Processing URDF: {input_path}")
    print(f"Scale: {scale_factor:g}")

    content = input_path.read_text(encoding="utf-8")
    filenames = re.findall(r'filename="([^"]+)"', content)
    rename_map: dict[str, str] = {}

    for filename in filenames:
        path = Path(filename)
        safe_basename = re.sub(r"[^a-zA-Z0-9_.]", "_", path.name)
        safe_filename = str(path.with_name(safe_basename)).replace("\\", "/")
        if safe_filename != filename:
            rename_map[filename] = safe_filename

    for old_relative, new_relative in rename_map.items():
        old_path = mesh_root / old_relative
        new_path = mesh_root / new_relative
        if not old_path.exists():
            raise FileNotFoundError(f"Referenced mesh does not exist: {old_path}")
        if not new_path.exists() or new_path.stat().st_mtime < old_path.stat().st_mtime:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            print(f"Copied mesh: {old_relative} -> {new_relative}")

    def replace_filename(match: re.Match[str]) -> str:
        filename = match.group(1)
        return f'filename="{rename_map.get(filename, filename)}"'

    content = re.sub(r'filename="([^"]+)"', replace_filename, content)

    def replace_name(match: re.Match[str]) -> str:
        return f'name="{_safe_name(match.group(1))}"'

    content = re.sub(r'\bname="([^"]+)"', replace_name, content)
    root = ET.fromstring(content)

    for mesh in root.findall(".//mesh"):
        existing_scale = mesh.get("scale", "1 1 1")
        mesh.set("scale", _scale_values(existing_scale, scale_factor))

    for origin in root.findall(".//origin"):
        if "xyz" in origin.attrib:
            origin.set("xyz", _scale_values(origin.attrib["xyz"], scale_factor))

    for joint in root.findall(".//joint"):
        if joint.get("type") != "prismatic":
            continue
        limit = joint.find("limit")
        if limit is not None:
            for attribute in ("lower", "upper"):
                if attribute in limit.attrib:
                    limit.set(
                        attribute,
                        _scale_values(limit.attrib[attribute], scale_factor),
                    )

    for inertial in root.findall(".//inertial"):
        mass = inertial.find("mass")
        if mass is not None and "value" in mass.attrib:
            mass.set(
                "value",
                _scale_values(mass.attrib["value"], scale_factor**3),
            )
        inertia = inertial.find("inertia")
        if inertia is not None:
            for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if attribute in inertia.attrib:
                    inertia.set(
                        attribute,
                        _scale_values(
                            inertia.attrib[attribute],
                            scale_factor**5,
                        ),
                    )

    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    with output_path.open("a", encoding="utf-8") as file:
        file.write("\n")
    print(f"Wrote scaled URDF: {output_path}")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=_path, required=True, help="Input URDF")
    parser.add_argument("--output", type=_path, required=True, help="Output URDF")
    parser.add_argument(
        "--mesh-dir",
        type=_path,
        help="Root directory used by relative mesh paths; defaults to input parent",
    )
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    process_and_scale_urdf(
        input_urdf_path=args.input,
        output_urdf_path=args.output,
        mesh_dir=args.mesh_dir or args.input.parent,
        scale_factor=args.scale,
    )


if __name__ == "__main__":
    main()
