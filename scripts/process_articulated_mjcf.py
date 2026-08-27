#!/usr/bin/env python3
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Clean, scale, and validate articulated-object MJCF files.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


JOINT_REFERENCE_ATTRIBUTES = {
    "joint",
    "joint1",
    "joint2",
    "jointinparent",
}
BODY_REFERENCE_ATTRIBUTES = {
    "body",
    "body1",
    "body2",
    "target",
}


def _numbers(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _format_number(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.12g}"


def _scale_attribute(element: ET.Element, attribute: str, factor: float) -> None:
    value = element.get(attribute)
    if value is None:
        return
    element.set(
        attribute,
        " ".join(_format_number(number * factor) for number in _numbers(value)),
    )


def _prefix(name: str | None) -> str | None:
    if not name or name.startswith("articulate_"):
        return name
    return f"articulate_{name}"


def _remove_matching_children(
    parent: ET.Element,
    predicate,
) -> int:
    removed = 0
    for child in list(parent):
        if predicate(child):
            parent.remove(child)
            removed += 1
    return removed


def _resolve_texture_path(
    texture_file: str,
    input_xml: Path,
    asset_dir: Path,
) -> Path:
    candidates = [
        (input_xml.parent / texture_file).resolve(),
        (asset_dir / "output_mjcf" / texture_file).resolve(),
        (asset_dir / texture_file.lstrip("./")).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Texture {texture_file!r} was not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _convert_texture_to_png(
    texture: ET.Element,
    input_xml: Path,
    asset_dir: Path,
) -> bool:
    texture_file = texture.get("file")
    if not texture_file:
        return False

    suffix = Path(texture_file).suffix.lower()
    if suffix == ".png":
        return False
    if suffix not in {".jpg", ".jpeg"}:
        return False

    source = _resolve_texture_path(texture_file, input_xml, asset_dir)
    destination = source.with_suffix(".png")
    if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
        with Image.open(source) as image:
            image.save(destination, format="PNG")

    texture.set("file", str(Path(texture_file).with_suffix(".png")).replace("\\", "/"))
    return True


def _indent_and_write(root: ET.Element, output_path: Path) -> None:
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    with output_path.open("a", encoding="utf-8") as file:
        file.write("\n")


def clean_mjcf(input_path: Path, output_path: Path, asset_dir: Path) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"{input_path} is not a MuJoCo XML file")

    removed_floor = 0
    for parent in root.iter():
        removed_floor += _remove_matching_children(
            parent,
            lambda child: (
                child.tag == "default" and child.get("class") == "floor"
            )
            or (
                child.tag == "geom"
                and (
                    child.get("name") == "floor"
                    or child.get("class") == "floor"
                    or child.get("material") == "groundplane"
                )
            ),
        )

    removed_actuators = _remove_matching_children(
        root, lambda child: child.tag == "actuator"
    )

    body_renames: dict[str, str] = {}
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            old_name = body.get("name")
            new_name = _prefix(old_name)
            if old_name and new_name and old_name != new_name:
                body.set("name", new_name)
                body_renames[old_name] = new_name

    joint_renames: dict[str, str] = {}
    for tag in ("joint", "freejoint"):
        for joint in root.findall(f".//{tag}"):
            old_name = joint.get("name")
            new_name = _prefix(old_name)
            if old_name and new_name and old_name != new_name:
                joint.set("name", new_name)
                joint_renames[old_name] = new_name

    for element in root.iter():
        for attribute, value in list(element.attrib.items()):
            if attribute in JOINT_REFERENCE_ATTRIBUTES and value in joint_renames:
                element.set(attribute, joint_renames[value])
            elif attribute in BODY_REFERENCE_ATTRIBUTES and value in body_renames:
                element.set(attribute, body_renames[value])

    removed_ground_assets = 0
    converted_textures = 0
    removed_duplicate_textures = 0
    asset = root.find("asset")
    if asset is not None:
        removed_ground_assets += _remove_matching_children(
            asset,
            lambda child: (
                child.tag == "texture"
                and (
                    child.get("name") == "groundplane"
                    or child.get("type") == "skybox"
                )
            )
            or (
                child.tag == "material"
                and (
                    child.get("name") == "groundplane"
                    or child.get("texture") == "groundplane"
                )
            ),
        )

        seen_texture_names: set[str] = set()
        for texture in list(asset.findall("texture")):
            name = texture.get("name")
            if name and name in seen_texture_names:
                asset.remove(texture)
                removed_duplicate_textures += 1
                continue
            if name:
                seen_texture_names.add(name)
            converted_textures += int(
                _convert_texture_to_png(texture, input_path, asset_dir)
            )

    _indent_and_write(root, output_path)
    print(
        "MJCF cleaned:",
        f"floor={removed_floor},",
        f"actuator={removed_actuators},",
        f"ground_assets={removed_ground_assets},",
        f"joint_names={len(joint_renames)},",
        f"textures_deduplicated={removed_duplicate_textures},",
        f"textures_converted={converted_textures}",
    )
    print(f"Wrote: {output_path}")


def scale_mjcf(input_path: Path, output_path: Path, factor: float) -> None:
    if factor <= 0:
        raise ValueError(f"Scale must be positive, got {factor}")

    tree = ET.parse(input_path)
    root = tree.getroot()

    for element in root.iter():
        if "pos" in element.attrib:
            _scale_attribute(element, "pos", factor)
        if "fromto" in element.attrib:
            _scale_attribute(element, "fromto", factor)

        if element.tag in {"geom", "site"}:
            _scale_attribute(element, "size", factor)
            _scale_attribute(element, "margin", factor)
            _scale_attribute(element, "gap", factor)

        if element.tag == "mesh":
            if "scale" not in element.attrib:
                element.set(
                    "scale",
                    " ".join([_format_number(factor)] * 3),
                )
            else:
                _scale_attribute(element, "scale", factor)

        if element.tag == "joint" and element.get("type", "hinge") == "slide":
            for attribute in ("range", "ref", "springref", "margin"):
                _scale_attribute(element, attribute, factor)

        if element.tag == "inertial":
            _scale_attribute(element, "mass", factor**3)
            _scale_attribute(element, "diaginertia", factor**5)
            _scale_attribute(element, "fullinertia", factor**5)

        if element.tag == "geom" and "mass" in element.attrib:
            _scale_attribute(element, "mass", factor**3)

        if element.tag == "statistic":
            _scale_attribute(element, "center", factor)
            _scale_attribute(element, "extent", factor)

    _indent_and_write(root, output_path)
    print(f"MJCF scaled by {factor:g}")
    print(f"Wrote: {output_path}")


def validate_mjcf(input_path: Path, compile_model: bool) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()

    errors: list[str] = []
    if root.find(".//actuator") is not None:
        errors.append("actuator section still exists")
    if root.find(".//default[@class='floor']") is not None:
        errors.append("floor default still exists")
    if root.find(".//geom[@name='floor']") is not None:
        errors.append("floor geom still exists")
    if root.find(".//texture[@name='groundplane']") is not None:
        errors.append("groundplane texture still exists")

    texture_names: list[str] = [
        texture.get("name", "") for texture in root.findall(".//asset/texture")
    ]
    if len(texture_names) != len(set(texture_names)):
        errors.append("duplicate texture names exist")

    for joint in root.findall(".//joint") + root.findall(".//freejoint"):
        name = joint.get("name")
        if name and not name.startswith("articulate_"):
            errors.append(f"joint {name!r} lacks articulate_ prefix")

    if errors:
        raise ValueError("; ".join(errors))

    if compile_model:
        import mujoco

        model_spec = mujoco.MjSpec.from_file(str(input_path))
        model_spec.compile()

    print(f"MJCF validation passed: {input_path}")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Clean generated MJCF")
    clean_parser.add_argument("--input", type=_path, required=True)
    clean_parser.add_argument("--output", type=_path, required=True)
    clean_parser.add_argument("--asset-dir", type=_path, required=True)

    scale_parser = subparsers.add_parser("scale", help="Scale cleaned MJCF")
    scale_parser.add_argument("--input", type=_path, required=True)
    scale_parser.add_argument("--output", type=_path, required=True)
    scale_parser.add_argument("--scale", type=float, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate MJCF")
    validate_parser.add_argument("--input", type=_path, required=True)
    validate_parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile the model with MuJoCo",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "clean":
        clean_mjcf(args.input, args.output, args.asset_dir)
    elif args.command == "scale":
        scale_mjcf(args.input, args.output, args.scale)
    elif args.command == "validate":
        validate_mjcf(args.input, args.compile)


if __name__ == "__main__":
    main()
