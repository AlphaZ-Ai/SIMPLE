#!/usr/bin/env python3
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Export an articulated-object URDF to the USD layout expected by SIMPLE.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.urdf.is_file():
        raise FileNotFoundError(args.urdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # SimulationApp must start before importing omni or pxr modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        import omni.kit.app
        import omni.kit.commands
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf",
            True,
        )

        print(f"Importing URDF: {args.urdf}")
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.set_fix_base(True)
        import_config.set_import_inertia_tensor(True)
        import_config.set_convex_decomp(False)
        import_config.set_self_collision(False)
        import_config.set_default_drive_type(1)
        import_config.set_default_drive_strength(1e4)
        import_config.set_default_position_drive_damping(1e3)
        import_config.set_density(30.0)
        import_config.set_distance_scale(1.0)

        success, _ = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(args.urdf),
            import_config=import_config,
            dest_path=str(args.output),
        )
        if not success:
            raise RuntimeError(
                "URDF to USD conversion failed; check the URDF and mesh paths"
            )

        stage = Usd.Stage.Open(str(args.output))
        if stage is None:
            raise RuntimeError(f"Could not open generated USD: {args.output}")

        old_root_prim = next(
            (
                child
                for child in stage.GetPseudoRoot().GetChildren()
                if child.GetName()
                not in {"Looks", "Render", "OmniverseKit_Persp"}
            ),
            None,
        )
        if old_root_prim is None:
            raise RuntimeError("Generated USD does not contain a model root")

        old_path = old_root_prim.GetPath()
        robot_name = old_root_prim.GetName()
        UsdGeom.Xform.Define(stage, "/root")

        new_path = Sdf.Path(f"/root/{robot_name}")
        root_edit = Sdf.BatchNamespaceEdit()
        root_edit.Add(old_path, new_path)
        if not stage.GetRootLayer().Apply(root_edit):
            raise RuntimeError(f"Could not move {old_path} to {new_path}")

        joint_edit = Sdf.BatchNamespaceEdit()
        renamed_count = 0
        for prim in list(stage.Traverse()):
            prim_name = prim.GetName()
            if prim_name.startswith("joint_"):
                joint_edit.Add(
                    prim.GetPath(),
                    prim.GetPath().ReplaceName(
                        prim_name.replace(
                            "joint_",
                            "articulate_joint_",
                            1,
                        )
                    ),
                )
                renamed_count += 1

            if prim.IsA(UsdGeom.Gprim):
                collision_attribute = prim.CreateAttribute(
                    "physics:collisionEnabled",
                    Sdf.ValueTypeNames.Bool,
                )
                collision_attribute.Set(False)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(
                        False
                    )

        if renamed_count and not stage.GetRootLayer().Apply(joint_edit):
            raise RuntimeError("Could not rename USD joint prims")

        stage.SetDefaultPrim(stage.GetPrimAtPath("/root"))
        stage.GetRootLayer().Save()
        print(f"Renamed {renamed_count} USD joint prim(s)")
        print(f"Wrote USD: {args.output}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
