# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Shared Isaac SimulationApp creation helpers.
"""

from __future__ import annotations

import os
from simple.utils import env_flag

HYDRA_WAIT_IDLE = "/app/hydraEngine/waitIdle"
HYDRA_RENDER_COMPLETE = "/app/updateOrder/checkForHydraRenderComplete"
THROTTLING_ENABLE_ASYNC = "/exts/isaacsim.core.throttling/enable_async"
LOG_OUTPUT_STREAM_LEVEL = "/log/outputStreamLevel"

# isaacsim.robot_motion.lula ships a prebuilt liblula_kinematics.so linked against the
# pre-C++11 ABI of urdfdom, while pinocchio (pulled in by curobo, which `simple.envs`
# imports long before Kit starts) loads a __cxx11-ABI liburdfdom_model.so.3.0 under the
# same SONAME. The first one loaded wins, so lula can never resolve its urdf symbols in
# this process and every extension depending on it fails to start. SIMPLE does its
# kinematics with curobo and never imports lula, so keep that whole subtree out of the
# dependency solve instead of letting it fail loudly. Set SIMPLE_ISAAC_ENABLE_LULA=1 to
# put them back (and get the load errors with them).
LULA_DEPENDENT_EXTENSIONS = (
    "isaacsim.robot_motion.lula",
    "isaacsim.robot_motion.lula_test_widget",
    "isaacsim.robot_motion.motion_generation",
    "isaacsim.robot_setup.xrdf_editor",
    "isaacsim.robot.manipulators.examples",
    "isaacsim.cortex.framework",
    "isaacsim.cortex.behaviors",
    "isaacsim.examples.interactive",
    "omni.isaac.lula",
    "omni.isaac.lula_test_widget",
    "omni.isaac.motion_generation",
    "omni.isaac.robot_description_editor",
    "omni.isaac.cortex",
    "omni.isaac.cortex.sample_behaviors",
    "omni.isaac.examples",
    "omni.isaac.franka",
    "omni.isaac.universal_robots",
)


def _compact_dict(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def create_simulation_app(
    SimulationApp,
    *,
    headless: bool,
    renderer: str = "RayTracedLighting",
    width: int | None = None,
    height: int | None = None,
    anti_aliasing: int | None = None,
    hide_ui: bool | None = None,
    multi_gpu: bool = False,
    webrtc: bool = False,
):
    experience = os.getenv("SIMPLE_ISAAC_EXPERIENCE", "").strip()
    zero_delay = env_flag("SIMPLE_ISAAC_ZERO_DELAY", default=True)
    disable_throttling_async = env_flag("SIMPLE_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)
    # Kit prints hundreds of harmless warnings per startup (omni.isaac.* deprecation
    # shims, unresolved USD asset references). They are still written to the kit log
    # file; only the console stream is quieted. Set SIMPLE_ISAAC_LOG_LEVEL=warning to
    # get them back on stdout.
    log_level = os.getenv("SIMPLE_ISAAC_LOG_LEVEL", "error").strip()
    enable_lula = env_flag("SIMPLE_ISAAC_ENABLE_LULA", default=False)

    settings: list[tuple[str, object, object]] = []
    if log_level:
        settings.append((LOG_OUTPUT_STREAM_LEVEL, log_level, log_level))
    if zero_delay:
        settings.append((HYDRA_WAIT_IDLE, 1, True))
        settings.append((HYDRA_RENDER_COMPLETE, 1000, 1000))
    if disable_throttling_async:
        settings.append((THROTTLING_ENABLE_ASYNC, "false", False))
    extra_args = [f"--{key}={arg_value}" for key, arg_value, _ in settings]
    if not enable_lula:
        extra_args += [
            f"--/app/extensions/excluded/{index}={name}"
            for index, name in enumerate(LULA_DEPENDENT_EXTENSIONS)
        ]
    if webrtc:
        # Enable livestreaming as part of the initial dependency solve. Calling
        # enable_extension() after startup instead triggers a re-solve that ignores
        # /app/extensions/excluded and drags the whole lula subtree back in.
        extra_args += ["--enable", "omni.kit.livestream.webrtc"]

    sim_cfg = _compact_dict({
        "headless": headless,
        "renderer": renderer,
        "multi_gpu": multi_gpu,
        "anti_aliasing": anti_aliasing,
        "hide_ui": hide_ui,
        "width": width,
        "height": height,
        "experience": experience,
    })
    if extra_args:
        sim_cfg["extra_args"] = extra_args

    app = SimulationApp(sim_cfg)

    for key, _, runtime_value in settings:
        app.set_setting(key, runtime_value)

    return app
