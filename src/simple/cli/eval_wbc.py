# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Evaluate action-chunk policies through the external SONIC controller.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from simple.agents.sonic_eval_agent import Psi0HttpPolicy, ReplayChunkPolicy
from simple.evals.sonic_wbc import SonicWbcEvalConfig, run_sonic_wbc_eval


def main(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument(help="replay or psi0")],
    data_dir: Annotated[str, typer.Option()],
    eval_dir: Annotated[str, typer.Option()] = "data/evals_wbc",
    episode_start: Annotated[int, typer.Option()] = 0,
    num_episodes: Annotated[int, typer.Option()] = 1,
    dr_level: Annotated[int, typer.Option()] = 0,
    render_hz: Annotated[int, typer.Option()] = 50,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8014,
    policy_timeout: Annotated[float, typer.Option()] = 60.0,
) -> None:
    if policy == "replay":
        policy_impl = ReplayChunkPolicy()
    elif policy == "psi0":
        policy_impl = Psi0HttpPolicy(host, port, timeout=policy_timeout)
    else:
        raise typer.BadParameter("policy must be replay or psi0")
    config = SonicWbcEvalConfig(
        env_id=env_id,
        data_dir=data_dir,
        eval_dir=eval_dir,
        episode_start=episode_start,
        num_episodes=num_episodes,
        dr_level=dr_level,
        render_hz=render_hz,
        sim_mode=sim_mode,
        headless=headless,
        save_video=save_video,
    )
    result = run_sonic_wbc_eval(config, policy_impl)
    print(json.dumps({"success_rate": result.success_rate, "eval_dir": result.eval_dir}))


def typer_main() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
