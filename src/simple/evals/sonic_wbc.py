# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Closed-loop evaluation through the external SONIC lockstep controller.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import tyro

import simple.envs as _  # noqa: F401
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.lockstep import LockstepTimeout
from simple.agents.sonic_eval_agent import PolicyError, SonicActionPolicy, SonicEvalAgent
from simple.cli.render_decoupled_wbc import _load_episode_configs, _load_episodes
from simple.envs.sonic_loco_manip import SonicLocoManipEnv
from simple.envs.wrappers import VideoRecorder
from simple.robots.g1_sonic import G1Sonic


CONTROL_DT = 0.02
MIN_MODEL_BUDGET_STEPS = 1500


class TerminationReason(str, Enum):
    TASK_SUCCESS = "task_success"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"
    LOCKSTEP_TIMEOUT = "lockstep_timeout"


def episode_budget_steps(policy_name: str, source_length: int) -> int:
    if source_length <= 0:
        raise ValueError("source episode must contain at least one frame")
    if policy_name == "replay":
        return source_length + 10
    return max(2 * source_length, MIN_MODEL_BUDGET_STEPS)


@dataclass(frozen=True)
class SonicWbcEvalConfig:
    env_id: str
    data_dir: str
    eval_dir: str
    episode_start: int = 0
    num_episodes: int = 1
    dr_level: int = 0
    render_hz: int = 50
    sim_mode: str = "mujoco_isaac"
    headless: bool = True
    save_video: bool = True


@dataclass
class EpisodeResult:
    episode_index: int
    source_length: int
    budget_steps: int
    executed_steps: int
    budget_fraction: float
    task_success: bool
    termination_reason: str
    progressing_at_timeout: bool
    episode_wall_seconds: float
    simulation_seconds: float
    inference_count: int
    session_id: int
    ack_count: int


@dataclass
class SonicWbcEvalResult:
    policy: str
    episodes: list[EpisodeResult]
    success_rate: float
    budget_exhausted_count: int
    model_error_count: int
    lockstep_timeout_count: int
    success_rate_is_lower_bound: bool
    eval_dir: str
    exec_horizon: int = 30


def _sonic_config() -> dict[str, Any]:
    config = tyro.cli(
        SimLoopConfig,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
        args=[],
    )
    result = config.load_wbc_yaml()
    result["ENV_NAME"] = "simple"
    return result


def _progress_sample(env: SonicLocoManipEnv, info: dict[str, Any]) -> dict[str, float]:
    pelvis = np.asarray(env.mjData.body("pelvis").xpos[:2], dtype=np.float64)
    target = np.asarray(info.get("target", [np.nan] * 7), dtype=np.float64)
    table = env.task.layout.actors.get("table")
    table_xy = (
        np.asarray(table.pose.position[:2], dtype=np.float64)
        if table is not None
        else np.full(2, np.nan)
    )
    target_xy = target[:2]
    return {
        "sim_time": float(env.mjData.time),
        "reward": float(env.task.reward),
        "pelvis_target_xy": float(np.linalg.norm(pelvis - target_xy)),
        "target_z": float(target[2]),
        "target_table_xy": float(np.linalg.norm(target_xy - table_xy)),
    }


def _progressing_at_timeout(samples: list[dict[str, float]]) -> tuple[bool, dict[str, float]]:
    if not samples:
        return False, {}
    end_time = samples[-1]["sim_time"]
    window = [sample for sample in samples if sample["sim_time"] >= end_time - 5.0]
    first, last = window[0], window[-1]
    metrics = {
        "reward_gain": last["reward"] - first["reward"],
        "pelvis_target_improvement": first["pelvis_target_xy"] - last["pelvis_target_xy"],
        "target_height_gain": last["target_z"] - first["target_z"],
        "target_table_improvement": first["target_table_xy"] - last["target_table_xy"],
    }
    progressing = bool(
        metrics["reward_gain"] > 0
        or metrics["pelvis_target_improvement"] >= 0.05
        or metrics["target_height_gain"] >= 0.05
        or metrics["target_table_improvement"] >= 0.05
    )
    return progressing, metrics


def run_sonic_wbc_eval(
    config: SonicWbcEvalConfig,
    policy: SonicActionPolicy,
) -> SonicWbcEvalResult:
    if config.render_hz != 50:
        raise ValueError("SONIC lockstep evaluation requires exactly 50 Hz")
    if config.episode_start < 0 or config.num_episodes <= 0:
        raise ValueError("episode_start must be non-negative and num_episodes positive")

    output = Path(config.eval_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    episodes = _load_episodes(config.data_dir)
    episode_configs = _load_episode_configs(config.data_dir)
    episode_ids = sorted(episodes.keys()) if hasattr(episodes, "keys") else list(range(len(episodes)))
    selected = episode_ids[config.episode_start : config.episode_start + config.num_episodes]
    if not selected:
        raise ValueError("no episodes selected")
    missing = [episode_id for episode_id in selected if episode_id not in episode_configs]
    if missing:
        raise ValueError(
            f"no environment_config in {config.data_dir}/meta/episodes.jsonl for episodes {missing}; "
            "point --data-dir at the SIMPLE teleop dataset (…/<env-id>/level-N), not a "
            "training-converted copy"
        )

    sonic_config = _sonic_config()
    sim_dt = float(sonic_config["SIMULATE_DT"])
    if not np.isclose(sim_dt, 0.005):
        raise ValueError(f"expected 5 ms physics step, got {sim_dt}")

    raw_env = gym.make(
        config.env_id,
        sim_mode=config.sim_mode,
        render_hz=config.render_hz,
        physics_dt=sim_dt,
        headless=config.headless,
        webrtc=False,
        max_episode_steps=30000,
        sonic_config=sonic_config,
    )
    sonic_env: SonicLocoManipEnv = raw_env.unwrapped  # type: ignore[assignment]
    sonic_env.sync_isaac_every_substep = False
    sonic_env.set_timing_event_sink(events.append)
    robot = sonic_env.task.robot
    if not isinstance(robot, G1Sonic):
        raise TypeError("SONIC evaluation requires G1Sonic")
    env = (
        VideoRecorder(raw_env, video_folder=str(output), framerate=config.render_hz)
        if config.save_video
        else raw_env
    )
    agent = SonicEvalAgent(robot, sonic_config, policy, event_sink=events.append)
    sonic_env.substep_callback = lambda: agent.cache_proprio(robot.prepare_obs())
    agent._inj.enter_token_mode()

    results: list[EpisodeResult] = []
    controller_initialized = False
    fatal_error: BaseException | None = None
    try:
        for episode_id in selected:
            episode_data = episodes[episode_id]
            env_conf = episode_configs[episode_id]
            if env_conf.get("uid") != sonic_env.task.uid:
                env_conf["uid"] = sonic_env.task.uid

            observation, info = env.reset(
                options={
                    "state_dict": env_conf,
                    "task_id": f"episode_{episode_id}",
                    "dr_level": config.dr_level,
                }
            )
            row0 = episode_data.iloc[0]
            base_pose = np.asarray(row0["observation.base_pose"], dtype=np.float64)
            base_vel = np.asarray(row0["observation.base_vel"], dtype=np.float64)
            joint_state = np.asarray(row0["observation.state"], dtype=np.float64)

            def apply_start_pose(
                restore_base_vel: bool,
                *,
                publish_state: bool,
                sync_isaac: bool,
            ) -> None:
                sonic_env.mjData.qpos[:7] = base_pose
                for name, value in zip(robot.joint_names, joint_state):
                    sonic_env.mujoco.joints[name].qpos = value
                sonic_env.mjData.qvel[:] = 0.0
                if restore_base_vel:
                    sonic_env.mjData.qvel[:6] = base_vel
                mujoco.mj_forward(sonic_env.mjModel, sonic_env.mjData)
                if publish_state:
                    agent.cache_proprio(robot.prepare_obs())
                if sync_isaac and sonic_env.isaac is not None:
                    sonic_env.isaac.step(sonic_env.mujoco)

            apply_start_pose(True, publish_state=False, sync_isaac=True)
            agent.reset_evaluation(episode_data, episode_id)

            if not controller_initialized:
                agent.begin_lockstep_session(robot.prepare_obs())
                for _ in range(152):
                    snapshot = agent.commit_lockstep_idle_boundary()
                    startup_action = agent._forward_controller_action(snapshot)
                    for _substep in range(4):
                        sonic_env.mujoco.apply_action(startup_action)
                        sonic_env.mujoco.step(render=False)
                        apply_start_pose(False, publish_state=True, sync_isaac=False)
                controller_initialized = True
                observation, info = env.reset(
                    options={
                        "state_dict": env_conf,
                        "task_id": f"episode_{episode_id}",
                        "dr_level": config.dr_level,
                    }
                )
                apply_start_pose(True, publish_state=False, sync_isaac=True)

            agent.begin_lockstep_session(robot.prepare_obs())
            session_id = int(agent.lockstep_barrier.session_id)  # type: ignore[union-attr]
            observation = sonic_env._get_obs()
            if isinstance(env, VideoRecorder):
                env._init_writers(observation)

            progress: list[dict[str, float]] = []
            budget = episode_budget_steps(policy.name, len(episode_data))
            termination = TerminationReason.BUDGET_EXHAUSTED
            started_ns = time.perf_counter_ns()
            error_message = None
            steps = 0

            while steps < budget:
                next_step = steps + 1
                sonic_env.set_timing_context(
                    {"episode": episode_id, "session": session_id, "step": next_step}
                )
                events.append(
                    {
                        "kind": "policy_turn_start",
                        "perf_ns": time.perf_counter_ns(),
                        "episode": episode_id,
                        "session": session_id,
                        "step": next_step,
                    }
                )
                try:
                    action = agent.get_evaluation_action(
                        observation,
                        instruction=sonic_env.task.instruction,
                    )
                except LockstepTimeout as exc:
                    termination = TerminationReason.LOCKSTEP_TIMEOUT
                    error_message = str(exc)
                    break
                except (PolicyError, OSError, RuntimeError, ValueError) as exc:
                    termination = TerminationReason.MODEL_ERROR
                    error_message = f"{type(exc).__name__}: {exc}"
                    break

                events.append(
                    {
                        "kind": "env_step_begin",
                        "perf_ns": time.perf_counter_ns(),
                        "episode": episode_id,
                        "session": session_id,
                        "step": next_step,
                    }
                )
                observation, _reward, terminated, _truncated, info = env.step(action)
                events.append(
                    {
                        "kind": "env_step_end",
                        "perf_ns": time.perf_counter_ns(),
                        "episode": episode_id,
                        "session": session_id,
                        "step": next_step,
                        "sim_time": float(sonic_env.mjData.time),
                    }
                )
                steps += 1
                progress.append(_progress_sample(sonic_env, info))
                sonic_env.update_viewer()
                sonic_env.update_reward()
                if terminated:
                    termination = TerminationReason.TASK_SUCCESS
                    break

            episode_wall = (time.perf_counter_ns() - started_ns) / 1e9
            if isinstance(env, VideoRecorder):
                env.release()
            progressing, progress_metrics = _progressing_at_timeout(progress)
            if termination != TerminationReason.BUDGET_EXHAUSTED:
                progressing = False

            control_acks = [
                event
                for event in events
                if event.get("kind") == "ack_complete"
                and event.get("phase") == "control"
                and int(event["ack"][1]) == session_id
            ]
            result = EpisodeResult(
                episode_index=episode_id,
                source_length=len(episode_data),
                budget_steps=budget,
                executed_steps=steps,
                budget_fraction=steps / budget,
                task_success=termination == TerminationReason.TASK_SUCCESS,
                termination_reason=termination.value,
                progressing_at_timeout=progressing,
                episode_wall_seconds=episode_wall,
                simulation_seconds=steps * CONTROL_DT,
                inference_count=len(agent.inference_records),
                session_id=session_id,
                ack_count=len(control_acks),
            )
            results.append(result)
            sonic_env.set_timing_context(None)

        successes = sum(result.task_success for result in results)
        exhausted = sum(
            result.termination_reason == TerminationReason.BUDGET_EXHAUSTED.value
            for result in results
        )
        model_errors = sum(
            result.termination_reason == TerminationReason.MODEL_ERROR.value
            for result in results
        )
        lockstep_timeouts = sum(
            result.termination_reason == TerminationReason.LOCKSTEP_TIMEOUT.value
            for result in results
        )
        final = SonicWbcEvalResult(
            policy=policy.name,
            episodes=results,
            success_rate=successes / len(results),
            budget_exhausted_count=exhausted,
            model_error_count=model_errors,
            lockstep_timeout_count=lockstep_timeouts,
            success_rate_is_lower_bound=any(
                result.termination_reason == TerminationReason.BUDGET_EXHAUSTED.value
                and result.progressing_at_timeout
                for result in results
            ),
            eval_dir=str(output),
            exec_horizon=agent.exec_horizon,
        )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    **asdict(final),
                    "episodes": [asdict(result) for result in results],
                },
                indent=2,
            )
            + "\n"
        )
        (output / "eval_stats.txt").write_text(
            "\n".join(
                [
                    f"episode_{result.episode_index}: {result.task_success} ({result.termination_reason})"
                    for result in results
                ]
                + [f"success rate: {final.success_rate:.6f}"]
            )
            + "\n"
        )
        return final
    except BaseException as exc:
        fatal_error = exc
        raise
    finally:
        if fatal_error is not None:
            (output / "fatal_error.txt").write_text(
                f"{type(fatal_error).__name__}: {fatal_error}\n"
            )
        try:
            agent.close()
        finally:
            raw_env.close()
