# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Physics replay of recorded full-body WBC teleop: streams each recorded 64-dim
token to the live SONIC controller and lets physics play out. Use --replay-dir
to save an MP4 of the run.
"""

from __future__ import annotations

import json
import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import subprocess
import sys
import time
import shutil
import mujoco
import numpy as np
import typer
import gymnasium as gym
from typing_extensions import Annotated, TYPE_CHECKING
import simple.envs as _  # import all envs  # noqa: F401
from tqdm import tqdm

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.cli.render_decoupled_wbc import _load_episodes, _load_episode_configs
from simple.agents.replay_wbc_agent import ReplayWbcAgent
from simple.envs.wrappers import VideoRecorder
from simple.envs.video_writer import VideoWriter
from simple.robots.g1_sonic import G1Sonic
from simple.sensors import CameraCfg
import transforms3d as t3d


def _snapshot_mujoco_state(sonic_env: SonicLocoManipEnv) -> dict[str, np.ndarray]:
    return {
        "qpos": sonic_env.mjData.qpos.copy(),
        "qvel": sonic_env.mjData.qvel.copy(),
    }


def _save_mujoco_state_trace(
    *,
    states: list[dict[str, np.ndarray]],
    replay_dir: str,
    ep_idx: int,
) -> str:
    """Persist the exact MuJoCo states consumed by the offline render pass."""
    if not states:
        raise ValueError(f"Cannot save an empty MuJoCo state trace for episode {ep_idx}")

    os.makedirs(replay_dir, exist_ok=True)
    trace_path = os.path.abspath(
        os.path.join(replay_dir, f"episode_{ep_idx}_mujoco_state_trace.npz")
    )
    np.savez_compressed(
        trace_path,
        qpos=np.stack([state["qpos"] for state in states]),
        qvel=np.stack([state["qvel"] for state in states]),
    )
    print(f"[ReplayWbc] Saved MuJoCo state trace: {trace_path}", flush=True)
    return trace_path


def _save_offline_isaac_render_job(
    *,
    trace_path: str,
    env_id: str,
    env_conf: dict,
    replay_dir: str,
    ep_idx: int,
    dr_level: int | None,
    render_hz: int,
    sim_dt: float,
    sonic_config: dict,
    debug_render: bool,
    success: bool,
    third_person_isaac: bool,
) -> str:
    """Write the versioned handoff consumed by the fresh Isaac process."""
    job_path = os.path.abspath(
        os.path.join(replay_dir, f"episode_{ep_idx}_isaac_render_job.json")
    )
    tmp_path = f"{job_path}.tmp-{os.getpid()}"
    payload = {
        "schema_version": 1,
        "env_id": env_id,
        "environment_config": env_conf,
        "replay_dir": os.path.abspath(replay_dir),
        "episode_index": ep_idx,
        "dr_level": dr_level,
        "render_hz": render_hz,
        "physics_dt": sim_dt,
        "sonic_config": sonic_config,
        "debug_render": debug_render,
        "success": success,
        "third_person_isaac": third_person_isaac,
        "trace_path": os.path.abspath(trace_path),
    }
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp_path, job_path)
    print(f"[ReplayWbc] Saved offline Isaac render job: {job_path}", flush=True)
    return job_path


def _run_offline_isaac_render_subprocess(trace_path: str, job_path: str) -> None:
    """Launch Kit in a clean interpreter after physics resources are closed."""
    command = [
        sys.executable,
        "-u",
        "-m",
        "simple.cli.render_replay_wbc",
        "--trace",
        trace_path,
        "--job",
        job_path,
    ]
    print(f"[ReplayWbc] Starting offline Isaac subprocess: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _render_state_trace_with_isaac(
    *,
    env_id: str,
    env_conf: dict,
    states: list[dict[str, np.ndarray]],
    replay_dir: str,
    ep_idx: int,
    dr_level: int | None,
    render_hz: int,
    sim_dt: float,
    sonic_config: dict,
    debug_render: bool,
    success: bool,
    third_person_isaac: bool = False,
    save_trace: bool = True,
) -> None:
    """Render a MuJoCo state trace through Isaac after controller playback.

    This is deliberately a second pass: the external SONIC controller is not
    delayed by IsaacSim stepping/rendering during physics replay, but the saved
    video still contains one Isaac frame per replay state.
    """
    if not states:
        return

    print(f"[ReplayWbc] Offline Isaac render Ep {ep_idx}: {len(states)} states", flush=True)
    if save_trace:
        _save_mujoco_state_trace(states=states, replay_dir=replay_dir, ep_idx=ep_idx)
    render_env = gym.make(
        env_id,
        sim_mode="mujoco_isaac",
        render_hz=render_hz,
        physics_dt=sim_dt,
        headless=True,
        webrtc=False,
        max_episode_steps=30000,
        sonic_config=sonic_config,
    )
    sonic_render_env: SonicLocoManipEnv = render_env.unwrapped  # type: ignore
    if debug_render:
        sonic_render_env.task.metadata["debug"] = True

    third_person_camera = None

    def _look_at_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        forward = target_pos - camera_pos
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = -forward  # USD/Isaac cameras look along local -Z.
        x_axis = np.cross(world_up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return t3d.quaternions.mat2quat(rot)  # wxyz

    try:
        print("[ReplayWbc] Offline Isaac reset begin", flush=True)
        render_env.reset(
            options={
                "state_dict": env_conf,
                "task_id": f"episode_{ep_idx}",
                "dr_level": dr_level,
            }
        )
        if third_person_isaac and sonic_render_env.isaac is not None:
            isaac_layout = sonic_render_env.isaac.task.layout
            if "third_person_isaac" not in isaac_layout.cameras:
                def _env_vec(name: str, default: list[float]) -> np.ndarray:
                    raw = os.environ.get(name, "").strip()
                    if not raw:
                        return np.array(default, dtype=np.float64)
                    vals = [float(x.strip()) for x in raw.split(",")]
                    if len(vals) != 3:
                        raise ValueError(f"{name} must be three comma-separated floats, got {raw!r}")
                    return np.array(vals, dtype=np.float64)

                # Default reviewed camera: side_look_x055_y100_z200_p45.
                # Override with SIMPLE_THIRD_PERSON_POS="x,y,z" and
                # SIMPLE_THIRD_PERSON_EULERS_DEG="roll,pitch,yaw" for paper/demo
                # camera trials without changing code each time.
                third_person_pos = _env_vec("SIMPLE_THIRD_PERSON_POS", [0.55, 1.00, 2.00])
                if os.environ.get("SIMPLE_THIRD_PERSON_EULERS_DEG"):
                    third_person_eulers_deg = _env_vec("SIMPLE_THIRD_PERSON_EULERS_DEG", [0.0, 45.0, 0.0])
                else:
                    third_person_yaw_deg = float(np.rad2deg(np.arctan2(-third_person_pos[1], -third_person_pos[0])))
                    third_person_eulers_deg = np.array([0.0, 45.0, third_person_yaw_deg], dtype=np.float64)
                print(
                    f"[ReplayWbc] third_person_isaac camera pos={third_person_pos.tolist()} "
                    f"eulers_deg={third_person_eulers_deg.tolist()}",
                    flush=True,
                )
                isaac_layout.add_camera(
                    "third_person_isaac",
                    CameraCfg(
                        uid="third_person_isaac",
                        mount="eye_on_base",
                        width=int(os.environ.get("SIMPLE_THIRD_PERSON_WIDTH", "960")),
                        height=int(os.environ.get("SIMPLE_THIRD_PERSON_HEIGHT", "540")),
                        focal_length=1.88,
                        fov=np.deg2rad(71.28),
                        near=0.2,
                        far=5.0,
                        pose=dict(
                            position=third_person_pos.tolist(),
                            eulers=np.deg2rad(third_person_eulers_deg),
                        ),
                    ),
                )
            sonic_render_env.isaac.add_cameras()
            sonic_render_env.isaac.update_layout()
        isaac_camera_keys = None
        if sonic_render_env.isaac is not None and hasattr(sonic_render_env.isaac, "cameras"):
            isaac_camera_keys = list(sonic_render_env.isaac.cameras.keys())
        print(f"[ReplayWbc] Offline Isaac reset done; isaac cameras={isaac_camera_keys}", flush=True)

        video_folder = os.path.join(replay_dir, f"episode_{ep_idx}")
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder, ignore_errors=True)
        os.makedirs(video_folder, exist_ok=True)

        writers: dict[str, VideoWriter] = {}
        try:
            # file=sys.__stderr__: once SimulationApp is up, Kit replaces
            # sys.stderr with its StreamInterceptor, which rebuffers tqdm's
            # \r updates into log lines (or drops them) — no live bar.  The
            # saved real stderr bypasses the interceptor.
            for state in tqdm(states, desc=f"  Offline Isaac render (Ep {ep_idx})", unit="frame",
                              leave=False, file=sys.__stderr__, dynamic_ncols=True):
                sonic_render_env.mjData.qpos[:] = state["qpos"]
                sonic_render_env.mjData.qvel[:] = state["qvel"]
                mujoco.mj_forward(sonic_render_env.mjModel, sonic_render_env.mjData)
                if sonic_render_env.isaac is not None:
                    sonic_render_env.isaac.step(sonic_render_env.mujoco)
                frame_dict = sonic_render_env._render_frame()
                if third_person_isaac:
                    frame_dict = {"third_person_isaac": frame_dict["third_person_isaac"]}

                if not writers:
                    print(f"[ReplayWbc] Offline Isaac frame keys: {list(frame_dict.keys())}")
                    for key, frame in frame_dict.items():
                        print(f"[ReplayWbc] Writing {key}: shape={frame.shape}, dtype={frame.dtype}")
                        writers[key] = VideoWriter(
                            os.path.join(video_folder, f"{key}.mp4"),
                            render_hz,
                            frame.shape[:2][::-1],
                        )
                for key, frame in frame_dict.items():
                    writers[key].write(frame)
        finally:
            print(f"[ReplayWbc] Offline Isaac loop finished; writers={list(writers.keys())}", flush=True)
            for writer in writers.values():
                writer.release(success)
    except BaseException as exc:
        import traceback
        print(f"[ReplayWbc] Offline Isaac render exception: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        render_env.close()


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    data_dir: Annotated[str, typer.Option(help="Recorded dataset root; defaults to data/teleop_wbc/<env_id>/level-0.")] = "",
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,
    render_hz: Annotated[int, typer.Option()] = 50,
    num_episodes: Annotated[int, typer.Option()] = -1,
    episode_start: Annotated[int, typer.Option()] = 0,
    replay_dir: Annotated[str, typer.Option(help="If set, write an MP4 of each replayed episode here (visual check).")] = "",
    debug_render: Annotated[bool, typer.Option(help="For mujoco_isaac, save the env debug split-view render instead of Isaac-only frames.")] = False,
    offline_isaac_render: Annotated[bool, typer.Option(help="Run controller playback in MuJoCo only, then render the saved MuJoCo state trace through IsaacSim after playback.")] = False,
    third_person_isaac: Annotated[bool, typer.Option(help="During offline Isaac render, save an Isaac-only third-person camera video following the robot.")] = False,
    loop_episodes: Annotated[bool, typer.Option()] = False,
    dr_level: Annotated[int | None, typer.Option()] = None,
    settle_timeout: Annotated[float, typer.Option(help="Seconds to wait for the SONIC controller to engage and hold the robot before starting token playback for each episode.")] = 10.0,
    sim_lockstep: Annotated[bool, typer.Option(help="Advance the external SONIC controller only at numbered simulation-time control boundaries.")] = False,
):
    """Stream recorded WBC tokens through the live SONIC controller and physics.

    The external SONIC controller must be running (and able to consume the
    replayed token — see ReplayWbcAgent) for this to reproduce the recording.
    """
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    # Load episodes + per-episode scene configs from the recorded dataset.
    if not data_dir:
        data_dir = f"data/teleop_wbc/{env_id}/level-0"
    print(f"Loading dataset from {data_dir} ...")
    episodes = _load_episodes(data_dir)
    is_mapping = hasattr(episodes, "keys")
    episode_ids = sorted(episodes.keys()) if is_mapping else list(range(len(episodes)))
    total_episodes = len(episode_ids)
    if num_episodes < 0:
        num_episodes = total_episodes
    if not loop_episodes:
        # Bug #4: the loop is range(episode_start, episode_start + num_episodes),
        # so an unclamped count (or num_episodes=-1 => total_episodes) overshoots
        # past the last episode when episode_start > 0.  Cap it to what remains.
        num_episodes = max(0, min(num_episodes, total_episodes - episode_start))
    print(f"Loaded {total_episodes} episodes, will replay {num_episodes}")

    episode_configs = _load_episode_configs(data_dir)
    if episode_configs:
        print(f"Loaded environment configs for {len(episode_configs)} episodes")
    else:
        print("WARNING: No environment_config found in episodes.jsonl")

    # Environment.
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt  # 0.02 s (50 Hz) — paces the loop / on-screen viewer

    requested_sim_mode = sim_mode
    playback_sim_mode = "mujoco" if offline_isaac_render and "isaac" in sim_mode and replay_dir else sim_mode
    if offline_isaac_render:
        if "isaac" not in requested_sim_mode:
            print("[ReplayWbc] --offline-isaac-render requested without an Isaac sim mode; using normal playback.")
        elif not replay_dir:
            print("[ReplayWbc] --offline-isaac-render needs --replay-dir to save video; using normal live rendering.")
        else:
            print(
                "[ReplayWbc] Offline Isaac render enabled: controller playback "
                f"runs with sim_mode={playback_sim_mode}, then renders {requested_sim_mode} from saved states."
            )

    print(f"Creating environment: {env_id} (sim_mode={playback_sim_mode})")
    env = gym.make(
        env_id,
        sim_mode=playback_sim_mode,
        render_hz=render_hz,
        physics_dt=sim_dt,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=30000,
        sonic_config=sonic_config,
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert isinstance(robot, G1Sonic)
    # This replay path drives an external SONIC controller. Keep MuJoCo/lowstate
    # updates at physics rate, but only synchronize Isaac once per 50 Hz control
    # frame so rendering does not delay every controller heartbeat substep.
    sonic_env.sync_isaac_every_substep = False
    if debug_render:
        task.metadata["debug"] = True

    offline_render_enabled = bool(replay_dir) and offline_isaac_render and "isaac" in requested_sim_mode and playback_sim_mode == "mujoco"
    live_video_recording = bool(replay_dir) and not offline_render_enabled
    offline_render_jobs: list[tuple[str, str]] = []

    if live_video_recording:
        env = VideoRecorder(env, video_folder=replay_dir, framerate=render_hz)

    # Agent owns the DDS bridge + token publisher; create it AFTER gym.make so the
    # env's ChannelFactoryInitialize has run.
    agent = ReplayWbcAgent(robot, sonic_config, sim_lockstep=sim_lockstep)

    # Keep rt/lowstate flowing at the physics rate (mirrors teleop_wbc): the
    # controller aborts if state is older than 500 ms, and env.reset() pauses
    # physics long enough to trip that.
    sonic_env.substep_callback = lambda: agent.cache_proprio(robot.prepare_obs())
    if not sim_lockstep:
        agent.start_state_heartbeat()
    # The controller sits in "LowState not ready" until rt/lowstate flows
    # (first cache_proprio, at episode 0's pose seeding), then runs its 3 s
    # INIT ramp (InitControl) before it can engage.  Ramp lowcmds already carry
    # real gains, so the settle loop's kp check alone cannot tell ramp from
    # engaged policy — give the settle a time floor covering the ramp (+ engage
    # margin), armed when the first lowstate goes out.  SONIC then warms up
    # during the kinematic hold instead of the ramp eating the first ~150
    # frames of playback.
    sonic_warmup_until: float | None = None
    lockstep_controller_initialized = False


    # Per-episode task-success bookkeeping (terminated == task.check_success).
    ep_results: dict[int, bool] = {}

    try:
        pbar = tqdm(
            range(episode_start, episode_start + num_episodes),
            desc="Episodes", unit="ep", position=0, leave=True,
        )
        for ep_idx in pbar:
            # Bug #3: non-loop selection must map the loop counter through
            # episode_ids too — using raw ep_idx as the dataset key skips every
            # episode whose key is non-contiguous (e.g. keys {0,2,5,7}).  Bug #4's
            # clamp keeps ep_idx < total_episodes here, so this index is in range.
            src_ep_id = episode_ids[ep_idx % total_episodes] if loop_episodes else episode_ids[ep_idx]

            if is_mapping:
                if src_ep_id not in episodes:
                    print(f"Episode {src_ep_id} not found, skipping")
                    continue
            elif src_ep_id >= total_episodes:
                print(f"Episode {src_ep_id} not found, skipping")
                continue

            ep_data = episodes[src_ep_id]

            # Reset to the exact recorded scene.  A dataset can lack an
            # environment_config (absent/partial meta/episodes.jsonl), so skip
            # such episodes instead of crashing the whole run on a KeyError.
            env_conf = episode_configs.get(src_ep_id)
            if env_conf is None:
                print(f"Episode {src_ep_id}: no environment_config, skipping")
                continue
            if env_conf.get("uid") != task.uid:
                print(
                    f"[warn] env/dataset task mismatch: dataset uid={env_conf.get('uid')!r} "
                    f"vs task.uid={task.uid!r} - overriding env_conf uid"
                )
                env_conf["uid"] = task.uid
            observation, info = env.reset(
                options={
                    "state_dict": env_conf,
                    "task_id": f"episode_{ep_idx}",
                    "dr_level": dr_level,
                }
            )

            # --- Seed the recorded START configuration ------------------------
            # env.reset spawns init_qpos zeros (G1Sonic.init_joint_states) at the
            # layout base pose, NOT the recorded start.  Write row0 into mjData so
            # the external controller engages onto the exact recorded pose; the
            # settle loop then holds this seeded stance kinematically (there is no
            # crane in replay) until the controller takes over.
            row0 = ep_data.iloc[0]
            mjData = sonic_env.mjData
            joints = sonic_env.mujoco.joints  # == robot.joints, keyed by joint name
            obs_state = np.asarray(row0["observation.state"], dtype=np.float64)
            base_pose_seed = np.asarray(row0["observation.base_pose"], dtype=np.float64)
            base_vel_seed = (
                np.asarray(row0["observation.base_vel"], dtype=np.float64)
                if "observation.base_vel" in row0 else None
            )

            def _apply_start_pose(
                restore_base_vel: bool,
                *,
                publish_state: bool = True,
                sync_isaac: bool = True,
            ):
                """Write the recorded start configuration into mjData, refresh
                kinematics, and republish the seeded proprio to the controller.

                Seeds the episode (restore_base_vel=True) and, with
                restore_base_vel=False, re-asserts the seeded stance each settle
                step so the floating-base robot cannot collapse before the
                controller engages (no crane in replay).
                """
                # Floating base qpos[:7] = [x,y,z, qw,qx,qy,qz]; 43 body+hand joints
                # mapped by NAME (order-agnostic: recorded over robot.joint_names).
                mjData.qpos[:7] = base_pose_seed
                for jname, jval in zip(robot.joint_names, obs_state):
                    if jname in joints:  # defensive; every joint_name is a key
                        joints[jname].qpos = jval
                # Zero velocities, then (seed only) restore the recorded base vel.
                mjData.qvel[:] = 0.0
                if restore_base_vel and base_vel_seed is not None:
                    mjData.qvel[:6] = base_vel_seed
                # Forward kinematics so body/site/camera poses + proprio reflect it.
                # If IsaacSim rendering is enabled, sync it immediately after this
                # direct MuJoCo state write; env.step() already syncs Isaac during
                # dynamic playback, but the replay seeding/settle hold bypasses
                # env.step().
                mujoco.mj_forward(sonic_env.mjModel, mjData)
                # Refresh the heartbeat proprio cache before any Isaac sync can
                # block, so rt/lowstate publishes the SEEDED pose promptly.
                if publish_state:
                    agent.cache_proprio(robot.prepare_obs())
                if sync_isaac and sonic_env.isaac is not None:
                    sonic_env.isaac.step(sonic_env.mujoco)

            _apply_start_pose(
                restore_base_vel=True,
                publish_state=not sim_lockstep,
            )
            # First lowstate is on the wire now — arm the INIT-ramp floor once.
            if not sim_lockstep and sonic_warmup_until is None:
                sonic_warmup_until = time.monotonic() + 4.0
            # Re-read observation so the settle loop / video / _init_writers start
            # from the seeded frame.
            observation = sonic_env._get_obs()
            # -----------------------------------------------------------------

            agent.load_episode(ep_data)
            agent.reset_policy()

            if sim_lockstep:
                # Lockstep uses no planner model. Switch the input manager to its
                # streamed-token delegate before the first inline Input() call;
                # SONIC_AUTO_START performs the WAIT_FOR_CONTROL transition.
                agent._inj.enter_token_mode()

                if not lockstep_controller_initialized:
                    agent.begin_lockstep_session(robot.prepare_obs())
                    for _ in range(152):
                        startup_snapshot = agent.commit_lockstep_idle_boundary()
                        startup_action = agent._forward_controller_action(startup_snapshot)
                        for _substep in range(4):
                            sonic_env.mujoco.apply_action(startup_action)
                            sonic_env.mujoco.step(render=False)
                            _apply_start_pose(
                                restore_base_vel=False,
                                publish_state=True,
                                sync_isaac=False,
                            )
                    lockstep_controller_initialized = True

                    # Startup is a separate session and trace. Rebuild/reset the
                    # episode so formal playback starts at simulator time ~0.005.
                    observation, info = env.reset(
                        options={
                            "state_dict": env_conf,
                            "task_id": f"episode_{ep_idx}",
                            "dr_level": dr_level,
                        }
                    )
                    mjData = sonic_env.mjData
                    joints = sonic_env.mujoco.joints
                    _apply_start_pose(
                        restore_base_vel=True,
                        publish_state=False,
                    )

                agent.begin_lockstep_session(robot.prepare_obs())
                observation = sonic_env._get_obs()

            # --- Wait for the external controller to engage and hold the robot ---
            # The live SONIC controller is the balancer; until it engages, we hold
            # the seeded start stance kinematically (no crane in replay) so the
            # floating-base robot does not collapse during the engage window.
            # Once it is holding with real gains, start token playback.
            #
            # Latch control FIRST via a PLANNER-mode start command: the controller
            # only advances WAIT_FOR_CONTROL -> CONTROL when operator_state.start is
            # set, and STREAMED_MOTION (token mode) never sets it.  We switch into
            # token mode after the settle loop, once the controller is holding.
            if not sim_lockstep:
                agent._inj.engage()
                settle_deadline = time.monotonic() + settle_timeout
                settle_pbar = tqdm(desc=f"  Waiting for controller (Ep {ep_idx})",
                                   unit="step", position=1, leave=False)
                try:
                    while (not agent.controller_holding or time.monotonic() < sonic_warmup_until) \
                            and time.monotonic() < settle_deadline:
                        step_start = time.monotonic()
                        # Hold the seeded stance kinematically (do NOT step dynamics —
                        # nothing keeps the floating base upright without a crane).  The
                        # heartbeat thread keeps publishing this pose so the controller
                        # can engage onto it.
                        _apply_start_pose(restore_base_vel=False)
                        observation = sonic_env._get_obs()
                        sonic_env.update_viewer()
                        sleep_time = control_dt - (time.monotonic() - step_start)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        settle_pbar.update(1)
                finally:
                    settle_pbar.close()

                if not agent.controller_holding:
                    print(
                        f"[ReplayWbc] Ep {ep_idx}: controller never engaged within "
                        f"{settle_timeout:.0f}s — is the SONIC controller running and "
                        "engaged? Replaying anyway, but with no controller holding the "
                        "robot it will not track the recorded motion."
                    )

            # Start the debug video at the first replayed frame (drop the settle).
            episode_state_trace: list[dict[str, np.ndarray]] = []
            if offline_render_enabled:
                episode_state_trace.append(_snapshot_mujoco_state(sonic_env))
            if isinstance(env, VideoRecorder):
                env._init_writers(observation)

            # Flip the controller into STREAMED_MOTION so GetExternalTokenState()
            # forwards the token we are about to stream (planner=false -> ZMQ
            # toggle -> use_zmq_stream=true).  Must follow engage()/settle so
            # control is already latched.
            if not sim_lockstep:
                agent._inj.enter_token_mode()

            # --- Token-driven physics playback ---
            episode_steps = len(ep_data)
            step_pbar = tqdm(
                total=episode_steps, desc=f"  Frames (Ep {ep_idx})", unit="frame",
                leave=False, position=1, bar_format="{desc} [{bar}] {n_fmt}/{total_fmt}",
            )
            ep_success = False
            try:
                while True:
                    step_start = time.monotonic()
                    try:
                        action = agent.get_action(observation)
                    except StopIteration:
                        break

                    observation, reward, terminated, truncated, info = env.step(action)
                    if offline_render_enabled:
                        episode_state_trace.append(_snapshot_mujoco_state(sonic_env))
                    sonic_env.update_viewer()
                    sonic_env.update_reward()

                    if terminated or truncated:
                        # terminated is task.check_success; truncated is only
                        # the TimeLimit wrapper (= failure to finish in time).
                        ep_success = terminated
                        break

                    step_pbar.update(1)
                    if not sim_lockstep:
                        sleep_time = control_dt - (time.monotonic() - step_start)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
            finally:
                step_pbar.close()

            ep_results[src_ep_id] = ep_success
            n_ok = sum(ep_results.values())
            pbar.set_postfix_str(f"success {n_ok}/{len(ep_results)}")
            print(f"[ReplayWbc] Ep {ep_idx} (src {src_ep_id}): "
                  f"{'SUCCESS' if ep_success else 'FAIL (tokens ended / timeout)'}")

            if offline_render_enabled:
                trace_path = _save_mujoco_state_trace(
                    states=episode_state_trace,
                    replay_dir=replay_dir,
                    ep_idx=ep_idx,
                )
                job_path = _save_offline_isaac_render_job(
                    trace_path=trace_path,
                    env_id=env_id,
                    env_conf=dict(env_conf),
                    replay_dir=replay_dir,
                    ep_idx=ep_idx,
                    dr_level=dr_level,
                    render_hz=render_hz,
                    sim_dt=sim_dt,
                    sonic_config=sonic_config,
                    debug_render=debug_render,
                    success=ep_success,
                    third_person_isaac=third_person_isaac,
                )
                offline_render_jobs.append((trace_path, job_path))

            if isinstance(env, VideoRecorder):
                env.release()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        if ep_results:
            n_ok = sum(ep_results.values())
            print(f"\n[ReplayWbc] Success rate: {n_ok}/{len(ep_results)} "
                  f"({100.0 * n_ok / len(ep_results):.0f}%)")
            for eid in sorted(ep_results):
                print(f"    episode {eid}: {'success' if ep_results[eid] else 'fail'}")
        env.close()
        agent.close()

        if offline_render_jobs:
            for trace_path, job_path in offline_render_jobs:
                _run_offline_isaac_render_subprocess(trace_path, job_path)


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
