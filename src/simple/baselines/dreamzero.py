# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
DreamZero G1 Whole-Body baseline adapter.

This agent is the client-side translator between SIMPLE's G1 whole-body envs
and a remote DreamZero policy server (see dreamzero-g1-clean/eval_utils/
serve_dreamzero_g1_simple.py). It:

  1. Takes the env's observation dict (43D joint_qpos, stereo head camera)
     and slices/packs it into the 32D Psi0-flat proprioception that the
     DreamZero G1 model was trained on.
  2. POSTs to the remote server's /act endpoint (same protocol as Psi0Agent,
     via HttpActionClient).
  3. Unpacks the returned (chunk_len, 36) action tensor into per-frame
     ActionCmd objects that the G1 wholebody task's apply_action expects
     (upper-body target_qpos + waist_qpos + 8-dim locomotion command).

The 36-D action layout is identical to Psi0's because the training data
(USC-PSI-Lab/psi-data/simple/*) is Psi0-flat:

    action[0:14]   hands  (left thumb 3, middle 2, index 2, right 7)
    action[14:28]  arms   (left arm 7, right arm 7)
    action[28:31]  waist  (roll, pitch, yaw)
    action[31]     target base height (~0.75 standing)
    action[32:34]  vx, vy
    action[34]     turning flag
    action[35]     target yaw

If you retrain DreamZero with a different action layout, update the
unpacking code below AND from_dreamzero_upper_joints().
"""

import os

import numpy as np
import requests

from simple.agents.primitive_agent import PrimitiveAgent
from simple.core.action import ActionCmd
from simple.baselines.client import HttpActionClient


# joint_qpos slicing to reconstruct the 32D Psi0-flat state the model was
# trained on. Must match scripts/postprocess_psi0.py in the training repo.
STATE_SLICES = [
    ("left_hand_thumb",  29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index",  32, 34),
    ("right_hand",       36, 43),
    ("left_arm",         15, 22),
    ("right_arm",        22, 29),
]


def from_dreamzero_upper_joints(dz_action):
    """Reorder the 28D upper-body slice of a DreamZero action into the
    joint_names[15:] order expected by G1Wholebody's apply_action.

    Layout in = [ arms(14), left_thumb(3), left_index(2), left_middle(2), right_hand(7) ]
    Layout out = robot.joint_names[15:] order (hands first, then arms).
    """
    return np.concatenate([
        dz_action[14:28],  # arms    (14)
        dz_action[0:3],    # L thumb  (3)
        dz_action[5:7],    # L index  (2)   (note: index before middle in robot order)
        dz_action[3:5],    # L middle (2)
        dz_action[7:14],   # R hand   (7)
    ])


class DreamzeroAgent(PrimitiveAgent):
    """Client for a remote DreamZero G1 policy server.

    Example:
        env = gym.make("simple/G1WholebodyXMovePickTeleop-v0", sim_mode="mujoco")
        agent = DreamzeroAgent(env.unwrapped.task.robot,
                               host="172.17.0.1", port=22085,
                               upsample_factor=1)
        obs, info = env.reset()
        for _ in range(max_steps):
            action = agent.get_action(obs, instruction=task.instruction, info=info)
            obs, *_ = env.step(action)

    Args:
        robot: the G1 wholebody robot instance from task.robot.
        host, port: HTTP host/port of the DreamZero server (speaks SIMPLE's
            HttpActionClient protocol — /act endpoint, numpy-serialized JSON).
        upsample_factor: how many sim steps each predicted action is held for.
            Set to sim_fps / model_fps. DreamZero G1 SIMPLE is trained at 50Hz
            action horizon; SIMPLE's G1 env also runs at 50Hz, so 1 is correct
            for the default checkpoint.
    """

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        upsample_factor: int = 1,
        action_horizon: int = 24,
        client=None,
        **kwargs,
    ):
        super().__init__(robot, **kwargs)

        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = client or HttpActionClient(self.server_ip, self.server_port)

        # Wait for server, then query model config so frame collection
        # matches training. Both containers are often launched in parallel —
        # don't give up on /config just because the server is still warming.
        self._wait_for_server()
        self._server_config = self._query_server_config()
        self.action_horizon = int(
            str(self._server_config.get("action_horizon", action_horizon)),
        )
        self._video_frames_per_chunk = self._server_config.get("video_frames_per_chunk", 8)
        self._video_stride = self._server_config.get("video_stride", self.action_horizon // 8)

        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()
        self._obs_frame_buffer: list[np.ndarray] = []

        # Last high-level command sent to the low-level controller. Kept as
        # part of the 32D proprio the policy expects (torso RPY + height).
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75], dtype=np.float32)
        self._reset_history = True

    def _wait_for_server(self, timeout_s: float = 900.0, poll_interval_s: float = 5.0) -> None:
        """Block until the server /health returns 200 or timeout elapses.

        Server startup (shard load + DeepSpeed init + compile warmup) can take
        3-5 min for the 14B full-ft model. Simulator init on the client side
        is comparable, so launching both in parallel is the usual workflow —
        wait here rather than giving up on the first failed /config.
        """
        import time
        url = f"http://{self.server_ip}:{self.server_port}/health"
        t0 = time.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    elapsed = time.time() - t0
                    print(f"[DreamzeroAgent] /health ok after {elapsed:.1f}s ({attempt} attempts)")
                    return
            except Exception:
                pass
            elapsed = time.time() - t0
            if elapsed > timeout_s:
                print(f"[DreamzeroAgent] /health still not responding after {elapsed:.0f}s — giving up, proceeding with defaults")
                return
            if attempt == 1 or attempt % 6 == 0:
                print(f"[DreamzeroAgent] waiting for server at {url} ... ({elapsed:.0f}s elapsed)")
            time.sleep(poll_interval_s)

    def _query_server_config(self) -> dict:
        """Query the DreamZero server for model config (action_horizon, video_stride, etc.)."""
        try:
            resp = requests.get(f"http://{self.server_ip}:{self.server_port}/config", timeout=5)
            if resp.status_code == 200:
                cfg = resp.json()
                print(f"[DreamzeroAgent] server config: {cfg}")
                return cfg
        except Exception as e:
            print(f"[DreamzeroAgent] could not query /config: {e}, using defaults")
        return {}

    def _make_session_id(self) -> str:
        return f"dreamzero-{os.getpid()}-{self._session_idx}"

    def get_action(
        self,
        observation,
        instruction=None,
        info=None,
        conditions=None,
        **kwargs,
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        # Always buffer observation frames for the next server query.
        self._obs_frame_buffer.append(observation["head_stereo_left"])

        if len(self._action_queue) == 0:
            # Subsample collected frames at video_stride to match training.
            # Training samples frames at offsets [0, stride, 2*stride, ...]
            # from the chunk anchor. The buffer contains observations from
            # the execution of the previous action chunk, starting at the
            # chunk boundary (offset 0).
            # Subsample frames at video_stride, aligned to the END so the most
            # recent observation is always included. The server takes the last
            # num_frame_per_block latent frames from the VAE output, so the
            # newest raw frame must be at the tail.
            # First query has 1 frame; subsequent queries have ~action_horizon.
            buf = self._obs_frame_buffer
            if len(buf) <= 1:
                frames = list(buf)
            else:
                # Sample backward from the last frame: [..., buf[-1-2s], buf[-1-s], buf[-1]]
                n_want = self._video_frames_per_chunk
                stride = self._video_stride
                indices = [len(buf) - 1 - i * stride for i in range(n_want)]
                indices = [max(0, idx) for idx in reversed(indices)]
                frames = [buf[i] for i in indices]
            video = np.stack(frames, axis=0)  # (T, H, W, 3)
            self._obs_frame_buffer = self._obs_frame_buffer[-1:]

            observations = {
                "rgb_head_stereo_left": video,
            }

            # Reconstruct 32D Psi0-flat state from 43D env qpos + last torso cmd.
            proprio = observation["joint_qpos"][None]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES]
                + [self._last_cmd_torso_rpyh[None]],
                axis=1,
            ).astype(np.float32)  # shape (1, 32)
            state_dict = {"states": states}

            # Tell server to reset its KV cache at episode boundary.
            history = {
                "session_id": self._session_id,
                "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                "step_index": int(self._global_step_idx),
            }
            if self._reset_history:
                history["reset"] = True
                self._reset_history = False

            pred_action, *_ = self.client.query_action(
                observations,
                instruction or "perform the task",
                state_dict,
                {},
                history=history,
                dataset="dreamzero_g1_simple",
            )
            # Receding horizon: execute only the first action_horizon actions
            # out of the full chunk. With action_horizon < chunk_len, the agent
            # re-queries the server more often with fresher observations.
            n_execute = min(self.action_horizon, pred_action.shape[0])
            print(f"[DreamzeroAgent] received chunk of {pred_action.shape[0]} actions, executing {n_execute}")

            # Unpack chunk -> ActionCmd queue. Each of the 36 dims maps to a
            # specific part of the G1 whole-body command as documented at the
            # top of this file. The same layout Psi0Agent uses.
            for i in range(n_execute):
                for _ in range(self.upsample_factor):
                    target_qpos = dict(zip(
                        self.robot.joint_names[15:],
                        from_dreamzero_upper_joints(pred_action[i][:28]),
                    ))

                    target_waist_qpos = {
                        "waist_roll_joint":  pred_action[i][28],
                        "waist_pitch_joint": pred_action[i][29],
                        "waist_yaw_joint":   pred_action[i][30],
                    }

                    command = [
                        pred_action[i][32],          # vx
                        pred_action[i][35],          # target yaw
                        pred_action[i][33],          # vy
                        pred_action[i][31] - 0.75,   # d_height (relative to standing 0.75)
                        pred_action[i][30],          # torso yaw
                        pred_action[i][29],          # torso pitch
                        pred_action[i][28],          # torso roll
                        pred_action[i][34],          # turning flag
                    ]
                    self.queue_action(ActionCmd(
                        "eval_move_actuators",
                        target_qpos=target_qpos,
                        action_command=command,
                        waist_qpos=target_waist_qpos,
                    ))

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "eval_move_actuators":
            self._last_cmd_torso_rpyh = np.array([
                action_cmd["action_command"][6],          # torso roll
                action_cmd["action_command"][5],          # torso pitch
                action_cmd["action_command"][4],          # torso yaw
                action_cmd["action_command"][3] + 0.75,   # absolute height
            ], dtype=np.float32)
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    def reset(self):
        super().reset()  # clear queue
        # Flush any unsaved predicted-video latents on the server so the last
        # episode's video is written to disk before we start a new session.
        try:
            requests.post(f"http://{self.server_ip}:{self.server_port}/flush", timeout=30)
        except Exception:
            pass
        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._obs_frame_buffer = []
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75], dtype=np.float32)
