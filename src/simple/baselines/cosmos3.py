# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
SIMPLE — Cosmos3 policy agent for the MOTION-PLANNING (`eval`) path.

The teleop/WBC path has `cosmos3_decoupled_wbc.py`; this is its non-WBC twin for the MP
`eval` CLI (mirrors `psix.py` : `psix_decoupled_wbc.py`). Same Cosmos server (`/predict`,
CosmosPredictClient), same flat 36-D `g1_simple` action — only the env-side ActionCmd
differs: MP uses `eval_move_actuators` (direct actuators), not `vla_cmd` (WBC goal).

Self-contained (copies CosmosPredictClient + action layout) so it doesn't import the
decoupled-WBC agent's WBC dependencies, which the MP path does not use.
"""

import base64
import io

import numpy as np
import requests
from PIL import Image

from simple.agents.primitive_agent import PrimitiveAgent
from simple.core.action import ActionCmd

DEFAULT_IMAGE_SIZE = 256

# joint order — consistent with scripts/postprocess_psi0.py (same as psix.py / cosmos3_decoupled_wbc.py)
STATE_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]


def from_psi0_upper_joints(psi0_action):
    return np.concatenate([
        psi0_action[14:28],
        psi0_action[0:3],  # left thumb
        psi0_action[5:7],  # left index
        psi0_action[3:5],  # left middle
        psi0_action[7:14],  # right hand
    ])


def _frame_to_png_b64(frame_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_rgb).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CosmosPredictClient:
    """POST a single ego frame + proprio state to the Cosmos server's ``/predict``; returns (T, D)."""

    def __init__(self, server_ip, server_port, domain_name="g1_simple",
                 image_size=DEFAULT_IMAGE_SIZE, view_point="ego_view", timeout=1200):
        self._server = f"http://{server_ip}:{server_port}"
        self._domain_name = domain_name
        self._image_size = image_size
        self._view_point = view_point
        self._timeout = timeout

    def predict(self, frame_rgb, prompt, state, history: dict = {}) -> np.ndarray:
        payload = {
            "image": _frame_to_png_b64(frame_rgb),
            "prompt": prompt,
            "domain_name": self._domain_name,
            "image_size": self._image_size,
            "view_point": self._view_point,
            "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
            **history,
        }
        resp = requests.post(self._server + "/predict", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return np.asarray(resp.json()["action"], dtype=np.float32)  # (T, D)


class Cosmos3Agent(PrimitiveAgent):
    def __init__(self, robot, host: str, port: int, upsample_factor=1,
                 domain_name: str = "g1_simple", image_size: int = DEFAULT_IMAGE_SIZE,
                 client=None, **kwargs):
        super().__init__(robot, **kwargs)
        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor
        self.client = client or CosmosPredictClient(host, port, domain_name=domain_name, image_size=image_size)
        self._global_step_idx = 0
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75])  # FIXME hardcoded for g1 wholebody
        self._reset_history = True

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            frame_rgb = np.asarray(observation["head_stereo_left"], dtype=np.uint8)  # [H,W,3] RGB
            proprio = observation["joint_qpos"][None]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + [self._last_cmd_torso_rpyh[None]],
                axis=1,
            ).astype(np.float32)  # (1, 32)

            history = {"reset": True} if self._reset_history else {}
            self._reset_history = False

            pred_action = self.client.predict(
                frame_rgb, instruction or "bend to pick up the object", states, history
            )  # (T, D)
            print(f"Received {pred_action.shape[0]} actions from server.")
            for i in range(pred_action.shape[0]):
                for _ in range(self.upsample_factor):
                    target_qpos = dict(
                        zip(self.robot.joint_names[15:], from_psi0_upper_joints(pred_action[i][:28]))
                    )
                    target_waist_qpos = {
                        "waist_yaw_joint": pred_action[i][30],
                        "waist_roll_joint": pred_action[i][28],
                        "waist_pitch_joint": pred_action[i][29],
                    }
                    # MP env action layout (matches psix.py): [vx, target_yaw, vy, d_height, torso yaw/pitch/roll, turning]
                    command = [
                        pred_action[i][32],          # vx
                        pred_action[i][35],          # target yaw
                        pred_action[i][33],          # vy
                        pred_action[i][31] - 0.75,   # d_height
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
                action_cmd["action_command"][6],  # torso roll
                action_cmd["action_command"][5],  # torso pitch
                action_cmd["action_command"][4],  # torso yaw
                action_cmd["action_command"][3] + 0.75,
            ], dtype=np.float32)
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    def reset(self):
        super().reset()
        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.75])
