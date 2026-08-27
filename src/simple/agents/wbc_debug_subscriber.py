# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Subscriber for the SONIC controller's per-tick ZMQ debug state ("g1_debug",
PUB port 5557) plus a faithful Python port of the controller's LowCmd
constants, so one msgpack packet yields BOTH the 64-dim whole-body token and
the full low-level command it belongs to.

Why this exists: the controller's DDS messages (rt/lowcmd, interim
rt/wbc_token) arrive through independent callbacks, so a reader pairing
"latest token" with "latest low_cmd" can straddle a tick boundary.  The
g1_debug packet is assembled by the controller from a single state-logger
entry, so its fields are consistent by construction.

Pairing convention (same as the real-robot recording pipeline): within one
packet, ``token_state`` is the token computed THIS tick, while
``last_action`` / ``last_*_hand_action`` were snapshotted at the start of the
tick and therefore hold the command computed the PREVIOUS tick (20 ms
earlier).  Real-robot datasets recorded via g1_data_server.py carry exactly
this offset, so consuming the packet as-is keeps sim data consistent with
real data.

LowCmd reconstruction: g1_deploy_onnx_ref.cpp:CreatePolicyCommand builds the
DDS command as

    q_target[i]  = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]
    tau_ff[i]    = 0.0
    dq_target[i] = 0.0
    kp[i]        = kps[i]      # constants from policy_parameters.hpp
    kd[i]        = kds[i]

and the packet's ``last_action`` is packed with the exact same q_target
formula (zmq_output_handler.hpp), so ``last_action[i] == motor_cmd[i].q``.
The remaining fields are constants, ported below from policy_parameters.hpp.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Constants ported from gear_sonic_deploy .../include/policy_parameters.hpp
# (all 29-dim arrays are in MuJoCo/hardware motor order, i.e. LowCmd index)
# ---------------------------------------------------------------------------

# Motor armature constants (used for PID gain computation)
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

# Control parameters for PID gain computation (10 Hz, zeta = 2.0; the C++
# uses this pi literal, kept verbatim for bit-parity with the controller)
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0

# stiffness = armature * natural_freq^2
_STIFF_5020 = _ARMATURE_5020 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFF_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFF_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFF_4010 = _ARMATURE_4010 * _NATURAL_FREQ * _NATURAL_FREQ

# damping = 2.0 * damping_ratio * armature * natural_freq
_DAMP_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMP_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMP_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMP_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

# Effort limits (used for action scale computation)
_EFFORT_5020 = 25.0
_EFFORT_7520_14 = 88.0
_EFFORT_7520_22 = 139.0
_EFFORT_4010 = 5.0

# Per-joint motor type, MuJoCo/hardware order: 6 left leg, 6 right leg,
# 3 waist, 7 left arm, 7 right arm.  "2x" marks the doubled-gain 5020 joints
# (ankles + waist roll/pitch).
_MOTOR_TYPES = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020_2x", "5020_2x",  # left leg
    "7520_22", "7520_22", "7520_14", "7520_22", "5020_2x", "5020_2x",  # right leg
    "7520_14", "5020_2x", "5020_2x",                                   # waist y/r/p
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",            # left arm
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",            # right arm
]

_KP_BY_TYPE = {
    "5020": _STIFF_5020, "5020_2x": 2.0 * _STIFF_5020,
    "7520_14": _STIFF_7520_14, "7520_22": _STIFF_7520_22, "4010": _STIFF_4010,
}
_KD_BY_TYPE = {
    "5020": _DAMP_5020, "5020_2x": 2.0 * _DAMP_5020,
    "7520_14": _DAMP_7520_14, "7520_22": _DAMP_7520_22, "4010": _DAMP_4010,
}
# action_scale = 0.25 * effort_limit / stiffness.  NOTE: the C++ uses the
# plain (not doubled) stiffness here even for the "2x"-gain joints.
_SCALE_BY_TYPE = {
    "5020": 0.25 * _EFFORT_5020 / _STIFF_5020,
    "5020_2x": 0.25 * _EFFORT_5020 / _STIFF_5020,
    "7520_14": 0.25 * _EFFORT_7520_14 / _STIFF_7520_14,
    "7520_22": 0.25 * _EFFORT_7520_22 / _STIFF_7520_22,
    "4010": 0.25 * _EFFORT_4010 / _STIFF_4010,
}

KPS = np.array([_KP_BY_TYPE[t] for t in _MOTOR_TYPES], dtype=np.float64)
KDS = np.array([_KD_BY_TYPE[t] for t in _MOTOR_TYPES], dtype=np.float64)
G1_ACTION_SCALE = np.array([_SCALE_BY_TYPE[t] for t in _MOTOR_TYPES], dtype=np.float64)

# Default joint angles (standing pose), MuJoCo/hardware order
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0,       # left arm
    0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0,       # right arm
], dtype=np.float64)

# MuJoCo order in IsaacLab index (isaaclab_to_mujoco in policy_parameters.hpp)
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)

WBC_TOKEN_DIM = 64
NUM_BODY_MOTORS = 29


def raw_action_from_q_target(q_target: np.ndarray) -> np.ndarray:
    """Invert the controller's scaling: recover the raw IsaacLab-order policy
    output from a MuJoCo-order q_target (= packet last_action)."""
    raw = np.empty(NUM_BODY_MOTORS, dtype=np.float64)
    raw[ISAACLAB_TO_MUJOCO] = (q_target - DEFAULT_ANGLES) / G1_ACTION_SCALE
    return raw


class WbcDebugSubscriber:
    """Background-thread SUB to the SONIC controller's g1_debug publisher.

    ``get_synced()`` returns the newest packet as one consistent snapshot:
    the 64-dim token plus the reconstructed LowCmd arrays (q_target /
    dq_target / tau_ff / kp / kd) and the Dex3 hand targets (motor order,
    same indexing as HandCmd_.motor_cmd).
    """

    def __init__(self, host: str = "localhost", port: int = 5557, topic: str = "g1_debug"):
        import msgpack  # deferred: only this consumer needs it
        self._msgpack = msgpack

        self._topic_bytes = topic.encode()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)  # always keep only the newest
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_t: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[WbcDebug] subscribed to tcp://{host}:{port} topic={topic}")

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                payload = raw[len(self._topic_bytes):]
                data = self._msgpack.unpackb(payload, raw=False)
                snapshot = self._build_snapshot(data)
            except Exception as e:
                print(f"[WbcDebug] decode error: {e}")
                continue
            if snapshot is not None:
                with self._lock:
                    self._latest = snapshot
                    self._latest_t = time.monotonic()

    @staticmethod
    def _build_snapshot(data: dict) -> dict | None:
        q_target = np.asarray(data.get("last_action", []), dtype=np.float64)
        if q_target.shape != (NUM_BODY_MOTORS,):
            return None  # controller not in CONTROL yet / malformed packet
        token = np.asarray(data.get("token_state", []), dtype=np.float64)
        return {
            "index": int(data.get("index", -1)),
            "token": token if token.shape == (WBC_TOKEN_DIM,) else None,
            # Full LowCmd_ reconstruction (MuJoCo/hardware motor order):
            "q_target": q_target,
            "dq_target": np.zeros(NUM_BODY_MOTORS, dtype=np.float64),
            "tau_ff": np.zeros(NUM_BODY_MOTORS, dtype=np.float64),
            "kp": KPS.copy(),
            "kd": KDS.copy(),
            # Dex3 targets, HandCmd_ motor order (setAllJointsCommand is
            # an identity index mapping in dex3_hands.hpp):
            "left_hand_action": np.asarray(data.get("last_left_hand_action", np.zeros(7)), dtype=np.float64),
            "right_hand_action": np.asarray(data.get("last_right_hand_action", np.zeros(7)), dtype=np.float64),
        }

    def get_synced(self) -> dict | None:
        """Newest packet as a fresh copy, or None before the first packet."""
        with self._lock:
            if self._latest is None:
                return None
            return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in self._latest.items()}

    @property
    def age_s(self) -> float:
        """Seconds since the last packet (inf before any arrives)."""
        with self._lock:
            t = self._latest_t
        return float("inf") if t is None else time.monotonic() - t

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()
        self._ctx.term()
