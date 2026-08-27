# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Physics replay agent for full-body WBC teleop: injects each recorded 64-dim
whole-body token into the live SONIC controller over ZMQ (as a protocol-v4
``pose`` ``token_state``, exactly where the controller's ``zmq_manager`` reads
its external token) so the recorded token actually drives real physics.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import zmq

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.agents.lockstep import CommandSnapshot, LockstepAck, LockstepBarrier
from simple.agents.lockstep_unitree_bridge import LockstepUnitreeBridge

# The recorded LeRobot `action` is [token(64) | left_hand(7) | right_hand(7)].
from simple.cli._decoupled_wbc_recording import WBC_TOKEN_DIM


class _TokenInjector:
    """Feeds the recorded 64-dim token into the SONIC controller over ZMQ.

    SIMPLE takes the pico_manager's role: it BINDS the PUB that the controller's
    ``zmq_manager`` SUBs connect to (``command``/``planner``/``pose`` on
    ``tcp://*:PORT``) and drives the recorded token in as a protocol-v4 ``pose``
    ``token_state``.  When an external token is present the controller copies it
    into ``token_state_data_`` and sets ``is_using_encoder_ = false`` (overriding
    its local encoder), so the recorded token — not the controller's idle output
    — drives the policy.

    Command sequence (verified against the controller's zmq_manager /
    zmq_endpoint_interface and the recorder's own pico_manager):
      1. ``engage()``          -> PLANNER-mode start, latches operator_state.start
                                  so WAIT_FOR_CONTROL advances to CONTROL.
      2. ``enter_token_mode()`` -> planner=false -> STREAMED_MOTION -> ZMQ toggle,
                                  the gate that makes GetExternalTokenState()
                                  forward our token.
      3. ``send_token()`` per control step, then ``stop()`` at the end.

    Replaces the old dead DDS path (rt/wbc_token_replay), which nothing subscribed
    to, so the token never reached physics.
    """

    def __init__(self, sonic_config: dict, token_dim: int = WBC_TOKEN_DIM):
        # The controller SUBs connect to tcp://<host>:PORT; we bind the PUB.  Host
        # is fixed to '*' on the bind side (mirrors pico_manager); PICO_PORT picks
        # the port the controller was launched against.
        port = sonic_config.get("PICO_PORT", 5556)
        self._token_dim = token_dim
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(f"tcp://*:{port}")
        time.sleep(0.5)  # slow-joiner warmup before the first send
        # Initial idle command, like the pico_manager's startup.
        self._sock.send(build_command_message(start=False, stop=False, planner=False))
        self._frame = 0
        print(f"[ReplayWbc] token injector bound PUB on tcp://*:{port} (ZMQ pose token_state)")

    def engage(self) -> None:
        """Latch ``operator_state.start`` in PLANNER mode so the controller's state
        machine advances WAIT_FOR_CONTROL -> CONTROL.  MUST precede token mode:
        STREAMED_MOTION never latches start.  Resent a few times (PUB is lossy on
        a fresh SUB join)."""
        for _ in range(5):
            self._sock.send(build_command_message(start=True, stop=False, planner=True))
            time.sleep(0.05)

    def enter_token_mode(self) -> None:
        """planner=false -> STREAMED_MOTION -> ZMQ toggle -> use_zmq_stream=true,
        the gate that makes GetExternalTokenState() forward our token_state."""
        for _ in range(3):
            self._sock.send(build_command_message(start=True, stop=False, planner=False))
            time.sleep(0.02)

    def send_token(
        self,
        token,
        left_hand=None,
        right_hand=None,
        *,
        lockstep: LockstepAck | None = None,
    ) -> None:
        """Emit one protocol-v4 ``pose`` frame carrying the recorded token.

        The controller stores it in ``external_token_state_`` and copies it into
        ``token_state_data_`` next tick, overriding its local encoder.  ``token``
        must be length ``token_dim`` (64); optional 7-DOF hand joints ride along.
        """
        tok = np.asarray(token, dtype=np.float32).reshape(self._token_dim)  # f32 / [64]
        frame_index = lockstep.action_seq if lockstep is not None else self._frame
        data = {
            "token_state": tok,
            # Optional, controller-side logging only.
            "frame_index": np.array([frame_index], dtype=np.int64),
        }
        if left_hand is not None:
            data["left_hand_joints"] = np.asarray(left_hand, dtype=np.float32).reshape(7)
        if right_hand is not None:
            data["right_hand_joints"] = np.asarray(right_hand, dtype=np.float32).reshape(7)
        if lockstep is not None:
            data["lockstep_session"] = np.array([lockstep.session_id], dtype=np.int64)
            data["lockstep_sim_step"] = np.array([lockstep.sim_step_seq], dtype=np.int64)
            data["lockstep_action_seq"] = np.array([lockstep.action_seq], dtype=np.int64)
        self._sock.send(pack_pose_message(data, topic="pose", version=4))
        if lockstep is None:
            self._frame += 1

    def send_lockstep_token(
        self,
        token: np.ndarray,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
        envelope: LockstepAck,
    ) -> None:
        self.send_token(
            token,
            left_hand,
            right_hand,
            lockstep=envelope,
        )

    def stop(self) -> None:
        self._sock.send(build_command_message(start=False, stop=True, planner=True))

    def close(self) -> None:
        self._sock.close(linger=0)


class ReplayWbcAgent:
    """Replays recorded WBC teleop episodes through the live SONIC controller.

    Per control step it injects the recorded token for the current frame over ZMQ
    (via ``_TokenInjector`` — the channel the controller actually reads) and
    forwards the controller's freshest ``rt/lowcmd`` (plus hand cmds) to physics.
    The DDS state heartbeat / proprio caching mirror ``PicoWbcAgent`` so the
    external controller's 500 ms state watchdog never trips across ``env.reset()``.

    Parameters:
        - num_pad_frames: extra frames after the episode data is exhausted during
          which the last recorded token keeps being published (lets the physics
          settle) before ``get_action`` raises ``StopIteration``.
    """

    def __init__(
        self,
        robot: G1Sonic,
        sonic_config: dict,
        num_pad_frames: int = 10,
        *,
        sim_lockstep: bool = False,
        lockstep_event_sink=None,
    ):
        self.robot = robot
        self.sim_dt = sonic_config["SIMULATE_DT"]
        self.sim_lockstep = bool(sim_lockstep)

        # DDS bridge: subscribes rt/lowcmd (+ rt/dex3/*/cmd), publishes rt/lowstate.
        # Created here (after the env's ChannelFactoryInitialize) exactly like
        # PicoWbcAgent._init_unitree_bridge.
        if self.sim_lockstep:
            self.unitree_bridge = LockstepUnitreeBridge(sonic_config)
        else:
            self.unitree_bridge = UnitreeSdk2Bridge(sonic_config)

        # Token transport (ZMQ): SIMPLE takes the pico_manager's role and injects
        # the recorded token where the controller's zmq_manager actually reads it
        # (protocol-v4 `pose` token_state), replacing the dead DDS publisher that
        # nothing subscribed to.
        self._inj = _TokenInjector(sonic_config)

        self.lockstep_barrier: LockstepBarrier | None = None
        if self.sim_lockstep:
            self.lockstep_barrier = LockstepBarrier(
                self.unitree_bridge,
                self._inj.send_lockstep_token,
                physics_dt=self.sim_dt,
                control_dt=4 * self.sim_dt,
                event_sink=lockstep_event_sink,
            )

        # Heartbeat plumbing (proprio is sampled on the sim thread, published off it).
        self._pub_lock = threading.Lock()
        self._last_proprio = None
        self._hb_stop: threading.Event | None = None
        self._hb_thread: threading.Thread | None = None

        # Episode playback state (set via load_episode).
        self._episode_data: pd.DataFrame | None = None
        self._data_row_index = 0
        self._num_pad_frames = num_pad_frames

    # ------------------------------------------------------------------
    # Token transport (ZMQ inject side — see _TokenInjector)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_token_dim(action_vec: np.ndarray) -> None:
        """Bug #5: guard the lower bound so a short recorded action (e.g. a
        non-WBC dataset whose `action` isn't [token(64) | hands(14)]) surfaces a
        clear 'token dim < N' error here instead of a confusing reshape failure
        inside the injector."""
        if action_vec.shape[0] < WBC_TOKEN_DIM:
            raise ValueError(
                f"token dim < {WBC_TOKEN_DIM}: recorded action supplies only "
                f"{action_vec.shape[0]} leading dims, need {WBC_TOKEN_DIM} for the "
                "WBC token. Is this a WBC recording (action = [token(64) | hands(14)])?"
            )

    # ------------------------------------------------------------------
    # Episode playback
    # ------------------------------------------------------------------

    def load_episode(self, episode_data: pd.DataFrame):
        """Load a new episode's recorded frames and rewind to the first frame."""
        self._episode_data = episode_data
        self._data_row_index = 0

    def reset_policy(self):
        """Clear the bridge's command-freshness flags for a new episode.

        The SONIC controller runs externally — we cannot reset it from here; this
        just prevents stale commands from the previous episode from reading as a
        fresh engagement (same rationale as PicoWbcAgent.reset_policy).
        """
        if not self.sim_lockstep:
            self.unitree_bridge.reset()
        self._data_row_index = 0

    def begin_lockstep_session(
        self, proprio: dict, *, session_id: int | None = None
    ) -> CommandSnapshot:
        if self.lockstep_barrier is None:
            raise RuntimeError("lockstep session requested while --sim-lockstep is disabled")
        return self.lockstep_barrier.begin_session(
            proprio,
            mj_time=float(proprio["time"]),
            session_id=session_id,
        )

    def commit_lockstep_idle_boundary(self) -> CommandSnapshot:
        if self.lockstep_barrier is None:
            raise RuntimeError("lockstep boundary requested while disabled")
        return self.lockstep_barrier.commit_boundary(None)

    @property
    def controller_holding(self) -> bool:
        """True when the controller is commanding with real position gains, i.e.
        it — not us — is holding the robot up (copied from PicoWbcAgent)."""
        if not self.unitree_bridge.cmd_received():
            return False
        if self.sim_lockstep:
            lc, _, _ = self.unitree_bridge.get_latest_command_messages()
        else:
            lc = self.unitree_bridge.low_cmd
        return max(lc.motor_cmd[i].kp for i in range(29)) > 5.0

    def get_action(self, observation, **kwargs) -> ActionCmd:
        """Publish the current frame's recorded token and forward the controller's
        freshest LowCmd to physics.

        Data-row advancement mirrors ReplayDecoupledAgent: read the row at
        ``_data_row_index``, then increment.  After the recorded frames run out,
        keep publishing the last token for ``num_pad_frames`` steps, then raise
        ``StopIteration``.
        """
        if self._episode_data is None:
            raise RuntimeError("No episode data loaded. Call load_episode() first.")

        if self._data_row_index >= len(self._episode_data) + self._num_pad_frames:
            raise StopIteration("Episode data exhausted")

        # Hold the last recorded token during the padding window.
        if self._data_row_index >= len(self._episode_data):
            row = self._episode_data.iloc[-1]
        else:
            row = self._episode_data.iloc[self._data_row_index]

        # Recorded action = [token(64) | left_hand(7) | right_hand(7)].  Inject the
        # 64-dim token (+ hands) over ZMQ where the controller's zmq_manager reads
        # it; the controller (asynchronously) decodes it -> rt/lowcmd, which the
        # bridge has cached in `low_cmd`.
        action_vec = np.asarray(row["action"], dtype=np.float64)
        action = self.dispatch_action_vector(action_vec)
        self._data_row_index += 1
        return action

    def dispatch_action_vector(self, action_vec: np.ndarray) -> ActionCmd:
        """Send one whole-body action through the same replay transport.

        Closed-loop evaluators use this method so replay and model-generated
        actions share the exact token split, lockstep barrier, and acknowledged
        command snapshot path.
        """
        action_vec = np.asarray(action_vec)
        if action_vec.ndim != 1:
            raise ValueError(f"whole-body action must be one-dimensional, got {action_vec.shape}")
        if not np.isfinite(action_vec).all():
            raise ValueError("whole-body action contains non-finite values")
        self._check_token_dim(action_vec)
        left_hand = action_vec[WBC_TOKEN_DIM:WBC_TOKEN_DIM + 7] if action_vec.shape[0] >= WBC_TOKEN_DIM + 7 else None
        right_hand = action_vec[WBC_TOKEN_DIM + 7:WBC_TOKEN_DIM + 14] if action_vec.shape[0] >= WBC_TOKEN_DIM + 14 else None
        snapshot = None
        if self.lockstep_barrier is not None:
            snapshot = self.lockstep_barrier.commit_boundary(
                action_vec[:WBC_TOKEN_DIM], left_hand, right_hand
            )
        else:
            self._inj.send_token(action_vec[:WBC_TOKEN_DIM], left_hand, right_hand)

        # Drive physics with the controller's freshest cached rt/lowcmd, ~1 step
        # behind the token just published (do NOT block on the decode round-trip).
        # During playback the controller is engaged (the settle loop waited for
        # it); the robot is no longer held in-process (the CLI settle loop holds
        # the seeded stance kinematically before playback — there is no crane here).
        return self._forward_controller_action(snapshot)

    def _forward_controller_action(
        self, snapshot: CommandSnapshot | None = None
    ) -> ActionCmd:
        """The controller's cached rt/lowcmd (+ hand cmds) as a physics action.

        Hands come from the controller's rt/dex3/*/cmd; with no operator they hold
        default until the controller-side token->hand path exists (see docstring).
        """
        if snapshot is None:
            low_cmd = self.unitree_bridge.low_cmd
            left_hand_cmd = self.unitree_bridge.left_hand_cmd
            right_hand_cmd = self.unitree_bridge.right_hand_cmd
        else:
            low_cmd = snapshot.low_cmd
            left_hand_cmd = snapshot.left_hand_cmd
            right_hand_cmd = snapshot.right_hand_cmd
        return ActionCmd(
            "wbc_torque",
            low_cmd=low_cmd,
            use_sensor=self.unitree_bridge.use_sensor,
            left_hand_cmd=left_hand_cmd,
            right_hand_cmd=right_hand_cmd,
        )

    # ------------------------------------------------------------------
    # State heartbeat (keeps the external controller alive across resets)
    # ------------------------------------------------------------------

    def cache_proprio(self, proprio):
        """Hand the freshest sim proprio to the heartbeat publisher (sim thread)."""
        if self.lockstep_barrier is not None:
            self.lockstep_barrier.cache_substep(
                proprio, mj_time=float(proprio["time"])
            )
            return
        with self._pub_lock:
            self._last_proprio = proprio

    def start_state_heartbeat(self):
        """Publish rt/lowstate from a side thread at ~200 Hz (as PicoWbcAgent does)."""

        if self.sim_lockstep:
            raise RuntimeError("wall-clock state heartbeat is forbidden in lockstep mode")

        def _beat():
            while not self._hb_stop.is_set():
                with self._pub_lock:
                    proprio = self._last_proprio
                if proprio is not None:
                    self.unitree_bridge.PublishLowState(proprio)  # type: ignore
                    if self.unitree_bridge.joystick:  # type: ignore
                        self.unitree_bridge.PublishWirelessController()  # type: ignore
                time.sleep(0.005)

        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(target=_beat, daemon=True)
        self._hb_thread.start()

    def close(self):
        if self._hb_stop is not None:
            self._hb_stop.set()
        # Tell the controller to stop, then tear down the ZMQ PUB.
        try:
            self._inj.stop()
        except Exception:  # noqa: BLE001 — a best-effort stop must never break close
            pass
        self._inj.close()
        if self.lockstep_barrier is not None:
            self.lockstep_barrier.close()
