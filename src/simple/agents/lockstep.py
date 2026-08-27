# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Simulation-clock lockstep primitives for the external SONIC controller.

The protocol is intentionally small: Python owns session and sequence numbers,
publishes a frozen boundary state plus its token, and waits for an identical
acknowledgement on body, left-hand, and right-hand command topics.
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


LOCKSTEP_MAGIC = 0x534C4B31
NO_ACTION_SEQ = 0xFFFFFFFF
DEFAULT_ACK_TIMEOUT = 2.0
DEFAULT_RETRY_PERIOD = 0.050


@dataclass(frozen=True, slots=True)
class LockstepAck:
    magic: int
    session_id: int
    sim_step_seq: int
    action_seq: int

    def __post_init__(self) -> None:
        values = (self.magic, self.session_id, self.sim_step_seq, self.action_seq)
        if any(not 0 <= int(value) <= 0xFFFFFFFF for value in values):
            raise ValueError(f"lockstep acknowledgement is outside uint32: {values}")

    @property
    def reserve(self) -> tuple[int, int, int, int]:
        return (self.magic, self.session_id, self.sim_step_seq, self.action_seq)


@dataclass(frozen=True, slots=True)
class CommandSnapshot:
    ack: LockstepAck
    low_cmd: Any
    left_hand_cmd: Any
    right_hand_cmd: Any
    arrival_ns: tuple[int, int, int]


class LockstepBridge(Protocol):
    def publish_low_state(self, proprio: dict[str, Any], envelope: LockstepAck) -> None: ...

    def clear_ack_state(self) -> None: ...

    def wait_for_ack(self, expected: LockstepAck, timeout: float) -> CommandSnapshot | None: ...

    def latest_ack_diagnostics(self) -> dict[str, Any]: ...


class LockstepTimeout(RuntimeError):
    """Raised after a frozen boundary receives no complete three-topic ack."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"SONIC lockstep acknowledgement timed out: {diagnostics}")


class LockstepProtocolError(RuntimeError):
    """Raised before publishing a request that violates the fixed protocol."""


TokenSender = Callable[[np.ndarray, np.ndarray | None, np.ndarray | None, LockstepAck], None]


class LockstepBarrier:
    """Owns one Python-side lockstep session.

    ``begin_session`` performs the step-zero capability handshake. The seeded
    state is then ready for the first ``commit_boundary`` (step 1). After each
    command, exactly ``substeps_per_control`` calls to ``cache_substep`` are
    required before the next boundary can be committed.
    """

    def __init__(
        self,
        bridge: LockstepBridge,
        send_token: TokenSender,
        *,
        physics_dt: float,
        control_dt: float = 0.02,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        retry_period: float = DEFAULT_RETRY_PERIOD,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        ratio = control_dt / physics_dt
        rounded = round(ratio)
        if not math.isfinite(ratio) or rounded <= 0 or abs(ratio - rounded) > 1e-12:
            raise ValueError(
                f"control_dt/physics_dt must be a positive integer, got {control_dt}/{physics_dt}={ratio}"
            )
        if rounded != 4:
            raise ValueError(f"current SONIC lockstep requires exactly four substeps, got {rounded}")
        if ack_timeout <= 0 or retry_period <= 0 or retry_period >= ack_timeout:
            raise ValueError("retry period and acknowledgement timeout are inconsistent")

        self.bridge = bridge
        self._send_token = send_token
        self.physics_dt = float(physics_dt)
        self.control_dt = float(control_dt)
        self.substeps_per_control = int(rounded)
        self.ack_timeout = float(ack_timeout)
        self.retry_period = float(retry_period)
        self._event_sink = event_sink

        self.state = "IDLE"
        self.session_id: int | None = None
        self.sim_step_seq = 0
        self.action_seq = -1
        self._substeps_since_boundary = 0
        self._last_sim_time: float | None = None
        self._frozen_proprio: dict[str, Any] | None = None
        self._last_snapshot: CommandSnapshot | None = None

    def _emit(self, kind: str, **values: Any) -> None:
        if self._event_sink is not None:
            self._event_sink({"kind": kind, "wall_ns": time.monotonic_ns(), **values})

    @staticmethod
    def _snapshot_payload(snapshot: CommandSnapshot) -> dict[str, Any]:
        body = snapshot.low_cmd
        left = snapshot.left_hand_cmd
        right = snapshot.right_hand_cmd
        return {
            "body_q": [float(body.motor_cmd[i].q) for i in range(29)],
            "body_dq": [float(body.motor_cmd[i].dq) for i in range(29)],
            "body_kp": [float(body.motor_cmd[i].kp) for i in range(29)],
            "body_kd": [float(body.motor_cmd[i].kd) for i in range(29)],
            "body_tau": [float(body.motor_cmd[i].tau) for i in range(29)],
            "left_q": [float(left.motor_cmd[i].q) for i in range(7)],
            "right_q": [float(right.motor_cmd[i].q) for i in range(7)],
            "arrival_ns": list(snapshot.arrival_ns),
        }

    @staticmethod
    def _new_session(previous: int | None) -> int:
        while True:
            value = secrets.randbits(32)
            if value != 0 and value != previous:
                return value

    @staticmethod
    def _copy_proprio(proprio: dict[str, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for key, value in proprio.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _validate_proprio(self, proprio: dict[str, Any], mj_time: float) -> float:
        if "time" not in proprio:
            raise LockstepProtocolError("proprio is missing simulator time")
        observed = float(proprio["time"])
        expected = float(mj_time)
        if not math.isfinite(observed) or not math.isfinite(expected):
            raise LockstepProtocolError(f"non-finite simulator time: obs={observed}, mj={expected}")
        if observed != expected:
            raise LockstepProtocolError(
                f"observation time is not the exact mjData.time: obs={observed!r}, mj={expected!r}"
            )
        if self._last_sim_time is not None and observed < self._last_sim_time:
            raise LockstepProtocolError(
                f"simulator time moved backwards within session: {self._last_sim_time!r} -> {observed!r}"
            )
        self._last_sim_time = observed
        return observed

    def begin_session(
        self,
        proprio: dict[str, Any],
        *,
        mj_time: float,
        session_id: int | None = None,
    ) -> CommandSnapshot:
        if self.state not in {"IDLE", "ACTIVE", "FAILED"}:
            raise LockstepProtocolError(f"cannot begin session while state={self.state}")
        previous = self.session_id
        chosen = self._new_session(previous) if session_id is None else int(session_id)
        if chosen == 0 or not 0 < chosen <= 0xFFFFFFFF or chosen == previous:
            raise LockstepProtocolError(f"invalid or reused session id: {chosen}")

        self.session_id = chosen
        self.sim_step_seq = 0
        self.action_seq = -1
        self._substeps_since_boundary = 0
        self._last_sim_time = None
        self._validate_proprio(proprio, mj_time)
        self._frozen_proprio = self._copy_proprio(proprio)
        self._last_snapshot = None
        self.bridge.clear_ack_state()
        self.state = "SESSION_HANDSHAKE"

        expected = LockstepAck(LOCKSTEP_MAGIC, chosen, 0, NO_ACTION_SEQ)

        def publish() -> None:
            assert self._frozen_proprio is not None
            self.bridge.publish_low_state(self._frozen_proprio, expected)

        snapshot = self._publish_and_wait(expected, publish, phase="session_ready")
        self.state = "ACTIVE"
        self._last_snapshot = snapshot
        self._emit("session_active", session=chosen)
        return snapshot

    def cache_substep(self, proprio: dict[str, Any], *, mj_time: float) -> None:
        if self.state != "ACTIVE" or self.session_id is None:
            raise LockstepProtocolError("substep arrived without an active session")
        self._validate_proprio(proprio, mj_time)
        if self._substeps_since_boundary >= self.substeps_per_control:
            raise LockstepProtocolError("simulator advanced past an uncommitted control boundary")

        last_action = self.action_seq if self.action_seq >= 0 else NO_ACTION_SEQ
        duplicate = LockstepAck(
            LOCKSTEP_MAGIC, self.session_id, self.sim_step_seq, last_action
        )
        self.bridge.publish_low_state(proprio, duplicate)
        self._substeps_since_boundary += 1
        self._emit(
            "substep",
            session=self.session_id,
            step=self.sim_step_seq,
            action=last_action,
            ordinal=self._substeps_since_boundary,
            sim_time=float(mj_time),
        )
        if self._substeps_since_boundary == self.substeps_per_control:
            self._frozen_proprio = self._copy_proprio(proprio)
            self._emit(
                "boundary_frozen",
                session=self.session_id,
                next_step=self.sim_step_seq + 1,
                sim_time=float(mj_time),
            )

    def commit_boundary(
        self,
        token: np.ndarray | None,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> CommandSnapshot:
        if self.state != "ACTIVE" or self.session_id is None:
            raise LockstepProtocolError("cannot commit without an active session")
        if self._frozen_proprio is None:
            raise LockstepProtocolError("no frozen boundary state is available")
        if self.sim_step_seq > 0 and self._substeps_since_boundary != self.substeps_per_control:
            raise LockstepProtocolError(
                f"next boundary requires {self.substeps_per_control} substeps, got {self._substeps_since_boundary}"
            )

        next_step = self.sim_step_seq + 1
        if next_step >= NO_ACTION_SEQ:
            raise LockstepProtocolError("sim step sequence exhausted uint32 protocol space")

        if token is None:
            next_action = NO_ACTION_SEQ
            token_array = None
        else:
            next_action = self.action_seq + 1
            if next_action >= NO_ACTION_SEQ:
                raise LockstepProtocolError("action sequence exhausted uint32 protocol space")
            token_array = np.asarray(token, dtype=np.float32).reshape(64)
        expected = LockstepAck(LOCKSTEP_MAGIC, self.session_id, next_step, next_action)
        frozen = self._frozen_proprio

        def publish() -> None:
            if token_array is not None:
                self._send_token(token_array, left_hand, right_hand, expected)
            self.bridge.publish_low_state(frozen, expected)

        snapshot = self._publish_and_wait(expected, publish, phase="control")
        self.sim_step_seq = next_step
        if token_array is not None:
            self.action_seq = next_action
        self._substeps_since_boundary = 0
        self._frozen_proprio = None
        self._last_snapshot = snapshot
        return snapshot

    def _publish_and_wait(
        self,
        expected: LockstepAck,
        publish: Callable[[], None],
        *,
        phase: str,
    ) -> CommandSnapshot:
        started = time.monotonic()
        retries = 0
        publish()
        self._emit("request_sent", phase=phase, ack=expected.reserve, retry=0)
        while True:
            elapsed = time.monotonic() - started
            remaining = self.ack_timeout - elapsed
            if remaining <= 0:
                self.state = "FAILED"
                diagnostics = {
                    "phase": phase,
                    "expected": expected.reserve,
                    "waited_seconds": elapsed,
                    "retry_count": retries,
                    "latest_acks": self.bridge.latest_ack_diagnostics(),
                    "last_sim_time": self._last_sim_time,
                }
                self._emit("timeout", **diagnostics)
                raise LockstepTimeout(diagnostics)
            snapshot = self.bridge.wait_for_ack(
                expected, timeout=min(self.retry_period, remaining)
            )
            if snapshot is not None:
                if self._event_sink is not None:
                    self._emit(
                        "ack_complete",
                        phase=phase,
                        ack=expected.reserve,
                        retry_count=retries,
                        latency_seconds=time.monotonic() - started,
                        command=self._snapshot_payload(snapshot),
                    )
                return snapshot
            retries += 1
            publish()
            self._emit("request_sent", phase=phase, ack=expected.reserve, retry=retries)

    @property
    def last_snapshot(self) -> CommandSnapshot | None:
        return self._last_snapshot

    def close(self) -> None:
        self.state = "CLOSED"
