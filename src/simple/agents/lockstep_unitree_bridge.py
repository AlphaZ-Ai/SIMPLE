# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
SIMPLE-owned atomic command bridge for SONIC simulation lockstep.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from unitree_sdk2py.utils.crc import CRC

from simple.agents.lockstep import CommandSnapshot, LockstepAck


class _LowStatePublisherProxy:
    def __init__(self, owner: "LockstepUnitreeBridge", publisher: Any) -> None:
        self._owner = owner
        self._publisher = publisher

    def Write(self, message: Any) -> Any:  # noqa: N802 - Unitree SDK API
        envelope = self._owner._publishing_envelope
        if envelope is None:
            raise RuntimeError("lockstep LowState write has no envelope")
        message.reserve[:] = envelope.reserve
        message.crc = self._owner._crc.Crc(message)
        return self._publisher.Write(message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._publisher, name)


class LockstepUnitreeBridge(UnitreeSdk2Bridge):
    """Wrap the upstream bridge without modifying ``third_party/gear_sonic``."""

    def __init__(self, config: dict[str, Any]):
        self._command_lock = threading.RLock()
        self._command_condition = threading.Condition(self._command_lock)
        self._publish_lock = threading.RLock()
        self._publishing_envelope: LockstepAck | None = None
        self._arrival_ns = {"body": 0, "left": 0, "right": 0}
        self._crc = CRC()
        super().__init__(config)
        self.low_state_puber = _LowStatePublisherProxy(self, self.low_state_puber)

    @staticmethod
    def _ack_from_message(message: Any) -> LockstepAck | None:
        reserve = tuple(int(value) for value in message.reserve)
        if len(reserve) != 4:
            return None
        try:
            return LockstepAck(*reserve)
        except ValueError:
            return None

    def reset(self) -> None:
        with self._command_condition:
            self.low_cmd_received = False
            self.left_hand_cmd_received = False
            self.right_hand_cmd_received = False
            self.new_low_cmd = False
            self.new_left_hand_cmd = False
            self.new_right_hand_cmd = False
            self._arrival_ns = {"body": 0, "left": 0, "right": 0}
            self._command_condition.notify_all()

    def clear_ack_state(self) -> None:
        self.reset()

    def LowCmdHandler(self, msg: Any) -> None:  # noqa: N802 - Unitree SDK callback
        with self._command_condition:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True
            self._arrival_ns["body"] = time.monotonic_ns()
            self._command_condition.notify_all()

    def LeftHandCmdHandler(self, msg: Any) -> None:  # noqa: N802
        with self._command_condition:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True
            self._arrival_ns["left"] = time.monotonic_ns()
            self._command_condition.notify_all()

    def RightHandCmdHandler(self, msg: Any) -> None:  # noqa: N802
        with self._command_condition:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True
            self._arrival_ns["right"] = time.monotonic_ns()
            self._command_condition.notify_all()

    def cmd_received(self) -> bool:
        with self._command_lock:
            return bool(
                self.low_cmd_received
                or self.left_hand_cmd_received
                or self.right_hand_cmd_received
            )

    def publish_low_state(self, proprio: dict[str, Any], envelope: LockstepAck) -> None:
        with self._publish_lock:
            self._publishing_envelope = envelope
            try:
                super().PublishLowState(proprio)
            finally:
                self._publishing_envelope = None

    def _matching_snapshot(self, expected: LockstepAck) -> CommandSnapshot | None:
        if not (
            self.low_cmd_received
            and self.left_hand_cmd_received
            and self.right_hand_cmd_received
        ):
            return None
        acks = (
            self._ack_from_message(self.low_cmd),
            self._ack_from_message(self.left_hand_cmd),
            self._ack_from_message(self.right_hand_cmd),
        )
        if any(ack != expected for ack in acks):
            return None
        return CommandSnapshot(
            ack=expected,
            low_cmd=copy.deepcopy(self.low_cmd),
            left_hand_cmd=copy.deepcopy(self.left_hand_cmd),
            right_hand_cmd=copy.deepcopy(self.right_hand_cmd),
            arrival_ns=(
                self._arrival_ns["body"],
                self._arrival_ns["left"],
                self._arrival_ns["right"],
            ),
        )

    def wait_for_ack(
        self, expected: LockstepAck, timeout: float
    ) -> CommandSnapshot | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._command_condition:
            while True:
                snapshot = self._matching_snapshot(expected)
                if snapshot is not None:
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._command_condition.wait(remaining)

    def latest_ack_diagnostics(self) -> dict[str, Any]:
        with self._command_lock:
            now = time.monotonic_ns()

            def item(received: bool, message: Any, key: str) -> dict[str, Any]:
                ack = self._ack_from_message(message) if received else None
                arrival = self._arrival_ns[key]
                return {
                    "ack": ack.reserve if ack is not None else None,
                    "age_ms": (now - arrival) / 1e6 if arrival else None,
                }

            return {
                "body": item(self.low_cmd_received, self.low_cmd, "body"),
                "left": item(self.left_hand_cmd_received, self.left_hand_cmd, "left"),
                "right": item(self.right_hand_cmd_received, self.right_hand_cmd, "right"),
            }

    def get_latest_command_messages(self) -> tuple[Any, Any, Any]:
        with self._command_lock:
            return (
                copy.deepcopy(self.low_cmd),
                copy.deepcopy(self.left_hand_cmd),
                copy.deepcopy(self.right_hand_cmd),
            )
