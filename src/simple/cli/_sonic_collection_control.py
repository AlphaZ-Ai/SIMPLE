# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Private control channel for deterministic hardware-free SONIC collection.
"""

from __future__ import annotations

import json
import os
import select
import socket
import time
from typing import Any


CONTROL_FD_ENV = "SIMPLE_SONIC_COLLECTION_FD"
_MAX_MESSAGE_BYTES = 64 * 1024


class JsonLineChannel:
    """Small, local, newline-delimited JSON channel over a stream socket."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buffer = bytearray()

    @property
    def fileno(self) -> int:
        return self._sock.fileno()

    def send(self, event: str, **fields: Any) -> None:
        if not event:
            raise ValueError("control event must not be empty")
        message = {"event": event, **fields}
        encoded = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("control message exceeds 64 KiB")
        self._sock.sendall(encoded + b"\n")

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("timed out waiting for collection control message")
            readable, _, _ = select.select([self._sock], [], [], remaining)
            if not readable:
                raise TimeoutError("timed out waiting for collection control message")
            chunk = self._sock.recv(4096)
            if not chunk:
                raise EOFError("collection control channel closed")
            self._buffer.extend(chunk)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise ValueError("collection control message exceeds 64 KiB")

        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        message = json.loads(line)
        if not isinstance(message, dict) or not isinstance(message.get("event"), str):
            raise ValueError("collection control message needs a string 'event'")
        return message

    def expect(self, event: str, timeout: float | None = None) -> dict[str, Any]:
        message = self.receive(timeout)
        if message["event"] != event:
            raise RuntimeError(
                f"expected collection event {event!r}, got {message['event']!r}"
            )
        return message

    def close(self) -> None:
        self._sock.close()


def inherited_collection_channel() -> JsonLineChannel | None:
    """Return the opt-in inherited channel, leaving manual teleop untouched."""

    raw_fd = os.environ.get(CONTROL_FD_ENV)
    if raw_fd is None:
        return None
    try:
        fd = int(raw_fd)
    except ValueError as exc:
        raise RuntimeError(
            f"{CONTROL_FD_ENV} must be an integer file descriptor"
        ) from exc
    if fd < 0:
        raise RuntimeError(f"{CONTROL_FD_ENV} must be non-negative")
    sock = socket.socket(fileno=fd)
    os.set_inheritable(fd, False)
    return JsonLineChannel(sock)
