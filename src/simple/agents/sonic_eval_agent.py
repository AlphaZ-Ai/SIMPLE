# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Action-chunk policies connected to the external SONIC lockstep controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import os
import uuid
from time import perf_counter, perf_counter_ns
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import requests

from simple.baselines.client import RequestMessage, ResponseMessage
from simple.agents.replay_wbc_agent import ReplayWbcAgent
from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic


ACTION_DIM = 78
ACTION_CHUNK_SIZE = 30
STATE_DIM = 43


class PolicyError(RuntimeError):
    """A policy request or response violated the evaluator contract."""


class SonicActionPolicy(Protocol):
    """Policy contract used by the SONIC evaluator."""

    name: str
    exec_horizon: int

    def reset(self, episode_data: pd.DataFrame, episode_index: int) -> None: ...

    def infer(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class InferenceRecord:
    episode_index: int
    chunk_index: int
    start_action_index: int
    wall_seconds: float
    shape: tuple[int, int]
    minimum: float
    maximum: float


class ReplayChunkPolicy:
    """Return recorded actions in the same chunks as a served policy."""

    name = "replay"

    def __init__(self, chunk_size: int = ACTION_CHUNK_SIZE, exec_horizon: int | None = None):
        self.chunk_size = chunk_size
        self.exec_horizon = chunk_size if exec_horizon is None else exec_horizon
        if not 1 <= self.exec_horizon <= chunk_size:
            raise PolicyError(f"exec horizon must be within 1..{chunk_size}, got {self.exec_horizon}")
        self._actions: np.ndarray | None = None
        self._cursor = 0

    def reset(self, episode_data: pd.DataFrame, episode_index: int) -> None:
        del episode_index
        actions = np.stack(episode_data["action"].to_numpy()).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise PolicyError(f"recorded actions must have shape [T,{ACTION_DIM}], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise PolicyError("recorded actions contain non-finite values")
        self._actions = actions
        self._cursor = 0

    def infer(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        del image, state, instruction
        if self._actions is None:
            raise PolicyError("replay policy has not been reset")
        if self._cursor >= len(self._actions):
            return np.repeat(self._actions[-1][None], self.chunk_size, axis=0)
        end = min(self._cursor + self.chunk_size, len(self._actions))
        chunk = self._actions[self._cursor:end]
        self._cursor = min(self._cursor + self.exec_horizon, len(self._actions))
        if len(chunk) < self.chunk_size:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1][None], self.chunk_size - len(chunk), axis=0)],
                axis=0,
            )
        return chunk.copy()


class Psi0HttpPolicy:
    """Synchronous client for ``serve_psi0_sonic_http``."""

    name = "psi0"

    def __init__(self, host: str, port: int, *, timeout: float = 60.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        # An RTC server serialises on one client at a time and, absent this id,
        # identifies the client by its TCP source port. 
        self.client_id = f"simple-eval-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.request_count = 0
        self.episode_index = -1
        self.request_records: list[dict[str, Any]] = []
        # reset() sets this per episode; declared here so infer() never reads it unset
        self._reset_pending = True
        try:
            health = self.session.get(f"{self.base_url}/health", timeout=timeout)
            health.raise_for_status()
            info_response = self.session.get(f"{self.base_url}/info", timeout=timeout)
            info_response.raise_for_status()
            self.info = info_response.json()
        except requests.RequestException as exc:
            raise PolicyError(f"Psi0 service preflight failed: {exc}") from exc
        self.exec_horizon = self._validate_info(self.info)

    @staticmethod
    def _validate_info(info: dict[str, Any]) -> int:
        action = info.get("action", {})
        if action.get("action_dim") != ACTION_DIM:
            raise PolicyError(f"server action_dim must be {ACTION_DIM}, got {action}")
        if action.get("action_chunk_size") != ACTION_CHUNK_SIZE:
            raise PolicyError(f"server chunk size must be {ACTION_CHUNK_SIZE}, got {action}")
        exec_horizon = action.get("action_exec_horizon")
        if not isinstance(exec_horizon, int) or not 1 <= exec_horizon <= ACTION_CHUNK_SIZE:
            raise PolicyError(
                f"server execution horizon must be an integer within 1..{ACTION_CHUNK_SIZE}, got {action}"
            )
        # RTC is allowed: the server conditions each chunk on the previous one, which
        # this evaluator makes coherent by executing exactly action_exec_horizon rows
        # per chunk (the shift the server assumes) and by sending history["reset"] on
        # the first request of every episode. rtc_mode=train is refused because its
        # frozen prefix is Tp-Ta wide and only valid on a --model.rtc checkpoint.
        rtc_mode = info.get("rtc_mode", "off")
        if info.get("rtc_enabled") and rtc_mode not in ("test_time", "train"):
            raise PolicyError(f"unsupported server rtc_mode: {rtc_mode!r}")
        if info.get("rtc_enabled") and exec_horizon >= ACTION_CHUNK_SIZE:
            raise PolicyError(
                "an RTC server must run with action_exec_horizon < action_chunk_size; "
                f"got {exec_horizon} of {ACTION_CHUNK_SIZE}"
            )
        image_keys = set(info.get("expected_keys", {}).get("image", {}))
        if image_keys != {"observation.images.egocentric"}:
            raise PolicyError(f"unexpected server image keys: {sorted(image_keys)}")
        return exec_horizon

    def reset(self, episode_data: pd.DataFrame, episode_index: int) -> None:
        del episode_data
        self.episode_index = episode_index
        self.request_records = []
        # An RTC server keeps one previous_action across requests. Without this the
        # first chunk of an episode is conditioned on the last chunk of the previous
        # episode (or of a previous eval run against the same server process).
        self._reset_pending = True

    def infer(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        image_array = np.ascontiguousarray(image, dtype=np.uint8)
        state_array = np.ascontiguousarray(state, dtype=np.float32)
        history: dict[str, Any] = {"client_id": self.client_id}
        if self._reset_pending:
            history["reset"] = True
        self._reset_pending = False
        started = perf_counter()
        record = {
            "request_index": self.request_count,
            "episode": self.episode_index,
            "instruction": instruction,
            "image": {
                "key": "observation.images.egocentric",
                "shape": list(image_array.shape),
                "dtype": str(image_array.dtype),
                "minimum": int(image_array.min()),
                "maximum": int(image_array.max()),
                "channel_mean": image_array.mean(axis=(0, 1)).tolist(),
                "sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
            },
            "state": {
                "shape": list(state_array.shape),
                "dtype": str(state_array.dtype),
                "minimum": float(state_array.min()),
                "maximum": float(state_array.max()),
                "values": state_array.tolist(),
                "sha256": hashlib.sha256(state_array.tobytes()).hexdigest(),
            },
            "fields": {
                "history": dict(history),
                "condition": {},
                "dataset_name": "simple",
                "gt_action": [],
            },
        }
        message = RequestMessage(
            image={"observation.images.egocentric": image_array},
            instruction=instruction,
            history=history,
            state={"states": state_array},
            condition={},
            gt_action=[],
            dataset_name="simple",
            timestamp=str(datetime.now()).replace(" ", "_").replace(":", "-"),
        )
        try:
            response = self.session.post(
                f"{self.base_url}/act",
                json=message.serialize(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            record.update(
                {
                    "wall_seconds": perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.request_records.append(record)
            raise PolicyError(f"Psi0 request failed: {exc}") from exc
        self.request_count += 1
        try:
            decoded = ResponseMessage.deserialize(response.json())
        except Exception as exc:
            record.update(
                {
                    "wall_seconds": perf_counter() - started,
                    "http_status": response.status_code,
                    "error": f"invalid response: {response.text}",
                }
            )
            self.request_records.append(record)
            raise PolicyError(f"invalid Psi0 response: {response.text}") from exc
        action = np.asarray(decoded.action, dtype=np.float32)
        record.update(
            {
                "wall_seconds": perf_counter() - started,
                "http_status": response.status_code,
                "response": {
                    "shape": list(action.shape),
                    "dtype": str(action.dtype),
                    "finite": bool(np.isfinite(action).all()),
                    "minimum": float(action.min()),
                    "maximum": float(action.max()),
                },
            }
        )
        self.request_records.append(record)
        return action


class SonicEvalAgent(ReplayWbcAgent):
    """Consume action chunks and dispatch each row through SONIC lockstep."""

    def __init__(
        self,
        robot: G1Sonic,
        sonic_config: dict[str, Any],
        policy: SonicActionPolicy,
        *,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.policy = policy
        self.exec_horizon = int(getattr(policy, "exec_horizon", ACTION_CHUNK_SIZE))
        if not 1 <= self.exec_horizon <= ACTION_CHUNK_SIZE:
            raise PolicyError(
                f"policy execution horizon must be within 1..{ACTION_CHUNK_SIZE}, got {self.exec_horizon}"
            )
        self._eval_event_sink = event_sink
        self._queue: deque[np.ndarray] = deque()
        self._episode_index = -1
        self._executed_actions = 0
        self._chunk_index = 0
        self.inference_records: list[InferenceRecord] = []
        self.action_chunks: list[np.ndarray] = []
        super().__init__(
            robot,
            sonic_config,
            num_pad_frames=0,
            sim_lockstep=True,
            lockstep_event_sink=event_sink,
        )

    def _emit(self, kind: str, **values: Any) -> None:
        if self._eval_event_sink is not None:
            self._eval_event_sink({"kind": kind, "perf_ns": perf_counter_ns(), **values})

    def reset_evaluation(self, episode_data: pd.DataFrame, episode_index: int) -> None:
        self.policy.reset(episode_data, episode_index)
        self._queue.clear()
        self._episode_index = episode_index
        self._executed_actions = 0
        self._chunk_index = 0
        self.inference_records = []
        self.action_chunks = []

    def get_evaluation_action(
        self,
        observation: dict[str, Any],
        *,
        instruction: str,
    ) -> ActionCmd:
        if not self._queue:
            image = np.asarray(observation["head_stereo_left"])
            state = np.asarray(observation["joint_qpos"], dtype=np.float32)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise PolicyError(f"policy image must be HxWx3 uint8, got {image.shape} {image.dtype}")
            if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
                raise PolicyError(f"policy state must be finite [{STATE_DIM}], got {state.shape}")
            self._emit(
                "inference_start",
                episode=self._episode_index,
                chunk=self._chunk_index,
                start_action=self._executed_actions,
            )
            started = perf_counter()
            chunk = np.asarray(
                self.policy.infer(image=image, state=state, instruction=instruction),
                dtype=np.float32,
            )
            elapsed = perf_counter() - started
            # The psi servers return only the rows they expect to be executed
            # (action_exec_horizon), not the full predicted chunk; a replay policy
            # returns the whole chunk. Accept anything from exec_horizon to the full
            # chunk and slice below.
            if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not (
                self.exec_horizon <= chunk.shape[0] <= ACTION_CHUNK_SIZE
            ):
                raise PolicyError(
                    f"policy action must have shape [{self.exec_horizon}..{ACTION_CHUNK_SIZE},"
                    f"{ACTION_DIM}], got {chunk.shape}"
                )
            if not np.isfinite(chunk).all():
                raise PolicyError("policy action contains non-finite values")
            self._emit(
                "inference_end",
                episode=self._episode_index,
                chunk=self._chunk_index,
                start_action=self._executed_actions,
                wall_seconds=elapsed,
            )
            self.inference_records.append(
                InferenceRecord(
                    episode_index=self._episode_index,
                    chunk_index=self._chunk_index,
                    start_action_index=self._executed_actions,
                    wall_seconds=elapsed,
                    shape=chunk.shape,
                    minimum=float(chunk.min()),
                    maximum=float(chunk.max()),
                )
            )
            self.action_chunks.append(chunk.copy())
            self._queue.extend(chunk[: self.exec_horizon])
            self._chunk_index += 1

        action = self._queue.popleft()
        result = self.dispatch_action_vector(action)
        self._executed_actions += 1
        return result

    @property
    def queued_actions(self) -> int:
        return len(self._queue)

    @property
    def executed_actions(self) -> int:
        return self._executed_actions
