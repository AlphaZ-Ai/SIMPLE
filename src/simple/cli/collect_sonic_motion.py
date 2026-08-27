# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Deterministically record a SONIC reference motion without VR hardware.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import zmq

from simple.cli._sonic_collection_control import CONTROL_FD_ENV, JsonLineChannel


DEFAULT_ENV_ID = "simple/G1WholebodyXMoveBendCarryBoxSonic-v0"
DEFAULT_FPS = 50.0
HEADER_SIZE = 1280
REQUIRED_PORTS = (5556, 5557, 13579)
FIRST_FRAME_SETTLE_S = 0.2
FIRST_FRAME_INDEX_JUMP = 1000


class CollectionError(RuntimeError):
    """A collection precondition or supervised child process failed."""


def _load_csv(path: Path, width: int) -> np.ndarray:
    if not path.is_file():
        raise CollectionError(f"missing motion file: {path}")
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise CollectionError(f"failed to read motion file {path}: {exc}") from exc
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != width:
        raise CollectionError(
            f"{path}: expected a two-dimensional array with {width} columns, "
            f"got {data.shape}"
        )
    if len(data) == 0:
        raise CollectionError(f"{path}: motion contains no frames")
    if not np.isfinite(data).all():
        raise CollectionError(f"{path}: motion contains non-finite values")
    return data


@dataclass(frozen=True)
class MotionData:
    source_dir: Path
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_quat: np.ndarray

    @classmethod
    def load(cls, source_dir: Path) -> "MotionData":
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.is_dir():
            raise CollectionError(f"motion directory does not exist: {source_dir}")
        joint_pos = _load_csv(source_dir / "joint_pos.csv", 29)
        joint_vel = _load_csv(source_dir / "joint_vel.csv", 29)
        body_quat_flat = _load_csv(source_dir / "body_quat.csv", 56)
        lengths = {len(joint_pos), len(joint_vel), len(body_quat_flat)}
        if len(lengths) != 1:
            raise CollectionError(
                "motion row count mismatch: "
                f"joint_pos={len(joint_pos)}, joint_vel={len(joint_vel)}, "
                f"body_quat={len(body_quat_flat)}"
            )
        return cls(
            source_dir=source_dir,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_quat=body_quat_flat.reshape(-1, 14, 4),
        )

    @property
    def frame_count(self) -> int:
        return len(self.joint_pos)


def synthetic_hand_waveform(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic 14D validation overlay used by the old probe."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    rows = np.arange(frame_count, dtype=np.float32)
    denominator = max(frame_count - 1, 1)
    phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * rows / denominator))
    left_scale = np.array([0.05, 0.10, 0.15, 0.20, 0.12, 0.08, 0.04], dtype=np.float32)
    right_scale = np.array([0.04, 0.08, 0.12, 0.16, 0.20, 0.10, 0.05], dtype=np.float32)
    return phase[:, None] * left_scale, (1.0 - phase[:, None]) * right_scale


def _pack_message(
    topic: str, version: int, fields: list[tuple[str, str, np.ndarray]]
) -> bytes:
    header_fields: list[dict[str, Any]] = []
    payload = bytearray()
    for name, dtype, value in fields:
        array = np.ascontiguousarray(value)
        header_fields.append({"name": name, "dtype": dtype, "shape": list(array.shape)})
        payload.extend(array.tobytes(order="C"))
    count = int(fields[0][2].shape[0]) if fields and fields[0][2].ndim else 1
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": header_fields,
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if len(encoded) > HEADER_SIZE:
        raise ValueError(
            f"protocol header is {len(encoded)} bytes; limit is {HEADER_SIZE}"
        )
    return (
        topic.encode("ascii") + encoded + b"\0" * (HEADER_SIZE - len(encoded)) + payload
    )


def build_command_message(*, start: bool, stop: bool) -> bytes:
    fields = [
        ("start", "u8", np.array([start], dtype=np.uint8)),
        ("stop", "u8", np.array([stop], dtype=np.uint8)),
        ("planner", "u8", np.array([0], dtype=np.uint8)),
    ]
    return _pack_message("command", 1, fields)


def build_pose_message(
    motion: MotionData,
    row: int,
    frame_index: int,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    catch_up: bool = False,
) -> bytes:
    fields = [
        ("joint_pos", "f32", motion.joint_pos[row].reshape(1, 29)),
        ("joint_vel", "f32", motion.joint_vel[row].reshape(1, 29)),
        ("body_quat_w", "f32", motion.body_quat[row].reshape(1, 14, 4)),
        ("frame_index", "i64", np.array([frame_index], dtype=np.int64)),
        ("catch_up", "u8", np.array([catch_up], dtype=np.uint8)),
        (
            "left_hand_joints",
            "f32",
            np.asarray(left_hand, dtype=np.float32).reshape(1, 7),
        ),
        (
            "right_hand_joints",
            "f32",
            np.asarray(right_hand, dtype=np.float32).reshape(1, 7),
        ),
    ]
    return _pack_message("pose", 1, fields)


class MotionPublisher:
    """Publish a held pose, then exactly one audited pass over the source rows."""

    def __init__(
        self,
        motion: MotionData,
        audit_path: Path,
        log_path: Path,
        *,
        port: int = 5556,
        fps: float = DEFAULT_FPS,
    ):
        self.motion = motion
        self.audit_path = audit_path
        self.log_path = log_path
        self.port = port
        self.fps = fps
        self.ready = threading.Event()
        self.prepare_first = threading.Event()
        self.first_sent = threading.Event()
        self.release_motion = threading.Event()
        self.motion_done = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.audit: dict[str, np.ndarray] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="sonic-motion-publisher", daemon=True
        )
        self._thread.start()
        if not self.ready.wait(timeout=5.0):
            self.raise_if_failed()
            raise CollectionError("publisher did not bind port 5556 within 5 seconds")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise CollectionError(
                f"motion publisher failed: {self.error}"
            ) from self.error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise CollectionError("motion publisher did not stop within 5 seconds")
        self.raise_if_failed()

    def _run(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.LINGER, 0)
        left_hands, right_hands = synthetic_hand_waveform(self.motion.frame_count)
        sent_global: list[int] = []
        sent_rows: list[int] = []
        sent_wall: list[float] = []
        sent_monotonic: list[float] = []
        sent_left: list[np.ndarray] = []
        sent_right: list[np.ndarray] = []
        global_index = 0
        next_motion_row = 0
        first_was_sent = False
        last_command = 0.0
        next_tick = time.monotonic()

        try:
            with self.log_path.open("w", buffering=1) as log:
                endpoint = f"tcp://127.0.0.1:{self.port}"
                publisher.bind(endpoint)
                log.write(
                    f"bound={endpoint} motion={self.motion.source_dir} "
                    f"frames={self.motion.frame_count} fps={self.fps}\n"
                )
                self.ready.set()
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                        continue
                    next_tick = max(next_tick + 1.0 / self.fps, now)

                    if now - last_command >= 0.1:
                        publisher.send(build_command_message(start=True, stop=False))
                        last_command = now

                    audit_row: int | None = None
                    if self.prepare_first.is_set() and not first_was_sent:
                        # The held pose accumulated while the controller loaded.
                        # A deliberate index discontinuity plus catch_up=true
                        # resets that backlog at the first source frame.
                        global_index += FIRST_FRAME_INDEX_JUMP
                        row = 0
                        audit_row = 0
                        next_motion_row = 1
                        first_was_sent = True
                    elif (
                        self.release_motion.is_set()
                        and next_motion_row < self.motion.frame_count
                    ):
                        row = next_motion_row
                        audit_row = row
                        next_motion_row += 1
                    elif next_motion_row >= self.motion.frame_count and first_was_sent:
                        row = self.motion.frame_count - 1
                    else:
                        row = 0

                    left = (
                        left_hands[row] if first_was_sent else np.zeros(7, np.float32)
                    )
                    right = (
                        right_hands[row] if first_was_sent else np.zeros(7, np.float32)
                    )
                    publisher.send(
                        build_pose_message(
                            self.motion,
                            row,
                            global_index,
                            left,
                            right,
                            catch_up=(
                                first_was_sent and not self.release_motion.is_set()
                            ),
                        )
                    )

                    if audit_row is not None:
                        sent_global.append(global_index)
                        sent_rows.append(audit_row)
                        sent_wall.append(time.time())
                        sent_monotonic.append(time.monotonic())
                        sent_left.append(left.copy())
                        sent_right.append(right.copy())
                        if audit_row == 0:
                            log.write(
                                f"first_source_frame global_index={global_index} "
                                f"catch_up=true index_jump={FIRST_FRAME_INDEX_JUMP}\n"
                            )
                            self.first_sent.set()
                        if next_motion_row >= self.motion.frame_count:
                            log.write(
                                f"motion_complete frames={len(sent_rows)} "
                                f"global_index={global_index}\n"
                            )
                            self.motion_done.set()
                    global_index += 1

                try:
                    publisher.send(build_command_message(start=False, stop=True))
                except zmq.ZMQError:
                    pass
                log.write("publisher_stopped\n")
        except BaseException as exc:  # keep the original exception for the supervisor
            self.error = exc
            self.ready.set()
            self.first_sent.set()
            self.motion_done.set()
        finally:
            if sent_rows:
                self.audit = {
                    "global_frame": np.asarray(sent_global, dtype=np.int64),
                    "source_row": np.asarray(sent_rows, dtype=np.int64),
                    "wall_time": np.asarray(sent_wall, dtype=np.float64),
                    "monotonic_time": np.asarray(sent_monotonic, dtype=np.float64),
                    "left_hand": np.stack(sent_left),
                    "right_hand": np.stack(sent_right),
                    "joint_pos": self.motion.joint_pos,
                    "joint_vel": self.motion.joint_vel,
                    "body_quat": self.motion.body_quat,
                }
                np.savez_compressed(self.audit_path, **self.audit)
            publisher.close(0)
            context.term()


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise CollectionError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise CollectionError(
                f"output directory must be absent or empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    (output_dir / "dataset").mkdir()
    (output_dir / "logs").mkdir()
    return output_dir


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # A just-finished local collection leaves harmless TIME_WAIT entries.
        # Match libzmq/TCP-server reuse semantics so those do not block the
        # next deterministic run, while an active listener still fails bind.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _wait_for_port(
    port: int,
    process: subprocess.Popen[bytes],
    publisher: MotionPublisher,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        publisher.raise_if_failed()
        code = process.poll()
        if code is not None:
            raise CollectionError(f"controller exited during startup with code {code}")
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.25)
        except OSError:
            time.sleep(0.25)
        else:
            connection.close()
            return
    raise CollectionError(
        f"controller port {port} did not become ready in {timeout:.0f}s"
    )


def _wait_for_event(
    channel: JsonLineChannel,
    event: str,
    timeout: float,
    processes: list[tuple[str, subprocess.Popen[bytes]]],
    publisher: MotionPublisher,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        publisher.raise_if_failed()
        for name, process in processes:
            code = process.poll()
            if code is not None:
                raise CollectionError(
                    f"{name} exited with code {code} while waiting for {event!r}"
                )
        try:
            message = channel.receive(timeout=min(0.5, deadline - time.monotonic()))
        except TimeoutError:
            continue
        if message["event"] != event:
            raise CollectionError(
                f"expected control event {event!r}, got {message['event']!r}"
            )
        return message
    raise CollectionError(f"timed out after {timeout:.0f}s waiting for {event!r}")


def terminate_process_group(
    process: subprocess.Popen[bytes] | None, *, timeout: float = 5.0
) -> str:
    """Terminate only the new session owned by ``process``; escalate if needed."""

    if process is None:
        return "not_started"
    if process.poll() is not None:
        return f"already_exited:{process.returncode}"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=timeout)
        return f"terminated:{process.returncode}"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)
        return f"killed:{process.returncode}"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _controller_command(root: Path, output_dir: Path) -> list[str]:
    deploy_root = root / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    binary = deploy_root / "target/release/g1_deploy_onnx_ref"
    environment = Path.home() / "tools/sonic_env.sh"
    shell = 'source "$1"; cd "$2"; source scripts/setup_env.sh; exec "$3" "${@:4}"'
    arguments = [
        "lo",
        "policy/release/model_decoder.onnx",
        "reference/example/",
        "--obs-config",
        "policy/release/observation_config.yaml",
        "--encoder-file",
        "policy/release/model_encoder.onnx",
        "--planner-file",
        "planner/target_vel/V2/planner_sonic.onnx",
        "--input-type",
        "zmq_manager",
        "--output-type",
        "all",
        "--zmq-host",
        "localhost",
        "--default-motion",
        "squat_001__A359",
        "--disable-crc-check",
        "--logs-dir",
        str(output_dir / "logs/controller_runtime"),
    ]
    return [
        "bash",
        "-c",
        shell,
        "sonic-controller",
        str(environment),
        str(deploy_root),
        str(binary),
        *arguments,
    ]


def _preflight(root: Path, output_dir: Path) -> None:
    deploy_root = root / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    binary = deploy_root / "target/release/g1_deploy_onnx_ref"
    environment = Path.home() / "tools/sonic_env.sh"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        build_log = output_dir / "logs/controller_build.log"
        with build_log.open("wb") as log:
            result = subprocess.run(
                ["bash", str(root / "scripts/build_sonic_controller.sh")],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise CollectionError(
                f"controller build failed with code {result.returncode}; see {build_log}"
            )
    required = [
        binary,
        environment,
        deploy_root / "policy/release/model_decoder.onnx",
        deploy_root / "policy/release/model_encoder.onnx",
        deploy_root / "policy/release/observation_config.yaml",
        deploy_root / "planner/target_vel/V2/planner_sonic.onnx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CollectionError("missing controller dependencies: " + ", ".join(missing))
    occupied = [port for port in REQUIRED_PORTS if not _port_is_free(port)]
    if occupied:
        raise CollectionError(
            "required ports are already in use; refusing to stop unrelated processes: "
            + ", ".join(map(str, occupied))
        )


def collect_motion(
    motion_dir: Path,
    output_dir: Path,
    *,
    env_id: str = DEFAULT_ENV_ID,
    fps: float = DEFAULT_FPS,
    startup_timeout: float = 600.0,
) -> dict[str, Any]:
    if fps != DEFAULT_FPS:
        raise CollectionError("the SONIC recorder currently requires exactly 50 Hz")
    motion = MotionData.load(motion_dir)
    output_dir = ensure_output_dir(output_dir)
    root = _repository_root()
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at.isoformat(),
        "motion_dir": str(motion.source_dir),
        "motion_frames": motion.frame_count,
        "fps": fps,
        "env_id": env_id,
        "dr_level": 0,
        "camera_shape_expected": [480, 640, 3],
        "dataset_dir": str(output_dir / "dataset"),
        "publisher_audit": str(output_dir / "publisher_audit.npz"),
        "events": {},
        "commands": {},
        "cleanup": {},
    }
    summary_path = output_dir / "run_summary.json"
    _write_summary(summary_path, summary)

    controller: subprocess.Popen[bytes] | None = None
    teleop: subprocess.Popen[bytes] | None = None
    parent_sock: socket.socket | None = None
    child_sock: socket.socket | None = None
    channel: JsonLineChannel | None = None
    controller_log = None
    teleop_log = None
    publisher = MotionPublisher(
        motion,
        output_dir / "publisher_audit.npz",
        output_dir / "logs/publisher.log",
        fps=fps,
    )

    try:
        _preflight(root, output_dir)
        summary["events"]["preflight_s"] = time.monotonic() - started_monotonic
        publisher.start()
        summary["events"]["publisher_bound_s"] = time.monotonic() - started_monotonic

        controller_command = _controller_command(root, output_dir)
        summary["commands"]["controller"] = shlex.join(controller_command)
        controller_environment = os.environ.copy()
        controller_environment.update(
            {
                "SONIC_AUTO_START": "1",
                "SONIC_FORCE_UNITREE_DDS": "1",
                "G1_ELBOW_POSE": "down",
                "CUDA_VISIBLE_DEVICES": controller_environment.get(
                    "CUDA_VISIBLE_DEVICES", "0"
                ),
            }
        )
        controller_log = (output_dir / "logs/controller.log").open("wb")
        controller = subprocess.Popen(
            controller_command,
            cwd=root,
            env=controller_environment,
            stdin=subprocess.DEVNULL,
            stdout=controller_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_port(5557, controller, publisher, startup_timeout)
        summary["events"]["controller_port_ready_s"] = (
            time.monotonic() - started_monotonic
        )

        parent_sock, child_sock = socket.socketpair()
        channel = JsonLineChannel(parent_sock)
        child_sock.set_inheritable(True)
        teleop_environment = os.environ.copy()
        teleop_environment.update(
            {
                CONTROL_FD_ENV: str(child_sock.fileno()),
                "MUJOCO_GL": "egl",
                "PYTHONUNBUFFERED": "1",
                "DISPLAY": teleop_environment.get("DISPLAY", ":1"),
                "G1_ELBOW_POSE": "down",
                "CUDA_VISIBLE_DEVICES": teleop_environment.get(
                    "CUDA_VISIBLE_DEVICES", "0"
                ),
            }
        )
        teleop_command = [
            str(root / ".venv/bin/python"),
            "-u",
            str(root / "src/simple/cli/teleop_wbc.py"),
            env_id,
            "--sim-mode",
            "mujoco",
            "--headless",
            "--record",
            "--dr-level",
            "0",
            "--render-hz",
            "50",
            "--num-episodes",
            "1",
            "--max-episode-steps",
            str(max(30000, motion.frame_count * 10)),
            "--save-dir",
            str(output_dir / "dataset"),
        ]
        summary["commands"]["teleop"] = shlex.join(teleop_command)
        teleop_log = (output_dir / "logs/teleop.log").open("wb")
        teleop = subprocess.Popen(
            teleop_command,
            cwd=root,
            env=teleop_environment,
            stdin=subprocess.DEVNULL,
            stdout=teleop_log,
            stderr=subprocess.STDOUT,
            pass_fds=(child_sock.fileno(),),
            start_new_session=True,
        )
        child_sock.close()
        child_sock = None

        ready = _wait_for_event(
            channel,
            "ready",
            startup_timeout,
            [("controller", controller), ("teleop", teleop)],
            publisher,
        )
        summary["events"]["recorder_ready_s"] = time.monotonic() - started_monotonic
        summary["recorder_ready"] = ready

        publisher.prepare_first.set()
        if not publisher.first_sent.wait(timeout=5.0):
            publisher.raise_if_failed()
            raise CollectionError("publisher did not send the first source frame")
        summary["events"]["first_source_frame_s"] = time.monotonic() - started_monotonic
        time.sleep(FIRST_FRAME_SETTLE_S)
        summary["events"]["first_source_settled_s"] = (
            time.monotonic() - started_monotonic
        )
        channel.send("start", frame_count=motion.frame_count, fps=fps)
        publisher.release_motion.set()
        summary["events"]["record_start_sent_s"] = time.monotonic() - started_monotonic

        saved = _wait_for_event(
            channel,
            "saved",
            motion.frame_count / fps + 180.0,
            [("controller", controller), ("teleop", teleop)],
            publisher,
        )
        summary["events"]["episode_saved_s"] = time.monotonic() - started_monotonic
        summary["recorder_saved"] = saved
        if not publisher.motion_done.wait(timeout=10.0):
            publisher.raise_if_failed()
            raise CollectionError("publisher did not complete all source frames")
        summary["events"]["publisher_done_s"] = time.monotonic() - started_monotonic

        try:
            teleop_code = teleop.wait(timeout=60.0)
        except subprocess.TimeoutExpired as exc:
            raise CollectionError(
                "teleop recorder did not exit within 60 seconds after save"
            ) from exc
        summary["teleop_exit_code"] = teleop_code
        if teleop_code != 0:
            raise CollectionError(f"teleop recorder exited with code {teleop_code}")
        if int(saved.get("frames", -1)) != motion.frame_count:
            raise CollectionError(
                f"recorder reported {saved.get('frames')} frames; expected "
                f"{motion.frame_count}"
            )

        publisher.stop()
        summary["publisher_frames"] = (
            0 if publisher.audit is None else len(publisher.audit["source_row"])
        )
        if summary["publisher_frames"] != motion.frame_count:
            raise CollectionError(
                f"publisher audit has {summary['publisher_frames']} rows; "
                f"expected {motion.frame_count}"
            )
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if child_sock is not None:
            child_sock.close()
        if channel is not None:
            channel.close()
        elif parent_sock is not None:
            parent_sock.close()
        if publisher._thread is not None and publisher._thread.is_alive():
            try:
                publisher.stop()
            except BaseException as exc:
                summary["cleanup"]["publisher"] = f"failed: {exc}"
            else:
                summary["cleanup"]["publisher"] = "stopped"
        else:
            summary["cleanup"]["publisher"] = "already_stopped"
        summary["cleanup"]["teleop"] = terminate_process_group(teleop)
        summary["cleanup"]["controller"] = terminate_process_group(controller)
        if teleop_log is not None:
            teleop_log.close()
        if controller_log is not None:
            controller_log.close()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_s"] = time.monotonic() - started_monotonic
        _write_summary(summary_path, summary)
    return summary


def main(
    motion_dir: Annotated[
        Path, typer.Argument(help="SONIC reference motion directory")
    ],
    output_dir: Annotated[Path, typer.Argument(help="New or empty output directory")],
    env_id: Annotated[
        str, typer.Option(help="SIMPLE teleop environment id")
    ] = DEFAULT_ENV_ID,
    startup_timeout: Annotated[
        float, typer.Option(help="Seconds allowed for controller and recorder startup")
    ] = 600.0,
) -> None:
    """Record one complete reference motion through the live SONIC controller."""

    try:
        summary = collect_motion(
            motion_dir,
            output_dir,
            env_id=env_id,
            startup_timeout=startup_timeout,
        )
    except (CollectionError, OSError, TimeoutError) as exc:
        typer.echo(f"collect-sonic-motion: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Recorded {summary['motion_frames']} frames to {summary['dataset_dir']}"
    )


def typer_main() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer_main()
