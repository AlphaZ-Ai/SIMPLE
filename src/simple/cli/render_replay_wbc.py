# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Render a saved replay state trace in a fresh IsaacSim process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


JOB_SCHEMA_VERSION = 1
REQUIRED_JOB_KEYS = {
    "schema_version",
    "env_id",
    "environment_config",
    "replay_dir",
    "episode_index",
    "dr_level",
    "render_hz",
    "physics_dt",
    "sonic_config",
    "debug_render",
    "success",
    "third_person_isaac",
}


def _load_job(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        job = json.load(handle)
    if not isinstance(job, dict):
        raise TypeError(f"Offline Isaac job must be a JSON object, got {type(job).__name__}")

    missing = sorted(REQUIRED_JOB_KEYS.difference(job))
    if missing:
        raise ValueError(f"Offline Isaac job is missing required keys: {missing}")
    if job["schema_version"] != JOB_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported offline Isaac job schema_version={job['schema_version']!r}; "
            f"expected {JOB_SCHEMA_VERSION}"
        )
    if not isinstance(job["environment_config"], dict):
        raise TypeError("environment_config must be a JSON object")
    if not isinstance(job["sonic_config"], dict):
        raise TypeError("sonic_config must be a JSON object")
    return job


def _load_trace(path: Path) -> list[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted({"qpos", "qvel"}.difference(arrays.files))
        if missing:
            raise ValueError(f"MuJoCo state trace is missing arrays: {missing}")
        qpos = np.asarray(arrays["qpos"]).copy()
        qvel = np.asarray(arrays["qvel"]).copy()

    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError(f"Trace arrays must be 2-D, got qpos={qpos.shape}, qvel={qvel.shape}")
    if len(qpos) == 0 or len(qvel) == 0:
        raise ValueError("MuJoCo state trace must contain at least one state")
    if len(qpos) != len(qvel):
        raise ValueError(f"Trace length mismatch: qpos={len(qpos)}, qvel={len(qvel)}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("MuJoCo state trace contains NaN or Inf")
    return [{"qpos": qpos[index], "qvel": qvel[index]} for index in range(len(qpos))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a saved MuJoCo replay trace through IsaacSim in this fresh process."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()

    trace_path = args.trace.resolve(strict=True)
    job_path = args.job.resolve(strict=True)
    job = _load_job(job_path)
    states = _load_trace(trace_path)

    recorded_trace_path = job.get("trace_path")
    if recorded_trace_path and Path(recorded_trace_path).resolve() != trace_path:
        raise ValueError(
            f"Job trace_path={recorded_trace_path!r} does not match CLI trace={str(trace_path)!r}"
        )

    print(
        f"[ReplayWbcRender] pid={os.getpid()} trace={trace_path} states={len(states)} "
        f"job={job_path}",
        flush=True,
    )

    # Import only after validating inputs so Kit is initialized by this fresh
    # interpreter, never by the process that owned DDS/ZMQ/MuJoCo playback.
    from simple.cli.replay_wbc import _render_state_trace_with_isaac

    _render_state_trace_with_isaac(
        env_id=str(job["env_id"]),
        env_conf=job["environment_config"],
        states=states,
        replay_dir=str(job["replay_dir"]),
        ep_idx=int(job["episode_index"]),
        dr_level=job["dr_level"],
        render_hz=int(job["render_hz"]),
        sim_dt=float(job["physics_dt"]),
        sonic_config=job["sonic_config"],
        debug_render=bool(job["debug_render"]),
        success=bool(job["success"]),
        third_person_isaac=bool(job["third_person_isaac"]),
        save_trace=False,
    )
    print("[ReplayWbcRender] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
