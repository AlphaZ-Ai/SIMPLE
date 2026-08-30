# Release note — Integrate SONIC whole-body control in SIMPLE

**Branch:** `pr-24` · **Date:** 2026-08-26

From VR teleoperation to closed-loop policy inference. SIMPLE now drives the external
SONIC whole-body controller through the full data lifecycle: record a G1 episode in
simulation, replay it through real physics and render it into a trainable LeRobot
dataset, then evaluate a served VLA policy with Isaac rendering and model inference
inside the control loop.

The enabling change is **simulation lockstep**: the controller advances on simulation
time instead of wall time.

## Why this was hard

SONIC is a separate C++ process that decodes a 64-dimensional body token, produces body
and hand commands, and balances the robot — on its own 50 Hz wall-clock thread.
Closed-loop evaluation needs an Isaac render and a policy request inside every control
frame, and those take hundreds of milliseconds of wall time. The legacy controller kept
stepping through that pause against frozen state, and its 500 ms lowstate watchdog
eventually killed the run.

Simulation lockstep makes one numbered 20 ms simulation boundary produce exactly one
`Control()` call.

| Scheduling | Rendering | Controller steps per simulated second |
|---|---|---:|
| Legacy wall clock | Off | 50.734 |
| Legacy wall clock | In loop | 362.905 |
| Simulation lockstep | Off | 50.000 |
| Simulation lockstep | In loop | 50.000 |

Lockstep is opt-in via `--sim-lockstep`. The default path and the real-robot deployment
path keep the original wall-clock behaviour untouched.

## Prerequisites

Build the controller first; the lockstep launchers deliberately do not build a missing
binary.

```bash
bash scripts/build_sonic_controller.sh third_party/GR00T-WholeBodyControl
```

These must then exist under `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/`:

```text
target/release/g1_deploy_onnx_ref
policy/release/model_decoder.onnx
policy/release/model_encoder.onnx
policy/release/observation_config.yaml
```

Ports 5556, 5557 and 13579 must be free — each launcher refuses to start rather than
stop an unknown process holding one. Headless simulation and rendering require
`HEADLESS=1` and `MUJOCO_GL=egl`.

## The pipeline — three stages, three launchers

### Stage 1 — collect: `scripts/run_teleop_wbc.sh`

One command brings up the SONIC controller, the Pico headset manager and the simulator,
and tears the whole stack down on a single Ctrl+C. The recorder writes LeRobot v2.1
episodes at 50 fps: a 43-dimensional joint state, a 78-dimensional action (64D body
token + two 7D hand targets), base pose and velocity, scene object poses, and the
egocentric RGB view.

```bash
bash scripts/run_teleop_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 \
  --record --save-dir data/teleop_wbc
```

For runs without a headset, `collect-sonic-motion` publishes one audited pass over a
SONIC reference motion through the same controller and recorder path.

### Stage 2 — replay: `scripts/run_replay_wbc.sh`

Replay rebuilds the recorded scene, seeds the exact start pose, and streams every
recorded token and hand target back through the live controller so MuJoCo physics plays
the motion out — the plausibility check a kinematic teleport replay cannot give. The
physical pass then hands its saved trace to a fresh Isaac process, which renders the
frames that become the LeRobot dataset for VLA models such as Psi0.

```bash
export HEADLESS=1 MUJOCO_GL=egl

bash scripts/run_replay_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 \
  --sim-lockstep \
  --data-dir <dataset>/level-0 \
  --episode-start 0 --num-episodes 1 \
  --sim-mode mujoco_isaac \
  --replay-dir /tmp/sonic-lockstep-replay \
  --dr-level 0
```

Rendering runs in a separate interpreter so controller native libraries and
SimulationApp are never initialized and destroyed in one process. Video suffixes report
the task predicate (`_success.mp4` / `_failed.mp4`). DR levels 0, 1 and 2 are validated.

### Stage 3 — evaluate: `scripts/run_eval_wbc.sh`

The evaluator starts one lockstep controller, runs the environment in the foreground,
and sends one egocentric image, a raw 43-dimensional state and the task instruction to
an HTTP policy service. The service returns a 30×78 action chunk; each row is executed
at 50 Hz. Isaac renders and the model infers *inside* the loop, and simulation time
waits.

```bash
# terminal 1 — the policy service (Psi0 project)
serve_psi0_sonic_http \
  --policy psi0 --port 8014 \
  --ckpt-step 40000 \
  --run-dir .runs/finetune/sonic-wbcbox.… \
  --rtc-mode test_time --action-exec-horizon 24

# terminal 2 — the evaluator
HEADLESS=0 bash scripts/run_eval_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/teleop_wbc/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0 \
  --host 127.0.0.1 --port 8014 \
  --episode-start 1 --num-episodes 1 \
  --dr-level 0 \
  --eval-dir /tmp/sonic-psi0-eval
```

Passing `replay` instead of `psi0` swaps in a recorded-action policy implementing the
identical chunk contract — the deterministic end-to-end reference for the evaluator.

## How the barrier works

Correctness rests on a tuple both sides agree on exactly: `(session, sim-step,
action-seq)`. Python stamps it into `LowState.reserve[0:4]` and recomputes the CRC; the
protocol-v4 ZMQ token message carries the same three fields.

1. Python freezes the latest MuJoCo proprioception and assigns session, simulation-step
   and action sequence.
2. The 78D action splits into a 64D body token and two 7D hand targets.
3. The token goes out over ZMQ, the robot state over DDS. Arrival order is unrestricted.
4. The C++ gate waits until both messages carry the same exact tuple.
5. The lockstep thread runs one `Control()` call and commits the tuple only after body
   and both hand command buffers are complete.
6. The 500 Hz writers stamp that committed tuple on the body and both hand messages.
7. Python waits for all three matching acknowledgements, then advances four 5 ms MuJoCo
   substeps.
8. Isaac synchronizes and renders after the barrier releases; the next numbered request
   waits for it.

Constants: `LOCKSTEP_MAGIC = 0x534C4B31`, `NO_ACTION_SEQ = 0xFFFFFFFF`, retry period
50 ms, ack timeout 2.0 s. No wall-clock sleep and no 200 Hz Python heartbeat run in
lockstep mode.

- The periodic 100 Hz input, 50 Hz control and 10 Hz planner threads collapse into one
  stoppable condition-variable thread; the 500 Hz command writers stay for packet-loss
  recovery.
- Session mismatch, step jumps, action jumps and frame/action mismatch are rejected
  without running `Control()`. Duplicate requests are idempotent.
- A new session clears controller histories that could leak across episodes while
  retaining initialized policy engines and `program_state=CONTROL`. Injected
  acknowledgements from a previous session cannot release the next barrier.
- Only the 500 ms wall-age stop is disabled, and only under `--sim-lockstep`. CRC,
  motor, temperature, model, dimension and sequence safety stay active.
- StateLogger runs in strict mode: the configured simulation stride is required and the
  timestamp-based fallback is rejected, so wall time cannot enter observations.

## Evidence

Reference input: the first real XMoveBendPick teleoperation episode at DR0 and 50 Hz.
Two in-loop-rendering runs and one no-render run produced bitwise identical qpos/qvel
traces — maximum difference 0.0.

| Measurement | Value |
|---|---:|
| Strict acknowledgements / trace frames | 1,164 / 1,165 |
| Commits per complete simulated second | 50 |
| Max request-to-ack latency | 11.3 – 12.1 ms (>1.98 s of timeout unused) |
| Ack waits overlapped by rendering | 0 |
| Flag-off regression replays | 5 baseline + 5 modified, all succeeded |
| Psi0 inference, median / p95 | 0.204 / 0.230 s |
| In-loop render, median / p95 | 0.113 / 0.135 s |

Five-episode closed-loop evaluation: every episode ran a fresh lockstep session, every
acknowledgement sequence and four-substep group was exact, and no model error, request
retry or lockstep timeout occurred. The measured task success rate (0/5 for the
evaluated checkpoint, with recorded-action references succeeding 5/5 in the same
environments) is a property of that checkpoint, not of this implementation.

## Also in this branch: real-time chunking (RTC) support

The evaluator originally required a non-RTC service returning and executing all 30 chunk
rows. It now reads `action_exec_horizon` from the service's `/info` response, executes
exactly that many rows per chunk — the shift an RTC server assumes — and sends
`history["reset"]` on the first request of every episode, so an episode's first chunk is
never conditioned on the previous episode's last one. This is what makes `--rtc-mode
test_time --action-exec-horizon 24` coherent end to end.

`rtc_mode=train` is refused: its frozen prefix is `Tp−Ta` wide and valid only on an
RTC-trained checkpoint. An RTC server must run with `action_exec_horizon <
action_chunk_size`. The `replay` policy takes a matching `exec_horizon` so the
deterministic reference tracks whatever the served policy does.

**Status:** these RTC changes are present in the working tree on `pr-24` but not yet
committed.

## Implementation map

| Path | Responsibility |
|---|---|
| `src/simple/agents/lockstep.py` | Session state machine, four-substep boundary, retransmission, timeout, three-channel barrier |
| `src/simple/agents/lockstep_unitree_bridge.py` | Atomic command snapshot; LowState reserve stamping and CRC without patching the upstream bridge |
| `src/simple/agents/pico_wbc_agent.py` | Teleoperation shim: headset ZMQ in, DDS state out, crane and engage handshake |
| `src/simple/agents/replay_wbc_agent.py` | Token transport, lockstep session API, replay action decoding, acknowledged command forwarding |
| `src/simple/agents/sonic_eval_agent.py` | Action-chunk policy interface, replay and Psi0 HTTP policies, queue, contract validation, 78D dispatch |
| `src/simple/evals/sonic_wbc.py` | Evaluation lifecycle, budgets, termination reasons, trace and result persistence |
| `src/simple/cli/collect_sonic_motion.py` | Deterministic hardware-free recording of a SONIC reference motion |
| `src/simple/cli/{teleop,replay,eval}_wbc.py` | The three entry points behind the three launchers |
| `src/simple/cli/render_replay_wbc.py` | Offline Isaac rendering of a saved MuJoCo trace, in a fresh process |
| `tests/fake_lockstep_controller.py` | Protocol test double speaking the real DDS and ZMQ message formats |

The C++ side — `sim_lockstep.hpp`, the lockstep thread in `g1_deploy_onnx_ref.cpp`,
strict StateLogger, ZMQ envelope parsing and hand command stamping — lives in the
GR00T-WholeBodyControl submodule at the revision this repository records.

## Reproducing the validation

```bash
.venv/bin/pytest -q tests/test_lockstep.py
.venv/bin/pytest -q tests/test_sonic_wbc_eval.py
.venv/bin/pytest -q tests/test_sonic_wbc_eval_integration.py

third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/run_tests \
  --gtest_filter='SimLockstep.*:StateLoggerLockstep.*'
```

Then run the Stage 2 replay command twice with a fresh controller process each time and
compare normalized `(step, action)` acknowledgements, the
`episode_0_mujoco_state_trace.npz` arrays, commits per complete simulated second, and
video frame count. Expected: 1,164 acknowledgements, 1,165 frames, 50 commits per
second, maximum trace difference 0.0. The replay-policy evaluator must produce the same
values and `termination_reason="task_success"`.

## Known limitations

- **Renderer reset variability.** Repeated rendering of a fixed MuJoCo state did not
  converge to one stable frame within 60 renders (final-frame RGB MAD 6.68–6.77 across
  fresh processes). Control timing is unaffected, but two runs from an identical state
  can see different first images. A bounded audit found no supported global seed for the
  active `RayTracedLighting` renderer in Isaac Sim 4.5.
- **The Psi0 service is stochastic by design.** Fresh diffusion noise per request; for
  one fixed observation, per-element action peak-to-peak was median 0.012, p95 0.026,
  max 0.302 over 50 requests. A request-level seed would make evaluation repeatable;
  that upstream extension is not implemented.
- **Frame count is not a determinism criterion with the flag off.** Baseline and
  modified binaries gave overlapping frame-count intervals [1151, 1167] and
  [1161, 1168] over five flag-off replays each.
- **The test double stops at the protocol.** It does not simulate SONIC inference,
  physics, controller histories, real C++ thread scheduling, GPU behaviour, or
  middleware under load. The same scenarios were run against the compiled controller.
- **Offline rendering is not an evaluation path.** Stage 2's offline render reproduces a
  saved trace in a second, open-loop pass. Closed-loop evaluation must keep Isaac inside
  the simulation loop and use lockstep.

## References

The design and validation plans this note was distilled from were removed in `d4bbf32`;
this file is the surviving record. For detail beyond it, read the code:

- Protocol and barrier: `src/simple/agents/lockstep.py`, `tests/test_lockstep.py`
- Evaluation lifecycle: `src/simple/evals/sonic_wbc.py`, `tests/test_sonic_wbc_eval.py`
- C++ lockstep: `sim_lockstep.hpp` and `g1_deploy_onnx_ref.cpp` in the
  GR00T-WholeBodyControl submodule at the revision this repository records
