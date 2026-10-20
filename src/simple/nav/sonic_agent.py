# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Drive the external SONIC controller's locomotion planner from code.

PicoWbcAgent already does what navigation needs on the robot side: it balances
the robot with the in-process decoupled WBC until it settles, publishes
rt/lowstate from a 200 Hz heartbeat, and forwards the controller's rt/lowcmd as
torques. What it lacks is a commander: in teleop the pico_manager publishes the
ZMQ `command` / `planner` messages. NavSonicAgent publishes them itself.

The controller must run free (no --sim-lockstep: the planner thread only exists
in that mode) with --planner-file and `--input-type zmq_manager`; see
`controller_command`.

Planner frame: `movement` and `facing` are unit vectors in the planner's frame,
whose +x is the robot heading when the planner initialized (engage). Speed is
`speed` in m/s, -1 for the mode default. A planner message older than 1 s makes
the controller fall back to IDLE, so `walk()` must be called continuously.
"""

import math
import time
from pathlib import Path

import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message
from simple.agents.pico_wbc_agent import PicoWbcAgent

# gear_sonic_deploy LocomotionMode (localmotion_kplanner.hpp)
IDLE, SLOW_WALK, WALK = 0, 1, 2
SLOW_WALK_SPEED = (0.1, 0.8)  # m/s range the planner accepts in SLOW_WALK


class NavSonicAgent(PicoWbcAgent):
    def __init__(self, robot, port: int = 5556, controller_ready=None, settle_s: float = 0.5):
        """controller_ready: callable -> True once the controller's planner has
        initialized (e.g. by watching its log); handover waits for it."""
        self._controller_ready = controller_ready or (lambda: True)
        self._settle_s = settle_s
        self._ready_since = None
        # the controller's ZMQ SUB sockets connect to localhost:<port>; we bind the PUB
        self._pub_ctx = zmq.Context.instance()
        self._pub = self._pub_ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{port}")
        super().__init__(robot, auto_drop=False)
        self.planner_engaged = False
        self.heading = 0.0  # planner-frame yaw we are asking for (rad)

    def _init_pico_streamer(self):
        # no headset: skip the TCP camera server
        self._streaming = None
        self._frame_buffer = None

    def controller_stiff(self) -> bool:
        """True once the controller commands real position gains (it holds the robot)."""
        lc = self.unitree_bridge.low_cmd
        return max(lc.motor_cmd[i].kp for i in range(29)) > 5.0

    @property
    def handed_over(self) -> bool:
        return getattr(self, "_handed_over", False)

    def get_action(self, observation, instruction=None, **kwargs):
        # PicoWbcAgent hands over as soon as the robot settles. In teleop the
        # controller has long been running by then; here it may still be
        # building TensorRT engines, and forwarding its idle zero-gain commands
        # drops the robot. Keep the stabilization WBC in charge until the
        # controller is engaged in PLANNER mode and actually holding.
        if not self.handed_over:
            ready = (self.robot.stabilized and self.planner_engaged and self.controller_stiff()
                     and self._controller_ready())
            if ready and self._ready_since is None:
                self._ready_since = time.monotonic()
            if ready and time.monotonic() - self._ready_since >= self._settle_s:
                self._handed_over = True
                print("[NavSonic] planner running, controller holding -> handing control to SONIC", flush=True)
            else:
                self._poll_manager_liveness()
                return self.get_stabilize_action(kwargs["privileged_info"]["proprio"])
        return super().get_action(observation, instruction=instruction, **kwargs)

    def reset_policy(self):
        super().reset_policy()
        self._handed_over = False
        self._ready_since = None

    # ---- controller commands ------------------------------------------------
    def engage_planner(self, repeats: int = 5):
        """Start the controller in PLANNER mode; the planner frame starts at the current heading."""
        for _ in range(repeats):
            self._pub.send(build_command_message(start=True, stop=False, planner=True))
            time.sleep(0.02)
        self.planner_engaged = True
        self.heading = 0.0

    def stop_controller(self):
        self._pub.send(build_command_message(start=False, stop=True, planner=True))

    def walk(self, vx: float, vy: float, wz: float, dt: float):
        """Body-frame velocity command -> planner message (call at >= 10 Hz).

        Yaw rate is integrated into the planner-frame facing; the walking
        direction is the commanded velocity direction rotated by that facing.
        """
        self.heading += wz * dt
        speed = math.hypot(vx, vy)
        facing = [math.cos(self.heading), math.sin(self.heading), 0.0]
        if speed < 0.05:
            mode, movement, speed = IDLE, facing, -1.0
        else:
            ang = self.heading + math.atan2(vy, vx)
            movement = [math.cos(ang), math.sin(ang), 0.0]
            mode = SLOW_WALK
            speed = min(max(speed, SLOW_WALK_SPEED[0]), SLOW_WALK_SPEED[1])
        self._pub.send(build_planner_message(mode, movement, facing, speed=speed, height=-1.0))

    def close(self):
        super().close()
        self._pub.close(linger=0)


def controller_command(root: Path, logs_dir: Path) -> list[str]:
    """Free-running SONIC controller with the locomotion planner (cf. collect_sonic_motion)."""
    deploy_root = root / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    binary = deploy_root / "target/release/g1_deploy_onnx_ref"
    shell = 'source "$1"; cd "$2"; source scripts/setup_env.sh; exec "$3" "${@:4}"'
    args = [
        "lo", "policy/release/model_decoder.onnx", "reference/example/",
        "--obs-config", "policy/release/observation_config.yaml",
        "--encoder-file", "policy/release/model_encoder.onnx",
        "--planner-file", "planner/target_vel/V2/planner_sonic.onnx",
        "--input-type", "zmq_manager", "--output-type", "all", "--zmq-host", "localhost",
        # without this the controller starts from a kick reference motion before the planner takes over
        "--default-motion", "squat_001__A359",
        "--disable-crc-check", "--logs-dir", str(logs_dir),
    ]
    # NB: log_ready_check(log) below watches this process's stdout for planner init
    return ["bash", "-c", shell, "sonic-controller", str(Path.home() / "tools/sonic_env.sh"),
            str(deploy_root), str(binary), *args]


PLANNER_READY_LINE = "Planner initialized successfully"


def log_ready_check(log_path: Path):
    """controller_ready callable: True once the controller logged planner init."""
    state = {"ready": False}

    def ready():
        if not state["ready"]:
            try:
                state["ready"] = PLANNER_READY_LINE in Path(log_path).read_text(errors="ignore")
            except FileNotFoundError:
                pass
        return state["ready"]

    return ready
