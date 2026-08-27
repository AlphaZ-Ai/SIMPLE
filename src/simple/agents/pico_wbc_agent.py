# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import json
import threading
import time

import cv2
import numpy as np
import zmq

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.pico.tcp_server import TCPControlServer
from simple.teleop.pico.tcp_video_sender import TCPVideoSender
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread
from .sonic_wbc_agent import SonicWbcAgent


class PicoWbcAgent(SonicWbcAgent):
    """Teleoperation agent that uses WBC policy from SONIC controller.

    This agent runs through executing the external SONIC whole-body controller.
    The SONIC controller (gear_sonic_deploy, `./deploy.sh sim --input-type zmq_manager`) 
    runs as a separate process, and the operator's poses reach it via the pico_manager over ZMQ:

        pico_manager --ZMQ pose/ctrl--> SONIC controller
        SONIC controller --DDS rt/lowcmd--> UnitreeSdk2Bridge 
        this agent --DDS rt/lowstate--> SONIC controller

    Per control step:
      1. forward the controller's latest LowCmd as ActionCmd("wbc_torque")
      2. poll the pico_manager's "ctrl" ZMQ topic for drop/reset events

    Sim proprio is cached every physics substep (`cache_proprio`) and published
    to DDS rt/lowstate at a steady 200 Hz by the heartbeat thread
    (`start_state_heartbeat`), matching gear_sonic's own sim loop.
    """

    def __init__(
        self,
        robot: G1Sonic, 
        auto_drop: bool = False,
        drop_settle_s: float = 3.0,
    ):
        super().__init__(robot)

        self.episodes_saved = 0
        self.num_episodes = 100
        # Headset overlay state, set by the teleop loop each tick:
        #   ""      -> nothing drawn
        #   "armed" -> yellow ARMED: POSE teleop active, waiting for the record
        #              gate (standing + controller holding) to open
        #   "rec"   -> red REC: frames are actually being saved
        #   "idle"  -> gray IDLE: episode saved, waiting for the scene reset
        self.rec_indicator = ""
        self.sim_dt = self.robot.sonic_config["SIMULATE_DT"]

        self._control_dt = 4 * self.sim_dt  # band descent + crane-hold PD integration

        self._dropping = False
        self._drop_rate = 0.15  # m/s
        self._reset_requested = False

        self._pub_lock = threading.Lock()
        self._last_proprio = None

        ## --- Pico VR streaming (camera feed to headset) ---
        self._init_pico_streamer()

        # --- SONIC controller plugins ---
        self._init_unitree_bridge()
        self._init_zmq_subscriber()
        self._init_token_subscriber()
        self._init_stabilize_wbc()
        
        self.auto_drop = auto_drop
        self.drop_settle_s = drop_settle_s

    @property
    def reset_requested(self) -> bool:
        """True if a reset was requested via the pico_manager. Clears on read."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    @property
    def is_teleop_active(self) -> bool:
        """True once the external SONIC controller is commanding the robot."""
        return self.unitree_bridge.cmd_received()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_pico_streamer(self):
        """Set up TCP server for VR camera streaming."""
        self._streaming: StreamingThread | None = None
        self._frame_buffer = FrameBuffer()

        tcp_server = TCPControlServer("0.0.0.0:13579")

        def on_open_camera(camera_req):
            print(f"[PicoWbc] OPEN_CAMERA: {camera_req}")
            if self._streaming and self._streaming.is_running():
                print("[PicoWbc] Already streaming, ignoring duplicate OPEN_CAMERA")
                return

            fps = camera_req.get("fps") or 60
            width = camera_req.get("width") or 2560
            height = camera_req.get("height") or 720
            bitrate = camera_req.get("bitrate") or 4_000_000
            hevc = bool(camera_req.get("enableMvHevc"))
            ip = camera_req.get("ip")
            port = camera_req.get("port")

            if not ip or not port:
                print("[PicoWbc] OPEN_CAMERA missing ip/port, cannot stream")
                return
            try:
                sender = TCPVideoSender(
                    ip=ip, port=port,
                    width=width, height=height, fps=fps,
                    bitrate=bitrate, hevc=hevc,
                )
            except ConnectionRefusedError:
                print(f"[PicoWbc] Connection refused to {ip}:{port}")
                return

            self._streaming = StreamingThread(
                frame_buffer=self._frame_buffer,
                fps=fps,
                publishers=[sender],
                on_ended=lambda: tcp_server.close_client(),
            )
            self._streaming.start()

        def on_close_camera():
            print("[PicoWbc] CLOSE_CAMERA received")
            if self._streaming:
                self._streaming.stop()
                self._streaming = None
            tcp_server.close_client()

        tcp_server.on_open_camera = on_open_camera
        tcp_server.on_close_camera = on_close_camera
        tcp_server.start()

    def _init_unitree_bridge(self):
        """DDS bridge: publishes rt/lowstate etc., subscribes to rt/lowcmd."""
        self.unitree_bridge = UnitreeSdk2Bridge(self.robot.sonic_config)
            
    def _init_zmq_subscriber(self):
        """ZMQ SUB socket on the pico_manager's 'ctrl' topic (drop/reset events)."""
        ##   "pose" — your VR poses (consumed by the C++ controller, NOT us)
        ##   "ctrl" — sim-control events: drop_robot, reset_env (consumed HERE)
        pico_host = self.robot.sonic_config.get("PICO_HOST", "localhost")
        pico_port = self.robot.sonic_config.get("PICO_PORT", 5556)

        self._ctrl_context = zmq.Context()
        self._ctrl_socket = self._ctrl_context.socket(zmq.SUB)
        self._ctrl_socket.setsockopt_string(zmq.SUBSCRIBE, "ctrl")
        self._ctrl_socket.setsockopt(zmq.CONFLATE, 1) 
        self._ctrl_socket.connect(f"tcp://{pico_host}:{pico_port}")

        self._planner_socket = self._ctrl_context.socket(zmq.SUB)
        self._planner_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
        self._planner_socket.setsockopt(zmq.CONFLATE, 1)
        self._planner_socket.connect(f"tcp://{pico_host}:{pico_port}")
        self._latest_planner = None

        # Liveness tap on EVERY topic.  "Is the operator engaged?" cannot be
        # answered from the planner topic alone: POSE mode publishes on
        # "pose" instead, so a perfectly engaged operator looked disengaged.
        # Any traffic at all means the manager left OFF, which is exactly the
        # question.
        self._live_socket = self._ctrl_context.socket(zmq.SUB)
        self._live_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._live_socket.setsockopt(zmq.CONFLATE, 1)
        self._live_socket.connect(f"tcp://{pico_host}:{pico_port}")
        self._manager_t = None

    def _init_stabilize_wbc(self):
        """In-process decoupled WBC used ONLY for the stabilization phase.

        Slim port of PicoDecoupledAgent._init_decoupled_policy: robot model +
        WBC policy (lower-body RL + upper-body interpolation), but no teleop
        policy / hand IK — stabilization just ramps to the WBC default pose,
        it never tracks the operator.  Once ``robot.stabilized`` latches, the
        external SONIC controller takes over and this policy sits idle.
        """
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

        sonic_cfg = self.robot.sonic_config
        enable_waist = sonic_cfg.get("enable_waist", False)
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )

        # Load the decoupled_wbc's own config (g1_29dof_gear_wbc.yaml) via
        # ControlLoopConfig — SIMPLE's sonic_config has VERSION=sonic_model12,
        # which get_wbc_policy does not accept.
        dwbc_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()
        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        self._wbc_policy = get_wbc_policy(
            "g1", self._dwbc_robot_model, wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )
        # The lower-body RL only balances when use_policy_action is on — the
        # in-process analogue of the SONIC controller's ']'.  Without it the
        # policy echoes the current leg positions (g1_gear_wbc_policy.py:309)
        # and the robot collapses.  teleop_decoupled_wbc.py:251 does the same.
        # The flag is persistent across _wbc_policy.reset(), so set-once here.
        self._wbc_policy.lower_body_policy.use_policy_action = True
        self._control_frequency = dwbc_config.control_frequency
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        """Convert SIMPLE observation dict to decoupled_wbc observation format.

        Ported from PicoDecoupledAgent._build_wbc_observation: the decoupled
        WBC expects q/dq/ddq/tau_est as full 43-dof configuration vectors
        (29 body + 7+7 hands) in the robot_model's joint order, with hand
        joints in NATURAL (thumb/index/middle) order.
        """
        rm = self._dwbc_robot_model
        obs = {}

        left_hand_q = sim_obs.get("left_hand_q", np.zeros(7))
        right_hand_q = sim_obs.get("right_hand_q", np.zeros(7))
        left_hand_dq = sim_obs.get("left_hand_dq", np.zeros(7))
        right_hand_dq = sim_obs.get("right_hand_dq", np.zeros(7))

        mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])

        obs["q"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=mjcf_to_natural_order(left_hand_q),
            right_hand_actuated_joint_values=mjcf_to_natural_order(right_hand_q),
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=mjcf_to_natural_order(left_hand_dq),
            right_hand_actuated_joint_values=mjcf_to_natural_order(right_hand_dq),
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("left_hand_ddq", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("right_hand_ddq", np.zeros(7))),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("left_hand_tau_est", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural_order(sim_obs.get("right_hand_tau_est", np.zeros(7))),
        )

        obs["floating_base_pose"] = sim_obs["floating_base_pose"]
        obs["floating_base_vel"] = sim_obs["floating_base_vel"]
        obs["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))

        obs["torso_quat"] = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6] if "secondary_imu_vel" in sim_obs else np.zeros(3)

        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))

        return obs

    def get_stabilize_action(self, proprio) -> ActionCmd:
        """Run the WBC pipeline ramping to the WBC default pose — used during stabilization.

        Ported from PicoDecoupledAgent.get_stabilize_action: drives the upper
        body toward the WBC default configuration (forearms pointing forward)
        while the lower-body RL policy balances, until ``robot.stabilized``
        latches and the external SONIC controller takes over.
        """
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )
        t_now = time.monotonic()
        control_freq = self._control_frequency

        # Fresh proprio, not the privileged_info snapshot from the previous
        # env.step — the balance policy is sensitive to a stale (20 ms old)
        # observation.  Matches PicoDecoupledAgent.get_stabilize_action.
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # Use the WBC's own default upper body pose, not the MJCF keyframe
        # defaults.  On the first step give a 2s ramp window so the arms move
        # smoothly to the default rather than snapping.
        is_first_step = self._cached_target_q is None
        if is_first_step:
            print("[PicoWbc] Stabilizing — ramping to the WBC default pose")
            self._stab_ticks = 0
        # Once-a-second trace: is the stand policy holding the robot up?
        self._stab_ticks += 1
        if self._stab_ticks % 50 == 0:
            base_vel = float(np.max(np.abs(self.robot.mjData.qvel[0:6])))
            print(f"[PicoWbc] stabilizing t={self._stab_ticks / 50:.0f}s "
                  f"pelvis_z={self.robot.pelvis_z:.3f} max|base qvel|={base_vel:.2e}")
        default_upper_body = self._dwbc_robot_model.get_initial_upper_body_pose()
        goal = {
            "target_upper_body_pose": default_upper_body,
            "navigate_cmd": np.asarray(DEFAULT_NAV_CMD),
            "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
            "target_time": t_now + (2.0 if is_first_step else 1 / control_freq),
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            target_waist=wbc_action["target_waist"],
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

    def _init_token_subscriber(self):
        """Subscribe to the controller's per-tick "g1_debug" ZMQ state stream
        (ZMQOutputHandler, PUB port 5557) for the recorded action.

        Replaces the interim ``rt/wbc_token`` DDS wire: that path sampled the
        token and the DDS low_cmd through independent callbacks, so the pair
        could straddle a tick boundary.  Each g1_debug packet is built by the
        controller from ONE state-logger entry, so the 64-dim token and the
        low-level command fields in it are tick-consistent by construction
        (see WbcDebugSubscriber for the exact pairing convention — identical
        to the real-robot recording pipeline).  ``get_action`` attaches the
        same packet snapshot as both ``token`` and ``wbc_debug`` on the
        ActionCmd, which is what the recorder saves.  If the stream never
        publishes, the snapshot stays None and the recorder writes zeros(64).
        """
        self._latest_token = None  # test injection seam (checked before the stream)
        self._token_lock = threading.Lock()
        self._wbc_debug = None
        host = self.robot.sonic_config.get("WBC_DEBUG_HOST", "localhost")
        port = self.robot.sonic_config.get("WBC_DEBUG_PORT", 5557)
        topic = self.robot.sonic_config.get("WBC_DEBUG_TOPIC", "g1_debug")
        try:
            from .wbc_debug_subscriber import WbcDebugSubscriber
            self._wbc_debug = WbcDebugSubscriber(host, port, topic)
        except Exception as e:
            print(f"[PicoWbc] g1_debug subscriber inactive ({e}); recording zeros(64).")

    @property
    def latest_token(self):
        """Latest 64-dim token as a fresh float64 array, or None.

        Injection seam for tests: set ``agent._latest_token = np.zeros(64)`` (or
        any 64-vector) to exercise the record path directly.
        """
        with self._token_lock:
            if self._latest_token is not None:
                return self._latest_token.copy()
        if self._wbc_debug is None:
            return None
        pkt = self._wbc_debug.get_synced()
        return None if pkt is None or pkt["token"] is None else pkt["token"]

    # ------------------------------------------------------------------
    # pico_manager ctrl topic (replaces _poll_pico_buttons)
    # ------------------------------------------------------------------

    def _poll_ctrl(self) -> dict | None:
        """Non-blocking poll of the 'ctrl' ZMQ topic.

        Returns a decoded field dict (e.g. ``{"drop_robot": array([True])}``)
        or ``None`` if no message is waiting.
        """
        if not self._ctrl_socket.poll(timeout=0):
            return None
        raw = self._ctrl_socket.recv(zmq.NOBLOCK)
        return self._unpack_pose_message(raw[len("ctrl"):])

    @staticmethod
    def _unpack_pose_message(data: bytes) -> dict:
        """Deserialize a packed pose message (topic prefix already stripped).

        Format: [1280-byte JSON header (null-padded)][concatenated binary fields]
        """
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE
        
        header_bytes = data[:HEADER_SIZE]
        payload = data[HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))
        
        dtype_map = {
            "f32": np.float32,
            "f64": np.float64,
            "i32": np.int32,
            "i64": np.int64,
            "bool": bool,
        }
        
        result = {}
        offset = 0
        for field in header["fields"]:
            dtype = dtype_map[field["dtype"]]
            shape = field["shape"]
            count = int(np.prod(shape)) if shape else 1
            nbytes = count * np.dtype(dtype).itemsize
            arr = np.frombuffer(payload[offset: offset + nbytes], dtype=dtype).reshape(shape)
            result[field["name"]] = arr
            offset += nbytes
        
        return result

    def _poll_planner(self) -> None:
        if self._planner_socket.poll(timeout=0):
            raw = self._planner_socket.recv(zmq.NOBLOCK)
            self._latest_planner = self._unpack_pose_message(raw[len("planner"):])
            self._latest_planner_t = time.monotonic()

    @property
    def latest_planner(self) -> dict | None:
        """Freshest planner command from the manager (None before any arrives)."""
        return self._latest_planner

    # locomotion gaits the planner cycles through with A+B / X+Y
    _GAITS = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN", 4: "IDLE_SQUAT"}

    def _poll_manager_liveness(self) -> None:
        if self._live_socket.poll(timeout=0):
            raw = self._live_socket.recv(zmq.NOBLOCK)
            self._manager_t = time.monotonic()
            # Which topic is streaming tells us the operator's mode: the
            # manager publishes "pose" only in POSE (full-body SMPL) and
            # "planner" in PLANNER / VR_3PT.  Nothing else reveals it — its
            # own mode prints go to the manager's terminal.
            if raw.startswith(b"pose"):
                self._manager_mode = "POSE (full-body)"
            elif raw.startswith(b"planner"):
                self._manager_mode = "PLANNER/VR_3PT"

    #: crane length that puts the pelvis at standing height (anchor is 1.0 m)
    STAND_BAND_LENGTH =  -0.2 # -0.25

    @property
    def controller_holding(self) -> bool:
        """True when the SONIC controller is commanding with real position
        gains, i.e. it — not us — is holding the robot up."""
        if not self.engaged:
            return False
        lc = self.unitree_bridge.low_cmd
        return max(lc.motor_cmd[i].kp for i in range(29)) > 5.0

    @property
    def settle_progress(self) -> str:
        """How close the controller is to being trusted with the robot."""
        need = getattr(self, "settle_frames", 150)
        have = getattr(self, "_stiff_frames", 0)
        return f"settling {have / 50:.1f}/{need / 50:.1f}s"

    @property
    def manager_mode(self) -> str:
        """Operator mode inferred from the manager's topic, plus the gait."""
        if not self.engaged:
            return "OFF"
        mode = getattr(self, "_manager_mode", "?")
        planner = self._latest_planner
        if planner is not None and "mode" in planner and mode.startswith("PLANNER"):
            gait = int(np.asarray(planner["mode"]).ravel()[0])
            mode = f"{mode} gait={self._GAITS.get(gait, gait)}"
        return mode

    @property
    def engaged(self) -> bool:
        """True while the pico_manager is streaming, in ANY mode.

        The manager sits silent in OFF and only starts publishing once the
        operator engages with A+X, so traffic is the engagement signal.
        (Commands on the DDS bus are not: an idle controller publishes too.)
        """
        # 3 s, not 0.5 s: POSE mode only publishes once its body-tracking
        # buffer fills, so a switch into POSE (or a tracking hiccup) leaves a
        # gap that is not a disengagement.  A real stop is unmistakable
        # anyway — the manager exits.
        t = getattr(self, "_manager_t", None)
        return t is not None and (time.monotonic() - t) < 3.0

    @property
    def latest_planner_age(self) -> float:
        """Seconds since the last planner message (inf before any arrives).

        CONFLATE keeps only the newest message, so in POSE mode (planner loop
        silent) the cached command goes stale — consumers should check this."""
        t = getattr(self, "_latest_planner_t", None)
        return float("inf") if t is None else time.monotonic() - t

    # ------------------------------------------------------------------
    # Rendering / streaming
    # ------------------------------------------------------------------

    def update_render_caches(self, observation: dict):
        if self._streaming and self._streaming.is_running():
            self._push_stereo_frame(observation)
        return observation

    def _push_stereo_frame(self, observation: dict) -> None:
        left = observation.get("head_stereo_left")
        right = observation.get("head_stereo_right")
        if left is None or right is None:
            return
        left_bgr = np.ascontiguousarray(left[..., ::-1])
        right_bgr = np.ascontiguousarray(right[..., ::-1])
        # Draw debug text on top-right of each eye
        text = f"{self.episodes_saved}/{self.num_episodes}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.0, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = left_bgr.shape[1] - tw - 100
        y = th + 10
        cv2.putText(left_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
        cv2.putText(right_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
        # Recording indicator, drawn directly BELOW the episodes counter (the
        # headset crops the frame edges; the counter's spot is known-visible):
        #   yellow ARMED -> POSE teleop live (A+X pressed), record gate not
        #                   open yet
        #   red    REC   -> frames are actually being saved
        #   gray   IDLE  -> episode saved, waiting for the scene reset
        if self.rec_indicator:
            y_rec = y + th + 16  # one line below the counter
            color, label = {
                "rec": ((0, 0, 255), "REC"),
                "idle": ((180, 180, 180), "IDLE"),
            }.get(self.rec_indicator, ((0, 220, 255), "ARMED"))
            for eye in (left_bgr, right_bgr):
                cv2.circle(eye, (x + 12, y_rec - th // 2), 12, color, -1)
                cv2.putText(eye, label, (x + 32, y_rec), font, scale, color, thickness)
        stereo = np.concatenate([left_bgr, right_bgr], axis=1)
        self._frame_buffer.put(stereo)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def get_action(self, observation, instruction=None, **kwargs):

        self._poll_manager_liveness()
        self._poll_planner()

        # Stabilization phase (mirrors PicoDecoupledAgent.get_action): until
        # the floating base settles after spawn/reset, run the in-process
        # decoupled WBC toward its default pose instead of forwarding the
        # external SONIC controller's commands.
        if not self.robot.stabilized:
            return self.get_stabilize_action(kwargs["privileged_info"]["proprio"])
        
        if self._cached_target_q is not None:
            self._cached_target_q = None
            self._cached_left_hand_q = None
            self._cached_right_hand_q = None
            print("[PicoWbc] Robot stabilized — handing control to the SONIC controller")

        ctrl = self._poll_ctrl()
        if ctrl is not None:
            if bool(ctrl.get("reset_env", [False])[0]):
                self._reset_requested = True
                print("[PicoWbc] Environment reset requested")


        # ONE g1_debug snapshot for both token and reconstructed low cmd, so
        # everything the recorder saves comes from the same controller tick.
        wbc_debug = self._wbc_debug.get_synced() if self._wbc_debug is not None else None
        with self._token_lock:
            injected = self._latest_token  # test seam wins over the stream
        if injected is not None:
            token = injected.copy()
        else:
            token = wbc_debug["token"] if wbc_debug is not None else None

        return ActionCmd(
            "wbc_torque",
            low_cmd=self.unitree_bridge.low_cmd,
            use_sensor=self.unitree_bridge.use_sensor,
            left_hand_cmd=self.unitree_bridge.left_hand_cmd,
            right_hand_cmd=self.unitree_bridge.right_hand_cmd,
            token=token,  # 64-dim SONIC whole-body token (or None)
            wbc_debug=wbc_debug,  # tick-synced token + reconstructed LowCmd
        )

    def reset_policy(self):
        """Reset for a new episode.

        The SONIC controller runs in an external process — we cannot reset it
        from here.  The operator must re-engage it from the VR controllers.
        We only clear the bridge's command-freshness flags so stale commands
        from the previous episode are not treated as engagement.
        """
        self.unitree_bridge.reset()
        # re-capture the crane hold pose from the new episode's spawn stance
        self._hold_q = None
        self._catch_ticks = None
        # Reset the stabilization WBC so the next episode ramps to the default
        # pose from scratch (mirrors PicoDecoupledAgent.reset_policy).
        self._wbc_policy.reset(init_time=time.monotonic())
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        print("[PicoWbc] Bridge reset — re-engage the SONIC controller from the VR controllers")

    def cache_proprio(self, proprio):
        """Hand the freshest sim proprio to the heartbeat publisher.

        Called from the physics-substep hook, which runs on the sim thread —
        the only place mjData is safe to read.  The DDS send itself is done by
        the heartbeat thread, so the physics loop never blocks on the network."""
        with self._pub_lock:
            self._last_proprio = proprio

    def start_state_heartbeat(self):
        """Publish rt/lowstate from a side thread at a steady 200 Hz.

        The controller aborts if state is older than 500 ms, and env.reset()
        pauses physics long enough to trip that — so a dedicated 200 Hz beat
        (fed by cache_proprio) keeps state flowing across resets and off the
        physics critical path.
        """

        def _beat():
            while not self._hb_stop.is_set():
                with self._pub_lock:
                    proprio = self._last_proprio  # prepare_obs() returns a
                    # fresh dict each call, so this reference is safe to publish
                    # outside the lock — the sim thread only ever rebinds it.
                if proprio is not None:
                    self.unitree_bridge.PublishLowState(proprio)  # type:ignore
                    if self.unitree_bridge.joystick:  # type:ignore
                        self.unitree_bridge.PublishWirelessController()  # type:ignore
                time.sleep(0.005)

        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(target=_beat, daemon=True)
        self._hb_thread.start()

    def close(self):
        if hasattr(self, "_hb_stop"):
            self._hb_stop.set()
        if getattr(self, "_wbc_debug", None) is not None:
            self._wbc_debug.close()
        if hasattr(self, "_ctrl_socket"):
            self._ctrl_socket.close()
        if hasattr(self, "_planner_socket"):
            self._planner_socket.close()
        if hasattr(self, "_live_socket"):
            self._live_socket.close()
        if hasattr(self, "_ctrl_context"):
            self._ctrl_context.term()
