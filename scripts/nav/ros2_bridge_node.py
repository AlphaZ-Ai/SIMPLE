# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
ROS2 side of the SIMPLE navigation bridge. Replaces the Unity simulator +
vehicle_simulator of the CMU autonomy stack (Navigation-Physical-Experiment):

    publishes   /registered_scan   sensor_msgs/PointCloud2  (frame "map", x y z intensity)
                /state_estimation  nav_msgs/Odometry        (frame "map" -> "sensor")
                TF map -> sensor
    subscribes  /cmd_vel           geometry_msgs/TwistStamped (body frame)
                /way_point, /goal_point  (forwarded to the sim's run log)

Runs on the system Python that ROS2 Jazzy's rclpy is built for (not SIMPLE's
venv); talks to scripts/nav/run_nav_sim.py over a local socket.

    source ~/ros_jazzy.sh
    python3 scripts/nav/ros2_bridge_node.py
"""

import argparse
import json
import threading
import time
from multiprocessing.connection import Client

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster

FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


class SimpleNavBridge(Node):
    def __init__(self, conn):
        super().__init__("simple_nav_bridge")
        self.conn = conn
        self.send_lock = threading.Lock()
        self.pub_scan = self.create_publisher(PointCloud2, "/registered_scan", 5)
        self.pub_odom = self.create_publisher(Odometry, "/state_estimation", 5)
        self.tf = TransformBroadcaster(self)
        # the local planner heads for its goalX/goalY parameter (0, 0) until a
        # waypoint arrives; give it the robot's own position so it stands still
        self.pub_waypoint = self.create_publisher(PointStamped, "/way_point", 5)
        self.hold_scans = 30  # ~3 s, before any goal is sent
        self.create_subscription(TwistStamped, "/cmd_vel", self.on_cmd_vel, 5)
        # forwarded only so the sim's run log records what the planners asked for
        self.create_subscription(PointStamped, "/way_point", lambda m: self.forward_point("waypoint", m), 5)
        self.create_subscription(PointStamped, "/goal_point", lambda m: self.forward_point("goal", m), 5)
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()
        self.get_logger().info("connected to SIMPLE nav sim")

    def on_cmd_vel(self, msg: TwistStamped):
        cmd = {"type": "cmd_vel", "vx": msg.twist.linear.x, "vy": msg.twist.linear.y, "wz": msg.twist.angular.z}
        with self.send_lock:
            self.conn.send_bytes(json.dumps(cmd).encode())

    def forward_point(self, kind, msg: PointStamped):
        with self.send_lock:
            self.conn.send_bytes(json.dumps({"type": kind, "x": msg.point.x, "y": msg.point.y}).encode())

    def read_loop(self):
        while rclpy.ok():
            try:
                header = json.loads(self.conn.recv_bytes())
                raw = self.conn.recv_bytes()
            except (EOFError, ConnectionResetError, OSError):
                self.get_logger().error("SIMPLE nav sim disconnected")
                rclpy.try_shutdown()
                return
            self.publish(header, np.frombuffer(raw, dtype=np.float32).reshape(-1, 3))

    def publish(self, header, xyz):
        stamp = self.get_clock().now().to_msg()  # sim runs in real time
        x, y, z, qw, qx, qy, qz = header["pose"]

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "map", "sensor"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = x, y, z
        o = odom.pose.pose.orientation
        o.w, o.x, o.y, o.z = qw, qx, qy, qz
        t = header["twist"]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = t[0], t[1], t[2]
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = t[3], t[4], t[5]
        self.pub_odom.publish(odom)
        if self.hold_scans > 0 and self.count_subscribers("/way_point") > 0:
            hold = PointStamped()
            hold.header.stamp, hold.header.frame_id = stamp, "map"
            hold.point.x, hold.point.y, hold.point.z = x, y, z
            self.pub_waypoint.publish(hold)
            self.hold_scans -= 1

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "map", "sensor"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
        r = tf.transform.rotation
        r.w, r.x, r.y, r.z = qw, qx, qy, qz
        self.tf.sendTransform(tf)

        cloud = np.zeros((len(xyz), 4), dtype=np.float32)
        cloud[:, :3] = xyz
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = stamp, "map"
        msg.height, msg.width = 1, len(xyz)
        msg.fields, msg.is_bigendian, msg.is_dense = FIELDS, False, True
        msg.point_step, msg.row_step = 16, 16 * len(xyz)
        msg.data = cloud.tobytes()
        self.pub_scan.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--authkey", default="simple-nav")
    args = ap.parse_args()

    while True:
        try:
            conn = Client(("127.0.0.1", args.port), authkey=args.authkey.encode())
            break
        except ConnectionRefusedError:
            print("waiting for scripts/nav/run_nav_sim.py ...", flush=True)
            time.sleep(1.0)

    rclpy.init()
    node = SimpleNavBridge(conn)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        conn.close()


if __name__ == "__main__":
    main()
