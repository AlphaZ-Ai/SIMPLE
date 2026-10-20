# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""
Bridge test client without ROS: turn toward a goal and walk to it using only
the scan/odometry messages and cmd_vel, exactly like the ROS bridge would.
Pure stdlib, so it runs on any Python.

    python scripts/nav/goal_client_test.py --goal -2.5 -1.0
"""

import argparse
import json
import math
import struct
import time
from multiprocessing.connection import Client


def yaw_of(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--authkey", default="simple-nav")
    ap.add_argument("--speed", type=float, default=0.25, help="m/s")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    conn = Client(("127.0.0.1", args.port), authkey=args.authkey.encode())
    gx, gy = args.goal
    t_start, last_print, start = time.time(), 0.0, None
    while True:
        header = json.loads(conn.recv_bytes())
        pts = conn.recv_bytes()
        x, y, z, qw, qx, qy, qz = header["pose"]
        if start is None:
            start = (x, y)
        yaw = yaw_of(qw, qx, qy, qz)
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        if dist < 0.25:
            vx, wz = 0.0, 0.0
        else:
            wz = max(-0.6, min(0.6, 1.5 * err))
            vx = args.speed if abs(err) < 0.35 else 0.0  # turn in place first
        conn.send_bytes(json.dumps({"type": "cmd_vel", "vx": vx, "vy": 0.0, "wz": wz}).encode())

        now = time.time() - t_start
        if now - last_print > 2.0:
            n = header["n"]
            print(f"t={now:5.1f}s pos=({x:+.2f},{y:+.2f}) z={z:.2f} yaw={math.degrees(yaw):+6.1f} "
                  f"dist={dist:.2f} cmd vx={vx:.2f} wz={wz:+.2f} scan={n} pts ({len(pts)//12})", flush=True)
            last_print = now
        if dist < 0.25:
            print(f"REACHED goal in {now:.1f}s (start {start[0]:+.2f},{start[1]:+.2f}) z={z:.2f}", flush=True)
            break
        if now > args.timeout:
            print(f"TIMEOUT at dist {dist:.2f} z={z:.2f}", flush=True)
            break
    conn.close()


if __name__ == "__main__":
    main()
