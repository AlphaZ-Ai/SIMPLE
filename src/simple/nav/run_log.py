# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Per-step log of a navigation run, for diagnosis and exact replay.

MuJoCo is the only physics in SIMPLE (IsaacSim just mirrors its state), so
saving the full MuJoCo qpos at every control step is enough to re-render a run
in IsaacSim afterwards (scripts/nav/render_nav_run.py) without re-simulating.
"""

import json

import numpy as np


class NavRunLog:
    def __init__(self, path: str, meta: dict):
        self.path = path
        self.meta = dict(meta)
        self.t, self.qpos, self.cmd, self.amo, self.contacts = [], [], [], [], []
        self.events = []

    def add(self, t: float, qpos: np.ndarray, cmd: dict, amo_command: list, house_contacts: list[str]):
        self.t.append(t)
        self.qpos.append(np.array(qpos, dtype=np.float64))
        self.cmd.append([cmd["vx"], cmd["vy"], cmd["wz"]])
        self.amo.append(list(amo_command))
        self.contacts.append(house_contacts)

    def event(self, t: float, text: str):
        self.events.append((t, text))

    def save(self):
        if not self.t:
            return
        np.savez_compressed(
            self.path,
            t=np.asarray(self.t),
            qpos=np.stack(self.qpos),
            cmd=np.asarray(self.cmd),
            amo=np.asarray(self.amo),
            contacts=np.array([json.dumps(c) for c in self.contacts]),
            events=np.array([json.dumps(e) for e in self.events]),
            meta=np.array(json.dumps(self.meta)),
        )
        print(f"[nav_sim] saved run log ({len(self.t)} steps) to {self.path}", flush=True)


def house_contacts(model, data, scene_body_id: int) -> list[str]:
    """Names of robot/other bodies currently touching the house collision boxes."""
    out = set()
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if scene_body_id in (b1, b2):
            other = b2 if b1 == scene_body_id else b1
            out.add(model.body(other).name)
    return sorted(out)
