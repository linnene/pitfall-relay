"""Minimal RCRS agents for the feasibility probe.

The dump agent connects one role, records its legal local world model for a
few timesteps, and rests.  It does not use server truth beyond what the
official perception gives this agent.
"""

import json
import os
import sys
import time
from multiprocessing import Process

from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN
from rcrs_core.connection.componentLauncher import ComponentLauncher
from rcrs_core.constants.constants import (
    DEFAULT_KERNEL_HOST_NAME,
    DEFAULT_KERNEL_PORT_NUMBER,
)
from rcrs_core.entities.area import Area
from rcrs_core.entities.blockade import Blockade
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.human import Human
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road


def as_int(value):
    if hasattr(value, "get_value"):
        value = value.get_value()
    return int(value)


def entity_snapshot(entity):
    name = entity.__class__.__name__
    item = {"id": as_int(entity.get_id()), "type": name}
    try:
        item["x"] = int(entity.get_x())
        item["y"] = int(entity.get_y())
    except Exception:
        pass
    if isinstance(entity, Human):
        item.update({
            "position": as_int(entity.get_position()),
            "hp": int(entity.get_hp()),
            "damage": int(entity.get_damage()),
            "buriedness": int(entity.get_buriedness()),
        })
    if isinstance(entity, Area):
        item["blockades"] = [as_int(b) for b in entity.get_blockades()]
    if isinstance(entity, Road):
        item["neighbours"] = [
            as_int(e.get_neighbour()) for e in entity.get_edges()
            if e.get_neighbour() is not None
        ]
    if isinstance(entity, Blockade):
        item["position"] = as_int(entity.get_position())
        item["repair_cost"] = int(entity.get_repaire_cost())
    if isinstance(entity, Building):
        item["fieryness"] = int(entity.get_fieryness())
        item["brokenness"] = int(entity.get_brokenness())
    if isinstance(entity, Refuge):
        item["bed_capacity"] = int(entity.get_bed_capacity())
        item["occupied_beds"] = int(entity.get_occupied_beds())
    return item


class DumpAgent(Agent):
    def __init__(self, role, out_path, max_steps=30):
        super().__init__(False)
        self.role = role
        self.out_path = out_path
        self.max_steps = max_steps
        self.name = role
        self.rows = []

    def get_requested_entities(self):
        return [getattr(URN.Entity, self.role)]

    def precompute(self):
        pass

    def think(self, time_step, change_set, heard):
        if time_step <= self.max_steps:
            entities = [entity_snapshot(e) for e in self.world_model.get_entities()]
            me = self.me()
            self.rows.append({
                "time": int(time_step),
                "agent_id": int(self.get_id().get_value()),
                "role": self.role,
                "self_position": as_int(me.get_position()),
                "self_x": int(me.get_x()),
                "self_y": int(me.get_y()),
                "self_hp": int(me.get_hp()),
                "self_damage": int(me.get_damage()),
                "heard": len(heard) if heard is not None else 0,
                "entities": entities,
            })
            if time_step == self.max_steps:
                with open(self.out_path, "w", encoding="utf-8") as fh:
                    json.dump(self.rows, fh, indent=2, sort_keys=True)
        self.send_rest(time_step)


def launch(role, out_path, port, host, max_steps):
    launcher = ComponentLauncher(port, host)
    agent = DumpAgent(role, out_path, max_steps=max_steps)
    launcher.connect(agent, launcher.generate_request_ID())


def main():
    role = os.environ["ROLE"]
    out_path = os.environ["OUT_PATH"]
    host = os.environ.get("KERNEL_HOST", DEFAULT_KERNEL_HOST_NAME)
    port = int(os.environ.get("KERNEL_PORT", DEFAULT_KERNEL_PORT_NUMBER))
    max_steps = int(os.environ.get("MAX_STEPS", "30"))
    launch(role, out_path, port, host, max_steps)


if __name__ == "__main__":
    main()
