"""Minimal scripted RCRS agents for one native clear/transport schedule.

A police agent clears blockades on the ambulance route.  An ambulance either
waits for a fixed start step (sequential schedule) or starts immediately
(early schedule), then rescues and delivers a civilian.  Agents use only their
own world model, never server truth.
"""

import json
import os
import sys
from collections import deque

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


class BaseAgent(Agent):
    def __init__(self, role, out_path, start_step=3):
        super().__init__(False)
        self.role = role
        self.out_path = out_path
        self.start_step = start_step
        self.name = role
        self.rows = []
        self.commands = []

    def get_requested_entities(self):
        return [getattr(URN.Entity, self.role)]

    def precompute(self):
        pass

    def log(self, time_step, event, **fields):
        row = {"time": int(time_step), "event": event}
        row.update(fields)
        self.rows.append(row)

    def command(self, time_step, kind, **fields):
        row = {"time": int(time_step), "command": kind}
        row.update(fields)
        self.commands.append(row)

    def finish(self):
        with open(self.out_path, "w", encoding="utf-8") as fh:
            json.dump({"role": self.role, "start_step": self.start_step,
                       "rows": self.rows, "commands": self.commands},
                      fh, indent=2, sort_keys=True)

    def area(self):
        return self.world_model.get_entity(self.me().get_position())

    def neighbours(self, area):
        out = []
        for edge in area.get_edges():
            n = edge.get_neighbour()
            if n is None or as_int(n) == 0:
                continue
            target = self.world_model.get_entity(n)
            if isinstance(target, (Road, Building)):
                if isinstance(target, Road) and target.get_blockades():
                    continue
                out.append(target)
        return out

    def path_to(self, target_id):
        start = self.area()
        if start is None:
            return None
        if as_int(start.get_id()) == target_id:
            return []
        target_id = int(target_id)
        queue = deque([(start, [])])
        seen = {as_int(start.get_id())}
        while queue:
            area, path = queue.popleft()
            for nxt in self.neighbours(area):
                nid = as_int(nxt.get_id())
                if nid in seen:
                    continue
                new_path = path + [nid]
                if nid == target_id:
                    return new_path
                seen.add(nid)
                queue.append((nxt, new_path))
        return None

    def move_to(self, time_step, target_id):
        path = self.path_to(target_id)
        if path is None:
            self.log(time_step, "no_path", target=target_id,
                     position=as_int(self.me().get_position()))
            self.send_rest(time_step)
            return False
        if not path:
            return True
        self.command(time_step, "move", target=int(target_id), path=path)
        self.send_move(time_step, path)
        return False


class PoliceAgent(BaseAgent):
    def __init__(self, out_path, start_step=3):
        super().__init__("POLICE_FORCE", out_path, start_step)
        self.route_roads = [279, 266, 971, 976, 975, 974]
        self.done = False
        self.route_seen = False

    def blockades(self):
        out = []
        for entity in self.world_model.get_entities():
            if isinstance(entity, Blockade):
                out.append(entity)
        return out

    def think(self, time_step, change_set, heard):
        if time_step < self.start_step:
            self.send_rest(time_step)
            return
        blockades = self.blockades()
        if blockades:
            self.route_seen = True
            # Prefer blockades on the ambulance route or the current road.
            current = as_int(self.me().get_position())
            target = None
            for b in blockades:
                pos = as_int(b.get_position())
                if pos in self.route_roads or pos == current:
                    target = b
                    break
            if target is not None:
                road = as_int(target.get_position())
                if current != road:
                    self.move_to(time_step, road)
                    return
                self.command(time_step, "clear", blockade=as_int(target.get_id()),
                             road=road, repair_cost=int(target.get_repaire_cost()))
                self.send_clear(time_step, target.get_id())
                return
        # Do not declare the route clear before collapse blockades have had a
        # chance to appear in local perception.
        if not self.done and time_step >= 25 and not blockades:
            self.done = True
            self.log(time_step, "route_clear",
                     blockades_known=0,
                     position=as_int(self.me().get_position()))
            self.send_say(time_step, "ROUTE_CLEAR")
            self.finish()
        self.send_rest(time_step)


class AmbulanceAgent(BaseAgent):
    def __init__(self, out_path, start_step=3):
        super().__init__("AMBULANCE_TEAM", out_path, start_step)
        self.target_building = 254
        self.refuge = 255
        self.loaded = None
        self.delivered = False
        self.started = False

    def civilians_at(self, building_id):
        out = []
        for entity in self.world_model.get_entities():
            if isinstance(entity, Civilian) and as_int(entity.get_position()) == building_id:
                out.append(entity)
        return out

    def think(self, time_step, change_set, heard):
        if time_step < self.start_step:
            self.send_rest(time_step)
            return
        if not self.started:
            self.started = True
            self.log(time_step, "ambulance_start",
                     position=as_int(self.me().get_position()))
        if self.delivered:
            self.send_rest(time_step)
            return
        current = as_int(self.me().get_position())
        if self.loaded is None:
            civilians = self.civilians_at(current)
            if current == self.target_building and civilians:
                target = civilians[0]
                cid = as_int(target.get_id())
                if int(target.get_buriedness()) > 0:
                    self.command(time_step, "rescue", civilian=cid,
                                 buriedness=int(target.get_buriedness()))
                    self.send_rescue(time_step, target.get_id())
                    return
                self.command(time_step, "load", civilian=cid)
                self.send_load(time_step, target.get_id())
                self.loaded = cid
                self.log(time_step, "loaded", civilian=cid)
                return
            if current != self.target_building:
                self.move_to(time_step, self.target_building)
                return
            self.send_rest(time_step)
            return
        if current != self.refuge:
            self.move_to(time_step, self.refuge)
            return
        self.command(time_step, "unload", civilian=self.loaded)
        self.send_unload(time_step)
        self.delivered = True
        self.log(time_step, "delivered", civilian=self.loaded,
                 position=current)
        self.finish()
        self.send_rest(time_step)


class FireAgent(BaseAgent):
    def __init__(self, out_path, start_step=3):
        super().__init__("FIRE_BRIGADE", out_path, start_step)

    def think(self, time_step, change_set, heard):
        if time_step == self.start_step:
            self.log(time_step, "fire_idle", reason="no_active_fire_observed")
            self.finish()
        self.send_rest(time_step)


def launch(role, out_path, start_step, port, host):
    launcher = ComponentLauncher(port, host)
    cls = {"POLICE_FORCE": PoliceAgent,
           "AMBULANCE_TEAM": AmbulanceAgent,
           "FIRE_BRIGADE": FireAgent}[role]
    agent = cls(out_path, start_step=start_step)
    launcher.connect(agent, launcher.generate_request_ID())


def main():
    role = os.environ["ROLE"]
    out_path = os.environ["OUT_PATH"]
    start_step = int(os.environ.get("START_STEP", "3"))
    host = os.environ.get("KERNEL_HOST", DEFAULT_KERNEL_HOST_NAME)
    port = int(os.environ.get("KERNEL_PORT", DEFAULT_KERNEL_PORT_NUMBER))
    launch(role, out_path, start_step, port, host)


if __name__ == "__main__":
    main()
