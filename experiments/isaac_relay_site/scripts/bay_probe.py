"""Development probe: can a narrow docking bay produce a native Nav2 abort?

Three robots drive straight into three parallel dead-end bays of decreasing
clear width.  The probe does not cancel goals, does not toggle obstacles at
runtime, and does not compress the budget.  A native Nav2 ABORTED is the only
accepted failure evidence; a timeout is reported as a timeout, not a failure.
"""

import argparse
import json
import math
import os
import sys
import time

from isaacsim import SimulationApp

SIM_APP = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np
import omni.usd
from action_msgs.msg import GoalStatus
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

sys.path.insert(0, "/work")
from flexnav import config, mapgen, nav2_stack, robots, scene  # noqa: E402
from flexnav.ros_client import FleetRos, quaternion_to_yaw  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
SIM_APP.update()

OUT_DIR = os.environ.get("RUN_OUT", "/work/out/bay_probe")
RENDER = False

STATUS_NAME = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def sim_time(world):
    return float(world.current_time)


def configure_layout(widths=None, shape="straight", single=False):
    config.ROOM_HALF_X = 7.0
    config.ROOM_HALF_Y = 5.0
    config.WALL_THICKNESS = 0.2
    config.WALL_HEIGHT = 1.5
    config.MAP_RESOLUTION = 0.05
    config.MAP_ORIGIN = (-config.ROOM_HALF_X, -config.ROOM_HALF_Y)
    config.ROBOT_NAMES = ["robot0"] if single else ["robot0", "robot1", "robot2"]
    config.LANES = {"robot0": 0.0} if single else {
        "robot0": -3.0, "robot1": 0.0, "robot2": 3.0}
    widths = widths or [1.8, 1.4, 1.2]
    config.BAY_WIDTHS = dict(zip(config.ROBOT_NAMES, widths))
    config.BAY_SHAPE = shape
    config.ROBOT_STARTS = {n: (-4.5, y) for n, y in config.LANES.items()}
    walls = [
        (0.0, 5.0, 14.0, 0.2),
        (0.0, -5.0, 14.0, 0.2),
        (7.0, 0.0, 0.2, 10.0),
        (-7.0, 0.0, 0.2, 10.0),
    ]
    config.GOALS = {}
    for name, lane in config.LANES.items():
        width = config.BAY_WIDTHS[name]
        if shape == "l":
            # L-shaped corridor: enter east, turn north at the end.
            half = width / 2.0
            south_x0, south_x1 = 2.0, 4.5 + half + 0.2
            north_x0, north_x1 = 2.0, 4.5 - half
            walls.extend([
                ((south_x0 + south_x1) / 2.0, lane - half - 0.1,
                 south_x1 - south_x0, 0.2),
                ((north_x0 + north_x1) / 2.0, lane + half + 0.1,
                 north_x1 - north_x0, 0.2),
                (4.5 - half - 0.1, lane + (half + 1.9) / 2.0,
                 0.2, 1.9 - half),
                (4.5 + half + 0.1, lane + (-half + 1.9) / 2.0,
                 0.2, 1.9 + half),
                (4.5, lane + 1.9, width + 0.4, 0.2),
            ])
            config.GOALS[name] = (4.5, lane + 1.2)
        else:
            side_y = width / 2.0 + 0.1
            walls.append((3.5, lane + side_y, 3.0, 0.2))
            walls.append((3.5, lane - side_y, 3.0, 0.2))
            walls.append((5.1, lane, 0.2, width + 0.4))
            config.GOALS[name] = (4.2, lane)
    config.STATIC_WALLS = walls
    # scene.py expects these hidden dynamic props; keep them outside the room.
    config.DETOUR_OBSTACLE = ("detour", 20.0, 20.0, 0.2, 0.2)
    config.BLOCK_FRONT = ("block_front", 20.0, 21.0, 0.2, 0.2)
    config.BLOCK_REAR = ("block_rear", 20.0, 22.0, 0.2, 0.2)
    config.GOAL_TIMEOUT_S = 240.0
    config.NAV2_START_TIMEOUT_S = 180.0


class Probe:
    def __init__(self, goal_yaw=0.0):
        self.goal_yaw = goal_yaw
        os.makedirs(OUT_DIR, exist_ok=True)
        self.events = []
        self.world = World(
            stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0
        )
        self.room = scene.RoomScene(self.world)
        self.room.build()
        robots.add_clock_graph()
        self.names = list(config.ROBOT_NAMES)
        self.info = {}
        for name in self.names:
            x, y = config.ROBOT_STARTS[name]
            self.info[name] = robots.add_robot(self.world, name, x, y)
        self.map_yaml = mapgen.write_map(os.path.join(OUT_DIR, "map"))
        self.stacks = {}
        self.ros = None
        self.step_count = 0

    def log(self, kind, **fields):
        rec = {"event": kind, "sim_t": round(sim_time(self.world), 3),
               "wall_t": time.time()}
        rec.update(fields)
        self.events.append(rec)
        print("EVENT", json.dumps(rec, ensure_ascii=False), flush=True)

    def step(self):
        self.ros.spin(0.0)
        now_wall = time.time()
        for name in self.names:
            t = self.ros.cmd_vel[name]
            terminal = (
                self.ros.actions.get(name) is not None
                and self.ros.actions[name]["result"] is not None
            )
            if terminal or now_wall - self.ros.cmd_stamp[name] > 1.0:
                t = type(t)()
            action = self.info[name]["controller"].forward(
                np.array([float(t.linear.x), float(t.angular.z)])
            )
            self.info[name]["robot"].apply_wheel_actions(action)
        self.world.step(render=RENDER)
        for name in self.names:
            x, y, yaw = self.pose(name)
            self.ros.publish_state(name, x, y, yaw, 1.0 / 60.0)
        self.ros.spin(0.0)
        self.step_count += 1

    def pose(self, name):
        prim = self.world.stage.GetPrimAtPath(f"/World/{name}/chassis_link")
        mat = omni.usd.get_world_transform_matrix(prim)
        t = mat.ExtractTranslation()
        q = mat.ExtractRotationQuat()
        yaw = math.atan2(
            2.0 * (q.GetReal() * q.GetImaginary()[2] + q.GetImaginary()[0] * q.GetImaginary()[1]),
            1.0 - 2.0 * (q.GetImaginary()[1] ** 2 + q.GetImaginary()[2] ** 2),
        )
        return float(t[0]), float(t[1]), float(yaw)

    def start(self):
        self.world.reset()
        self.world.play()
        for name in self.names:
            robots.set_lidar_enabled(name, False)
        self.ros = FleetRos(self.names, starts=config.ROBOT_STARTS)
        nav2_dir = os.path.join(OUT_DIR, "nav2")
        nodes = [
            "controller_server", "smoother_server", "planner_server",
            "behavior_server", "bt_navigator", "waypoint_follower",
            "velocity_smoother", "map_server",
        ]
        for name in self.names:
            params = nav2_stack.render_params(name, nav2_dir)
            stack = nav2_stack.Nav2Stack(
                name, params, self.map_yaml,
                os.path.join(OUT_DIR, "logs", f"nav2_{name}.log"),
            )
            stack.start()
            self.stacks[name] = stack
            status = nav2_stack.wait_for_lifecycle(
                self.ros.node, [name], nodes,
                config.NAV2_START_TIMEOUT_S, pump=self.step,
            )
            pending = [k for k, v in status.items() if v != 3]
            if pending:
                for _, node_name in pending:
                    state = nav2_stack.force_activate(
                        self.ros.node, name, node_name, timeout_s=40.0
                    )
                    self.log("lifecycle_forced", robot=name, node=node_name,
                             state=state)
                status = nav2_stack.wait_for_lifecycle(
                    self.ros.node, [name], nodes, timeout_s=30.0, pump=self.step
                )
                pending = [k for k, v in status.items() if v != 3]
                if pending:
                    raise RuntimeError(f"Nav2 lifecycle failed for {name}: {pending}")
            if not self.ros.action_ready(name, timeout_s=30.0):
                raise RuntimeError(f"{name} action server unavailable")
            time.sleep(2.0)
        for name in self.names:
            robots.set_lidar_enabled(name, True)
        time.sleep(1.0)
        self.log("nav2_ready", sim_t=sim_time(self.world))

    def costmap_value(self, name, wx, wy):
        msg = self.ros.costmap[name]
        if msg is None:
            return None
        info = msg.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if col < 0 or row < 0 or col >= info.width or row >= info.height:
            return None
        return int(msg.data[row * info.width + col])

    def run(self, timeout_s):
        goals = {name: tuple(config.GOALS[name]) for name in self.names}
        for name, (gx, gy) in goals.items():
            self.ros.send_goal(name, gx, gy, self.goal_yaw)
        self.log("goals_sent", goals={n: list(g) for n, g in goals.items()},
                 yaw=self.goal_yaw)
        deadline = sim_time(self.world) + timeout_s
        while sim_time(self.world) < deadline:
            self.step()
            self.ros.poll_actions()
            if all(
                self.ros.actions[n] is not None
                and self.ros.actions[n]["result"] is not None
                for n in self.names
            ):
                break
        rows = {}
        for name in self.names:
            rec = self.ros.actions[name]
            px, py, pyaw = self.pose(name)
            rows[name] = {
                "goal": goals[name],
                "bay_clear_width_m": config.BAY_WIDTHS[name],
                "status": STATUS_NAME.get(rec["status"], str(rec["status"])) if rec else None,
                "result_received": bool(rec and rec["result"] is not None),
                "cancelled_by_client": bool(rec and rec["cancelled_by_client"]),
                "final_pose_world": [round(px, 3), round(py, 3), round(pyaw, 3)],
                "xy_error_m": round(math.hypot(px - goals[name][0], py - goals[name][1]), 3),
                "sim_t": round(sim_time(self.world), 3),
                "plan_msgs": self.ros.plan_count[name],
                "cmd_vel_msgs": self.ros.cmd_count[name],
                "costmap_entrance": self.costmap_value(name, 2.2, config.LANES[name]),
                "costmap_goal": self.costmap_value(name, 4.2, config.LANES[name]),
            }
        out = {"layout": "bay_probe", "widths": config.BAY_WIDTHS,
               "robots": rows, "events": self.events}
        path = os.path.join(OUT_DIR, "bay_probe.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print("RESULT", json.dumps(out, indent=2, sort_keys=True), flush=True)
        return out

    def shutdown(self):
        for stack in self.stacks.values():
            stack.stop()
        if self.ros is not None:
            self.ros.shutdown()
        try:
            self.world.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--widths", default="1.8,1.4,1.2",
                        help="comma-separated bay clear widths for robot0/1/2")
    parser.add_argument("--shape", choices=["straight", "l"], default="straight")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--goal-yaw", type=float, default=None)
    args = parser.parse_args()
    widths = [float(x) for x in args.widths.split(",")]
    if args.single:
        widths = widths[:1]
    configure_layout(widths, args.shape, single=args.single)
    goal_yaw = args.goal_yaw
    if goal_yaw is None:
        goal_yaw = math.pi / 2.0 if args.shape == "l" else 0.0
    probe = Probe(goal_yaw=goal_yaw)
    try:
        probe.start()
        probe.run(args.timeout)
    finally:
        probe.shutdown()


if __name__ == "__main__":
    main()
