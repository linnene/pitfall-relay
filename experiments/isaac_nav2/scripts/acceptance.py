"""Isaac Sim + Nav2 acceptance runner.

Modes:
    --mode smoke     one robot, one goal, validates action -> wheels -> result
    --mode full      cases 1-3 in one simulation session
    --mode case4     fresh reset, permuted goals, interface/cross-talk checks
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

from isaacsim import SimulationApp

SIM_APP = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np
import omni.usd
from action_msgs.msg import GoalStatus
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

sys.path.insert(0, "/work")
from flexnav import config, mapgen, nav2_stack, robots, scene
from flexnav.ros_client import FleetRos, quaternion_to_yaw

enable_extension("isaacsim.ros2.bridge")
SIM_APP.update()

OUT_DIR = os.environ.get("RUN_OUT", "/work/out/run")
RENDER = "--no-render" not in sys.argv

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


class Acceptance:
    def __init__(self, mode):
        self.mode = mode
        self.out_dir = OUT_DIR
        os.makedirs(self.out_dir, exist_ok=True)
        self.events = []
        self.rounds = []
        self.world = World(
            stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0
        )
        self.room = scene.RoomScene(self.world)
        self.room.build()
        robots.add_clock_graph()
        self.names = ["robot1"] if mode == "smoke" else list(config.ROBOT_NAMES)
        self.info = {}
        for name in self.names:
            x, y = config.ROBOT_STARTS[name]
            self.info[name] = robots.add_robot(self.world, name, x, y)
        self.map_yaml = mapgen.write_map(os.path.join(self.out_dir, "map"))
        self.stacks = {}
        self.ros = None
        self.path_samples = {n: [] for n in self.names}
        self.last_obs = {n: None for n in self.names}
        self.step_count = 0
        self.probes = []

    # --- infrastructure -------------------------------------------------
    def start(self):
        self.world.reset()
        self.world.play()
        # Lidar rendering is not needed for lifecycle bringup; turning it off
        # keeps DDS service responses from missing their deadline under load.
        for name in self.names:
            robots.set_lidar_enabled(name, False)
        self.ros = FleetRos(self.names, starts=config.ROBOT_STARTS)
        nav2_dir = os.path.join(self.out_dir, "nav2")
        nav2_nodes = [
            "controller_server", "smoother_server", "planner_server",
            "behavior_server", "bt_navigator", "waypoint_follower",
            "velocity_smoother", "map_server",
        ]
        all_status = {}
        # Start stacks one at a time: three simultaneous lifecycle bringups
        # overload the single DDS/CPU burst and make configure calls time out.
        for name in self.names:
            params = nav2_stack.render_params(name, nav2_dir)
            stack = nav2_stack.Nav2Stack(
                name, params, self.map_yaml,
                os.path.join(self.out_dir, "logs", f"nav2_{name}.log"),
            )
            stack.start()
            self.stacks[name] = stack
            status = nav2_stack.wait_for_lifecycle(
                self.ros.node,
                [name],
                nav2_nodes,
                config.NAV2_START_TIMEOUT_S,
                pump=self.step,
            )
            all_status.update(status)
            self.log_event(
                "nav2_lifecycle_one",
                robot=name,
                status={f"{k[0]}/{k[1]}": v for k, v in status.items()},
            )
            pending = [k for k, v in status.items() if v != 3]
            if pending:
                # The lifecycle manager gives up when a change_state response
                # misses its deadline; retry the transition directly.
                for _, node_name in pending:
                    state = nav2_stack.force_activate(
                        self.ros.node, name, node_name, timeout_s=40.0
                    )
                    self.log_event(
                        "lifecycle_forced", robot=name, node=node_name, state=state
                    )
                status = nav2_stack.wait_for_lifecycle(
                    self.ros.node, [name],
                    nav2_nodes,
                    timeout_s=30.0, pump=self.step,
                )
                pending = [k for k, v in status.items() if v != 3]
                if pending:
                    raise RuntimeError(f"Nav2 lifecycle did not activate for {name}: {pending}")
            if not self.ros.action_ready(name, timeout_s=30.0):
                raise RuntimeError(f"{name} navigate_to_pose action server unavailable")
            time.sleep(2.0)
        for name in self.names:
            robots.set_lidar_enabled(name, True)
        time.sleep(1.0)
        self.log_event("nav2_ready", sim_t=sim_time(self.world))

    def shutdown(self):
        for stack in self.stacks.values():
            stack.stop()
        if self.ros is not None:
            self.ros.shutdown()
            self.ros = None
        try:
            self.world.stop()
        except Exception:
            pass

    def schedule_probe(self, delay_s, label, fn):
        self.probes.append(
            {"due": sim_time(self.world) + delay_s, "label": label, "fn": fn, "done": False}
        )

    def run_probes(self):
        for probe in self.probes:
            if probe["done"] or sim_time(self.world) < probe["due"]:
                continue
            probe["done"] = True
            try:
                fields = probe["fn"]()
            except Exception as exc:  # a probe must never break the run
                fields = {"probe_error": repr(exc)}
            self.log_event("probe", label=probe["label"], **fields)

    def log_event(self, kind, **fields):
        rec = {"event": kind, "sim_t": round(sim_time(self.world), 3), "wall_t": time.time()}
        rec.update(fields)
        self.events.append(rec)
        print("EVENT", json.dumps(rec, ensure_ascii=False), flush=True)

    # --- simulation step ------------------------------------------------
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
        if self.step_count % 10 == 0:
            for name in self.names:
                p, q = self.info[name]["robot"].get_world_pose()
                self.path_samples[name].append(
                    {
                        "t": round(sim_time(self.world), 3),
                        "x": round(float(p[0]), 4),
                        "y": round(float(p[1]), 4),
                        "yaw": round(float(quaternion_to_yaw(q)), 4),
                    }
                )

    def pose(self, name):
        # base_link is the chassis rigid body, not the USD root: the two have
        # a fixed local offset and Nav2 plans in the base_link frame.
        prim = self.world.stage.GetPrimAtPath(f"/World/{name}/chassis_link")
        mat = omni.usd.get_world_transform_matrix(prim)
        t = mat.ExtractTranslation()
        q = mat.ExtractRotationQuat()
        yaw = math.atan2(
            2.0 * (q.GetReal() * q.GetImaginary()[2] + q.GetImaginary()[0] * q.GetImaginary()[1]),
            1.0 - 2.0 * (q.GetImaginary()[1] ** 2 + q.GetImaginary()[2] ** 2),
        )
        return float(t[0]), float(t[1]), float(yaw)

    # --- mission --------------------------------------------------------
    def send_goals(self, goals, yaw=0.0):
        sent = {}
        for name, (gx, gy) in goals.items():
            rec = self.ros.send_goal(name, gx, gy, yaw)
            sent[name] = rec
        self.log_event("goals_sent", goals={n: list(g) for n, g in goals.items()}, yaw=yaw)
        return sent

    def results_done(self):
        done = self.ros.poll_actions()
        for name in done:
            rec = self.ros.actions[name]
            result = rec["result"]
            rec["final_pose_at_result"] = self.pose(name)
            self.log_event(
                "action_result",
                robot=name,
                status=STATUS_NAME.get(rec["status"], str(rec["status"])),
                error_code=None,  # Humble NavigateToPose.Result is std_msgs/Empty
                error_msg=str(getattr(result, "error_msg", "")) if result else None,
                cancelled_by_client=rec["cancelled_by_client"],
            )
        return done

    def run_round(self, label, goals, timeout_s, trigger=None, yaw=0.0):
        self.send_goals(goals, yaw=yaw)
        t0 = sim_time(self.world)
        deadline = t0 + timeout_s
        trigger_fired = False
        while sim_time(self.world) < deadline:
            self.step()
            self.results_done()
            self.run_probes()
            if trigger is not None and not trigger_fired:
                if trigger(self):
                    trigger_fired = True
            if all(
                self.ros.actions[n] is not None and self.ros.actions[n]["result"] is not None
                for n in self.names
            ):
                break
        timed_out = []
        for name in self.names:
            rec = self.ros.actions[name]
            if rec is None or rec["result"] is None:
                timed_out.append(name)
                if rec is not None and rec["handle"] is not None:
                    rec["cancelled_by_client"] = True
                    rec["handle"].cancel_goal_async()
                    self.log_event("external_cancel", robot=name, reason="round_timeout")
        round_rec = self.summarize_round(label, goals)
        round_rec["timed_out"] = timed_out
        self.rounds.append(round_rec)
        print("ROUND", json.dumps(round_rec, ensure_ascii=False, indent=2), flush=True)
        return round_rec

    def summarize_round(self, label, goals):
        out = {"label": label, "robots": {}}
        for name in self.names:
            rec = self.ros.actions[name]
            gx, gy = goals[name]
            px, py, pyaw = rec.get("final_pose_at_result", self.pose(name)) if rec else self.pose(name)
            px2, py2, pyaw2 = self.pose(name)
            goal_yaw = rec["goal"][2] if rec else 0.0
            result = rec["result"] if rec else None
            out["robots"][name] = {
                "goal": [gx, gy],
                "status": STATUS_NAME.get(rec["status"], str(rec["status"])) if rec else None,
                "error_code": None,  # Humble NavigateToPose.Result is std_msgs/Empty
                "error_msg": str(getattr(result, "error_msg", "")) if result else None,
                "cancelled_by_client": bool(rec["cancelled_by_client"]) if rec else None,
                "final_pose_world": [round(px, 3), round(py, 3), round(pyaw, 3)],
                "pose_world_after_round": [round(px2, 3), round(py2, 3), round(pyaw2, 3)],
                "xy_error_m": round(math.hypot(px - gx, py - gy), 3),
                "yaw_error_rad": round(
                    abs(math.atan2(math.sin(pyaw - goal_yaw), math.cos(pyaw - goal_yaw))), 3
                ),
                "feedback_samples": len(rec["feedback"]) if rec else 0,
                "last_feedback": rec["feedback"][-1] if rec and rec["feedback"] else None,
                "cmd_vel_msgs": self.ros.cmd_count[name],
                "plan_msgs": self.ros.plan_count[name],
                "odom_msgs": self.ros.odom_count[name],
                "scan_msgs": self.ros.scan_count[name],
                "costmap_msgs": self.ros.costmap_count[name],
            }
        return out

    def capture_ros_graph(self):
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = (
            "/opt/ros/humble/lib/python3.10/site-packages:"
            "/opt/ros/humble/local/lib/python3.10/dist-packages"
        )
        written = {}
        for label, cmd in (
            ("topics", "ros2 topic list -t"),
            ("nodes", "ros2 node list"),
            ("actions", "ros2 action list -t"),
        ):
            path = os.path.join(self.out_dir, f"ros2_{label}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                subprocess.run(
                    ["bash", "-lc", f"source /opt/ros/humble/setup.bash && {cmd}"],
                    stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=60,
                )
            written[label] = path
        return written

    def nav2_log_excerpt(self, name, patterns, limit=40):
        path = os.path.join(self.out_dir, "logs", f"nav2_{name}.log")
        if not os.path.exists(path):
            return []
        hits = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if any(p.lower() in line.lower() for p in patterns):
                    hits.append(line.rstrip())
        return hits[-limit:]

    # --- obstacle evidence ----------------------------------------------
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

    def scan_min_toward(self, name, wx, wy):
        msg = self.ros.scan[name]
        if msg is None:
            return None
        px, py, pyaw = self.pose(name)
        bearing = math.atan2(wy - py, wx - px)
        rel = math.atan2(math.sin(bearing - pyaw), math.cos(bearing - pyaw))
        best = None
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            ang = msg.angle_min + i * msg.angle_increment
            d = abs(math.atan2(math.sin(ang - rel), math.cos(ang - rel)))
            if d > math.radians(6.0):
                continue
            if best is None or r < best:
                best = float(r)
        return None if best is None else round(best, 3)

    def recent_displacement(self, name, window_s=2.0):
        pts = self.path_samples[name]
        if len(pts) < 2:
            return 0.0
        t_end = pts[-1]["t"]
        ref = pts[0]
        for sample in reversed(pts):
            if sample["t"] <= t_end - window_s:
                ref = sample
                break
        return round(math.hypot(pts[-1]["x"] - ref["x"], pts[-1]["y"] - ref["y"]), 3)

    def path_lateral_deviation(self, name, y_ref):
        pts = [s for s in self.path_samples[name]]
        if not pts:
            return 0.0
        return round(max(abs(s["y"] - y_ref) for s in pts), 3)

    # --- run modes ------------------------------------------------------
    def run_smoke(self):
        goal = {self.names[0]: (3.0, 0.0)}
        rec = self.run_round("smoke_single_goal", goal, timeout_s=180.0)
        return rec

    def run_full(self):
        # Case 1: three concurrent goals, no dynamic obstacle.
        t_goal = sim_time(self.world)
        self.run_round("case1_three_goals", config.ROBOT_GOALS_CASE1, timeout_s=config.GOAL_TIMEOUT_S)
        self.log_event("case1_wall_duration_s", value=round(sim_time(self.world) - t_goal, 1))

        # Case 2: return goals, then place an avoidable obstacle in robot1's lane.
        def case2_trigger(ctx):
            px, py, _ = ctx.pose("robot1")
            if px < config.CASE2_TRIGGER_X:
                obstacle_xy = (config.DETOUR_OBSTACLE[1], config.DETOUR_OBSTACLE[2])
                ctx.log_event(
                    "dynamic_obstacle_inserted",
                    case="case2",
                    obstacle="detour_robot1",
                    pose_world=list(obstacle_xy),
                    robot1_pose=[round(px, 2), round(py, 2)],
                    robot1_scan_min_before=ctx.scan_min_toward("robot1", *obstacle_xy),
                    robot1_costmap_before=ctx.costmap_value("robot1", *obstacle_xy),
                )
                ctx.room.enable_dynamic(config.DETOUR_OBSTACLE[0])
                ctx.log_event(
                    "dynamic_obstacle_enabled",
                    obstacle="detour_robot1",
                    visible=ctx.room.is_dynamic_enabled("detour_robot1"),
                )
                for delay in (2.0, 6.0):
                    ctx.schedule_probe(
                        delay,
                        f"case2_detour_t+{delay:g}s",
                        lambda: {
                            "robot1_pose": [round(v, 2) for v in ctx.pose("robot1")],
                            "scan_min_at_obstacle": ctx.scan_min_toward("robot1", *obstacle_xy),
                            "costmap_value_at_obstacle": ctx.costmap_value("robot1", *obstacle_xy),
                            "lateral_deviation_m": ctx.path_lateral_deviation(
                                "robot1", config.LANE_Y["robot1"]
                            ),
                        },
                    )
                return True
            return False

        case2 = self.run_round(
            "case2_dynamic_detour", config.ROBOT_GOALS_CASE2,
            timeout_s=config.GOAL_TIMEOUT_S, trigger=case2_trigger, yaw=math.pi,
        )
        # Evidence after the robot passed or finished.
        obstacle_xy = (config.DETOUR_OBSTACLE[1], config.DETOUR_OBSTACLE[2])
        case2["detour_evidence"] = {
            "scan_min_at_obstacle": self.scan_min_toward("robot1", *obstacle_xy),
            "costmap_value_at_obstacle": self.costmap_value("robot1", *obstacle_xy),
            "max_lateral_deviation_robot1_m": self.path_lateral_deviation(
                "robot1", config.LANE_Y["robot1"]
            ),
            "plan_msgs_robot1": self.ros.plan_count["robot1"],
        }
        self.log_event("case2_evidence", **case2["detour_evidence"])
        self.log_event(
            "nav2_log_excerpt_case2",
            lines=self.nav2_log_excerpt("robot1", ["Received a goal", "Planning", "passing new path"]),
        )

        # Case 3: send eastbound goals, then box robot1 in completely.
        def case3_trigger(ctx):
            px, py, _ = ctx.pose("robot1")
            if px > config.CASE3_TRIGGER_X:
                ctx.log_event(
                    "route_blocked",
                    case="case3",
                    robot1_pose=[round(px, 2), round(py, 2)],
                    barriers=[config.BLOCK_FRONT, config.BLOCK_REAR],
                    scan_min_ahead=ctx.scan_min_toward(
                        "robot1", min(7.0, px + 3.0), py
                    ),
                )
                ctx.room.enable_dynamic(config.BLOCK_FRONT[0])
                ctx.room.enable_dynamic(config.BLOCK_REAR[0])
                px2, py2, _ = ctx.pose("robot1")
                ctx.schedule_probe(
                    2.0,
                    "case3_blocked_t+2s",
                    lambda: {
                        "robot1_pose": [round(v, 2) for v in ctx.pose("robot1")],
                        "scan_min_at_front_wall": ctx.scan_min_toward(
                            "robot1", config.BLOCK_FRONT[1], py2
                        ),
                        "costmap_value_at_front_wall": ctx.costmap_value(
                            "robot1", config.BLOCK_FRONT[1], py2
                        ),
                    },
                )
                return True
            return False

        case3 = self.run_round(
            "case3_blocked_route", config.ROBOT_GOALS_CASE3,
            timeout_s=config.CASE3_TIMEOUT_S, trigger=case3_trigger,
        )
        # Let the deadman turn the last command into zero and confirm the
        # aborted robot is physically stopped before sampling the terminal state.
        for _ in range(180):
            self.step()
        self.log_event(
            "post_abort_settle",
            sim_t=round(sim_time(self.world), 3),
            displacement_last_2s={
                name: self.recent_displacement(name, 2.0) for name in self.names
            },
        )
        final = {}
        for name in self.names:
            px, py, _ = self.pose(name)
            final[name] = {
                "pose_world": [round(px, 3), round(py, 3)],
                "last_feedback": (
                    self.ros.actions[name]["feedback"][-1]
                    if self.ros.actions[name] and self.ros.actions[name]["feedback"]
                    else None
                ),
                "cmd_vel_msgs": self.ros.cmd_count[name],
                "scan_msgs": self.ros.scan_count[name],
                "odom_msgs": self.ros.odom_count[name],
                "displacement_last_2s_m": self.recent_displacement(name, 2.0),
                "tf_edges": sorted([list(e) for e in self.ros.tf_edges[name]]),
                "tf_static_edges": sorted([list(e) for e in self.ros.tf_static_edges[name]]),
            }
        self.log_event("terminal_state", robots=final)
        self.log_event(
            "nav2_log_excerpt_case3",
            lines=self.nav2_log_excerpt(
                "robot1",
                ["Failed to create plan", "Navigation failed", "aborting", "Failed to", "recovery"],
            ),
        )
        return {"case1": self.rounds[0], "case2": case2, "case3": case3}

    def run_case4(self):
        rec = self.run_round("case4_new_goals", config.ROBOT_GOALS_CASE4, timeout_s=config.GOAL_TIMEOUT_S)
        interfaces = {}
        cross_talk = {}
        for name in self.names:
            foreign = [
                list(edge)
                for edge in self.ros.tf_edges[name]
                if any(frame != "map" and not frame.startswith(f"{name}/") for frame in edge)
            ]
            cross_talk[name] = {
                "foreign_tf_edges": foreign,
                "namespace_clean": not foreign,
            }
            interfaces[name] = {
                "action_result": rec["robots"][name]["status"],
                "goal": rec["robots"][name]["goal"],
                "odom_msgs": self.ros.odom_count[name],
                "scan_msgs": self.ros.scan_count[name],
                "cmd_vel_msgs": self.ros.cmd_count[name],
                "plan_msgs": self.ros.plan_count[name],
                "costmap_msgs": self.ros.costmap_count[name],
                "tf_edges": sorted([list(e) for e in self.ros.tf_edges[name]]),
                "tf_static_edges": sorted([list(e) for e in self.ros.tf_static_edges[name]]),
                "topics": {
                    "goal_action": f"/{name}/navigate_to_pose",
                    "odom": f"/{name}/odom",
                    "scan": f"/{name}/scan",
                    "cmd_vel": f"/{name}/cmd_vel",
                    "plan": f"/{name}/plan",
                    "global_costmap": f"/{name}/global_costmap/costmap",
                    "local_costmap": f"/{name}/local_costmap/costmap",
                    "tf": f"/{name}/tf",
                    "tf_static": f"/{name}/tf_static",
                },
            }
        graph_files = self.capture_ros_graph()
        self.log_event(
            "case4_interfaces",
            interfaces=interfaces,
            cross_talk=cross_talk,
            ros_graph_files=graph_files,
        )
        return {
            "round": rec,
            "interfaces": interfaces,
            "cross_talk": cross_talk,
            "ros_graph_files": graph_files,
        }

    # --- top level ------------------------------------------------------
    def run(self):
        self.start()
        try:
            if self.mode == "smoke":
                result = self.run_smoke()
            elif self.mode == "full":
                result = self.run_full()
            else:
                result = self.run_case4()
            summary = {
                "mode": self.mode,
                "names": self.names,
                "map_yaml": self.map_yaml,
                "robot_starts": {n: list(config.ROBOT_STARTS[n]) for n in self.names},
                "result": result,
                "rounds": self.rounds,
                "events": self.events,
                "path_samples": self.path_samples,
            }
            path = os.path.join(self.out_dir, f"summary_{self.mode}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
            print("SUMMARY_WRITTEN", path, flush=True)
            return summary
        finally:
            self.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "case4"], required=True)
    parser.add_argument("--no-render", action="store_true")
    args, _ = parser.parse_known_args()
    run = Acceptance(args.mode)
    try:
        run.run()
        code = 0
    except Exception as exc:
        print("FATAL", repr(exc), flush=True)
        traceback.print_exc()
        code = 1
    finally:
        SIM_APP.close()
    print("EXIT", code, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
