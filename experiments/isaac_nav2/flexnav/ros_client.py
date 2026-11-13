"""ROS 2 side of the acceptance run: cmd_vel bridge, telemetry, action clients."""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


def yaw_to_quaternion(yaw):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def quaternion_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FleetRos:
    def __init__(self, names, starts=None):
        rclpy.init()
        self.node = Node(
            "nav2_acceptance_client",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.names = names
        self.cmd_vel = {n: Twist() for n in names}
        self.cmd_count = {n: 0 for n in names}
        self.cmd_stamp = {n: 0.0 for n in names}
        self.odom = {n: None for n in names}
        self.odom_count = {n: 0 for n in names}
        self.scan = {n: None for n in names}
        self.scan_count = {n: 0 for n in names}
        self.plan = {n: None for n in names}
        self.plan_count = {n: 0 for n in names}
        self.costmap = {n: None for n in names}
        self.costmap_count = {n: 0 for n in names}
        self.tf_edges = {n: set() for n in names}
        self.tf_static_edges = {n: set() for n in names}
        self.actions = {n: None for n in names}
        self.starts = starts or {}
        self.odom_pubs = {n: self.node.create_publisher(Odometry, f"/{n}/odom", 20) for n in names}
        self.tf_pubs = {n: self.node.create_publisher(TFMessage, f"/{n}/tf", 20) for n in names}
        self.prev_odom = {n: None for n in names}
        self.action_clients = {n: ActionClient(self.node, NavigateToPose, f"/{n}/navigate_to_pose") for n in names}

        transient = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        for n in names:
            self.node.create_subscription(Twist, f"/{n}/cmd_vel", self._cmd_cb(n), 10)
            self.node.create_subscription(Odometry, f"/{n}/odom", self._odom_cb(n), 10)
            self.node.create_subscription(LaserScan, f"/{n}/scan", self._scan_cb(n), 10)
            self.node.create_subscription(Path, f"/{n}/plan", self._plan_cb(n), 10)
            self.node.create_subscription(
                OccupancyGrid, f"/{n}/global_costmap/costmap", self._costmap_cb(n), transient
            )
            self.node.create_subscription(TFMessage, f"/{n}/tf", self._tf_cb(n), 10)
            self.node.create_subscription(
                TFMessage, f"/{n}/tf_static", self._tf_static_cb(n), transient
            )

    # --- subscriptions -------------------------------------------------
    def _cmd_cb(self, name):
        def cb(msg):
            self.cmd_vel[name] = msg
            self.cmd_count[name] += 1
            self.cmd_stamp[name] = time.time()
        return cb

    def _odom_cb(self, name):
        def cb(msg):
            self.odom[name] = msg
            self.odom_count[name] += 1
        return cb

    def _scan_cb(self, name):
        def cb(msg):
            self.scan[name] = msg
            self.scan_count[name] += 1
        return cb

    def _plan_cb(self, name):
        def cb(msg):
            self.plan[name] = msg
            self.plan_count[name] += 1
        return cb

    def _costmap_cb(self, name):
        def cb(msg):
            self.costmap[name] = msg
            self.costmap_count[name] += 1
        return cb

    def _tf_cb(self, name):
        def cb(msg):
            for t in msg.transforms:
                self.tf_edges[name].add((t.header.frame_id, t.child_frame_id))
        return cb

    def _tf_static_cb(self, name):
        def cb(msg):
            for t in msg.transforms:
                self.tf_static_edges[name].add((t.header.frame_id, t.child_frame_id))
        return cb

    # --- actions -------------------------------------------------------
    def action_ready(self, name, timeout_s=10.0):
        return self.action_clients[name].wait_for_server(timeout_sec=timeout_s)

    def send_goal(self, name, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = float(math.sin(yaw / 2.0))
        goal.pose.pose.orientation.w = float(math.cos(yaw / 2.0))
        client = self.action_clients[name]
        send_future = client.send_goal_async(goal, feedback_callback=self._feedback_cb(name))
        record = {
            "goal": (float(x), float(y), float(yaw)),
            "send_future": send_future,
            "handle": None,
            "result_future": None,
            "feedback": [],
            "result": None,
            "status": None,
            "cancelled_by_client": False,
        }
        self.actions[name] = record
        return record

    def _feedback_cb(self, name):
        def cb(msg):
            rec = self.actions.get(name)
            if not rec:
                return
            fb = msg.feedback
            rec["feedback"].append(
                {
                    "t": time.time(),
                    "distance_remaining": float(fb.distance_remaining),
                    "navigation_time_s": float(fb.navigation_time.sec + fb.navigation_time.nanosec * 1e-9),
                    "number_of_recoveries": int(fb.number_of_recoveries),
                    "current_pose": (
                        float(fb.current_pose.pose.position.x),
                        float(fb.current_pose.pose.position.y),
                    ),
                }
            )
        return cb

    def poll_actions(self):
        """Advance goal/result futures; returns names whose result just arrived."""
        done = []
        for name, rec in self.actions.items():
            if rec is None:
                continue
            if rec["handle"] is None and rec["send_future"] is not None and rec["send_future"].done():
                rec["handle"] = rec["send_future"].result()
                if rec["handle"] is not None and rec["handle"].accepted:
                    rec["result_future"] = rec["handle"].get_result_async()
                else:
                    rec["status"] = GoalStatus.STATUS_UNKNOWN
            if rec["result_future"] is not None and rec["result_future"].done() and rec["result"] is None:
                response = rec["result_future"].result()
                rec["result"] = response.result
                rec["status"] = response.status
                done.append(name)
        return done

    def spin(self, timeout_s=0.0):
        rclpy.spin_once(self.node, timeout_sec=timeout_s)

    def shutdown(self):
        for client in self.action_clients.values():
            client.destroy()
        self.node.destroy_node()
        rclpy.shutdown()


    def publish_state(self, name, x, y, yaw, dt):
        """Publish ground-truth odometry and odom->base TF for Nav2.

        The map->odom static transform is the robot start pose, so this keeps
        Nav2's map frame equal to the Isaac world frame without odometry drift.
        """
        sx, sy = self.starts.get(name, (0.0, 0.0))
        ox, oy = x - sx, y - sy
        stamp = self.node.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = f"{name}/odom"
        odom.child_frame_id = f"{name}/base_link"
        odom.pose.pose.position.x = float(ox)
        odom.pose.pose.position.y = float(oy)
        odom.pose.pose.orientation.z = float(math.sin(yaw / 2.0))
        odom.pose.pose.orientation.w = float(math.cos(yaw / 2.0))
        prev = self.prev_odom[name]
        if prev is not None and dt > 0.0:
            odom.twist.twist.linear.x = float((ox - prev[0]) / dt)
            odom.twist.twist.linear.y = float((oy - prev[1]) / dt)
            dyaw = math.atan2(math.sin(yaw - prev[2]), math.cos(yaw - prev[2]))
            odom.twist.twist.angular.z = float(dyaw / dt)
        self.prev_odom[name] = (ox, oy, yaw)
        self.odom_pubs[name].publish(odom)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = f"{name}/odom"
        tf.child_frame_id = f"{name}/base_link"
        tf.transform.translation.x = float(ox)
        tf.transform.translation.y = float(oy)
        tf.transform.rotation.z = float(math.sin(yaw / 2.0))
        tf.transform.rotation.w = float(math.cos(yaw / 2.0))
        self.tf_pubs[name].publish(TFMessage(transforms=[tf]))
