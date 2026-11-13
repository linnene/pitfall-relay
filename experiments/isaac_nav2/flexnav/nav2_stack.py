"""Render per-robot Nav2 params and manage one namespaced Nav2 launch process."""

import os
import signal
import subprocess
import time

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState

LAUNCH_FILE = "/work/nav2_launch/robot_nav2.launch.py"
PARAMS_TEMPLATE = "/work/config/nav2_params.yaml.in"
ROS_SETUP = "source /opt/ros/humble/setup.bash"


def render_params(namespace, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(PARAMS_TEMPLATE, "r", encoding="utf-8") as fh:
        text = fh.read()
    path = os.path.join(out_dir, f"{namespace}_nav2_params.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace("__NS__", namespace))
    return path


class Nav2Stack:
    def __init__(self, namespace, params_file, map_yaml, log_path):
        self.namespace = namespace
        self.params_file = params_file
        self.map_yaml = map_yaml
        self.log_path = log_path
        self.proc = None
        self.log_file = None

    def start(self):
        cmd = (
            f"{ROS_SETUP} && exec ros2 launch {LAUNCH_FILE} "
            f"namespace:={self.namespace} params_file:={self.params_file} "
            f"map:={self.map_yaml} use_sim_time:=true autostart:=true"
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Keep the Isaac/kit PYTHONPATH (and /work) out of the ROS CLI process:
        # /work contains a launch/ directory that would shadow the ROS package.
        env["PYTHONPATH"] = (
            "/opt/ros/humble/lib/python3.10/site-packages:"
            "/opt/ros/humble/local/lib/python3.10/dist-packages"
        )
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        return self.proc

    def stop(self, timeout=15.0):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            deadline = time.time() + timeout
            while self.proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def wait_for_lifecycle(node, namespaces, node_names, timeout_s, pump=None):
    """Wait until every lifecycle node reports ACTIVE; return a status dict."""
    clients = {}
    for ns in namespaces:
        for name in node_names:
            srv = f"/{ns}/{name}/get_state"
            clients[(ns, name)] = node.create_client(GetState, srv)
    deadline = time.time() + timeout_s
    status = {key: None for key in clients}
    while time.time() < deadline:
        for key, client in clients.items():
            if status[key] == State.PRIMARY_STATE_ACTIVE:
                continue
            if not client.service_is_ready():
                continue
            future = client.call_async(GetState.Request())
            rclpy_spin_until(node, future, 0.2)
            if future.result() is not None:
                status[key] = future.result().current_state.id
        if all(v == State.PRIMARY_STATE_ACTIVE for v in status.values()):
            return status
        if pump is not None:
            pump()
        time.sleep(0.2)
    return status


def rclpy_spin_until(node, future, timeout_s):
    import rclpy

    deadline = time.time() + timeout_s
    while not future.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def force_activate(node, namespace, name, timeout_s=30.0):
    """Configure/activate a lifecycle node if its manager gave up on it."""
    import rclpy

    get_client = node.create_client(GetState, f"/{namespace}/{name}/get_state")
    change_client = node.create_client(ChangeState, f"/{namespace}/{name}/change_state")
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        if not get_client.service_is_ready() or not change_client.service_is_ready():
            rclpy.spin_once(node, timeout_sec=0.1)
            continue
        future = get_client.call_async(GetState.Request())
        rclpy_spin_until(node, future, 2.0)
        if future.result() is None:
            time.sleep(0.2)
            continue
        last = future.result().current_state.id
        if last == State.PRIMARY_STATE_ACTIVE:
            return last
        transition = (
            Transition.TRANSITION_CONFIGURE
            if last == State.PRIMARY_STATE_UNCONFIGURED
            else Transition.TRANSITION_ACTIVATE
        )
        req = ChangeState.Request()
        req.transition.id = transition
        future = change_client.call_async(req)
        rclpy_spin_until(node, future, 20.0)
        time.sleep(0.5)
    return last
