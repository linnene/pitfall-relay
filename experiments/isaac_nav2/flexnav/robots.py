"""Spawn Carter robots, their drive controller and ROS 2 sensor/odometry graph.

WheeledRobot owns the articulation and makes PhysX register it before play;
the DifferentialController is Isaac's stock wheel controller and is applied
through apply_wheel_actions. OmniGraph publishes odometry, TF and the RTX
lidar scan.
"""

import math

import numpy as np
import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf

from . import config

CARTER_USD = "/Isaac/Robots/Carter/carter_v1.usd"
WHEEL_RADIUS = 0.24
WHEEL_DISTANCE = 0.53342
SPAWN_Z = 0.3


def _lidar_render_product_path(render_product):
    for attr in ("path", "_path", "render_product_path"):
        value = getattr(render_product, attr, None)
        if isinstance(value, str) and value:
            return value
    for meth in ("get_prim_path", "get_path"):
        fn = getattr(render_product, meth, None)
        if callable(fn):
            value = fn()
            if isinstance(value, str) and value:
                return value
    raise RuntimeError(f"cannot obtain render product path from {render_product!r}")


def _build_odom_graph(robot_root, name, x, y):
    """Static map->odom TF; odometry itself is published from true sim pose."""
    graph_path = f"/World/{name}_graph"
    ns = f"/{name}"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("tf_map_odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("tf_map_odom.inputs:topicName", "tf_static"),
                ("tf_map_odom.inputs:parentFrameId", "map"),
                ("tf_map_odom.inputs:childFrameId", f"{name}/odom"),
                ("tf_map_odom.inputs:nodeNamespace", ns),
                ("tf_map_odom.inputs:staticPublisher", True),
                ("tf_map_odom.inputs:translation", (x, y, 0.0)),
                ("tf_map_odom.inputs:rotation", (1.0, 0.0, 0.0, 0.0)),
            ],
            og.Controller.Keys.CONNECT: [
                ("tick.outputs:tick", "tf_map_odom.inputs:execIn"),
                ("sim_time.outputs:simulationTime", "tf_map_odom.inputs:timeStamp"),
            ],
        },
    )
    stage = omni.usd.get_context().get_stage()
    tf_prim = stage.GetPrimAtPath(f"{graph_path}/tf_map_odom")
    tf_prim.GetAttribute("inputs:translation").Set(Gf.Vec3d(float(x), float(y), 0.0))
    tf_prim.GetAttribute("inputs:rotation").Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
    return graph_path


def _add_lidar(robot_root, name):
    parent = f"{robot_root}/chassis_link"
    _, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="lidar",
        parent=parent,
        config="Example_Rotary",
        translation=(0.0, 0.0, 0.8),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )
    if sensor is None or not sensor.IsValid():
        raise RuntimeError(f"lidar prim was not created under {parent}")
    sensor_path = str(sensor.GetPath())
    render_product = rep.create.render_product(sensor.GetPath(), [1, 1], name=f"{name}_lidar")
    rp_path = _lidar_render_product_path(render_product)
    graph_path = f"/World/{name}_lidar_graph"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("helper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("helper.inputs:renderProductPath", rp_path),
                ("helper.inputs:topicName", "scan"),
                ("helper.inputs:frameId", f"{name}/base_link"),
                ("helper.inputs:nodeNamespace", f"/{name}"),
                ("helper.inputs:type", "laser_scan"),
                ("helper.inputs:fullScan", False),
            ],
            og.Controller.Keys.CONNECT: [
                ("tick.outputs:tick", "helper.inputs:execIn"),
            ],
        },
    )
    return sensor_path, rp_path


def add_robot(world, name, x, y, yaw=0.0):
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("Isaac Sim assets root is not configured")
    robot_root = f"/World/{name}"
    robot = world.scene.add(
        WheeledRobot(
            prim_path=robot_root,
            name=name,
            wheel_dof_names=["left_wheel", "right_wheel"],
            create_robot=True,
            usd_path=root + CARTER_USD,
            position=np.array([x, y, SPAWN_Z], dtype=float),
            orientation=np.array(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float
            ),
        )
    )
    controller = DifferentialController(
        name=f"{name}_differential", wheel_radius=WHEEL_RADIUS, wheel_base=WHEEL_DISTANCE
    )
    graph_path = _build_odom_graph(robot_root, name, x, y)
    sensor_path, rp_path = _add_lidar(robot_root, name)
    return {
        "name": name,
        "root": robot_root,
        "robot": robot,
        "controller": controller,
        "graph": graph_path,
        "lidar": sensor_path,
        "render_product": rp_path,
    }


def add_clock_graph():
    og.Controller.edit(
        {"graph_path": "/World/clock_graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("clock.inputs:topicName", "clock"),
                ("clock.inputs:nodeNamespace", ""),
            ],
            og.Controller.Keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
            ],
        },
    )


def set_lidar_enabled(name, enabled):
    """Toggle the RTX lidar publisher, e.g. to cut render load during startup."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(f"/World/{name}_lidar_graph/helper")
    prim.GetAttribute("inputs:enabled").Set(bool(enabled))
