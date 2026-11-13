# Isaac Sim 4.5 + Nav2 multi-robot acceptance

This directory holds the reproducible stack used to accept Nav2 autonomous
navigation for three Carters in Isaac Sim 4.5 before the FLEX migration.
The task layer only sends each robot a final `NavigateToPose` goal; Nav2 does
planning, obstacle avoidance, control and result reporting.

## Environment

- Host: `amgg` (Ubuntu 24.04, NVIDIA RTX 4090), one GPU per run.
- Base image: `isaacsim4.5.0-rosbridge:local` (Isaac Sim 4.5.0, ROS 2 Humble
  bridge). It is assumed to exist locally on the experiment host.
- Derived image: `docker/build_image.sh` builds `isaacsim4.5.0-nav2:local`
  by adding `ros-humble-navigation2`, `ros-humble-nav2-bringup`,
  `ros-humble-tf2-tools`, `ros-humble-xacro` and `python3-lark`.
- All simulation, ROS 2 nodes, tests and log processing run on `amgg`; the
  Mac only holds code and summaries.

## Architecture

Per robot `robotN` (`N` in 0,1,2), everything is namespaced:

| Interface | Topic / action |
| --- | --- |
| Goal | `/robotN/navigate_to_pose` (`nav2_msgs/action/NavigateToPose`) |
| Odometry | `/robotN/odom`, TF `robotN/odom -> robotN/base_link` |
| TF | `/robotN/tf`, `map -> robotN/odom -> robotN/base_link` |
| Lidar | `/robotN/scan` (`sensor_msgs/LaserScan`, frame `robotN/base_link`) |
| Velocity | `/robotN/cmd_vel` (`geometry_msgs/Twist`) |
| Plan / costmaps | `/robotN/plan`, `/robotN/global_costmap/costmap`, `/robotN/local_costmap/costmap` |

Control chain from goal to wheels:

1. Acceptance client sends `NavigateToPose` to the robot's bt_navigator.
2. `planner_server` (NavFn) plans on the static map plus the global obstacle
   layer; `controller_server` (DWB) follows the path using the local costmap.
3. `velocity_smoother` publishes `/robotN/cmd_vel`.
4. The Isaac process subscribes to `/robotN/cmd_vel` and feeds Isaac's stock
   `DifferentialController` (wheel radius 0.24 m, wheel base 0.53342 m).
5. `WheeledRobot.apply_wheel_actions` drives the PhysX wheel joints. The
   Isaac-side process publishes `/robotN/odom` (chassis pose relative to its
   start) and `robotN/odom -> robotN/base_link` TF at 60 Hz; a static
   `map -> robotN/odom` transform places the odom origin at the start pose.
   This uses the simulated true chassis pose so Nav2's map frame cannot drift
   from the Isaac world frame; it feeds feedback only, never control.

The static map (`runs/.../map/room_map.pgm|yaml`) is rasterised from the same
wall list that builds the USD room, so map and world cannot drift.

Using Isaac's wheel/articulation odometry alone accumulated up to ~0.5 m of
map error over a turning run, which made final-pose comparison meaningless.
The odometry here is therefore exact simulated odometry (no noise, no slip),
published by the Isaac-side bridge; Nav2 still plans and controls normally.

## Layout

```
config/nav2_params.yaml.in   per-robot Nav2 parameters (`__NS__` replaced)
nav2_launch/robot_nav2.launch.py  one namespaced Nav2 stack + map server
flexnav/                     scene, robots, map generation, ROS/Nav2 helpers
scripts/acceptance.py        smoke / full / case4 runners and evidence capture
tools/                       development probes kept for debugging
run.sh                       container entry point (PYTHONPATH/ROS env)
```

## Build and run on amgg

```bash
cd experiments/isaac_nav2/docker && bash build_image.sh

cd ..
mkdir -p runs/smoke1
docker run --rm --name flex_nav2_smoke --gpus device=1 \
  -e OMNI_KIT_ALLOW_ROOT=1 -e ROS_DOMAIN_ID=77 -e RUN_OUT=/runs/smoke1 \
  --shm-size=2g \
  -v "$PWD":/work \
  -v "$PWD/runs":/runs \
  --entrypoint bash isaacsim4.5.0-nav2:local \
  -lc "bash /work/run.sh --mode smoke"
```

`--mode full` runs cases 1-3 in one session; `--mode case4` starts a fresh
simulation and Nav2 stack with a different goal set. `RUN_OUT` is the run
directory; keep it outside the repository and leave raw data on `amgg`.

## Startup note

Three Nav2 stacks plus Isaac put a burst on DDS discovery. Startup starts the
stacks one by one and, if a lifecycle manager's `change_state` response misses
its deadline, the runner retries the transition directly and checks all eight
lifecycle nodes before sending goals. RTX lidars are disabled until all stacks
are active to reduce render load.

## Acceptance evidence

Each run writes `summary_<mode>.json` with per-robot action results, final
pose error, path samples, plan/odom/scan/cmd_vel counts, TF edges, timed
probe samples for inserted obstacles and Nav2 log excerpts. Raw Nav2 logs are
under `RUN_OUT/logs/`.
