# Nav2 acceptance in Isaac Sim 4.5

Question: with only a final `NavigateToPose` goal per robot, can the shipped
Nav2 stack plan, avoid obstacles, drive the existing Isaac wheel controller
and report results for three concurrent Carters, including an explicit native
failure when one route is fully blocked?

## Setup

- Isaac Sim 4.5.0 headless, ROS 2 Humble bridge, derived image
  `isaacsim4.5.0-nav2:local` (see `../docker/Dockerfile`).
- Three Carter v1 robots, namespaces `robot0`/`robot1`/`robot2`.
- Room 14 x 10 m with two lane dividers and two static pillars; the occupancy
  map is rasterised from the same wall list as the USD scene.
- Each robot has its own Nav2 stack (planner, controller, behaviours,
  bt_navigator, velocity smoother, map server, lifecycle manager).
- One RTX lidar per robot publishes `/robotN/scan`; `WheeledRobot` +
  `DifferentialController` execute `/robotN/cmd_vel`.
- ROS domain 77, one GPU, isolated container, all data under
  `/data/amgg_flex_demo/flex-isaac-nav2-acceptance-d5d7773d-bfa4-4421-b77d-bff0bb82c0d0/runs/`.

## Goal-to-wheel chain

`NavigateToPose` action -> `bt_navigator` -> NavFn global plan ->
DWB local controller -> `cmd_vel_nav` -> velocity smoother -> `/robotN/cmd_vel`
-> rclpy bridge in the Isaac process -> Isaac `DifferentialController`
-> `WheeledRobot.apply_wheel_actions` -> PhysX wheel joints -> chassis motion
-> exact simulated odometry published as `/robotN/odom` and TF
`robotN/odom -> robotN/base_link`, with static `map -> robotN/odom` at the start pose.

## Results (run `runs/full9` and `runs/case4g`)

All final poses are measured on the simulated chassis (`base_link`), after the
action result. Recorded acceptance bound for `xy` is 0.30 m (Nav2 goal
tolerance 0.20 m plus stopping transient); yaw bound 0.35 rad.

### Case 1: three concurrent goals

| robot | goal | status | final (x,y,yaw) | xy err | yaw err |
| --- | --- | --- | --- | ---: | ---: |
| robot0 | (6.0,-3.35) | SUCCEEDED | (5.899,-3.319,-0.220) | 0.106 | 0.220 |
| robot1 | (6.0,0.0) | SUCCEEDED | (5.835,0.047,-0.155) | 0.171 | 0.155 |
| robot2 | (6.0,3.35) | SUCCEEDED | (5.903,3.384,-0.202) | 0.103 | 0.202 |

### Case 2: obstacle inserted after motion

| robot | goal | status | xy err |
| --- | --- | --- | ---: |
| robot0 | (-6.0,-3.35) | SUCCEEDED | 0.142 |
| robot1 | (-6.0,0.0) | SUCCEEDED | 0.167 |
| robot2 | (-6.0,3.35) | SUCCEEDED | 0.174 |

Obstacle at (-1.0, 0.0), enabled at sim t=63.9 while robot1 was at x=2.99 and
moving west. Robot1 scan toward the obstacle: 3.352 m at +2 s, 1.692 m at
+6 s; global costmap cell at the obstacle: 99 (lethal) in both probes; max
lateral deviation while passing: 1.11 m; 93 plan messages; robot1 arrived
with 1 recovery. No waypoints were added externally.

### Case 3: route fully blocked

| robot | goal | status | final (x,y) | xy err | recoveries |
| --- | --- | --- | --- | ---: | ---: |
| robot0 | (6.0,-3.35) | SUCCEEDED | (5.886,-3.419) | 0.133 | 0 |
| robot1 | (6.0,0.0) | ABORTED | (2.846,-0.481) | 3.191 | 18 |
| robot2 | (6.0,3.35) | SUCCEEDED | (5.895,3.283) | 0.125 | 0 |

Barriers at x=0.4 and x=4.2, spanning robot1's lane, enabled at sim t=168.8
when robot1 was at x=1.80. Front wall in scan: 1.621 m, costmap 100. Nav2
planner logged `GridBased failed to generate a valid path to (6.00, 0.00)` and
`[navigate_to_pose] [ActionServer] Aborting handle`; action status ABORTED,
`cancelled_by_client=false`. After the abort, 6 s settle: displacement over
the last 2 s is 0.0 m for all three robots.

### Case 4: reset, new goals, interfaces

Fresh process: `runs/case4g`. Goals were different from case 1.

| robot | goal | status | final (x,y,yaw) | xy err |
| --- | --- | --- | --- | ---: |
| robot0 | (3.0,-3.35) | SUCCEEDED | (2.933,-3.301,-0.207) | 0.083 |
| robot1 | (5.0,0.30) | SUCCEEDED | (4.832,0.225,0.246) | 0.184 |
| robot2 | (2.0,3.60) | SUCCEEDED | (2.044,3.449,0.225) | 0.157 |

Per-robot TF contains only `map -> robotN/odom` (static) and
`robotN/odom -> robotN/base_link` (dynamic); no foreign TF edges. ROS graph
listings are in `runs/case4g/ros2_topics.txt`, `ros2_nodes.txt`,
`ros2_actions.txt`.
