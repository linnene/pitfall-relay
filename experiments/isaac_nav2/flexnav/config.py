"""Shared layout and scenario configuration for the Isaac Sim Nav2 acceptance.

Coordinates are metres in the Isaac world frame. The static occupancy map is
generated from the same wall list that builds the USD scene, so map and world
cannot drift apart.
"""

ROOM_HALF_X = 7.0
ROOM_HALF_Y = 5.0
WALL_THICKNESS = 0.2
WALL_HEIGHT = 1.5

# Static walls, given as (center_x, center_y, size_x, size_y).
STATIC_WALLS = [
    (0.0, 5.0, 14.0, WALL_THICKNESS),    # north outer wall
    (0.0, -5.0, 14.0, WALL_THICKNESS),   # south outer wall
    (7.0, 0.0, WALL_THICKNESS, 10.0),    # east outer wall
    (-7.0, 0.0, WALL_THICKNESS, 10.0),   # west outer wall
    (0.0, 1.7, 11.0, WALL_THICKNESS),    # lane divider A/B
    (0.0, -1.7, 11.0, WALL_THICKNESS),   # lane divider B/C
    (0.0, -3.35, 0.6, 0.6),              # pillar in lane 0
    (0.0, 3.35, 0.6, 0.6),               # pillar in lane 2
]

# The three lane centre lines and the start/goal x positions.
LANE_Y = {"robot0": -3.35, "robot1": 0.0, "robot2": 3.35}
START_X = -6.0
GOAL_X = 6.0

ROBOT_NAMES = ["robot0", "robot1", "robot2"]
ROBOT_STARTS = {name: (START_X, LANE_Y[name]) for name in ROBOT_NAMES}
ROBOT_GOALS_CASE1 = {name: (GOAL_X, LANE_Y[name]) for name in ROBOT_NAMES}
ROBOT_GOALS_CASE2 = {name: (START_X, LANE_Y[name]) for name in ROBOT_NAMES}
ROBOT_GOALS_CASE3 = {name: (GOAL_X, LANE_Y[name]) for name in ROBOT_NAMES}
# Case 4 permutes goal positions so interface checks do not repeat case 1.
ROBOT_GOALS_CASE4 = {
    "robot0": (3.0, -3.35),
    "robot1": (5.0, 0.30),
    "robot2": (2.0, 3.60),
}

# Dynamic obstacles are authored hidden and non-colliding, then toggled on.
# (prim_name, center_x, center_y, size_x, size_y)
DETOUR_OBSTACLE = ("detour_robot1", -1.0, 0.0, 0.7, 0.7)
BLOCK_FRONT = ("block_front", 4.2, 0.0, 0.2, 3.3)
BLOCK_REAR = ("block_rear", 0.4, 0.0, 0.2, 3.3)

# Trigger conditions, in world x of robot1.
CASE2_TRIGGER_X = 3.0    # robot1 westbound, obstacle three metres ahead
CASE3_TRIGGER_X = 1.8    # robot1 eastbound, both barriers clear of the robot

MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-ROOM_HALF_X, -ROOM_HALF_Y)

# Nav2 / mission timing.
GOAL_TIMEOUT_S = 240.0
CASE3_TIMEOUT_S = 240.0
NAV2_START_TIMEOUT_S = 180.0
