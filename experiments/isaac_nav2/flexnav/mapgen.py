"""Generate the static occupancy map for the acceptance room.

The map is rasterised from the same axis-aligned wall list used to build the
USD scene. Unknown space is not used: the whole room is surveyed.
"""

import os

from . import config

FREE = 254
OCCUPIED = 0


def rasterize(walls, resolution=config.MAP_RESOLUTION):
    width = int(round((2 * config.ROOM_HALF_X) / resolution))
    height = int(round((2 * config.ROOM_HALF_Y) / resolution))
    ox, oy = config.MAP_ORIGIN
    grid = bytearray([FREE]) * (width * height)
    for cx, cy, sx, sy in walls:
        x0 = int(max(0, (cx - sx / 2 - ox) / resolution))
        x1 = int(min(width, (cx + sx / 2 - ox) / resolution + 1))
        y0 = int(max(0, (cy - sy / 2 - oy) / resolution))
        y1 = int(min(height, (cy + sy / 2 - oy) / resolution + 1))
        for gy in range(y0, y1):
            row = height - 1 - gy  # PGM rows run from max y downwards
            base = row * width
            for gx in range(x0, x1):
                grid[base + gx] = OCCUPIED
    return width, height, grid


def write_map(out_dir, walls=None):
    """Write room_map.pgm and room_map.yaml; return the YAML path."""
    walls = config.STATIC_WALLS if walls is None else walls
    width, height, grid = rasterize(walls)
    os.makedirs(out_dir, exist_ok=True)
    pgm_path = os.path.join(out_dir, "room_map.pgm")
    yaml_path = os.path.join(out_dir, "room_map.yaml")
    with open(pgm_path, "wb") as fh:
        fh.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        fh.write(bytes(grid))
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            "image: room_map.pgm\n"
            f"resolution: {config.MAP_RESOLUTION}\n"
            f"origin: [{config.MAP_ORIGIN[0]}, {config.MAP_ORIGIN[1]}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        )
    return yaml_path


if __name__ == "__main__":
    import sys

    print(write_map(sys.argv[1] if len(sys.argv) > 1 else "."))
