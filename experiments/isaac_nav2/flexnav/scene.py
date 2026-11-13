"""Build the acceptance room and its switchable obstacles in Isaac Sim."""

import numpy as np
from isaacsim.core.api.objects import FixedCuboid
from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics

from . import config


def _set_collision(prim, enabled):
    for p in [prim] + list(prim.GetChildren()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
            if not attr:
                attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
            attr.Set(bool(enabled))


def _set_visible(prim, visible):
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


class RoomScene:
    def __init__(self, world):
        self.world = world
        self.stage = world.stage
        self.dynamic_paths = {}

    def _box(self, name, center, size, color, height=None, visible=True, z_center=None):
        z = size[2] / 2.0 if height is None else height / 2.0
        if z_center is not None:
            z = z_center
        obj = FixedCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array([center[0], center[1], z], dtype=float),
            scale=np.array(size, dtype=float),
            color=np.array(color, dtype=float),
            visible=visible,
        )
        self.world.scene.add(obj)
        return obj

    def build(self):
        # Floor: top surface exactly at z=0 so wheel contact starts cleanly.
        self._box("Floor", (0.0, 0.0), (14.0, 10.0, 0.1), (0.55, 0.58, 0.62), z_center=-0.05)
        # Static walls and pillars, shared with the occupancy map.
        for i, (cx, cy, sx, sy) in enumerate(config.STATIC_WALLS):
            self._box(f"wall_{i:02d}", (cx, cy), (sx, sy, config.WALL_HEIGHT),
                      (0.22, 0.26, 0.32))
        # Dynamic obstacles: hidden and non-colliding until the scenario
        # enables them. They are intentionally absent from the static map.
        for name, cx, cy, sx, sy in (
            config.DETOUR_OBSTACLE,
            config.BLOCK_FRONT,
            config.BLOCK_REAR,
        ):
            self._box(name, (cx, cy), (sx, sy, 1.2), (0.85, 0.55, 0.10), visible=False)
            path = f"/World/{name}"
            prim = self.stage.GetPrimAtPath(path)
            _set_collision(prim, False)
            _set_visible(prim, False)
            self.dynamic_paths[name] = path
        # Lights.
        dome = UsdLux.DomeLight.Define(self.stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(700.0)
        sun = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/DistantLight"))
        sun.CreateIntensityAttr(1100.0)
        sun.CreateAngleAttr(0.5)

    def enable_dynamic(self, name):
        path = self.dynamic_paths[name]
        prim = self.stage.GetPrimAtPath(path)
        _set_visible(prim, True)
        _set_collision(prim, True)
        return path

    def is_dynamic_enabled(self, name):
        prim = self.stage.GetPrimAtPath(self.dynamic_paths[name])
        return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
