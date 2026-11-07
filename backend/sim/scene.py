"""Places to fly: solid geometry and how each surface looks and behaves."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    material: str
    fit: bool = False  # stretch the texture once across each face instead of tiling it
    solid: bool = True  # False for rugs, pictures and other things nothing can bump into


@dataclass(frozen=True)
class Scene:
    name: str
    bounds: Box | None = None  # the inside of a room; None means open air
    boxes: tuple[Box, ...] = ()
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw
    floor_material: str = "wood_floor"
    wall_material: str = "wall"
    ceiling_material: str = "ceiling"
    wind_mps: float = 0.0
    wind_direction_deg: float = 0.0
    gust_mps: float = 0.0
    flow_quality: float = 1.0  # how much texture the floor gives the optical-flow sensor
    fog_distance_m: float = 0.0

    def collide(self, x: float, y: float, z: float, radius: float):
        """Deepest contact between a sphere and the walls or boxes, as (normal, depth).

        The floor is left out: the physics model rests the landing gear on it.
        """
        best = None
        best_depth = 0.0
        bounds = self.bounds
        if bounds is not None:
            for depth, normal in (
                (bounds.lo[0] + radius - x, (1.0, 0.0, 0.0)),
                (x + radius - bounds.hi[0], (-1.0, 0.0, 0.0)),
                (bounds.lo[1] + radius - y, (0.0, 1.0, 0.0)),
                (y + radius - bounds.hi[1], (0.0, -1.0, 0.0)),
                (z + radius - bounds.hi[2], (0.0, 0.0, -1.0)),
            ):
                if depth > best_depth:
                    best, best_depth = normal, depth
        for box in self.boxes:
            if not box.solid:
                continue
            lo, hi = box.lo, box.hi
            if (x < lo[0] - radius or x > hi[0] + radius or y < lo[1] - radius
                    or y > hi[1] + radius or z < lo[2] - radius or z > hi[2] + radius):
                continue
            nearest_x = min(max(x, lo[0]), hi[0])
            nearest_y = min(max(y, lo[1]), hi[1])
            nearest_z = min(max(z, lo[2]), hi[2])
            dx, dy, dz = x - nearest_x, y - nearest_y, z - nearest_z
            squared = dx * dx + dy * dy + dz * dz
            if squared > 0.0:
                distance = math.sqrt(squared)
                depth = radius - distance
                if depth > best_depth:
                    best, best_depth = (dx / distance, dy / distance, dz / distance), depth
            else:
                # Centre inside the box: push out through the nearest face.
                for depth, normal in (
                    (x - lo[0], (-1.0, 0.0, 0.0)), (hi[0] - x, (1.0, 0.0, 0.0)),
                    (y - lo[1], (0.0, -1.0, 0.0)), (hi[1] - y, (0.0, 1.0, 0.0)),
                    (z - lo[2], (0.0, 0.0, -1.0)), (hi[2] - z, (0.0, 0.0, 1.0)),
                ):
                    if depth + radius > best_depth:
                        best, best_depth = normal, depth + radius
        return None if best is None else (best, best_depth)

    def ground(self, x: float, y: float, z: float) -> tuple[float, str]:
        """Height and material of the surface under a point."""
        top, material = 0.0, self.floor_material
        for box in self.boxes:
            lo, hi = box.lo, box.hi
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and top < hi[2] <= z:
                top, material = hi[2], box.material
        return top, material
