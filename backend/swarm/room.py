"""The flying space: a fence, the solid things in it, and how far a point is from each.

World axes match the simulator: x forward, y left, z up, in metres, origin wherever the room
file puts it. The fence is the volume drones may use, which can be smaller than the real room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOOR_CLEARANCE = 0.25  # the floor is a surface too, but takeoff and landing have to reach it
SEEN_PREFIX = "seen-"  # the hub's name for what a camera found, as against what the swarm file lists
TALLER_M = 10.0  # past the fence, so a drone pushed inside a full-height obstacle only ever leaves sideways


@dataclass(frozen=True)
class Obstacle:
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


class Room:
    def __init__(self, lo, hi, obstacles=()):
        self.lo = np.array(lo, float)
        self.hi = np.array(hi, float)
        if np.any(self.hi <= self.lo):
            raise ValueError("the fence's hi corner must be above its lo corner on every axis")
        self.obstacles: dict[str, Obstacle] = {}
        self.version = 0  # goes up when the map changes, so paths know to replan
        self.fly_around = False  # nothing flies over a listed obstacle: on floor tags it would hide them
        for obstacle in obstacles:
            self.add(obstacle)

    @classmethod
    def from_scene(cls, scene, fence=None) -> "Room":
        """The map of a simulator scene. Open-air scenes need a fence as (lo, hi)."""
        if fence is None:
            if scene.bounds is None:
                raise ValueError(f"scene {scene.name!r} has no walls, so it needs a fence")
            fence = (scene.bounds.lo, scene.bounds.hi)
        solid = [Obstacle(f"{box.material}-{i}", box.lo, box.hi) for i, box in enumerate(scene.boxes) if box.solid]
        return cls(fence[0], fence[1], solid)

    @classmethod
    def from_dict(cls, data: dict) -> "Room":
        obstacles = [Obstacle(o.get("name", f"obstacle-{i}"), tuple(o["lo"]), tuple(o["hi"]))
                     for i, o in enumerate(data.get("obstacles", ()))]
        return cls(data["fence"]["lo"], data["fence"]["hi"], obstacles)

    @classmethod
    def load(cls, path: str | Path) -> "Room":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def add(self, obstacle: Obstacle) -> None:
        if any(h <= l for l, h in zip(obstacle.lo, obstacle.hi)):
            raise ValueError(f"obstacle {obstacle.name!r} has no volume")
        self.obstacles[obstacle.name] = obstacle
        self.version += 1

    def remove(self, name: str) -> bool:
        if self.obstacles.pop(name, None) is None:
            return False
        self.version += 1
        return True

    def surfaces(self, p, reach: float) -> list[tuple[np.ndarray, float]]:
        """Every surface within reach of p, as (unit normal pointing back at p, distance).

        The distance goes negative when p is past the surface.
        """
        out = []
        for axis in range(3):
            below, above = p[axis] - self.lo[axis], self.hi[axis] - p[axis]
            if below < reach:
                normal = np.zeros(3)
                normal[axis] = 1.0
                out.append((normal, below))
            if above < reach:
                normal = np.zeros(3)
                normal[axis] = -1.0
                out.append((normal, above))
        for obstacle in self.obstacles.values():
            lo, hi = np.array(obstacle.lo), np.array(obstacle.hi)
            nearest = np.clip(p, lo, hi)
            offset = p - nearest
            distance = float(np.linalg.norm(offset))
            if distance > reach:
                continue
            if distance > 1e-9:
                out.append((offset / distance, distance))
                continue
            # Inside the box: leave through the closest face.
            faces = np.concatenate([p - lo, hi - p])
            face = int(np.argmin(faces))
            normal = np.zeros(3)
            normal[face % 3] = -1.0 if face < 3 else 1.0
            out.append((normal, -float(faces[face])))
        return out

    def clearance(self, p) -> float:
        """Distance from p to the nearest wall, ceiling or obstacle, negative inside one.

        The floor is left out, since drones start and finish on it.
        """
        p = np.asarray(p, float)
        best = float(min(np.min(p[:2] - self.lo[:2]), np.min(self.hi[:2] - p[:2]), self.hi[2] - p[2]))
        for obstacle in self.obstacles.values():
            lo, hi = np.array(obstacle.lo), np.array(obstacle.hi)
            outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
            distance = float(np.linalg.norm(outside))
            if distance == 0.0:
                distance = -float(min(np.min(p - lo), np.min(hi - p)))
            best = min(best, distance)
        return best

    def json(self) -> dict:
        return {
            "fence": {"lo": self.lo.tolist(), "hi": self.hi.tolist()},
            "obstacles": [{"name": o.name, "lo": list(o.lo), "hi": list(o.hi)} for o in self.obstacles.values()],
            "version": self.version,
        }
