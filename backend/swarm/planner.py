"""Paths around known obstacles: A* on a 3D grid, then straightened.

The safety filter alone can't do this. It only ever steers away from the nearest surface, so a
drone with a sofa between it and its goal would stop in front of the sofa forever.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from swarm.room import FLOOR_CLEARANCE, Room

CELL = 0.2
CLIMB_COST = 1.5  # prefer going around over changing height: the barometer hold is the weak axis
_MOVES = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]


class PlanError(ValueError):
    pass


class Grid:
    """Which cells a drone's centre may occupy with the given clearance."""

    def __init__(self, room: Room, clearance: float, cylinders=(), floor: float = FLOOR_CLEARANCE):
        self.room, self.clearance, self.origin = room, clearance, room.lo
        shape = np.maximum(np.ceil((room.hi - room.lo) / CELL).astype(int) + 1, 2)
        self.shape = tuple(int(n) for n in shape)
        axes = [room.lo[i] + CELL * np.arange(shape[i]) for i in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        free = (x - room.lo[0] >= clearance) & (room.hi[0] - x >= clearance)
        free &= (y - room.lo[1] >= clearance) & (room.hi[1] - y >= clearance)
        free &= (z >= room.lo[2] + floor) & (room.hi[2] - z >= clearance)
        for o in room.obstacles.values():
            lo, hi = room.flight_box(o)
            dx = np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0.0)
            dy = np.maximum(np.maximum(lo[1] - y, y - hi[1]), 0.0)
            dz = np.maximum(np.maximum(lo[2] - z, z - hi[2]), 0.0)
            free &= dx * dx + dy * dy + dz * dz >= clearance * clearance
        for (cx, cy), radius in cylinders:
            free &= (x - cx) ** 2 + (y - cy) ** 2 >= radius * radius
        self.free = free

    def cell(self, p) -> tuple[int, int, int]:
        index = np.rint((np.asarray(p, float) - self.origin) / CELL).astype(int)
        return tuple(int(v) for v in np.clip(index, 0, np.array(self.shape) - 1))

    def point(self, cell) -> np.ndarray:
        return self.origin + CELL * np.array(cell, float)

    def nearest_free(self, p, limit: float = 1.0):
        """The closest usable cell, for a drone that has been pushed into a margin."""
        p = np.asarray(p, float)
        centre = self.cell(p)
        span = int(limit / CELL) + 1
        best, best_distance = None, math.inf
        for offset in itertools.product(range(-span, span + 1), repeat=3):
            cell = tuple(c + o for c, o in zip(centre, offset))
            if not all(0 <= c < n for c, n in zip(cell, self.shape)) or not self.free[cell]:
                continue
            distance = float(np.linalg.norm(self.point(cell) - p))
            if distance < best_distance:
                best, best_distance = cell, distance
        return best if best_distance <= limit else None

    def line_clear(self, a, b) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(int(np.linalg.norm(b - a) / (CELL / 2)), 1)
        return all(self.free[self.cell(a + (b - a) * i / steps)] for i in range(steps + 1))


def _search(grid: Grid, start, goal) -> list | None:
    free = grid.free
    gx, gy, gz = goal
    costs = {move: math.sqrt(move[0] ** 2 + move[1] ** 2 + (CLIMB_COST * move[2]) ** 2) for move in _MOVES}

    def estimate(c):
        return math.sqrt((c[0] - gx) ** 2 + (c[1] - gy) ** 2 + (CLIMB_COST * (c[2] - gz)) ** 2)

    order = itertools.count()
    frontier = [(estimate(start), next(order), start)]
    best = {start: 0.0}
    came = {}
    nx, ny, nz = grid.shape
    while frontier:
        _, _, here = heapq.heappop(frontier)
        if here == goal:
            path = [here]
            while here in came:
                here = came[here]
                path.append(here)
            return path[::-1]
        base = best[here]
        for move, step in costs.items():
            cell = (here[0] + move[0], here[1] + move[1], here[2] + move[2])
            if not (0 <= cell[0] < nx and 0 <= cell[1] < ny and 0 <= cell[2] < nz) or not free[cell]:
                continue
            # No cutting corners: the diagonal has to be free along both of its sides.
            if (move[0] and move[1] and not (free[here[0] + move[0], here[1], here[2]]
                                             and free[here[0], here[1] + move[1], here[2]])):
                continue
            cost = base + step
            if cost < best.get(cell, math.inf):
                best[cell] = cost
                came[cell] = here
                heapq.heappush(frontier, (cost + estimate(cell), next(order), cell))
    return None


def plan_path(room: Room, start, goal, clearance: float, cylinders=()) -> list[np.ndarray]:
    """Waypoints from start to goal, keeping clearance from every surface, ending exactly at goal."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    grid = Grid(room, clearance, cylinders)
    if grid.line_clear(start, goal):
        return [start, goal]
    begin = grid.cell(start) if grid.free[grid.cell(start)] else grid.nearest_free(start)
    finish = grid.cell(goal) if grid.free[grid.cell(goal)] else grid.nearest_free(goal, limit=CELL * 1.5)
    if begin is None:
        raise PlanError(f"start {start.round(2).tolist()} is too close to a surface to plan from")
    if finish is None:
        raise PlanError(f"goal {goal.round(2).tolist()} is too close to a surface")
    cells = _search(grid, begin, finish)
    if cells is None:
        raise PlanError(f"no path from {start.round(2).tolist()} to {goal.round(2).tolist()} with {clearance:.2f} m clearance")
    points = [start, *(grid.point(c) for c in cells), goal]
    straight = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.line_clear(points[i], points[j]):
            j -= 1
        straight.append(points[j])
        i = j
    return straight
