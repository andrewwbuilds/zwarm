"""The last check before any stick moves: change each requested velocity as little as possible
so that no drone can reach another drone, a wall or an obstacle.

Each rule says the gap may only close at a rate proportional to what's left of it:

    n . (v_i - v_j) >= -gain * (distance - keep_out)        between two drones
    n . v_i         >= -gain * (distance - keep_out)        against a surface

As a gap shrinks to its keep-out distance the allowed closing speed shrinks to zero, and inside it
the rule demands separation. This is a control barrier function, solved for all drones together.

A neighbour that can't be commanded (lost link, lost tracking, not ours) gets its gap handled
entirely by the drone that can still act, using that neighbour's measured velocity.

Distances are measured between stopping points, not current positions, because these drones take
about half a second to respond. Vertical gaps count for less, because air pushed down by a drone
overhead makes flying underneath it unsafe at distances that would be fine side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swarm.room import Room


@dataclass(frozen=True)
class Limits:
    max_speed: float = 0.5  # m/s on each horizontal axis
    max_climb: float = 0.3
    drone_radius: float = 0.12  # prop tip reach
    pose_error: float = 0.06  # how far off a tracked position may be
    margin: float = 0.14  # spare, on top of radius and pose error, between two drones
    surface_margin: float = 0.1  # spare between a drone and a surface
    gain: float = 1.0  # 1/s: how fast a gap may close per metre left. Must stay well under 1/response
    response: float = 0.5  # s from command to the drone mostly obeying it
    latency: float = 0.15  # s from a camera frame to the resulting stick reaching the drone
    vertical_stretch: float = 2.5  # a vertical gap has to be this much bigger than a horizontal one
    moving_margin: float = 0.25  # spare around tracked obstacles such as people

    @property
    def separation(self) -> float:
        return 2 * (self.drone_radius + self.pose_error) + self.margin

    @property
    def surface_clearance(self) -> float:
        return self.drone_radius + self.pose_error + self.surface_margin

    @property
    def lookahead(self) -> float:
        return self.latency + 0.5 * self.response


@dataclass
class Body:
    """One drone as the filter sees it."""

    id: str
    p: np.ndarray
    v: np.ndarray
    want: np.ndarray = field(default_factory=lambda: np.zeros(3))
    controllable: bool = True
    extra_gap: float = 0.0  # how much further to stay away, for a pose that may be off
    descending: bool = False  # landing: the floor is where it's meant to go


@dataclass
class Moving:
    """A tracked thing that isn't a drone: a tall column at p, moving at v."""

    id: str
    p: np.ndarray
    v: np.ndarray
    radius: float = 0.3


@dataclass
class Result:
    velocities: dict[str, np.ndarray]
    limited: set[str]  # drones whose request had to change
    infeasible: bool  # the rules contradict each other: the output is the best compromise
    closest: float  # smallest stretched gap between stopping points, minus the separation


def _gap(a: np.ndarray, b: np.ndarray, stretch: float) -> tuple[np.ndarray, float]:
    """Unit direction from b to a in the stretched metric's gradient, and the stretched distance."""
    offset = a - b
    scaled = np.array([offset[0], offset[1], offset[2] / stretch])
    distance = float(np.linalg.norm(scaled))
    if distance < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    gradient = np.array([scaled[0], scaled[1], scaled[2] / stretch]) / distance
    return gradient, distance


def _project(start: np.ndarray, rows: np.ndarray, floors: np.ndarray, sweeps: int = 300, tolerance: float = 1e-4):
    """Closest point to start with rows @ x >= floors (Hildreth's method). Returns it and the worst miss."""
    x = start.copy()
    if len(rows) == 0:
        return x, 0.0
    weights = np.zeros(len(rows))
    norms = np.einsum("ij,ij->i", rows, rows)
    for _ in range(sweeps):
        worst = 0.0
        for k in range(len(rows)):
            step = (floors[k] - rows[k] @ x) / norms[k]
            new = max(0.0, weights[k] + step)
            change = new - weights[k]
            if change != 0.0:
                x += change * rows[k]
                weights[k] = new
            worst = max(worst, abs(change) * norms[k] ** 0.5)
        if worst < tolerance:
            break
    return x, float(np.max(floors - rows @ x))


def filter_velocities(bodies: list[Body], room: Room, limits: Limits, moving: list[Moving] = ()) -> Result:
    free = [b for b in bodies if b.controllable]
    slot = {b.id: 3 * i for i, b in enumerate(free)}
    size = 3 * len(free)
    want = np.zeros(size)
    for b in free:
        want[slot[b.id]:slot[b.id] + 3] = b.want
    ahead = {b.id: b.p + b.v * limits.lookahead for b in bodies}
    rows, floors = [], []

    def rule(entries: list[tuple[str, np.ndarray]], floor: float) -> None:
        row = np.zeros(size)
        for body_id, direction in entries:
            row[slot[body_id]:slot[body_id] + 3] = direction
        rows.append(row)
        floors.append(floor)

    closest = np.inf
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if not (a.controllable or b.controllable):
                continue
            n, distance = _gap(ahead[a.id], ahead[b.id], limits.vertical_stretch)
            keep = limits.separation + a.extra_gap + b.extra_gap
            closest = min(closest, distance - keep)
            floor = -limits.gain * (distance - keep)
            if a.controllable and b.controllable:
                rule([(a.id, n), (b.id, -n)], floor)
            elif a.controllable:
                rule([(a.id, n)], floor + n @ b.v)
            else:
                rule([(b.id, -n)], floor - n @ a.v)

    keep = limits.surface_clearance
    reach = keep + limits.max_speed / limits.gain + 0.2
    for b in free:
        for n, distance in room.surfaces(ahead[b.id], reach + b.extra_gap):
            if b.descending and n[2] > 0.5:
                continue
            rule([(b.id, n)], -limits.gain * (distance - keep - b.extra_gap))
        for thing in moving:
            offset = (ahead[b.id] - (thing.p + thing.v * limits.lookahead))[:2]
            distance = float(np.linalg.norm(offset))
            n = np.array([1.0, 0.0, 0.0]) if distance < 1e-6 else np.array([offset[0] / distance, offset[1] / distance, 0.0])
            gap = distance - (thing.radius + limits.drone_radius + limits.pose_error + limits.moving_margin)
            closest = min(closest, gap)
            rule([(b.id, n)], -limits.gain * gap + n @ thing.v)

    # Speed limits. These apply last, so a drone never exceeds them even to escape.
    for b in free:
        for axis in range(3):
            top = limits.max_climb if axis == 2 else limits.max_speed
            for sign in (1.0, -1.0):
                direction = np.zeros(3)
                direction[axis] = sign
                rule([(b.id, direction)], -top)

    table = np.array(rows) if rows else np.zeros((0, size))
    x, miss = _project(want, table, np.array(floors))
    out, limited = {}, set()
    for b in free:
        chosen = x[slot[b.id]:slot[b.id] + 3].copy()
        chosen[:2] = np.clip(chosen[:2], -limits.max_speed, limits.max_speed)
        chosen[2] = np.clip(chosen[2], -limits.max_climb, limits.max_climb)
        out[b.id] = chosen
        if np.linalg.norm(chosen - b.want) > 0.02:
            limited.add(b.id)
    return Result(out, limited, miss > 0.02, float(closest))
