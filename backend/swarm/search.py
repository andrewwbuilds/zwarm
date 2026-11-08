"""Searching a room with several drones: track which floor has been looked at, and share out the rest.

Coverage counts looks at each floor patch, using only the map and the drones' poses. A patch needs a
few looks, since one frame can miss. Each free drone gets the viewpoint worth the most new floor per
second of flying, and floor another drone is already heading to see doesn't count, so they spread out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from swarm.eyes import DEFAULT_TILT, EDGE_MARGIN, RELIABLE_REACH, blocked, in_view, to_camera
from swarm.room import Room

CELL = 0.5
FLOOR_Z = 0.1  # the lowest thing worth finding: anything taller is in the picture sooner
SEARCH_Z = 1.0
LOOKS_NEEDED = 5  # at the 5 looks a second the hub takes, about one second in view
TRAVEL_SPEED = 0.4
TURN_SPEED = 0.55
DWELL_S = 2.0
RESET_RADIUS = 2.0  # a new obstacle may have hidden floor that was counted as seen, so look again
KEEP_AWAY = 1.2  # metres between a viewpoint and any other drone or its goal
MIN_GAIN = 2  # a viewpoint must offer at least this many new patches


@dataclass
class Viewpoint:
    p: np.ndarray
    yaw: float
    cells: np.ndarray  # indices of the floor patches it would see


class Coverage:
    def __init__(self, room: Room, tilts=(DEFAULT_TILT,)):
        self.room = room
        lo, hi = room.lo, room.hi
        xs = np.arange(lo[0] + CELL / 2, hi[0], CELL)
        ys = np.arange(lo[1] + CELL / 2, hi[1], CELL)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        self.cells = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, FLOOR_Z)], axis=1)
        self.counts = np.zeros(len(self.cells), int)
        self._version = -1
        self.open = np.ones(len(self.cells), bool)
        self.candidates = np.zeros((0, 3))
        self.candidate_yaws = np.zeros(0)
        self.tilts = tuple(sorted(set(tilts) or {DEFAULT_TILT}))
        self.sees: dict[float, np.ndarray] = {}  # camera tilt -> which viewpoint sees which floor patch
        self.clearance_needed = 0.38

    def _view(self, p, yaw, tilt: float = DEFAULT_TILT) -> np.ndarray:
        mask = in_view(p, yaw, self.cells, reach=RELIABLE_REACH, margin=EDGE_MARGIN, tilt=tilt)
        if mask.any():
            lines = self.cells[mask]
            clear = np.ones(len(lines), bool)
            for o in self.room.obstacles.values():
                clear &= ~blocked(p, lines, o.lo, o.hi)
            indices = np.flatnonzero(mask)
            mask[indices[~clear]] = False
        return mask

    def _refresh(self) -> None:
        if self._version == self.room.version:
            return
        self._version = self.room.version
        self.open = np.ones(len(self.cells), bool)
        for o in self.room.obstacles.values():
            inside = np.all((self.cells[:, :2] >= np.array(o.lo[:2])) & (self.cells[:, :2] <= np.array(o.hi[:2])), axis=1)
            self.open &= ~inside
        places = []
        step = CELL
        for x in np.arange(self.room.lo[0] + step, self.room.hi[0], step):
            for y in np.arange(self.room.lo[1] + step, self.room.hi[1], step):
                p = np.array([x, y, SEARCH_Z])
                if self.room.clearance(p) >= self.clearance_needed:
                    places.append(p)
        yaws = [i * math.pi / 4 for i in range(8)]
        self.candidates = np.array([p for p in places for _ in yaws]).reshape(-1, 3)
        self.candidate_yaws = np.array([yaw for _ in places for yaw in yaws])
        self.sees = {tilt: np.array([self._view(p, yaw, tilt) for p, yaw in zip(self.candidates, self.candidate_yaws)]).reshape(
            len(self.candidates), len(self.cells)) for tilt in self.tilts}

    def _reachable(self) -> np.ndarray:
        reachable = np.zeros(len(self.cells), bool)
        for sees in self.sees.values():
            if len(sees):
                reachable |= sees.any(axis=0)
        return reachable

    def look(self, p, yaw: float, tilt: float = DEFAULT_TILT) -> None:
        self.counts[self._view(p, yaw, tilt)] += 1

    def obstacle_appeared(self, lo, hi) -> None:
        """Whatever was behind this was counted as seen when it wasn't, so those patches go back on the list."""
        center = (np.array(lo[:2]) + np.array(hi[:2])) / 2
        near = np.linalg.norm(self.cells[:, :2] - center, axis=1) < RESET_RADIUS
        self.counts[near] = 0

    def remaining(self) -> np.ndarray:
        """Floor patches that can be seen from somewhere and haven't been looked at enough."""
        self._refresh()
        return self.open & self._reachable() & (self.counts < LOOKS_NEEDED)

    def share_seen(self) -> float:
        self._refresh()
        reachable = self.open & self._reachable()
        total = int(reachable.sum())
        return 1.0 if total == 0 else float((reachable & (self.counts >= LOOKS_NEEDED)).sum()) / total

    def best(self, p, yaw: float, claimed: np.ndarray, avoid=(), keep_away: float = 0.7, banned=(), tilt: float = DEFAULT_TILT) -> Viewpoint | None:
        """The viewpoint giving this drone the most new floor per second."""
        wanted = self.remaining() & ~claimed
        if not wanted.any() or not len(self.candidates) or tilt not in self.sees:
            return None
        sees = self.sees[tilt]
        gain = (sees & wanted).sum(axis=1).astype(float)
        gain[gain < MIN_GAIN] = 0.0
        distance = np.linalg.norm(self.candidates[:, :2] - np.asarray(p, float)[:2], axis=1)
        turn = np.abs((self.candidate_yaws - yaw + math.pi) % (2 * math.pi) - math.pi)
        score = gain / (DWELL_S + distance / TRAVEL_SPEED + turn / TURN_SPEED)
        for q in avoid:
            score[np.linalg.norm(self.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) < keep_away] = 0.0
        for q in banned:
            score[np.linalg.norm(self.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) < 0.3] = 0.0
        index = int(np.argmax(score))
        if score[index] <= 0.0:
            return None
        return Viewpoint(self.candidates[index].copy(), float(self.candidate_yaws[index]), np.flatnonzero(sees[index] & wanted))

    def standoff(self, p, target, avoid=(), keep_away: float = 0.7, tilt: float = DEFAULT_TILT) -> Viewpoint | None:
        """The nearest place to get a clear close look at a thing, near the middle of the picture."""
        self._refresh()
        if not len(self.candidates):
            return None
        target = np.asarray(target, float)
        offset = target - self.candidates
        flat = np.hypot(offset[:, 0], offset[:, 1])
        axis, left, _ = to_camera(offset, self.candidate_yaws, tilt)
        bearing = np.arctan2(left, axis)
        usable = (flat >= 0.5) & in_view(self.candidates, self.candidate_yaws, target, reach=RELIABLE_REACH, margin=EDGE_MARGIN, tilt=tilt)
        for q in avoid:
            usable &= np.linalg.norm(self.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) >= keep_away
        # Straight down the camera's axis is the sharpest view and the best floor position.
        ideal = min(2.0, max(0.8, (SEARCH_Z - float(target[2])) / max(math.tan(tilt), 1e-3)))
        best, best_cost = None, math.inf
        for index in np.flatnonzero(usable):
            cost = (float(np.linalg.norm(self.candidates[index, :2] - np.asarray(p, float)[:2]))
                    + 0.5 * abs(float(bearing[index])) + abs(float(flat[index]) - ideal))
            if cost >= best_cost:
                continue
            clear = True
            for o in self.room.obstacles.values():
                # The thing itself is on the map as an obstacle, so a box around the target doesn't block the view of it.
                if np.all(target >= np.array(o.lo) - 0.2) and np.all(target <= np.array(o.hi) + 0.2):
                    continue
                if blocked(self.candidates[index], target[None, :], o.lo, o.hi)[0]:
                    clear = False
                    break
            if clear:
                best, best_cost = index, cost
        if best is None:
            return None
        return Viewpoint(self.candidates[best].copy(), float(self.candidate_yaws[best]), np.zeros(0, int))
