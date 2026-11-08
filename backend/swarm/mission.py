"""What the swarm is trying to do: explore, map, inspect or search a region, and who does which part.

The hub and coordinator keep the drones flying and apart. This only chooses where each one goes next,
and it plugs into the hub the way a search does: hub.search is anything with step(now), state, label
and coverage. Hub actions such as hold and land stop that object, so the operator can always override
a mission through the hub, with or without the model.

The camera points down at a fixed tilt and can't see above the horizon. So what gets covered is the
floor and low surfaces, and a thing is only inspected from where it is actually in the picture.

Work lives in a pool. A drone holds a piece only while it flies to it or looks at it, so when a drone
drops out its piece is free again and the next free drone takes it.

tick() runs on the flight loop. Anything that may be slow (whoever listens to events, map capture, the
inspector) runs on a background thread instead, and a failure there can't reach the flight loop.
"""

from __future__ import annotations

import logging
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from swarm.commands import clean_label
from swarm.coordinator import Phase
from swarm.eyes import DEFAULT_TILT, EDGE_MARGIN, RELIABLE_REACH, blocked, in_view
from swarm.search import CELL, LOOKS_NEEDED, MIN_GAIN, SEARCH_Z, Coverage, Viewpoint

logger = logging.getLogger(__name__)

KINDS = ("explore", "map", "inspect", "search")
REGION_NAMES = ("all", "north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west", "center")
_FILLER = {"the", "a", "side", "half", "part", "area", "areas", "corner", "quarter", "of", "room", "section", "end", "this",
           "entire", "whole", "floor", "in", "on", "to", "please", "space", "map"}
_SHORT = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
          "sw": "southwest", "eastnorth": "northeast", "westnorth": "northwest", "eastsouth": "southeast",
          "westsouth": "southwest", "centre": "center", "middle": "center", "central": "center", "": "all",
          "everything": "all", "everywhere": "all"}

_VAGUE = re.compile(r"(these |those |all |the )*(things?|stuff|objects?|items?|everything|anything|them|it)?")

WARN_BEFORE_LIMIT_S = 30.0
RETURN_SPEED = 0.4  # m/s, a cautious guess at the speed back to the pad when working out when to leave
LANDING_TIME_S = 8.0  # to line up over the pad and descend
MIN_FLIGHT_LEFT_S = 15.0  # with less than this, a new mission isn't worth starting
DROP_GRACE_S = 1.0  # a short tracking gap isn't a lost drone
DWELL_S = 1.0  # at a floor viewpoint: enough for the looks a patch of floor needs
VIEW_DWELL_S = 4.0  # at an inspection viewpoint: long enough to settle and give a sharp frame
VIEW_SETTLE_S = 0.5  # the inspector only sees frames once a drone has stopped swinging into place
MOVE_LIMIT_S = 30.0  # enough to cross a large room at the speed limit and go around something
KEEP_AWAY = 0.7
AT_EDGE_M = 0.25  # this close to a place in the region counts as being in it
OBSTACLE_RESET_M = 1.0  # smaller than a search's 2 m: a noisy detector mustn't keep sending a mapping flight back
VIEW_BEARINGS = 8
VIEW_GAPS = (0.5, 0.7, 0.9, 1.2, 1.5)  # metres from the edge of a thing to a drone inspecting it
VIEWS_PER_THING = 4  # all the way round: damage can be on any side
VIEWS_PER_THING_IN_REGION = 2
MAX_THINGS = 6
MAX_FLOOR_VIEWS = 12
MIN_VIEW_Z = 0.7
FAILED_VIEW_TRIES = 2
ANALYSIS_LIMIT_S = 180.0  # a stuck model mustn't leave a mission "analyzing" for ever
LOOK_EVERY = 0.2
VIDEO_LATE_S = 0.15  # measured: video arrives 100 to 200 ms after the picture was taken


class MissionError(ValueError):
    """A mission that can't be done. The message is a sentence an operator can read."""


class Regions:
    """Named parts of the floor, computed from the fence. North is +x unless north_yaw_deg says otherwise."""

    def __init__(self, room, north_yaw_deg: float = 0.0):
        turn = math.radians(north_yaw_deg)
        self.center = (room.lo[:2] + room.hi[:2]) / 2
        self.north = np.array([math.cos(turn), math.sin(turn)])
        self.east = np.array([math.sin(turn), -math.cos(turn)])  # to the right when facing north
        corners = np.array([[x, y] for x in (room.lo[0], room.hi[0]) for y in (room.lo[1], room.hi[1])]) - self.center
        self.half_north = float(np.max(np.abs(corners @ self.north)))
        self.half_east = float(np.max(np.abs(corners @ self.east)))

    def resolve(self, text) -> str:
        """A region's proper name from what someone might say, such as 'the north side'."""
        words = re.sub(r"[^a-z ]+", " ", str(text if text is not None else "all").lower()).split()
        said = "".join(word for word in words if word not in _FILLER)
        said = _SHORT.get(said, said)
        for name in REGION_NAMES:
            if said == name.replace("-", ""):
                return name
        raise MissionError(f'I don\'t know a region called "{text}". I know: {", ".join(REGION_NAMES)}.')

    def mask(self, name: str, points) -> np.ndarray:
        """Which of these room points lie in the region."""
        offset = np.atleast_2d(points)[:, :2] - self.center
        north, east = offset @ self.north, offset @ self.east
        if name == "center":
            return (np.abs(north) <= self.half_north / 2) & (np.abs(east) <= self.half_east / 2)
        inside = np.ones(len(offset), bool)
        if "north" in name:
            inside &= north >= 0.0
        if "south" in name:
            inside &= north < 0.0
        if "east" in name:
            inside &= east >= 0.0
        if "west" in name:
            inside &= east < 0.0
        return inside


def spoken(region: str) -> str:
    if region == "all":
        return "the whole room"
    return "the center" if region == "center" else f"the {region} side"


class RegionCoverage(Coverage):
    """Coverage with one region switched on. Looks count everywhere, and only the region's floor is asked for.

    Viewpoints are limited to the region too, so a focused mission really moves the drones there.
    One of these lasts the whole session, so what's been seen survives a change of region or mission.
    """

    def __init__(self, room, tilts, regions: Regions, usable=None):
        super().__init__(room, tilts)
        self.regions = regions
        self.region = "all"
        self.usable = usable  # usable(p, yaw): whether a drone can hold this viewpoint at all
        self.usable_known: dict[tuple, bool] = {}
        self.hub_looks = True  # off when real camera frames are counted instead of the detector's answers
        self.in_region = np.ones(len(self.cells), bool)
        self.ever = np.zeros(len(self.cells), bool)  # seen at some point, even if counted again since
        self.all_candidates = np.zeros((0, 3))
        self.all_yaws = np.zeros(0)
        self.all_sees: dict[float, np.ndarray] = {}
        self.anywhere = np.zeros(len(self.cells), bool)

    def look(self, p, yaw: float, tilt: float = DEFAULT_TILT) -> None:
        """What the hub calls whenever its detector has answered."""
        if self.hub_looks:
            super().look(p, yaw, tilt)

    def count(self, p, yaw: float, tilt: float = DEFAULT_TILT) -> None:
        super().look(p, yaw, tilt)

    def _can_hold(self, p, yaw: float) -> bool:
        key = (round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), round(float(yaw), 2))
        if key not in self.usable_known:
            try:
                self.usable_known[key] = bool(self.usable(p, yaw))
            except Exception:
                logger.exception("[mission] viewpoint check failed; treating it as usable")
                self.usable_known[key] = True
        return self.usable_known[key]

    def _refresh(self) -> None:
        if self._version == self.room.version:
            return
        super()._refresh()
        if self.usable is not None and len(self.candidates):
            keep = np.array([self._can_hold(p, yaw) for p, yaw in zip(self.candidates, self.candidate_yaws)], bool)
            self.candidates, self.candidate_yaws = self.candidates[keep], self.candidate_yaws[keep]
            self.sees = {tilt: sees[keep] for tilt, sees in self.sees.items()}
        self.all_candidates, self.all_yaws, self.all_sees = self.candidates, self.candidate_yaws, self.sees
        self.anywhere = np.zeros(len(self.cells), bool)
        for sees in self.all_sees.values():
            if len(sees):
                self.anywhere |= sees.any(axis=0)
        self._narrow()

    def _narrow(self) -> None:
        keep = self.regions.mask(self.region, self.all_candidates)
        self.candidates, self.candidate_yaws = self.all_candidates[keep], self.all_yaws[keep]
        self.sees = {tilt: sees[keep] for tilt, sees in self.all_sees.items()}
        self.in_region = self.regions.mask(self.region, self.cells)

    def _reachable(self) -> np.ndarray:
        return super()._reachable() & self.in_region

    def set_region(self, name: str) -> None:
        self._refresh()
        self.region = name
        self._narrow()

    def forget_region(self) -> None:
        """Fly this region again from the start. What was seen before still counts as ever seen."""
        self.ever |= self.counts >= LOOKS_NEEDED
        self.counts[self.in_region] = 0

    def obstacle_appeared(self, lo, hi) -> None:
        center = (np.array(lo[:2]) + np.array(hi[:2])) / 2
        near = np.linalg.norm(self.cells[:, :2] - center, axis=1) < OBSTACLE_RESET_M
        self.counts[near] = 0
        self.ever[near] = False

    def _seen(self) -> np.ndarray:
        return self.ever | (self.counts >= LOOKS_NEEDED)

    def stats(self, name: str) -> dict:
        """How much of a region's floor has ever been seen, whichever region is switched on."""
        self._refresh()
        floor = self.regions.mask(name, self.cells) & self.open
        reachable = floor & self.anywhere
        seen = reachable & self._seen()
        total = int(reachable.sum())
        return {"share_seen": 1.0 if total == 0 else round(float(seen.sum()) / total, 3),
                "unseen_m2": round((total - int(seen.sum())) * CELL * CELL, 2),
                "out_of_view_m2": round(int((floor & ~self.anywhere).sum()) * CELL * CELL, 2)}

    def _unseen_in(self, name: str) -> np.ndarray:
        return self.regions.mask(name, self.cells) & self.open & self.anywhere & ~self._seen()

    def suggest(self, name: str) -> Viewpoint | None:
        """The viewpoint showing the most unseen floor in this region, for a person or a planner to go on."""
        self._refresh()
        wanted = self._unseen_in(name)
        if not wanted.any() or not self.all_sees or not len(self.all_candidates):
            return None
        sees = next(iter(self.all_sees.values()))
        gain = (sees & wanted).sum(axis=1)
        inside = gain * self.regions.mask(name, self.all_candidates)
        if inside.max() > 0:
            gain = inside
        index = int(np.argmax(gain))
        return Viewpoint(self.all_candidates[index].copy(), float(self.all_yaws[index]), np.flatnonzero(sees[index] & wanted))

    def leftovers(self, p, claimed: np.ndarray, avoid, keep_away: float, banned, tilt: float) -> Viewpoint | None:
        """The closest viewpoint with anything at all left to see: the last patches best() thinks too small."""
        wanted = self.remaining() & ~claimed
        if not wanted.any() or tilt not in self.sees or not len(self.candidates):
            return None
        sees = self.sees[tilt]
        distance = np.linalg.norm(self.candidates[:, :2] - np.asarray(p, float)[:2], axis=1)
        distance[~(sees & wanted).any(axis=1)] = math.inf
        for q in avoid:
            distance[np.linalg.norm(self.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) < keep_away] = math.inf
        for q in banned:
            distance[np.linalg.norm(self.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) < 0.3] = math.inf
        index = int(np.argmin(distance))
        if not math.isfinite(distance[index]):
            return None
        return Viewpoint(self.candidates[index].copy(), float(self.candidate_yaws[index]), np.flatnonzero(sees[index] & wanted))

    def cover(self, name: str, limit: int) -> list[Viewpoint]:
        """A few viewpoints that between them show a region's floor, biggest view first."""
        self._refresh()
        if not self.all_sees or not len(self.all_candidates):
            return []
        wanted = self.regions.mask(name, self.cells) & self.open & self.anywhere
        inside = self.regions.mask(name, self.all_candidates)
        sees = next(iter(self.all_sees.values()))
        views = []
        while wanted.any() and len(views) < limit:
            gain = (sees & wanted).sum(axis=1) * inside
            index = int(np.argmax(gain))
            if gain[index] < MIN_GAIN:
                break
            views.append(Viewpoint(self.all_candidates[index].copy(), float(self.all_yaws[index]),
                                   np.flatnonzero(sees[index] & wanted)))
            wanted &= ~sees[index]
        return views


def views_of(thing, coverage: RegionCoverage, tilt: float, want: int) -> list[Viewpoint]:
    """Places to look at a thing from, on different sides, with its middle in the picture.

    The camera looks tilt below level, so the height sets how far back a drone has to be. Something
    tall may have no such place under the ceiling, and then this finds none.
    """
    room = coverage.room
    half = float(max(thing.size[0], thing.size[1])) / 2
    base, top = max(0.0, float(thing.p[2] - thing.size[2] / 2)), float(thing.p[2] + thing.size[2] / 2)
    aim = np.array([thing.p[0], thing.p[1], (base + top) / 2])
    parts = np.array([aim, [thing.p[0], thing.p[1], top], [thing.p[0], thing.p[1], base]])
    ceiling = float(room.hi[2]) - coverage.clearance_needed
    best: dict[int, tuple[float, np.ndarray, float]] = {}
    for side in range(VIEW_BEARINGS):
        out = np.array([math.cos(2 * math.pi * side / VIEW_BEARINGS), math.sin(2 * math.pi * side / VIEW_BEARINGS)])
        for gap in VIEW_GAPS:
            flat = half + gap
            xy = aim[:2] + out * flat
            ideal = aim[2] + flat * math.tan(tilt)  # puts its middle on the camera's axis
            for z in {round(min(max(ideal, MIN_VIEW_Z), ceiling), 2), min(SEARCH_Z, ceiling)}:
                p = np.array([xy[0], xy[1], z])
                if room.clearance(p) < coverage.clearance_needed:
                    continue
                yaw = math.atan2(aim[1] - p[1], aim[0] - p[0])
                showing = in_view(p, yaw, parts, reach=RELIABLE_REACH, margin=EDGE_MARGIN, tilt=tilt)
                if not showing[0]:
                    continue
                hidden = False
                for o in room.obstacles.values():
                    # The thing itself is on the map, so its own box doesn't count as blocking the view.
                    if np.all(aim >= np.array(o.lo) - 0.2) and np.all(aim <= np.array(o.hi) + 0.2):
                        continue
                    if blocked(p, aim[None, :], o.lo, o.hi)[0]:
                        hidden = True
                        break
                if hidden or (coverage.usable is not None and not coverage._can_hold(p, yaw)):
                    continue
                score = int(showing.sum()) - 0.3 * abs(z - SEARCH_Z) - 0.3 * flat
                if side not in best or score > best[side][0]:
                    best[side] = (score, p, yaw)
    chosen: list[int] = []
    for side in sorted(best, key=lambda s: -best[s][0]):
        apart = all(min((side - c) % VIEW_BEARINGS, (c - side) % VIEW_BEARINGS) >= 2 for c in chosen)
        if apart and len(chosen) < want:
            chosen.append(side)
    return [Viewpoint(best[side][1], best[side][2], np.zeros(0, int)) for side in chosen]


class _Later:
    """Runs things in order on one background thread, so a slow or broken one can't touch the flight loop."""

    def __init__(self):
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True, name="mission-later").start()

    def __call__(self, job, *args) -> None:
        self.jobs.put((job, args))

    def _run(self) -> None:
        while True:
            job, args = self.jobs.get()
            try:
                job(*args)
            except Exception:
                logger.exception("[mission] background job failed")

    def wait(self, timeout: float = 5.0) -> bool:
        """Until everything asked for so far has run."""
        done = threading.Event()
        self(done.set)
        return done.wait(timeout)


class Sweep:
    """Shares a region's unseen floor among the flying drones. Sits in hub.search while it runs."""

    def __init__(self, manager: "MissionManager", coverage: RegionCoverage, label: str):
        self.manager, self.coverage, self.label = manager, coverage, label
        self.state = "searching"  # searching, done, failed, stopped. The hub stops anything "searching"
        self.views: dict[str, Viewpoint] = {}
        self.dwell: dict[str, tuple[float, int]] = {}  # arrived when, and how many looks its view had by then
        self.banned: list[np.ndarray] = []

    def release(self, drone_id: str) -> None:
        self.views.pop(drone_id, None)
        self.dwell.pop(drone_id, None)

    def task_text(self, drone_id: str) -> str | None:
        view = self.views.get(drone_id)
        if view is None:
            return None
        if not len(view.cells):
            return f"moving into {spoken(self.coverage.region)}"
        return f"looking at unseen floor from ({view.p[0]:.1f}, {view.p[1]:.1f})"

    def step(self, now: float) -> None:
        if self.state != "searching":
            return
        manager = self.manager
        flying = manager.crew(now)
        if not flying:
            if not manager.can_still_fly(now):
                self.state = "failed"
            return
        hub = manager.hub
        wanted = self.coverage.remaining()
        claimed = np.zeros(len(self.coverage.cells), bool)
        free = []
        for d in flying:
            order, view = hub.orders.get(d.id), self.views.get(d.id)
            manager.give_up_if_slow(d, order, now)
            if order is not None and order.stage == "failed":
                self.banned.append(order.goal.copy())
            elif view is None or order is None:
                pass
            elif order.stage == "done":
                looks = int(self.coverage.counts[view.cells].sum())
                arrived, looks_then = self.dwell.setdefault(d.id, (now, looks))
                if now - arrived < DWELL_S:
                    claimed[view.cells] = True
                    continue
                if len(view.cells) and looks == looks_then:
                    self.banned.append(order.goal.copy())  # no picture counted from there, so going back won't help
            elif not len(view.cells) or wanted[view.cells].any():
                claimed[view.cells] = True
                continue
            hub.orders.pop(d.id, None)
            self.release(d.id)
            free.append(d)
        for d in free:
            view = self._next_view(d, claimed) or self._stand_by(d)
            if view is not None:
                claimed[view.cells] = True
                self.views[d.id] = view
                hub.order(d, view.p, view.yaw, now)
        if not any(d.id in hub.orders for d in flying) and not manager.taking_off():
            self.state = "done"

    def _next_view(self, d, claimed: np.ndarray) -> Viewpoint | None:
        hub = self.manager.hub
        avoid = [other.tracked.p for other in hub.co.drones.values()
                 if other is not d and other.phase in (Phase.AIR, Phase.CHECK, Phase.TAKEOFF, Phase.LANDING)]
        avoid += [order.goal for other_id, order in hub.orders.items() if other_id != d.id]
        return self.coverage.best(d.tracked.p, d.tracked.yaw, claimed, avoid, keep_away=KEEP_AWAY, banned=self.banned,
                                  tilt=hub.tilt_of(d.id))

    def _stand_by(self, d) -> Viewpoint | None:
        """Nothing left for this drone to look at. If it's outside the region, bring it in anyway."""
        coverage = self.coverage
        if coverage.regions.mask(coverage.region, d.tracked.p)[0] or not len(coverage.candidates):
            return None
        hub = self.manager.hub
        distance = np.linalg.norm(coverage.candidates[:, :2] - d.tracked.p[:2], axis=1)
        taken = [other.tracked.p for other in hub.co.drones.values() if other is not d and other.phase != Phase.GROUND]
        taken += [order.goal for other_id, order in hub.orders.items() if other_id != d.id]
        for q in taken:
            distance[np.linalg.norm(coverage.candidates[:, :2] - np.asarray(q, float)[:2], axis=1) < KEEP_AWAY] = math.inf
        index = int(np.argmin(distance))
        if not math.isfinite(distance[index]):
            return None
        return Viewpoint(coverage.candidates[index].copy(), float(coverage.candidate_yaws[index]), np.zeros(0, int))


@dataclass
class ViewTask:
    view: Viewpoint
    text: str
    target: str | None = None
    thing_id: int | None = None
    state: str = "waiting"  # waiting, held, looking, done, skipped
    holder: str | None = None
    last_holder: str | None = None
    arrived: float = 0.0
    tries: int = 0


class Survey:
    """Visits a list of views, one drone at each, and lingers so the inspector gets a sharp frame."""

    def __init__(self, manager: "MissionManager", coverage: RegionCoverage, tasks: list[ViewTask], label: str):
        self.manager, self.coverage, self.tasks, self.label = manager, coverage, tasks, label
        self.state = "searching"

    def held_by(self, drone_id: str) -> ViewTask | None:
        return next((t for t in self.tasks if t.holder == drone_id), None)

    def release(self, drone_id: str) -> None:
        task = self.held_by(drone_id)
        if task is not None:
            task.state, task.last_holder, task.holder = "waiting", drone_id, None

    def settled(self, now: float) -> list[str]:
        """Drones that have been at their view long enough to give a steady picture."""
        return [t.holder for t in self.tasks if t.state == "looking" and now - t.arrived >= VIEW_SETTLE_S]

    def task_text(self, drone_id: str) -> str | None:
        task = self.held_by(drone_id)
        return None if task is None else task.text

    def counts(self) -> dict:
        return {"done": sum(t.state == "done" for t in self.tasks), "skipped": sum(t.state == "skipped" for t in self.tasks),
                "total": len(self.tasks)}

    def step(self, now: float) -> None:
        if self.state != "searching":
            return
        manager = self.manager
        hub = manager.hub
        memory = hub.memory
        flying = manager.crew(now)
        for task in self.tasks:
            if task.thing_id is not None and task.thing_id not in memory.things and task.state in ("waiting", "held", "looking"):
                # On a closer look it wasn't there, and the memory has forgotten it.
                if task.holder is not None:
                    hub.orders.pop(task.holder, None)
                task.state, task.holder = "skipped", None
        if not flying and not manager.can_still_fly(now):
            self.state = "failed"
            return
        for d in flying:
            task = self.held_by(d.id)
            if task is None:
                continue
            order = hub.orders.get(d.id)
            if order is None or order.stage == "failed":
                hub.orders.pop(d.id, None)
                task.tries += 1
                task.state = "skipped" if task.tries >= FAILED_VIEW_TRIES else "waiting"
                task.holder = None
            elif order.stage == "done":
                if task.state != "looking":
                    task.state, task.arrived = "looking", now
                elif now - task.arrived >= VIEW_DWELL_S:
                    hub.orders.pop(d.id, None)
                    task.state, task.holder = "done", None
        for d in flying:
            if self.held_by(d.id) is None:
                self._give_work(d, now, flying)
        if all(t.state in ("done", "skipped") for t in self.tasks):
            self.state = "done"

    def _give_work(self, d, now: float, flying: list) -> None:
        hub = self.manager.hub
        idle = {other.id for other in flying if self.held_by(other.id) is None}  # these are about to be sent elsewhere too
        avoid = [other.tracked.p for other in hub.co.drones.values()
                 if other is not d and other.id not in idle and other.phase in (Phase.AIR, Phase.CHECK, Phase.TAKEOFF, Phase.LANDING)]
        avoid += [order.goal for other_id, order in hub.orders.items() if other_id != d.id]
        best, best_distance = None, math.inf
        for task in self.tasks:
            if task.state != "waiting":
                continue
            if any(np.linalg.norm(task.view.p[:2] - np.asarray(q)[:2]) < KEEP_AWAY for q in avoid):
                continue
            distance = float(np.linalg.norm(task.view.p[:2] - d.tracked.p[:2]))
            if distance < best_distance:
                best, best_distance = task, distance
        if best is None:
            return
        if best.last_holder is not None and best.last_holder != d.id:
            self.manager.say("mission", f"{d.id} took over {best.last_holder}'s work: {best.text}.", event="reassigned",
                             drone=d.id, from_drone=best.last_holder)
        best.state, best.holder = "held", d.id
        hub.order(d, best.view.p, best.view.yaw, now)


def _names(items) -> str:
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _marker_filter(tracker):
    """A viewpoint with no floor marker in the picture costs a drone its position, so it's no use."""
    from swarm.tracking import Lens, anchors_in_view

    optics = next(iter(tracker.drones.values()))
    lens = optics.lens or Lens.from_fov()
    tilt = math.radians(optics.tilt_deg)
    return lambda p, yaw: bool(anchors_in_view(tracker.anchors, lens, tilt, p, yaw))


class MissionManager:
    def __init__(self, hub, frames=None, findings=None, capture=None, flight_limit_s: float = 270.0, on_event=None,
                 north_yaw_deg: float = 0.0, inspector=None):
        self.hub = hub
        self.frames = frames
        self.findings = findings
        self.capture = capture
        self.flight_limit_s = flight_limit_s
        self.on_event = on_event
        self.inspector = inspector if inspector is not None else self._default_inspector()
        self.regions = Regions(hub.co.room, north_yaw_deg)
        self.later = _Later()
        self.events: deque = deque(maxlen=500)
        tracker = getattr(hub.co, "tracker", None)
        self.usable = _marker_filter(tracker) if tracker is not None else None
        if tracker is not None:
            hub.limit_heading = lambda d, yaw: tracker.allowed_yaw(d.id, d.tracked.p, yaw)  # never turn away from every tag
        self.coverage = self._new_coverage()
        self.coverage.hub_looks = frames is None  # a floor patch is seen when a picture was taken, detector or not
        self.coverage.remaining()  # the first map of viewpoints takes a moment: do it now, not in flight

        self.kind: str | None = None
        self.state = "idle"  # idle, starting, running, analyzing, done, stopped, failed
        self.region = "all"
        self.target: str | None = None
        self.reason = ""
        self.work = None
        self.started_at = 0.0
        self.ended_at: float | None = None
        self.capture_name: str | None = None
        self.progress_said = 0
        self.analyzing_since = 0.0

        self.took_off: dict[str, float] = {}
        self.warned: set[str] = set()
        self.limited: set[str] = set()
        self.total_air: dict[str, float] = {}  # all flights so far: the battery may not have been changed
        self.total_warned: set[str] = set()
        self.out: dict[str, str] = {}  # drones the current mission has lost, and why
        self.flew: set[str] = set()  # drones that have been part of the current mission
        self.bad_since: dict[str, float] = {}
        self.next_look = 0.0
        self.last_frame: dict[str, int] = {}
        self.errors = 0

    # ---------------------------------------------------------------- who can work
    def crew(self, now: float) -> list:
        """Drones that can take work right now."""
        co = self.hub.co
        return [d for d in co.drones.values() if d.phase == Phase.AIR and not d.landing_requested and co._steerable(d, now)]

    def taking_off(self) -> bool:
        co = self.hub.co
        return bool(co.takeoff_queue) or any(d.phase in (Phase.TAKEOFF, Phase.CHECK) for d in co.drones.values())

    def can_still_fly(self, now: float) -> bool:
        """Is anyone flying, or about to, that this mission hasn't written off?"""
        if self.taking_off():
            return True
        return any(d.phase == Phase.AIR and not d.landing_requested and d.id not in self.out for d in self.hub.co.drones.values())

    def _flight_seconds(self, drone_id: str, now: float) -> float:
        return max(0.0, now - self.took_off[drone_id]) if drone_id in self.took_off else 0.0

    # ---------------------------------------------------------------- telling people
    def say(self, kind: str, text: str, /, **data) -> None:
        """Note something worth telling the operator or the agent. Never waits and never raises."""
        event = {"t": round(self.hub.now, 2), "kind": kind, "text": text, "data": data}
        self.events.append(event)
        if self.on_event is not None:
            self.later(self.on_event, kind, text, data)

    def settle(self, timeout: float = 5.0) -> bool:
        """Wait until the background thread has caught up. For tests and benches."""
        return self.later.wait(timeout)

    # ---------------------------------------------------------------- asking for things
    def start(self, kind: str, region: str = "all", target: str | None = None) -> dict:
        """Begin a mission, replacing any running one. Raises MissionError with a sentence if it can't be done."""
        with self.hub.lock:
            now = self.hub.now
            kind = str(kind).strip().lower()
            if kind not in KINDS:
                raise MissionError(f'I can {", ".join(KINDS[:-1])} or {KINDS[-1]}, not "{kind}".')
            region = self.regions.resolve(region)
            self._check_can_start(kind, now)
            if kind == "map" and self.capture is None:
                raise MissionError("Map capture isn't set up here, so I can't build a map. I can explore instead.")
            work_for = {"explore": self._plan_sweep, "map": self._plan_sweep, "inspect": self._plan_survey,
                        "search": self._plan_search}[kind]
            plan, region, target = work_for(kind, region, target, now)
            if self.active:
                self._finish("stopped", f"The {self.kind} mission was replaced by a new {kind} mission.", now)
            self._begin(kind, region, target, plan, now)
            return self.status()

    def focus(self, region: str) -> dict:
        """Same mission, new region. Every drone gets new work at once."""
        with self.hub.lock:
            now = self.hub.now
            if not self.active:
                raise MissionError("There's no mission running to change. Start one first.")
            region = self.regions.resolve(region)
            work_for = {"explore": self._plan_sweep, "map": self._plan_sweep, "inspect": self._plan_survey,
                        "search": self._plan_search}[self.kind]
            plan, region, target = work_for(self.kind, region, self.target, now)
            self.region, self.target = region, target
            self._install(plan, now)
            self.progress_said = int(self._share_seen() * 4)
            if self.state == "analyzing":
                self.state = "running"
                self._inspector_begin(target)
            self.say("mission", f"Focus moved to {spoken(region)}. {_names(d.id for d in self.crew(now))} moving there.",
                     event="focus", region=region)
            return self.status()

    def stop(self) -> dict:
        """End the mission. Drones hold where they are, except any already on their way down."""
        with self.hub.lock:
            now = self.hub.now
            if self.active:
                self.hub._stop_search()
                self.hub.orders.clear()
                self._halt()
                self.hub.co.takeoff_queue.clear()
                self._finish("stopped", f"{self.kind.capitalize()} mission stopped. Holding position.", now)
            return self.status()

    def status(self) -> dict:
        with self.hub.lock:
            now = self.hub.now
            end = self.ended_at if self.ended_at is not None else now
            out = {
                "kind": self.kind,
                "state": self.state,
                "region": self.region,
                "target": self.target,
                "share_seen": round(self._share_seen(), 3),
                "elapsed_s": round(end - self.started_at, 1) if self.kind else 0.0,
                "tasks": {d.id: self._task_text(d, now) for d in self.hub.co.drones.values()},
                "flight_s": {drone_id: round(self._flight_seconds(drone_id, now), 1) for drone_id in self.hub.co.drones},
                "flight_limit_s": self.flight_limit_s,
                "reason": self.reason,
            }
            if isinstance(self.work, Survey):
                out["views"] = self.work.counts()
            return out

    def gaps(self) -> list[dict]:
        """For each region: how much is seen, and the best place to look from next."""
        with self.hub.lock:
            rows = []
            for name in REGION_NAMES:
                row = {"name": name, **self.coverage.stats(name), "viewpoint": None}
                view = self.coverage.suggest(name)
                if view is not None:
                    row["viewpoint"] = {"x": round(float(view.p[0]), 2), "y": round(float(view.p[1]), 2),
                                        "z": round(float(view.p[2]), 2), "yaw_deg": round(math.degrees(view.yaw), 1)}
                rows.append(row)
            return rows

    @property
    def active(self) -> bool:
        return self.state in ("starting", "running", "analyzing")

    # ---------------------------------------------------------------- planning, before anything changes
    def _check_can_start(self, kind: str, now: float) -> None:
        flying = [d for d in self.hub.co.drones.values() if d.phase == Phase.AIR and not d.landing_requested]
        if not flying and not self.taking_off():
            raise MissionError(f"Nothing is flying. Take off first, then ask me to {kind}.")
        if flying and not self.taking_off():
            left = [self.flight_limit_s - self._flight_seconds(d.id, now) for d in flying]
            if max(left) < MIN_FLIGHT_LEFT_S:
                raise MissionError(f"Every drone is within {MIN_FLIGHT_LEFT_S:.0f} s of its flight limit. Land and change batteries first.")

    def _new_coverage(self) -> RegionCoverage:
        tilts = {self.hub.tilt_of(drone_id) for drone_id in self.hub.co.drones}
        return RegionCoverage(self.hub.co.room, tilts, self.regions, self.usable)

    def _plan_sweep(self, kind: str, region: str, target, now: float):
        self.coverage._refresh()
        if not (self.regions.mask(region, self.coverage.cells) & self.coverage.anywhere).any():
            raise MissionError(f"No place a drone can hold shows any of the floor of {spoken(region)}.")
        return None, region, None

    def _plan_survey(self, kind: str, region: str, target, now: float):
        coverage = self.coverage
        coverage._refresh()
        label = None if target is None else clean_label(str(target))
        if label is not None and _VAGUE.fullmatch(label):
            label = None
        if label is not None:
            try:
                named = self.regions.resolve(label)
                region, label = (named if region == "all" else region), None
            except MissionError:
                pass  # not a region, so it's the name of a thing
        tilt = next(iter(coverage.all_sees), None)
        if tilt is None:
            raise MissionError("There's no place a drone can hold in this room, so I can't plan an inspection.")
        memory = self.hub.memory
        tasks: list[ViewTask] = []
        if label is not None:
            things = [t for t in memory.find(label) if not t.moves]
            if not things:
                raise MissionError(f"I haven't seen a {label}. Explore or search first, or give me a region to inspect.")
            inside = [t for t in things if region == "all" or self.regions.mask(region, t.p[None, :])[0]]
            if not inside:
                raise MissionError(f"I haven't seen a {label} in {spoken(region)}.")
            for thing in inside[:MAX_THINGS]:
                tasks += self._thing_tasks(thing, tilt, VIEWS_PER_THING)
            if not tasks:
                raise MissionError(f"The camera looks {math.degrees(tilt):.0f} degrees down and can't look up, and I can't "
                                   f"find a clear place below the ceiling limit to get the {label} in the picture.")
            return tasks, region, label
        things = [t for t in memory.known() if not t.moves and self.regions.mask(region, t.p[None, :])[0]]
        for thing in things[:MAX_THINGS]:
            tasks += self._thing_tasks(thing, tilt, VIEWS_PER_THING_IN_REGION)
        for view in coverage.cover(region, MAX_FLOOR_VIEWS):
            tasks.append(ViewTask(view, f"inspecting the floor from ({view.p[0]:.1f}, {view.p[1]:.1f})"))
        if not tasks:
            raise MissionError(f"I can't find anywhere to inspect {spoken(region)} from.")
        return tasks, region, None

    def _thing_tasks(self, thing, tilt: float, count: int) -> list[ViewTask]:
        return [ViewTask(view, f"inspecting the {thing.label} from ({view.p[0]:.1f}, {view.p[1]:.1f})", thing.label, thing.id)
                for view in views_of(thing, self.coverage, tilt, count)]

    def _plan_search(self, kind: str, region: str, target, now: float):
        if not target:
            raise MissionError("Tell me what to search for.")
        label = clean_label(str(target))
        if not self.hub.eyes.knows(label):
            raise MissionError(f'The camera can\'t recognise "{label}", so I didn\'t start a search.')
        return None, region, label

    # ---------------------------------------------------------------- starting and ending
    def _begin(self, kind: str, region: str, target, plan, now: float) -> None:
        self.kind, self.region, self.target = kind, region, target
        self.state = "running" if self.crew(now) else "starting"
        self.reason = ""
        self.started_at, self.ended_at = now, None
        self.out.clear()
        self.bad_since.clear()
        self.flew = {d.id for d in self.crew(now)}
        if kind == "map":
            self.coverage.set_region(region)
            self.coverage.forget_region()  # the floor has to be seen again while the photos are being saved
            self.capture_name = time.strftime("map-%Y%m%d-%H%M%S")
            self.later(self._start_capture, self.capture_name)
        self._install(plan, now)
        self.progress_said = int(self._share_seen() * 4)
        if kind == "inspect":
            self._inspector_begin(target)
        self.say("mission", f"{kind.capitalize()} mission started on {spoken(region)}"
                            f"{f' for the {target}' if target else ''}.", event="started", mission=kind, region=region, target=target)

    def _install(self, plan, now: float) -> None:
        """Put this work in the hub's hands and take every drone off whatever it was doing."""
        hub = self.hub
        kind = self.kind
        if kind == "search":
            before = hub.search
            reply = hub.find(self.target)
            if hub.search is before:
                self.state, self.reason = "failed", reply
                self.ended_at = now
                raise MissionError(reply)
            hub.orders.clear()
            self._halt()
            if self.region != "all":
                coverage = self._new_coverage()
                coverage.set_region(self.region)
                hub.search.coverage = coverage
            self.work = hub.search
            return
        hub._stop_search()
        hub.orders.clear()
        self._halt()
        self.coverage.set_region(self.region)
        if kind == "inspect":
            work = Survey(self, self.coverage, plan, f"things to inspect in {spoken(self.region)}")
        else:
            if self.coverage.share_seen() >= 1.0:
                self.coverage.forget_region()  # asked to go there anyway, so take a second look
            work = Sweep(self, self.coverage, f"unseen floor in {spoken(self.region)}")
        hub.search = self.work = work

    def _halt(self) -> None:
        co = self.hub.co
        for d in co.drones.values():
            if d.phase == Phase.AIR and not d.landing_requested:
                co._set_goal(d, d.tracked.p.copy())

    def _finish(self, state: str, reason: str, now: float) -> None:
        was_kind = self.kind
        self.state, self.reason, self.ended_at = state, reason, now
        text = reason
        if was_kind == "map" and self.capture_name is not None:
            name, self.capture_name = self.capture_name, None
            self.later(self._stop_capture_and_build, name)
            text += " Building the 3D map from the photos."
        if was_kind == "inspect":
            self._inspector_end()
        event = {"done": "done", "stopped": "stopped"}.get(state, "failed")
        self.say("mission", text, event=event, mission=was_kind, region=self.region, target=self.target,
                 share_seen=round(self._share_seen(), 3), elapsed_s=round(now - self.started_at, 1))

    def _start_capture(self, name: str) -> None:
        try:
            self.capture.start(name)
        except Exception as error:
            logger.exception("[mission] map capture didn't start")
            self.say("fault", f"Map capture didn't start: {error}", event="capture_failed")

    def _stop_capture_and_build(self, name: str) -> None:
        try:
            self.capture.stop()
            self.capture.build(name)
        except Exception as error:
            logger.exception("[mission] map build didn't start")
            self.say("fault", f"The map build didn't start: {error}", event="capture_failed")

    # ---------------------------------------------------------------- the inspector
    def _default_inspector(self):
        if self.frames is None or self.findings is None:
            return None
        try:
            from swarm.inspect import Inspector
            return Inspector(self.findings, self.frames, hub=self.hub, on_event=lambda kind, text, data: self.say(kind, text, **data))
        except Exception:
            logger.exception("[mission] no inspector: inspections will fly but nothing will be analysed")
            return None

    def _inspector_begin(self, target) -> None:
        if self.inspector is None:
            self.say("fault", "No camera or findings store is connected, so this inspection will fly but report nothing.",
                     event="no_inspector")
            return
        try:
            self.inspector.start(target)
        except Exception:
            logger.exception("[mission] the inspector didn't start")
            self.say("fault", "The inspector didn't start, so this inspection will fly but report nothing.", event="no_inspector")

    def _inspector_end(self) -> None:
        if self.inspector is not None:
            try:
                self.inspector.stop()
            except Exception:
                logger.exception("[mission] the inspector didn't stop")

    def _inspector_status(self) -> dict:
        if self.inspector is None:
            return {}
        try:
            return dict(self.inspector.status())
        except Exception:
            logger.exception("[mission] couldn't ask the inspector how it's doing")
            return {}

    def _inspector_busy(self) -> bool:
        status = self._inspector_status()
        return bool(status.get("queued") or status.get("busy"))

    def _feed_inspector(self, now: float) -> None:
        """Only frames from a drone settled at a viewpoint are worth a slow vision call, not ones taken on the way."""
        if self.inspector is None or self.state != "running" or not isinstance(self.work, Survey):
            return
        drones = self.hub.co.drones
        self.inspector.tick(now, {drone_id: drones[drone_id].tracked for drone_id in self.work.settled(now)})

    # ---------------------------------------------------------------- the loop
    def tick(self, now: float) -> None:
        """Called after hub.tick. Never raises: a failure here must not cost a control packet."""
        try:
            with self.hub.lock:
                self._watch_flight_time(now)
        except Exception:
            self.errors += 1
            logger.exception("[mission] flight timer failed")
        try:
            with self.hub.lock:
                self._count_looks(now)
                if self.active:
                    self._watch_crew(now)
                    self._watch_work(now)
                    self._feed_inspector(now)
        except Exception:
            self.errors += 1
            logger.exception("[mission] watching the mission failed; the drones carry on with their last goals")

    def _return_time(self, d) -> float:
        distance = float(np.linalg.norm(d.tracked.p[:2] - np.array(d.config.pad[:2])))
        return distance / RETURN_SPEED + LANDING_TIME_S

    def _watch_flight_time(self, now: float) -> None:
        for d in self.hub.co.drones.values():
            if d.phase in (Phase.TAKEOFF, Phase.CHECK, Phase.AIR, Phase.LANDING):
                if d.id not in self.took_off:
                    self.took_off[d.id] = now
                    self.warned.discard(d.id)
                    self.limited.discard(d.id)
            elif d.id in self.took_off:
                self.total_air[d.id] = self.total_air.get(d.id, 0.0) + now - self.took_off.pop(d.id)
                continue
            else:
                continue
            flown = now - self.took_off[d.id]
            total = self.total_air.get(d.id, 0.0) + flown
            if total >= self.flight_limit_s and flown < self.flight_limit_s and d.id not in self.total_warned:
                self.total_warned.add(d.id)
                self.say("mission", f"{d.id} has flown {total:.0f} s in total. Unless its battery was changed, land it soon.",
                         event="flight_warning", drone=d.id, flight_s=round(total, 1))
            if d.phase not in (Phase.AIR, Phase.CHECK):
                continue
            leave_in = self.flight_limit_s - flown - self._return_time(d)
            if leave_in <= 0.0:
                # A failed landing is for a person to look at. Another toggle could take off again.
                if not d.landing_requested and "land" not in d.faults:
                    self._land_for_limit(d, now, flown)
            elif leave_in <= WARN_BEFORE_LIMIT_S and d.id not in self.warned:
                self.warned.add(d.id)
                self.say("mission", f"{d.id} has about {leave_in:.0f} s before it has to head back to land.",
                         event="flight_warning", drone=d.id, flight_s=round(flown, 1))

    def _land_for_limit(self, d, now: float, flown: float) -> None:
        """Bring back just this one drone, the same way the coordinator brings back all of them."""
        co = self.hub.co
        if d.phase == Phase.CHECK:
            d.check = None
            d.phase = Phase.AIR
        self.hub.orders.pop(d.id, None)
        d.landing_requested = True
        d.land_requested_at = now
        co._set_goal(d, np.array([d.config.pad[0], d.config.pad[1], max(d.tracked.p[2], 0.7)]))
        if d.id in self.limited:
            return  # someone said hold while it was on its way back, and it still has to land
        self.limited.add(d.id)
        text = f"{d.id} has flown {flown:.0f} s and is coming back to land before its {self.flight_limit_s:.0f} s limit."
        if self.active and d.id in self.flew:
            self._lose(d, "reached its flight limit", now, say=False)
            others = [other.id for other in self.crew(now)]
            text += f" {_names(others)} take over its work." if others else " Nobody is left to take over its work."
        self.say("mission", text, event="flight_limit", drone=d.id, flight_s=round(flown, 1))

    def _lose(self, d, why: str, now: float, say: bool = True) -> None:
        self.out[d.id] = why
        self.bad_since.pop(d.id, None)
        self.hub.orders.pop(d.id, None)
        release = getattr(self.work, "release", None)
        if release is not None:
            release(d.id)
        if say:
            others = [other.id for other in self.crew(now)]
            carry_on = f"Its work goes back to the pool and {_names(others)} carry on." if others else "Nobody is left to take its work."
            self.say("fault", f"{d.id} {why}. {carry_on}", event="drone_lost", drone=d.id, reason=why, others=others,
                     share_seen=round(self._share_seen(), 3))

    def _why_out(self, d, now: float) -> str:
        if d.phase == Phase.GROUND:
            return "has landed"
        if d.phase == Phase.STOPPED:
            return "has had its motors cut"
        if d.phase == Phase.LANDING or d.landing_requested:
            return f"is landing ({'; '.join(d.faults.values())})" if d.faults else "is landing"
        if not d.link.connected:
            return "lost its control link"
        return "lost tracking"

    def _watch_crew(self, now: float) -> None:
        crew = {d.id for d in self.crew(now)}
        for d in self.hub.co.drones.values():
            if d.id in crew:
                self.bad_since.pop(d.id, None)
                self.flew.add(d.id)
                if d.id in self.out:
                    del self.out[d.id]
                    self.hub.co._set_goal(d, d.tracked.p.copy())
                    self.say("mission", f"{d.id} is back and taking work again.", event="drone_back", drone=d.id)
                continue
            if d.id not in self.flew or d.id in self.out:
                continue
            # Tracking can come back by itself, so give it a moment. A landing can't.
            if d.phase == Phase.AIR and not d.landing_requested:
                since = self.bad_since.setdefault(d.id, now)
                if now - since < DROP_GRACE_S:
                    continue
            self._lose(d, self._why_out(d, now), now)

    def _count_looks(self, now: float) -> None:
        """Keep the shared picture of the floor up to date.

        With real frames, each new picture from a drone counts, so a mission doesn't depend on the
        detector working. Without them only the hub knows, and it only counts for its current search.
        """
        hub = self.hub
        if now < self.next_look:
            return
        if self.frames is None and hub.search is not None and hub.search.coverage is self.coverage:
            return
        self.next_look = now + LOOK_EVERY
        for d in hub.co.drones.values():
            if d.phase not in (Phase.AIR, Phase.CHECK) or d.tracked.age(now) > 0.3:
                continue
            p, yaw = d.tracked.p, d.tracked.yaw
            if self.frames is not None:
                frame = self.frames.latest(d.id)
                if frame is None or frame.number == self.last_frame.get(d.id):
                    continue
                self.last_frame[d.id] = frame.number
                p, yaw = hub._pose_at(d.id, now - VIDEO_LATE_S)
            self.coverage.count(p, yaw, hub.tilt_of(d.id))

    def _watch_work(self, now: float) -> None:
        hub, work = self.hub, self.work
        if self.state == "starting" and self.crew(now):
            self.state = "running"
        if self.state == "analyzing":
            if not self._inspector_busy() or now - self.analyzing_since > ANALYSIS_LIMIT_S:
                self._finish_done(now)
            return
        if hub.search is not work:
            self._finish("stopped", f"The {self.kind} mission ended because someone started something else.", now)
            return
        if work.state == "stopped":
            self._finish("stopped", f"The {self.kind} mission ended on the operator's command.", now)
        elif work.state == "failed":
            self._finish("failed", f"No drones are left flying, so the {self.kind} mission ended with "
                                   f"{self._share_seen() * 100:.0f}% of the floor seen.", now)
        elif self.kind == "search":
            self._watch_search(now)
        elif work.state == "done":
            if self.kind == "inspect":
                self._inspector_end()  # what's already queued still gets checked
            if self.kind == "inspect" and self._inspector_busy():
                self.state, self.analyzing_since = "analyzing", now
                self.say("mission", "All the views are done. Waiting for the photo analysis to finish.", event="analyzing")
            else:
                self._finish_done(now)
        else:
            self._say_progress()

    def _watch_search(self, now: float) -> None:
        search = self.work
        if search.state == "found":
            where = "" if search.found_at is None else f" at ({search.found_at[0]:.1f}, {search.found_at[1]:.1f})"
            self._finish("done", f"Found the {self.target}{where}.", now)
        elif search.state == "exhausted":
            self._finish("done", f"Searched {search.coverage.share_seen() * 100:.0f}% of the floor of {spoken(self.region)} "
                                 f"and didn't find a {self.target}.", now)

    def _finish_done(self, now: float) -> None:
        elapsed = now - self.started_at
        if self.kind == "inspect":
            counts, seen = self.work.counts(), self._inspector_status()
            text = f"Inspection of {self.target or spoken(self.region)} finished: {counts['done']} of {counts['total']} views in {elapsed:.0f} s."
            if seen:
                text += f" {seen.get('checked', 0)} photos checked, {seen.get('findings', 0)} possible issues to review."
            if seen.get("failed"):
                text += f" {seen['failed']} photos couldn't be checked."
        else:
            text = f"{self.kind.capitalize()} of {spoken(self.region)} finished: {self._share_seen() * 100:.0f}% of the floor seen in {elapsed:.0f} s."
        self._finish("done", text, now)

    def _say_progress(self) -> None:
        mark = int(self._share_seen() * 4)
        if self.progress_said < mark < 4:
            self.progress_said = mark
            self.say("mission", f"{mark * 25}% of {spoken(self.region)} seen so far.", event="progress", share_seen=mark * 0.25)

    # ---------------------------------------------------------------- what's going on
    def _share_seen(self) -> float:
        work = self.work
        if work is not None and self.kind == "search":
            return float(work.coverage.share_seen())
        return float(self.coverage.share_seen())

    def _task_text(self, d, now: float) -> str:
        words = {Phase.GROUND: "on the ground", Phase.TAKEOFF: "taking off", Phase.CHECK: "checking its response",
                 Phase.LANDING: "landing", Phase.STOPPED: "motors cut"}
        if d.phase != Phase.AIR:
            return words[d.phase]
        if d.landing_requested:
            return "flying back to its pad to land"
        if d.id in self.out:
            return self.out[d.id]
        if self.active and self.work is not None:
            ask = getattr(self.work, "task_text", None)
            text = ask(d.id) if ask is not None else None
            if text is None and self.kind == "search" and d.id in self.hub.orders:
                text = f"searching for the {self.target}"
            if text is not None:
                return text
        return "holding"
