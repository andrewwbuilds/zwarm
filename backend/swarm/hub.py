"""The fleet's shared brain: what any drone sees, every drone can act on.

Drones can't talk to each other. Each one only sends video and receives sticks, so "telling the other
drones" means this hub writes a sighting to the shared memory and the coordinator flies everyone
around it. The coordinator stays in charge of safety: everything here only chooses goals and headings.

A failure in here must never stop the flight loop, so tick() catches and logs anything raised while
thinking. The drones then keep flying their last goals under the safety filter.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from swarm.commands import EXPLANATION, parse
from swarm.coordinator import Coordinator, Phase
from swarm.eyes import DEFAULT_TILT, EDGE_MARGIN, RELIABLE_REACH, blocked, in_view, locate
from swarm.memory import Memory, Thing
from swarm.room import Obstacle
from swarm.safety import Moving
from swarm.search import SEARCH_Z, Coverage

logger = logging.getLogger(__name__)

LOOK_EVERY = 0.2  # seconds between looks through each camera
POSE_HISTORY = 60  # ticks of pose kept, about 3 s, to match a late picture with where the drone was
FENCE_SLACK = 0.5  # how far outside the fence a position may land before it's thrown away
OBSTACLE_MARGIN = 0.1  # added around a seen thing, since its size and place are estimates
OBSTACLE_MOVE = 0.15  # the map only changes when the estimate has shifted this far, since every change replans
TURN_FIRST = math.radians(25)  # turn to look where it's going before moving off
TURN_TIMEOUT_S = 8.0
FACED = math.radians(14)
GO_TIMEOUT_S = 75.0
HINT_MAX_AGE_S = 3600.0
CLOSE_LOOK_S = 2.0
CLOSE_TIMEOUT_S = 45.0
CLOSE_SIGHTINGS = 3
NOBODY_WAIT_S = 15.0
MOVER_RADIUS = 0.35
DWELL_S = 1.0


def _action(method):
    """One thing someone can ask for: takes the lock, keeps a record, and always answers."""

    @functools.wraps(method)
    def run(self, *args, **kwargs):
        with self.lock:
            shown = ", ".join([*(repr(a) for a in args), *(f"{k}={v!r}" for k, v in kwargs.items())])
            self.memory.note(self.now, "action", f"{method.__name__}({shown})")
            try:
                reply = method(self, *args, **kwargs)
            except Exception:
                logger.exception("[hub] %s failed", method.__name__)
                reply = "That failed inside the hub, and it may not have finished. Check the log."
            self.memory.note(self.now, "reply", reply)
            self.memory.commit()
            return reply

    return run


def _angle(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Order:
    """Turn to look, go there, then face the way asked."""

    goal: np.ndarray
    face: float | None
    stage: str  # turn, go, face, done, failed
    since: float
    why: str = ""


class Search:
    """Share out the floor until the thing is seen, then get a close look before saying it's found."""

    def __init__(self, hub: "Hub", label: str, now: float, hint: Thing | None = None):
        self.hub, self.label, self.started = hub, label, now
        self.coverage = hub.new_coverage()
        self.state = "searching"  # searching, closing, found, exhausted, failed, stopped
        self.views: dict[str, object] = {}
        self.dwell: dict[str, tuple[float, int]] = {}  # arrived when, and how many looks its view had by then
        self.banned: list[np.ndarray] = []
        self.hint = hint.id if hint else None
        self.thing_id: int | None = None
        self.closer: str | None = None
        self.close_started = 0.0
        self.close_seen_from = 0
        self.found_at: np.ndarray | None = None
        self.obstacle_count = len(hub.co.room.obstacles)
        self.nobody_since: float | None = None

    def flying(self) -> list:
        """Drones that can take work now. One with no link or tracking may be up there, but it won't do as it's told."""
        co = self.hub.co
        return [d for d in co.drones.values() if d.phase == Phase.AIR and not d.landing_requested and co._steerable(d, self.hub.now)]

    def step(self, now: float) -> None:
        if self.state in ("found", "exhausted", "failed", "stopped"):
            return
        hub = self.hub
        flying = self.flying()
        if not flying and not any(d.phase in (Phase.TAKEOFF, Phase.CHECK) for d in hub.co.drones.values()) and not hub.co.takeoff_queue:
            self.state = "failed"
            hub.report(f"No drones are left flying, so I stopped looking for the {self.label}.")
            return
        if self.state == "closing":
            self._close_in(now, flying)
        if self.state == "searching":
            spotted = next((t for t in hub.memory.find(self.label) if t.sure and t.last_seen >= self.started), None)
            if spotted is not None:
                self._start_close_look(spotted, now, flying)
            else:
                self._share_out(now, flying)

    # ---------------------------------------------------------------- finding it
    def _start_close_look(self, thing: Thing, now: float, flying: list) -> None:
        hub = self.hub
        if not flying:
            return
        self.thing_id = thing.id
        self.state = "closing"
        self.close_started = now
        self.close_seen_from = thing.sightings
        for d in flying:
            hub.orders.pop(d.id, None)
        self.views.clear()
        closest = min(flying, key=lambda d: float(np.linalg.norm(d.tracked.p - thing.p)))
        self.closer = closest.id
        for d in flying:
            if d is not closest:
                hub.co._set_goal(d, d.tracked.p.copy())
        hub.report(f"{closest.id} may have spotted a {self.label} near {_where(thing.p)}. Going closer to check.")
        others = [d.tracked.p for d in flying if d is not closest]
        view = self.coverage.standoff(closest.tracked.p, thing.p, avoid=others, tilt=hub.tilt_of(closest.id))
        if view is None:
            hub.aim(closest, math.atan2(thing.p[1] - closest.tracked.p[1], thing.p[0] - closest.tracked.p[0]))
        else:
            hub.order(closest, view.p, view.yaw, now)

    def _close_in(self, now: float, flying: list) -> None:
        hub = self.hub
        thing = hub.memory.things.get(self.thing_id)
        if thing is None:
            hub.report(f"That {self.label} wasn't there on a closer look. Still searching.")
            self._resume(now)
            return
        closer = hub.co.drones.get(self.closer)
        if closer is None or closer not in flying:
            self._resume(now)
            return
        order = hub.orders.get(closer.id)
        settled = order is None or order.stage in ("done", "failed")
        if settled and closer.id not in self.dwell:
            self.dwell[closer.id] = (now, 0)
            self.close_seen_from = thing.sightings
        settled_for = now - self.dwell[closer.id][0] if closer.id in self.dwell else -1.0
        looked = settled_for >= CLOSE_LOOK_S
        if looked and thing.sightings - self.close_seen_from >= CLOSE_SIGHTINGS and thing.sure:
            self.state = "found"
            self.found_at = thing.p.copy()
            hub.report(f"Found the {self.label} at {_where(thing.p)}, seen by {_names(thing.seen_by)}.")
            return
        if now - self.close_started > CLOSE_TIMEOUT_S or settled_for > 8.0:
            hub.report(f"I couldn't confirm the {self.label} near {_where(thing.p)}. Still searching.")
            self.banned.append(thing.p.copy())
            hub.memory.forget(thing.id)
            self._resume(now)

    def _resume(self, now: float) -> None:
        self.state = "searching"
        self.thing_id = None
        self.closer = None
        self.dwell.clear()
        for d in self.flying():
            self.hub.orders.pop(d.id, None)

    # ---------------------------------------------------------------- sharing the floor
    def _share_out(self, now: float, flying: list) -> None:
        hub = self.hub
        claimed = np.zeros(len(self.coverage.cells), bool)
        free = []
        for d in flying:
            order = hub.orders.get(d.id)
            view = self.views.get(d.id)
            if order is not None and order.stage == "failed":
                self.banned.append(order.goal.copy())
                hub.orders.pop(d.id, None)
                self.views.pop(d.id, None)
                self.dwell.pop(d.id, None)
                free.append(d)
            elif order is None or order.stage == "done":
                if view is not None and order is not None:
                    looks = int(self.coverage.counts[view.cells].sum())
                    arrived_at, looks_then = self.dwell.setdefault(d.id, (now, looks))
                    if now - arrived_at < DWELL_S:
                        claimed[view.cells] = True
                        continue
                    if len(view.cells) and looks == looks_then:
                        # Its floor wasn't seen from there, such as when the tracker overruled the
                        # heading. Offering the same viewpoint again would never end the search.
                        self.banned.append(order.goal.copy())
                hub.orders.pop(d.id, None)
                self.views.pop(d.id, None)
                self.dwell.pop(d.id, None)
                free.append(d)
            elif view is not None:
                claimed[view.cells] = True
        for d in free:
            hint = hub.memory.things.get(self.hint) if self.hint is not None else None
            avoid = [other.tracked.p for other in flying if other is not d]
            avoid += [o.goal for other_id, o in hub.orders.items() if other_id != d.id]
            view = None
            if hint is not None:
                view = self.coverage.standoff(d.tracked.p, hint.p, avoid=avoid, tilt=hub.tilt_of(d.id))
                self.hint = None
            if view is None:
                view = self.coverage.best(d.tracked.p, d.tracked.yaw, claimed, avoid, banned=self.banned, tilt=hub.tilt_of(d.id))
            if view is None:
                continue
            if len(view.cells):
                claimed[view.cells] = True
            self.views[d.id] = view
            hub.order(d, view.p, view.yaw, now)
        busy = any(d.id in hub.orders for d in flying)
        waiting_to_fly = any(d.phase in (Phase.TAKEOFF, Phase.CHECK) for d in hub.co.drones.values()) or bool(hub.co.takeoff_queue)
        if flying and not busy and not waiting_to_fly:
            self.state = "exhausted"
            hub.report(f"I searched {self.coverage.share_seen() * 100:.0f}% of the floor I can see and didn't find a {self.label}.")


class Hub:
    def __init__(self, coordinator: Coordinator, memory: Memory, eyes, report=None, tilts: dict | None = None):
        self.co = coordinator
        self.tilts = tilts or {}  # drone id -> camera tilt in radians, where it differs from DEFAULT_TILT
        self.memory = memory
        self.eyes = eyes
        self.on_report = report
        self.reports: list[str] = []
        self.orders: dict[str, Order] = {}
        self.search: Search | None = None
        self.next_look = 0.0
        self.mapped: dict[int, Obstacle] = {}
        self.now = 0.0
        self.think_errors = 0
        self.lock = threading.RLock()  # the flight loop and whoever gives commands are different threads
        self.poses: dict[str, deque] = {drone_id: deque(maxlen=POSE_HISTORY) for drone_id in self.co.drones}
        # A tracker that needs certain things kept in view can overrule the way a drone is turned.
        self.limit_heading = lambda drone, yaw: yaw
        self._sync_obstacles()

    # ---------------------------------------------------------------- the loop
    def tick(self, now: float, poses: dict, moving=()) -> None:
        with self.lock:
            self.now = now
            seen_movers = [Moving(f"seen-{t.id}", t.p + t.v * min(now - t.last_seen, 1.0), t.v.copy(), MOVER_RADIUS)
                           for t in self.memory.movers(now)]
            self.co.tick(now, poses, [*moving, *seen_movers])
            try:
                for drone_id, d in self.co.drones.items():
                    self.poses[drone_id].append((now, d.tracked.p.copy(), d.tracked.yaw))
                if now >= self.next_look:
                    self.next_look = now + LOOK_EVERY
                    self._look(now)
                    self._sync_obstacles()
                    self.memory.commit()
                self._drive(now)
                if self.search is not None:
                    self.search.step(now)
            except Exception:
                self.think_errors += 1
                logger.exception("[hub] thinking failed; the drones carry on with their last goals")

    def tilt_of(self, drone_id: str) -> float:
        return self.tilts.get(drone_id, DEFAULT_TILT)

    def new_coverage(self) -> Coverage:
        return Coverage(self.co.room, {self.tilt_of(drone_id) for drone_id in self.co.drones})

    def report(self, text: str) -> None:
        self.reports.append(text)
        self.memory.note(self.now, "report", text)
        if self.on_report:
            self.on_report(text)

    # ---------------------------------------------------------------- seeing
    def _look(self, now: float) -> None:
        for d in self.co.drones.values():
            if d.phase not in (Phase.AIR, Phase.CHECK) or d.tracked.age(now) > 0.3:
                continue
            detections = self.eyes.look(d.id, now)
            if detections is None:
                continue
            # A real camera's picture is already old by now, and it's the drone's pose back then that counts.
            p, yaw = self._pose_at(d.id, now - max((x.age for x in detections), default=0.0))
            tilt = self.tilt_of(d.id)
            seen_now = set()
            for detection in detections:
                where = locate(p, yaw, detection, tilt)
                if np.any(where[:2] < self.co.room.lo[:2] - FENCE_SLACK) or np.any(where[:2] > self.co.room.hi[:2] + FENCE_SLACK):
                    continue  # a bad ray: nothing outside the room can be in it
                seen_now.add(self.memory.record(detection.label, where, detection.size, detection.confidence,
                                                float(np.linalg.norm(where - p)), d.id, now).id)
            self._note_misses(p, yaw, tilt, seen_now)
            if self.search is not None:
                self.search.coverage.look(p, yaw, tilt)

    def _pose_at(self, drone_id: str, t: float) -> tuple[np.ndarray, float]:
        history = self.poses[drone_id]
        return min(history, key=lambda entry: abs(entry[0] - t))[1:] if history else (self.co.drones[drone_id].tracked.p.copy(), self.co.drones[drone_id].tracked.yaw)

    def _note_misses(self, p, yaw: float, tilt: float, seen_now: set[int]) -> None:
        """Looked straight at something the memory holds and saw nothing there: count it against it."""
        for thing in list(self.memory.things.values()):
            if thing.id in seen_now or thing.moves:
                continue
            # Its base: a tilted camera can have the bottom of something tall in view but not its middle.
            base = np.array([thing.p[0], thing.p[1], max(0.0, thing.p[2] - thing.size[2] / 2)])
            if not in_view(p, yaw, base, reach=RELIABLE_REACH, margin=EDGE_MARGIN, tilt=tilt)[0]:
                continue
            own = self.mapped.get(thing.id)
            hidden = any(blocked(p, base[None, :], o.lo, o.hi)[0] for o in self.co.room.obstacles.values() if o is not own)
            if not hidden:
                self.memory.missed(thing.id)

    def _sync_obstacles(self) -> None:
        """Share what's been seen: put it on the map every drone plans against."""
        current = {t.id: t for t in self.memory.obstacles()}
        for thing_id in [i for i in self.mapped if i not in current]:
            self.co.remove_obstacle(self.mapped.pop(thing_id).name)
        for thing_id, thing in current.items():
            half = np.maximum(thing.size / 2, 0.05) + OBSTACLE_MARGIN
            lo = np.maximum(thing.p - half, self.co.room.lo)
            hi = np.minimum(thing.p + half, self.co.room.hi)
            lo[2] = max(self.co.room.lo[2], lo[2])
            if np.any(hi <= lo):
                continue
            old = self.mapped.get(thing_id)
            if old is not None and np.max(np.abs(np.array(old.lo) - lo)) < OBSTACLE_MOVE and np.max(np.abs(np.array(old.hi) - hi)) < OBSTACLE_MOVE:
                continue
            obstacle = Obstacle(f"seen-{thing.label}-{thing_id}", tuple(lo.tolist()), tuple(hi.tolist()))
            self.mapped[thing_id] = obstacle
            self.co.add_obstacle(obstacle)
            if old is None and self.search is not None:
                self.search.coverage.obstacle_appeared(lo, hi)

    # ---------------------------------------------------------------- moving and turning
    def aim(self, d, yaw: float) -> None:
        d.config = replace(d.config, yaw=_angle(self.limit_heading(d, yaw), 0.0))

    def order(self, d, goal, face: float | None, now: float) -> None:
        self.orders[d.id] = Order(np.asarray(goal, float), face, "turn", now)

    def _drive(self, now: float) -> None:
        for drone_id, order in list(self.orders.items()):
            d = self.co.drones[drone_id]
            if d.phase != Phase.AIR or d.landing_requested:
                del self.orders[drone_id]
                continue
            self._advance(d, order, now)
        for d in self.co.drones.values():
            # Without an order it may still be flying somewhere, such as back to its pad. Look that way too.
            if d.id not in self.orders and d.phase == Phase.AIR and d.goal is not None and not d.arrived:
                self._look_along_path(d)

    def _advance(self, d, order: Order, now: float) -> None:
        p = d.tracked.p
        if order.stage == "turn":
            offset = order.goal - p
            if math.hypot(offset[0], offset[1]) > 0.4:
                self.aim(d, math.atan2(offset[1], offset[0]))
                if abs(_angle(d.config.yaw, d.tracked.yaw)) > TURN_FIRST and now - order.since < TURN_TIMEOUT_S:
                    return
            problems = self.co.goto({d.id: order.goal})
            if problems:
                order.stage, order.why = "failed", "; ".join(problems)
                return
            order.stage, order.since = "go", now
        if order.stage == "go":
            if "path" in d.faults or now - order.since > GO_TIMEOUT_S:
                order.stage, order.why = "failed", d.faults.get("path", "took too long")
                self.co._set_goal(d, p.copy())
                return
            if not d.arrived:
                self._look_along_path(d)
                return
            order.stage, order.since = "face", now
        if order.stage == "face":
            if order.face is None:
                order.stage = "done"
                return
            self.aim(d, order.face)
            # d.config.yaw, not order.face: the tracker may have overruled the heading asked for.
            if abs(_angle(d.config.yaw, d.tracked.yaw)) < FACED or now - order.since > TURN_TIMEOUT_S:
                order.stage = "done"

    def _look_along_path(self, d) -> None:
        if not d.path or d.leg >= len(d.path):
            return
        offset = d.path[d.leg] - d.tracked.p
        if math.hypot(offset[0], offset[1]) > 0.4:
            self.aim(d, math.atan2(offset[1], offset[0]))

    # ---------------------------------------------------------------- what can be asked
    # Plain methods taking plain values, each returning a sentence for the operator. Text from a
    # person goes through command(), which only parses and calls one of these. All are thread-safe.
    def command(self, text: str) -> str:
        """Do what was said and answer. Anything not understood changes nothing."""
        with self.lock:
            self.memory.note(self.now, "said", text)
            self.memory.commit()
            c = parse(text, tuple(self.co.drones))
            if c.verb == "refused":
                return c.problem
            if c.verb == "help":
                return EXPLANATION
            if c.verb == "unknown":
                return "I didn't understand that, so I did nothing. " + EXPLANATION
            if c.verb == "goto":
                if c.point is not None:
                    return self.go_to(*c.point, drone=c.drone)
                return self.go_to_thing(c.label, drone=c.drone)
            if c.verb == "form":
                return self.form(c.shape)
            if c.verb in ("find", "where", "forget"):
                return getattr(self, c.verb)(c.label)
            return {"estop": self.estop, "takeoff": self.takeoff, "land": self.land, "land_here": self.land_here,
                    "hold": self.hold, "status": self.status, "list": self.seen}[c.verb]()

    def _stop_search(self) -> None:
        """End any search and the moves it ordered. A plain "go to" for one drone leaves the others' moves alone."""
        if self.search is not None and self.search.state in ("searching", "closing"):
            self.search.state = "stopped"
            self.orders.clear()

    @_action
    def estop(self) -> str:
        """Cut every motor. The drones fall."""
        self.co.estop()
        self._stop_search()
        self.orders.clear()
        return "Motors cut on every drone. They are falling."

    @_action
    def takeoff(self) -> str:
        problems = self.co.takeoff(self.now)
        return "Not taking off: " + "; ".join(problems) + "." if problems else "Taking off, one at a time."

    @_action
    def land(self) -> str:
        """Fly back to the pads, then land one after another."""
        self._stop_search()
        self.orders.clear()
        self.co.land(self.now)
        return "Coming back to the pads to land."

    @_action
    def land_here(self) -> str:
        self._stop_search()
        self.orders.clear()
        self.co.land_here(self.now)
        return "Landing where they are."

    @_action
    def hold(self) -> str:
        """Stop moving and end any search. The motors stay on."""
        self._stop_search()
        self.orders.clear()
        self.co.hold()
        return "Holding position."

    @_action
    def form(self, shape: str) -> str:
        """Line up the flying drones. shape is 'line', 'ring' or 'grid'."""
        if not [d for d in self.co.drones.values() if d.phase == Phase.AIR]:
            return "Nothing is flying yet."
        self._stop_search()
        self.orders.clear()
        center = (self.co.room.lo[:2] + self.co.room.hi[:2]) / 2
        problems = self.co.formation(shape, center, 1.0, SEARCH_Z)
        return f"Can't do that: {'; '.join(problems)}." if problems else f"Forming a {shape}."

    def _flying_for(self, drone: str | None) -> tuple[list, str | None]:
        flying = [d for d in self.co.drones.values() if d.phase == Phase.AIR and not d.landing_requested]
        if drone is None:
            return flying, None if flying else "Nothing is flying yet."
        if drone not in self.co.drones:
            return [], f"There is no drone called {drone}."
        flying = [d for d in flying if d.id == drone]
        return flying, None if flying else f"{drone} isn't flying."

    @_action
    def go_to(self, x: float, y: float, z: float | None = None, drone: str | None = None) -> str:
        """Send one drone to a room position in metres: the named one, or else the nearest."""
        flying, problem = self._flying_for(drone)
        if problem:
            return problem
        goal = np.array([x, y, SEARCH_Z if z is None else z], float)
        d = min(flying, key=lambda f: float(np.linalg.norm(f.tracked.p - goal)))
        problems = self.co.check_goals({d.id: goal})
        if problems:
            return f"Can't go there: {'; '.join(problems)}."
        self._stop_search()
        self.order(d, goal, None, self.now)
        return f"{d.id} is going to {_where(goal)}."

    @_action
    def go_to_thing(self, label: str, drone: str | None = None) -> str:
        """Send one drone to a clear view of a remembered thing."""
        flying, problem = self._flying_for(drone)
        if problem:
            return problem
        things = self.memory.find(label)
        if not things:
            return f"I haven't seen a {label}. Search for one first."
        thing = things[0]
        d = min(flying, key=lambda f: float(np.linalg.norm(f.tracked.p - thing.p)))
        others = [f.tracked.p for f in flying if f is not d]
        view = self.new_coverage().standoff(d.tracked.p, thing.p, avoid=others, tilt=self.tilt_of(d.id))
        if view is None:
            return f"I can't find a clear place to see the {label} from."
        self._stop_search()
        self.order(d, view.p, view.yaw, self.now)
        return f"{d.id} is going to the {label}."

    @_action
    def find(self, label: str) -> str:
        """Search the room until it's confirmed or all the floor has been seen. See hub.search.state."""
        if not self.eyes.knows(label):
            return f"The camera can't recognise \"{label}\", so I didn't start a search."
        drones = self.co.drones.values()
        if all(d.phase in (Phase.GROUND, Phase.LANDING, Phase.STOPPED) for d in drones) and not self.co.takeoff_queue:
            return f"Nothing is flying. Take off first, then ask me to find the {label}."
        hints = [t for t in self.memory.find(label) if self.now - t.last_seen <= HINT_MAX_AGE_S and not t.moves]
        self._stop_search()
        self.search = Search(self, label, self.now, hints[0] if hints else None)
        if hints:
            return f"I last saw a {label} near {_where(hints[0].p)}. Checking there first, then the rest of the room."
        return f"Looking for a {label}."

    @_action
    def where(self, label: str) -> str:
        things = self.memory.find(label)
        if not things:
            return f"I haven't seen a {label}."
        parts = [f"{_where(t.p)}, seen {_ago(self.now - t.last_seen)} by {_names(t.seen_by)}" for t in things[:3]]
        doubt = "" if things[0].sure else " I'm not certain it's really one."
        return f"{label.capitalize()}: " + "; ".join(parts) + "." + doubt

    @_action
    def seen(self) -> str:
        """Everything the fleet remembers."""
        things = self.memory.known()
        if not things:
            return "Nothing yet."
        return "I've seen: " + "; ".join(f"{t.label} at {_where(t.p)}" for t in things) + "."

    @_action
    def forget(self, label: str | None = None) -> str:
        """Forget every thing with this label, or everything when label is None."""
        count = self.memory.forget_label(label)
        self._sync_obstacles()
        if label is None:
            return f"Forgot all {count} things."
        return f"Forgot {count} {label}." if count else f"I had nothing about a {label}."

    @_action
    def status(self) -> str:
        parts = []
        for d in self.co.drones.values():
            fault = f" ({'; '.join(d.faults.values())})" if d.faults else ""
            parts.append(f"{d.id} {d.phase.value} at {_where(d.tracked.p)}{fault}")
        text = ", ".join(parts) + f". I remember {len(self.memory.known())} things."
        if self.search is not None:
            text += f" Search for {self.search.label}: {self.search.state}, {self.search.coverage.share_seen() * 100:.0f}% of the floor seen."
        return text


def _where(p) -> str:
    return f"({p[0]:.1f}, {p[1]:.1f})"


def _names(names) -> str:
    return " and ".join(sorted(names))


def _ago(seconds: float) -> str:
    if seconds < 2:
        return "just now"
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    return f"{seconds / 60:.0f} min ago"
