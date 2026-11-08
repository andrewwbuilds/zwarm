"""Runs the fleet: one tick reads poses, decides every drone's velocity together, and sends sticks.

It takes its clock and poses as arguments and talks to drones only through links, so the same code
flies the simulator faster than real time and real drones at real time.

S20 takeoff and landing share one toggle bit, so a repeat reverses the action. Each toggle is sent
once and judged by what the tracker then sees. If it doesn't work, that's a fault for the operator,
never a retry. A land toggle sent to a drone already on the floor starts its motors, so none goes
out to a drone whose last sightings or still camera say it's down.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm.control import Airframe, Pose, StickMapper, Tracked
from swarm.planner import PlanError, plan_path
from swarm.room import Room
from swarm.safety import Body, Limits, Moving, filter_velocities

logger = logging.getLogger(__name__)

TTL = 0.3  # the drone's backend centres the sticks if no newer command arrives in this time
STALE_S = 0.3  # no pose for this long and the drone stops being steered
LOST_LAND_S = 2.5  # no pose for this long while flying and it is told to land where it is
TAKEOFF_STAGGER_S = 1.0
LAND_STAGGER_S = 1.5
TAKEOFF_TIMEOUT_S = 8.0
LAND_TIMEOUT_S = 15.0
PAD_APPROACH_TIMEOUT_S = 45.0
AIRBORNE_HEIGHT = 0.4
GROUNDED_HEIGHT = 0.15
ARRIVED = 0.15
LOOKAHEAD = 0.35
OVERSPEED_FACTOR = 3.0  # this many times the speed limit, sustained, and the drone is landed
FENCE_BREACH = 0.4  # this far past the fence and the drone is landed
STUCK_S = 4.0
FOLLOW_WINDOW_S = 3.0
SIGMA_HOLD = 0.3  # m of position uncertainty at which a drone stops being steered and holds
SIGMA_LAND = 0.6  # and at which it is told to land where it is
CHECK_NEEDS_FIX_S = 0.5  # the response check can only judge what a real sighting showed
DOWN_AFTER_S = 1.0  # this long with no sighting before a low or falling last sighting means it came down
LOW_SIGHTING = 0.4  # m: no flight holds this low, so a drone last seen here was on its way to the floor
FALL_DROP = 0.25  # m the sightings fell by, over FALL_WINDOW_S, while nothing asked it to go down
FALL_WINDOW_S = 1.5
STILL_AFTER_S = 2.5  # no sighting for this long and a still camera: the fall is over and it's on the floor
SEEN_UP_FIX_S = 1.0  # a sighting this recent, this high, proves it is in the air
VIDEO_STALE_S = 2.0  # no frame for this long and takeoff is refused: position comes only from video
NEXT_TAKEOFF_HEIGHT = 0.25  # the next drone goes once the last one is seen this far up
CHECK_SPEED = 0.4  # m/s for the small test moves after takeoff
CHECK_MOVE_S = 1.2
CHECK_STOP_S = 2.5
CHECK_SAMPLE_S = 1.0  # how much of the end of a stop gets measured
CHECK_MIN_SPEED = 0.08  # it has to have got at least this fast, the right way
CHECK_WRONG_WAY = 0.2  # going backwards this fast part way through a move ends the check at once
CHECK_STILL_SPEED = 0.13  # stopped, whatever it was doing before
CHECK_STOP_RATIO = 0.5  # or at least slowed this much: no flow sensor means it coasts
CHECK_MAX_COAST = 0.75  # m covered after being told to stop: past this it fails at once


class Phase(str, Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CHECK = "check"  # airborne, proving it does what the sticks say before going anywhere
    AIR = "air"
    LANDING = "landing"
    STOPPED = "stopped"


@dataclass
class ResponseCheck:
    """Move a little along each body axis, then stop. Finds reversed or dead sticks, and a drone
    that won't stop because its flow sensor can't read the floor, before it goes near anything."""

    directions: list[np.ndarray]
    home: np.ndarray
    stage: int = 0  # even: moving, odd: stopping
    stage_at: float = 0.0
    samples: list[tuple[float, float]] = field(default_factory=list)  # when, and how far along by the tags
    moved: float = 0.0
    stop_at: np.ndarray = field(default_factory=lambda: np.zeros(3))
    results: list[dict] = field(default_factory=list)
    retried: bool = False


@dataclass(frozen=True)
class DroneConfig:
    id: str
    pad: tuple[float, float, float]  # where it sits before takeoff, and returns to
    yaw: float = 0.0  # radians, the heading it holds in flight


@dataclass
class Drone:
    config: DroneConfig
    link: object
    mapper: StickMapper
    tracked: Tracked = field(default_factory=Tracked)
    phase: Phase = Phase.GROUND
    goal: np.ndarray | None = None
    path: list = field(default_factory=list)
    leg: int = 1
    path_version: int = -1
    toggled_at: float = 0.0
    landing_requested: bool = False
    land_requested_at: float = 0.0
    blind_landing: bool = False
    faults: dict[str, str] = field(default_factory=dict)
    arrived: bool = False
    limited: bool = False
    commanded: np.ndarray = field(default_factory=lambda: np.zeros(3))
    overspeed_since: float | None = None
    progress_at: float = 0.0
    progress_best: float = math.inf
    last_replan: float = -math.inf
    follow_errors: deque = field(default_factory=deque)
    follow_error: float = 0.0
    follow_worst: float = 0.0
    check: ResponseCheck | None = None
    check_results: list[dict] = field(default_factory=list)
    sticks_wrong: bool = False  # failed its check in a way that means steering it would make things worse
    unknown_air: bool = False  # takeoff was sent and nothing was seen after, so it may or may not be up
    came_down: bool = False  # it looked to be on the floor. If it turns out not to be, it goes back to holding
    was_down: bool = False  # it looked down and then wasn't, so it gets no more toggles without the operator
    land_here_wanted: bool = False  # asked to land on the spot when no toggle could go out yet
    check_held: bool = False  # the operator said hold: no test moves until they ask for work again
    backend_grounded: bool = False  # a land toggle went out and didn't work, so its backend wrongly thinks it's down

    @property
    def id(self) -> str:
        return self.config.id

    def fault(self, key: str, message: str) -> None:
        if self.faults.get(key) != message:
            self.faults[key] = message
            logger.warning("[swarm] %s: %s", self.id, message)

    def clear(self, key: str) -> None:
        self.faults.pop(key, None)


def formation_points(shape: str, count: int, center, spacing: float, z: float, turn: float = 0.0) -> list[np.ndarray]:
    """Slots for a formation: 'line', 'ring' (spacing is the gap between neighbours) or 'grid'."""
    if shape == "line":
        local = [((i - (count - 1) / 2) * spacing, 0.0) for i in range(count)]
    elif shape == "ring":
        radius = spacing / (2 * math.sin(math.pi / count)) if count > 1 else 0.0
        local = [(radius * math.cos(2 * math.pi * i / count), radius * math.sin(2 * math.pi * i / count)) for i in range(count)]
    elif shape == "grid":
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        local = [((c - (columns - 1) / 2) * spacing, (r - (rows - 1) / 2) * spacing)
                 for r in range(rows) for c in range(columns)][:count]
    else:
        raise ValueError(f"unknown formation {shape!r}")
    cos_t, sin_t = math.cos(turn), math.sin(turn)
    return [np.array([center[0] + x * cos_t - y * sin_t, center[1] + x * sin_t + y * cos_t, z]) for x, y in local]


class Coordinator:
    def __init__(self, room: Room, configs: list[DroneConfig], links: dict, limits: Limits | None = None,
                 airframe: Airframe | None = None, visible=None, tracker=None):
        self.room = room
        self.tracker = tracker  # an InsideOutTracker, told what each drone was commanded to do
        self.visible = visible  # whether the tracker can see a given point; None means it always can
        self.limits = limits or Limits()
        self.airframe = airframe or Airframe()
        self.drones = {c.id: Drone(c, links[c.id], StickMapper(self.airframe)) for c in configs}
        self.takeoff_queue: list[str] = []
        self.last_takeoff: str | None = None
        self.next_takeoff = 0.0
        self.next_landing = 0.0
        self.moving: list[Moving] = []
        self.now = 0.0
        self.closest = math.inf
        self.infeasible_ticks = 0
        self.estopped = False
        self.last_tick: float | None = None

    # --------------------------------------------------------------- operator commands
    def preflight(self, now: float) -> list[str]:
        """Reasons the fleet must not take off. Empty means go."""
        problems = []
        drones = list(self.drones.values())
        for d in drones:
            if d.phase != Phase.GROUND:
                problems.append(f"{d.id} is not on the ground ({d.phase.value})")
            if not d.link.connected:
                problems.append(f"{d.id} has no control link")
            if getattr(d.link, "problem", None):
                problems.append(f"{d.id}: {d.link.problem}")
            if d.tracked.age(now) > STALE_S:
                problems.append(f"{d.id} is not being tracked")
                continue
            p = d.tracked.p
            if p[2] > GROUNDED_HEIGHT + 0.1:
                problems.append(f"{d.id} reads {p[2]:.2f} m up, expected on the floor")
            miss = float(np.linalg.norm(p[:2] - np.array(d.config.pad[:2])))
            if miss > 0.3:
                problems.append(f"{d.id} is {miss:.2f} m from its pad")
            if self.room.clearance(np.array([p[0], p[1], 1.0])) < self.limits.surface_clearance:
                problems.append(f"{d.id} is too close to a wall or obstacle to climb")
            if d.unknown_air:
                problems.append(f"{d.id} may be on the floor or hovering unseen: check it by eye, then reset it")
            if self.tracker is not None and not self.tracker.expected(d.id):
                problems.append(f"{d.id} would see no floor markers hovering over its pad")
            if self.tracker is not None and self.tracker.video_age(d.id, now) > VIDEO_STALE_S:
                problems.append(f"{d.id} has sent no video for {VIDEO_STALE_S:.0f} s or more, and it would fly with no position")
        for i, a in enumerate(drones):
            for b in drones[i + 1:]:
                gap = float(np.linalg.norm(np.array(a.config.pad[:2]) - np.array(b.config.pad[:2])))
                if gap < self.limits.separation:
                    problems.append(f"pads {a.id} and {b.id} are {gap:.2f} m apart, need {self.limits.separation:.2f} m")
        return problems

    def takeoff(self, now: float | None = None) -> list[str]:
        now = self.now if now is None else now
        problems = self.preflight(now)
        if problems:
            return problems
        self.estopped = False
        self.last_takeoff = None
        for d in self.drones.values():
            self._forget_requests(d)
        self.takeoff_queue = list(self.drones)
        self.next_takeoff = now
        return []

    def check_goals(self, goals: dict) -> list[str]:
        problems = []
        limits = self.limits
        items = list(goals.items())
        for drone_id, goal in items:
            if drone_id not in self.drones:
                problems.append(f"no drone called {drone_id}")
                continue
            if self.drones[drone_id].phase != Phase.AIR:
                problems.append(f"{drone_id} isn't ready ({self.drones[drone_id].phase.value})")
            p = np.asarray(goal, float)
            if self.room.clearance(p) < limits.surface_clearance:
                problems.append(f"{drone_id}'s goal {p.round(2).tolist()} is too close to a wall or obstacle")
            if p[2] < 0.4:
                problems.append(f"{drone_id}'s goal is only {p[2]:.2f} m up")
            if self.visible is not None and not self.visible(p):
                problems.append(f"{drone_id}'s goal {p.round(2).tolist()} is outside what the cameras can see")
        for i, (a, pa) in enumerate(items):
            for b, pb in items[i + 1:]:
                offset = np.asarray(pa, float) - np.asarray(pb, float)
                gap = math.hypot(offset[0], offset[1], offset[2] / limits.vertical_stretch)
                if gap < limits.separation:
                    problems.append(f"goals for {a} and {b} are {gap:.2f} m apart, need {limits.separation:.2f} m")
        return problems

    def goto(self, goals: dict) -> list[str]:
        goals = {k: np.asarray(v, float) for k, v in goals.items()}
        problems = self.check_goals(goals)
        if problems:
            return problems
        for drone_id, goal in goals.items():
            d = self.drones[drone_id]
            d.landing_requested = d.land_here_wanted = False
            self._set_goal(d, goal)
        return []

    def formation(self, shape: str, center, spacing: float, z: float, turn: float = 0.0) -> list[str]:
        """Send every flying drone to a formation, choosing who goes where to shorten the total flight."""
        flying = [d for d in self.drones.values() if d.phase == Phase.AIR]
        if not flying:
            return ["no drones are flying"]
        slots = formation_points(shape, len(flying), center, spacing, z, turn)
        cost = np.array([[np.linalg.norm(d.tracked.p - slot) for slot in slots] for d in flying])
        rows, columns = linear_sum_assignment(cost)
        return self.goto({flying[r].id: slots[c] for r, c in zip(rows, columns)})

    def hold(self) -> None:
        for d in self.drones.values():
            d.landing_requested = d.land_here_wanted = False
            if d.phase == Phase.AIR:
                self._set_goal(d, d.tracked.p.copy())
        self.hold_checks()
        self.takeoff_queue.clear()

    def hold_checks(self) -> None:
        """No test moves until the operator asks for work again. A drone still climbing waits too."""
        for d in self.drones.values():
            if d.phase in (Phase.TAKEOFF, Phase.CHECK):
                d.check_held = True

    def resume_checks(self) -> None:
        for d in self.drones.values():
            d.check_held = False

    def land(self, now: float | None = None) -> None:
        """Bring each flying drone back over its pad, then land them one after another.

        A drone still climbing can't take the toggle yet, so it's asked to land as soon as it's up.
        """
        now = self.now if now is None else now
        self.takeoff_queue.clear()
        self.next_landing = now
        for d in self.drones.values():
            if d.phase == Phase.CHECK:
                d.check = None
                d.phase = Phase.AIR
            if d.phase == Phase.TAKEOFF:
                d.landing_requested = True
                d.land_requested_at = now
            if d.phase == Phase.AIR:
                d.landing_requested = True
                d.land_requested_at = now
                self._go_to_pad(d)

    def _go_to_pad(self, d: Drone) -> None:
        self._set_goal(d, np.array([d.config.pad[0], d.config.pad[1], max(d.tracked.p[2], 0.7)]))

    def land_here(self, now: float | None = None) -> None:
        """Land where they are. A drone that can't take the toggle yet, climbing or with its link down, gets it when it can."""
        now = self.now if now is None else now
        self.takeoff_queue.clear()
        for d in self.drones.values():
            if d.phase == Phase.CHECK:
                d.check = None
                d.phase = Phase.AIR
            if d.phase in (Phase.TAKEOFF, Phase.AIR):
                d.land_here_wanted = True
            if d.phase == Phase.AIR:
                self._toggle_land(d, now)

    def estop(self) -> None:
        """Cut every motor. The drones fall."""
        self.takeoff_queue.clear()
        self.estopped = True
        for d in self.drones.values():
            d.link.command("estop")
            d.phase = Phase.STOPPED
            d.goal = None
            d.path = []
            self._forget_requests(d)

    @staticmethod
    def _forget_requests(d: Drone) -> None:
        """Nothing asked of it before this still applies, including a landing from the last flight."""
        d.landing_requested = d.land_here_wanted = d.blind_landing = d.check_held = False
        d.check = None

    def reset_to_pads(self) -> list[str]:
        """The operator has put the landed drones back on their pads."""
        problems = [f"{d.id} is {d.phase.value}" for d in self.drones.values()
                    if d.phase not in (Phase.GROUND, Phase.STOPPED)]
        if problems:
            return problems
        for d in self.drones.values():
            d.phase = Phase.GROUND
            d.unknown_air = d.came_down = d.was_down = d.backend_grounded = d.sticks_wrong = False
            self._forget_requests(d)
            d.faults.clear()
            d.goal = None
            d.path = []
            d.tracked = Tracked()
            if self.tracker is not None:
                self.tracker.reset(d.id)
        self.estopped = False
        return []

    def add_obstacle(self, obstacle) -> None:
        self.room.add(obstacle)

    def remove_obstacle(self, name: str) -> bool:
        return self.room.remove(name)

    # --------------------------------------------------------------- the tick
    def tick(self, now: float, poses: dict[str, Pose], moving: list[Moving] = ()) -> None:
        dt = 0.05 if self.last_tick is None else max(now - self.last_tick, 1e-3)
        self.last_tick = self.now = now
        self.moving = list(moving)
        for drone_id, pose in poses.items():
            if drone_id in self.drones:
                self.drones[drone_id].tracked.update(pose)
        self._send_takeoffs(now)
        for d in self.drones.values():
            self._supervise(d, now)
        self._send_landings(now)
        self._fly(now, dt)

    def _set_goal(self, d: Drone, goal: np.ndarray, cylinders=()) -> None:
        d.goal = np.asarray(goal, float)
        d.arrived = False
        d.progress_best = math.inf
        d.progress_at = self.now
        self._plan(d, cylinders)

    def _plan(self, d: Drone, cylinders=()) -> None:
        try:
            d.path = plan_path(self.room, d.tracked.p.copy(), d.goal, self.limits.surface_clearance + 0.05, cylinders)
            d.leg = 1
            d.path_version = self.room.version
            d.clear("path")
        except PlanError as error:
            d.path = []
            d.fault("path", str(error))

    def _send_takeoffs(self, now: float) -> None:
        if self.takeoff_queue and now >= self.next_takeoff:
            if self.last_takeoff is not None:
                last = self.drones[self.last_takeoff]
                if last.phase == Phase.TAKEOFF and last.tracked.p[2] < NEXT_TAKEOFF_HEIGHT:
                    return
            d = self.drones[self.takeoff_queue.pop(0)]
            if d.link.command("takeoff"):
                d.phase = Phase.TAKEOFF
                d.toggled_at = now
                self.last_takeoff = d.id
                self.next_takeoff = now + TAKEOFF_STAGGER_S
            else:
                d.fault("takeoff", "takeoff could not be sent")
                self.takeoff_queue.clear()

    def _looks_down(self, d: Drone, now: float) -> str | None:
        """Why this drone seems to be on the floor already, or None.

        Goes by real sightings and the camera. The pose estimate only echoes the commands, and it has been
        seen rising while the drone lay on the floor.
        """
        seen = d.tracked.seen_log
        if not seen:
            return None
        silent = now - seen[-1][0]
        if silent < DOWN_AFTER_S:
            return None
        z = seen[-1][1]
        asked = None if d.goal is None else float(d.goal[2])
        if z < LOW_SIGHTING and (asked is None or asked >= LOW_SIGHTING):
            return f"last seen only {z:.2f} m up"
        drop = d.tracked.seen_drop(FALL_WINDOW_S)
        if drop > FALL_DROP and (asked is None or z < asked - FALL_DROP):
            return f"last seen falling, {drop:.2f} m in {FALL_WINDOW_S:.1f} s"
        if silent >= STILL_AFTER_S and self.tracker is not None and self.tracker.camera_still(d.id, now):
            return "its camera has stopped moving and it sees no floor markers"
        return None

    def _mark_down(self, d: Drone, why: str) -> None:
        """Treat it as on the floor: no sticks, and above all no toggle. A sighting up high brings it back."""
        d.phase = Phase.GROUND
        d.unknown_air = d.came_down = True
        d.goal = None
        d.path = []
        self._forget_requests(d)
        getattr(d.link, "assume_grounded", lambda: None)()  # so the backend's own Land all can't start it either
        d.fault("down", f"{why}, so it's probably on the floor. No landing was sent, because that toggle would start its "
                        "motors. Check it by eye before going near it")

    def _back_in_air(self, d: Drone) -> None:
        """Written off as down, and now seen flying. It holds, and any toggle waits for the operator."""
        d.phase = Phase.AIR
        d.unknown_air = d.came_down = False
        d.was_down = d.backend_grounded = True
        d.clear("down")
        d.fault("land", "it was seen flying after it looked to be down. It's holding and won't land by itself: press LAND")
        self._set_goal(d, d.tracked.p.copy())

    def _toggle_land(self, d: Drone, now: float, blind: bool = False) -> None:
        down = self._looks_down(d, now)
        if down:
            self._mark_down(d, down)
            return
        if blind and d.was_down:
            d.fault("tracking", "position lost, and it already looked down once, so it won't be landed blind: check it by eye")
            return
        if d.backend_grounded and d.link.connected:
            getattr(d.link, "assume_flying", lambda: None)()
            d.backend_grounded = False
        if d.link.command("land"):
            d.phase = Phase.LANDING
            d.toggled_at = now
            d.blind_landing = blind
            d.land_here_wanted = False
            d.goal = None
            d.path = []
            d.clear("land_link")
        else:
            # Nothing went out, so a retry can't reverse anything. Not a failed landing either.
            d.fault("land_link", "landing could not be sent; the link is down")

    def _send_landings(self, now: float) -> None:
        if now < self.next_landing:
            return
        for d in self.drones.values():
            if d.phase != Phase.AIR or not d.landing_requested:
                continue
            over_pad = np.linalg.norm(d.tracked.p[:2] - np.array(d.config.pad[:2])) < 0.25
            slow = np.linalg.norm(d.tracked.v[:2]) < 0.25
            late = now - d.land_requested_at > PAD_APPROACH_TIMEOUT_S
            if (over_pad and slow) or late:
                if late:
                    d.fault("land", "could not reach its pad in time; landing where it is")
                self._toggle_land(d, now)
                self.next_landing = now + LAND_STAGGER_S
                return

    def _supervise(self, d: Drone, now: float) -> None:
        age = d.tracked.age(now)
        height = d.tracked.p[2]
        if age > STALE_S and d.phase in (Phase.AIR, Phase.CHECK, Phase.TAKEOFF, Phase.LANDING):
            d.fault("tracking", f"no pose for {age:.1f} s")
        elif age <= STALE_S:
            d.clear("tracking")
        if d.link.connected:
            d.clear("link")
        elif d.phase in (Phase.AIR, Phase.CHECK, Phase.TAKEOFF, Phase.LANDING):
            d.fault("link", "control link lost; its backend has centred the sticks")

        if d.phase == Phase.TAKEOFF:
            if age <= STALE_S and height > AIRBORNE_HEIGHT and abs(d.tracked.v[2]) < 0.15:
                self._begin_check(d, now)
            elif now - d.toggled_at > TAKEOFF_TIMEOUT_S:
                if age <= STALE_S and height > AIRBORNE_HEIGHT:
                    self._begin_check(d, now)
                else:
                    d.phase = Phase.GROUND
                    d.fault("takeoff", "takeoff was sent but it was not seen to climb; not sending again, the toggle "
                                       "would reverse it. It may be hovering unseen: check it by eye")
                    d.unknown_air = self.tracker is not None
                    if not d.unknown_air:
                        self._forget_requests(d)
                    self.takeoff_queue.clear()
        elif d.phase == Phase.GROUND and d.unknown_air:
            if age <= STALE_S and d.tracked.fix_age < SEEN_UP_FIX_S and height > AIRBORNE_HEIGHT:
                if d.came_down:
                    self._back_in_air(d)
                else:
                    d.unknown_air = False
                    d.clear("takeoff")
                    self._begin_check(d, now)
        elif d.phase == Phase.LANDING:
            seen_up = age <= STALE_S and d.tracked.fix_age < SEEN_UP_FIX_S and height > AIRBORNE_HEIGHT
            if age <= STALE_S and height < GROUNDED_HEIGHT and abs(d.tracked.v[2]) < 0.15:
                blind = d.blind_landing
                d.phase = Phase.GROUND
                self._forget_requests(d)
                d.sticks_wrong = False
                d.clear("land")
                if blind and d.tracked.fix_age >= SEEN_UP_FIX_S:
                    # The estimate assumes a descent, so with no sighting this says nothing about the drone.
                    d.unknown_air = d.came_down = True
                    d.fault("land", "landed blind and touchdown was never seen; check it by eye")
            elif now - d.toggled_at > LAND_TIMEOUT_S:
                if d.blind_landing and not seen_up:
                    d.fault("land", "landing was sent blind and touchdown was not seen; check it by eye")
                    d.phase = Phase.GROUND
                    d.unknown_air = d.came_down = True
                    self._forget_requests(d)
                else:
                    d.fault("land", "landing was sent but it's still in the air, so the packet was probably lost. "
                                    "It's holding. Press LAND to send it once more")
                    d.phase = Phase.AIR
                    d.landing_requested = d.blind_landing = False
                    d.backend_grounded = True  # told to land, so its backend won't pass on another landing
                    self._set_goal(d, d.tracked.p.copy())
        elif d.phase in (Phase.AIR, Phase.CHECK):
            if age > LOST_LAND_S and d.link.connected:
                d.fault("tracking", f"no pose for {age:.1f} s; landing it where it is")
                self._toggle_land(d, now, blind=True)
                return
            sigma = d.tracked.sigma
            if sigma > SIGMA_LAND and d.link.connected:
                d.check = None
                d.fault("tracking", f"position uncertain by {sigma:.2f} m after {d.tracked.fix_age:.0f} s without a "
                                    "floor marker; landing it where it is")
                self._toggle_land(d, now, blind=True)
                return
            if sigma > SIGMA_HOLD:
                d.fault("tracking", f"position uncertain by {sigma:.2f} m; holding until it sees a floor marker")
            elif age <= STALE_S:
                d.clear("tracking")
            if age <= STALE_S:
                self._watch_for_runaway(d, now)

    def _begin_check(self, d: Drone, now: float) -> None:
        d.phase = Phase.CHECK
        d.arrived = False
        yaw = d.tracked.yaw
        axes = [np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])]
        directions = []
        for axis in axes:
            def room_for(sign):
                # Walls are checked further out: a drone that fails this test may coast a long way.
                ahead = d.tracked.p + sign * axis * 0.8
                far = d.tracked.p + sign * axis * 1.6
                others = [np.linalg.norm(ahead[:2] - other.tracked.p[:2]) - self.limits.separation
                          for other in self.drones.values() if other is not d and other.tracked.t is not None]
                return min([self.room.clearance(far) - self.limits.surface_clearance, *others])
            directions.append(axis if room_for(1.0) >= room_for(-1.0) else -axis)
        d.check = ResponseCheck(directions, d.tracked.p.copy(), 0, now)
        d.path = []
        d.goal = None

    def _check_step(self, d: Drone, now: float) -> np.ndarray:
        check = d.check
        if d.tracked.fix_age > CHECK_NEEDS_FIX_S:
            # Nothing real to judge it by, and the estimate would only echo the command. Wait.
            check.stage_at = now
            check.samples = []
            return np.zeros(3)
        if check.stage % 2 == 1:
            coast = float(np.linalg.norm(d.tracked.seen_p[:2] - check.stop_at[:2]))
            if coast > CHECK_MAX_COAST:
                d.check_results = check.results
                d.check = None
                d.fault("response", f"it drifted {coast:.2f} m after being told to stop, so it isn't "
                                    "holding position; landing it where it is")
                self._toggle_land(d, now)
                return np.zeros(3)
        direction = check.directions[check.stage // 2]
        moving = check.stage % 2 == 0
        length = CHECK_MOVE_S if moving else CHECK_STOP_S
        elapsed = now - check.stage_at
        # Judged by how far sightings say it went. A speed estimate would need steady frames, and
        # part of it echoes the command, which would hide sticks that do nothing.
        along = float(d.tracked.seen_p @ direction)
        if not check.samples or check.samples[-1][1] != along:
            check.samples.append((now, along))
        if moving and elapsed > 0.8 and along - check.samples[0][1] < -CHECK_WRONG_WAY:
            # Don't wait for the end of the move: every correction would now make it worse.
            d.check_results = check.results
            d.check = None
            d.sticks_wrong = True
            d.fault("response", f"it moved the wrong way ({along - check.samples[0][1]:+.2f} m), so a stick is reversed "
                                "or the heading is wrong; landing it where it is")
            self._toggle_land(d, now)
            return np.zeros(3)
        if elapsed < length:
            return direction * min(CHECK_SPEED, self.limits.max_speed) if moving else np.zeros(3)
        # The start of a move is still getting going, so judge its second half, and the end of a stop.
        since = now - (CHECK_MOVE_S / 2 if moving else CHECK_SAMPLE_S)
        window = [(t, x) for t, x in check.samples if t >= since]
        if len(window) < 2:  # sightings are sparse: use what there is
            window = check.samples[-2:]
        span = max(window[-1][0] - window[0][0], 0.25) if window else 1.0
        measured = (window[-1][1] - window[0][1]) / span if window else 0.0
        check.samples = []
        check.stage += 1
        check.stage_at = now
        if moving:
            check.moved = measured
            check.stop_at = d.tracked.seen_p.copy()
            return np.zeros(3)
        check.results.append({"moved": round(check.moved, 3), "after_stop": round(measured, 3)})
        problem = self._judge_check(check.moved, measured)
        if problem and check.moved < CHECK_MIN_SPEED and check.moved > -CHECK_MIN_SPEED and not check.retried:
            # A gust can cancel out a move this small, so give it one more go.
            check.retried = True
            check.stage -= 2
            check.results.pop()
            return np.zeros(3)
        if problem:
            d.check_results = check.results
            d.check = None
            d.sticks_wrong = check.moved < -CHECK_MIN_SPEED
            d.fault("response", f"{problem}; landing it where it is")
            self._toggle_land(d, now)
            return np.zeros(3)
        if check.stage >= 2 * len(check.directions):
            d.check_results = check.results
            speed = sum(r["moved"] for r in check.results) / len(check.results)
            # Only ever raise it. A short move may not reach full speed, and believing a drone is
            # slower than it is makes it fly too fast. Steady flight can correct it downward later.
            d.mapper.scale = max(d.mapper.scale, min(StickMapper.GAIN_RANGE[1], speed / min(CHECK_SPEED, self.limits.max_speed)))
            home = check.home
            # The first floor marker can be read during the climb, so that height may be far too low.
            home[2] = max(home[2], d.tracked.p[2])
            d.check = None
            d.phase = Phase.AIR
            self._set_goal(d, home)
            return np.zeros(3)
        return np.zeros(3)

    @staticmethod
    def _judge_check(moved: float, after_stop: float) -> str | None:
        if moved < -CHECK_MIN_SPEED:
            return f"it moved the wrong way ({moved:+.2f} m/s by the tags), so a stick is reversed or the heading is wrong"
        if moved < CHECK_MIN_SPEED:
            return f"it barely responded to the sticks ({moved:+.2f} m/s)"
        if after_stop > CHECK_STILL_SPEED and after_stop > CHECK_STOP_RATIO * moved:
            return (f"it kept going after being told to stop ({moved:+.2f} then {after_stop:+.2f} m/s), "
                    "so its flow sensor probably can't read this floor")
        return None

    def _watch_for_runaway(self, d: Drone, now: float) -> None:
        """The drone isn't doing what it's told: sticks reversed, flow sensor lost, wind. Land it."""
        limits = self.limits
        speed = float(np.linalg.norm(d.tracked.v[:2]))
        past_fence = float(np.max(np.concatenate([self.room.lo[:2] - d.tracked.p[:2], d.tracked.p[:2] - self.room.hi[:2]])))
        if speed > OVERSPEED_FACTOR * limits.max_speed:
            if d.overspeed_since is None:
                d.overspeed_since = now
        else:
            d.overspeed_since = None
        if past_fence > FENCE_BREACH:
            d.fault("runaway", f"{past_fence:.2f} m past the fence; landing it")
            self._toggle_land(d, now)
        elif d.overspeed_since is not None and now - d.overspeed_since > 0.5:
            d.fault("runaway", f"flying at {speed:.1f} m/s, far above the limit; landing it")
            self._toggle_land(d, now)

    # --------------------------------------------------------------- velocities
    def _steerable(self, d: Drone, now: float) -> bool:
        return (d.phase in (Phase.AIR, Phase.CHECK, Phase.LANDING) and d.link.connected
                and d.tracked.age(now) <= STALE_S and not d.sticks_wrong and d.tracked.sigma <= SIGMA_HOLD)

    def _want(self, d: Drone, now: float) -> np.ndarray:
        limits = self.limits
        if d.phase == Phase.CHECK:
            return self._check_step(d, now)
        if d.phase == Phase.LANDING:
            # The descent is the flight controller's. All that's left is not to drift, which a
            # drone with no flow sensor would otherwise do, into whatever is beside it.
            want = np.zeros(3)
            if np.linalg.norm(d.tracked.v[:2]) > 0.1:
                want[:2] = -0.8 * d.tracked.v[:2]
            return want
        if d.goal is None:
            return np.zeros(3)
        p = d.tracked.p
        error = d.goal - p
        distance = float(np.linalg.norm(error))
        if d.path_version != self.room.version:
            self._plan(d)
        if distance < ARRIVED:
            d.arrived = True
        elif distance > 2 * ARRIVED:
            d.arrived = False
        if d.arrived:
            return np.zeros(3)
        if not d.path:
            return np.zeros(3)
        self._watch_progress(d, now, distance)
        target, final = self._carrot(d, p)
        offset = target - p
        horizontal = float(np.linalg.norm(offset[:2]))
        speed = limits.max_speed if not final else min(limits.max_speed, 1.0 * horizontal)
        want = np.zeros(3)
        if horizontal > 1e-6:
            want[:2] = offset[:2] / horizontal * speed
        want[2] = max(-limits.max_climb, min(limits.max_climb, 1.0 * offset[2]))
        return want

    def _carrot(self, d: Drone, p: np.ndarray) -> tuple[np.ndarray, bool]:
        path = d.path
        while d.leg < len(path) - 1 and np.linalg.norm(path[d.leg] - p) < LOOKAHEAD:
            d.leg += 1
        a, b = path[d.leg - 1], path[d.leg]
        along = b - a
        length = float(np.linalg.norm(along))
        if length < 1e-6:
            return b, d.leg == len(path) - 1
        unit = along / length
        at = float((p - a) @ unit) + LOOKAHEAD
        final = d.leg == len(path) - 1
        if at >= length:
            return b, final
        return a + unit * max(at, 0.0), False

    def _watch_progress(self, d: Drone, now: float, distance: float) -> None:
        """Blocked by another drone parked in the way: plan again, going around it."""
        if distance < d.progress_best - 0.05:
            d.progress_best = distance
            d.progress_at = now
            d.clear("stuck")
            return
        if now - d.progress_at < STUCK_S:
            return
        if now - d.progress_at > 15.0:
            d.fault("stuck", f"no progress for {now - d.progress_at:.0f} s, {distance:.2f} m from its goal")
        if now - d.last_replan < STUCK_S:
            return
        d.last_replan = now
        cylinders = []
        for other in self.drones.values():
            if other is d or other.phase != Phase.AIR or np.linalg.norm(other.tracked.v[:2]) > 0.25:
                continue
            q = other.tracked.p
            if np.linalg.norm(q[:2] - d.tracked.p[:2]) > self.limits.separation + 0.05:
                cylinders.append(((float(q[0]), float(q[1])), self.limits.separation))
        if cylinders:
            # If nothing fits around the parked drones, keep the path and wait for them to move.
            try:
                d.path = plan_path(self.room, d.tracked.p.copy(), d.goal, self.limits.surface_clearance + 0.05, cylinders)
                d.leg = 1
            except PlanError:
                pass

    def _sidestep(self, d: Drone, bodies: list[Body]) -> None:
        """Two drones meeting head on would stop nose to nose, so each keeps to its right."""
        limits = self.limits
        reach = 2.2 * limits.separation
        for other in bodies:
            if other.id == d.id:
                continue
            offset = other.p[:2] - d.tracked.p[:2]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6 or distance > reach:
                continue
            toward = offset / distance
            body = next(b for b in bodies if b.id == d.id)
            if body.want[:2] @ toward <= 0.02:
                continue
            strength = min(1.0, (reach - distance) / (reach - limits.separation))
            right = np.array([toward[1], -toward[0], 0.0])
            body.want = body.want + right * limits.max_speed * 0.8 * strength

    def _fly(self, now: float, dt: float) -> None:
        bodies = []
        for d in self.drones.values():
            if d.tracked.t is None or d.phase == Phase.STOPPED:
                continue
            if d.phase == Phase.GROUND and not d.unknown_air and (d.tracked.age(now) > STALE_S or d.tracked.p[2] < 0.3):
                continue  # on the floor: flight heights keep everyone well above it
            steerable = self._steerable(d, now)
            age = d.tracked.age(now)
            extra = min(1.0, max(0.0, age - STALE_S) * 0.4)
            extra = max(extra, min(1.0, 2 * d.tracked.sigma - self.limits.pose_error))
            if d.unknown_air:
                extra = max(extra, 0.5)
            velocity = d.tracked.v if age <= STALE_S else np.zeros(3)
            if age <= STALE_S:
                position = d.tracked.now(now)
            else:
                # Its sticks centre when commands stop, but it takes a moment to stop moving, so
                # expect it where it would have come to rest, not where it was last seen.
                position = d.tracked.p + d.tracked.v * min(age, STALE_S + self.limits.response)
            want = self._want(d, now) if steerable else np.zeros(3)
            bodies.append(Body(d.id, position, velocity, want, steerable, extra, descending=d.phase == Phase.LANDING))
        for d in self.drones.values():
            if self._steerable(d, now) and d.phase != Phase.LANDING:
                self._sidestep(d, bodies)
        result = filter_velocities(bodies, self.room, self.limits, self.moving)
        self.closest = result.closest
        if result.infeasible:
            self.infeasible_ticks += 1
        for d in self.drones.values():
            if d.phase == Phase.STOPPED:
                continue
            velocity = result.velocities.get(d.id)
            flying = d.phase in (Phase.TAKEOFF, Phase.CHECK, Phase.AIR, Phase.LANDING)
            if velocity is None:
                d.commanded = np.zeros(3)
                d.limited = False
                if d.link.connected and d.phase != Phase.GROUND:
                    d.link.send_sticks(0.0, 0.0, 0.0, 0.0, TTL)
                if self.tracker is not None:
                    self.tracker.note_command(d.id, np.zeros(3), 0.0, flying, d.phase == Phase.LANDING)
                continue
            if d.phase == Phase.LANDING:
                velocity = np.array([velocity[0], velocity[1], 0.0])
            d.commanded = velocity
            d.limited = d.id in result.limited
            yaw = d.tracked.yaw
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            asked = (velocity[0] * cos_yaw + velocity[1] * sin_yaw, -velocity[0] * sin_yaw + velocity[1] * cos_yaw)
            seen = (d.tracked.v[0] * cos_yaw + d.tracked.v[1] * sin_yaw, -d.tracked.v[0] * sin_yaw + d.tracked.v[1] * cos_yaw)
            d.mapper.learn(now, asked[0], asked[1], seen[0], seen[1], dt)
            self._watch_following(d, now, velocity)
            sticks = d.mapper.sticks(velocity, yaw, d.config.yaw, dt)
            d.link.send_sticks(*sticks, TTL)
            if self.tracker is not None:
                self.tracker.note_command(d.id, velocity, -sticks[1] * self.airframe.yaw_rate_per_stick, flying,
                                          d.phase == Phase.LANDING)

    def _watch_following(self, d: Drone, now: float, velocity: np.ndarray) -> None:
        """How far the drone's speed has been from what was asked, over the last few seconds."""
        errors = d.follow_errors
        errors.append((now, float(np.sum((velocity[:2] - d.tracked.v[:2]) ** 2))))
        while errors and errors[0][0] < now - FOLLOW_WINDOW_S:
            errors.popleft()
        if now - errors[0][0] < FOLLOW_WINDOW_S * 0.9:
            return
        d.follow_error = math.sqrt(sum(e for _, e in errors) / len(errors))
        d.follow_worst = max(d.follow_worst, d.follow_error)

    # --------------------------------------------------------------- reporting
    def all_arrived(self) -> bool:
        flying = [d for d in self.drones.values() if d.phase in (Phase.AIR, Phase.CHECK)]
        return bool(flying) and all(d.arrived for d in flying)

    def all_phase(self, phase: Phase) -> bool:
        return all(d.phase == phase for d in self.drones.values())

    def json(self) -> dict:
        now = self.now
        return {
            "t": round(now, 2),
            "room": self.room.json(),
            "limits": {"separation": self.limits.separation, "surface_clearance": self.limits.surface_clearance,
                       "max_speed": self.limits.max_speed},
            "closest": None if not math.isfinite(self.closest) else round(self.closest, 3),
            "estopped": self.estopped,
            "preflight": self.preflight(now) if self.all_phase(Phase.GROUND) else [],
            "moving": [{"id": m.id, "p": m.p.round(3).tolist(), "radius": m.radius} for m in self.moving],
            "anchors": None if self.tracker is None else {
                "known": [{"id": a.id, "x": a.x, "y": a.y, "size": a.size} for a in self.tracker.anchors.values()],
                "drones": {k: {**v, "expected": self.tracker.expected(k)} for k, v in self.tracker.report(now).items()},
            },
            "drones": [{
                "id": d.id,
                "phase": d.phase.value,
                "p": d.tracked.p.round(3).tolist(),
                "v": d.tracked.v.round(3).tolist(),
                "yaw": round(math.degrees(d.tracked.yaw), 1),
                "pad": list(d.config.pad),
                "pose_age": None if not math.isfinite(d.tracked.age(now)) else round(d.tracked.age(now), 3),
                "sigma": round(d.tracked.sigma, 3),
                "fix_age": round(d.tracked.fix_age, 2) if math.isfinite(d.tracked.fix_age) else None,
                "link": d.link.connected,
                "latency": getattr(d.link, "latency", None),
                "goal": None if d.goal is None else d.goal.round(3).tolist(),
                "path": [point.round(3).tolist() for point in d.path],
                "arrived": d.arrived,
                "limited": d.limited,
                "commanded": d.commanded.round(3).tolist(),
                "speed_per_stick": round(d.mapper.speed_per_stick, 2),
                "follow_error": round(d.follow_error, 3),
                "check": d.check_results,
                "faults": list(d.faults.values()),
            } for d in self.drones.values()],
        }
