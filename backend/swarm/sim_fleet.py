"""Several simulated S20s in one room, with a tracker and links that misbehave on purpose.

Each drone is a full sim.Simulator fed the real control packets, built by the same model and
protocol adapter the backend uses. This adds what one simulator can't see: other drones, the air
they push down, and how close anything actually came. The controller never reads this truth.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from control.strategies import LinearStrategy
from models.s2x_rc import S2xDroneModel
from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from sim.config import DroneSpec, FlightTuning
from sim.simulator import Simulator
from swarm.control import Pose

STEP = 1 / 400
PACKET_PERIOD = 1 / 80
CLIENT = ("172.16.10.2", 50000)
AXES = ("throttle", "yaw", "pitch", "roll")


@dataclass(frozen=True)
class Conditions:
    """How badly the world behaves. The defaults are hopes for a decent real setup, not measurements."""

    pose_rate: float = 30.0
    pose_latency: float = 0.07  # camera frame to pose available
    pose_noise_xy: float = 0.01
    pose_noise_z: float = 0.03
    pose_noise_yaw: float = 0.02
    pose_miss: float = 0.02  # chance a frame's marker isn't found
    link_delay: tuple[float, float] = (0.02, 0.06)  # coordinator to flight controller
    link_loss: float = 0.01
    speed_scale: tuple[float, float] = (0.8, 1.25)  # how wrong the assumed stick response may be
    deadband: tuple[float, float] = (0.03, 0.055)
    hover_height: tuple[float, float] = (1.0, 1.25)
    draft: float = 0.1  # m/s of indoor air movement


class _Air:
    """The simulator's wind plus the downwash from drones overhead."""

    def __init__(self, wind):
        self.wind = wind
        self.extra = (0.0, 0.0, 0.0)

    def sample(self, dt: float):
        x, y, z = self.wind.sample(dt)
        return x + self.extra[0], y + self.extra[1], z + self.extra[2]


class SimLink:
    def __init__(self, fleet: "SimFleet", drone_id: str):
        self.fleet, self.id = fleet, drone_id
        self.connected = True
        self.deaf = False  # commands vanish though the link looks fine, like a stalled radio
        self.commands: list[tuple[float, str]] = []
        self.sent = 0

    def send_sticks(self, throttle, yaw, pitch, roll, ttl) -> None:
        self.sent += 1
        if self.connected and not self.deaf:
            self.fleet.deliver(self.id, ("axes", (throttle, yaw, pitch, roll), ttl))

    def command(self, name: str) -> bool:
        self.commands.append((self.fleet.t, name))
        if not self.connected:
            return False
        if not self.deaf:
            self.fleet.deliver(self.id, ("command", name, None), reliable=True)
        return True


class SimDrone:
    def __init__(self, drone_id, sim, model, rc):
        self.id, self.sim, self.model, self.rc = drone_id, sim, model, rc
        self.axes = (0.0, 0.0, 0.0, 0.0)
        self.deadline = None
        self.next_packet = 0.0
        self.override = None  # sticks forced by a fault, ignoring every command
        self.air = _Air(sim.wind)
        sim.wind = self.air


class SimFleet:
    def __init__(self, scene, pads: dict, seed: int = 1, conditions: Conditions | None = None):
        self.scene = scene
        self.conditions = conditions or Conditions()
        self.rng = random.Random(seed)
        self.t = 0.0
        self.drones: dict[str, SimDrone] = {}
        self.links: dict[str, SimLink] = {}
        self.inbox: list[tuple[float, int, str, tuple]] = []
        self.order = 0
        self.pending: deque = deque()
        self.next_frame = 0.0
        self.hidden: dict[str, float] = {}  # drone: hidden from the tracker until this time
        self.moving_paths: dict[str, object] = {}
        self.min_separation = math.inf
        self.min_clearance = math.inf
        self.collisions: set[tuple[str, str]] = set()
        self.contacts: set[str] = set()
        self.people: list = []  # tracked obstacles to score against: functions of time
        self.person_hits: set[str] = set()
        self.min_person_gap = math.inf
        c = self.conditions
        base = FlightTuning.calibrated()
        for i, (drone_id, pad) in enumerate(pads.items()):
            tuning = replace(
                base,
                max_speed_mps=base.max_speed_mps * self.rng.uniform(*c.speed_scale),
                stick_deadband=self.rng.uniform(*c.deadband),
                takeoff_height_m=self.rng.uniform(*c.hover_height),
            )
            sim = Simulator(scene=scene, seed=seed * 101 + i, tuning=tuning, spec=DroneSpec.calibrated())
            sim.place(pad[0], pad[1], pad[3] if len(pad) > 3 else 0.0)
            if c.draft:
                sim.wind.set(c.draft, self.rng.uniform(0, 360), c.draft)
            model = S2xDroneModel()
            model.set_strategy(LinearStrategy())
            rc = S2xRCProtocolAdapter("127.0.0.1")
            rc.sock.close()
            self.drones[drone_id] = SimDrone(drone_id, sim, model, rc)
            self.links[drone_id] = SimLink(self, drone_id)
        for _ in range(int(2.5 / STEP)):  # gyro calibration after power-on
            self._step_physics()
        self.t = 0.0
        for d in self.drones.values():
            d.next_packet = 0.0

    # ------------------------------------------------------------ what the coordinator touches
    def deliver(self, drone_id: str, message: tuple, reliable: bool = False) -> None:
        c = self.conditions
        if not reliable and self.rng.random() < c.link_loss:
            return
        self.order += 1
        self.inbox.append((self.t + self.rng.uniform(*c.link_delay), self.order, drone_id, message))

    def _capture(self) -> None:
        """A camera frame: it happens as the world runs, whether or not anyone is reading poses."""
        c = self.conditions
        self.next_frame += 1 / c.pose_rate
        for d in self.drones.values():
            if self.t < self.hidden.get(d.id, 0.0) or self.rng.random() < c.pose_miss:
                continue
            q = d.sim.drone
            self.pending.append((self.t + c.pose_latency, d.id, Pose(
                self.t,
                (q.x + self.rng.gauss(0, c.pose_noise_xy), q.y + self.rng.gauss(0, c.pose_noise_xy),
                 q.z - d.sim.spec.gear_height_m + self.rng.gauss(0, c.pose_noise_z)),
                q.yaw() + self.rng.gauss(0, c.pose_noise_yaw))))
        while len(self.pending) > 400:
            self.pending.popleft()

    def poses(self) -> dict[str, Pose]:
        """The newest pose from each drone that has come out of the tracker's pipeline by now."""
        out = {}
        while self.pending and self.pending[0][0] <= self.t:
            _, drone_id, pose = self.pending.popleft()
            out[drone_id] = pose
        return out

    # ------------------------------------------------------------ the world
    def advance(self, seconds: float) -> None:
        end = self.t + seconds
        while self.t < end:
            if self.t >= self.next_frame:
                self._capture()
            self._step_physics()
            self.t += STEP

    def _step_physics(self) -> None:
        for item in sorted(self.inbox, key=lambda e: e[:2]) if self.inbox else ():
            if item[0] > self.t:
                break
            self.inbox.remove(item)
            self._apply(item[2], item[3])
        items = list(self.drones.values())
        for d in items:
            d.air.extra = self._downwash(d, items)
            if self.t >= d.next_packet:
                d.next_packet += PACKET_PERIOD
                self._send_packet(d)
            d.sim.step(STEP)
        self._score(items)

    def _apply(self, drone_id: str, message: tuple) -> None:
        d = self.drones[drone_id]
        kind, value, ttl = message
        if kind == "axes":
            d.axes = value
            d.deadline = self.t + ttl
        elif value == "takeoff":
            d.model.takeoff()
        elif value == "land":
            d.model.land()
        elif value == "estop":
            d.model.emergency_stop()

    def _send_packet(self, d: SimDrone) -> None:
        if d.deadline is not None and self.t >= d.deadline:
            d.axes = (0.0, 0.0, 0.0, 0.0)
            d.deadline = None
        axes = d.override if d.override is not None else d.axes
        d.model.update(PACKET_PERIOD, dict(zip(AXES, axes)))
        d.sim.receive_udp(CLIENT, d.rc.build_control_packet(d.model))

    @staticmethod
    def _downwash(d: SimDrone, items: list[SimDrone]) -> tuple[float, float, float]:
        down = 0.0
        q = d.sim.drone
        for other in items:
            if other is d or other.sim.drone.on_ground:
                continue
            r = other.sim.drone
            below = r.z - q.z
            reach = math.hypot(r.x - q.x, r.y - q.y)
            if 0.0 < below < 2.0 and reach < 0.3:  # estimate: a narrow column fading over 2 m
                down += 1.5 * (1 - reach / 0.3) * (1 - below / 2.0)
        return (0.0, 0.0, -down)

    def _score(self, items: list[SimDrone]) -> None:
        for i, a in enumerate(items):
            qa = a.sim.drone
            if not qa.on_ground and a.sim.fc.state not in ("READY", "SPINUP"):
                radius = a.sim.spec.collision_radius_m
                if self.scene.collide(qa.x, qa.y, qa.z, radius) is not None:
                    self.contacts.add(a.id)
            for b in items[i + 1:]:
                qb = b.sim.drone
                if qa.on_ground and qb.on_ground:
                    continue
                gap = math.sqrt((qa.x - qb.x) ** 2 + (qa.y - qb.y) ** 2 + (qa.z - qb.z) ** 2)
                if gap < self.min_separation:
                    self.min_separation = gap
                if gap < a.sim.spec.collision_radius_m + b.sim.spec.collision_radius_m:
                    self.collisions.add((a.id, b.id))

    def score_people(self) -> None:
        for person in self.people:
            thing = person(self.t)
            for d in self.drones.values():
                q = d.sim.drone
                gap = math.hypot(q.x - thing.p[0], q.y - thing.p[1]) - thing.radius
                self.min_person_gap = min(self.min_person_gap, gap)
                if gap < d.sim.spec.collision_radius_m:
                    self.person_hits.add(d.id)

    def truth(self, drone_id: str) -> np.ndarray:
        d = self.drones[drone_id]
        q = d.sim.drone
        return np.array([q.x, q.y, q.z - d.sim.spec.gear_height_m])

    def crashed(self) -> list[str]:
        return [d.id for d in self.drones.values() if d.sim.fc.state == "CRASHED"]
