"""The fixed test set: every mission, flown on many seeds, scored against the simulator's truth.

A mission passes only if nothing touched anything and it ended as it should. Run it with
`python -m swarm bench`. Every seed also changes how the simulated drones respond to the sticks.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from sim import flight_controller
from sim.scene import Box, Scene
from sim.scenes import SCENES
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.room import Obstacle, Room
from swarm.safety import Limits, Moving
from swarm.sim_fleet import Conditions, SimFleet

TICK = 0.05
ROOM = Box((-4.0, -3.0, 0.0), (4.0, 3.0, 2.6), "wall")


def hall(*boxes: Box, bounds: Box = ROOM) -> Scene:
    return Scene(name="hall", bounds=bounds, boxes=tuple(boxes))


def solid(lo, hi) -> Box:
    return Box(lo, hi, "wood")


class Flight:
    """One mission in progress: the coordinator, the world, and the helpers a script yields."""

    def __init__(self, scene, pads, seed, conditions=None, limits=None, fence=None):
        self.fleet = SimFleet(scene, pads, seed, conditions)
        self.room = Room.from_scene(scene, fence)
        configs = [DroneConfig(k, (v[0], v[1], 0.0), v[3] if len(v) > 3 else 0.0) for k, v in pads.items()]
        self.co = Coordinator(self.room, configs, self.fleet.links, limits)
        self.moving: list = []
        self.notes: list[str] = []
        self.failed: list[str] = []
        self.exempt: set[str] = set()  # drones flown into things on purpose by the mission
        self.velocity_gain_scale = 1.0

    def tick(self) -> None:
        self.fleet.advance(TICK)
        self.fleet.score_people()
        moving = [m(self.fleet.t) for m in self.moving]
        self.co.tick(self.fleet.t, self.fleet.poses(), moving)

    def wait(self, seconds: float):
        yield self.fleet.t + seconds, None

    def until(self, test, timeout: float, what: str):
        yield self.fleet.t + timeout, (test, what)

    def takeoff(self):
        # The tracker needs a moment, and a noisy one can drop out at the instant preflight looks.
        yield from self.until(lambda: not self.co.preflight(self.fleet.t), 5.0, "preflight to pass")
        problems = self.co.takeoff(self.fleet.t)
        if problems:
            self.failed.append(f"preflight: {problems}")
            return
        yield from self.until(lambda: self.co.all_phase(Phase.AIR), 30.0, "takeoff")

    def go(self, goals: dict, timeout: float = 60.0, what: str = "arrive"):
        problems = self.co.goto(goals)
        if problems:
            self.failed.append(f"goto refused: {problems}")
            return
        yield from self.until(self.co.all_arrived, timeout, what)

    def form(self, shape, center, spacing, z, turn=0.0, timeout: float = 60.0):
        problems = self.co.formation(shape, center, spacing, z, turn)
        if problems:
            self.failed.append(f"formation refused: {problems}")
            return
        yield from self.until(self.co.all_arrived, timeout, f"{shape} formation")

    def land(self, timeout: float = 90.0):
        self.co.land(self.fleet.t)
        yield from self.until(lambda: self.co.all_phase(Phase.GROUND), timeout, "landing")

    def run(self, script) -> None:
        for deadline, check in script:
            while self.fleet.t < deadline:
                self.tick()
                if check is not None and check[0]():
                    break
            else:
                if check is not None:
                    self.failed.append(f"timed out waiting for {check[1]}")
                    return


@dataclass
class Outcome:
    name: str
    seed: int
    passed: bool
    collisions: int
    contacts: int
    crashed: int
    min_separation: float
    mission: bool
    duration: float
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- pads and helpers
THREE = {"a": (-2.0, -1.2), "b": (-2.0, 0.0), "c": (-2.0, 1.2)}
TWO = {"a": (-2.0, 0.0), "b": (2.0, 0.0)}
Z = 1.0


def setup(pads=THREE, boxes=(), conditions=None, limits=None, bounds=ROOM, seed=1):
    return Flight(hall(*boxes, bounds=bounds), pads, seed, conditions, limits)


# ----------------------------------------------------------------------------- the missions
def hover(seed):
    f = setup(seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.wait(10.0)
        yield from f.land()
    return f, script()


def swap_two(seed):
    f = setup(TWO, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 0.0, Z), "b": (-2.0, 0.0, Z)})
        yield from f.go({"a": (-2.0, 0.0, Z), "b": (2.0, 0.0, Z)})
        yield from f.land()
    return f, script()


def cross_three(seed):
    pads = {"a": (-2.0, -1.5), "b": (-2.0, 1.5), "c": (0.0, -2.3)}
    f = setup(pads, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 1.5, Z), "b": (2.0, -1.5, Z), "c": (0.0, 2.3, Z)})
        yield from f.go({"a": (-2.0, -1.5, Z), "b": (-2.0, 1.5, Z), "c": (0.0, -2.3, Z)})
        yield from f.land()
    return f, script()


def rotate_ring(seed):
    f = setup(seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.form("ring", (0.0, 0.0), 1.2, Z)
        for turn in (2.1, 4.2, 0.0):
            yield from f.form("ring", (0.0, 0.0), 1.2, Z, turn)
        yield from f.land()
    return f, script()


def formations(seed):
    f = setup(seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.form("line", (0.0, 0.0), 1.0, Z)
        yield from f.form("ring", (1.0, 0.0), 1.0, Z)
        yield from f.form("line", (0.0, 0.0), 1.0, Z, math.pi / 2)
        yield from f.land()
    return f, script()


def pillar(seed):
    f = setup(boxes=[solid((-0.4, -0.8, 0.0), (0.4, 0.8, 2.6))], seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 1.2, Z), "b": (2.0, 0.0, Z), "c": (2.0, -1.2, Z)})
        yield from f.go({"a": (-2.0, -1.2, Z), "b": (-2.0, 0.0, Z), "c": (-2.0, 1.2, Z)})
        yield from f.land()
    return f, script()


def wall_gap(seed):
    """A wall across the room with one door: only one drone fits through at a time."""
    boxes = [solid((-0.1, -3.0, 0.0), (0.1, -0.6, 2.6)), solid((-0.1, 0.6, 0.0), (0.1, 3.0, 2.6))]
    f = setup(boxes=boxes, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, -1.2, Z), "b": (2.0, 0.0, Z), "c": (2.0, 1.2, Z)}, timeout=90.0)
        yield from f.go({"a": (-2.0, -1.2, Z), "b": (-2.0, 0.0, Z), "c": (-2.0, 1.2, Z)}, timeout=90.0)
        yield from f.land()
    return f, script()


def u_trap(seed):
    """A cup facing the drones: flying straight at the goal ends in a dead end."""
    boxes = [solid((0.8, -1.5, 0.0), (1.0, 1.5, 2.6)), solid((-0.6, -1.5, 0.0), (1.0, -1.3, 2.6)),
             solid((-0.6, 1.3, 0.0), (1.0, 1.5, 2.6))]
    pads = {"a": (-1.0, -0.6), "b": (-1.0, 0.6), "c": (-2.5, 0.0)}
    f = setup(pads, boxes=boxes, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.5, -0.8, Z), "b": (2.5, 0.8, Z), "c": (2.8, 2.0, Z)}, timeout=90.0)
        yield from f.wait(1.0)
        f.co.land_here(f.fleet.t)
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND), 30.0, "landing")
    return f, script()


def furniture(seed):
    """The simulator's living room, sofa and table and all."""
    scene = SCENES["room"]()
    pads = {"a": (-2.0, -0.8), "b": (-2.0, 0.4), "c": (-1.0, 1.4)}
    f = Flight(scene, pads, seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.5, -1.0, Z), "b": (2.4, 0.3, Z), "c": (-1.0, -1.2, Z)})
        yield from f.go({"a": (-2.0, 1.2, Z), "b": (0.4, 1.5, Z), "c": (2.5, -0.5, 1.4)})
        yield from f.go({"a": (-2.0, -0.8, Z), "b": (-2.0, 0.4, Z), "c": (-1.0, 1.4, Z)})
        yield from f.land()
    return f, script()


def new_obstacle(seed):
    """Someone puts a screen in the way mid-flight and adds it to the map."""
    f = setup(seed=seed)
    screen = solid((2.0, -2.2, 0.0), (2.2, 2.2, 2.6))
    def script():
        yield from f.takeoff()
        problems = f.co.goto({"a": (3.2, -1.0, Z), "b": (3.2, 0.0, Z), "c": (3.2, 1.0, Z)})
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(3.0)
        f.co.add_obstacle(Obstacle("screen", screen.lo, screen.hi))
        f.fleet.scene = hall(screen)
        for d in f.fleet.drones.values():
            d.sim.drone.scene = f.fleet.scene
        yield from f.until(f.co.all_arrived, 60.0, "arrive around the new obstacle")
        yield from f.land()
    return f, script()


def walking_person(seed):
    """A tracked person walks through the formation at 0.8 m/s, and back."""
    f = setup(seed=seed)
    def person(t):
        phase = (0.8 * max(t - 14.0, 0.0)) % 10.0
        y = -2.8 + phase if phase < 5.0 else 2.2 - (phase - 5.0)
        return Moving("person", np.array([0.0, y, 0.9]), np.array([0.0, 0.8 if phase < 5.0 else -0.8, 0.0]), 0.3)
    def script():
        yield from f.takeoff()
        f.moving.append(person)
        f.fleet.people.append(person)
        yield from f.form("line", (0.0, 0.0), 1.0, Z)
        yield from f.wait(14.0)
        f.moving.clear()
        f.fleet.people.clear()
        yield from f.form("line", (0.0, 0.0), 1.0, Z)
        yield from f.land()
    return f, script()


def reversed_sticks(seed):
    """The roll stick is wired backwards on one drone. The check must catch it before the mission."""
    f = setup(seed=seed)
    send = f.fleet.links["b"].send_sticks
    f.fleet.links["b"].send_sticks = lambda t, y, p, r, ttl: send(t, y, p, -r, ttl)
    def script():
        yield from f.wait(0.5)
        problems = f.co.takeoff(f.fleet.t)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.until(lambda: "response" in f.co.drones["b"].faults, 25.0, "the check to catch it")
        yield from f.until(lambda: f.co.drones["b"].phase == Phase.GROUND, 20.0, "it to land")
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.AIR, 20.0, "the others to pass")
        f.co.land(f.fleet.t)
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND), 40.0, "landing")
    return f, script()


def tracking_dropout(seed):
    """One drone vanishes from the tracker for 1.5 s while crossing: it must coast and resume."""
    f = setup(seed=seed)
    def script():
        yield from f.takeoff()
        problems = f.co.goto({"a": (2.0, 1.2, Z), "b": (2.0, 0.0, Z), "c": (2.0, -1.2, Z)})
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(4.0)
        f.fleet.hidden["b"] = f.fleet.t + 1.5
        yield from f.until(f.co.all_arrived, 60.0, "arrive after the dropout")
        yield from f.land()
    return f, script()


def tracking_lost(seed):
    """One drone is gone for good: it must land itself and the others must carry on."""
    f = setup(seed=seed)
    def script():
        yield from f.takeoff()
        problems = f.co.goto({"a": (2.0, 1.2, Z), "b": (2.0, 0.0, Z), "c": (2.0, -1.2, Z)})
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(4.0)
        f.fleet.hidden["b"] = math.inf
        yield from f.until(lambda: f.co.drones["a"].arrived and f.co.drones["c"].arrived, 60.0, "the others to arrive")
        yield from f.wait(16.0)
        if f.fleet.drones["b"].sim.fc.state != "READY":
            f.failed.append("the lost drone did not land")
        f.co.land(f.fleet.t)
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND), 60.0, "landing")
    return f, script()


def deaf_drone(seed):
    """One drone stops hearing commands while hovering in the way. The others must go around it."""
    pads = {"a": (-2.0, 0.0), "b": (0.0, 0.0), "c": (2.0, 0.0)}
    f = setup(pads, seed=seed)
    def script():
        yield from f.takeoff()
        f.fleet.links["b"].deaf = True
        yield from f.wait(0.5)
        f.co.drones["b"].link.connected = False
        problems = f.co.goto({"a": (2.0, 0.8, Z), "c": (-2.0, -0.8, Z)})
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.until(lambda: f.co.drones["a"].arrived and f.co.drones["c"].arrived, 60.0, "going around it")
        f.co.drones["b"].link.connected = True
        f.fleet.links["b"].deaf = False
        yield from f.wait(1.0)
        yield from f.land()
    return f, script()


def rogue_drone(seed):
    """One drone ignores every command and flies straight at another. Only the target can act."""
    pads = {"a": (-3.0, 0.0), "b": (0.0, 0.0), "c": (0.0, 1.5)}
    f = setup(pads, seed=seed)
    f.exempt = {"a"}
    def script():
        yield from f.takeoff()
        yield from f.wait(2.0)
        f.fleet.drones["a"].override = (0.0, 0.0, 0.12, 0.0)
        f.co.drones["a"].link.connected = False  # the coordinator knows it can't steer this one
        yield from f.wait(7.0)
        f.fleet.drones["a"].override = None
        f.co.drones["a"].link.connected = True
        yield from f.wait(1.0)
        yield from f.land()
    return f, script()


def blank_floor(seed):
    """A floor the flow sensor can't read: sticks set tilt, not speed, and nothing holds position.

    These can't fly a mission. They must find that out right after takeoff, and land.
    """
    scene = Scene(name="blank", bounds=ROOM, flow_quality=0.1)
    f = Flight(scene, THREE, seed)
    def script():
        yield from f.wait(0.5)
        problems = f.co.takeoff(f.fleet.t)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND) and not f.co.takeoff_queue
                           and all("response" in d.faults or "takeoff" in d.faults for d in f.co.drones.values()),
                           40.0, "every drone to fail its check and land")
    return f, script()


def sluggish(seed):
    """Flight controllers that take twice as long to reach the speed they're asked for."""
    pads = {"a": (-2.0, -1.5), "b": (-2.0, 1.5), "c": (0.0, -2.3)}
    f = setup(pads, seed=seed)
    f.velocity_gain_scale = 0.5
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 1.5, Z), "b": (2.0, -1.5, Z), "c": (0.0, 2.3, Z)}, timeout=90.0)
        yield from f.land()
    return f, script()


def drafty(seed):
    """Air moving at 0.5 m/s with gusts, like a fan or an open door."""
    pads = {"a": (-2.0, -1.5), "b": (-2.0, 1.5), "c": (0.0, -2.3)}
    f = setup(pads, conditions=Conditions(draft=0.5), seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 1.5, Z), "b": (2.0, -1.5, Z), "c": (0.0, 2.3, Z)}, timeout=90.0)
        yield from f.land()
    return f, script()


def refused_takeoff(seed):
    """One battery is too flat to take off. The toggle must not be sent twice, and the rest must not follow."""
    f = setup(seed=seed)
    f.fleet.drones["b"].sim.drone.battery.soc = 0.02
    def script():
        yield from f.wait(0.5)
        problems = f.co.takeoff(f.fleet.t)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(12.0)
        if "takeoff" not in f.co.drones["b"].faults:
            f.failed.append("the failed takeoff was not reported")
        if f.co.drones["c"].phase != Phase.GROUND:
            f.failed.append("the next drone took off anyway")
        f.co.land(f.fleet.t)
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND), 40.0, "landing")
    return f, script()


def frozen_coordinator(seed):
    """The coordinator hangs for 3 s mid-crossing. Commands expire and the stock hold takes over."""
    pads = {"a": (-2.0, -1.5), "b": (-2.0, 1.5), "c": (0.0, -2.3)}
    f = setup(pads, seed=seed)
    def script():
        yield from f.takeoff()
        problems = f.co.goto({"a": (2.0, 1.5, Z), "b": (2.0, -1.5, Z), "c": (0.0, 2.3, Z)})
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(3.5)
        f.fleet.advance(3.0)
        yield from f.until(f.co.all_arrived, 60.0, "arrive after the hang")
        yield from f.land()
    return f, script()


def bad_tracking(seed):
    conditions = Conditions(pose_rate=20.0, pose_latency=0.14, pose_noise_xy=0.025, pose_noise_z=0.06, pose_noise_yaw=0.05,
                            pose_miss=0.12, link_delay=(0.03, 0.12), link_loss=0.08)
    pads = {"a": (-2.0, -1.5), "b": (-2.0, 1.5), "c": (0.0, -2.3)}
    f = setup(pads, conditions=conditions, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 1.5, Z), "b": (2.0, -1.5, Z), "c": (0.0, 2.3, Z)}, timeout=90.0)
        yield from f.land()
    return f, script()


def wrong_gains(seed):
    """The sticks respond very differently from what was assumed."""
    conditions = Conditions(speed_scale=(0.5, 1.8))
    f = setup(TWO, conditions=conditions, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.go({"a": (2.0, 0.0, Z), "b": (-2.0, 0.0, Z)}, timeout=90.0)
        yield from f.go({"a": (-2.0, 0.0, Z), "b": (2.0, 0.0, Z)}, timeout=90.0)
        yield from f.land()
    return f, script()


def six_in_a_gym(seed):
    """More drones than the hardware today, to see that it holds together as the room fills."""
    bounds = Box((-5.0, -5.0, 0.0), (5.0, 5.0, 3.0), "wall")
    pads = {name: (-3.0 + 1.2 * i, -3.0) for i, name in enumerate("abcdef")}
    f = setup(pads, bounds=bounds, seed=seed)
    def script():
        yield from f.takeoff()
        yield from f.form("ring", (0.0, 0.0), 1.1, Z)
        yield from f.form("grid", (0.0, 1.0), 1.1, Z)
        yield from f.form("line", (0.0, 3.0), 1.1, Z)
        yield from f.land(timeout=150.0)
    return f, script()


MISSIONS = {m.__name__: m for m in (
    hover, swap_two, cross_three, rotate_ring, formations, pillar, wall_gap, u_trap, furniture, new_obstacle,
    walking_person, tracking_dropout, tracking_lost, deaf_drone, rogue_drone, bad_tracking, wrong_gains, sluggish, drafty,
    blank_floor, reversed_sticks, refused_takeoff, frozen_coordinator, six_in_a_gym,
)}


def fly(name: str, seed: int) -> Outcome:
    flight, script = MISSIONS[name](seed)
    normal_gain = flight_controller._VELOCITY_GAIN
    flight_controller._VELOCITY_GAIN = normal_gain * flight.velocity_gain_scale
    try:
        flight.run(script)
    finally:
        flight_controller._VELOCITY_GAIN = normal_gain
    fleet = flight.fleet
    notes = list(flight.failed)
    for drone_id, link in fleet.links.items():
        toggles = [command for _, command in link.commands if command in ("takeoff", "land")]
        repeats = [a for a, b in zip(toggles, toggles[1:]) if a == b]
        if repeats or toggles.count("takeoff") > 1:
            flight.failed.append(f"{drone_id} sent a toggle twice: {toggles}")
            notes.append(flight.failed[-1])
    if flight.co.infeasible_ticks:
        notes.append(f"{flight.co.infeasible_ticks} infeasible ticks")
    for d in flight.co.drones.values():
        notes += [f"{d.id}: {message}" for message in d.faults.values()]
    crashed = fleet.crashed()
    contacts = (fleet.contacts | fleet.person_hits) - flight.exempt
    if fleet.min_person_gap < math.inf:
        notes.append(f"closest to the person {fleet.min_person_gap:.2f} m")
    ok = not flight.failed
    passed = ok and not fleet.collisions and not contacts and not crashed
    return Outcome(name, seed, passed, len(fleet.collisions), len(contacts), len(crashed),
                   round(fleet.min_separation, 3), ok, round(fleet.t, 1), notes)


def _job(job):
    return fly(*job)


def run(names=None, seeds: int = 10, workers: int | None = None, first_seed: int = 1) -> list[Outcome]:
    jobs = [(name, seed) for name in (names or MISSIONS) for seed in range(first_seed, first_seed + seeds)]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


def report(outcomes: list[Outcome]) -> dict:
    rows = []
    for name in dict.fromkeys(o.name for o in outcomes):
        group = [o for o in outcomes if o.name == name]
        rows.append({
            "mission": name,
            "passed": sum(o.passed for o in group),
            "flights": len(group),
            "collisions": sum(o.collisions for o in group),
            "contacts": sum(o.contacts for o in group),
            "crashes": sum(o.crashed for o in group),
            "closest_m": min(o.min_separation for o in group),
            "typical_closest_m": round(statistics.median(o.min_separation for o in group), 3),
            "longest_s": max(o.duration for o in group),
        })
    return {
        "flights": len(outcomes),
        "passed": sum(o.passed for o in outcomes),
        "collisions": sum(o.collisions for o in outcomes),
        "contacts": sum(o.contacts for o in outcomes),
        "crashes": sum(o.crashed for o in outcomes),
        "missions": rows,
        "failures": [{"mission": o.name, "seed": o.seed, "collisions": o.collisions, "contacts": o.contacts,
                      "crashed": o.crashed, "closest_m": o.min_separation, "notes": o.notes}
                     for o in outcomes if not o.passed],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm bench")
    parser.add_argument("missions", nargs="*", help="which to run; all of them by default")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args(argv)
    started = time.time()
    outcomes = run(args.missions or None, args.seeds, args.workers, args.first_seed)
    result = report(outcomes)
    limits = Limits()
    print(f"{'mission':<20}{'pass':>8}{'hit':>5}{'wall':>6}{'crash':>7}{'closest':>9}{'median':>8}{'longest':>9}")
    for row in result["missions"]:
        print(f"{row['mission']:<20}{row['passed']:>5}/{row['flights']:<2}{row['collisions']:>5}{row['contacts']:>6}"
              f"{row['crashes']:>7}{row['closest_m']:>8.2f}m{row['typical_closest_m']:>7.2f}m{row['longest_s']:>8.0f}s")
    print(f"\n{result['passed']}/{result['flights']} flights passed, {result['collisions']} drone collisions, "
          f"{result['contacts']} obstacle contacts, {result['crashes']} crashes, in {time.time() - started:.0f} s.")
    print(f"Contact between drones is {2 * 0.12:.2f} m. The controller tries to keep {limits.separation:.2f} m.")
    for failure in result["failures"][:30]:
        print(f"  FAIL {failure['mission']} seed {failure['seed']}: {failure['notes'] or failure}")
    if args.json:
        with open(args.json, "w") as out:
            json.dump(result, out, indent=1)
    return 0 if result["passed"] == result["flights"] else 1
