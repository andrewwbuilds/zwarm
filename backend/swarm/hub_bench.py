"""The fixed test set for the shared memory: avoiding what only cameras can see, taking commands, searching.

Run it with `python -m swarm.hub_bench`. Every flight uses the simulated flight controller and the
simulated camera in swarm.eyes, so these are simulator results, not drone results. A flight passes
only if nothing touched anything and the mission's own checks hold.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import random
import statistics
import time

import numpy as np

from sim.scene import Box
from swarm import bench
from swarm.bench import ROOM, Flight, hall, solid
from swarm.coordinator import Phase
from swarm.eyes import Item, SimEyes, Walker
from swarm.hub import Hub
from swarm.memory import Memory
from swarm.safety import Moving

THREE = {"a": (-3.0, -1.5), "b": (-3.0, 0.0), "c": (-3.0, 1.5)}
TWO = {"a": (-3.0, -1.0), "b": (-3.0, 1.0)}
Z = 1.0
FOUND_WITHIN = 0.5  # metres from the truth for a search to count as right
SEARCH_LIMIT = 300.0


class HubFlight(Flight):
    """A flight where the hub is in charge and the map starts without the hidden things."""

    def __init__(self, pads, seed, known=(), hidden=(), items=(), walkers=(), memory=None, false_rate=0.005):
        hidden_items = [Item(label, box.lo, box.hi) for label, box in hidden]
        super().__init__(hall(*known, *(box for _, box in hidden)), pads, seed)
        for i, _ in enumerate(hidden, start=len(known)):
            self.room.remove(f"wood-{i}")
        self.eyes = SimEyes(self.fleet, [*hidden_items, *items], walkers, seed, false_rate)
        self.memory = memory or Memory()
        self.hub = Hub(self.co, self.memory, self.eyes)

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.fleet.score_people()
        self.hub.tick(self.fleet.t, self.fleet.poses(), [m(self.fleet.t) for m in self.moving])

    def say(self, text: str) -> str:
        return self.hub.command(text)

    def lift(self):
        yield from self.until(lambda: not self.co.preflight(self.fleet.t), 5.0, "preflight to pass")
        reply = self.say("take off")
        if not reply.startswith("Taking off"):
            self.failed.append(reply)
            return
        yield from self.until(lambda: self.co.all_phase(Phase.AIR), 40.0, "takeoff")

    def down(self):
        self.say("land")
        yield from self.until(lambda: self.co.all_phase(Phase.GROUND), 120.0, "landing")

    def check(self, passed: bool, problem: str) -> None:
        if not passed:
            self.failed.append(problem)

    def search_over(self):
        return self.hub.search is not None and self.hub.search.state in ("found", "exhausted", "failed")

    def seek(self, label: str):
        self.say(f"find the {label}")
        yield from self.until(self.search_over, SEARCH_LIMIT, f"the search for {label} to finish")

    def check_found(self, item: Item) -> None:
        search = self.hub.search
        self.check(search.state == "found", f"search ended {search.state}: {self.hub.reports[-1:]}")
        if search.found_at is not None:
            error = float(np.linalg.norm(search.found_at[:2] - item.center[:2]))
            self.notes.append(f"found {error:.2f} m from the truth at {self.fleet.t:.0f} s")
            self.check(error <= FOUND_WITHIN, f"said found {error:.2f} m from the truth")


def on_floor(label: str, x: float, y: float, size: float = 0.3, height: float = 0.4, **kwargs) -> Item:
    return Item(label, (x - size / 2, y - size / 2, 0.0), (x + size / 2, y + size / 2, height), **kwargs)


def random_spot(seed: int, keep_clear=()) -> tuple[float, float]:
    """A place on the floor that depends only on the seed: the test set is fixed."""
    rng = random.Random(seed * 31 + 5)
    while True:
        x, y = rng.uniform(-3.6, 3.6), rng.uniform(-2.6, 2.6)
        if x > -2.0 and all(lo[0] - 0.4 > x or x > hi[0] + 0.4 or lo[1] - 0.4 > y or y > hi[1] + 0.4 for lo, hi in keep_clear):
            return x, y


FURNITURE = [("couch", solid((0.0, -2.6, 0.0), (0.9, -1.0, 0.9))), ("bed", solid((1.2, 0.6, 0.0), (2.6, 2.4, 0.7))),
             ("dining table", solid((-1.2, 0.3, 0.0), (-0.3, 1.5, 1.4)))]


# ----------------------------------------------------------------------------- avoiding things
def unseen_boxes(seed):
    """Tall boxes nobody told the map about stand across every drone's straight line."""
    boxes = [("chair", solid((-0.2, -1.9, 0.0), (0.4, -1.1, 1.8))), ("couch", solid((-0.2, -0.4, 0.0), (0.4, 0.4, 1.8))),
             ("potted plant", solid((-0.2, 1.1, 0.0), (0.4, 1.9, 1.8)))]
    f = HubFlight(THREE, seed, hidden=boxes)
    def script():
        yield from f.lift()
        for name, y in (("a", -1.5), ("b", 0.0), ("c", 1.5)):
            f.say(f"drone {name} go to 3 {y}")
        yield from f.until(lambda: all(np.linalg.norm(d.tracked.p[:2] - np.array([3.0, y])) < 0.3
                                       for d, y in zip(f.co.drones.values(), (-1.5, 0.0, 1.5))), 120.0, "arriving past the boxes")
        for label, box in boxes:
            center = (np.array(box.lo[:2]) + np.array(box.hi[:2])) / 2
            f.check(any(t.label == label and np.linalg.norm(t.p[:2] - center) < 0.6 for t in f.memory.obstacles()),
                    f"the {label} isn't on the map where it really is")
        yield from f.down()
    return f, script()


def told_by_another(seed):
    """Drone b's camera is dead and a box is in its way. Only what a saw can save it."""
    box = ("chair", solid((-0.5, -0.6, 0.0), (0.5, 0.6, 1.8)))
    f = HubFlight({"a": (-3.0, -1.0), "b": (-3.0, 1.0)}, seed, hidden=[box])
    f.eyes.blind.add("b")
    def script():
        yield from f.lift()
        f.say("drone a go to -1.5 0")
        yield from f.until(lambda: len(f.memory.obstacles()) == 1, 60.0, "a to see the box")
        f.say("drone a go to -2.5 -2")
        yield from f.until(lambda: "a" not in f.hub.orders or f.hub.orders["a"].stage == "done", 40.0, "a to clear out")
        f.say("drone b go to -2 0")
        yield from f.until(lambda: f.hub.orders.get("b") is None or f.hub.orders["b"].stage == "done", 40.0, "b to line up")
        f.say("drone b go to 3 0")
        yield from f.until(lambda: f.hub.orders.get("b") is None or f.hub.orders["b"].stage == "done", 90.0, "b to go past")
        f.check(np.linalg.norm(f.co.drones["b"].tracked.p[:2] - np.array([3.0, 0.0])) < 0.4, "b didn't get there")
        yield from f.down()
    return f, script()


def _crossing_person(t):
    phase = (0.6 * max(t - 18.0, 0.0)) % 9.0
    y = -2.6 + phase if phase < 4.5 else 1.9 - (phase - 4.5)
    return Moving("person", np.array([1.0, y, 0.9]), np.array([0.0, 0.6 if phase < 4.5 else -0.6, 0.0]), 0.25)


def unseen_person(seed):
    """Someone walks across in front of the drones, known only from their cameras."""
    walker = Walker("person", _crossing_person)
    f = HubFlight(THREE, seed, walkers=[walker])
    f.fleet.people.append(_crossing_person)
    def script():
        yield from f.lift()
        for name, y in (("a", -1.5), ("b", 0.0), ("c", 1.5)):
            f.say(f"drone {name} go to 3.2 {y}")
        yield from f.wait(40.0)
        f.check(any(t.label == "person" for t in f.memory.known()), "nobody saw the person")
        yield from f.down()
    return f, script()


def false_alarms(seed):
    """A noisy detector, plus a box in the middle of the route that isn't real. It must not stop the flight."""
    ghost = on_floor("chair", 0.0, 0.0, size=0.8, height=1.6, until=8.0)
    f = HubFlight(TWO, seed, items=[ghost], false_rate=0.15)
    def script():
        yield from f.lift()
        f.say("drone a go to 3 0.8")
        f.say("drone b go to 3 -0.8")
        yield from f.until(lambda: all(np.linalg.norm(d.tracked.p[0] - 3.0) < 0.3 for d in f.co.drones.values()), 90.0, "arriving")
        f.check(not any(t.label == "chair" and t.sightings > 0 and np.linalg.norm(t.p[:2]) < 1.0 for t in f.memory.obstacles()),
                "the box that wasn't real stayed on the map")
        yield from f.down()
    return f, script()


# ----------------------------------------------------------------------------- searching
def search_open(seed):
    """Something on the floor of an empty room, somewhere new on every seed."""
    target = on_floor("backpack", *random_spot(seed))
    f = HubFlight(THREE, seed, items=[target])
    def script():
        yield from f.lift()
        yield from f.seek("backpack")
        f.check_found(target)
        yield from f.down()
    return f, script()


def search_furniture(seed):
    """The same, among furniture no one mapped, which hides floor from the cameras."""
    spots = [(box.lo, box.hi) for _, box in FURNITURE]
    target = on_floor("backpack", *random_spot(seed, spots))
    f = HubFlight(THREE, seed, hidden=FURNITURE, items=[target])
    def script():
        yield from f.lift()
        yield from f.seek("backpack")
        f.check_found(target)
        yield from f.down()
    return f, script()


def search_absent(seed):
    """It isn't there. The drones must say so, not imagine it, even with a noisy detector."""
    f = HubFlight(THREE, seed, hidden=FURNITURE, false_rate=0.05)
    def script():
        yield from f.lift()
        yield from f.seek("backpack")
        search = f.hub.search
        f.check(search.state == "exhausted", f"search ended {search.state}: {f.hub.reports[-1:]}")
        share = search.coverage.share_seen()
        f.notes.append(f"saw {share * 100:.0f}% of the floor at {f.fleet.t:.0f} s")
        f.check(share >= 0.95, f"gave up with {share * 100:.0f}% seen")
        yield from f.down()
    return f, script()


def search_decoy(seed):
    """Something that only looks a bit like the target is nearer than the real one."""
    decoy = on_floor("backpack", -0.5, -1.5, confidence=0.5)
    target = on_floor("backpack", 2.8, 1.8)
    f = HubFlight(THREE, seed, items=[decoy, target])
    def script():
        yield from f.lift()
        yield from f.seek("backpack")
        f.check_found(target)
        yield from f.down()
    return f, script()


def search_stale(seed):
    """Memory says it's in one place from an earlier flight, but it has been moved."""
    memory = Memory()
    for i in range(6):
        memory.record("backpack", (-1.0, -1.5, 0.2), (0.3, 0.3, 0.4), 0.85, 1.5, "a", -100.0 + i)
    x, y = random_spot(seed)
    target = on_floor("backpack", 1.6 + abs(x) * 0.5, y)  # too far for the first look, so the old spot is checked first
    f = HubFlight(THREE, seed, items=[target], memory=memory)
    def script():
        yield from f.lift()
        yield from f.seek("backpack")
        f.check_found(target)
        old = [t for t in memory.find("backpack") if np.linalg.norm(t.p[:2] - np.array([-1.0, -1.5])) < 0.6]
        f.check(not old, "the old sighting wasn't corrected")
        yield from f.down()
    return f, script()


def search_lost_drone(seed):
    """One drone drops out of tracking part way through. It lands itself and the others finish the job."""
    target = on_floor("backpack", *random_spot(seed))
    f = HubFlight(THREE, seed, items=[target])
    def script():
        yield from f.lift()
        f.say("find the backpack")
        yield from f.wait(6.0)
        f.fleet.hidden["b"] = math.inf
        yield from f.until(f.search_over, SEARCH_LIMIT, "the search to finish")
        f.check_found(target)
        f.say("land")
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.GROUND and f.co.drones["c"].phase == Phase.GROUND, 120.0, "landing")
    return f, script()


# ----------------------------------------------------------------------------- commands
def spoken_mission(seed):
    """A whole flight run only by what the operator says, checking the answers."""
    target = on_floor("backpack", 2.5, -1.5)
    box = ("couch", solid((0.0, 0.8, 0.0), (1.0, 2.0, 1.0)))
    f = HubFlight(THREE, seed, hidden=[box], items=[target])
    def script():
        f.check("didn't understand" in f.say("do a backflip"), "an unknown phrase got an action")
        f.check(f.fleet.links["a"].commands == [], "an unknown phrase sent a command")
        f.check("Nothing is flying" in f.say("find the backpack"), "searching on the ground should be refused")
        f.check("can't recognise" in f.say("find the unicorn"), "an unknown object should be refused")
        yield from f.lift()
        yield from f.seek("bag")
        f.check_found(target)
        answer = f.say("where is the backpack")
        f.check("(2." in answer, f"wrong answer to where: {answer}")
        f.check("backpack" in f.say("what have you seen"), "the list left out the backpack")
        f.check("whole fleet" in f.say("drone a land"), "naming a drone must not land them all")
        f.check(all(d.phase == Phase.AIR for d in f.co.drones.values()), "a refused command still acted")
        yield from f.wait(1.0)
        f.say("line up")
        yield from f.until(f.co.all_arrived, 90.0, "the line")
        yield from f.down()
    return f, script()


def stop_mid_search(seed):
    """Saying stop must hold everything at once and end the search."""
    target = on_floor("backpack", 3.2, 2.2)
    f = HubFlight(THREE, seed, items=[target])
    def script():
        yield from f.lift()
        f.say("find the backpack")
        yield from f.wait(9.0)
        f.say("stop")
        f.check(f.hub.search.state == "stopped", "the search carried on")
        yield from f.wait(4.0)
        speeds = [float(np.linalg.norm(d.tracked.v[:2])) for d in f.co.drones.values()]
        f.check(max(speeds) < 0.2, f"still moving after stop: {max(speeds):.2f} m/s")
        before = [d.tracked.p.copy() for d in f.co.drones.values()]
        yield from f.wait(5.0)
        moved = max(float(np.linalg.norm(d.tracked.p[:2] - b[:2])) for d, b in zip(f.co.drones.values(), before))
        f.check(moved < 0.4, f"drifted {moved:.2f} m after stop")
        yield from f.down()
    return f, script()


MISSIONS = {m.__name__: m for m in (
    unseen_boxes, told_by_another, unseen_person, false_alarms, search_open, search_furniture, search_absent, search_decoy,
    search_stale, search_lost_drone, spoken_mission, stop_mid_search,
)}
SEARCHES = {"search_open", "search_furniture", "search_absent", "search_decoy", "search_stale", "search_lost_drone", "spoken_mission"}


def fly(name: str, seed: int) -> bench.Outcome:
    """Scored by the same rules as every other mission on the swarm bench."""
    bench.MISSIONS[name] = MISSIONS[name]
    try:
        return bench.fly(name, seed)
    finally:
        del bench.MISSIONS[name]


def _job(job):
    logging.disable(logging.WARNING)  # a drone lost on purpose reports itself on every tick
    return fly(*job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm.hub_bench")
    parser.add_argument("missions", nargs="*", help="which to run; all of them by default")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args(argv)
    started = time.time()
    names = args.missions or list(MISSIONS)
    jobs = [(name, seed) for name in names for seed in range(args.first_seed, args.first_seed + args.seeds)]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        outcomes = pool.map(_job, jobs, chunksize=1)
    result = bench.report(outcomes)
    print(f"{'mission':<20}{'pass':>8}{'hit':>5}{'wall':>6}{'crash':>7}{'closest':>9}{'median s':>10}{'longest s':>11}")
    for row in result["missions"]:
        times = [o.duration for o in outcomes if o.name == row["mission"]]
        print(f"{row['mission']:<20}{row['passed']:>5}/{row['flights']:<2}{row['collisions']:>5}{row['contacts']:>6}"
              f"{row['crashes']:>7}{row['closest_m']:>8.2f}m{statistics.median(times):>10.0f}{row['longest_s']:>11.0f}")
    print(f"\n{result['passed']}/{result['flights']} flights passed, {result['collisions']} drone collisions, "
          f"{result['contacts']} obstacle contacts, {result['crashes']} crashes, in {time.time() - started:.0f} s.")
    for failure in result["failures"][:40]:
        print(f"  FAIL {failure['mission']} seed {failure['seed']}: {failure['notes'] or failure}")
    if args.json:
        with open(args.json, "w") as out:
            json.dump(result, out, indent=1)
    return 0 if result["passed"] == result["flights"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
