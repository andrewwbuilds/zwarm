"""The fixed test set for missions: covering a region, losing a drone, changing focus, inspecting, the flight limit.

Run it with `python -m swarm.mission_bench`. Every flight uses the simulated flight controller and the
simulated camera, so these are simulator results, not drone results. A flight passes only if nothing
touched anything and the mission's own checks hold. The floor_ missions get positions only from the
real tag detector on rendered frames, which is slower and closer to the real thing. Run them with --floor.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import re
import statistics
import time

import numpy as np

from swarm import bench, floor_bench
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.eyes import EDGE_MARGIN, RELIABLE_REACH, SimEyes, in_view
from swarm.hub import Hub
from swarm.hub_bench import FURNITURE, THREE, HubFlight, on_floor
from swarm.memory import Memory
from swarm.mission import AT_EDGE_M, MissionError, MissionManager
from swarm.sim_anchors import SimVideo
from swarm.sim_cameras import demo_world
from swarm.tracking import DroneOptics, InsideOutTracker

MISSION_LIMIT = 260.0  # simulated seconds for a mission to finish, inside the 270 s flight limit
COVERED = 0.95  # the share of a region's visible floor that counts as covered
DEMO_VIDEO_HZ = 5.0  # what `python -m zwarm sim` renders at. A real S20 sends more, which only helps


class FakeCapture:
    """Stands in for mapping.fleet.FleetCapture and records what it was asked to do."""

    def __init__(self):
        self.calls: list[tuple] = []

    def start(self, name):
        self.calls.append(("start", name))

    def stop(self):
        self.calls.append(("stop",))

    def build(self, name=None):
        self.calls.append(("build", name))

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


class FakeInspector:
    """Stands in for swarm.inspect.Inspector and notes which drones it was shown, and when."""

    def __init__(self):
        self.started: list = []
        self.watching = False
        self.shown: list[tuple[float, str]] = []

    def start(self, target=None):
        self.started.append(target)
        self.watching = True

    def stop(self):
        self.watching = False

    def status(self):
        return {"queued": 0, "busy": 0, "checked": len(self.shown), "findings": 0}

    def tick(self, now, poses=None):
        self.shown += [(now, drone_id) for drone_id in poses or ()]


class Scored:
    """What a mission flight adds, whichever way the drones get their positions."""

    def add_missions(self, limit=270.0, capture=None, north=0.0) -> None:
        self.seen_events: list[tuple] = []
        self.capture = capture
        self.inspector = FakeInspector()
        self.missions = MissionManager(self.hub, capture=capture, flight_limit_s=limit, north_yaw_deg=north, inspector=self.inspector,
                                       on_event=lambda kind, text, data: self.seen_events.append((kind, text, data)))
        self.numbers: dict[str, float] = {}

    def over(self) -> bool:
        return not self.missions.active

    def happened(self, event: str) -> list[dict]:
        return [e["data"] for e in self.missions.events if e["data"].get("event") == event]

    def inside(self, region: str, drones=None) -> bool:
        flying = [d for d in self.co.drones.values() if d.phase == Phase.AIR and (drones is None or d.id in drones)]
        coverage = self.missions.coverage
        return bool(flying) and all(self.missions.regions.mask(region, d.tracked.p)[0]
                                    or np.linalg.norm(coverage.candidates[:, :2] - d.tracked.p[:2], axis=1).min() < AT_EDGE_M for d in flying)

    def run_mission(self, kind, region="all", target=None, what=None):
        started = self.fleet.t
        try:
            self.missions.start(kind, region, target)
        except MissionError as error:
            self.failed.append(f"refused: {error}")
            return
        yield from self.until(self.over, MISSION_LIMIT, what or f"the {kind} mission to finish")
        self.numbers["mission_s"] = round(self.fleet.t - started, 1)

    def check_covered(self, least: float = COVERED) -> None:
        share = self.missions.status()["share_seen"]
        self.numbers["share"] = share
        self.check(share >= least, f"only {share * 100:.0f}% covered: {self.missions.reason}")


class MissionFlight(Scored, HubFlight):
    """Positions come straight from the simulator, a little noisy and late. Fast, so most of the test set uses it."""

    def __init__(self, pads, seed, hidden=(), items=(), limit=270.0, capture=None, north=0.0, false_rate=0.005):
        HubFlight.__init__(self, pads, seed, hidden=hidden, items=items, false_rate=false_rate)
        self.add_missions(limit, capture, north)

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.fleet.score_people()
        self.hub.tick(self.fleet.t, self.fleet.poses(), [m(self.fleet.t) for m in self.moving])
        self.missions.tick(self.fleet.t)


class DemoFlight(Scored, bench.Flight):
    """The demo room: furniture on the map, more that isn't, and staged damage. Nothing is rendered here."""

    def __init__(self, seed, capture=None):
        self.world = demo_world()
        bench.Flight.__init__(self, self.world.scene, self.world.pads, seed, fence=self.world.fence)
        self.room = self.world.room()
        configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in self.world.pads.items()]
        self.co = Coordinator(self.room, configs, self.fleet.links)
        self.memory = Memory()
        self.hub = Hub(self.co, self.memory, SimEyes(self.fleet, self.world.items, (), seed))
        self.add_missions(capture=capture)

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.hub.tick(self.fleet.t, self.fleet.poses())
        self.missions.tick(self.fleet.t)

    def check(self, passed: bool, problem: str) -> None:
        if not passed:
            self.failed.append(problem)


class DemoFloorFlight(Scored, bench.Flight):
    """The demo room on floor tags, as on a real flight: its furniture hides tags and gets flown near. Slow, since frames are rendered."""

    def __init__(self, seed, capture=None, limit=270.0):
        self.world = demo_world()
        bench.Flight.__init__(self, self.world.scene, self.world.pads, seed, fence=self.world.fence)
        self.room = self.world.room()
        configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in self.world.pads.items()]
        optics = {c.id: DroneOptics(c.pad[0], c.pad[1], c.yaw) for c in configs}
        self.tracker = InsideOutTracker(optics, self.world.anchors, self.world.fence)
        self.video = SimVideo(self.fleet, self.tracker, self.world.scene, seed=seed, rate=DEMO_VIDEO_HZ)
        self.co = Coordinator(self.room, configs, self.fleet.links, tracker=self.tracker)
        self.memory = Memory()
        self.hub = Hub(self.co, self.memory, SimEyes(self.fleet, self.world.items, (), seed))
        self.add_missions(limit, capture)

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.video.step()
        self.hub.tick(self.fleet.t, self.tracker.poses(self.fleet.t))
        self.missions.tick(self.fleet.t)

    def check(self, passed: bool, problem: str) -> None:
        if not passed:
            self.failed.append(problem)


class FloorMissionFlight(Scored, floor_bench.FloorFlight):
    """Positions come only from the real tag detector on rendered frames, as on a real flight. Much slower."""

    def __init__(self, seed, boxes=(), items=(), limit=270.0, capture=None):
        floor_bench.FloorFlight.__init__(self, seed, boxes=boxes)
        self.eyes = SimEyes(self.fleet, items, (), seed)
        self.memory = Memory()
        self.hub = Hub(self.co, self.memory, self.eyes)
        self.add_missions(limit, capture)

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.video.step()
        self.hub.tick(self.fleet.t, self.tracker.poses(self.fleet.t))
        self.missions.tick(self.fleet.t)
        for key, d in self.co.drones.items():
            if d.phase == Phase.AIR:
                self.errors.append((float(np.linalg.norm(d.tracked.p[:2] - self.fleet.truth(key)[:2])), d.tracked.sigma))

    def check(self, passed: bool, problem: str) -> None:
        if not passed:
            self.failed.append(problem)


def explore_region(seed):
    """Three drones cover the north half of an empty room, starting from the south."""
    f = MissionFlight(THREE, seed)
    def script():
        yield from f.lift()
        yield from f.run_mission("explore", "north")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered()
        yield from f.down()
    return f, script()


def explore_all_furniture(seed):
    """The whole room, with furniture nobody mapped, hiding floor from the cameras."""
    f = MissionFlight(THREE, seed, hidden=FURNITURE)
    def script():
        yield from f.lift()
        yield from f.run_mission("explore", "all")
        f.check(f.missions.state in ("done", "failed"), f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered(0.85)
        yield from f.down()
    return f, script()


def lost_drone(seed):
    """One drone drops out of tracking part way through. It lands itself and the other two finish the job."""
    f = MissionFlight(THREE, seed)
    def script():
        yield from f.lift()
        started = f.fleet.t
        f.missions.start("explore", "north")
        yield from f.wait(12.0)
        at_loss = f.missions.status()["share_seen"]
        f.numbers["share_at_loss"] = at_loss
        f.fleet.hidden["b"] = math.inf
        yield from f.until(f.over, MISSION_LIMIT, "the mission to finish without b")
        f.numbers["mission_s"] = round(f.fleet.t - started, 1)
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered()
        lost = f.happened("drone_lost")
        f.check(any(e["drone"] == "b" for e in lost), f"nobody reported b lost: {lost}")
        f.check(f.co.drones["b"].phase != Phase.AIR or f.co.drones["b"].landing_requested, "b is still flying")
        f.say("land")
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.GROUND and f.co.drones["c"].phase == Phase.GROUND, 120.0, "landing")
    return f, script()


def focus_moves_all(seed):
    """Partway through the whole room, the operator says to focus on one side. Everyone goes there at once."""
    f = MissionFlight(THREE, seed)
    def script():
        yield from f.lift()
        started = f.fleet.t
        f.missions.start("explore", "all")
        yield from f.wait(15.0)
        focus_at = f.fleet.t
        f.numbers["share_before_focus"] = f.missions.status()["share_seen"]
        f.missions.focus("west")
        f.check(f.missions.status()["region"] == "west", "the region didn't change")
        yield from f.until(lambda: f.inside("west"), 45.0, "every drone to reach the west side")
        f.numbers["all_there_s"] = round(f.fleet.t - focus_at, 1)
        yield from f.until(f.over, MISSION_LIMIT, "the focused mission to finish")
        f.numbers["mission_s"] = round(f.fleet.t - started, 1)
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered()
        yield from f.down()
    return f, script()


def map_with_focus(seed):
    """Map, then 'stop mapping and focus on the north side'. Capture runs throughout and the build waits for the landing."""
    capture = FakeCapture()
    f = MissionFlight(THREE, seed, capture=capture)
    def script():
        yield from f.lift()
        f.missions.start("map", "all")
        yield from f.wait(10.0)
        f.missions.settle()
        f.check(capture.names() == ["start"], f"capture should be running: {capture.calls}")
        f.missions.focus("north")
        yield from f.until(f.over, MISSION_LIMIT, "the map to finish")
        f.missions.settle()
        f.check(capture.names() == ["start"], f"nothing should build with drones up: {capture.calls}")
        f.check_covered()
        f.missions.focus("south")  # it finished, and the operator can still send them on with the same photos
        f.check(f.missions.active, "a finished map couldn't be refocused")
        yield from f.wait(4.0)
        f.missions.settle()
        f.check(capture.names() == ["start"], f"refocus should keep the same capture: {capture.calls}")
        f.missions.stop()
        yield from f.down()
        yield from f.wait(1.0)
        f.missions.settle()
        f.check(capture.names() == ["start", "stop", "build"], f"capture got {capture.calls}")
        f.check(capture.calls[0][1] == capture.calls[2][1], "built a different map from the one captured")
    return f, script()


def stop_midway(seed):
    """Stopping must end the mission, hold the drones and still build what was captured."""
    capture = FakeCapture()
    f = MissionFlight(THREE, seed, capture=capture)
    def script():
        yield from f.lift()
        f.missions.start("map", "all")
        yield from f.wait(12.0)
        f.missions.stop()
        f.check(f.missions.state == "stopped", f"state is {f.missions.state}")
        yield from f.wait(4.0)
        speeds = [float(np.linalg.norm(d.tracked.v[:2])) for d in f.co.drones.values()]
        f.check(max(speeds) < 0.2, f"still moving after stop: {max(speeds):.2f} m/s")
        before = [d.tracked.p.copy() for d in f.co.drones.values()]
        yield from f.wait(5.0)
        moved = max(float(np.linalg.norm(d.tracked.p[:2] - b[:2])) for d, b in zip(f.co.drones.values(), before))
        f.check(moved < 0.4, f"drifted {moved:.2f} m after stop")
        f.missions.settle()
        f.check(capture.names() == ["start"], f"nothing should build with drones up: {capture.calls}")
        yield from f.down()
        yield from f.wait(1.0)
        f.missions.settle()
        f.check(capture.names() == ["start", "stop", "build"], f"capture got {capture.calls}")
    return f, script()


def operator_hold(seed):
    """A hold through the hub, the path that works with the model down, must end the mission too."""
    f = MissionFlight(THREE, seed)
    def script():
        yield from f.lift()
        f.missions.start("explore", "all")
        yield from f.wait(8.0)
        f.say("stop")
        yield from f.wait(0.2)
        f.check(f.missions.state == "stopped", f"state is {f.missions.state}")
        yield from f.down()
    return f, script()


def inspect_thing(seed):
    """Find a couch nobody mapped, then inspect it. Every view must really have it in the picture."""
    couch = FURNITURE[0]
    f = MissionFlight(THREE, seed, hidden=[couch])
    center = (np.array(couch[1].lo) + np.array(couch[1].hi)) / 2
    show = f.inspector.tick
    def noted(now, poses=None):
        for drone_id in poses or ():
            q = f.fleet.drones[drone_id].sim.drone
            if not in_view(f.fleet.truth(drone_id), q.yaw(), center, reach=RELIABLE_REACH, margin=EDGE_MARGIN)[0]:
                f.check(False, f"{drone_id} gave the inspector a view without the couch in the picture")
        show(now, poses)
    f.inspector.tick = noted
    def script():
        yield from f.lift()
        f.say("drone a go to -1 -1.5")
        yield from f.until(lambda: f.memory.find("couch"), 60.0, "somebody to see the couch")
        started = f.fleet.t
        yield from f.run_mission("inspect", target="sofa")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        counts = f.missions.status()["views"]
        f.numbers["views"], f.numbers["mission_s"] = counts["done"], round(f.fleet.t - started, 1)
        f.check(counts["done"] >= 2, f"only {counts} views done")
        f.check(f.inspector.started == ["couch"], f"inspector started with {f.inspector.started}")
        f.check(not f.inspector.watching, "the inspector was left watching")
        f.check(len({drone_id for _, drone_id in f.inspector.shown}) >= 2, f"only {set(f.inspector.shown)} shown")
        yield from f.down()
    return f, script()


def inspect_region(seed):
    """Inspect a side of the room: things on it, then its floor."""
    f = MissionFlight(THREE, seed, hidden=[FURNITURE[1]])
    def script():
        yield from f.lift()
        f.say("drone b go to 0 1.5")
        yield from f.until(lambda: f.memory.find("bed"), 60.0, "somebody to see the bed")
        yield from f.run_mission("inspect", "west")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        counts = f.missions.status()["views"]
        f.numbers["views"] = counts["done"]
        f.check(counts["done"] >= 3, f"only {counts} views done")
        yield from f.down()
    return f, script()


def inspect_lost_drone(seed):
    """A drone drops out in the middle of an inspection. Its view goes to another drone."""
    f = MissionFlight(THREE, seed)
    def script():
        yield from f.lift()
        f.missions.start("inspect", "north")
        yield from f.wait(10.0)
        f.fleet.hidden["c"] = math.inf
        yield from f.until(f.over, MISSION_LIMIT, "the inspection to finish without c")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        counts = f.missions.status()["views"]
        f.numbers["views"] = counts["done"]
        f.check(counts["done"] + counts["skipped"] == counts["total"] and counts["done"] >= counts["total"] - 1, f"{counts}")
        f.say("land")
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.GROUND and f.co.drones["b"].phase == Phase.GROUND, 120.0, "landing")
    return f, script()


def flight_limit(seed):
    """Everyone runs out of flight time mid-mission. They come back and land by the limit, and it gets reported."""
    limit = 60.0
    f = MissionFlight(THREE, seed, limit=limit)
    landed: dict[str, float] = {}
    def all_down():
        for d in f.co.drones.values():
            if d.phase == Phase.GROUND and d.id not in landed:
                landed[d.id] = f.fleet.t
        return len(landed) == 3
    def script():
        yield from f.lift()
        f.missions.start("explore", "all")
        yield from f.until(all_down, 120.0, "everyone to land")
        f.numbers["last_landed_s"] = round(max(landed.values()), 1)
        f.check(len(f.happened("flight_limit")) == 3, f"reported {len(f.happened('flight_limit'))} of 3")
        f.check(f.missions.state in ("failed", "done"), f"ended {f.missions.state}")
    return f, script()


def search_region(seed):
    """Search delegates to the hub's own search, limited to one side."""
    target = on_floor("backpack", 2.4, 1.6)
    f = MissionFlight(THREE, seed, items=[target])
    def script():
        yield from f.lift()
        yield from f.run_mission("search", "north", "bag")
        f.check(f.missions.state == "done" and "Found" in f.missions.reason, f"ended {f.missions.state}: {f.missions.reason}")
        yield from f.down()
    return f, script()


# ----------------------------------------------------------------------------- the demo room
def demo_room_inspect(seed):
    """Inspect the whole demo room. Every piece of staged damage must be in the picture of a settled drone."""
    f = DemoFlight(seed)
    framed: set[str] = set()
    show = f.inspector.tick
    def noted(now, poses=None):
        for drone_id in poses or ():
            camera, yaw = f.fleet.truth(drone_id), f.fleet.drones[drone_id].sim.drone.yaw()
            framed.update(d.name for d in f.world.damage if in_view(camera, yaw, np.array(d.where), reach=RELIABLE_REACH)[0])
        show(now, poses)
    f.inspector.tick = noted
    def script():
        yield from f.takeoff()
        started = f.fleet.t
        f.missions.start("inspect", "all")
        yield from f.until(f.over, MISSION_LIMIT, "the inspection to finish")
        f.numbers["mission_s"] = round(f.fleet.t - started, 1)
        f.numbers["views"] = f.missions.status()["views"]["done"]
        f.numbers["damage_framed"] = len(framed)
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check(len(framed) == len(f.world.damage), f"never framed: {sorted({d.name for d in f.world.damage} - framed)}")
        yield from f.land()
    return f, script()


def demo_room_map(seed):
    """Map the whole demo room, with four things nobody put on the map, and hear which areas are left."""
    capture = FakeCapture()
    f = DemoFlight(seed, capture)
    def script():
        yield from f.takeoff()
        started = f.fleet.t
        f.missions.start("map", "all")
        yield from f.until(f.over, MISSION_LIMIT, "the map to finish")
        f.numbers["mission_s"] = round(f.fleet.t - started, 1)
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered()
        f.missions.settle()
        f.check(capture.names() == ["start"], f"nothing should build with drones up: {capture.calls}")
        gaps = {gap["name"]: gap for gap in f.missions.gaps()}
        f.check(gaps["all"]["share_seen"] >= COVERED, f"gaps says only {gaps['all']['share_seen']}")
        yield from f.land()
        yield from f.wait(1.0)
        f.missions.settle()
        f.check(capture.names() == ["start", "stop", "build"], f"capture got {capture.calls}")
    return f, script()


# ----------------------------------------------------------------------------- on floor tags, as on a real flight
def floor_explore(seed):
    """Every drone's position comes from the tags in its own picture, so a view with no tag in it costs the drone."""
    f = FloorMissionFlight(seed)
    def script():
        yield from f.takeoff()
        yield from f.run_mission("explore", "all")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered(0.9)
        f.numbers["sigma_worst"] = round(max(row[1] for row in f.errors), 2)
        f.numbers["error_xy_worst"] = round(max(row[0] for row in f.errors), 2)
        yield from f.land()
    return f, script()


def floor_lost_drone(seed):
    """One drone's video stops for good. It lands by itself and the other two finish."""
    f = FloorMissionFlight(seed)
    def script():
        yield from f.takeoff()
        f.missions.start("explore", "all")
        yield from f.wait(8.0)
        f.fleet.hidden["b"] = math.inf
        yield from f.until(f.over, MISSION_LIMIT, "the mission to finish without b")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered(0.9)
        f.check(any(e["drone"] == "b" for e in f.happened("drone_lost")), "nobody reported b lost")
        f.hub.land()
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.GROUND and f.co.drones["c"].phase == Phase.GROUND, 90.0, "landing")
    return f, script()


def floor_focus(seed):
    """Focus moves all three to one side, using only the tags to get there."""
    f = FloorMissionFlight(seed)
    def script():
        yield from f.takeoff()
        f.missions.start("explore", "all")
        yield from f.wait(6.0)
        focus_at = f.fleet.t
        f.missions.focus("west")
        yield from f.until(lambda: f.inside("west"), 45.0, "every drone to reach the west side")
        f.numbers["all_there_s"] = round(f.fleet.t - focus_at, 1)
        yield from f.until(f.over, MISSION_LIMIT, "the focused mission to finish")
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check_covered(0.9)
        yield from f.land()
    return f, script()


def floor_demo_inspect(seed):
    """Inspect the whole demo room on floor tags. Nothing may lose its position by flying over or up against the furniture."""
    f = DemoFloorFlight(seed)
    def script():
        yield from f.takeoff()
        yield from f.run_mission("inspect", "all")
        f.numbers.update({f"views_{key}": value for key, value in f.missions.status()["views"].items()})
        f.numbers["lost"] = len(f.happened("drone_lost"))
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check(not f.happened("drone_lost"), f"lost a drone: {f.happened('drone_lost')}")
        yield from f.land()
    return f, script()


def floor_demo_blind(seed):
    """One drone's video stops for good, so it lands blind. The other two must still finish the inspection."""
    f = DemoFloorFlight(seed)
    def script():
        yield from f.takeoff()
        f.missions.start("inspect", "all")
        yield from f.wait(8.0)
        f.fleet.hidden["2"] = math.inf
        started = f.fleet.t
        yield from f.until(f.over, MISSION_LIMIT, "the inspection to finish without 2")
        f.numbers["mission_s"] = round(f.fleet.t - started, 1)
        f.numbers.update({f"views_{key}": value for key, value in f.missions.status()["views"].items()})
        f.check(f.missions.state == "done", f"ended {f.missions.state}: {f.missions.reason}")
        f.check(any(e["drone"] == "2" for e in f.happened("drone_lost")), "nobody reported 2 lost")
        f.hub.land()
        yield from f.until(lambda: f.co.drones["1"].phase == Phase.GROUND and f.co.drones["3"].phase == Phase.GROUND, 90.0, "landing")
    return f, script()


MISSIONS = {m.__name__: m for m in (
    explore_region, explore_all_furniture, lost_drone, focus_moves_all, map_with_focus, stop_midway, operator_hold,
    inspect_thing, inspect_region, inspect_lost_drone, flight_limit, search_region, demo_room_inspect, demo_room_map,
    floor_explore, floor_lost_drone, floor_focus, floor_demo_inspect, floor_demo_blind,
)}
FAST = [name for name in MISSIONS if not name.startswith("floor_")]


def fly(name: str, seed: int):
    """Scored by the same rules as every other mission on the swarm bench. Returns the outcome and its numbers."""
    made = {}
    def mission(seed):
        flight, script = MISSIONS[name](seed)
        made["flight"] = flight
        return flight, script
    bench.MISSIONS[name] = mission
    try:
        outcome = bench.fly(name, seed)
    finally:
        del bench.MISSIONS[name]
    flight = made["flight"]
    flight.missions.settle()
    numbers = {**flight.numbers, "errors": flight.missions.errors, "hub_errors": flight.hub.think_errors}
    if flight.missions.errors or flight.hub.think_errors:
        outcome.notes.append(f"{flight.missions.errors} mission errors, {flight.hub.think_errors} hub errors")
    return outcome, numbers


def _job(job):
    logging.disable(logging.WARNING)  # a drone lost on purpose reports itself on every tick
    return fly(*job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm.mission_bench")
    parser.add_argument("missions", nargs="*", help="which to run; all of them by default")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--floor", action="store_true", help="also fly the slower missions that use floor tags")
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args(argv)
    started = time.time()
    names = args.missions or (list(MISSIONS) if args.floor else FAST)
    jobs = [(name, seed) for name in names for seed in range(args.first_seed, args.first_seed + args.seeds)]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    outcomes = [outcome for outcome, _ in results]
    result = bench.report(outcomes)
    print(f"{'mission':<24}{'pass':>8}{'hit':>5}{'wall':>6}{'crash':>7}{'closest':>9}  numbers (median)")
    for row in result["missions"]:
        group = [numbers for outcome, numbers in results if outcome.name == row["mission"]]
        keys = sorted({key for numbers in group for key in numbers} - {"errors", "hub_errors"})
        shown = "  ".join(f"{key}={statistics.median(n[key] for n in group if key in n):g}" for key in keys)
        print(f"{row['mission']:<24}{row['passed']:>5}/{row['flights']:<2}{row['collisions']:>5}{row['contacts']:>6}"
              f"{row['crashes']:>7}{row['closest_m']:>8.2f}m  {shown}")
    print(f"\n{result['passed']}/{result['flights']} flights passed, {result['collisions']} drone collisions, "
          f"{result['contacts']} obstacle contacts, {result['crashes']} crashes, in {time.time() - started:.0f} s. Simulator results.")
    for failure in result["failures"][:40]:
        print(f"  FAIL {failure['mission']} seed {failure['seed']}: {failure['notes'] or failure}")
    if args.json:
        with open(args.json, "w") as out:
            json.dump({**result, "numbers": [numbers for _, numbers in results]}, out, indent=1)
    return 0 if result["passed"] == result["flights"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
