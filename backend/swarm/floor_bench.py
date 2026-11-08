"""The fixed test set for flying from floor tags: `python -m swarm floor-bench`.

Every flight uses the real tag detector on rendered frames of the real tag layout, with no pose
source but the drone's own camera. Scored against the simulator's truth. This says what the method
can do in simulation. Nothing here has flown, and the real lens, sharpness and exposure are unknown.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import time
from dataclasses import replace

import numpy as np

from sim import flight_controller
from sim.scene import Box, Scene
from swarm import bench
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.safety import Limits
from swarm.sim_anchors import FloorCamera, SimVideo, VideoConditions
from swarm.sim_fleet import Conditions
from swarm.tracking import AnchorReader, DroneOptics, InsideOutTracker, Lens, anchors_section, fit_focal, load_anchors, tag_grid

WHITE_SQUARE_M = 0.0767  # the printed tags
HALL = Box((-5.0, -4.0, 0.0), (5.0, 4.0, 2.8), "wall")
FENCE = ((-3.0, -2.5, 0.0), (1.7, 2.5, 2.4))  # the tag grid, plus room to see its first row from behind it
Z = 1.0
THREE = {"a": (-2.5, 1.2), "b": (-2.5, 0.0), "c": (-2.5, -1.2)}
CROSSING = {"a": (-2.5, 1.2), "b": (-2.5, -1.2), "c": (-0.5, -2.1)}
CROSSED = {"a": (1.2, -1.2, Z), "b": (1.2, 1.2, Z), "c": (-0.5, 2.1, Z)}


_default_limits = None  # set by fly() for the length of one flight


class FloorFlight(bench.Flight):
    def __init__(self, seed, pads=THREE, boxes=(), conditions=None, limits=None, video=None, tags=None, assumed_fov=90.0, frame_rate=None):
        limits = limits or _default_limits
        scene = Scene(name="hall", bounds=HALL, boxes=tuple(boxes))
        super().__init__(scene, pads, seed, conditions, limits, FENCE)
        configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in pads.items()]
        section = anchors_section(WHITE_SQUARE_M, tags if tags is not None else tag_grid())
        lens = Lens.from_fov(horizontal_fov_deg=assumed_fov)
        optics = {c.id: DroneOptics(c.pad[0], c.pad[1], c.yaw, lens=lens) for c in configs}
        self.tracker = InsideOutTracker(optics, load_anchors(section), FENCE)
        self.video = SimVideo(self.fleet, self.tracker, scene, video or VideoConditions(), seed, **({"rate": frame_rate} if frame_rate else {}))
        self.co = Coordinator(self.room, configs, self.fleet.links, limits, tracker=self.tracker)
        self.errors: list[tuple[float, float, float, float]] = []
        self.velocity_gain_scale = 1.0

    def tick(self) -> None:
        self.fleet.advance(bench.TICK)
        self.video.step()
        self.co.tick(self.fleet.t, self.tracker.poses(self.fleet.t), [])
        for key, d in self.co.drones.items():
            if d.phase in (Phase.CHECK, Phase.AIR):
                truth = self.fleet.truth(key)
                turn = (d.tracked.yaw - self.fleet.drones[key].sim.drone.yaw() + math.pi) % (2 * math.pi) - math.pi
                self.errors.append((float(np.linalg.norm(d.tracked.p[:2] - truth[:2])), abs(float(d.tracked.p[2] - truth[2])),
                                    abs(math.degrees(turn)), d.tracked.sigma))


def _there_and_back(f, out, back, timeout=90.0):
    yield from f.takeoff()
    yield from f.go(out, timeout)
    yield from f.go(back, timeout)
    yield from f.land()


def _home(pads):
    return {k: (v[0], v[1], Z) for k, v in pads.items()}


def hover(seed):
    f = FloorFlight(seed)
    def script():
        yield from f.takeoff()
        yield from f.wait(10.0)
        yield from f.land()
    return f, script()


def swap(seed):
    pads = {"a": (-2.5, 0.0), "b": (1.2, 0.0)}
    f = FloorFlight(seed, pads)
    return f, _there_and_back(f, {"a": (1.2, 0.0, Z), "b": (-2.5, 0.0, Z)}, _home(pads))


def cross(seed):
    f = FloorFlight(seed, CROSSING)
    return f, _there_and_back(f, CROSSED, _home(CROSSING))


def ring(seed):
    f = FloorFlight(seed)
    def script():
        yield from f.takeoff()
        yield from f.form("ring", (-0.5, 0.0), 1.2, Z)
        for turn in (2.1, 4.2, 0.0):
            yield from f.form("ring", (-0.5, 0.0), 1.2, Z, turn)
        yield from f.land()
    return f, script()


def formations(seed):
    f = FloorFlight(seed)
    def script():
        yield from f.takeoff()
        yield from f.form("line", (-0.5, 0.0), 1.0, Z, math.pi / 2)
        yield from f.form("ring", (0.0, 0.0), 1.0, Z)
        yield from f.form("line", (0.5, 0.0), 1.0, Z, math.pi / 2)
        yield from f.land()
    return f, script()


def pillar(seed):
    f = FloorFlight(seed, boxes=[Box((-0.7, -0.6, 0.0), (-0.3, 0.6, 2.8), "wood")])
    return f, _there_and_back(f, {"a": (1.2, 1.2, Z), "b": (1.2, 0.0, Z), "c": (1.2, -1.2, Z)}, _home(THREE))


def wall_gap(seed):
    """A wall across the grid with one door, hiding the tags beyond it until a drone is through."""
    f = FloorFlight(seed, boxes=[Box((-0.6, -4.0, 0.0), (-0.4, -0.6, 2.8), "wood"), Box((-0.6, 0.6, 0.0), (-0.4, 4.0, 2.8), "wood")])
    return f, _there_and_back(f, {"a": (1.2, 1.2, Z), "b": (1.2, 0.0, Z), "c": (1.2, -1.2, Z)}, _home(THREE), 120.0)


def under_table(seed):
    """A table sits on two of the tags, so they can't be seen. The drones fly over it."""
    f = FloorFlight(seed, boxes=[Box((-0.3, -1.3, 0.0), (0.6, 0.3, 0.7), "wood")])
    high = 1.25
    return f, _there_and_back(f, {"a": (1.2, 1.2, high), "b": (1.2, 0.0, high), "c": (1.2, -1.2, high)}, _home(THREE))


def missing_tags(seed):
    """Every other tag isn't there, as if the layout were only half finished."""
    f = FloorFlight(seed, tags=[t for t in tag_grid() if t["tag"] % 2 == 0])
    return f, _there_and_back(f, {"a": (1.2, 1.2, Z), "b": (1.2, 0.0, Z), "c": (1.2, -1.2, Z)}, _home(THREE))


def _crossing_with(f, during):
    yield from f.takeoff()
    problems = f.co.goto(CROSSED)
    if problems:
        f.failed.append(str(problems))
        return
    yield from during()
    yield from f.until(f.co.all_arrived, 90.0, "arrive")
    yield from f.land()


def brief_dropouts(seed):
    """One drone goes blind for 0.7 s, again and again, as blur from a fast move would do."""
    f = FloorFlight(seed, CROSSING)
    def during():
        for _ in range(5):
            yield from f.wait(1.5)
            f.fleet.hidden["a"] = f.fleet.t + 0.7
    return f, _crossing_with(f, during)


def covered_view(seed):
    """One drone sees no tags for 6 s mid-crossing. It must hold on its stock hold and carry on, or land."""
    f = FloorFlight(seed, CROSSING)
    def during():
        yield from f.wait(3.0)
        f.fleet.hidden["a"] = f.fleet.t + 6.0
    return f, _crossing_with(f, during)


def blind_for_good(seed):
    """One drone's video stops for good. It must land itself, and the others must finish."""
    f = FloorFlight(seed, CROSSING)
    def script():
        yield from f.takeoff()
        problems = f.co.goto(CROSSED)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(3.0)
        f.fleet.hidden["a"] = math.inf
        others = (f.co.drones["b"], f.co.drones["c"])
        yield from f.until(lambda: all(d.arrived for d in others), 70.0, "the others to arrive")
        yield from f.until(lambda: f.fleet.drones["a"].sim.fc.state == "READY", 45.0, "the blind drone to land")
        f.co.land(f.fleet.t)
        yield from f.until(lambda: f.co.all_phase(Phase.GROUND), 60.0, "landing")
    return f, script()


def frozen_coordinator(seed):
    f = FloorFlight(seed, CROSSING)
    def during():
        yield from f.wait(3.5)
        f.fleet.advance(3.0)
    return f, _crossing_with(f, during)


def reversed_sticks(seed):
    f = FloorFlight(seed)
    send = f.fleet.links["b"].send_sticks
    f.fleet.links["b"].send_sticks = lambda t, y, p, r, ttl: send(t, y, p, -r, ttl)
    def script():
        yield from f.until(lambda: not f.co.preflight(f.fleet.t), 5.0, "preflight")
        problems = f.co.takeoff(f.fleet.t)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.until(lambda: "response" in f.co.drones["b"].faults, 30.0, "the check to catch it")
        yield from f.until(lambda: f.fleet.drones["b"].sim.fc.state == "READY", 25.0, "it to land")
        yield from f.until(lambda: f.co.drones["a"].phase == Phase.AIR, 25.0, "the others to pass")
        f.co.land(f.fleet.t)
        yield from f.until(lambda: f.fleet.crashed() or all(d.sim.fc.state == "READY" for d in f.fleet.drones.values()), 50.0, "landing")
    return f, script()


def refused_takeoff(seed):
    """A flat battery: the toggle isn't repeated, and with nothing seen, nobody assumes either way."""
    f = FloorFlight(seed)
    f.fleet.drones["a"].sim.drone.battery.soc = 0.02
    def script():
        yield from f.until(lambda: not f.co.preflight(f.fleet.t), 5.0, "preflight")
        problems = f.co.takeoff(f.fleet.t)
        if problems:
            f.failed.append(str(problems))
            return
        yield from f.wait(12.0)
        if "takeoff" not in f.co.drones["a"].faults:
            f.failed.append("the failed takeoff was not reported")
        if f.co.drones["b"].phase != Phase.GROUND:
            f.failed.append("the next drone took off anyway")
    return f, script()


def _crossing(seed, **kwargs):
    f = FloorFlight(seed, CROSSING, **kwargs)
    return f, _there_and_back(f, CROSSED, _home(CROSSING), 120.0)


def wrong_gains(seed):
    return _crossing(seed, conditions=Conditions(speed_scale=(0.5, 1.8)))


def sluggish(seed):
    flight, script = _crossing(seed)
    flight.velocity_gain_scale = 0.5
    return flight, script


def drafty(seed):
    return _crossing(seed, conditions=Conditions(draft=0.5))


def soft_video(seed):
    return _crossing(seed, video=VideoConditions(render_scale=1.0, blur_sigma=2.0))


def dim_room(seed):
    """A 1/30 s exposure instead of 1/60 s: twice the blur."""
    return _crossing(seed, video=VideoConditions(exposure_s=1 / 30))


def late_video(seed):
    """Video 100 ms later than the tracker assumes."""
    return _crossing(seed, video=VideoConditions(latency_s=0.25))


def slow_video(seed):
    """Only 4 frames a second get through, as on a weak link or an overloaded Mac."""
    return _crossing(seed, frame_rate=4.0)


def wrong_lens(seed):
    """The lens is really 82 degrees wide and the tracker assumes 90: what flying without `fit-focal` risks."""
    return _crossing(seed, video=VideoConditions(fov_deg=82.0))


def fitted_lens(seed):
    """The same lens, after `fit-focal` measured it from a hand-held hover over the grid."""
    video = VideoConditions(fov_deg=82.0)
    anchors = load_anchors(anchors_section(WHITE_SQUARE_M, tag_grid()))
    scene = Scene(name="hall", bounds=HALL)
    frames = []
    for i in range(12):
        camera = FloorCamera(scene, anchors, 45.0, video, seed + i)
        c, s = math.cos(0.08 * i - 0.4), math.sin(0.08 * i - 0.4)
        image = camera.render(-1.5 + 0.1 * i, 0.5 - 0.1 * i, 1.0 + 0.02 * i, (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
        frames.append(AnchorReader(anchors).find(image))
    fitted = fit_focal(frames)
    fov = 2 * math.degrees(math.atan(1184 / 2 / fitted["focal_px"]))
    return _crossing(seed, video=video, assumed_fov=fov)


MISSIONS = {m.__name__: m for m in (
    hover, swap, cross, ring, formations, pillar, wall_gap, under_table, missing_tags, brief_dropouts, covered_view,
    blind_for_good, frozen_coordinator, reversed_sticks, refused_takeoff, wrong_gains, sluggish, drafty, soft_video,
    dim_room, late_video, slow_video, wrong_lens, fitted_lens,
)}


def fly(name: str, seed: int, max_speed: float | None = None) -> dict:
    global _default_limits
    _default_limits = Limits(max_speed=max_speed) if max_speed else None
    try:
        flight, script = MISSIONS[name](seed)
    finally:
        _default_limits = None
    gain = flight_controller._VELOCITY_GAIN
    flight_controller._VELOCITY_GAIN = gain * flight.velocity_gain_scale
    try:
        flight.run(script)
    finally:
        flight_controller._VELOCITY_GAIN = gain
    fleet = flight.fleet
    for drone_id, link in fleet.links.items():
        toggles = [command for _, command in link.commands if command in ("takeoff", "land")]
        if toggles.count("takeoff") > 1 or any(a == b for a, b in zip(toggles, toggles[1:])):
            flight.failed.append(f"{drone_id} sent a toggle twice: {toggles}")
    notes = list(flight.failed)
    for d in flight.co.drones.values():
        notes += [f"{d.id}: {message}" for message in d.faults.values()]
    out = {"mission": name, "seed": seed, "hits": len(fleet.collisions), "walls": len(fleet.contacts), "crashes": len(fleet.crashed()),
           "closest": round(fleet.min_separation, 3), "finished": not flight.failed, "duration": round(fleet.t, 1), "notes": notes,
           "rendered": flight.video.rendered, "culled": flight.video.culled}
    out["passed"] = out["finished"] and not (out["hits"] or out["walls"] or out["crashes"])
    if flight.errors:
        rows = np.array(flight.errors)
        out.update(xy_median=float(np.median(rows[:, 0])), xy_p95=float(np.percentile(rows[:, 0], 95)), xy_max=float(rows[:, 0].max()),
                   z_p95=float(np.percentile(rows[:, 1], 95)), yaw_p95=float(np.percentile(rows[:, 2], 95)),
                   within_2_sigma=float(np.mean(rows[:, 0] <= 2 * np.maximum(rows[:, 3], 1e-3))), sigma_max=float(rows[:, 3].max()))
    return out


def _job(job):
    return fly(*job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm floor-bench")
    parser.add_argument("missions", nargs="*")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-speed", type=float, help="m/s per axis; the default is Limits' 0.5")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    started = time.time()
    names = args.missions or list(MISSIONS)
    jobs = [(n, s, args.max_speed) for n in names for s in range(args.first_seed, args.first_seed + args.seeds)]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    print(f"{'mission':<20}{'pass':>7}{'hit':>4}{'wall':>5}{'crash':>6}{'closest':>8}{'xy med':>8}{'xy p95':>8}{'xy max':>8}{'z p95':>7}{'yaw95':>7}{'in2s':>6}{'sig max':>8}")
    for name in names:
        g = [r for r in results if r["mission"] == name]
        def med(key):
            values = [r[key] for r in g if key in r]
            return statistics.median(values) if values else math.nan
        print(f"{name:<20}{sum(r['passed'] for r in g):>4}/{len(g):<2}{sum(r['hits'] for r in g):>4}{sum(r['walls'] for r in g):>5}"
              f"{sum(r['crashes'] for r in g):>6}{min(r['closest'] for r in g):>8.2f}{med('xy_median'):>8.3f}{med('xy_p95'):>8.3f}"
              f"{max((r.get('xy_max', 0) for r in g)):>8.2f}{med('z_p95'):>7.3f}{med('yaw_p95'):>7.1f}{med('within_2_sigma'):>6.2f}{max((r.get('sigma_max', 0) for r in g)):>8.2f}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} flights passed, {sum(r['hits'] for r in results)} drone collisions, {sum(r['walls'] for r in results)} "
          f"wall or obstacle contacts, in {time.time() - started:.0f} s. Simulated: not proven on hardware.")
    for r in [r for r in results if not r["passed"]][:40]:
        print(f"  FAIL {r['mission']} seed {r['seed']}: hit {r['hits']} wall {r['walls']} closest {r['closest']} {r['notes'][:3]}")
    if args.json:
        with open(args.json, "w") as out:
            json.dump(results, out, indent=1)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
