"""Plan, predict and fly open-loop trajectories.

The S20 reports nothing back, so a plan is a timed list of stick positions. Its expected
path comes from the simulator's flight tuning; the prediction flies the exact packets the
drone would get through the simulator.

Plan language, one command per line ('#' starts a comment). x is forward from the
takeoff spot, y left, z up, in metres; headings are degrees, positive to the left.

    takeoff                 spin up and climb to the hover height
    land
    goto X Y [Z]            fly straight to a point without turning
    move F L [U]            fly relative to where the nose points
    yaw DEG | turn DEG      turn in place to a heading, or by an angle
    hold S                  hover for S seconds
    speed M/S | climb M/S | rate DEG/S | settle S
"""

from __future__ import annotations

import bisect
import json
import math
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from control.strategies import LinearStrategy
from sim.config import DEFAULT_FIRMWARE, FlightTuning

AXES = ("throttle", "yaw", "pitch", "roll")
NEUTRAL = (0.0, 0.0, 0.0, 0.0)
DEADBAND = 0.06  # just above the flight controller's estimated 0.05
TICK = 1 / 50


class PlanError(ValueError):
    pass


@dataclass
class Segment:
    t: float
    dur: float
    sticks: tuple[float, float, float, float]  # throttle, yaw, pitch, roll
    action: str | None
    line: int
    text: str
    end: tuple[float, float, float, float]  # x, y, z, heading in degrees


@dataclass
class Plan:
    text: str
    segments: list[Segment]
    warnings: list[str]

    @property
    def duration(self) -> float:
        return self.segments[-1].t + self.segments[-1].dur if self.segments else 0.0

    def find(self, t: float) -> tuple[int, Segment | None]:
        for i, seg in enumerate(self.segments):
            if t < seg.t + seg.dur:
                return i, seg
        return len(self.segments), None

    def path(self) -> list[list[float]]:
        points = [[0.0, 0.0, 0.0, 0.0, 0.0]]
        for seg in self.segments:
            points.append([round(seg.t + seg.dur, 3), *(round(v, 3) for v in seg.end)])
        return points

    def json(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "segments": [asdict(seg) for seg in self.segments],
            "path": self.path(),
            "warnings": self.warnings,
        }


def sticks(values: dict) -> tuple[float, float, float, float]:
    return tuple(values.get(axis, 0.0) for axis in AXES)


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def plan(text: str, tuning: FlightTuning | None = None) -> Plan:
    tune = tuning or FlightTuning.calibrated()
    top_speed, top_climb, top_descent = tune.max_speed_mps, tune.max_climb_mps, tune.max_descent_mps
    top_rate = math.degrees(tune.max_yaw_rate_rad_s)
    x = y = z = heading = 0.0
    airborne = False
    speed, climb, rate, settle = 1.0, 0.5, 90.0, 1.5
    segments: list[Segment] = []
    warnings: list[str] = []
    clock = 0.0

    def add(n, source, dur, sticks=NEUTRAL, action=None):
        nonlocal clock
        segments.append(Segment(round(clock, 4), round(dur, 4), tuple(round(s, 4) for s in sticks), action, n, source,
                                (x, y, z, heading)))
        clock += dur

    def lift(n, source):
        nonlocal z, airborne
        if not airborne:
            warnings.append(f"line {n}: no takeoff before this, so the plan assumes the drone already hovers "
                            f"at {tune.takeoff_height_m:.1f} m")
            airborne = True
            z = tune.takeoff_height_m

    def fly(n, source, dx, dy, dz):
        nonlocal x, y, z
        lift(n, source)
        h = math.radians(heading)
        forward = dx * math.cos(h) + dy * math.sin(h)
        left = -dx * math.sin(h) + dy * math.cos(h)
        vertical = min(climb, top_climb if dz > 0 else top_descent)
        dur = max(math.hypot(forward, left) / speed, abs(dz) / vertical)
        if dur < 1e-3:
            return
        x, y, z = x + dx, y + dy, z + dz
        if z > tune.flow_max_height_m:
            warnings.append(f"line {n}: above {tune.flow_max_height_m:.0f} m the flow sensor loses the floor, "
                            "so the drone drifts and sticks set tilt instead of speed")
        if z < 0.3:
            warnings.append(f"line {n}: ends {z:.2f} m above the floor")
        parts = {"pitch": forward / top_speed, "roll": -left / top_speed,
                 "throttle": dz / (top_climb if dz > 0 else top_descent)}
        main = {k: v / dur for k, v in parts.items() if abs(v / dur) >= DEADBAND}
        if main:
            add(n, source, dur, sticks(main))
        for k, v in parts.items():
            if abs(v) > 1e-4 and abs(v / dur) < DEADBAND:
                add(n, source, abs(v) / DEADBAND, sticks({k: math.copysign(DEADBAND, v)}))
        add(n, source, settle)

    def rotate(n, source, delta):
        nonlocal heading
        lift(n, source)
        if abs(delta) < 0.5:
            return
        stick = math.copysign(rate / top_rate, -delta)
        heading += delta
        add(n, source, abs(delta) / rate, (0.0, stick, 0.0, 0.0))
        add(n, source, min(settle, 1.0))

    for n, raw in enumerate(text.splitlines(), 1):
        source = raw.split("#")[0].strip()
        if not source:
            continue
        word, *rest = source.split()
        word = word.lower()
        try:
            args = [float(a) for a in rest]
        except ValueError:
            raise PlanError(f"line {n}: arguments must be numbers: {source}") from None

        def need(low, high=None):
            high = low if high is None else high
            if not low <= len(args) <= high:
                count = str(low) if low == high else f"{low} to {high}"
                raise PlanError(f"line {n}: {word} takes {count} numbers")

        if word == "takeoff":
            need(0)
            if airborne:
                raise PlanError(f"line {n}: already flying")
            airborne, z = True, tune.takeoff_height_m
            add(n, source, tune.spinup_s + tune.takeoff_height_m / tune.takeoff_climb_mps + 1.2, action="takeoff")
        elif word == "land":
            need(0)
            if not airborne:
                raise PlanError(f"line {n}: not flying")
            dur = max(z - 0.3, 0) / tune.landing_descent_mps + 0.3 / 0.25 + 1.5
            airborne, z = False, 0.0
            add(n, source, dur, action="land")
        elif word in ("goto", "move"):
            need(2, 3)
            if word == "goto":
                tx, ty = args[0], args[1]
                tz = args[2] if len(args) == 3 else (z if airborne else tune.takeoff_height_m)
                lift(n, source)
                fly(n, source, tx - x, ty - y, tz - z)
            else:
                h = math.radians(heading)
                f, lft, up = args[0], args[1], args[2] if len(args) == 3 else 0.0
                fly(n, source, f * math.cos(h) - lft * math.sin(h), f * math.sin(h) + lft * math.cos(h), up)
        elif word == "yaw":
            need(1)
            rotate(n, source, wrap(args[0] - heading))
        elif word == "turn":
            need(1)
            rotate(n, source, args[0])
        elif word == "hold":
            need(1)
            if args[0] <= 0:
                raise PlanError(f"line {n}: hold needs a positive time")
            add(n, source, args[0])
        elif word == "speed":
            need(1)
            if not DEADBAND * top_speed <= args[0] <= top_speed:
                raise PlanError(f"line {n}: speed must be {DEADBAND * top_speed:.2f} to {top_speed:.1f} m/s")
            speed = args[0]
        elif word == "climb":
            need(1)
            if not DEADBAND * top_climb <= args[0] <= top_climb:
                raise PlanError(f"line {n}: climb must be {DEADBAND * top_climb:.2f} to {top_climb:.1f} m/s")
            climb = args[0]
        elif word == "rate":
            need(1)
            if not DEADBAND * top_rate <= args[0] <= top_rate:
                raise PlanError(f"line {n}: rate must be {DEADBAND * top_rate:.0f} to {top_rate:.0f} deg/s")
            rate = args[0]
        elif word == "settle":
            need(1)
            if args[0] < 0:
                raise PlanError(f"line {n}: settle can't be negative")
            settle = args[0]
        else:
            raise PlanError(f"line {n}: unknown command {word!r}")

    if not segments:
        raise PlanError("the plan is empty")
    if airborne:
        warnings.append("the plan ends in the air: the drone keeps hovering")
    return Plan(text, segments, warnings)


def predict(p: Plan, scene=None, firmware: str = DEFAULT_FIRMWARE, start=(0.0, 0.0, 0.0),
            wind: dict | None = None, seed: int = 1, rate_hz: float = 80.0, until: float | None = None,
            tuning: FlightTuning | None = None) -> dict:
    """Fly the plan's exact control packets through the simulator. Poses are in the plan's frame."""
    from models.s2x_rc import S2xDroneModel
    from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
    from sim.scene import Scene
    from sim.simulator import Simulator

    sim = Simulator(firmware=firmware, scene=scene or Scene(name="open"), seed=seed, tuning=tuning)
    sx, sy, syaw = start
    sim.place(sx, sy, syaw)
    if wind:
        sim.wind.set(wind.get("speed", 0.0), wind.get("direction", 0.0), wind.get("gust", 0.0))
    model = S2xDroneModel()
    model.set_strategy(LinearStrategy())
    rc = S2xRCProtocolAdapter("127.0.0.1")
    rc.sock.close()
    client = ("172.16.10.2", 50000)
    period, step = 1 / rate_hz, 1 / 400
    cos_s, sin_s = math.cos(-syaw), math.sin(-syaw)

    def send(sticks):
        model.update(period, dict(zip(AXES, sticks)))
        sim.receive_udp(client, rc.build_control_packet(model))

    due = 0.0
    while sim.t < sim.tuning.boot_s + 0.5:
        if sim.t >= due:
            send(NEUTRAL)
            due += period
        sim.step(step)
    begin = sim.t
    fired: set[int] = set()
    track: list[list[float]] = []
    first_event = len(sim.fc.events)
    end = p.duration + 3.0 if until is None else until
    while sim.t - begin < end:
        t = sim.t - begin
        if sim.t >= due:
            index, seg = p.find(t)
            if seg is not None and seg.action and index not in fired:
                fired.add(index)
                getattr(model, seg.action)()
            send(seg.sticks if seg is not None else NEUTRAL)
            due += period
        sim.step(step)
        if not track or t - track[-1][0] >= 0.1:
            d = sim.drone
            dx, dy = d.x - sx, d.y - sy
            track.append([round(t, 2), round(dx * cos_s - dy * sin_s, 3), round(dx * sin_s + dy * cos_s, 3),
                          round(d.z - sim.spec.gear_height_m, 3), round(wrap(math.degrees(d.yaw() - syaw)), 1)])
    events = [[round(t - begin, 2), m] for t, m in list(sim.fc.events)[first_event:]]
    return {"track": track, "events": events, "state": sim.fc.state}


def pose(track: list[list[float]], t: float) -> list[float]:
    times = [point[0] for point in track]
    return track[min(bisect.bisect_left(times, t), len(track) - 1)]


def unwrap(track: list[list[float]]) -> list[list[float]]:
    """The same track with its heading counting whole turns instead of wrapping at 180."""
    out, total, last = [], 0.0, None
    for point in track:
        total += 0.0 if last is None else wrap(point[4] - last)
        last = point[4]
        out.append([*point[:4], (track[0][4] if track else 0.0) + total])
    return out


def ratio(want: float, got: float, floor: float) -> float:
    if abs(got) < floor or want * got <= 0:
        return 1.0
    return min(2.0, max(0.5, want / got))


def rescale(seg: Segment, ratios: dict) -> None:
    products = [s * seg.dur * ratios.get(axis, 1.0) for axis, s in zip(AXES, seg.sticks)]
    peak = max(abs(v) for v in products)
    if peak == 0:
        return
    dur = min(max(seg.dur, peak), peak / DEADBAND)
    seg.dur = round(dur, 4)
    seg.sticks = tuple(round(v / dur, 4) for v in products)


def refine(p: Plan, passes: int = 3, **kw) -> tuple[Plan, dict]:
    """Rescale each leg, in order, until the simulator flies it to its target, since nothing on the drone can."""
    groups: dict[int, list[int]] = {}
    for i, seg in enumerate(p.segments):
        if seg.action is None and any(other.line == seg.line and any(other.sticks) for other in p.segments):
            groups.setdefault(seg.line, []).append(i)
    for idx in groups.values():
        moving = [p.segments[i] for i in idx if any(p.segments[i].sticks)]
        for _ in range(passes):
            first, last = p.segments[idx[0]], p.segments[idx[-1]]
            track = unwrap(predict(p, until=last.t + last.dur + 0.2, **kw)["track"])
            a, b, goal = pose(track, first.t), pose(track, last.t + last.dur), first.end
            if any(seg.sticks[1] for seg in moving):
                ratios = {"yaw": ratio(goal[3] - a[4], b[4] - a[4], 3.0)}
            else:
                h = math.radians(a[4])
                c, s_ = math.cos(h), math.sin(h)
                wf, wl = (goal[0] - a[1]) * c + (goal[1] - a[2]) * s_, -(goal[0] - a[1]) * s_ + (goal[1] - a[2]) * c
                gf, gl = (b[1] - a[1]) * c + (b[2] - a[2]) * s_, -(b[1] - a[1]) * s_ + (b[2] - a[2]) * c
                ratios = {"pitch": ratio(wf, gf, 0.05), "roll": ratio(wl, gl, 0.05),
                          "throttle": ratio(goal[2] - a[3], b[3] - a[3], 0.05)}
            if all(abs(r - 1) < 0.02 for r in ratios.values()):
                break
            for seg in moving:
                rescale(seg, ratios)
            clock = 0.0
            for seg in p.segments:
                seg.t = round(clock, 4)
                clock += seg.dur
    return p, predict(p, **kw)


class Runner:
    """Flies a plan through the backend's flight controller, and records the run."""

    def __init__(self, flight, p: Plan, folder: Path | None = None, sim_url: str | None = None,
                 meta: dict | None = None):
        self.flight = flight
        self.plan = p
        self.folder = folder
        self.sim_url = sim_url
        self.meta = meta or {}
        self.reason: str | None = None
        self.index = -1
        self.started = 0.0
        self.origin = None
        self.frames = 0
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.files: dict = {}

    @property
    def running(self) -> bool:
        return self.started > 0 and not self.stopping.is_set()

    def start(self) -> None:
        model = self.flight.model
        self.saved = (getattr(model, "strategy", None), getattr(model, "speed_index", None))
        if self.sim_url:
            state = self.peek()
            if state:
                self.origin = [*state["position"][:2], state["attitude"][2]]
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "frames").mkdir(exist_ok=True)
            (self.folder / "plan.json").write_text(json.dumps(
                {"text": self.plan.text, **self.plan.json(), "meta": self.meta, "origin": self.origin}, indent=1))
            self.files = {name: open(self.folder / name, "w", buffering=1)
                          for name in ("commands.jsonl", "frames.csv") + (("truth.jsonl",) if self.sim_url else ())}
        model.set_strategy(LinearStrategy())
        if hasattr(model, "set_speed_index"):
            model.set_speed_index(2)
        self.started = time.monotonic()
        threading.Thread(target=self.loop, daemon=True, name="trajectory").start()
        if self.sim_url and self.folder is not None:
            threading.Thread(target=self.watch, daemon=True, name="trajectory-truth").start()

    def clock(self) -> float:
        return time.monotonic() - self.started

    def loop(self) -> None:
        fired: set[int] = set()
        model = self.flight.model
        while not self.stopping.is_set():
            t = self.clock()
            index, seg = self.plan.find(t)
            if seg is None:
                self.stop("done")
                return
            action = None
            if seg.action and index not in fired:
                fired.add(index)
                action = seg.action
                getattr(model, action)()
            self.index = index
            self.flight.set_axes_from("trajectory", *seg.sticks)
            self.write("commands.jsonl", {"t": round(t, 4), "i": index, "sticks": seg.sticks, "action": action})
            self.stopping.wait(TICK)

    def stop(self, reason: str) -> None:
        with self.lock:
            if self.reason is not None:
                return
            self.reason = reason
            self.stopping.set()
        model = self.flight.model
        self.flight.set_axes_from("trajectory", 0.0, 0.0, 0.0, 0.0)
        for axis in AXES:
            setattr(model, axis, model.center_value)
        strategy, speed_index = self.saved
        if strategy is not None:
            model.set_strategy(strategy)
        if speed_index is not None and hasattr(model, "set_speed_index"):
            model.set_speed_index(speed_index)
        if self.folder is not None:
            (self.folder / "result.json").write_text(json.dumps(
                {"reason": reason, "elapsed": round(self.clock(), 3), "frames": self.frames}, indent=1))
            for f in self.files.values():
                f.close()

    def frame(self, jpeg: bytes) -> None:
        if not self.running or self.folder is None:
            return
        n = self.frames
        self.frames += 1
        (self.folder / "frames" / f"{n:05d}.jpg").write_bytes(jpeg)
        self.write("frames.csv", None, f"{n},{self.clock():.4f}\n")

    def write(self, name: str, record: dict | None, line: str | None = None) -> None:
        f = self.files.get(name)
        if f is None or f.closed:
            return
        try:
            f.write(line if line is not None else json.dumps(record) + "\n")
        except ValueError:
            pass

    def peek(self) -> dict | None:
        try:
            with urllib.request.urlopen(f"{self.sim_url}/state", timeout=0.5) as reply:
                return json.load(reply)
        except OSError:
            return None

    def watch(self) -> None:
        while not self.stopping.wait(0.05):
            state = self.peek()
            if state:
                self.write("truth.jsonl", {"t": round(self.clock(), 3), "sim_t": state["t"], "state": state["state"],
                                           "position": state["position"], "velocity": state["velocity"],
                                           "attitude": state["attitude"]})

    def status(self) -> dict:
        seg = self.plan.segments[self.index] if 0 <= self.index < len(self.plan.segments) else None
        return {
            "running": self.running,
            "t": round(self.clock(), 2) if self.started else 0.0,
            "duration": round(self.plan.duration, 2),
            "index": self.index,
            "step": seg.text if seg else None,
            "reason": self.reason,
            "origin": self.origin,
            "folder": str(self.folder) if self.folder else None,
        }
