"""The drift benchmark: a fixed set of flights, flown with and without the stabilizer.

    uv run python -m sim.stability                every case, scored against the pass bars
    uv run python -m sim.stability --quick        a small sample
    uv run python -m sim.stability --json out.json

Each flight uses the real backend path: FlightController, the S2x model and packet builder, and
rendered, JPEG-compressed frames that reach the stabilizer late, as they would over Wi-Fi.

This measures the simulator, whose flight controller is an estimate. Good numbers here show that
the control law works against that estimate. They don't show how much a real drone will improve.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import cv2

from control.stabilizer import Gains, Stabilizer, decode_gray
from control.strategies import IncrementalStrategy, LinearStrategy
from models.s2x_rc import S2xDroneModel
from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from services.flight_controller import FlightController
from sim.camera import RATE, Camera, Pose
from sim.config import DroneSpec, FlightTuning
from sim.scene import Scene
from sim.simulator import Simulator
from slam.sequences import JPEG_QUALITY, hall, textured

CLIENT = ("172.16.10.2", 50000)
STEP = 1 / 400
CONTROL = 1 / 80
FLOW_LOST = bytes.fromhex("770e00000000000000000000" "0099")
BLANK = {"wall_material": "wall", "ceiling_material": "ceiling", "floor_material": "wall"}


def scene_named(name: str) -> Scene:
    if name == "hall":
        return hall()
    if name == "blank":
        return replace(hall(), boxes=(), **BLANK)
    return textured(name)


# A manoeuvre is a list of (seconds, throttle, yaw, pitch, roll). The drift is scored from `measure_from`.
MANOEUVRES = {
    "hover": ([(20.0, 0, 0, 0, 0)], 0),
    "stop_forward": ([(1.5, 0, 0, 0.4, 0), (6.0, 0, 0, 0, 0)], 1),
    "stop_strafe": ([(1.5, 0, 0, 0, 0.4), (6.0, 0, 0, 0, 0)], 1),
    "stop_keyboard": ([(0.8, 0, 0, 1, 0), (6.0, 0, 0, 0, 0)], 1),
    "turn": ([(4.8, 0, 0.5, 0, 0), (2.0, 0, 0, 0, 0)], 0),
    "turn_fast": ([(2.7, 0, 0.9, 0, 0), (2.0, 0, 0, 0, 0)], 0),
    "settled_turn": ([(5.0, 0, 0, 0, 0), (4.8, 0, 0.5, 0, 0), (2.0, 0, 0, 0, 0)], 1),
    "wall_hold": ([(15.0, 0, 0, 0, 0)], 0),
}


@dataclass(frozen=True)
class Case:
    manoeuvre: str
    scene: str = "hall"
    flow: bool = False
    wind: float = 0.0
    seed: int = 1
    delay_s: float = 0.15
    hfov: float = 90.0  # what the stabilizer assumes
    camera_fov: float = 90.0  # what the drone has
    max_tilt_deg: float = 25.0
    blind_after: float | None = None  # the video stops this long after the flight starts

    @property
    def label(self) -> str:
        parts = [self.manoeuvre, self.scene, "flow" if self.flow else "noflow", f"wind{self.wind:g}", f"seed{self.seed}"]
        if self.delay_s != 0.15:
            parts.append(f"delay{self.delay_s:g}")
        if self.camera_fov != self.hfov:
            parts.append(f"fov{self.camera_fov:g}")
        if self.max_tilt_deg != 25.0:
            parts.append(f"tilt{self.max_tilt_deg:g}")
        if self.blind_after is not None:
            parts.append("blind")
        return "/".join(parts)


class SimLink(S2xRCProtocolAdapter):
    """The real packet builder, sending into a simulator instead of a socket."""

    def __init__(self, sim: Simulator):
        super().__init__("127.0.0.1")
        self.sock.close()
        self.sim = sim

    def send_control_packet(self, packet) -> None:
        self.sim.receive_udp(CLIENT, bytes(packet))


def fly(case: Case, stabilized: bool, gains: Gains | None = None) -> dict:
    scene = scene_named(case.scene)
    tuning = replace(FlightTuning.calibrated(), max_tilt_rad=math.radians(case.max_tilt_deg))
    spec = replace(DroneSpec.calibrated(), camera_fov_deg=case.camera_fov)
    sim = Simulator(scene=scene, seed=case.seed, tuning=tuning, spec=spec)
    sim.place(*(wall_spot(scene) if case.manoeuvre == "wall_hold" else open_spot(scene)))
    if not case.flow:
        # research: a covered sensor still sends its report, with all zeros. Rugs and furniture would
        # otherwise give the flight controller a floor to lock on to, and this case is about losing it.
        sim.board._flow_frame = lambda: FLOW_LOST
    sim.wind.set(case.wind, 35.0, case.wind * 0.4)
    camera = Camera(scene, seed=case.seed, fov=math.radians(case.camera_fov)) if stabilized else None
    stabilizer = Stabilizer(gains, hfov_deg=case.hfov) if stabilized else None
    model = S2xDroneModel()
    keyboard = case.manoeuvre == "stop_keyboard"
    model.set_strategy(IncrementalStrategy() if keyboard else LinearStrategy())
    flight = FlightController(model, SimLink(sim), stabilizer=stabilizer)
    flight.last_control_source = "frontend"
    frames: list[tuple[float, int, bytes]] = []
    rng = random.Random(case.seed)
    count = [0]
    drone = sim.drone
    clock = {"control": 0.0, "video": 0.0}
    blind_at = [math.inf]
    peak_stick = [0.0]
    worst_correction = {"after_blind": 0.0}

    def run(seconds: float, *sticks) -> None:
        flight.set_axes_from("frontend", *sticks)
        end = sim.t + seconds
        while sim.t < end:
            t = sim.t
            if t >= clock["control"]:
                flight.tick(CONTROL, t)
                clock["control"] += CONTROL
                if stabilizer is not None:
                    command = max(abs(stabilizer.forward.command), abs(stabilizer.lateral.command))
                    peak_stick[0] = max(peak_stick[0], command)
                    # The backend can't know the video stopped until the last frame has arrived.
                    if t > blind_at[0] + case.delay_s + 0.5:
                        add = stabilizer.correction(t, 0.0, 0.0, 0.0, 0.0)
                        worst_correction["after_blind"] = max(worst_correction["after_blind"], abs(add[0]), abs(add[1]))
            if camera is not None:
                if t >= clock["video"]:
                    clock["video"] += 1 / RATE
                    if t < blind_at[0]:
                        image = camera.render(Pose(drone.x, drone.y, drone.z, drone.rotation()))
                        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        jitter = rng.uniform(-0.02, 0.02)
                        count[0] += 1
                        heapq.heappush(frames, (t + case.delay_s + jitter, count[0], jpeg.tobytes()))
                while frames and frames[0][0] <= t:
                    arrived, _, data = heapq.heappop(frames)
                    stabilizer.observe(decode_gray(data), arrived)
            sim.step(STEP)

    run(2.5, 0.0, 0.0, 0.0, 0.0)
    model.takeoff()
    run(6.0, 0.0, 0.0, 0.0, 0.0)
    if sim.fc.state != "FLYING":
        return {"label": case.label, "failed": f"never reached a hover: {sim.fc.state}"}

    start_impacts = sum(1 for _, m in sim.fc.events if "Bumped" in m or "Crashed" in m)
    steps, measure_from = MANOEUVRES[case.manoeuvre]
    if case.blind_after is not None:
        blind_at[0] = sim.t + case.blind_after
    anchor = (drone.x, drone.y)
    peak = 0.0
    speed_end = 0.0
    for index, (seconds, throttle, yaw, pitch, roll) in enumerate(steps):
        if index == measure_from:
            anchor = (drone.x, drone.y)
            peak = 0.0
        end = sim.t + seconds
        while sim.t < end:
            run(min(0.25, end - sim.t), float(throttle), float(yaw), float(pitch), float(roll))
            peak = max(peak, math.hypot(drone.x - anchor[0], drone.y - anchor[1]))
    speed_end = math.hypot(drone.vx, drone.vy)
    impacts = [m for _, m in sim.fc.events if "Bumped" in m or "Crashed" in m][start_impacts:]
    return {
        "label": case.label,
        "drift": round(math.hypot(drone.x - anchor[0], drone.y - anchor[1]), 3),
        "peak": round(peak, 3),
        "speed_end": round(speed_end, 3),
        "impacts": len(impacts),
        "crashed": sim.fc.state == "CRASHED",
        "fault": stabilizer.fault if stabilizer else None,
        "gain": round(stabilizer.gain_scale, 2) if stabilizer else None,
        "state": stabilizer.state if stabilizer else None,
        "lost": stabilizer.lost if stabilizer else 0,
        "frames": stabilizer.frames if stabilizer else 0,
        "peak_stick": round(peak_stick[0], 3),
        "after_blind": round(worst_correction["after_blind"], 3),
    }


def wall_spot(scene: Scene) -> tuple[float, float, float]:
    """A spot about 1 m from a wall, facing it."""
    return scene.bounds.hi[0] - 1.1, 0.0, 0.0


def open_spot(scene: Scene) -> tuple[float, float, float]:
    """The middle of the room, so a flight has space before it can bump anything."""
    lo, hi = scene.bounds.lo, scene.bounds.hi
    return (lo[0] + hi[0]) / 2 - 1.0, (lo[1] + hi[1]) / 2, 0.0


def fixed_set(quick: bool = False) -> list[Case]:
    seeds = (1, 2) if quick else (1, 2, 3)
    scenes = ("hall",) if quick else ("hall", "room", "garage", "gym")
    winds = (0.0, 0.5)
    cases = []
    for manoeuvre in MANOEUVRES:
        for scene in scenes:
            for flow in (False, True):
                for wind in winds:
                    for seed in seeds:
                        cases.append(Case(manoeuvre, scene, flow, wind, seed))
    return cases


def robustness_set() -> list[Case]:
    """Assumptions the real drone could break. These fly stabilized only and are judged for harm, not gain."""
    cases = []
    for manoeuvre in ("hover", "stop_forward", "turn"):
        for seed in (1, 2):
            for kw in ({"delay_s": 0.3}, {"delay_s": 0.45}, {"camera_fov": 70.0}, {"camera_fov": 110.0},
                       {"max_tilt_deg": 15.0}, {"max_tilt_deg": 35.0}):
                cases.append(Case(manoeuvre, "hall", False, 0.5, seed, **kw))
    for seed in (1, 2):
        cases.append(Case("hover", "blank", False, 0.0, seed))
        cases.append(Case("hover", "hall", False, 0.5, seed, blind_after=5.0))
    return cases


def _pair(case: Case) -> tuple[dict, dict]:
    return fly(case, False), fly(case, True)


def _solo(case: Case) -> dict:
    return fly(case, True)


def median(values) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def reduction(before: float, after: float) -> float:
    return 100.0 * (1 - after / before) if before > 1e-6 else 0.0


def score(pairs: list[tuple[dict, dict]], robust: list[dict]) -> tuple[list[str], bool]:
    lines: list[str] = []
    ok = True

    def group(name: str, want, floor: float | None) -> None:
        nonlocal ok
        picked = [(b, a) for b, a in pairs if want(b["label"])]
        if not picked:
            return
        before, after = median(b["drift"] for b, _ in picked), median(a["drift"] for _, a in picked)
        worst_b, worst_a = max(b["drift"] for b, _ in picked), max(a["drift"] for _, a in picked)
        cut = reduction(before, after)
        passed = floor is None or cut >= floor
        ok &= passed
        bar = "report" if floor is None else f"bar {floor:.0f}%"
        lines.append(f"{'PASS' if passed else 'FAIL'}  {name:27} median {before:5.2f} -> {after:5.2f} m ({cut:+4.0f}%)  "
                     f"worst {worst_b:5.2f} -> {worst_a:5.2f} m  n={len(picked):3}  {bar}")

    no_flow = lambda label: "/noflow/" in label
    group("hover, no flow", lambda s: s.startswith("hover/") and no_flow(s), 70)
    group("stop, no flow", lambda s: s.startswith("stop_") and no_flow(s), 50)
    group("turn right after takeoff", lambda s: s.startswith("turn") and no_flow(s), 50)
    group("turn after 5 s of hover", lambda s: s.startswith("settled_turn") and no_flow(s), 50)
    group("near a wall, no flow", lambda s: s.startswith("wall_hold/") and no_flow(s), 50)
    group("everything, flow working", lambda s: "/flow/" in s, None)

    regressed = [(b, a) for b, a in pairs if "/flow/" in b["label"] and a["drift"] > b["drift"] + 0.25]
    passed = not regressed
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  flow working: {len(regressed)} flights ended over 0.25 m worse")
    for b, a in regressed[:6]:
        lines.append(f"      {b['label']}: {b['drift']:.2f} -> {a['drift']:.2f} m")

    hits_b = sum(b["impacts"] for b, _ in pairs)
    hits_a = sum(a["impacts"] for _, a in pairs)
    crash_b, crash_a = sum(b["crashed"] for b, _ in pairs), sum(a["crashed"] for _, a in pairs)
    passed = hits_a <= hits_b and crash_a <= crash_b
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  impacts {hits_b} -> {hits_a}, crashes {crash_b} -> {crash_a}")
    faults = [a["label"] for _, a in pairs if a.get("fault")]
    passed = not faults
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  stabilizer gave up on {len(faults)} of {len(pairs)} flights")
    for label in faults[:6]:
        lines.append(f"      {label}")

    if robust:
        lines.append("")
        lines.append("Broken assumptions, stabilized only (no flow, 0.5 m/s wind):")
        kinds: dict[str, list[dict]] = {}
        for r in robust:
            parts = r["label"].split("/")
            kind = "/".join(parts[5:]) or parts[1]
            kinds.setdefault(kind, []).append(r)
        for kind, rs in kinds.items():
            drift = max(r.get("drift", 99) for r in rs)
            crashes = sum(bool(r.get("crashed")) for r in rs)
            faults = sum(bool(r.get("fault")) for r in rs)
            gain = min(r.get("gain") or 1 for r in rs)
            passed = crashes == 0
            ok &= passed
            lines.append(f"{'PASS' if passed else 'FAIL'}  {kind:14} worst drift {drift:5.2f} m  crashes {crashes}  "
                         f"gave up {faults}  lowest gain {gain:.2f}  n={len(rs)}")
        blind = [r for r in robust if r["label"].endswith("blind")]
        if blind:
            leak = max(r["after_blind"] for r in blind)
            passed = leak == 0.0
            ok &= passed
            lines.append(f"{'PASS' if passed else 'FAIL'}  largest correction 0.5 s after the video stops: {leak:.3f}")
    return lines, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--json", help="write every flight's result here")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cases = fixed_set(args.quick)
    extra = [] if args.no_robust else robustness_set()
    with ProcessPoolExecutor(args.workers) as pool:
        pairs = list(pool.map(_pair, cases, chunksize=2))
        robust = list(pool.map(_solo, extra, chunksize=2))
    failed = [r for pair in pairs for r in pair if "failed" in r] + [r for r in robust if "failed" in r]
    pairs = [p for p in pairs if "failed" not in p[0] and "failed" not in p[1]]
    robust = [r for r in robust if "failed" not in r]
    lines, ok = score(pairs, robust)
    print(f"{len(pairs)} flight pairs, {len(robust)} robustness flights, {len(failed)} never reached a hover")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"pairs": pairs, "robust": robust, "failed": failed,
                       "cases": [asdict(c) for c in cases + extra]}, f, indent=1)
    return 0 if ok and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
