"""Fit the simulator to calibration flights flown on the real drone.

    uv run python -m sim.calibrate                 every run in backend/runs with measurements
    uv run python -m sim.calibrate runs/20261107-101500 --flight-min 11.5
    uv run python -m sim.calibrate --dry-run        report without saving

A run is a trajectory the web console flew and recorded. Its measured.json holds what was
measured by hand (hover height, final height, landing spot); spin runs are measured from
their video. Each run's recorded commands are replayed through the simulator, and each
unknown is searched until the simulator matches. Results go to sim/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2

from control.strategies import LinearStrategy
from control.trajectory import AXES, NEUTRAL, wrap
from sim import video
from sim.config import CALIBRATION, DEFAULT_FIRMWARE, DroneSpec, FlightTuning, fitted
from sim.scene import Scene
from sim.simulator import Simulator

RUNS = Path(__file__).resolve().parents[1] / "runs"
MEASURES = ("hover_height", "final_height", "landed_x", "landed_y", "yaw_deg")
UNKNOWNS = (  # what to fit, what pins it down, and the range a toy drone could have
    ("takeoff_height_m", ("hover_height",), (0.4, 2.5)),
    ("max_climb_mps", ("final_height",), (0.3, 4.0)),
    ("max_descent_mps", ("final_height",), (0.3, 3.0)),
    ("max_speed_mps", ("landed_x", "landed_y"), (0.5, 12.0)),
    ("max_yaw_rate_rad_s", ("yaw_deg",), (0.5, 8.0)),
)


@dataclass
class Run:
    name: str
    segments: list
    commands: list
    measured: dict
    frames: Path | None = None
    times: list = field(default_factory=list)

    def window(self) -> tuple[float, float]:
        """When the takeoff ends and the landing starts."""
        up = next((s["t"] + s["dur"] for s in self.segments if s["action"] == "takeoff"), 0.0)
        down = next((s["t"] for s in self.segments if s["action"] == "land"), self.commands[-1]["t"])
        return up, down

    def uses(self, axis: str, sign: float) -> bool:
        index = AXES.index(axis)
        return any(c["sticks"][index] * sign > 0 for c in self.commands)


def load(folder: Path) -> Run | None:
    try:
        plan = json.loads((folder / "plan.json").read_text())
        commands = [json.loads(line) for line in (folder / "commands.jsonl").read_text().splitlines() if line]
    except (OSError, ValueError):
        return None
    if not commands:
        return None
    try:
        measured = json.loads((folder / "measured.json").read_text())
    except (OSError, ValueError):
        measured = {}
    run = Run(folder.name, plan["segments"], commands, {k: v for k, v in measured.items() if k in MEASURES})
    frames = folder / "frames.csv"
    if frames.exists():
        rows = [line.split(",") for line in frames.read_text().splitlines() if line]
        run.frames, run.times = folder / "frames", [float(t) for _, t in rows]
    return run


def replay(run: Run, tuning: FlightTuning, spec: DroneSpec | None = None, seed: int = 1) -> list[list[float]]:
    """Fly a run's recorded commands through the simulator: [t, x, y, height, heading] in its own frame."""
    from models.s2x_rc import S2xDroneModel
    from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter

    sim = Simulator(firmware=DEFAULT_FIRMWARE, scene=Scene(name="open"), seed=seed,
                    spec=spec or DroneSpec.calibrated(), tuning=tuning)
    model = S2xDroneModel()
    model.set_strategy(LinearStrategy())
    rc = S2xRCProtocolAdapter("127.0.0.1")
    rc.sock.close()
    client = ("172.16.10.2", 50000)
    period = 1 / 80
    current, due, i = NEUTRAL, 0.0, 0
    while sim.t < sim.tuning.boot_s + 0.5:
        if sim.t >= due:
            model.update(period, dict(zip(AXES, NEUTRAL)))
            sim.receive_udp(client, rc.build_control_packet(model))
            due += period
        sim.step(1 / 400)
    begin, heading, last = sim.t, 0.0, sim.drone.yaw()
    track = []
    end = run.commands[-1]["t"] + 6.0
    while sim.t - begin < end:
        t = sim.t - begin
        while i < len(run.commands) and run.commands[i]["t"] <= t:
            command = run.commands[i]
            current = tuple(command["sticks"])
            if command["action"]:
                getattr(model, command["action"])()
            i += 1
        if i >= len(run.commands) and t > run.commands[-1]["t"]:
            current = NEUTRAL
        if sim.t >= due:
            model.update(period, dict(zip(AXES, current)))
            sim.receive_udp(client, rc.build_control_packet(model))
            due += period
        sim.step(1 / 400)
        d = sim.drone
        heading += wrap(math.degrees(d.yaw() - last))
        last = d.yaw()
        if not track or t - track[-1][0] >= 0.05:
            track.append([t, d.x, d.y, d.z - sim.spec.gear_height_m, heading])
    return track


def observe(run: Run, track: list[list[float]]) -> dict:
    """What the measurements would read on the simulated flight."""
    up, down = run.window()
    at = lambda t: min(track, key=lambda p: abs(p[0] - t))
    return {
        "hover_height": at(up)[3],
        "final_height": at(down - 0.1)[3],
        "landed_x": track[-1][1],
        "landed_y": track[-1][2],
        "yaw_deg": at(down - 0.1)[4] - at(up)[4],
    }


def error(runs: list[Run], tuning: FlightTuning, keys: tuple, spec: DroneSpec | None = None) -> float:
    total = 0.0
    for run in runs:
        sim = observe(run, replay(run, tuning, spec))
        for key in keys:
            if key in run.measured:
                scale = 30.0 if key == "yaw_deg" else 0.1  # a 30 degree miss weighs like a 10 cm one
                total += ((sim[key] - run.measured[key]) / scale) ** 2
    return total


def search(cost, low: float, high: float, steps: int = 16) -> float:
    """Golden-section search for the minimum of a function on [low, high]."""
    g = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = cost(d)
    return (a + b) / 2


def fit(runs: list[Run], tuning: FlightTuning, spec: DroneSpec | None = None, log=print) -> FlightTuning:
    for name, keys, (low, high) in UNKNOWNS:
        chosen = [r for r in runs if any(k in r.measured for k in keys)]
        if name == "max_climb_mps":
            chosen = [r for r in chosen if r.uses("throttle", +1)]
        if name == "max_descent_mps":
            chosen = [r for r in chosen if r.uses("throttle", -1)]
        if name == "max_yaw_rate_rad_s":
            chosen = [r for r in chosen if r.uses("yaw", +1) or r.uses("yaw", -1)]
        if name == "max_speed_mps":
            chosen = [r for r in chosen if r.uses("pitch", +1) or r.uses("pitch", -1)
                      or r.uses("roll", +1) or r.uses("roll", -1)]
        if not chosen:
            continue
        value = search(lambda v: error(chosen, replace(tuning, **{name: v}), keys, spec), low, high)
        log(f"fit {name:20s} {getattr(tuning, name):7.3f} -> {value:7.3f}  from {', '.join(r.name for r in chosen)}")
        tuning = replace(tuning, **{name: value})
    return tuning


def endurance(spec: DroneSpec, tuning: FlightTuning) -> float:
    """Minutes the simulated drone hovers on a full battery before it lands itself."""
    from sim.pilot import FLY, Pilot

    pilot = Pilot(scene=Scene(name="open"), step_hz=100, spec=spec, tuning=tuning)
    pilot.fly(2.5)
    pilot.hold(FLY, 0.3)
    pilot.fly(5.0)
    started = pilot.sim.t
    while pilot.fc.state in ("CLIMB", "FLYING") and pilot.sim.t - started < 3600:
        pilot.fly(5.0)
    return (pilot.sim.t - started) / 60


def spin(run: Run) -> dict | None:
    if run.frames is None or not (run.uses("yaw", +1) or run.uses("yaw", -1)):
        return None
    images = [cv2.imread(str(run.frames / f"{i:05d}.jpg")) for i in range(len(run.times))]
    pairs = [(img, t) for img, t in zip(images, run.times) if img is not None]
    if len(pairs) < 30:
        return None
    return video.spin([img for img, _ in pairs], [t for _, t in pairs])


def calibrate(folders: list[Path] | None = None, flight_min: float | None = None, save: bool = True) -> list[str]:
    """Fit every unknown the runs can pin down. Returns the report, line by line."""
    lines: list[str] = []
    folders = folders or sorted(p for p in RUNS.glob("*") if p.is_dir())
    runs = [r for r in (load(f) for f in folders) if r is not None]
    values = dict(fitted())
    spec = replace(DroneSpec(), **{k: v for k, v in values.items() if hasattr(DroneSpec(), k)})
    tuning = replace(FlightTuning(), **{k: v for k, v in values.items() if hasattr(FlightTuning(), k)})

    fovs = []
    for run in runs:
        result = spin(run)
        if result is None and run.frames is not None and (run.uses("yaw", +1) or run.uses("yaw", -1)):
            lines.append(f"video {run.name}: no reliable full turn. Spin again from the middle of a room, "
                         "at least 1.5 m from walls, with things in view all around.")
        if result:
            fovs.append(result["fov_deg"])
            run.measured.setdefault("yaw_deg", result["yaw_deg"])
            lines.append(f"video {run.name}: turned {result['yaw_deg']:.1f} deg, field of view "
                         f"{result['fov_deg']:.1f} deg ({result['agreement']} full-turn matches agree)")
    if fovs:
        spec = replace(spec, camera_fov_deg=sum(fovs) / len(fovs))
    measured = [r for r in runs if r.measured]
    if not measured and flight_min is None and not fovs:
        return ["Nothing to fit: fly a calibration program from the web console and enter its measurements."]

    before = {r.name: observe(r, replay(r, tuning, spec)) for r in measured}
    tuning = fit(measured, tuning, spec, log=lines.append)
    if flight_min is not None:
        minutes = endurance(spec, tuning)
        for _ in range(8):
            if abs(minutes - flight_min) < 0.01 * flight_min:
                break
            spec = replace(spec, rated_flight_min=spec.rated_flight_min * flight_min / minutes)
            minutes = endurance(spec, tuning)
        lines.append(f"fit rated_flight_min      -> {spec.rated_flight_min:7.2f}  (simulated hover now {minutes:.2f} min)")

    lines += ["", "run              measure          real  sim before  sim after"]
    for run in measured:
        after = observe(run, replay(run, tuning, spec))
        for key, real in run.measured.items():
            lines.append(f"{run.name:16s} {key:13s} {real:7.2f}  {before[run.name][key]:9.2f}  {after[key]:9.2f}")

    changed = {}
    for name in FlightTuning.__dataclass_fields__:
        if getattr(tuning, name) != getattr(FlightTuning(), name):
            changed[name] = round(getattr(tuning, name), 4)
    for name in ("camera_fov_deg", "rated_flight_min"):
        if getattr(spec, name) != getattr(DroneSpec(), name):
            changed[name] = round(getattr(spec, name), 4)
    if not save:
        return lines + ["", "(dry run: nothing saved)"]
    CALIBRATION.write_text(json.dumps({"values": changed, "runs": [r.name for r in runs],
                                       "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\n")
    return lines + ["", f"Saved {len(changed)} values to sim/calibration.json. Power-cycle the simulator to use them."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: all in backend/runs)")
    parser.add_argument("--flight-min", type=float, help="measured hover time on a full battery, minutes")
    parser.add_argument("--dry-run", action="store_true", help="report, don't save")
    args = parser.parse_args()
    print("\n".join(calibrate(args.runs or None, args.flight_min, not args.dry_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
