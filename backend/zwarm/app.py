"""Puts the parts together: Hub, MissionManager, Agent, FindingStore, FleetCapture and Frames, behind HTTP.

server.py's Runtime and build_app stay as they are. Runtime ticks whatever it holds as `co`, and the
routes call it for hold, land and E-stop. So it gets a Front: one flight tick is hub.tick then
missions.tick, and a stop button goes through the agent's halt, which also ends the mission and any
model turn in progress. A plain coordinator hold wouldn't: a running mission would just send new goals.

Runtime and the hub share one lock. Nothing here does network or disk work under it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import statistics
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response

from agent.agent import Agent
from agent.nemotron import NemotronClient, NemotronError
from mapping.fleet import FleetCapture, MapError, tracked_pose_of
from swarm.coordinator import Coordinator, DroneConfig
from swarm.frames import StreamFrames
from swarm.hub import Hub
from swarm.inspect import DATA_ROOT, FindingStore, Inspector
from swarm.links import BackendLink
from swarm.memory import Memory
from swarm.mission import MissionManager
from swarm.server import Runtime, SwarmFile, build_app, real_runtime
from swarm.tracking import DroneOptics, InsideOutTracker, SwarmFileError

logger = logging.getLogger(__name__)

PORT = 8010
EVENTS_KEPT = 2000
EVENTS_PER_ANSWER = 500
FRAME_WAIT_S = 8.0  # how long preflight waits for a first frame from each drone
TICK_S_KEPT = 400  # ticks behind the timing numbers in /zwarm/state, about 20 s
MAX_POINTS = 50000
SAME_BUTTON_S = 0.5


def plain(value, depth: int = 0):
    """JSON-safe, so an odd value in an event's data can't break the event route."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v, depth + 1) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return plain(value.tolist(), depth + 1)
    return str(value)


class EventLog:
    """What the console shows, numbered so it can ask for what it hasn't seen. Memory only, so the flight loop can write to it."""

    def __init__(self, clock=time.monotonic, keep: int = EVENTS_KEPT):
        self.clock = clock
        self.lock = threading.Lock()
        self.items: deque = deque(maxlen=keep)
        self.count = 0

    def add(self, kind: str, text: str, data=None) -> None:
        with self.lock:
            self.count += 1
            self.items.append({"n": self.count, "t": round(float(self.clock()), 2), "kind": kind, "text": str(text), "data": plain(data)})

    def since(self, after: int = 0, limit: int = EVENTS_PER_ANSWER) -> list[dict]:
        with self.lock:
            return [event for event in self.items if event["n"] > after][:limit]


class Front:
    """What Runtime and server.py's routes hold as `co`. Anything not named here goes to the real coordinator."""

    def __init__(self, hub: Hub, missions: MissionManager):
        self.hub, self.missions = hub, missions
        self.agent: Agent | None = None
        self.events: EventLog | None = None
        self.tick_s: deque = deque(maxlen=TICK_S_KEPT)
        self.tick_errors = 0
        self.last_button: str | None = None
        self.last_button_at = 0.0

    def __getattr__(self, name):
        return getattr(self.hub.co, name)

    def tick(self, now: float, poses: dict, moving=()) -> None:
        started = time.perf_counter()
        try:
            self.hub.tick(now, poses, moving)
        except Exception:
            # Runtime's thread would die with this, and then nothing would fly the drones at all.
            self.tick_errors += 1
            logger.exception("[zwarm] a flight tick failed")
        self.missions.tick(now)
        self.tick_s.append(time.perf_counter() - started)

    def _halt(self, verb: str, button: str) -> None:
        # The console sends a stop button to two hostnames at once, and it only needs doing once.
        now = time.monotonic()
        if self.last_button == verb and now - self.last_button_at < SAME_BUTTON_S:
            return
        self.last_button, self.last_button_at = verb, now
        if self.events is not None:
            self.events.add("said", f"{button} button")
        if self.agent is not None:
            self.agent.halt(verb)
        else:
            getattr(self.hub, verb)()

    def hold(self) -> None:
        self._halt("hold", "HOLD")

    def land(self, now=None) -> None:
        self._halt("land", "LAND")

    def land_here(self, now=None) -> None:
        self._halt("land_here", "LAND HERE")

    def estop(self) -> None:
        # The hub first and on its own: nothing below it can be allowed to get in the way of this one.
        self.hub.estop()
        self._halt("estop", "E-STOP")

    def timing(self) -> dict:
        ticks = sorted(self.tick_s)
        if not ticks:
            return {"ticks": 0, "errors": self.tick_errors}
        return {"ticks": len(ticks), "errors": self.tick_errors, "median_ms": round(statistics.median(ticks) * 1000, 2),
                "p99_ms": round(ticks[int(len(ticks) * 0.99)] * 1000, 2), "max_ms": round(ticks[-1] * 1000, 2)}


@dataclass
class Zwarm:
    runtime: Runtime
    front: Front
    hub: Hub
    missions: MissionManager
    agent: Agent
    findings: FindingStore
    capture: object
    frames: object
    events: EventLog
    simulated: bool = False
    anchors: dict = field(default_factory=dict)
    north_yaw_deg: float = 0.0
    fleet: object = None  # the simulated fleet, for the truth only a simulator knows
    world: object = None

    def state(self) -> dict:
        hub = self.hub
        co = hub.co
        with hub.lock:
            mission = self.missions.status()
            flown = mission.get("flight_s", {})
            drones = [{"id": d.id, "phase": d.phase.value, "p": d.tracked.p.round(3).tolist(), "yaw": round(math.degrees(d.tracked.yaw), 1),
                       "sigma": round(float(d.tracked.sigma), 3),
                       "fix_age": round(float(d.tracked.fix_age), 2) if math.isfinite(d.tracked.fix_age) else None,
                       "flight_s": flown.get(d.id, 0.0), "faults": [str(f) for f in d.faults.values()], "connected": bool(d.link.connected),
                       "goal": None if d.goal is None else d.goal.round(3).tolist(), "path": [p.round(3).tolist() for p in d.path]}
                      for d in co.drones.values()]
            room = {"lo": co.room.lo.tolist(), "hi": co.room.hi.tolist(), "north_yaw_deg": self.north_yaw_deg,
                    "obstacles": [{"name": o.name, "lo": list(o.lo), "hi": list(o.hi)} for o in co.room.obstacles.values()],
                    "anchors": [{"id": a.id, "x": a.x, "y": a.y} for a in self.anchors.values()]}
            coverage = self.missions.coverage_map()
            things = [{"id": t.id, "label": t.label, "x": round(float(t.p[0]), 2), "y": round(float(t.p[1]), 2), "z": round(float(t.p[2]), 2),
                       "sure": t.sure, "seen_by": sorted(t.seen_by)} for t in hub.memory.known()]
            out = {"simulated": self.simulated, "t": round(hub.now, 2), "estopped": co.estopped, "mission": mission, "drones": drones, "room": room,
                   "coverage": coverage, "things": things, "flight_limit_s": self.missions.flight_limit_s,
                   "closest": None if not math.isfinite(co.closest) else round(co.closest, 3)}
        out["agent"] = self.agent.state()
        out["pending_plan"] = self.agent.pending()
        out["loop"] = {**self.front.timing(), "overruns": self.runtime.overruns, "mission_errors": self.missions.errors, "hub_errors": hub.think_errors}
        if self.fleet is not None:
            out["sim"] = self._sim_truth()
        return out

    def _sim_truth(self) -> dict:
        fleet = self.fleet
        stats = self.frames.stats() if hasattr(self.frames, "stats") else {}
        return {"collisions": sorted(f"{a}-{b}" for a, b in fleet.collisions), "contacts": sorted(fleet.contacts), "crashed": fleet.crashed(),
                "min_separation": None if not math.isfinite(fleet.min_separation) else round(fleet.min_separation, 3), "camera": plain(stats)}

    def close(self) -> None:
        inspector = self.missions.inspector
        if inspector is not None:
            inspector.close(timeout=1.0)
        self.findings.close()


def _stop_all(*stoppers):
    def close() -> None:
        for stop in stoppers:
            try:
                stop()
            except Exception:
                logger.exception("[zwarm] something didn't shut down cleanly")
    return close


def compose(coordinator, clock, poses, frames, eyes, data_dir: Path, *, advance=None, close=lambda: None, client=None, analyze=None, capture=None,
            tilts=None, anchors=None, north_yaw_deg: float = 0.0, flight_limit_s: float = 270.0, auto_approve: bool = False, fleet=None,
            world=None, runtime: Runtime | None = None, findings_path: Path | None = None) -> Zwarm:
    """Everything on top of a coordinator. Passing client, analyze or capture replaces a live part, which is how tests run with no network."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(clock)
    # In memory because the hub commits from the flight loop, which must never wait on a disk.
    hub = Hub(coordinator, Memory(":memory:"), eyes, report=lambda text: events.add("report", text), tilts=tilts)
    findings = FindingStore(findings_path or data_dir / "findings.sqlite")

    def said_by_map(kind: str, text: str, data=None) -> None:
        events.add("mission", text, {"map": True, "name": (data or {}).get("name")})

    if capture is None:
        capture = FleetCapture(frames, data_dir / "maps", drones=list(coordinator.drones), on_event=said_by_map, pose_of=tracked_pose_of(hub))
    agent_holder: dict = {}

    def said_by_mission(kind: str, text: str, data=None) -> None:
        events.add(kind if kind in ("mission", "finding", "fault", "report") else "mission", text, data)
        agent = agent_holder.get("agent")
        if agent is not None:
            agent.observe(kind, text, data)

    inspector = Inspector(findings, frames, hub=hub, analyze=analyze)
    missions = MissionManager(hub, frames, findings, capture, flight_limit_s, on_event=said_by_mission, north_yaw_deg=north_yaw_deg, inspector=inspector)
    inspector.on_event = lambda kind, text, data: missions.say(kind, text, **data)
    agent = Agent(client if client is not None else NemotronClient(), hub, missions, findings, capture, on_event=events.add, auto_approve=auto_approve)
    agent_holder["agent"] = agent
    front = Front(hub, missions)
    front.agent, front.events = agent, events
    if runtime is None:
        runtime = Runtime(front, clock, poses, advance=advance, close=close)
    else:
        previous_close, runtime.co = runtime.close, front
        runtime.close = lambda: (close(), previous_close())
    runtime.lock = hub.lock  # commands land between ticks, whichever way they come in
    zw = Zwarm(runtime, front, hub, missions, agent, findings, capture, frames, events, simulated=fleet is not None,
               anchors=dict(anchors or {}), north_yaw_deg=north_yaw_deg, fleet=fleet, world=world)
    closing = runtime.close
    runtime.close = _stop_all(closing, zw.close)
    return zw


# ------------------------------------------------------------------------------- HTTP
def build(zw: Zwarm) -> FastAPI:
    """server.py's app, with the /zwarm routes added. server.py keeps /state, /takeoff, /hold, /land, /land-here and /estop."""
    app = build_app(zw.runtime)

    @app.get("/zwarm/state")
    def state():
        return plain(zw.state())

    @app.post("/zwarm/chat")
    def chat(body: dict = Body(...)):
        text = body.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Send {\"text\": \"what you want the fleet to do\"}.")
        return plain(zw.agent.handle(text))

    @app.get("/zwarm/events")
    def events(after: int = 0):
        return zw.events.since(after)

    @app.post("/zwarm/plan/{plan_id}/approve")
    def approve(plan_id: str):
        return plain(zw.agent.approve(plan_id))

    @app.post("/zwarm/plan/{plan_id}/reject")
    def reject(plan_id: str):
        return plain(zw.agent.reject(plan_id))

    @app.get("/zwarm/findings")
    def findings():
        return plain(zw.findings.list())

    @app.post("/zwarm/findings/{finding_id}")
    def set_finding(finding_id: int, body: dict = Body(...)):
        try:
            return plain(zw.findings.set_status(finding_id, str(body.get("status"))))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"There's no finding {finding_id}.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @app.get("/zwarm/findings/{finding_id}/photo")
    def photo(finding_id: int):
        try:
            return Response(zw.findings.photo(finding_id), media_type="image/jpeg")
        except KeyError:
            raise HTTPException(status_code=404, detail=f"There's no photo for finding {finding_id}.")

    @app.get("/zwarm/video/{drone}.jpg")
    def video(drone: str):
        frame = zw.frames.latest(drone)
        if frame is None:
            raise HTTPException(status_code=404, detail=f"No video from drone {drone} yet.")
        return Response(frame.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/zwarm/map/build")
    def build_map():
        try:
            return plain(zw.capture.build())
        except MapError as error:
            raise HTTPException(status_code=409, detail=str(error))
        except ValueError as error:
            raise HTTPException(status_code=409, detail=f"There are no photos to build from yet: {error}")

    @app.get("/zwarm/map")
    def map_status():
        return plain(zw.capture.status())

    @app.get("/zwarm/map/points")
    def map_points(limit: int = MAX_POINTS):
        # Encoded in one go. FastAPI's own encoder would hold the GIL for a second on a full cloud.
        return Response(json.dumps(zw.capture.points(min(max(limit, 0), MAX_POINTS))), media_type="application/json")

    return app


# ------------------------------------------------------------------------------- the simulator
def sim(seed: int = 1, data_dir: Path | None = None, rate_hz: float = 5.0, inside_out: bool = False, **parts) -> Zwarm:
    """The demo room with a simulated fleet, rendered cameras and a simulated detector. Numbers from here are simulator numbers."""
    from swarm.eyes import SimEyes
    from swarm.sim_cameras import SimFrames, demo_world
    from swarm.sim_fleet import SimFleet

    world = demo_world()
    fleet = SimFleet(world.scene, world.pads, seed)
    configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in world.pads.items()]
    frames = parts.pop("frames", None) or SimFrames(fleet, rate_hz=rate_hz)
    eyes = SimEyes(fleet, world.items, seed=seed)
    data_dir = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix=f"zwarm-sim-{time.strftime('%H%M%S')}-"))
    tracker = None
    if inside_out:
        optics = {c.id: DroneOptics(c.pad[0], c.pad[1], c.yaw) for c in configs}
        tracker = InsideOutTracker(optics, world.anchors, world.fence)
    coordinator = Coordinator(world.room(), configs, fleet.links, tracker=tracker)
    poses = (lambda: tracker.poses(fleet.t)) if tracker else fleet.poses
    stoppers = [frames.stop] if hasattr(frames, "stop") else []
    if tracker:
        stoppers.append(tracker.stop)
    zw = compose(coordinator, lambda: fleet.t, poses, frames, eyes, data_dir, advance=fleet.advance, close=_stop_all(*stoppers), anchors=world.anchors,
                 fleet=fleet, world=world, **parts)
    if hasattr(frames, "start"):
        frames.start()
    if tracker:
        tracker.start(frames)
    return zw


# ------------------------------------------------------------------------------- real drones
@dataclass
class Readiness:
    config: SwarmFile | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.problems


def load_swarm_file(path) -> tuple[SwarmFile | None, str | None]:
    try:
        return SwarmFile.load(path), None
    except SwarmFileError as error:
        return None, str(error)
    except FileNotFoundError:
        return None, f"There's no swarm file at {path}."
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"The swarm file {path} can't be read: {error!r}."


def check_ready(path, need_model: bool = True, frame_wait_s: float = FRAME_WAIT_S) -> Readiness:
    """Everything that has to be true before fly. Reads capabilities and video, and sends no flight command."""
    out = Readiness()
    config, problem = load_swarm_file(path)
    if config is None:
        out.problems.append(problem)
        return out
    out.config = config
    if not config.anchors:
        out.problems.append("The swarm file lists no floor anchors, and they are the only source of position.")
    else:
        tracker = InsideOutTracker(config.optics(), config.anchors, (config.room.lo, config.room.hi), config.video_latency_s)
        for d in config.drones:
            if not tracker.expected(d.id):
                out.problems.append(f"Drone {d.id} would see no floor anchor hovering over its pad at 1 m. Move the pad or add tags.")
    if not all(d.focal_px for d in config.drones):
        out.warnings.append("No fitted focal_px for every drone, so positions rest on a guessed field of view. Run `python -m swarm fit-focal`.")
    if need_model:
        try:
            NemotronClient()._api_key()
        except NemotronError:
            out.problems.append("NVIDIA_API_KEY isn't set in backend/.env, so the planner can't run. Pass --no-model to fly on fixed phrases only.")
    try:
        from ultralytics import YOLO

        from swarm.yolo_eyes import weights_path
        YOLO(weights_path())
        out.notes.append("YOLO detector loaded.")
    except Exception as error:
        out.problems.append(f"The YOLO detector won't load ({error}). Run `uv sync --inexact --extra follow` and check yolov10n.pt or YOLO_WEIGHTS.")
    if shutil.which("colmap") is None and not os.environ.get("MAP_BUILD_SSH"):
        out.warnings.append("COLMAP isn't on the PATH and MAP_BUILD_SSH isn't set, so a map can be captured but not built. `brew install colmap`.")
    for d in config.drones:
        link = BackendLink(d.backend, d.id)
        link.inspect()
        if link.problem:
            out.problems.append(f"Drone {d.id} at {d.backend}: {link.problem}. Is its Pi up? `tools/pi/fleet.sh status`.")
    frames = StreamFrames(config.backends())
    frames.start()
    deadline = time.monotonic() + frame_wait_s
    try:
        while time.monotonic() < deadline and not all(frames.latest(d.id) for d in config.drones):
            time.sleep(0.1)
        for d in config.drones:
            frame = frames.latest(d.id)
            if frame is None:
                out.problems.append(f"Drone {d.id} at {d.backend}: no video in {frame_wait_s:.0f} s from /mjpeg.")
            else:
                out.notes.append(f"Drone {d.id}: video arriving, {len(frame.jpeg)} byte frames.")
    finally:
        threading.Thread(target=frames.stop, daemon=True).start()  # a stuck read has a 5 s timeout, and no one needs to wait for it
    return out


def fly(path, data_dir: Path | None = None, need_model: bool = True, north_yaw_deg: float | None = None, **parts) -> tuple[Zwarm | None, Readiness]:
    """Real drones from a swarm file. Returns (None, why) if anything needed isn't there: this never starts half-working."""
    from swarm.yolo_eyes import YoloEyes

    readiness = check_ready(path, need_model)
    if not readiness.ready:
        return None, readiness
    config = readiness.config
    raw = json.loads(Path(path).read_text())
    runtime = real_runtime(config)
    eyes = YoloEyes(list(config.backends()), runtime.frames)
    eyes.start()
    north = float(raw.get("north_yaw_deg", 0.0)) if north_yaw_deg is None else north_yaw_deg
    zw = compose(runtime.co, runtime.clock, runtime.get_poses, runtime.frames, eyes, Path(data_dir or DATA_ROOT) / "flights" / time.strftime("%Y%m%d-%H%M%S"),
                 close=eyes.stop, tilts={d.id: math.radians(d.camera_tilt_deg) for d in config.drones}, anchors=config.anchors, north_yaw_deg=north,
                 flight_limit_s=float(raw.get("flight_limit_s", 270.0)), runtime=runtime, findings_path=Path(data_dir or DATA_ROOT) / "findings.sqlite", **parts)
    return zw, readiness

