"""The swarm's operator API: a loop that ticks the coordinator, and HTTP around it.

Works the same with simulated drones or real ones. The frontend's /swarm page talks to this.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import signal
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sim.scene import Box, Scene
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.room import Obstacle, Room
from swarm.safety import Limits
from swarm.sim_fleet import Conditions, SimFleet
from swarm.tracking import (
    DEFAULT_TILT_DEG,
    Anchor,
    DroneOptics,
    InsideOutTracker,
    Lens,
    anchors_in_view,
    anchors_section,
    load_anchors,
    tag_grid,
)

logger = logging.getLogger(__name__)

TICK_S = 0.05
SHUTDOWN_LAND_S = 25.0
# A page from one of these can use the API. The console runs on this Mac or a machine on its network.
# Anything else in the same browser, such as a public website in another tab, mustn't be able to fly the fleet.
LOCAL_ORIGIN = r"^https?://(localhost|\[::1\]|127(\.\d+){3}|10(\.\d+){3}|192\.168(\.\d+){2}|172\.(1[6-9]|2\d|3[01])(\.\d+){2}|[\w-]+\.local|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])(\.\d+){2})(:\d+)?$"


class Runtime:
    """Owns the loop, and takes the lock for every coordinator call so commands land between ticks."""

    def __init__(self, coordinator: Coordinator, clock, poses, moving=lambda now: [], advance=None, close=lambda: None):
        self.co = coordinator
        self.clock = clock
        self.get_poses = poses
        self.get_moving = moving
        self.advance = advance  # moves a simulated world along between ticks
        self.close = close
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="swarm-loop")
        self.overruns = 0
        self.errors = 0
        self.land_by: float | None = None  # when the shutdown landing gives up
        self.closing = threading.Lock()
        self.closed = False

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        due = time.monotonic()
        while not self.stopping.is_set():
            try:
                with self.lock:
                    if self.advance is not None:
                        self.advance(TICK_S)
                    now = self.clock()
                    self.co.tick(now, self.get_poses(), self.get_moving(now))
            except Exception:
                # If this thread died, nothing would fly the drones at all, and no stop button would do anything.
                self.errors += 1
                if self.errors == 1 or self.errors % 200 == 0:
                    logger.exception("[swarm] a flight tick failed (%d so far)", self.errors)
            due += TICK_S
            late = time.monotonic() - due
            if late > TICK_S:
                self.overruns += 1
                due = time.monotonic()
            elif late < 0:
                self.stopping.wait(-late)

    def airborne(self) -> bool:
        return any(d.phase in (Phase.TAKEOFF, Phase.CHECK, Phase.AIR, Phase.LANDING) for d in self.co.drones.values())

    def land_everything(self) -> None:
        """Land anything in the air and wait for it. The loop and the HTTP routes keep running, so E-STOP still works."""
        with self.lock:
            first = self.land_by is None
            if first:
                self.land_by = time.monotonic() + SHUTDOWN_LAND_S
            flying = self.airborne()
            if first and flying:
                logger.warning("[swarm] shutting down with drones in the air: landing them where they are")
                self.co.land_here()
        deadline = time.monotonic() + SHUTDOWN_LAND_S
        while flying and time.monotonic() < deadline:
            time.sleep(0.2)
            with self.lock:
                flying = self.airborne()
        self.stopping.set()
        self.thread.join(timeout=2.0)
        self.close()


def build_app(runtime: Runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        yield
        runtime.shut_down()

    app = FastAPI(title="Swarm", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def answer(problems: list[str]) -> dict:
        if problems:
            raise HTTPException(status_code=409, detail=problems)
        return {"ok": True}

    @app.get("/state")
    def state():
        with runtime.lock:
            return {**runtime.co.json(), "overruns": runtime.overruns, "simulated": runtime.advance is not None}

    @app.post("/takeoff")
    def takeoff():
        with runtime.lock:
            return answer(runtime.co.takeoff())

    @app.post("/formation")
    def formation(body: dict = Body(...)):
        try:
            shape, center = str(body["shape"]), [float(v) for v in body["center"]]
            spacing, z = float(body.get("spacing", 1.0)), float(body.get("z", 1.0))
            turn = float(body.get("turn", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=[f"bad formation: {error}"])
        with runtime.lock:
            try:
                return answer(runtime.co.formation(shape, center, spacing, z, turn))
            except ValueError as error:
                raise HTTPException(status_code=400, detail=[str(error)])

    @app.post("/goto")
    def goto(body: dict = Body(...)):
        try:
            goals = {str(k): [float(a) for a in v] for k, v in body["goals"].items()}
            if any(len(v) != 3 for v in goals.values()):
                raise ValueError("each goal needs x, y and z")
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=[f"bad goals: {error}"])
        with runtime.lock:
            return answer(runtime.co.goto(goals))

    @app.post("/hold")
    def hold():
        with runtime.lock:
            runtime.co.hold()
        return {"ok": True}

    @app.post("/land")
    def land():
        with runtime.lock:
            runtime.co.land()
        return {"ok": True}

    @app.post("/land-here")
    def land_here():
        with runtime.lock:
            runtime.co.land_here()
        return {"ok": True}

    @app.post("/estop")
    def estop():
        with runtime.lock:
            runtime.co.estop()
        return {"ok": True}

    @app.post("/reset-pads")
    def reset_pads():
        """The operator has put the landed drones back on their pads."""
        with runtime.lock:
            return answer(runtime.co.reset_to_pads())

    @app.post("/obstacles")
    def add_obstacle(body: dict = Body(...)):
        try:
            obstacle = Obstacle(str(body["name"]), tuple(float(v) for v in body["lo"]), tuple(float(v) for v in body["hi"]))
            with runtime.lock:
                runtime.co.add_obstacle(obstacle)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=[str(error)])
        return {"ok": True}

    @app.delete("/obstacles/{name}")
    def remove_obstacle(name: str):
        with runtime.lock:
            if not runtime.co.remove_obstacle(name):
                raise HTTPException(status_code=404, detail=[f"no obstacle called {name}"])
        return {"ok": True}

    return app


# ------------------------------------------------------------------------------- the two setups
@dataclass
class DroneEntry:
    id: str
    backend: str
    pad: tuple[float, float]
    yaw_deg: float = 0.0
    camera_tilt_deg: float = DEFAULT_TILT_DEG
    focal_px: float | None = None  # at 1184 x 656, from `fit-focal`; until then the 90 degree guess is used


@dataclass
class SwarmFile:
    """Everything about one real installation, from a JSON file (see example.json)."""

    room: Room
    drones: list[DroneEntry]
    anchors: dict[str, Anchor]
    limits: Limits = field(default_factory=Limits)
    video_latency_s: float = 0.15

    @classmethod
    def load(cls, path: str | Path) -> "SwarmFile":
        data = json.loads(Path(path).read_text())
        drones = [DroneEntry(str(d["id"]), str(d["backend"]), tuple(d["pad"]), float(d.get("yaw_deg", 0.0)),
                             float(d.get("camera_tilt_deg", DEFAULT_TILT_DEG)),
                             None if d.get("focal_px") is None else float(d["focal_px"]))
                  for d in data["drones"]]
        return cls(Room.from_dict(data), drones, load_anchors(data.get("anchors")), Limits(**data.get("limits", {})),
                   float(data.get("video_latency_s", 0.15)))

    def optics(self) -> dict[str, DroneOptics]:
        return {d.id: DroneOptics(d.pad[0], d.pad[1], math.radians(d.yaw_deg), d.camera_tilt_deg,
                                  None if d.focal_px is None else Lens.from_focal(d.focal_px))
                for d in self.drones}

    def backends(self) -> dict[str, str]:
        return {d.id: d.backend for d in self.drones}


def real_runtime(config: SwarmFile) -> Runtime:
    from swarm.frames import StreamFrames
    from swarm.links import BackendLink

    links = {d.id: BackendLink(d.backend, d.id) for d in config.drones}
    frames = StreamFrames(config.backends())
    tracker = InsideOutTracker(config.optics(), config.anchors, (config.room.lo, config.room.hi), config.video_latency_s)
    configs = [DroneConfig(d.id, (d.pad[0], d.pad[1], 0.0), math.radians(d.yaw_deg)) for d in config.drones]
    first = next(iter(tracker.drones.values()))

    def visible(p) -> bool:
        lens = first.lens or Lens.from_fov()
        return bool(anchors_in_view(config.anchors, lens, math.radians(first.tilt_deg), p, math.radians(config.drones[0].yaw_deg)))

    coordinator = Coordinator(config.room, configs, links, config.limits, visible=visible, tracker=tracker)
    for link in links.values():
        link.start()
    frames.start()
    tracker.start(frames)

    def close() -> None:
        tracker.stop()
        frames.stop()
        for link in links.values():
            link.stop()

    runtime = Runtime(coordinator, time.monotonic, lambda: tracker.poses(time.monotonic()), close=close)
    runtime.tracker, runtime.frames = tracker, frames
    return runtime


DEMO_ROOM = Box((-5.0, -4.0, 0.0), (5.0, 4.0, 2.8), "wall")
DEMO_FENCE = ((-3.0, -2.5, 0.0), (1.7, 2.5, 2.4))
DEMO_OBSTACLES = (Box((-0.7, -0.6, 0.0), (-0.3, 0.6, 2.8), "wood"), Box((0.8, 1.3, 0.0), (1.4, 2.4, 0.9), "wood"))
DEMO_PADS = {"1": (-2.5, 1.2), "2": (-2.5, 0.0), "3": (-2.5, -1.2)}


def sim_runtime(seed: int = 1, conditions: Conditions | None = None, inside_out: bool = False, video_rate: float = 6.0) -> Runtime:
    """A simulated fleet on the real tag layout.

    Poses come straight from the simulator with added noise, or with inside_out from rendering and
    reading each drone's frames. That is slower than real time at full rate, so video_rate is lowered.
    """
    scene = Scene(name="demo", bounds=DEMO_ROOM, boxes=DEMO_OBSTACLES)
    fleet = SimFleet(scene, DEMO_PADS, seed, conditions)
    room = Room.from_scene(scene, DEMO_FENCE)
    configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in DEMO_PADS.items()]
    if not inside_out:
        return Runtime(Coordinator(room, configs, fleet.links), lambda: fleet.t, fleet.poses, advance=fleet.advance)
    from swarm.sim_anchors import SimVideo

    optics = {c.id: DroneOptics(c.pad[0], c.pad[1], c.yaw) for c in configs}
    tracker = InsideOutTracker(optics, load_anchors(anchors_section(0.0767, tag_grid())), DEMO_FENCE)
    video = SimVideo(fleet, tracker, scene, seed=seed, rate=video_rate)

    def advance(dt: float) -> None:
        fleet.advance(dt)
        video.step()

    runtime = Runtime(Coordinator(room, configs, fleet.links, tracker=tracker), lambda: fleet.t, lambda: tracker.poses(fleet.t), advance=advance)
    runtime.tracker = tracker
    return runtime
