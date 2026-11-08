"""The swarm's operator API: a loop that ticks the coordinator, and HTTP around it.

Works the same with simulated drones or real ones. The frontend's /swarm page talks to this.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        due = time.monotonic()
        while not self.stopping.is_set():
            with self.lock:
                if self.advance is not None:
                    self.advance(TICK_S)
                now = self.clock()
                self.co.tick(now, self.get_poses(), self.get_moving(now))
            due += TICK_S
            late = time.monotonic() - due
            if late > TICK_S:
                self.overruns += 1
                due = time.monotonic()
            elif late < 0:
                self.stopping.wait(-late)

    def airborne(self) -> bool:
        return any(d.phase in (Phase.TAKEOFF, Phase.CHECK, Phase.AIR, Phase.LANDING) for d in self.co.drones.values())

    def shut_down(self) -> None:
        """Never leave drones hovering because the program quit: land them first."""
        with self.lock:
            flying = self.airborne()
            if flying:
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
    marker: int
    backend: str
    pad: tuple[float, float]
    yaw_deg: float = 0.0


@dataclass
class SwarmFile:
    """Everything about one real installation, from a JSON file (see example.json)."""

    room: Room
    drones: list[DroneEntry]
    marker_m: float
    mount_height_m: float
    cameras: list[dict]
    limits: Limits = field(default_factory=Limits)
    anchor_m: float = 0.15
    anchors: dict[int, tuple] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "SwarmFile":
        data = json.loads(Path(path).read_text())
        tracking = data.get("tracking", {})
        drones = [DroneEntry(str(d["id"]), int(d["marker"]), str(d["backend"]), tuple(d["pad"]), float(d.get("yaw_deg", 0.0)))
                  for d in data["drones"]]
        anchors = tracking.get("anchors", {})
        limits = Limits(**data.get("limits", {}))
        return cls(Room.from_dict(data), drones, float(tracking.get("marker_m", 0.07)),
                   float(tracking.get("mount_height_m", 0.05)), list(tracking.get("cameras", [])), limits,
                   float(anchors.get("marker_m", 0.15)),
                   {int(k): tuple(v) for k, v in anchors.get("positions", {}).items()})


def real_runtime(config: SwarmFile) -> Runtime:
    import math

    from swarm.links import BackendLink
    from swarm.tracking import CameraModel, CameraWorker, PoseStore

    if not config.cameras:
        raise ValueError("the swarm file lists no cameras: without tracking nothing can fly")
    links = {d.id: BackendLink(d.backend, d.id) for d in config.drones}
    store = PoseStore({d.marker: d.id for d in config.drones}, config.mount_height_m)
    workers = []
    for entry in config.cameras:
        model = CameraModel.load(entry["calibration"])
        if model.rvec is None:
            raise ValueError(f"{entry['calibration']} has no camera position: run locate-camera first")
        workers.append(CameraWorker(entry["source"], model, config.marker_m, store))
    configs = [DroneConfig(d.id, (d.pad[0], d.pad[1], 0.0), math.radians(d.yaw_deg)) for d in config.drones]
    coordinator = Coordinator(config.room, configs, links, config.limits,
                              visible=lambda p: any(worker.sees(p) for worker in workers))

    for link in links.values():
        link.start()
    for worker in workers:
        worker.start()

    def close() -> None:
        for worker in workers:
            worker.stop()
        for link in links.values():
            link.stop()

    return Runtime(coordinator, time.monotonic, store.poses, store.moving_things, close=close)


DEMO_ROOM = Box((-4.0, -3.0, 0.0), (4.0, 3.0, 2.6), "wall")
DEMO_OBSTACLES = (Box((-0.4, -0.9, 0.0), (0.4, 0.9, 2.6), "wood"), Box((1.8, 1.2, 0.0), (2.6, 3.0, 2.6), "wood"))
DEMO_PADS = {"1": (-2.5, -1.2), "2": (-2.5, 0.0), "3": (-2.5, 1.2)}


def sim_runtime(seed: int = 1, conditions: Conditions | None = None) -> Runtime:
    scene = Scene(name="demo", bounds=DEMO_ROOM, boxes=DEMO_OBSTACLES)
    fleet = SimFleet(scene, DEMO_PADS, seed, conditions)
    configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in DEMO_PADS.items()]
    coordinator = Coordinator(Room.from_scene(scene), configs, fleet.links)
    return Runtime(coordinator, lambda: fleet.t, fleet.poses, advance=fleet.advance)
