"""Frame sequences with ground truth: flown in the simulator, or loaded from a real recording."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from sim.camera import Camera, Pose, RATE, WIDTH, HEIGHT
from sim.config import DroneSpec
from sim.pilot import FLY, Pilot
from sim.scene import Box, Scene
from sim.scenes import SCENES
from slam.camera import Intrinsics
from slam.geometry import camera_pose_from_body

JPEG_QUALITY = 60  # about what the real camera's 34 KB frames work out to at this size


@dataclass
class Frame:
    t: float
    image: np.ndarray  # BGR
    truth: np.ndarray | None = None  # 4x4 camera-to-world pose, in metres
    command: tuple | None = None  # the sticks flown: roll, pitch, throttle, yaw, each -1 to 1


def textured(name: str = "room", wall: str = "brick", ceiling: str = "brick") -> Scene:
    """A simulator scene with detail on every wall: a blank wall gives any tracker nothing to hold on to."""
    return replace(SCENES[name](), wall_material=wall, ceiling_material=ceiling)


def hall() -> Scene:
    """A 9 x 9 m hall with brick walls and detailed objects spread around: room for loops, detail everywhere.

    Every surface carries texture, and each wall has its own poster, so no view is blank and no two places look alike.
    """
    objects = (
        Box((-3.8, -3.0, 0.0), (-3.0, -1.0, 1.0), "wood"),
        Box((-3.9, 1.0, 0.0), (-3.4, 3.2, 2.0), "books", fit=True),
        Box((-1.5, 3.4, 0.0), (1.5, 3.9, 1.6), "books", fit=True),
        Box((2.0, 3.0, 0.0), (3.6, 3.8, 0.8), "fabric"),
        Box((3.4, -1.0, 0.0), (3.9, 1.5, 1.2), "wood"),
        Box((2.5, -3.6, 0.0), (3.8, -2.6, 0.9), "fabric"),
        Box((-0.8, -3.95, 0.4), (0.8, -3.9, 1.9), "window", fit=True),
        Box((3.95, -0.6, 1.0), (3.99, 0.8, 2.2), "painting", fit=True),
        Box((-3.99, -0.8, 1.0), (-3.95, 0.6, 2.3), "door", fit=True),
        Box((-0.9, 3.95, 1.7), (0.9, 3.99, 2.8), "tv", fit=True),
        Box((-1.8, -1.8, 0.0), (1.8, 1.8, 0.004), "rug", fit=True),
    )
    return Scene(name="hall", bounds=Box((-4.0, -4.0, 0.0), (4.0, 4.0, 3.2), "brick"), boxes=objects,
                 floor_material="wood_floor", wall_material="brick", ceiling_material="brick", start=(-1.2, -1.2, 0.0))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class SimFlight:
    """Fly the simulated drone with a scripted pilot and record what its camera sees."""

    def __init__(self, scene: "str | Scene" = "room", seed: int = 1, render_scale: float = 0.36, jpeg: bool = True):
        self.scene = scene if isinstance(scene, Scene) else SCENES[scene]()
        self.pilot = Pilot(scene=self.scene, seed=seed)
        self.camera = Camera(self.scene, render_scale=render_scale, seed=seed)
        self.jpeg = jpeg
        self.frames: list[Frame] = []
        self.intrinsics = Intrinsics.from_fov(WIDTH, HEIGHT, DroneSpec().camera_fov_deg)
        self._period = 1.0 / RATE
        self._next = 0.0
        self.drone = self.pilot.sim.drone
        self.pilot.fly(2.5)
        self.pilot.hold(FLY, 0.2)
        self._until(lambda: self.pilot.fc.state == "FLYING", 12.0)

    # ------------------------------------------------------------------ recording
    def _capture(self, command: tuple | None = None) -> None:
        d = self.drone
        image = self.camera.render(Pose(d.x, d.y, d.z, d.rotation()))
        if self.jpeg:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.frames.append(Frame(self.pilot.sim.t, image, camera_pose_from_body([d.x, d.y, d.z], d.rotation()), command))

    def _step(self, **sticks) -> None:
        self.pilot.fly(self._period, **sticks)
        raw = [sticks.get(name, 128) for name in ("roll", "pitch", "throttle", "yaw")]
        self._capture(tuple((v - 128) / 127 for v in raw))

    def _until(self, done, timeout: float, **sticks) -> bool:
        end = self.pilot.sim.t + timeout
        while self.pilot.sim.t < end:
            self._step(**sticks)
            if done():
                return True
        return False

    # --------------------------------------------------------------- manoeuvres
    def hover(self, seconds: float) -> "SimFlight":
        self._until(lambda: False, seconds)
        return self

    def forward(self, metres: float, speed: float = 0.5) -> "SimFlight":
        """Fly along the nose by this distance. speed is the stick deflection, 0 to 1."""
        d = self.drone
        x0, y0 = d.x, d.y
        stick = round(128 + 127 * speed)
        self._until(lambda: math.hypot(d.x - x0, d.y - y0) >= metres - 0.25 * speed * 2, 30.0, pitch=stick)
        self._until(lambda: False, 0.6)
        return self

    def strafe(self, metres: float, speed: float = 0.5) -> "SimFlight":
        d = self.drone
        x0, y0 = d.x, d.y
        stick = round(128 + 127 * speed * math.copysign(1, -metres))
        self._until(lambda: math.hypot(d.x - x0, d.y - y0) >= abs(metres) - 0.25 * speed * 2, 30.0, roll=stick)
        self._until(lambda: False, 0.6)
        return self

    def turn(self, degrees: float, rate: float = 0.5) -> "SimFlight":
        """Turn left (positive) or right by this many degrees."""
        d = self.drone
        target = wrap(d.yaw() + math.radians(degrees))
        stick = round(128 + 127 * rate * -math.copysign(1, degrees))
        self._until(lambda: abs(wrap(target - d.yaw())) < math.radians(4), 20.0, yaw=stick)
        self._until(lambda: False, 0.5)
        return self

    def arc(self, degrees: float, speed: float = 0.45, turn: float = 0.22) -> "SimFlight":
        """Fly forward while turning left (positive) or right, until the heading has changed by this much."""
        d = self.drone
        start = d.yaw()
        stick_yaw = round(128 + 127 * turn * -math.copysign(1, degrees))
        stick_pitch = round(128 + 127 * speed)
        swept = lambda: abs(wrap(d.yaw() - start)) if abs(degrees) < 180 else None
        target, travelled, last = math.radians(abs(degrees)), 0.0, start
        end = self.pilot.sim.t + 60.0
        while travelled < target and self.pilot.sim.t < end:
            self._step(pitch=stick_pitch, yaw=stick_yaw)
            travelled += abs(wrap(d.yaw() - last))
            last = d.yaw()
        self._until(lambda: False, 0.6)
        return self

    def climb(self, metres: float) -> "SimFlight":
        d = self.drone
        z0 = d.z
        stick = 255 if metres > 0 else 0
        self._until(lambda: abs(d.z - z0) >= abs(metres) - 0.2, 15.0, throttle=stick)
        self._until(lambda: False, 0.6)
        return self

    def square(self, side: float, speed: float = 0.5, left: bool = True) -> "SimFlight":
        for _ in range(4):
            self.forward(side, speed).turn(90 if left else -90)
        return self

    # --------------------------------------------------------------------- output
    @property
    def truth_positions(self) -> np.ndarray:
        return np.array([f.truth[:3, 3] for f in self.frames])

    def cache(self) -> "SimFlight":
        """Drop the simulator so the flight can be pickled and reused."""
        self.camera = self.pilot = self.drone = None
        return self

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.frames):
            cv2.imwrite(str(folder / f"{i:05d}.jpg"), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.save(folder / "truth.npy", np.array([f.truth for f in self.frames]))
        np.save(folder / "times.npy", np.array([f.t for f in self.frames]))


def load_recording(folder: Path, step: int = 1) -> list[Frame]:
    """Frames from a wifi_recorder capture: frames/NNNNN.jpg with times from events.jsonl."""
    folder = Path(folder)
    stamps = {}
    for line in (folder / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "frame":
            stamps[row["n"]] = int.from_bytes(bytes.fromhex(row["envelope"])[32:40], "little") / 1e6
    frames = []
    for n in sorted(stamps)[::step]:
        image = cv2.imread(str(folder / "frames" / f"{n:05d}.jpg"))
        if image is not None:
            frames.append(Frame(stamps[n], image))
    return frames
