"""What each simulated S20 sees: rendered video from its true pose, tilted down like the real camera.

SimFrames does the same job as swarm.frames.StreamFrames, so the tracker, the detector, inspections
and map capture all run unchanged on it. One worker thread renders in turn for every drone and rests
between frames, so the flight loop keeps most of the CPU. latest() only reads a dict.

demo_world() is a room to fly in: furniture on the map, things that aren't, AprilTag anchors on the
floor, and staged damage painted on the floor and on low furniture. A frame from here is a simulator
frame and every one is stamped SIMULATED. Sharpness and exposure are guesses about the real camera.

Run `python -m swarm.sim_cameras --out DIR` to save sample frames and measure the render time.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sim import textures
from sim.camera import HEIGHT, LIGHT, SIZES, WIDTH, Camera, Pose as CameraPose, VideoEncoder
from sim.scene import Box, Scene
from swarm.eyes import Item
from swarm.frames import Frame
from swarm.room import Obstacle, Room
from swarm.tag41h12 import INNER_CELLS, TOTAL_CELLS
from swarm.tracking import Anchor, anchors_section, grid_layout, tag_image

logger = logging.getLogger(__name__)

FOV = math.radians(90.0)  # estimate: horizontal, what Lens.from_fov assumes
RENDER_SCALE = 0.54  # 640 x 360, what the video header advertises. Real sharpness is unmeasured
LATENCY_S = 0.15  # measured: real video arrives 100 to 200 ms late
MAX_BUSY = 0.5  # the worker's share of one core; if it would take more, the frame rate drops instead
IDLE_S = 0.01
ON_SURFACE = 0.006  # metres: a pixel this close to a decal's plane is on that surface
MARGIN_CELLS = 2  # white paper around a printed tag
PAPER, INK = 0.85, 0.07  # brightness of a tag's white and black


# ----------------------------------------------------------------------------- decals
@dataclass(frozen=True)
class Decal:
    """A flat picture on a surface: a floor marker, a crack. Paint only, so nothing can bump into it."""

    levels: tuple[np.ndarray, ...] = field(compare=False)  # BGRA floats, each half the size of the last
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right: tuple[float, float, float] = (1.0, 0.0, 0.0)  # room direction of the picture's columns
    down: tuple[float, float, float] = (0.0, 1.0, 0.0)  # room direction of its rows
    half_w: float = 0.5
    half_h: float = 0.5
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # out of the surface
    face: int = 5  # which entry of the renderer's LIGHT the surface gets
    lo: tuple[float, float, float] = (0.0, 0.0, 0.0)  # bounding box, for working out what can see it
    hi: tuple[float, float, float] = (0.0, 0.0, 0.0)


def make_decal(image: np.ndarray, center, right, down, width: float, height: float) -> Decal:
    """image is BGRA floats. right and down are unit vectors along an axis-aligned surface's plane."""
    right, down, center = (np.asarray(v, float) for v in (right, down, center))
    normal = np.cross(right, down) * -1.0  # right x down points into a picture seen from the front
    axis = int(np.argmax(np.abs(normal)))
    corners = [center + a * right * width / 2 + b * down * height / 2 for a in (-1, 1) for b in (-1, 1)]
    levels = [image.astype(np.float32)]
    while min(levels[-1].shape[:2]) > 4:
        h, w = levels[-1].shape[:2]
        levels.append(cv2.resize(levels[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
    return Decal(tuple(levels), tuple(float(v) for v in center), tuple(float(v) for v in right), tuple(float(v) for v in down),
                 width / 2, height / 2, tuple(float(v) for v in normal), axis * 2 + (1 if normal[axis] > 0 else 0),
                 tuple(float(v) - 0.001 for v in np.min(corners, axis=0)), tuple(float(v) + 0.001 for v in np.max(corners, axis=0)))


def flat_decal(image: np.ndarray, x: float, y: float, width: float, height: float, z: float = 0.0, yaw_deg: float = 0.0) -> Decal:
    """On a floor or a tabletop, seen from above. The top of the picture points along yaw, 0 being +x."""
    yaw = math.radians(yaw_deg)
    return make_decal(image, (x, y, z), (math.sin(yaw), -math.cos(yaw), 0.0), (-math.cos(yaw), -math.sin(yaw), 0.0), width, height)


def side_decal(image: np.ndarray, box: Box, face: str, along: float, z: float, width: float, height: float) -> Decal:
    """On one side of a box, face being "-x", "+x", "-y" or "+y". along is metres from that side's middle."""
    axis = 0 if face[1] == "x" else 1
    sign = -1.0 if face[0] == "-" else 1.0
    other = 1 - axis
    middle = [(box.lo[0] + box.hi[0]) / 2, (box.lo[1] + box.hi[1]) / 2, z]
    middle[axis] = box.hi[axis] if sign > 0 else box.lo[axis]
    normal = np.zeros(3)
    normal[axis] = sign
    right = np.cross(-normal, (0.0, 0.0, 1.0))  # to a viewer looking at the side, with up being up
    middle[other] += along * right[other]
    return make_decal(image, middle, right, (0.0, 0.0, -1.0), width, height)


def paper_width(anchor: Anchor) -> float:
    """The whole printed sheet. An anchor's `size` measures only the white square in its middle."""
    return anchor.size * (TOTAL_CELLS + 2 * MARGIN_CELLS) / INNER_CELLS


def anchor_decal(anchor: Anchor) -> Decal:
    """A printed tag on the floor where the swarm file says, with its white paper."""
    code = tag_image(anchor.tag, 24, MARGIN_CELLS)
    shade = (INK + (PAPER - INK) * code.astype(np.float32) / 255.0)[..., None]
    image = np.concatenate([shade, shade, shade, np.ones_like(shade)], axis=2)
    width = paper_width(anchor)
    return flat_decal(image, anchor.x, anchor.y, width, width, yaw_deg=anchor.yaw_deg)


def anchor_decals(anchors) -> tuple[Decal, ...]:
    """Decals for a tracker's anchors, given as its dict or as any list of Anchors."""
    return tuple(anchor_decal(a) for a in (anchors.values() if isinstance(anchors, dict) else anchors))


# ----------------------------------------------------------------------------- pictures of damage
def _noise(rng, width: int, height: int, base: int = 4) -> np.ndarray:
    return cv2.resize(textures.fbm(rng, base), (width, height), interpolation=cv2.INTER_CUBIC)


def _bgra(color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.dstack([np.clip(color, 0, 1), np.clip(alpha, 0, 1)]).astype(np.float32)


def crack_image(rng, width_m: float, height_m: float, thick_m: float = 0.022, per_m: int = 420) -> np.ndarray:
    """A jagged split with branches, starting wide and thinning out, with crumbled edges."""
    w, h = max(8, round(width_m * per_m)), max(8, round(height_m * per_m))
    core = np.zeros((h, w), np.uint8)
    stack = [(0.04 * w, rng.uniform(0.35, 0.65) * h, 0.0, thick_m * per_m, 0)]
    while stack:
        x, y, heading, thick, depth = stack.pop()
        while thick >= 1.0 and 0 <= x < w and 0 <= y < h:
            heading = (0.55 if depth == 0 else 0.8) * heading + rng.normal(0.0, 0.4 if depth == 0 else 0.6)
            step = rng.uniform(0.025, 0.06) * per_m
            nx, ny = x + step * math.cos(heading), y + step * math.sin(heading)
            cv2.line(core, (round(x), round(y)), (round(nx), round(ny)), 255, max(1, round(thick)), cv2.LINE_AA)
            if depth < 3 and rng.random() < 0.16:
                turn = float(rng.choice((-1, 1))) * rng.uniform(0.5, 1.1)
                stack.append((nx, ny, heading + turn, thick * 0.6, depth + 1))
            x, y = nx, ny
            thick *= 0.985 if depth == 0 else 0.9
    edge = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge = cv2.GaussianBlur(edge, (0, 0), 2.0).astype(np.float32) / 255.0
    edge *= np.clip(0.6 + 2.2 * _noise(rng, w, h, 16), 0, 1)
    ink = cv2.GaussianBlur(core, (0, 0), 0.8).astype(np.float32) / 255.0
    color = np.empty((h, w, 3), np.float32)
    color[...] = (0.17, 0.18, 0.19)  # crumbled material around the split
    color = color * (1 - ink[..., None]) + np.asarray((0.02, 0.02, 0.03), np.float32) * ink[..., None]
    return _bgra(color, np.maximum(ink, 0.45 * edge))


def scorch_image(rng, pixels: int = 400) -> np.ndarray:
    """A burn: charred black in the middle, brown soot fading out to an uneven edge, with streaks."""
    grid = (np.arange(pixels, dtype=np.float32) + 0.5) / pixels * 2 - 1
    u, v = np.meshgrid(grid, grid)
    r = np.sqrt(u * u + v * v)
    angle = np.arctan2(v, u)
    streaks = np.zeros_like(r)
    for lobes, amount in ((5, 0.06), (9, 0.05), (23, 0.03)):
        streaks += amount * np.sin(angle * lobes + rng.uniform(0, 6.28))
    outline = 0.72 + 0.34 * _noise(rng, pixels, pixels, 3) + streaks
    depth = np.clip((outline - r) / 0.42, 0, 1)
    speckle = _noise(rng, pixels, pixels, 32)
    alpha = np.clip(depth ** 0.7 + 0.5 * speckle * (depth > 0), 0, 1)
    char = np.asarray((0.015, 0.018, 0.02), np.float32)
    soot = np.asarray((0.05, 0.10, 0.16), np.float32)
    mix = np.clip(depth * 1.7, 0, 1)[..., None]
    color = soot * (1 - mix) + char * mix
    # Pale ash and cracked char right at the heart of the burn.
    heart = np.clip((0.3 + 0.8 * _noise(rng, pixels, pixels, 12) - r) / 0.2, 0, 1)[..., None] * 0.5
    color = color * (1 - heart) + np.asarray((0.22, 0.23, 0.24), np.float32) * heart
    return _bgra(color, alpha)


def rust_image(rng, pixels: int = 320) -> np.ndarray:
    """Blistered orange corrosion with dark pits, spreading in patches."""
    grid = (np.arange(pixels, dtype=np.float32) + 0.5) / pixels * 2 - 1
    u, v = np.meshgrid(grid, grid)
    r = np.sqrt(u * u + v * v)
    patches = _noise(rng, pixels, pixels, 4)
    alpha = np.clip((0.55 + 1.3 * patches - r * 0.85) / 0.12, 0, 1)
    tone = 1.0 + 1.2 * _noise(rng, pixels, pixels, 24)
    color = np.asarray((0.07, 0.20, 0.40), np.float32) * tone[..., None]
    pits = (_noise(rng, pixels, pixels, 64) > 0.2)[..., None]
    color = np.where(pits, np.asarray((0.03, 0.07, 0.14), np.float32), color)
    return _bgra(color, alpha)


def floor_image(rng, width_m: float, depth_m: float, per_m: int = 130) -> np.ndarray:
    """Sealed concrete that never repeats, so map building has real features to match frame to frame.

    Flecks are the features. The mottling stays faint so that plain floor doesn't read as a stain.
    The picture's columns run along -y and its rows along -x, as flat_decal lays them.
    """
    w, h = round(depth_m * per_m), round(width_m * per_m)
    shade = np.full((h, w), 0.56, np.float32) + 0.05 * _noise(rng, w, h, 5) + 0.03 * _noise(rng, w, h, 24)
    flecks = np.zeros((h, w), np.float32)
    count = int(width_m * depth_m * 900)
    for x, y, size, tone in zip(rng.integers(0, w, count), rng.integers(0, h, count), rng.integers(1, 3, count), rng.uniform(-0.13, 0.13, count)):
        cv2.circle(flecks, (int(x), int(y)), int(size), float(tone), -1, cv2.LINE_AA)
    shade = shade + cv2.GaussianBlur(flecks, (0, 0), 0.6)
    color = np.dstack([shade * 0.98, shade, shade * 1.01])
    return _bgra(color, np.ones((h, w), np.float32))


# ----------------------------------------------------------------------------- the camera
def tilted(rotation, tilt: float) -> tuple:
    """A body-to-world rotation with the camera pitched down by tilt about the body's left axis."""
    c, s = math.cos(tilt), math.sin(tilt)
    r = rotation
    return (r[0] * c - r[2] * s, r[1], r[0] * s + r[2] * c,
            r[3] * c - r[5] * s, r[4], r[3] * s + r[5] * c,
            r[6] * c - r[8] * s, r[7], r[6] * s + r[8] * c)


def level(yaw: float) -> tuple:
    """The rotation of a body hovering level, pointing along yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


class DecalCamera(Camera):
    """The simulator's camera, plus decals painted on whatever the depth pass says they are on."""

    def __init__(self, scene, decals=(), render_scale: float = RENDER_SCALE, fov: float = FOV, seed: int = 0):
        super().__init__(scene, render_scale=render_scale, fov=fov, seed=seed)
        self.decals = tuple(decals)
        self._cast = None

    def cast(self, pose: CameraPose):
        self._cast = super().cast(pose)
        return self._cast

    def shade(self, pose: CameraPose) -> np.ndarray:
        color = super().shade(pose)
        if self.decals:
            dist, _, _, rays = self._cast
            for decal in self.decals:
                self._paint(color, pose, dist, rays, decal)
        return color

    def _paint(self, color: np.ndarray, pose: CameraPose, dist: np.ndarray, rays, decal: Decal) -> None:
        rect = self.footprint(decal, pose)
        if rect is None:
            return
        part = (slice(rect[0], rect[1]), slice(rect[2], rect[3]))
        t = dist[part]
        cx, cy, cz = decal.center
        with np.errstate(invalid="ignore", over="ignore"):
            x, y, z = pose.x - cx + t * rays[0][part], pose.y - cy + t * rays[1][part], pose.z - cz + t * rays[2][part]
            n, r, d = decal.normal, decal.right, decal.down
            off = x * n[0] + y * n[1] + z * n[2]
            along = x * r[0] + y * r[1] + z * r[2]
            across = x * d[0] + y * d[1] + z * d[2]
            hit = (np.abs(off) < ON_SURFACE) & (np.abs(along) < decal.half_w) & (np.abs(across) < decal.half_h)
        if not hit.any():
            return
        # The sharpest level that isn't finer than a pixel, so far-off markers blur instead of breaking up.
        facing = abs(float(np.median(rays[0][part][hit] * n[0] + rays[1][part][hit] * n[1] + rays[2][part][hit] * n[2])))
        spread = float(np.median(t[hit])) * self.pixel / max(facing, 0.3)
        texel = 2 * decal.half_w / decal.levels[0].shape[1]
        index = int(np.clip(math.log2(max(spread / texel, 1.0)), 0, len(decal.levels) - 1))
        image = decal.levels[index]
        h, w = image.shape[:2]
        map_x = ((along / decal.half_w + 1) * w / 2 - 0.5).astype(np.float32)
        map_y = ((across / decal.half_h + 1) * h / 2 - 0.5).astype(np.float32)
        picture = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = picture[..., 3:] * hit[..., None]
        light = (LIGHT[decal.face] * np.clip(1.12 - 0.035 * t, 0.62, 1.1))[..., None]
        color[part] = color[part] * (1 - alpha) + picture[..., :3] * light * alpha


@dataclass(frozen=True)
class DecalScene(Scene):
    """A scene that carries its own decals, so SimFrames(fleet) finds them through fleet.scene."""

    decals: tuple[Decal, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class SimFrame(Frame):
    """A Frame, plus what only a simulator can know, for scoring anything that reads the video."""

    taken: float = 0.0  # when the picture was taken; t is when it arrived
    truth: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # camera x, y, z and yaw when taken


class SimFrames:
    """Implements swarm.frames.Frames from each simulated drone's true pose.

    Times are on the simulation's clock, the same one tick(now) gets. The rate is per drone. A drone
    whose link is down, or that's listed in `blind`, stops getting new frames, like real lost video.
    """

    def __init__(self, fleet, tilt_deg: float = 45.0, rate_hz: float = 5.0, *, decals=None, render_scale: float = RENDER_SCALE,
                 latency_s: float = LATENCY_S, max_busy: float = MAX_BUSY, clock=None, seed: int = 0):
        self.fleet = fleet
        self.tilt = math.radians(tilt_deg)
        self.period = 1.0 / rate_hz
        self.latency = latency_s
        self.max_busy = max_busy
        self.clock = clock or (lambda: fleet.t)
        if decals is None:
            decals = getattr(fleet.scene, "decals", ())
        self.camera = DecalCamera(fleet.scene, decals, render_scale, seed=seed)  # builds the atlas, so not on a worker
        self.encoders = {k: VideoEncoder(seed=seed + i) for i, k in enumerate(fleet.drones)}
        self.blind: set[str] = set()
        self.lock = threading.Lock()  # never held across a render, so latest() can't wait for one
        self.render_lock = threading.Lock()
        self.frames: dict[str, Frame] = {}
        self.waiting: deque = deque()
        self.numbers = {k: 0 for k in fleet.drones}
        start = self.clock()
        self.next_shot = {k: start + i * self.period / max(1, len(fleet.drones)) for i, k in enumerate(fleet.drones)}
        self.render_ms: deque = deque(maxlen=300)
        self.encode_ms: deque = deque(maxlen=300)
        self.sent = 0
        self.errors = 0
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is None:
            self.stopping.clear()
            self.thread = threading.Thread(target=self._run, daemon=True, name="sim-cameras")
            self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def latest(self, drone_id: str) -> Frame | None:
        with self.lock:
            now = self.clock()
            while self.waiting and self.waiting[0].t <= now:
                frame = self.waiting.popleft()
                self.frames[frame.drone] = frame
            return self.frames.get(drone_id)

    def render(self, drone_id: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """A BGR picture from this drone's camera right now, and where the camera was. Slow: not for the flight loop."""
        q = self.fleet.drones[drone_id].sim.drone
        pose = CameraPose(q.x, q.y, q.z, tilted(q.rotation(), self.tilt))
        return self.camera.render(pose), (q.x, q.y, q.z, q.yaw())

    def capture(self, drone_id: str, delay: bool = False) -> SimFrame:
        """Render one frame now. Without delay it's readable at once, for tests and lockstep benches."""
        with self.render_lock:
            taken = self.clock()
            started = time.perf_counter()
            image, truth = self.render(drone_id)
            rendered = time.perf_counter()
            jpeg = self.encoders[drone_id].compress(image, SIZES[1])
            self.render_ms.append((rendered - started) * 1000)
            self.encode_ms.append((time.perf_counter() - rendered) * 1000)
            self.numbers[drone_id] += 1
            frame = SimFrame(drone_id, taken + (self.latency if delay else 0.0), jpeg, self.numbers[drone_id], taken, truth)
        with self.lock:
            if delay:
                self.waiting.append(frame)
            else:
                self.frames[drone_id] = frame
        self.sent += 1
        return frame

    def stats(self) -> dict:
        render, encode = list(self.render_ms), list(self.encode_ms)
        if not render:
            return {"frames": self.sent, "errors": self.errors}
        return {"frames": self.sent, "errors": self.errors, "render_ms": round(statistics.mean(render), 1),
                "render_ms_worst": round(max(render), 1), "encode_ms": round(statistics.mean(encode), 1)}

    def _due(self, now: float) -> str | None:
        """The drone that has waited longest for a frame, skipping any that can't send one."""
        best = None
        for drone in self.next_shot:
            if self.next_shot[drone] > now:
                continue
            link = getattr(self.fleet, "links", {}).get(drone)
            if drone in self.blind or (link is not None and not link.connected):
                self.next_shot[drone] = now + self.period
            elif best is None or self.next_shot[drone] < self.next_shot[best]:
                best = drone
        return best

    def _run(self) -> None:
        while not self.stopping.is_set():
            now = self.clock()
            drone = self._due(now)
            if drone is None:
                self.stopping.wait(IDLE_S)
                continue
            # Falling behind drops frames instead of building up a burst to catch up with.
            self.next_shot[drone] = max(self.next_shot[drone] + self.period, now)
            started = time.perf_counter()
            try:
                self.capture(drone, delay=True)
            except Exception:
                self.errors += 1
                if self.errors == 1:
                    logger.exception("[sim-cameras] a render failed; carrying on")
            self.stopping.wait((time.perf_counter() - started) * (1.0 / self.max_busy - 1.0))


# ----------------------------------------------------------------------------- the demo room
FENCE = ((-3.0, -2.5, 0.0), (3.0, 2.5, 2.6))
PADS = {"1": (-2.2, -1.2), "2": (-2.2, 0.0), "3": (-2.2, 1.2)}
TAG_WIDTH_M = 0.14  # the printed tags' whole width, the same as the real ones
TAG_SPACING_M = 1.25


@dataclass(frozen=True)
class Damage:
    """One piece of staged damage: the truth to score an inspection against."""

    name: str
    kind: str  # crack, scorch or rust
    on: str  # what it's on: floor, or a thing's name
    where: tuple[float, float, float]  # its middle
    size: float  # across, in metres
    severity: str  # what a person looking at it would probably say
    facing: tuple[float, float] = (0.0, 0.0)  # for a mark on a side: the way that side faces


@dataclass(frozen=True)
class DemoWorld:
    scene: DecalScene
    pads: dict[str, tuple[float, float]]
    fence: tuple
    furniture: tuple[Obstacle, ...]  # on the room's map from the start
    items: tuple[Item, ...]  # in the room, but only the cameras can find them
    anchors: dict[str, Anchor]
    damage: tuple[Damage, ...]

    def room(self) -> Room:
        """The map a mission starts with: the furniture, without the hidden items."""
        return Room(self.fence[0], self.fence[1], self.furniture)

    def anchors_section(self) -> dict:
        """The anchors as a swarm file lists them; swarm.tracking.load_anchors reads this back."""
        edge = next(iter(self.anchors.values())).size
        return anchors_section(edge, [{"tag": a.tag, "x": a.x, "y": a.y, "rotation_deg": a.yaw_deg} for a in self.anchors.values()])


def view_of(point, height: float = 1.0, yaw: float = 0.0, tilt_deg: float = 45.0) -> tuple[float, float, float, float]:
    """Where to hover at this height so the middle of the picture is on a point: (x, y, z, yaw)."""
    back = (height - point[2]) / math.tan(math.radians(tilt_deg))
    return (point[0] - back * math.cos(yaw), point[1] - back * math.sin(yaw), height, yaw)


def demo_world(seed: int = 11) -> DemoWorld:
    """A 6 x 5 m room with a concrete floor. North is +x, and the pads are at the south end.

    Everything sits below the drones' 1 m cruise so the 45 degree camera can look down on it.
    Materials and marks are chosen to read at the video's real size, not to look nice up close.
    """
    rng = np.random.default_rng(seed)
    # Things stand between the rows of the tag grid, so every tag stays readable and in the open.
    generator = Box((1.4, -2.0, 0.0), (2.3, -1.25, 0.6), "metal")
    crate = Box((1.525, 0.9, 0.0), (2.225, 1.6, 0.55), "wood")
    cabinet = Box((-1.1, -2.4, 0.0), (-0.2, -1.7, 0.5), "white")
    couch = Box((-1.1, 0.85, 0.0), (-0.15, 1.65, 0.7), "fabric")
    furniture = {"generator": generator, "crate": crate, "cabinet": cabinet, "couch": couch}
    hidden = (("backpack", Box((0.5, 1.95, 0.0), (0.85, 2.25, 0.45), "fabric")),
              ("suitcase", Box((-2.6, -2.4, 0.0), (-2.05, -2.1, 0.6), "leather")),
              ("chair", Box((2.5, -0.225, 0.0), (2.95, 0.225, 0.85), "wood")),
              ("potted plant", Box((-0.2, -0.2, 0.0), (0.2, 0.2, 0.7), "foliage")))

    damage = (Damage("long crack across the floor", "crack", "floor", (1.0, 0.0, 0.0), 1.4, "high"),
              Damage("burn mark on the floor", "scorch", "floor", (-0.625, -1.0, 0.0), 0.8, "high"),
              Damage("burn mark on top of the generator", "scorch", "generator", (1.85, -1.62, 0.6), 0.5, "high"),
              Damage("split in the side of the crate", "crack", "crate", (1.525, 1.25, 0.3), 0.6, "medium", (-1.0, 0.0)),
              Damage("rust on top of the cabinet", "rust", "cabinet", (-0.65, -2.05, 0.5), 0.55, "medium"),
              Damage("small crack in the floor", "crack", "floor", (2.0, -0.9, 0.0), 0.6, "low"))
    width, depth = FENCE[1][0] - FENCE[0][0], FENCE[1][1] - FENCE[0][1]
    floor = flat_decal(floor_image(rng, width, depth), (FENCE[0][0] + FENCE[1][0]) / 2, (FENCE[0][1] + FENCE[1][1]) / 2, depth, width)
    marks = (flat_decal(crack_image(rng, 1.4, 0.7), 1.0, 0.0, 1.4, 0.7, yaw_deg=90.0),
             flat_decal(scorch_image(rng), -0.625, -1.0, 0.8, 0.8),
             flat_decal(scorch_image(rng), 1.85, -1.62, 0.5, 0.5, z=0.6),
             side_decal(crack_image(rng, 0.6, 0.4, thick_m=0.016), crate, "-x", 0.0, 0.3, 0.6, 0.4),
             flat_decal(rust_image(rng), -0.65, -2.05, 0.55, 0.55, z=0.5),
             flat_decal(crack_image(rng, 0.6, 0.35, thick_m=0.012), 2.0, -0.9, 0.6, 0.35, yaw_deg=30.0))

    # A tag can't lie under furniture, and staged damage shouldn't deface one.
    boxes = (*furniture.values(), *(box for _, box in hidden))
    keep_off = (*boxes, *(mark for mark in marks if mark.center[2] == 0.0))

    def clear(anchor: Anchor) -> bool:
        reach = paper_width(anchor) / 2
        return not any(t.lo[0] - reach < anchor.x < t.hi[0] + reach and t.lo[1] - reach < anchor.y < t.hi[1] + reach for t in keep_off)

    edge = TAG_WIDTH_M * INNER_CELLS / TOTAL_CELLS
    anchors = {a.id: a for a in grid_layout((-2.5, -2.0), (2.5, 2.0), TAG_SPACING_M, edge) if clear(a)}
    scene = DecalScene(name="zwarm-demo", bounds=Box(FENCE[0], FENCE[1], "wall"), floor_material="concrete", boxes=boxes,
                       decals=(floor, *anchor_decals(anchors), *marks))
    return DemoWorld(scene, dict(PADS), FENCE, tuple(Obstacle(name, box.lo, box.hi) for name, box in furniture.items()),
                     tuple(Item(label, box.lo, box.hi) for label, box in hidden), anchors, damage)


# ----------------------------------------------------------------------------- cameras at chosen poses
class _Hover:
    """Answers what SimFrames asks of a drone. x, y, z and yaw can be changed at any time."""

    def __init__(self, x: float, y: float, z: float, yaw: float):
        self.x, self.y, self.z, self.heading = x, y, z, yaw
        self.sim, self.drone = self, self  # looks like SimDrone.sim.drone

    def rotation(self) -> tuple:
        return level(self.heading)

    def yaw(self) -> float:
        return self.heading


class StillFleet:
    """Stands in for a SimFleet when all you need is pictures from chosen poses: {name: (x, y, z, yaw)}."""

    def __init__(self, scene, spots: dict):
        self.scene = scene
        self.t = 0.0
        self.links: dict = {}
        self.drones = {k: _Hover(*v) for k, v in spots.items()}


# ----------------------------------------------------------------------------- python -m swarm.sim_cameras
def _save_samples(world: DemoWorld, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spots = {f"damage{i}-{d.kind}-{d.on}": view_of(d.where, 1.0, math.atan2(-d.facing[1], -d.facing[0]) if any(d.facing) else 0.0)
             for i, d in enumerate(world.damage)}
    spots.update({f"pad{k}": (p[0], p[1], 1.0, 0.0) for k, p in world.pads.items()})
    spots.update({"middle-north": (0.0, 0.0, 1.2, 0.0), "middle-east": (0.0, 0.5, 1.2, -math.pi / 2)})
    frames = SimFrames(StillFleet(world.scene, spots))
    for name in spots:
        (out / f"{name}.jpg").write_bytes(frames.capture(name).jpeg)
    print(f"saved {len(spots)} frames to {out}")


def _measure(world: DemoWorld, count: int) -> None:
    fleet = StillFleet(world.scene, {"a": (-1.0, 0.0, 1.0, 0.0), "b": (1.0, 0.0, 1.2, 0.7), "c": (0.0, -1.0, 1.0, -1.5)})
    frames = SimFrames(fleet)
    keys = list(fleet.drones)
    for i in range(count):
        frames.capture(keys[i % 3])
    print(f"simulator, {count} frames rendered at {frames.camera.width}x{frames.camera.height} and scaled to {WIDTH}x{HEIGHT}: {frames.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save sample frames from the demo room and time the renderer.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    started = time.perf_counter()
    world = demo_world()
    print(f"demo_world took {time.perf_counter() - started:.2f} s")
    if args.out is not None:
        _save_samples(world, args.out)
    _measure(world, args.count)


if __name__ == "__main__":
    main()
