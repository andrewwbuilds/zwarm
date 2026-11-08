"""What a drone's camera reports, and a simulated camera that reports it.

The hub only ever receives Detections: a label, directions and a rough distance relative to the
camera, which is all a real detector could give. It works out room positions from the tracker's pose.
The camera points down at a fixed tilt, so the ray through where a thing meets the floor gives a far
better position than guessing distance from its size. SimEyes reads the simulator's truth to decide
what is visible, then adds the errors. Its numbers are estimates. See swarm.yolo_eyes for the real one.

Angles are in the camera's frame: bearing positive to the left, elevation positive above its axis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np

HFOV = math.radians(90.0)
VFOV = math.radians(58.0)  # 16:9 at the 90 degree horizontal view the SLAM assumes
REACH = 3.5  # metres; past this a small object is too few pixels to name
RELIABLE_REACH = 3.0  # within this, not seeing something is evidence it isn't there
EDGE_MARGIN = math.radians(10.0)  # things at the edge of the picture don't count as looked at
DEFAULT_TILT = math.radians(45.0)  # how far below level the camera points
FLOOR_RAY_LIMIT = 2.0  # a floor position this many times off the size-based guess isn't believed

# Names a YOLO model trained on COCO can give that might sit in a room, and what people call them.
LABELS = {
    "person", "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup", "chair", "couch", "potted plant", "bed",
    "dining table", "tv", "laptop", "keyboard", "cell phone", "book", "clock", "vase", "teddy bear", "sports ball",
    "dog", "cat",
}


@dataclass(frozen=True)
class Detection:
    label: str
    bearing: float  # radians from the camera axis, positive to the left
    elevation: float  # radians from the camera axis, positive up
    range: float  # metres to its middle, a rough estimate from one camera
    size: tuple[float, float, float]
    confidence: float
    age: float = 0.0  # seconds between the picture being taken and this report
    foot: tuple[float, float] | None = None  # (bearing, elevation) of the nearest point where it meets the floor


class Eyes(Protocol):
    def look(self, drone_id: str, now: float) -> list[Detection] | None:
        """This drone's newest detections, or None if its camera has nothing to offer."""
        ...

    def knows(self, label: str) -> bool:
        ...


def camera_tilts(swarm_file: dict) -> dict[str, float]:
    """Per-drone camera tilts from a swarm file's "camera_tilt_deg", in radians."""
    return {str(d["id"]): math.radians(float(d["camera_tilt_deg"])) for d in swarm_file.get("drones", ()) if "camera_tilt_deg" in d}


def to_camera(offset, yaw, tilt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A room-frame offset as (along the camera axis, to its left, above it)."""
    offset = np.asarray(offset, float)
    x, y, z = offset[..., 0], offset[..., 1], offset[..., 2]
    forward = x * np.cos(yaw) + y * np.sin(yaw)
    left = -x * np.sin(yaw) + y * np.cos(yaw)
    return forward * math.cos(tilt) - z * math.sin(tilt), left, forward * math.sin(tilt) + z * math.cos(tilt)


def ray(yaw: float, tilt: float, bearing: float, elevation: float) -> np.ndarray:
    """The room-frame unit direction of a camera-frame angle."""
    axis = math.cos(elevation) * math.cos(bearing)
    left = math.cos(elevation) * math.sin(bearing)
    up = math.sin(elevation)
    forward = axis * math.cos(tilt) + up * math.sin(tilt)
    return np.array([forward * math.cos(yaw) - left * math.sin(yaw), forward * math.sin(yaw) + left * math.cos(yaw),
                     -axis * math.sin(tilt) + up * math.cos(tilt)])


def _on_floor(p, yaw: float, tilt: float, detection: Detection) -> np.ndarray | None:
    """Where a thing's middle is, from the ray to its foot, or None if that ray isn't believable."""
    direction = ray(yaw, tilt, *detection.foot)
    if p[2] <= 0.0 or direction[2] > -0.05:
        return None
    distance = -p[2] / direction[2]
    if detection.range > 0.0 and not 1 / FLOOR_RAY_LIMIT <= distance / detection.range <= FLOOR_RAY_LIMIT:
        return None
    flat = direction[:2] / np.linalg.norm(direction[:2])
    middle = p[:2] + direction[:2] * distance + flat * min(detection.size[0], detection.size[1]) / 2
    return np.array([middle[0], middle[1], detection.size[2] / 2])


def locate(p, yaw: float, detection: Detection, tilt: float = DEFAULT_TILT) -> np.ndarray:
    """Where a detection is in the room, given the drone's pose and its camera's tilt."""
    p = np.asarray(p, float)
    if detection.foot is not None:
        where = _on_floor(p, yaw, tilt, detection)
        if where is not None:
            return where
    return p + detection.range * ray(yaw, tilt, detection.bearing, detection.elevation)


def in_view(p, yaw, targets, reach: float = REACH, margin: float = 0.0, tilt: float = DEFAULT_TILT) -> np.ndarray:
    """Which target points are inside the picture of a camera at p, tilted down by tilt."""
    offset = np.atleast_2d(targets) - np.asarray(p, float)
    axis, left, up = to_camera(offset, yaw, tilt)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((axis > 0.0) & (np.linalg.norm(offset, axis=-1) <= reach)
                & (np.abs(left / axis) <= math.tan(HFOV / 2 - margin)) & (np.abs(up / axis) <= math.tan(VFOV / 2 - margin / 2)))


def blocked(a, b, lo, hi) -> np.ndarray:
    """Whether each straight line from a to b passes through the box."""
    a = np.broadcast_to(np.asarray(a, float), np.shape(b))
    b = np.asarray(b, float)
    along = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        low = (np.asarray(lo, float) - a) / along
        high = (np.asarray(hi, float) - a) / along
    near = np.minimum(low, high)
    far = np.maximum(low, high)
    # A line running parallel to a face is inside that slab or it misses the box completely.
    parallel = along == 0
    inside = (a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
    enter = np.maximum(near.max(axis=-1), 0.0)
    leave = np.minimum(far.min(axis=-1), 1.0)
    return enter <= leave


@dataclass
class Item:
    """Something in the simulated room for cameras to see. solid things are also in the scene."""

    label: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    confidence: float = 0.85
    until: float = math.inf  # a false object the detector stops reporting after this time

    @property
    def center(self) -> np.ndarray:
        return (np.array(self.lo) + np.array(self.hi)) / 2

    @property
    def size(self) -> np.ndarray:
        return np.array(self.hi) - np.array(self.lo)


@dataclass
class Walker:
    """A moving person or animal: where(t) returns a swarm.safety.Moving."""

    label: str
    where: object
    height: float = 1.7
    confidence: float = 0.85


class SimEyes:
    def __init__(self, fleet, items=(), walkers=(), seed: int = 1, false_rate: float = 0.005, tilts: dict | None = None):
        self.fleet = fleet
        self.items = list(items)
        self.walkers = list(walkers)
        self.rng = random.Random(seed * 7919 + 13)
        self.false_rate = false_rate  # chance per look of a one-frame detection of nothing
        self.tilts = tilts or {}
        self.blind: set[str] = set()
        self.biases: dict[tuple[str, int], tuple[float, float]] = {}
        self.mount_errors: dict[str, float] = {}

    def knows(self, label: str) -> bool:
        return label in LABELS

    def _bias(self, drone_id: str, key: int) -> tuple[float, float]:
        # One camera's distance guess is off the same way frame after frame, so averaging one
        # view doesn't remove it. Seeing it closer or from another drone does.
        pair = (drone_id, key)
        if pair not in self.biases:
            self.biases[pair] = (self.rng.gauss(0.0, 0.08), self.rng.gauss(0.0, math.radians(1.5)))
        return self.biases[pair]

    def _true_tilt(self, drone_id: str) -> float:
        # The picture also tips with the airframe, and no camera is mounted exactly as written down.
        if drone_id not in self.mount_errors:
            self.mount_errors[drone_id] = self.rng.gauss(0.0, math.radians(2.0))
        q = self.fleet.drones[drone_id].sim.drone
        return self.tilts.get(drone_id, DEFAULT_TILT) + self.mount_errors[drone_id] + q.pitch()

    def _visible(self, cam, yaw, tilt, points, skip=None) -> bool:
        seen = in_view(cam, yaw, points, tilt=tilt)
        if not seen.any():
            return False
        lines = np.atleast_2d(points)[seen]
        clear = np.ones(len(lines), bool)
        for box in self.fleet.scene.boxes:
            if box.solid and (box.lo, box.hi) != skip:
                clear &= ~blocked(cam, lines, box.lo, box.hi)
        return bool(clear.any())

    @staticmethod
    def _foot(cam, center, floor: float, half_depth: float) -> np.ndarray:
        """The middle of the near edge of its base: what the bottom centre of a detector's box lands on."""
        toward = cam[:2] - center[:2]
        toward = toward / max(float(np.linalg.norm(toward)), 1e-6)
        return np.array([*(center[:2] + toward * half_depth), floor])

    def _angles(self, offset, yaw, tilt, bias: float) -> tuple[float, float]:
        axis, left, up = (float(v) for v in to_camera(offset, yaw, tilt))
        return (math.atan2(left, axis) + bias + self.rng.gauss(0.0, math.radians(1.0)),
                math.atan2(up, math.hypot(axis, left)) + self.rng.gauss(0.0, math.radians(1.0)))

    def _report(self, drone_id, key, cam, yaw, tilt, label, center, size, confidence, foot=None, skip=None) -> Detection | None:
        offset = center - cam
        true_range = float(np.linalg.norm(offset))
        chance = 0.95 if true_range < 1.5 else 0.95 - 0.35 * (true_range - 1.5) / (REACH - 1.5)
        if self.rng.random() > chance:
            return None
        range_bias, bearing_bias = self._bias(drone_id, key)
        bearing, elevation = self._angles(offset, yaw, tilt, bearing_bias)
        distance = max(0.2, true_range * (1.0 + range_bias + self.rng.gauss(0.0, 0.06)))
        seen_size = tuple(float(s) * (1.0 + self.rng.gauss(0.0, 0.08)) for s in size)
        foot_angles = None
        if foot is not None and self._visible(cam, yaw, tilt, foot, skip=skip):
            foot_angles = self._angles(foot - cam, yaw, tilt, bearing_bias)
        return Detection(label, bearing, elevation, distance, seen_size,
                         min(0.99, max(0.41, confidence + self.rng.gauss(0.0, 0.05))), foot=foot_angles)

    def look(self, drone_id: str, now: float) -> list[Detection] | None:
        if drone_id in self.blind:
            return None
        cam = self.fleet.truth(drone_id)
        yaw = self.fleet.drones[drone_id].sim.drone.yaw()
        tilt = self._true_tilt(drone_id)
        found = []
        for key, item in enumerate(self.items):
            if now > item.until:
                continue
            center = item.center
            lo, hi = np.array(item.lo), np.array(item.hi)
            near = np.clip(cam, lo, hi)  # the part closest to the camera, which a wide thing shows first
            near = near + (center - near) * 0.05
            points = [np.array([x, y, z]) for x, y in ((center[0], center[1]), (near[0], near[1])) for z in (lo[2], center[2], hi[2])]
            skip = (item.lo, item.hi)
            if self._visible(cam, yaw, tilt, points, skip=skip):
                foot = self._foot(cam, center, lo[2], min(item.size[0], item.size[1]) / 2) if lo[2] < 0.05 else None
                seen = self._report(drone_id, key, cam, yaw, tilt, item.label, center, item.size, item.confidence, foot, skip)
                if seen is not None:
                    found.append(seen)
        for key, walker in enumerate(self.walkers, start=len(self.items)):
            moving = walker.where(now)
            center = np.array([moving.p[0], moving.p[1], walker.height / 2])
            points = [np.array([moving.p[0], moving.p[1], z]) for z in (0.0, walker.height / 2, walker.height)]
            if self._visible(cam, yaw, tilt, points):
                foot = self._foot(cam, center, 0.0, moving.radius)
                size = (2 * moving.radius, 2 * moving.radius, walker.height)
                seen = self._report(drone_id, key, cam, yaw, tilt, walker.label, center, size, walker.confidence, foot)
                if seen is not None:
                    found.append(seen)
        if self.rng.random() < self.false_rate:
            label = self.rng.choice(sorted(LABELS))
            found.append(Detection(label, self.rng.uniform(-HFOV / 2, HFOV / 2), self.rng.uniform(-VFOV / 2, VFOV / 2),
                                   self.rng.uniform(0.5, REACH), (0.4, 0.4, 0.4), self.rng.uniform(0.42, 0.7)))
        return found
