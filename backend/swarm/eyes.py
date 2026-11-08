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


class Eyes(Protocol):
    def look(self, drone_id: str, now: float) -> list[Detection] | None:
        """This drone's newest detections, or None if its camera has nothing to offer."""
        ...

    def knows(self, label: str) -> bool:
        ...


def locate(p, yaw: float, detection: Detection) -> np.ndarray:
    """Where a detection is in the room, given the drone's pose."""
    heading = yaw + detection.bearing
    flat = detection.range * math.cos(detection.elevation)
    return np.asarray(p, float) + np.array([flat * math.cos(heading), flat * math.sin(heading),
                                            detection.range * math.sin(detection.elevation)])


def in_view(p, yaw: float, targets, reach: float = REACH, margin: float = 0.0) -> np.ndarray:
    """Which target points are inside the picture of a level camera at p."""
    offset = np.atleast_2d(targets) - np.asarray(p, float)
    flat = np.hypot(offset[:, 0], offset[:, 1])
    bearing = (np.arctan2(offset[:, 1], offset[:, 0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    elevation = np.arctan2(offset[:, 2], flat)
    return ((np.linalg.norm(offset, axis=1) <= reach) & (np.abs(bearing) <= HFOV / 2 - margin)
            & (np.abs(elevation) <= VFOV / 2 - margin / 2))


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
    def __init__(self, fleet, items=(), walkers=(), seed: int = 1, false_rate: float = 0.005):
        self.fleet = fleet
        self.items = list(items)
        self.walkers = list(walkers)
        self.rng = random.Random(seed * 7919 + 13)
        self.false_rate = false_rate  # chance per look of a one-frame detection of nothing
        self.blind: set[str] = set()
        self.biases: dict[tuple[str, int], tuple[float, float]] = {}

    def knows(self, label: str) -> bool:
        return label in LABELS

    def _bias(self, drone_id: str, key: int) -> tuple[float, float]:
        # One camera's distance guess is off the same way frame after frame, so averaging one
        # view doesn't remove it. Seeing it closer or from another drone does.
        pair = (drone_id, key)
        if pair not in self.biases:
            self.biases[pair] = (self.rng.gauss(0.0, 0.08), self.rng.gauss(0.0, math.radians(1.5)))
        return self.biases[pair]

    def _visible(self, cam, yaw, points, skip=None) -> bool:
        seen = in_view(cam, yaw, points)
        if not seen.any():
            return False
        lines = np.atleast_2d(points)[seen]
        clear = np.ones(len(lines), bool)
        for box in self.fleet.scene.boxes:
            if box.solid and (box.lo, box.hi) != skip:
                clear &= ~blocked(cam, lines, box.lo, box.hi)
        return bool(clear.any())

    def _report(self, drone_id, key, cam, yaw, label, center, size, confidence) -> Detection | None:
        offset = center - cam
        true_range = float(np.linalg.norm(offset))
        chance = 0.95 if true_range < 1.5 else 0.95 - 0.35 * (true_range - 1.5) / (REACH - 1.5)
        if self.rng.random() > chance:
            return None
        range_bias, bearing_bias = self._bias(drone_id, key)
        bearing = math.atan2(offset[1], offset[0]) - yaw + bearing_bias + self.rng.gauss(0.0, math.radians(1.0))
        elevation = math.atan2(offset[2], math.hypot(offset[0], offset[1])) + self.rng.gauss(0.0, math.radians(1.0))
        distance = max(0.2, true_range * (1.0 + range_bias + self.rng.gauss(0.0, 0.06)))
        seen_size = tuple(float(s) * (1.0 + self.rng.gauss(0.0, 0.08)) for s in size)
        return Detection(label, bearing, elevation, distance, seen_size,
                         min(0.99, max(0.41, confidence + self.rng.gauss(0.0, 0.05))))

    def look(self, drone_id: str, now: float) -> list[Detection] | None:
        if drone_id in self.blind:
            return None
        cam = self.fleet.truth(drone_id)
        yaw = self.fleet.drones[drone_id].sim.drone.yaw()
        found = []
        for key, item in enumerate(self.items):
            if now > item.until:
                continue
            center = item.center
            lo, hi = np.array(item.lo), np.array(item.hi)
            nearest = np.clip(cam, lo, hi)  # the part closest to the camera, which a wide thing shows first
            points = [center, np.array([center[0], center[1], np.clip(cam[2], lo[2], hi[2])]), nearest + (center - nearest) * 0.05]
            if self._visible(cam, yaw, points, skip=(item.lo, item.hi)):
                seen = self._report(drone_id, key, cam, yaw, item.label, center, item.size, item.confidence)
                if seen is not None:
                    found.append(seen)
        for key, walker in enumerate(self.walkers, start=len(self.items)):
            moving = walker.where(now)
            center = np.array([moving.p[0], moving.p[1], walker.height / 2])
            points = [center, np.array([moving.p[0], moving.p[1], cam[2]])]
            if self._visible(cam, yaw, points):
                size = (2 * moving.radius, 2 * moving.radius, walker.height)
                seen = self._report(drone_id, key, cam, yaw, walker.label, center, size, walker.confidence)
                if seen is not None:
                    found.append(seen)
        if self.rng.random() < self.false_rate:
            label = self.rng.choice(sorted(LABELS))
            found.append(Detection(label, self.rng.uniform(-HFOV / 2, HFOV / 2), self.rng.uniform(-VFOV / 2, VFOV / 2),
                                   self.rng.uniform(0.5, REACH), (0.4, 0.4, 0.4), self.rng.uniform(0.42, 0.7)))
        return found
