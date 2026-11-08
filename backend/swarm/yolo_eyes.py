"""Real eyes: YOLO on each drone's newest picture, from the shared reader in swarm.frames.

One thread runs the detector, taking each drone's newest unseen frame in turn, so the flight loop
never waits for it. look() hands each result over once and returns None when nothing new has
arrived, because "no new picture" must not count as "looked and saw nothing".

YOLO only names what it was trained on. A plain box or a wall gets no detection at all, so this
finds and avoids the things in eyes.LABELS and nothing else.

Needs the follow extra: `uv sync --inexact --extra follow`.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from swarm.eyes import HFOV, LABELS, Detection

logger = logging.getLogger(__name__)

CONFIDENCE = 0.45  # below the memory's bar for "found", so doubtful things are still steered around
IMAGE_SIZE = 416  # 36 ms a picture on an M5 Pro's CPU, against 45 ms on its GPU
MAX_AGE_S = 0.8  # older than this, the pose it should be matched with is too uncertain
LATENCY_S = 0.15  # picture taken to frame arriving: an estimate, so measure it on the real drones
EDGE_PX = 3  # a box this close to the edge of the picture is cut off
MIN_BOX_PX = 12
IDLE_S = 0.01

# Rough real sizes in metres: width, depth, height. Only used for a first guess at distance.
REAL_SIZE = {
    "person": (0.5, 0.35, 1.7), "backpack": (0.32, 0.2, 0.45), "umbrella": (0.9, 0.9, 0.9), "handbag": (0.32, 0.15, 0.25),
    "suitcase": (0.45, 0.25, 0.65), "bottle": (0.08, 0.08, 0.25), "cup": (0.09, 0.09, 0.1), "chair": (0.5, 0.5, 0.9),
    "couch": (1.9, 0.9, 0.85), "potted plant": (0.4, 0.4, 0.7), "bed": (1.5, 2.0, 0.6), "dining table": (1.4, 0.9, 0.75),
    "tv": (1.1, 0.1, 0.65), "laptop": (0.33, 0.23, 0.23), "keyboard": (0.44, 0.13, 0.03), "cell phone": (0.07, 0.15, 0.01),
    "book": (0.15, 0.22, 0.03), "clock": (0.3, 0.05, 0.3), "vase": (0.15, 0.15, 0.3), "teddy bear": (0.3, 0.25, 0.4),
    "sports ball": (0.22, 0.22, 0.22), "dog": (0.3, 0.7, 0.55), "cat": (0.2, 0.45, 0.3),
}
# These stand on the floor, so the ray through the bottom of the box gives their position. A cup is
# usually on a table, where that ray would land well behind it.
ON_FLOOR = {"person", "backpack", "suitcase", "chair", "couch", "potted plant", "bed", "dining table", "dog", "cat",
            "teddy bear", "sports ball", "handbag"}


def _angles(u: float, v: float, width: int, height: int, focal: float) -> tuple[float, float]:
    """A pixel's (bearing, elevation) in the camera frame."""
    left, up = (width / 2 - u) / focal, (height / 2 - v) / focal
    return math.atan(left), math.atan2(up, math.hypot(1.0, left))


def detection_from_box(label: str, confidence: float, box, width: int, height: int, hfov: float = HFOV) -> Detection:
    """A detector's pixel box as camera-frame directions and a rough distance."""
    x1, y1, x2, y2 = (float(v) for v in box)
    focal = (width / 2) / math.tan(hfov / 2)
    bearing, elevation = _angles((x1 + x2) / 2, (y1 + y2) / 2, width, height, focal)
    real_w, real_d, real_h = REAL_SIZE[label]
    cut_vertically = y1 <= EDGE_PX or y2 >= height - EDGE_PX
    cut_sideways = x1 <= EDGE_PX or x2 >= width - EDGE_PX
    if not cut_vertically:
        along_axis = focal * real_h / max(y2 - y1, 1.0)
    elif not cut_sideways:
        along_axis = focal * real_w / max(x2 - x1, 1.0)
    else:
        along_axis = 0.8  # cut off on both axes: it fills the picture, so it's close
    distance = along_axis / max(math.cos(bearing) * math.cos(elevation), 0.1)
    foot = None
    if label in ON_FLOOR and y2 < height - EDGE_PX:
        foot = _angles((x1 + x2) / 2, y2, width, height, focal)
    return Detection(label, bearing, elevation, distance, (real_w, real_d, real_h), confidence, foot=foot, rough_range=cut_vertically)


def weights_path() -> str:
    configured = os.getenv("YOLO_WEIGHTS")
    if configured and os.path.exists(configured):
        return configured
    in_repo = Path(__file__).resolve().parents[2] / "yolov10n.pt"
    return str(in_repo) if in_repo.exists() else "yolov10n.pt"


@dataclass
class _Camera:
    last_number: int | None = None
    result: tuple[float, list[Detection]] | None = None
    pictures: int = 0
    last_picture_at: float | None = None
    error: str | None = None


class YoloEyes:
    """frames is a swarm.frames.Frames, such as a started StreamFrames. clock must be the one it uses."""

    def __init__(self, drones, frames, model=None, device: str = "cpu", image_size: int = IMAGE_SIZE, confidence: float = CONFIDENCE,
                 hfov: float = HFOV, latency: float = LATENCY_S, max_age: float = MAX_AGE_S, clock=time.monotonic):
        self.cameras = {str(drone): _Camera() for drone in drones}
        self.frames = frames
        self.model = model
        self.device, self.image_size, self.confidence = device, image_size, confidence
        self.hfov, self.latency, self.max_age, self.clock = hfov, latency, max_age, clock
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Load the detector now, so a missing package or file stops the flight before takeoff."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(weights_path())
        self.thread = threading.Thread(target=self._run, daemon=True, name="yolo-eyes")
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stopping.is_set():
            if not self.step():
                self.stopping.wait(IDLE_S)

    def step(self) -> bool:
        """Look at each drone's newest unseen picture once. True if there was any."""
        worked = False
        for drone_id in self.cameras:
            try:
                worked = self._see(drone_id) or worked
            except Exception as error:
                # One bad frame or lost stream must not blind the other drones.
                camera = self.cameras[drone_id]
                if camera.error != str(error):
                    logger.warning("[eyes] %s: %s", drone_id, error)
                camera.error = str(error)
        return worked

    def _see(self, drone_id: str) -> bool:
        camera = self.cameras[drone_id]
        frame = self.frames.latest(drone_id)
        if frame is None or frame.number == camera.last_number:
            return False
        camera.last_number = frame.number
        image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("a frame couldn't be decoded")
        detections = self._detect(image)
        with self.lock:
            camera.result = (frame.t, detections)
            camera.pictures += 1
            camera.last_picture_at = self.clock()
            camera.error = None
        return True

    def _detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        result = self.model.predict(image, imgsz=self.image_size, conf=self.confidence, verbose=False, device=self.device)[0]
        found = []
        for box, kind, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = self.model.names[int(kind)]
            if label not in LABELS or min(box[2] - box[0], box[3] - box[1]) < MIN_BOX_PX:
                continue
            found.append(detection_from_box(label, float(confidence), box, width, height, self.hfov))
        return found

    # ------------------------------------------------------------------ what the hub calls
    def look(self, drone_id: str, now: float) -> list[Detection] | None:
        camera = self.cameras.get(drone_id)
        if camera is None:
            return None
        with self.lock:
            result, camera.result = camera.result, None
        if result is None:
            return None
        age = self.clock() - result[0] + self.latency
        if age > self.max_age:
            return None
        return [replace(detection, age=age) for detection in result[1]]

    def knows(self, label: str) -> bool:
        return label in LABELS

    def health(self) -> dict:
        """For a pre-flight check: is every camera giving pictures the detector can use?"""
        now = self.clock()
        return {drone_id: {"pictures": c.pictures, "seconds_since_picture": None if c.last_picture_at is None else round(now - c.last_picture_at, 2),
                           "error": c.error} for drone_id, c in self.cameras.items()}
