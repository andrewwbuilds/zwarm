"""Where every drone is, from its own camera tilted down at AprilTags laid flat on the floor.

Each tag's position in the room is known, so seeing one gives the camera's position, height and
heading. Between sightings the pose is carried forward from what the drone was told to do, and its
uncertainty grows until the next sighting shrinks it. The coordinator widens its margins to match.

A drone on the floor can't see the floor: with the camera 45 degrees down, the top of the view
reaches only about 0.2 m ahead from 5 cm up. So every flight starts from its pad's position in the
swarm file, and the first real sighting comes during the climb.

Nothing here has been tested on a real drone. Accuracy figures come from the simulator's camera.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from swarm.control import Pose
from swarm.tag41h12 import TAGS, TOTAL_CELLS

logger = logging.getLogger(__name__)

DICTIONARY = cv2.aruco.DICT_4X4_50
MOVING_IDS = range(30, 40)
ANCHOR_IDS = range(40, 50)
MAX_TILT = math.radians(40)  # a marker tipped further than this is a misread: no drone flies like that
MAX_REPROJECTION_PX = 3.0
MOVING_STALE_S = 1.0


def detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), parameters)


def marker_image(marker_id: int, pixels: int = 800) -> np.ndarray:
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), marker_id, pixels)


def local_corners(size: float) -> np.ndarray:
    """A marker's corners in its own frame (x right, y up the printed page, z out), in OpenCV's order."""
    half = size / 2
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)


def flat_marker_corners(center, size: float, yaw: float = 0.0) -> np.ndarray:
    """World corners of a marker lying face up, with its printed top edge toward heading yaw."""
    up_page = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    return np.array([np.asarray(center, float) + x * right + y * up_page for x, y, _ in local_corners(size)])


@dataclass(frozen=True)
class CameraModel:
    """What a camera is (intrinsics) and where it is (world to camera)."""

    matrix: np.ndarray
    distortion: np.ndarray
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reprojection_px: float | None = None

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float) -> "CameraModel":
        """A rough lens model for testing. Run `calibrate-lens` before relying on heights."""
        focal = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
        return cls(matrix, np.zeros(5))

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        data = json.loads(Path(path).read_text())
        return cls(np.array(data["matrix"], np.float64), np.array(data["distortion"], np.float64),
                   None if data.get("rvec") is None else np.array(data["rvec"], np.float64),
                   None if data.get("tvec") is None else np.array(data["tvec"], np.float64),
                   data.get("reprojection_px"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "matrix": self.matrix.tolist(), "distortion": self.distortion.tolist(),
            "rvec": None if self.rvec is None else self.rvec.tolist(),
            "tvec": None if self.tvec is None else self.tvec.tolist(),
            "reprojection_px": self.reprojection_px,
        }, indent=1))


def lens_from_checkerboards(images: list[np.ndarray], inner_corners: tuple[int, int], square_m: float) -> CameraModel:
    """Intrinsics from photos of a checkerboard at varied angles."""
    columns, rows = inner_corners
    pattern = np.zeros((columns * rows, 3), np.float32)
    pattern[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    world, seen = [], []
    size = None
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, inner_corners)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        world.append(pattern)
        seen.append(corners)
    if len(seen) < 5:
        raise ValueError(f"the checkerboard was found in only {len(seen)} images; at least 5 are needed")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(world, seen, size, None, None)
    return CameraModel(matrix, distortion.ravel(), reprojection_px=float(error))


def locate_camera(image: np.ndarray, camera: CameraModel, anchors: dict[int, tuple], anchor_size: float) -> CameraModel:
    """Work out where a camera is from the anchors it can see."""
    corners, ids, _ = detector().detectMarkers(image)
    world, pixels = [], []
    for found, marker_id in zip(corners, () if ids is None else ids.ravel()):
        if int(marker_id) in anchors:
            world.append(flat_marker_corners(anchors[int(marker_id)], anchor_size))
            pixels.append(found.reshape(4, 2))
    if len(world) < 2:
        raise ValueError(f"only {len(world)} anchors in view; at least 2 are needed, and 4 spread out is better")
    world = np.concatenate(world)
    pixels = np.concatenate(pixels)
    ok, rvec, tvec = cv2.solvePnP(world, pixels, camera.matrix, camera.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("could not solve for the camera's position")
    projected, _ = cv2.projectPoints(world, rvec, tvec, camera.matrix, camera.distortion)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    return CameraModel(camera.matrix, camera.distortion, rvec.ravel(), tvec.ravel(), error)


def camera_position(camera: CameraModel) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(camera.rvec)
    return -rotation.T @ camera.tvec


def in_view(camera: CameraModel, point, width: int, height: int, margin_px: float = 60.0) -> bool:
    """Whether a marker at this world point would be fully inside this camera's image."""
    rotation, _ = cv2.Rodrigues(camera.rvec)
    if (rotation @ np.asarray(point, float) + camera.tvec)[2] <= 0:
        return False
    pixels, _ = cv2.projectPoints(np.asarray(point, float).reshape(1, 3), camera.rvec, camera.tvec,
                                  camera.matrix, camera.distortion)
    u, v = pixels.ravel()
    return margin_px <= u <= width - margin_px and margin_px <= v <= height - margin_px


@dataclass(frozen=True)
class Sighting:
    marker: int
    position: tuple[float, float, float]
    yaw: float
    size_px: float


def read_markers(image: np.ndarray, camera: CameraModel, marker_size: float, marker_detector=None) -> list[Sighting]:
    """Every marker in the image with a believable pose, in world coordinates."""
    corners, ids, _ = (marker_detector or detector()).detectMarkers(image)
    if ids is None:
        return []
    to_camera, _ = cv2.Rodrigues(camera.rvec)
    to_world = to_camera.T
    local = local_corners(marker_size)
    out = []
    for found, marker_id in zip(corners, ids.ravel()):
        marker_id = int(marker_id)
        if marker_id in ANCHOR_IDS:
            continue
        pixels = found.reshape(4, 2).astype(np.float64)
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            local, pixels, camera.matrix, camera.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        # A flat marker has two possible poses. Drones fly nearly level, so keep the one facing up.
        best = None
        for i in range(count):
            rotation, _ = cv2.Rodrigues(rvecs[i])
            world_rotation = to_world @ rotation
            tilt = math.acos(max(-1.0, min(1.0, float(world_rotation[2, 2]))))
            if best is None or tilt < best[0]:
                best = (tilt, world_rotation, tvecs[i].ravel(), float(np.ravel(errors)[i]))
        tilt, world_rotation, translation, error = best
        if tilt > MAX_TILT or error > MAX_REPROJECTION_PX:
            continue
        position = to_world @ (translation - camera.tvec)
        nose = world_rotation[:, 1]  # the marker's printed top edge
        side = float(np.mean(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1)))
        out.append(Sighting(marker_id, tuple(float(v) for v in position), math.atan2(nose[1], nose[0]), side))
    return out


class PoseStore:
    """The newest pose from any camera. Called from camera threads and read by the coordinator."""

    def __init__(self, marker_to_drone: dict[int, str], mount_height: float = 0.0):
        self.marker_to_drone = marker_to_drone
        self.mount_height = mount_height  # how far the marker sits above the floor when landed
        self.lock = threading.Lock()
        self.fresh: dict[str, Pose] = {}
        self.best_size: dict[str, tuple[float, float]] = {}
        self.moving: dict[int, Tracked] = {}
        self.counts: dict[str, int] = {}
        self.others = 0

    def add(self, t: float, sightings: list[Sighting]) -> None:
        with self.lock:
            for s in sightings:
                drone_id = self.marker_to_drone.get(s.marker)
                x, y, z = s.position
                if drone_id is not None:
                    # With overlapping cameras, a much smaller view of a recently seen marker is
                    # the less accurate one, so it doesn't replace the closer camera's.
                    size, at = self.best_size.get(drone_id, (0.0, -math.inf))
                    if s.size_px < 0.6 * size and t - at < 0.1:
                        continue
                    self.best_size[drone_id] = (s.size_px, t)
                    self.fresh[drone_id] = Pose(t, (x, y, z - self.mount_height), s.yaw)
                    self.counts[drone_id] = self.counts.get(drone_id, 0) + 1
                elif s.marker in MOVING_IDS:
                    tracked = self.moving.setdefault(s.marker, Tracked())
                    tracked.update(Pose(t, (x, y, z), s.yaw))
                else:
                    self.others += 1

    def poses(self) -> dict[str, Pose]:
        with self.lock:
            fresh, self.fresh = self.fresh, {}
            return fresh

    def moving_things(self, now: float, radius: float = 0.3) -> list[Moving]:
        with self.lock:
            return [Moving(f"marker-{marker}", tracked.now(now), tracked.v.copy(), radius)
                    for marker, tracked in self.moving.items() if tracked.age(now) < MOVING_STALE_S]


class CameraWorker:
    """Reads one camera as fast as it delivers, always working on the newest frame."""

    def __init__(self, source, camera: CameraModel, marker_size: float, store: PoseStore, clock=time.monotonic):
        self.source = source
        self.camera = camera
        self.marker_size = marker_size
        self.store = store
        self.clock = clock
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"camera-{source}")
        self.frames = 0
        self.last_frame = 0.0
        self.error: str | None = None
        self.width = int(camera.matrix[0, 2] * 2)  # replaced by the real size once a frame arrives
        self.height = int(camera.matrix[1, 2] * 2)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=2.0)

    def sees(self, point) -> bool:
        return in_view(self.camera, point, self.width, self.height)

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        marker_detector = detector()
        if not capture.isOpened():
            self.error = f"could not open camera {self.source}"
            logger.error("[tracking] %s", self.error)
            return
        while not self.stopping.is_set():
            ok, frame = capture.read()
            t = self.clock()
            if not ok:
                self.error = f"camera {self.source} stopped delivering frames"
                self.stopping.wait(0.1)
                continue
            self.error = None
            self.height, self.width = frame.shape[:2]
            self.frames += 1
            self.last_frame = t
            self.store.add(t, read_markers(frame, self.camera, self.marker_size, marker_detector))
        capture.release()
