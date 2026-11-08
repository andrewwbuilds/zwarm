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

TAG_FAMILY = "tagStandard41h12"
VIDEO_WIDTH, VIDEO_HEIGHT = 1184, 656  # measured: the S20's JPEG size
DEFAULT_FOV_DEG = 90.0  # estimate: horizontal
DEFAULT_TILT_DEG = 45.0
CAMERA_HEIGHT_M = 0.03  # estimate: lens above the floor when landed, taken off every height reading

MAX_BODY_TILT = math.radians(35)  # more than this is a misread: nothing flies at this angle
MAX_REPROJECTION_PX = 4.0
MAX_READ_RANGE_M = 4.0  # no real sighting is further than this; see read_range.py for what's usable
READABLE_M = 1.8  # straight-line distance a tag is counted on being read from, from 1 m up


def tag_image(tag_id: int, cell_pixels: int = 40, margin_cells: int = 2) -> np.ndarray:
    """A printed tag as the simulator draws it, with white paper around the whole 9-cell tag."""
    cells = TOTAL_CELLS + 2 * margin_cells
    image = np.full((cells, cells), 255, np.uint8)
    rows = TAGS[tag_id]
    for r in range(TOTAL_CELLS):
        for c in range(TOTAL_CELLS):
            if rows[r * TOTAL_CELLS + c] == "#":
                image[margin_cells + r, margin_cells + c] = 0
    return cv2.resize(image, None, fx=cell_pixels, fy=cell_pixels, interpolation=cv2.INTER_NEAREST)


def local_corners(size: float) -> np.ndarray:
    """A marker's corners in its own frame (x right, y up the printed page, z out), in OpenCV's order."""
    half = size / 2
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)


class SwarmFileError(ValueError):
    pass


@dataclass(frozen=True)
class Anchor:
    """A tag lying flat on the floor. `size` is the edge of its inner white square, not of the whole tag."""

    id: str  # "tag:12"
    x: float
    y: float
    size: float
    yaw_deg: float = 0.0  # the direction the top of the printed page points, 0 being +x

    @property
    def tag(self) -> int:
        return int(self.id.partition(":")[2])

    def corners(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        up_page = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        center = np.array([self.x, self.y, 0.0])
        return np.array([center + x * right + y * up_page for x, y, _ in local_corners(self.size)])


def load_anchors(section: dict) -> dict[str, Anchor]:
    """The swarm file's "anchors" section. Refuses a missing size, because every position scales with it."""
    if not isinstance(section, dict):
        raise SwarmFileError('"anchors" must be an object with tag_family, white_square_edge_m and tags')
    family = section.get("tag_family")
    if family != TAG_FAMILY:
        raise SwarmFileError(f"tag_family must be {TAG_FAMILY!r}, got {family!r}")
    edge = section.get("white_square_edge_m")
    if edge is None:
        raise SwarmFileError("white_square_edge_m is missing. Measure the white square inside a printed tag's black "
                             "ring, in metres. It is 5/9 of the tag's width, and every position and height scales with it.")
    edge = float(edge)
    if not 0.02 <= edge <= 0.5:
        raise SwarmFileError(f"white_square_edge_m is {edge}, which isn't a plausible size in metres")
    anchors = {}
    for entry in section.get("tags", ()):
        tag = int(entry["tag"])
        anchor = Anchor(f"tag:{tag}", float(entry["x"]), float(entry["y"]), edge, float(entry.get("rotation_deg", 0.0)))
        if anchor.id in anchors:
            raise SwarmFileError(f"tag {tag} is listed twice")
        anchors[anchor.id] = anchor
    if not anchors:
        raise SwarmFileError('"anchors" lists no tags')
    return anchors


def tag_grid(origin=(-2.0, 2.0), spacing: float = 1.0, columns: int = 5, rows: int = 5, start_id: int = 0,
             across_deg: float = -90.0, rotation_deg: float = 0.0) -> list[dict]:
    """Tags in reading order for someone standing at the near edge, facing +x.

    The first tag is near-left. Ids rise to the right along each row (-y), and the next row is further
    away (+x), so with the defaults id 0 is at (-2, 2) and id 24 at (2, -2), every tag's top toward +x.
    """
    across = (math.cos(math.radians(across_deg)), math.sin(math.radians(across_deg)))
    down = (-across[1], across[0])
    return [{"tag": start_id + r * columns + c,
             "x": round(origin[0] + c * spacing * across[0] + r * spacing * down[0], 3),
             "y": round(origin[1] + c * spacing * across[1] + r * spacing * down[1], 3),
             "rotation_deg": rotation_deg}
            for r in range(rows) for c in range(columns)]


def anchors_section(edge: float | None, tags: list[dict]) -> dict:
    return {"tag_family": TAG_FAMILY, "white_square_edge_m": edge, "tags": tags}


def grid_layout(lo, hi, spacing: float, size: float, first_id: int = 0) -> list[Anchor]:
    """Tags in a centred grid over a fence, as many as fit. For simulated rooms."""
    columns = int((hi[0] - lo[0]) // spacing) + 1
    rows = int((hi[1] - lo[1]) // spacing) + 1
    if first_id + columns * rows > len(TAGS):
        raise ValueError(f"{columns} x {rows} needs {columns * rows} tags; only {len(TAGS)} exist. Use wider spacing.")
    origin = ((lo[0] + hi[0] - (columns - 1) * spacing) / 2, (lo[1] + hi[1] - (rows - 1) * spacing) / 2)
    return [Anchor(f"tag:{t['tag']}", t["x"], t["y"], size) for t in tag_grid(origin, spacing, columns, rows, first_id, 0.0)]


@dataclass(frozen=True)
class Lens:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int
    reprojection_px: float | None = None

    @classmethod
    def from_fov(cls, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT, horizontal_fov_deg: float = DEFAULT_FOV_DEG) -> "Lens":
        """An ideal lens. The real S20's field of view is an estimate until `calibrate-lens` is run."""
        return cls.from_focal((width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2), width, height)

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
