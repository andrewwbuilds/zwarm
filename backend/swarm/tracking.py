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
    def from_focal(cls, focal: float, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT) -> "Lens":
        return cls(np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64), np.zeros(5), width, height)

    @classmethod
    def load(cls, path: str | Path) -> "Lens":
        data = json.loads(Path(path).read_text())
        return cls(np.array(data["matrix"], np.float64), np.array(data["distortion"], np.float64),
                   int(data["width"]), int(data["height"]), data.get("reprojection_px"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"matrix": self.matrix.tolist(), "distortion": self.distortion.tolist(),
                                          "width": self.width, "height": self.height,
                                          "reprojection_px": self.reprojection_px}, indent=1))

    def for_size(self, width: int, height: int) -> "Lens":
        if width == self.width and height == self.height:
            return self
        matrix = self.matrix.copy()
        matrix[0] *= width / self.width
        matrix[1] *= height / self.height
        return Lens(matrix, self.distortion, width, height, self.reprojection_px)


def lens_from_checkerboards(images: list[np.ndarray], inner_corners: tuple[int, int], square_m: float) -> Lens:
    """A lens model from photos of a checkerboard at varied angles."""
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
    return Lens(matrix, distortion.ravel(), size[0], size[1], float(error))


class AnchorReader:
    """Finds known tags in a frame. This is the only place a detector is called, for real and simulated
    frames alike. Not thread safe: each drone has its own."""

    MAX_HAMMING = 1  # corrected bits accepted: more risks reading the wrong tag, which is a wrong position
    MIN_MARGIN = 15.0  # how clearly the bits read

    def __init__(self, anchors: dict[str, Anchor], decimate: float = 1.0):
        from pupil_apriltags import Detector  # only the machine running the swarm needs this

        self.anchors = anchors
        self.detector = Detector(families=TAG_FAMILY, quad_decimate=decimate, nthreads=1)
        self.unknown: set[int] = set()

    def find(self, gray: np.ndarray) -> list[tuple[Anchor, np.ndarray]]:
        out = []
        for detection in self.detector.detect(gray):
            if detection.hamming > self.MAX_HAMMING or detection.decision_margin < self.MIN_MARGIN:
                continue
            anchor = self.anchors.get(f"tag:{detection.tag_id}")
            if anchor is None:
                self.unknown.add(int(detection.tag_id))
                continue
            # The detector goes bottom-left, bottom-right, top-right, top-left; local_corners starts top-left.
            out.append((anchor, np.asarray(detection.corners, np.float64)[::-1].copy()))
        return out


@dataclass(frozen=True)
class Fix:
    """One frame's measurement of where the drone is."""

    t: float
    p: tuple[float, float, float]
    yaw: float
    anchors: tuple[str, ...]
    range_m: float
    reprojection_px: float
    sigma_xy: float
    sigma_z: float
    sigma_yaw: float


def _body_from_camera(to_world: np.ndarray, tilt: float) -> tuple[float, float]:
    """(heading, tilt of the body away from level), undoing the camera's mount angle.

    Uses both the camera's forward and up axes, so it still works looking straight down.
    """
    forward = to_world[:, 2]
    up = -to_world[:, 1]
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    nose = cos_t * forward + sin_t * up
    body_up = -sin_t * forward + cos_t * up
    return math.atan2(nose[1], nose[0]), math.acos(max(-1.0, min(1.0, float(body_up[2]))))


def _fix_noise(count: int, range_m: float, side_px: float) -> tuple[float, float, float]:
    """How far off a sighting may be. Fitted to the simulator's camera; see docs/research."""
    side = max(side_px, 8.0)
    if count >= 2:
        return 0.015 + 0.012 * range_m, 0.02 + 0.015 * range_m, math.radians(1.5)
    pixel_angle = 1.0 / side
    return 0.02 + 0.5 * range_m * pixel_angle, 0.03 + 0.5 * range_m * pixel_angle, math.radians(2.0) + 0.5 * pixel_angle


def solve_fix(seen: list[tuple[Anchor, np.ndarray]], lens: Lens, tilt: float, t: float, prior_yaw: float | None = None,
              fence=None) -> Fix | None:
    """The drone's pose from the anchors in one frame, or None if nothing believable comes out."""
    if not seen:
        return None
    world = np.concatenate([anchor.corners() for anchor, _ in seen])
    pixels = np.concatenate([quad for _, quad in seen])
    try:
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(world, pixels, lens.matrix, lens.distortion, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    # A flat marker has two possible poses. Keep the one with the body level, and if both are
    # nearly level, the one that agrees with where we thought the nose was pointing.
    best, best_score = None, math.inf
    for i in range(count):
        rotation, _ = cv2.Rodrigues(rvecs[i])
        to_world = rotation.T
        heading, body_tilt = _body_from_camera(to_world, tilt)
        if body_tilt > MAX_BODY_TILT:
            continue
        position = -to_world @ tvecs[i].ravel()
        if position[2] <= 0.0:
            continue
        score = body_tilt
        if prior_yaw is not None:
            score += 0.5 * abs((heading - prior_yaw + math.pi) % (2 * math.pi) - math.pi)
        if score < best_score:
            best, best_score = (rvecs[i], tvecs[i]), score
    if best is None:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(world, pixels, lens.matrix, lens.distortion, best[0], best[1])
    rotation, _ = cv2.Rodrigues(rvec)
    to_world = rotation.T
    heading, body_tilt = _body_from_camera(to_world, tilt)
    position = -to_world @ tvec.ravel()
    projected, _ = cv2.projectPoints(world, rvec, tvec, lens.matrix, lens.distortion)
    reprojection = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    if body_tilt > MAX_BODY_TILT or reprojection > MAX_REPROJECTION_PX or position[2] <= 0.0:
        return None
    if fence is not None and (np.any(position[:2] < np.asarray(fence[0])[:2] - 1.5) or np.any(position[:2] > np.asarray(fence[1])[:2] + 1.5)):
        return None
    centers = np.array([[anchor.x, anchor.y, 0.0] for anchor, _ in seen])
    range_m = float(np.min(np.linalg.norm(centers - position, axis=1)))
    if range_m > MAX_READ_RANGE_M:
        return None
    side = float(np.max([np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) for _, quad in seen]))
    sigma_xy, sigma_z, sigma_yaw = _fix_noise(len(seen), range_m, side)
    return Fix(t, (float(position[0]), float(position[1]), float(position[2]) - CAMERA_HEIGHT_M), heading,
               tuple(anchor.id for anchor, _ in seen), range_m, reprojection, sigma_xy, sigma_z, sigma_yaw)


def anchors_in_view(anchors: dict[str, Anchor], lens: Lens, tilt: float, p, yaw: float, max_range: float = READABLE_M,
                    margin_px: float = 20.0) -> list[str]:
    """Which anchors should be fully in view and readable from a pose, by geometry alone."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    nose = np.array([cos_y, sin_y, 0.0])
    left = np.array([-sin_y, cos_y, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    forward = cos_t * nose - sin_t * world_up
    down = -(sin_t * nose + cos_t * world_up)
    right = -left
    to_camera = np.array([right, down, forward])
    camera = np.array([p[0], p[1], p[2] + CAMERA_HEIGHT_M])
    out = []
    for anchor in anchors.values():
        points = to_camera @ (anchor.corners() - camera).T
        if np.any(points[2] <= 0.02):
            continue
        if np.linalg.norm(np.array([anchor.x, anchor.y, 0.0]) - camera) > max_range:
            continue
        u = lens.matrix[0, 0] * points[0] / points[2] + lens.matrix[0, 2]
        v = lens.matrix[1, 1] * points[1] / points[2] + lens.matrix[1, 2]
        if u.min() >= margin_px and u.max() <= lens.width - margin_px and v.min() >= margin_px and v.max() <= lens.height - margin_px:
            out.append(anchor.id)
    return out


# Uncertainty growth while carrying the pose forward. Estimates, widened on purpose.
HOLD_DRIFT = 0.04  # m/s: position hold with neutral sticks still wanders
SPEED_ERROR = 0.3  # share of commanded speed the drone may not actually be doing
YAW_DRIFT = math.radians(1.5)  # rad/s with no yaw stick
PAD_SIGMA = 0.1  # how carefully a drone is expected to be set on its pad
PAD_YAW_SIGMA = math.radians(15)
FIX_MODEL_AFTER_S = 0.4  # with no sighting for this long, trust the command for speed
COMMAND_EXPIRES_S = 0.3  # the backend centres the sticks this long after the last command
RESPONSE_S = 0.5
GATE_M = 0.45
LANDING_DESCENT = 0.45  # m/s, an estimate of the stock auto-land


class Estimator:
    """One drone's pose. Sightings arrive late, so each is compared with where we thought it was then."""

    def __init__(self, x: float, y: float, yaw: float):
        self.p = np.array([x, y, 0.0])
        self.v = np.zeros(3)
        self.yaw = yaw
        self.sigma_xy = PAD_SIGMA
        self.sigma_z = 0.03
        self.sigma_yaw = PAD_YAW_SIGMA
        self.command = np.zeros(3)
        self.yaw_rate = 0.0
        self.flying = False
        self.landing = False
        self.t: float | None = None
        self.last_fix: Fix | None = None
        self.fix_at = -math.inf
        self.history: deque = deque(maxlen=120)
        self.rejected: list[Fix] = []
        self.fixes = 0
        self.rejections = 0

    def advance(self, now: float) -> None:
        if self.t is None:
            self.t = now
        dt = now - self.t
        if dt <= 0.0:
            return
        self.t = now
        if not self.flying:
            self.v = np.zeros(3)
            self.history.append((now, self.p.copy(), self.yaw))
            return
        if now - self.fix_at > FIX_MODEL_AFTER_S:
            self.v += (self.command - self.v) * min(1.0, dt / RESPONSE_S)
            if self.landing:
                self.v[2] = -LANDING_DESCENT
            else:
                self.v[2] = self.command[2]
        self.p = self.p + self.v * dt
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = 0.0
        self.yaw += self.yaw_rate * dt
        self.history.append((now, self.p.copy(), self.yaw))
        speed = float(np.linalg.norm(self.v[:2]))
        self.sigma_xy += (HOLD_DRIFT + SPEED_ERROR * speed + self.sigma_yaw * speed) * dt
        self.sigma_z += (HOLD_DRIFT + 0.3 * abs(self.v[2])) * dt
        self.sigma_yaw += (YAW_DRIFT + 0.25 * abs(self.yaw_rate)) * dt

    def note_command(self, velocity, yaw_rate: float, flying: bool, landing: bool) -> None:
        self.command = np.asarray(velocity, float).copy()
        self.yaw_rate = yaw_rate
        self.flying = flying
        self.landing = landing

    def add_fix(self, fix: Fix) -> bool:
        """Blend in a sighting. A wild one is ignored, unless several in a row agree with each other."""
        then = None
        for t, p, yaw in reversed(self.history):
            if t <= fix.t:
                then = (p, yaw)
                break
        if then is None:
            then = (self.p, self.yaw)
        measured = np.array(fix.p)
        gap = measured - then[0]
        reach = max(GATE_M, 4 * math.hypot(self.sigma_xy, fix.sigma_xy))
        if float(np.linalg.norm(gap[:2])) > reach:
            self.rejections += 1
            self.rejected.append(fix)
            del self.rejected[:-3]
            spread = max((np.linalg.norm(np.array(r.p[:2]) - measured[:2]) for r in self.rejected), default=0.0)
            if len(self.rejected) < 3 or spread > 0.25:
                return False
            # Three that agree: it's the estimate that's wrong, so start again from them.
            self.sigma_xy = max(self.sigma_xy, reach)
        self.rejected.clear()
        previous = self.last_fix
        self.last_fix = fix
        self.fix_at = fix.t
        self.fixes += 1
        gain_xy = self.sigma_xy ** 2 / (self.sigma_xy ** 2 + fix.sigma_xy ** 2)
        gain_z = self.sigma_z ** 2 / (self.sigma_z ** 2 + fix.sigma_z ** 2)
        gain_yaw = self.sigma_yaw ** 2 / (self.sigma_yaw ** 2 + fix.sigma_yaw ** 2)
        change = np.array([gain_xy * gap[0], gain_xy * gap[1], gain_z * gap[2]])
        self.p = self.p + change
        if self.p[2] < 0.0:
            self.p[2] = 0.0
        self.history = deque(((t, p + change, yaw) for t, p, yaw in self.history), maxlen=self.history.maxlen)
        if previous is not None and 0.0 < fix.t - previous.t < 0.5:
            dt = fix.t - previous.t
            self.v = self.v + 0.25 * np.array([gain_xy * gap[0], gain_xy * gap[1], gain_z * gap[2]]) / max(dt, 0.04)
        turn = (fix.yaw - then[1] + math.pi) % (2 * math.pi) - math.pi
        self.yaw += gain_yaw * turn
        self.sigma_xy = math.sqrt(1 - gain_xy) * self.sigma_xy
        self.sigma_z = math.sqrt(1 - gain_z) * self.sigma_z
        self.sigma_yaw = math.sqrt(1 - gain_yaw) * self.sigma_yaw
        return True

    def pose(self, now: float) -> Pose:
        return Pose(now, tuple(float(v) for v in self.p), self.yaw, sigma=max(self.sigma_xy, 0.5 * self.sigma_z),
                    v=self.v.copy(), fix_age=now - self.fix_at)


@dataclass(frozen=True)
class DroneOptics:
    x: float
    y: float
    yaw: float = 0.0
    tilt_deg: float = DEFAULT_TILT_DEG
    lens: Lens | None = None


class InsideOutTracker:
    """Every drone's Estimator, and what feeds it. Frames can arrive on any thread."""

    def __init__(self, drones: dict[str, DroneOptics], anchors: dict[str, Anchor], fence=None, video_latency: float = 0.15):
        self.drones = drones
        self.anchors = anchors
        self.fence = fence
        self.video_latency = video_latency  # receiving a frame to when it was really taken: an estimate
        self.lock = threading.Lock()
        self.estimators = {k: Estimator(d.x, d.y, d.yaw) for k, d in drones.items()}
        self.readers = {k: AnchorReader(anchors) for k in drones}
        self.seen: dict[str, tuple[str, ...]] = {k: () for k in drones}
        self.seen_at: dict[str, float] = {k: -math.inf for k in drones}
        self.frames: dict[str, int] = {k: 0 for k in drones}
        self.stopping = threading.Event()
        self.threads: list[threading.Thread] = []

    def add_frame(self, drone_id: str, t: float, image: np.ndarray) -> Fix | None:
        """Process one frame, taken at time t. Safe to call from a camera thread."""
        optics = self.drones[drone_id]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        lens = (optics.lens or Lens.from_fov()).for_size(gray.shape[1], gray.shape[0])
        found = self.readers[drone_id].find(gray)
        with self.lock:
            prior = self.estimators[drone_id].yaw
        fix = solve_fix(found, lens, math.radians(optics.tilt_deg), t, prior, self.fence)
        with self.lock:
            self.frames[drone_id] += 1
            if found:
                self.seen[drone_id] = tuple(anchor.id for anchor, _ in found)
                self.seen_at[drone_id] = t
            if fix is not None:
                self.estimators[drone_id].add_fix(fix)
        return fix

    def poses(self, now: float) -> dict[str, Pose]:
        with self.lock:
            out = {}
            for k, estimator in self.estimators.items():
                estimator.advance(now)
                out[k] = estimator.pose(now)
            return out

    def note_command(self, drone_id: str, velocity, yaw_rate: float, flying: bool, landing: bool) -> None:
        with self.lock:
            self.estimators[drone_id].note_command(velocity, yaw_rate, flying, landing)

    def reset(self, drone_id: str) -> None:
        """The drone has been put back on its pad by hand."""
        d = self.drones[drone_id]
        with self.lock:
            self.estimators[drone_id] = Estimator(d.x, d.y, d.yaw)

    def expected(self, drone_id: str, height: float = 1.0) -> list[str]:
        """The anchors this drone should see hovering over its pad."""
        d = self.drones[drone_id]
        return anchors_in_view(self.anchors, d.lens or Lens.from_fov(), math.radians(d.tilt_deg), (d.x, d.y, height), d.yaw)

    def report(self, now: float) -> dict[str, dict]:
        with self.lock:
            return {k: {
                "seeing": list(self.seen[k]) if now - self.seen_at[k] < 1.0 else [],
                "last_seen_s": None if not math.isfinite(self.seen_at[k]) else round(now - self.seen_at[k], 2),
                "frames": self.frames[k],
                "fixes": self.estimators[k].fixes,
                "rejected": self.estimators[k].rejections,
                "sigma_m": round(self.estimators[k].sigma_xy, 3),
            } for k in self.drones}

    def allowed_yaw(self, drone_id: str, p, yaw: float) -> float:
        """The nearest heading to yaw from which this drone would still see a tag."""
        d = self.drones[drone_id]
        lens = d.lens or Lens.from_fov()
        tilt = math.radians(d.tilt_deg)
        for step in range(0, 181, 15):
            for sign in (1, -1):
                turned = yaw + sign * math.radians(step)
                if anchors_in_view(self.anchors, lens, tilt, p, turned):
                    return turned
        return yaw

    def start(self, frames) -> None:
        """Read each drone's newest frame from `frames.latest(drone_id)`, which swarm.frames provides."""
        for drone_id in self.drones:
            thread = threading.Thread(target=self._read, args=(drone_id, frames), daemon=True, name=f"anchors-{drone_id}")
            thread.start()
            self.threads.append(thread)

    def _read(self, drone_id: str, frames) -> None:
        last = None
        while not self.stopping.is_set():
            frame = frames.latest(drone_id)
            if frame is None or frame.number == last:
                self.stopping.wait(0.005)
                continue
            last = frame.number
            image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                self.add_frame(drone_id, frame.t - self.video_latency, image)

    def stop(self) -> None:
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=2.0)


def fit_focal(frames: list[list[tuple[Anchor, np.ndarray]]], width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT,
              guess_fov_deg: float = DEFAULT_FOV_DEG) -> dict:
    """The focal length that best explains tags at known spacing, from frames showing two or more.

    Tags that all lie flat only pin down focal length when seen at an angle, which a tilted camera does.
    Corners that disagree with the room file's spacing show up here as a large remaining error.
    """
    from scipy.optimize import minimize_scalar

    usable = [seen for seen in frames if len(seen) >= 2]
    if not usable:
        raise ValueError("no frame showed two or more known tags together; hold the drone about 1 m up over the grid")
    guess = (width / 2) / math.tan(math.radians(guess_fov_deg) / 2)

    def error_at(focal: float) -> float:
        lens = Lens.from_focal(focal, width, height)
        total, count = 0.0, 0
        for seen in usable:
            world = np.concatenate([anchor.corners() for anchor, _ in seen])
            pixels = np.concatenate([quad for _, quad in seen])
            found, rvecs, tvecs, _ = cv2.solvePnPGeneric(world, pixels, lens.matrix, lens.distortion, flags=cv2.SOLVEPNP_IPPE)
            if not found:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(world, pixels, lens.matrix, lens.distortion, rvecs[0], tvecs[0])
            projected, _ = cv2.projectPoints(world, rvec, tvec, lens.matrix, lens.distortion)
            total += float(np.sum((projected.reshape(-1, 2) - pixels) ** 2))
            count += len(pixels)
        return total / max(count, 1)

    result = minimize_scalar(error_at, bounds=(guess * 0.5, guess * 1.8), method="bounded", options={"xatol": 0.5})
    return {"focal_px": round(float(result.x), 1), "error_px": round(math.sqrt(float(result.fun)), 2), "frames": len(usable),
            "error_at_guess_px": round(math.sqrt(error_at(guess)), 2), "width": width, "height": height}
