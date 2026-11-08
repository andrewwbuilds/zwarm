"""Floor markers in the simulator's camera, tilted as on the drone, through the S20's JPEG size.

Frames get what the real video should do to them: a low effective resolution, motion blur, and
compression down to the 34 KB a real frame measures. Exposure and sharpness are unmeasured guesses.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from sim.camera import HEIGHT, SIZES, WIDTH, Camera, Pose as CameraPose, VideoEncoder
from swarm.control import Pose
from swarm.tag41h12 import INNER_CELLS, TOTAL_CELLS
from swarm.tracking import Anchor, InsideOutTracker, Lens, anchors_in_view, tag_image

MARGIN_CELLS = 2  # white paper around the printed tag
RATE = 17.6  # measured: the S20's frames per second


@dataclass(frozen=True)
class VideoConditions:
    render_scale: float = 0.54  # 640 x 360, what the video header advertises. Unmeasured sharpness
    exposure_s: float = 1 / 60  # a guess for indoor light
    latency_s: float = 0.15  # frame taken to frame processed: real video arrives 100 to 200 ms late
    assumed_latency_s: float = 0.15  # what the tracker believes that to be
    fov_deg: float = 90.0  # what the simulated lens really is; the tracker may assume otherwise
    blur_sigma: float = 0.0  # extra softness at full frame size
    jpeg_bytes: int = SIZES[1]


def tilted(rotation, tilt: float) -> tuple:
    """Body-to-world rotation with the camera pitched down by tilt about the body's left axis."""
    c, s = math.cos(tilt), math.sin(tilt)
    r = rotation
    return (r[0] * c - r[2] * s, r[1], r[0] * s + r[2] * c,
            r[3] * c - r[5] * s, r[4], r[3] * s + r[5] * c,
            r[6] * c - r[8] * s, r[7], r[6] * s + r[8] * c)


def paper_scale() -> float:
    """How much wider the drawn paper is than the inner white square that `size` measures."""
    return (TOTAL_CELLS + 2 * MARGIN_CELLS) / INNER_CELLS


class FloorCamera:
    """Renders what a drone's tilted camera sees, with anchors painted on the floor."""

    def __init__(self, scene, anchors: dict[str, Anchor], tilt_deg: float, video: VideoConditions, seed: int = 0):
        self.scene = scene
        self.anchors = anchors
        self.tilt = math.radians(tilt_deg)
        self.video = video
        self.camera = Camera(scene, render_scale=video.render_scale, fov=math.radians(video.fov_deg), seed=seed)
        self.tiles = {key: tag_image(anchor.tag, 30, MARGIN_CELLS) for key, anchor in anchors.items()}
        self.scaled: dict = {}
        self.encoder = VideoEncoder(seed=seed)
        self.width, self.height = self.camera.width, self.camera.height
        focal = (self.width / 2) / self.camera.half
        self.matrix = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]], np.float64)

    def _hidden(self, anchor: Anchor, origin) -> bool:
        """Is something solid between the camera and this marker, or is it under furniture?"""
        target = np.array([anchor.x, anchor.y, 0.0])
        for box in self.scene.boxes:
            if not box.solid:
                continue
            lo, hi = np.array(box.lo), np.array(box.hi)
            direction = target - origin
            with np.errstate(divide="ignore", invalid="ignore"):
                a, b = (lo - origin) / direction, (hi - origin) / direction
            near = np.nanmax(np.minimum(a, b))
            far = np.nanmin(np.maximum(a, b))
            if near <= far and far > 0.0 and near < 1.0:
                return True
        return False

    def _paint(self, image: np.ndarray, x: float, y: float, z: float, to_camera: np.ndarray) -> None:
        origin = np.array([x, y, z])
        for key, anchor in self.anchors.items():
            if self._hidden(anchor, origin):
                continue
            corners = anchor.corners()
            center = corners.mean(0)
            outer = center + (corners - center) * paper_scale()
            points = to_camera @ (outer - origin).T
            if np.any(points[2] <= 0.03):
                continue
            u = self.matrix[0, 0] * points[0] / points[2] + self.matrix[0, 2]
            v = self.matrix[1, 1] * points[1] / points[2] + self.matrix[1, 2]
            if u.max() < 0 or u.min() > self.width or v.max() < 0 or v.min() > self.height:
                continue
            target = np.stack([u, v], axis=1).astype(np.float32)
            tile = self.tiles[key]
            side = max(float(np.linalg.norm(target[0] - target[1])), float(np.linalg.norm(target[1] - target[2])))
            pixels = int(np.clip(side * 2, 26, tile.shape[0]) // 13 * 13)  # whole pixels per cell, so renders can share
            small = self.scaled.get((key, pixels))
            if small is None:
                small = cv2.resize(tile, (pixels, pixels), interpolation=cv2.INTER_AREA)
                self.scaled[(key, pixels)] = small
            square = np.array([[0, 0], [pixels, 0], [pixels, pixels], [0, pixels]], np.float32)
            x0, y0 = max(int(u.min()) - 1, 0), max(int(v.min()) - 1, 0)
            x1, y1 = min(int(u.max()) + 2, self.width), min(int(v.max()) + 2, self.height)
            target = target - np.array([x0, y0], np.float32)
            warp = cv2.getPerspectiveTransform(square, target)
            painted = cv2.warpPerspective(small, warp, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full_like(small, 255), warp, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            patch = image[y0:y1, x0:x1]
            image[y0:y1, x0:x1] = patch * (1 - alpha) + (painted[..., None].astype(np.float32) / 255.0 * 0.85) * alpha

    def render(self, x: float, y: float, z: float, rotation, speed_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """A grayscale frame as it would decode from the drone's JPEG."""
        rot = tilted(rotation, self.tilt)
        pose = CameraPose(x, y, z, rot)
        color = self.camera.shade(pose)
        # Camera axes in world terms: x right is the rotation's -left column, y down is -up, z is forward.
        to_camera = np.array([[-rot[1], -rot[4], -rot[7]], [-rot[2], -rot[5], -rot[8]], [rot[0], rot[3], rot[6]]])
        self._paint(color, x, y, z, to_camera)
        color *= self.camera.vignette
        dy, dx = self.camera.rng.integers(0, 64, 2)
        color += self.camera.grain[dy:dy + color.shape[0], dx:dx + color.shape[1]]
        small = np.clip(color * 255, 0, 255).astype(np.uint8)
        image = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        image = _blur(image, speed_px[0] * self.video.exposure_s, speed_px[1] * self.video.exposure_s)
        if self.video.blur_sigma:
            image = cv2.GaussianBlur(image, (0, 0), self.video.blur_sigma)
        jpeg = self.encoder.compress(image, self.video.jpeg_bytes)
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)


def _blur(image: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    length = math.hypot(dx_px, dy_px)
    if length < 1.0:
        return image
    size = int(length) | 1
    kernel = np.zeros((size, size), np.float32)
    middle = size // 2
    ux, uy = dx_px / length, dy_px / length
    a = (int(round(middle - ux * length / 2)), int(round(middle - uy * length / 2)))
    b = (int(round(middle + ux * length / 2)), int(round(middle + uy * length / 2)))
    cv2.line(kernel, a, b, 1.0, 1)
    return cv2.filter2D(image, -1, kernel / kernel.sum())


class SimVideo:
    """Feeds an InsideOutTracker from a SimFleet: renders a frame when one is due, hands it over late."""

    def __init__(self, fleet, tracker: InsideOutTracker, scene, video: VideoConditions | None = None, seed: int = 0, rate: float = RATE):
        self.rate = rate
        self.fleet = fleet
        self.tracker = tracker
        self.video = video or VideoConditions()
        self.cameras = {k: FloorCamera(scene, tracker.anchors, tracker.drones[k].tilt_deg, self.video, seed + i)
                        for i, k in enumerate(tracker.drones)}
        self.next_frame = {k: i * 0.017 for i, k in enumerate(tracker.drones)}
        self.pending: deque = deque()
        self.rendered = 0
        self.culled = 0
        self.lens = Lens.from_fov(horizontal_fov_deg=self.video.fov_deg)

    def step(self) -> None:
        t = self.fleet.t
        for key, camera in self.cameras.items():
            while t >= self.next_frame[key]:
                self.next_frame[key] += 1 / self.rate
                if t < self.fleet.hidden.get(key, 0.0):
                    continue
                self.capture(key, camera, t)
        while self.pending and self.pending[0][0] <= t:
            _, key, taken, image = self.pending.popleft()
            self.tracker.add_frame(key, taken + self.video.latency_s - self.video.assumed_latency_s, image)

    def capture(self, key: str, camera: FloorCamera, t: float) -> None:
        drone = self.fleet.drones[key]
        q = drone.sim.drone
        optics = self.tracker.drones[key]
        # Skip the render when geometry says no marker could be in frame. It would only show floor.
        # A real detector can also report markers that aren't there; this never exercises that.
        wide = anchors_in_view(self.tracker.anchors, self.lens, math.radians(optics.tilt_deg), (q.x, q.y, q.z), q.yaw(),
                               max_range=4.0, margin_px=-80.0)
        if not wide:
            self.culled += 1
            return
        self.rendered += 1
        focal = self.lens.matrix[0, 0]
        depth = max(q.z / math.sin(math.radians(optics.tilt_deg)), 0.3)
        speed_px = (math.hypot(q.vx, q.vy) * focal / depth + abs(q.r) * focal, 0.0)
        image = camera.render(q.x, q.y, q.z, q.rotation(), speed_px)
        self.pending.append((t + self.video.latency_s, key, t, image))


def truth_pose(drone) -> Pose:
    q = drone.sim.drone
    return Pose(0.0, (q.x, q.y, q.z - drone.sim.spec.gear_height_m), q.yaw())
