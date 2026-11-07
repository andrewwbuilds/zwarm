"""The S20's front camera: renders the scene, JPEG-encodes it, and wraps it the way the drone does.

Wire format sources: tests/fixtures/s20_capture.json (measured) and docs/research/S2x.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from sim import textures
from sim.config import DroneSpec

WIDTH, HEIGHT = 1184, 656  # measured: JPEG size
ADVERTISED = (640, 360)  # measured: what the header claims
FOV = math.radians(DroneSpec().camera_fov_deg)
CHUNK = 1400  # measured
BODY = CHUNK - 10
HEADER = 40  # measured: the JPEG starts after this
SIZES = (34041, 34197, 34357)  # measured: min, median, max JPEG bytes
CLOCK = 0x000561ABDEE8A672  # measured: first captured header timestamp, microseconds
RATE = 17.6  # measured: 106 camera frames in 6.03 s
SKIP = 0.04  # measured: 102 of those 106 frames were sent

LIGHT = np.asarray([0.80, 0.86, 0.76, 0.92, 0.72, 1.0], np.float32)  # estimate: by face -x +x -y +y -z +z
SKY_LOW = np.asarray([0.93, 0.86, 0.78], np.float32)  # BGR
SKY_HIGH = np.asarray([0.85, 0.62, 0.42], np.float32)
DARK = np.asarray([0.35, 0.35, 0.35], np.float32)

_atlas = None


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    rotation: tuple  # body-to-world, row by row


def atlas():
    """Every texture as mip levels: level -> array (materials, size, size, 3) in BGR float."""
    global _atlas
    if _atlas is None:
        built = textures.build()
        names = list(built)
        levels = []
        images = [img[..., ::-1].astype(np.float32) / 255.0 for img, _ in built.values()]
        while True:
            levels.append(np.stack(images))
            size = images[0].shape[0]
            if size == 1:
                break
            images = [cv2.resize(img, (size // 2, size // 2), interpolation=cv2.INTER_AREA).reshape(size // 2, size // 2, 3)
                      for img in images]
        _atlas = (names, [tile for _, tile in built.values()], levels)
    return _atlas


class Camera:
    SKY, FLOOR, WALL, CEILING, FIRST_BOX = 0, 1, 2, 3, 4

    def __init__(self, scene, render_scale: float = 0.36, fov: float = FOV, seed: int = 0):
        self.scene = scene
        self.width = w = max(32, round(WIDTH * render_scale))
        self.height = h = max(18, round(HEIGHT * render_scale))
        self.half = math.tan(fov / 2)
        self.vhalf = self.half * h / w
        u = ((np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1) * self.half
        v = (1 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2) * self.vhalf
        uu, vv = np.meshgrid(u, v)
        norm = 1 / np.sqrt(1 + uu * uu + vv * vv)
        self.rays = (norm, -uu * norm, vv * norm)
        self.pixel = 2 * self.half / w
        radius = np.sqrt(((uu / self.half) ** 2 + (vv / self.half) ** 2))
        self.vignette = (1 - 0.35 * radius ** 2.2).astype(np.float32)[..., None]
        self.rng = np.random.default_rng(seed)
        self.grain = self.rng.normal(0, 0.018, (h + 64, w + 64, 3)).astype(np.float32)
        self.outdoor = scene.bounds is None

        names, tiles, self.levels = atlas()
        index = {name: i for i, name in enumerate(names)}
        fallback = index["concrete"]
        boxes = scene.boxes
        count = self.FIRST_BOX + len(boxes)
        self.material = np.zeros(count, np.int32)
        self.fit = np.zeros(count, bool)
        self.origin = np.zeros((count, 3), np.float32)
        self.extent = np.ones((count, 3), np.float32)
        self.tile = np.ones(count, np.float32)
        for k, name in ((self.FLOOR, scene.floor_material), (self.WALL, scene.wall_material),
                        (self.CEILING, scene.ceiling_material)):
            self.material[k] = index.get(name, fallback)
        for k, box in enumerate(boxes, self.FIRST_BOX):
            self.material[k] = index.get(box.material, fallback)
            if box.fit:
                self.fit[k] = True
                self.origin[k] = box.lo
                self.extent[k] = np.maximum(np.subtract(box.hi, box.lo), 1e-3)
                self.tile[k] = max(self.extent[k])
        for k in range(count):
            if not self.fit[k]:
                self.tile[k] = tiles[self.material[k]]
                self.extent[k] = self.tile[k]
        self.clouds = index["clouds"], tiles[index["clouds"]]

    # ------------------------------------------------------------------ geometry
    def cast(self, pose: Pose):
        x, y, z = pose.x, pose.y, pose.z
        r = pose.rotation
        bx, by, bz = self.rays
        dx = r[0] * bx + r[1] * by + r[2] * bz
        dy = r[3] * bx + r[4] * by + r[5] * bz
        dz = r[6] * bx + r[7] * by + r[8] * bz
        shape = dx.shape
        dist = np.full(shape, np.inf, np.float32)
        obj = np.zeros(shape, np.int16)
        axis = np.zeros(shape, np.int8)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ix, iy, iz = 1 / dx, 1 / dy, 1 / dz
            t = -z * iz
            np.copyto(dist, t, where=t > 0)
            np.copyto(obj, self.FLOOR, where=t > 0)
            np.copyto(axis, 2, where=t > 0)
            bounds = self.scene.bounds
            if bounds is not None:
                lo, hi = bounds.lo, bounds.hi
                for t, kind, ax in (((np.where(ix > 0, hi[0], lo[0]) - x) * ix, self.WALL, 0),
                                    ((np.where(iy > 0, hi[1], lo[1]) - y) * iy, self.WALL, 1),
                                    (np.where(iz > 0, (hi[2] - z) * iz, np.inf), self.CEILING, 2)):
                    hit = (t > 0) & (t < dist)
                    np.copyto(dist, t, where=hit)
                    np.copyto(obj, kind, where=hit)
                    np.copyto(axis, ax, where=hit)
            for k, box in enumerate(self.scene.boxes, self.FIRST_BOX):
                rect = self.footprint(box, pose)
                if rect is None:
                    continue
                part = (slice(rect[0], rect[1]), slice(rect[2], rect[3]))
                a, b = (box.lo[0] - x) * ix[part], (box.hi[0] - x) * ix[part]
                nx, fx = np.minimum(a, b), np.maximum(a, b)
                a, b = (box.lo[1] - y) * iy[part], (box.hi[1] - y) * iy[part]
                ny, fy = np.minimum(a, b), np.maximum(a, b)
                a, b = (box.lo[2] - z) * iz[part], (box.hi[2] - z) * iz[part]
                nz, fz = np.minimum(a, b), np.maximum(a, b)
                near = np.maximum(np.maximum(nx, ny), nz)
                far = np.minimum(np.minimum(fx, fy), fz)
                hit = (near <= far) & (near > 0) & (near < dist[part])
                if not hit.any():
                    continue
                which = np.where((nx >= ny) & (nx >= nz), 0, np.where(ny >= nz, 1, 2)).astype(np.int8)
                np.copyto(dist[part], near, where=hit)
                np.copyto(obj[part], k, where=hit)
                np.copyto(axis[part], which, where=hit)
        return dist, obj, axis, (dx, dy, dz)

    def trace(self, pose: Pose):
        """Distance to the first surface, what it is, and its axis, for each rendered pixel."""
        dist, obj, axis, _ = self.cast(pose)
        return dist, obj, axis

    def footprint(self, box, pose: Pose):
        """The rows and columns a box can cover, or None when it is out of view."""
        r = pose.rotation
        xs = np.asarray([box.lo[0], box.hi[0]]) - pose.x
        ys = np.asarray([box.lo[1], box.hi[1]]) - pose.y
        zs = np.asarray([box.lo[2], box.hi[2]]) - pose.z
        px, py, pz = (a.ravel() for a in np.meshgrid(xs, ys, zs))
        fwd = r[0] * px + r[3] * py + r[6] * pz
        if fwd.max() <= 0:
            return None
        if fwd.min() < 0.05:
            return 0, self.height, 0, self.width
        left = r[1] * px + r[4] * py + r[7] * pz
        up = r[2] * px + r[5] * py + r[8] * pz
        cols = (-left / fwd / self.half + 1) * self.width / 2
        rows = (1 - up / fwd / self.vhalf) * self.height / 2
        c0, c1 = max(0, int(cols.min()) - 1), min(self.width, int(math.ceil(cols.max())) + 1)
        r0, r1 = max(0, int(rows.min()) - 1), min(self.height, int(math.ceil(rows.max())) + 1)
        if c0 >= c1 or r0 >= r1:
            return None
        return r0, r1, c0, c1

    # ----------------------------------------------------------------- shading
    def shade(self, pose: Pose) -> np.ndarray:
        dist, obj, axis, (dx, dy, dz) = self.cast(pose)
        h, w = dist.shape
        color = np.empty((h, w, 3), np.float32)
        sky = obj == self.SKY
        solid = ~sky
        if sky.any():
            color[sky] = self.sky(pose, dx[sky], dy[sky], dz[sky])
        if solid.any():
            t = dist[solid]
            o = obj[solid].astype(np.int32)
            a = axis[solid]
            ddx, ddy, ddz = dx[solid], dy[solid], dz[solid]
            px, py, pz = pose.x + t * ddx, pose.y + t * ddy, pose.z + t * ddz
            along = np.choose(a, (ddx, ddy, ddz))
            face = a * 2 + (along < 0)
            uax = np.where(a == 0, 1, 0)
            vax = np.where(a == 2, 1, 2)
            u = (np.where(a == 0, py, px) - self.origin[o, uax]) / self.extent[o, uax]
            v = (np.where(a == 2, py, pz) - self.origin[o, vax]) / self.extent[o, vax]
            fit = self.fit[o]
            if fit.any():
                flip = fit & ((face == 0) | (face == 3))
                u = np.where(flip, 1 - u, u)
                v = np.where(fit & (a != 2), 1 - v, v)
            texels = t * self.pixel * self.levels[0].shape[1] / (self.tile[o] * np.maximum(np.abs(along), 0.2))
            level = np.clip(np.log2(np.maximum(texels, 1.0)), 0, len(self.levels) - 1).astype(np.int32)
            mat = self.material[o]
            surface = np.empty((t.size, 3), np.float32)
            for lv in np.unique(level):
                m = level == lv
                tex = self.levels[lv]
                size = tex.shape[1]
                iu = np.floor(u[m] * size).astype(np.int32)
                iv = np.floor(v[m] * size).astype(np.int32)
                f = fit[m]
                iu = np.where(f, np.clip(iu, 0, size - 1), iu & (size - 1))
                iv = np.where(f, np.clip(iv, 0, size - 1), iv & (size - 1))
                surface[m] = tex[mat[m], iv, iu]
            light = LIGHT[face]
            if self.outdoor:
                light = light * 1.05
            else:
                light = light * np.clip(1.12 - 0.035 * t, 0.62, 1.1)
            surface *= light[:, None]
            fog = self.scene.fog_distance_m
            if fog > 0:
                mix = (1 - np.exp(-t / fog))[:, None]
                surface = surface * (1 - mix) + SKY_LOW * mix
            color[solid] = surface
        return color

    def sky(self, pose: Pose, dx, dy, dz) -> np.ndarray:
        g = np.clip(dz, 0, 1)[:, None] ** 0.6
        out = SKY_LOW * (1 - g) + SKY_HIGH * g
        index, tile = self.clouds
        up = dz > 0.02
        if up.any():
            t = (600.0 - pose.z) / dz[up]
            tex = self.levels[2]
            size = tex.shape[1]
            iu = np.floor((pose.x + t * dx[up]) / tile * size).astype(np.int64) & (size - 1)
            iv = np.floor((pose.y + t * dy[up]) / tile * size).astype(np.int64) & (size - 1)
            cover = tex[index, iv, iu, :1] * np.clip(dz[up] * 6, 0, 1)[:, None]
            out[up] = out[up] * (1 - cover) + 0.97 * cover
        return out

    def render(self, pose: Pose) -> np.ndarray:
        """A full-size BGR frame, as the camera's JPEG would decode."""
        color = self.shade(pose)
        color *= self.vignette
        dy, dx = self.rng.integers(0, 64, 2)
        color += self.grain[dy:dy + color.shape[0], dx:dx + color.shape[1]]
        small = np.clip(color * 255, 0, 255).astype(np.uint8)
        image = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        mark(image)
        return image


def mark(image: np.ndarray) -> None:
    """Label every frame so a simulated feed can't be mistaken for the drone's."""
    origin = (image.shape[1] - 196, image.shape[0] - 14)
    cv2.putText(image, "SIMULATED", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, "SIMULATED", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


class VideoEncoder:
    """JPEG at the size the S20 sends, wrapped in its 40-byte frame header."""

    def __init__(self, seed: int = 0, sizes: tuple[int, int, int] = SIZES):
        self.rng = random.Random(seed)
        self.sizes = sizes
        self.quality = 70
        self.sequence = self.rng.randrange(256)
        self.counter = self.rng.randrange(256)
        self.gap = 1 / RATE

    def encode(self, image: np.ndarray, time_s: float) -> tuple[bytes, bytes]:
        low, median, high = self.sizes
        target = min(high, max(low + 4, round(self.rng.gauss(median, (high - low) / 6))))
        jpeg = self.compress(image, target)
        envelope = self.envelope(jpeg, self.sequence, self.counter, time_s)
        skipped = 1 if self.rng.random() < SKIP else 0
        self.sequence = (self.sequence + 1) & 0xFF
        self.counter = (self.counter + 1 + skipped) & 0xFF
        self.gap = (1 + skipped) / RATE
        return envelope, jpeg

    def compress(self, image: np.ndarray, target: int) -> bytes:
        quality = self.quality
        for _ in range(10):
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            size = len(buf)
            if size <= target or quality <= 5:
                break
            quality = max(5, quality - max(1, round((size / target - 1) * 30)))
        if size < target * 0.93 and quality < 95:
            self.quality = quality + 1
        else:
            self.quality = quality
        return pad(buf.tobytes(), target)

    def envelope(self, jpeg: bytes, sequence: int, camera_counter: int, time_s: float) -> bytes:
        head = bytearray(HEADER)
        head[0:4] = b"\x00\x00\x01\xa1"
        head[6] = camera_counter & 0xFF
        head[10] = sequence & 0xFF
        head[11:14] = b"\x10\x00\x04"
        head[16:20] = len(jpeg).to_bytes(4, "little")
        head[24:26] = ADVERTISED[1].to_bytes(2, "little")
        head[26:28] = ADVERTISED[0].to_bytes(2, "little")
        head[32:40] = (CLOCK + round(time_s * 1e6)).to_bytes(8, "little")
        return bytes(head) + jpeg


def pad(jpeg: bytes, size: int) -> bytes:
    """Grow a JPEG to exactly `size` bytes with comment segments after its JFIF header."""
    gap = size - len(jpeg)
    if gap < 4:
        return jpeg
    at = 2
    if jpeg[2:4] == b"\xff\xe0":
        at = 4 + int.from_bytes(jpeg[4:6], "big")
    segments = []
    while gap >= 4:
        n = min(gap, 65537)
        if 0 < gap - n < 4:
            n -= 4
        segments.append(b"\xff\xfe" + (n - 2).to_bytes(2, "big") + bytes(n - 4))
        gap -= n
    return jpeg[:at] + b"".join(segments) + jpeg[at:]


def packetize(frame_id: int, envelope: bytes) -> list[bytes]:
    """Split a frame into the S20's UDP datagrams: 40 40 <id> <count> <index> <length> ... 23 23."""
    total = max(1, math.ceil(len(envelope) / BODY))
    head = b"\x40\x40" + (frame_id & 0xFFFF).to_bytes(2, "little") + bytes([total])
    chunks = []
    for index in range(total):
        body = envelope[index * BODY:(index + 1) * BODY]
        chunks.append(head + bytes([index]) + (len(body) + 10).to_bytes(2, "little") + body + b"##")
    return chunks
