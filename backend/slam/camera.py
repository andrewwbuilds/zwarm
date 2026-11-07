"""The pinhole camera model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0  # radial distortion; the sim has none and the real lens is unmeasured

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, fx, fx, width / 2, height / 2)

    @property
    def k(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, factor: float) -> "Intrinsics":
        return Intrinsics(round(self.width * factor), round(self.height * factor), self.fx * factor,
                          self.fy * factor, self.cx * factor, self.cy * factor, self.k1)

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Pixels for points (rows) in the camera frame."""
        x = points_camera[:, 0] / points_camera[:, 2]
        y = points_camera[:, 1] / points_camera[:, 2]
        return np.stack([self.fx * x + self.cx, self.fy * y + self.cy], axis=1)

    def bearing(self, pixels: np.ndarray) -> np.ndarray:
        """Unit rays for pixels (rows)."""
        rays = np.stack([(pixels[:, 0] - self.cx) / self.fx, (pixels[:, 1] - self.cy) / self.fy,
                         np.ones(len(pixels))], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)
