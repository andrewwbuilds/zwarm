"""Measure how the drone is moving from its front camera alone.

The S20 tells the host nothing about its flight, so the video is the only feedback. Corners are
tracked between frames and fitted to the optical flow of a camera moving past a flat scene:

    u = x y wx - (1 + x^2) wy + y wz + (x c - a)
    v = (1 + y^2) wx - x y wy - x wz + (y c - b)

x and y are normalised image coordinates. wx, wy and wz are the camera's turn rates. a, b and c are
its velocity right, down and forward, each divided by the distance to the scene, so they come out in
1/s rather than m/s: without a depth sensor the true speed cannot be known, only its ratio to depth.

A turn and a sideways slide both move the image sideways. They differ in the x^2 term: a turn drags
the edges of the view faster than its centre, and a slide past a flat scene moves them equally. This
is what lets a slide be seen during a turn, and it is why the fit is more than an image shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

MAX_CORNERS = 160
MIN_CORNERS = 90  # find more corners once fewer than this survive
MIN_INLIERS = 24
MIN_SPREAD = 0.25  # of normalised x: inliers bunched in one spot say little about a turn
INLIER_PIXELS = 1.2
LK_WINDOW = (21, 21)
LK_LEVELS = 3
MIN_DT, MAX_DT = 1 / 60, 0.3
MAX_SPEED = 4.0  # 1/s: crossing the distance to the scene in a quarter second is a bad fit, not a flight
HOLD_PRIOR = 0.3  # how hard the turn rate is held to the sticks' request when the yaw stick is centred
TURN_PRIOR = 0.03  # and while turning: only the x^2 term should decide, so a slide can be seen apart from it


@dataclass(frozen=True)
class Motion:
    t: float
    dt: float
    valid: bool
    inliers: int = 0
    tracked: int = 0
    lateral: float = 0.0  # moving right, 1/s
    vertical: float = 0.0  # moving down, 1/s
    forward: float = 0.0  # moving forward, 1/s
    yaw_rate: float = 0.0  # turning right, rad/s
    pitch_rate: float = 0.0  # camera tilting down, rad/s
    roll_rate: float = 0.0  # camera rolling clockwise, rad/s
    residual: float = 0.0  # pixels


def unturn(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """The displacements left over once a turn to the right by this angle is taken out of the second frame."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    depth = cos_a - x1 * sin_a
    return (x1 * cos_a + sin_a) / depth - x0, y1 / depth - y0


def flow_basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rows of the flow model for each point, u rows then v rows. Columns: wx wy wz a b c."""
    n = len(x)
    one, zero = np.ones(n), np.zeros(n)
    u = np.stack([x * y, -(1 + x * x), y, -one, zero, x], axis=1)
    v = np.stack([1 + y * y, -x * y, -x, zero, -one, y], axis=1)
    return np.vstack([u, v])


def solve_flow(x, y, du, dv, turn_prior: float = 0.0, iterations: int = 4,
               inlier: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Robustly fit the flow model. du and dv are normalised displacements. Returns (parameters, inlier mask)."""
    n = len(x)
    basis = flow_basis(x, y)
    target = np.concatenate([du, dv])
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    if n >= 8:
        # Moving things in view agree with each other, not with the room: vote before fitting.
        for _ in range(24):
            pick = rng.choice(n, 4, replace=False)
            rows = np.concatenate([pick, pick + n])
            params = np.linalg.lstsq(basis[rows], target[rows], rcond=None)[0]
            error = (basis @ params - target).reshape(2, -1)
            mask = np.hypot(error[0], error[1]) < inlier * 2
            if mask.sum() > best.sum():
                best = mask
    if best.sum() < 6:
        best = np.ones(n, bool)
    params = np.zeros(6)
    for _ in range(iterations):
        rows = np.concatenate([best, best])
        a, b = basis[rows], target[rows]
        if turn_prior:
            # A pull on the turn rate, because a narrow or flat view can't tell it from a slide.
            a = np.vstack([a, turn_prior * max(int(best.sum()), 1) ** 0.5 * np.eye(6)[1]])
            b = np.append(b, 0.0)
        params = np.linalg.lstsq(a, b, rcond=None)[0]
        error = (basis @ params - target).reshape(2, -1)
        best = np.hypot(error[0], error[1]) < inlier
        if best.sum() < 6:
            break
    return params, best


class MotionTracker:
    """Feed it grayscale frames; it says how the camera moved since the last one."""

    def __init__(self, hfov_deg: float = 90.0):
        self.hfov = math.radians(hfov_deg)
        self.previous: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.last_t: float | None = None

    def reset(self) -> None:
        self.previous = self.points = self.last_t = None

    def update(self, gray: np.ndarray, t: float, expected_yaw_rate: float = 0.0, turning: bool = False) -> Motion:
        previous, points, last_t = self.previous, self.points, self.last_t
        self.previous, self.last_t = gray, t
        if previous is None or last_t is None or previous.shape != gray.shape:
            self.points = self._corners(gray, None)
            return Motion(t, 0.0, False)
        dt = t - last_t
        if dt < MIN_DT:  # a repeated frame says nothing
            self.previous, self.last_t = previous, last_t
            return Motion(t, dt, False)
        if dt > MAX_DT or points is None or len(points) < 8:
            self.points = self._corners(gray, None)
            return Motion(t, dt, False)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, winSize=LK_WINDOW,
                                                    maxLevel=LK_LEVELS)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, winSize=LK_WINDOW,
                                                        maxLevel=LK_LEVELS)
        good = (status.ravel() == 1) & (back_status.ravel() == 1)
        good &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 0.7
        a, b = points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
        tracked = len(a)
        if tracked < MIN_INLIERS:
            self.points = self._corners(gray, None)
            return Motion(t, dt, False, tracked=tracked)

        h, w = gray.shape
        focal = (w / 2) / math.tan(self.hfov / 2)
        x0, y0 = (a[:, 0] - w / 2) / focal, (a[:, 1] - h / 2) / focal
        du, dv = (b[:, 0] - a[:, 0]) / focal, (b[:, 1] - a[:, 1]) / focal
        # Fit around the rate the sticks asked for, so the prior can pull toward it rather than toward zero.
        expected = expected_yaw_rate * dt
        x1, y1 = x0 + du, y0 + dv
        params, mask = solve_flow(x0, y0, *unturn(x0, y0, x1, y1, expected), TURN_PRIOR if turning else HOLD_PRIOR,
                                  inlier=INLIER_PIXELS / focal, seed=int(t * 1000) & 0xFFFF)
        turned = expected + float(params[1])
        if turning:
            # The flow model is only first order, and a turn's next term looks exactly like flying
            # forward. Undo the measured turn exactly, then fit what's left, which is the real motion.
            params, mask = solve_flow(x0, y0, *unturn(x0, y0, x1, y1, turned), HOLD_PRIOR,
                                      inlier=INLIER_PIXELS / focal, seed=int(t * 1000) & 0xFFFF)
            turned += float(params[1])
        params = params.copy()
        params[1] = turned
        kept = b[mask]
        self.points = self._corners(gray, kept)
        inliers = int(mask.sum())
        spread = float(np.std(x0[mask])) if inliers else 0.0
        if inliers < MIN_INLIERS or spread < MIN_SPREAD:
            return Motion(t, dt, False, inliers=inliers, tracked=tracked)
        left_u, left_v = unturn(x0[mask], y0[mask], x1[mask], y1[mask], turned)
        fitted = params.copy()
        fitted[1] = 0.0
        error = flow_basis(x0[mask], y0[mask]) @ fitted - np.concatenate([left_u, left_v])
        rates = params / dt
        if max(abs(rates[3]), abs(rates[5])) > MAX_SPEED:
            return Motion(t, dt, False, inliers=inliers, tracked=tracked)
        return Motion(
            t, dt, True, inliers, tracked,
            lateral=float(rates[3]), vertical=float(rates[4]), forward=float(rates[5]),
            yaw_rate=float(rates[1]), pitch_rate=float(rates[0]), roll_rate=float(rates[2]),
            residual=float(np.sqrt(np.mean(error * error)) * focal),
        )

    def _corners(self, gray: np.ndarray, kept: np.ndarray | None) -> np.ndarray | None:
        if kept is not None and len(kept) >= MIN_CORNERS:
            return kept.reshape(-1, 1, 2).astype(np.float32)
        mask = None
        if kept is not None and len(kept):
            mask = np.full(gray.shape, 255, np.uint8)
            for x, y in kept.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 7, 0, -1)
        found = cv2.goodFeaturesToTrack(gray, MAX_CORNERS - (0 if kept is None else len(kept)), 0.01, 7, mask=mask)
        if found is None:
            return None if kept is None or not len(kept) else kept.reshape(-1, 1, 2).astype(np.float32)
        if kept is None or not len(kept):
            return found.astype(np.float32)
        return np.vstack([kept.reshape(-1, 1, 2).astype(np.float32), found.astype(np.float32)])
