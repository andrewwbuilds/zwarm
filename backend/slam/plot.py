"""Top-down plots of a trajectory against the truth."""

from __future__ import annotations

import cv2
import numpy as np

from slam.geometry import apply_sim3, umeyama


def topdown(estimated: np.ndarray, truth: np.ndarray | None = None, size: int = 640, loops: list | None = None,
            points: np.ndarray | None = None) -> np.ndarray:
    """A BGR image of the path seen from above (x right, y up). The estimate is aligned to the truth if given."""
    estimated = np.asarray(estimated, float)
    if truth is not None:
        truth = np.asarray(truth, float)
        estimated = apply_sim3(umeyama(estimated, truth), estimated)
    everything = [estimated] + ([truth] if truth is not None else [])
    stack = np.concatenate([e[:, :2] for e in everything])
    lo, hi = stack.min(0), stack.max(0)
    span = max(float((hi - lo).max()), 1e-6)
    margin = 40
    scale = (size - 2 * margin) / span

    def px(p):
        return (int(margin + (p[0] - lo[0]) * scale), int(size - margin - (p[1] - lo[1]) * scale))

    image = np.full((size, size, 3), 255, np.uint8)
    if points is not None and len(points):
        for p in points[::3]:
            x, y = px(p)
            if 0 <= x < size and 0 <= y < size:
                image[y, x] = (200, 200, 200)
    for path, color in ((truth, (0, 160, 0)), (estimated, (0, 0, 220))):
        if path is None:
            continue
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(image, px(a), px(b), color, 2, cv2.LINE_AA)
        cv2.circle(image, px(path[0]), 6, color, -1)
    cv2.putText(image, "green: truth   red: estimate", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(image, f"{span:.1f} m across", (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return image
