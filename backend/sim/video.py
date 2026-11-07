"""Measure rotation from the drone's own video, which is the only thing the S20 sends back.

A turn slides the picture sideways. Adding up the slide between frames gives the heading in
pixels; finding the frame where the view lines up with the start again marks exactly 360
degrees, which turns pixels into degrees and gives the camera's field of view.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

WIDTH = 320
FOV_RANGE = (140.0, 30.0)  # degrees: any lens a toy drone could have


def prepare(image: np.ndarray) -> np.ndarray:
    """A small grey strip from the middle of the frame, where a turn moves every pixel alike."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    small = cv2.resize(image, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA).astype(np.float32)
    rows, cols = small.shape
    strip = small[rows // 6: rows - rows // 6, cols // 4: cols - cols // 4]
    # Drop what stays put in the frame (vignetting, the JPEG block grid) so only the scene can match.
    detail = strip - cv2.GaussianBlur(strip, (0, 0), 6)
    return np.ascontiguousarray(cv2.GaussianBlur(detail, (0, 0), 1.2))


def slide(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far b's picture sits to the right of a's, and how sure that is (0 to 1)."""
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, _), response = cv2.phaseCorrelate(a, b, window)
    return dx, response


def spin(images: list[np.ndarray], times: list[float]) -> dict | None:
    """Field of view and heading over time from frames of the drone turning in place. Left is positive."""
    strips = [prepare(img) for img in images]
    pairs = [slide(strips[i - 1], strips[i]) for i in range(1, len(strips))]
    good = [i for i, (_, response) in enumerate(pairs) if response > 0.35]
    if len(good) < 10:
        return None
    # Plain walls give nothing to lock on to; a spin turns smoothly, so bridge them from either side.
    steps = np.interp(np.arange(len(pairs)), good, [pairs[i][0] for i in good])
    total = np.concatenate([[0.0], np.cumsum(steps)])
    moving = [i for i, s in enumerate(steps) if abs(s) > 1.5]
    if len(moving) < 10:
        return None
    ref = moving[0]
    half = WIDTH / 2
    lo, hi = (2 * math.pi * half / math.tan(math.radians(fov) / 2) for fov in FOV_RANGE)
    turns = []
    for j in moving[: len(moving) // 2]:
        best = None
        for k in range(j + 1, len(strips)):
            turned = abs(total[k] - total[j])
            if lo * 0.95 <= turned <= hi * 1.05:
                dx, response = slide(strips[j], strips[k])
                if response > 0.12 and (best is None or response > best[1]):
                    best = (abs(total[k] - total[j] - dx), response)
        if best is not None:
            turns.append(best)
    # A room that looks alike from both ends agrees on a false half turn, but only a real
    # full turn shows the same picture again, so it matches far more strongly.
    groups = [[t for t in turns if abs(t[0] - centre) < 0.03 * centre] for centre, _ in turns]
    groups = [g for g in groups if len(g) >= 3]
    if not groups:
        return None
    agreed = max(groups, key=lambda g: (np.median([r for _, r in g]), len(g)))
    if len(agreed) < 0.25 * len(turns):
        return None  # matches disagree: too close to a wall, or too little in view
    focal = float(np.median([t for t, _ in agreed])) / (2 * math.pi)
    heading = [math.degrees((t - total[ref]) / focal) for t in total]
    return {
        "fov_deg": math.degrees(2 * math.atan(half / focal)),
        "agreement": f"{len(agreed)} of {len(turns)}",
        "start": times[ref],
        "times": list(times),
        "heading": heading,
        "yaw_deg": heading[-1] - heading[ref],
    }
