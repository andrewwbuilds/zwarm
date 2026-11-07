"""Live localization: read the backend's video stream, track it, and show the video beside a top-down map."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from slam.camera import Intrinsics
from slam.system import OK, Config, Slam

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def jpeg_frames(stream, chunk: int = 65536) -> Iterator[bytes]:
    """JPEG images from a multipart MJPEG byte stream (anything with .read(n))."""
    buffer = b""
    while True:
        data = stream.read(chunk)
        if not data:
            return
        buffer += data
        while True:
            start = buffer.find(SOI)
            if start < 0:
                buffer = buffer[-1:]
                break
            end = buffer.find(EOI, start + 2)
            if end < 0:
                buffer = buffer[start:]
                break
            yield buffer[start:end + 2]
            buffer = buffer[end + 2:]


def draw_map(positions: np.ndarray, heading: float | None, loops: list, size: int = 480, unit: str = "units",
             scale: float = 1.0) -> np.ndarray:
    """Top-down path of the drone: x right, y up, the current position marked with an arrow."""
    image = np.full((size, size, 3), 250, np.uint8)
    if len(positions) < 2:
        cv2.putText(image, "waiting for tracking...", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
        return image
    xy = positions[:, [0, 2]] * scale  # map x right, map z forward: the floor plane for a level camera
    lo, hi = xy.min(0), xy.max(0)
    span = max(float((hi - lo).max()), 0.5)
    centre = (lo + hi) / 2
    pixels_per = (size - 80) / span
    px = lambda p: (int(size / 2 + (p[0] - centre[0]) * pixels_per), int(size / 2 - (p[1] - centre[1]) * pixels_per))
    for a, b in zip(xy[:-1], xy[1:]):
        cv2.line(image, px(a), px(b), (200, 80, 0), 2, cv2.LINE_AA)
    cv2.circle(image, px(xy[0]), 5, (0, 160, 0), -1)
    here = px(xy[-1])
    cv2.circle(image, here, 6, (0, 0, 220), -1)
    if heading is not None:
        tip = (int(here[0] + 22 * np.sin(heading)), int(here[1] - 22 * np.cos(heading)))
        cv2.arrowedLine(image, here, tip, (0, 0, 220), 2, tipLength=0.4)
    bar = 1.0 if span > 4 else 0.5
    cv2.line(image, (20, size - 20), (20 + int(bar * pixels_per), size - 20), (60, 60, 60), 2)
    cv2.putText(image, f"{bar:g} {unit}", (20, size - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return image


def run_live(url: str, scale: float | None, hfov: float, window: bool, save_dir: Path | None, seconds: float | None) -> None:
    width, height = 1184, 656
    slam = Slam(Intrinsics.from_fov(width, height, hfov), Config())
    unit = "m" if scale else "map units"
    started = time.time()
    count = 0
    print(f"Reading {url}. Press q in the window to quit, r to start over.")
    with urllib.request.urlopen(url, timeout=10) as stream:
        for jpeg in jpeg_frames(stream):
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            estimate = slam.process(image, time.time() - started)
            count += 1
            if window or save_dir:
                positions = slam.positions(metric=False)
                heading = None
                if estimate.pose is not None:
                    forward = estimate.pose[:3, 2]
                    heading = float(np.arctan2(forward[0], forward[2]))
                panel = draw_map(positions, heading, slam.loops, unit=unit, scale=scale or 1.0)
                video = cv2.resize(image, (int(480 * width / height), 480))
                status = f"{estimate.state}  tracked {estimate.tracked}  keyframes {len(slam.keyframes)}  loops {len(slam.loops)}"
                cv2.putText(video, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                if estimate.pose is not None and scale:
                    x, _, z = estimate.pose[:3, 3] * scale
                    cv2.putText(video, f"x {x:+.2f}  z {z:+.2f}  y {-estimate.pose[1, 3] * scale:+.2f} m", (10, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                combined = np.hstack([video, panel])
                if window:
                    cv2.imshow("S20 localization", combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        slam = Slam(Intrinsics.from_fov(width, height, hfov), Config())
                if save_dir and count % 30 == 0:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / "latest.png"), combined)
            if seconds and time.time() - started > seconds:
                break
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        trajectory = slam.trajectory()
        with open(save_dir / "trajectory.csv", "w") as f:
            f.write("frame,time,x,y,z\n")
            for i, t, pose in trajectory:
                f.write(f"{i},{t:.3f},{pose[0, 3] * (scale or 1):.4f},{pose[1, 3] * (scale or 1):.4f},{pose[2, 3] * (scale or 1):.4f}\n")
        print(f"Saved {len(trajectory)} poses to {save_dir}")
    cv2.destroyAllWindows()
    print(f"{count} frames, {len(slam.keyframes)} keyframes, {len(slam.loops)} loops closed.")
