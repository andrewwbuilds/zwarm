"""Picking photos for the reconstruction from every frame the camera sent.

Capture keeps everything. Choosing is done afterwards, with the whole flight in view, so a frame is judged
against its neighbours rather than against a guess made on the spot. Two frames that look alike are not
duplicates: a slow flight is supposed to produce neighbours that overlap, because that overlap is how the
photos get linked into one model. Only the sharpest frame of each short window is kept, so a blurry moment
costs one photo, not a gap.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SHARP_WIDTH = 592  # frames are scored at this width, plenty for blur
THUMB = (296, 164)  # and compared at this size for "identical"
NEIGHBOURHOOD = 15  # frames either side that set the standard for "sharp"
BLANK_FLOOR = 1.0  # sharpness below this means nothing to see (a blank wall or a covered lens)


@dataclass
class Report:
    total: int
    chosen: list  # original frame numbers, in flight order
    rejected: Counter = field(default_factory=Counter)

    @property
    def kept(self) -> int:
        return len(self.chosen)

    def summary(self) -> str:
        parts = [f"{n} {why}" for why, n in self.rejected.most_common() if n]
        return f"kept {self.kept} of {self.total} frames" + (f" ({', '.join(parts)})" if parts else "")


@contextlib.contextmanager
def quiet_jpeg_warnings():
    """The S20's JPEGs end with a byte or two of padding that libjpeg complains about for every frame."""
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def frame_files(raw: Path) -> list[Path]:
    return sorted(p for p in raw.glob("*.jpg"))


def _gray(image: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    return cv2.resize(gray, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp detail, near zero for blur or a blank wall."""
    return float(cv2.Laplacian(_gray(image, SHARP_WIDTH), cv2.CV_64F).var())


def score_frames(paths: list[Path], cache: Path | None = None) -> list[tuple[float, str]]:
    """(sharpness, file name) for each frame. Scores are cached next to the frames."""
    known = json.loads(cache.read_text()) if cache and cache.exists() else {}
    scores, changed = [], False
    for path in paths:
        size = path.stat().st_size
        entry = known.get(path.name)
        if entry and entry[0] == size:
            value = entry[1]
        else:
            image = cv2.imread(str(path))
            value = sharpness(image) if image is not None else 0.0
            known[path.name] = [size, value]
            changed = True
        scores.append((value, path.name))
    if cache and changed:
        cache.write_text(json.dumps(known))
    return scores


def _thumb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    return _gray(image, THUMB[0]).astype(np.int16)


def curate(root: Path, window: int = 3, blur_ratio: float = 0.6, identical_diff: float = 0.8) -> Report:
    """Choose photos from root/raw and copy them to root/images, numbered in flight order.

    window: frames per group, of which the sharpest is kept (3 gives about 6 photos a second from the S20).
    blur_ratio: a frame is blurry when it is under this share of its neighbourhood's median sharpness.
    identical_diff: mean pixel difference under which two frames count as the same picture.
    """
    raw = root / "raw"
    paths = frame_files(raw)
    if not paths:
        raise FileNotFoundError(f"No frames in {raw}. Record some first: uv run python -m mapping capture NAME")
    with quiet_jpeg_warnings():
        scores = score_frames(paths, raw / "scores.json")
    sharp = np.array([s for s, _ in scores])
    rejected: Counter = Counter()
    blurry = np.zeros(len(paths), bool)
    for i in range(len(paths)):
        around = sharp[max(0, i - NEIGHBOURHOOD): i + NEIGHBOURHOOD + 1]
        blurry[i] = sharp[i] < BLANK_FLOOR or sharp[i] < blur_ratio * float(np.median(around))
    rejected["blurry"] = int(blurry.sum())

    chosen, last = [], None
    for start in range(0, len(paths), window):
        group = [i for i in range(start, min(start + window, len(paths))) if not blurry[i]]
        if not group:
            continue
        best = max(group, key=lambda i: (sharp[i], -i))
        with quiet_jpeg_warnings():
            thumb = _thumb(paths[best])
        if last is not None and float(np.abs(thumb - last).mean()) < identical_diff:
            rejected["identical"] += 1
            continue
        chosen.append(best)
        last = thumb
    rejected["passed over for a sharper neighbour"] = len(paths) - int(blurry.sum()) - len(chosen) - rejected["identical"]

    images = root / "images"
    shutil.rmtree(images, ignore_errors=True)
    images.mkdir(parents=True)
    for n, i in enumerate(chosen, 1):
        shutil.copyfile(paths[i], images / f"{n:05d}.jpg")
    return Report(len(paths), chosen, rejected)


def raw_name(index: int, t_ms: int) -> str:
    return f"{index:06d}_{t_ms:08d}.jpg"


def frame_time_ms(path: Path) -> int | None:
    match = re.match(r"\d+_(\d+)\.jpg$", path.name)
    return int(match.group(1)) if match else None
