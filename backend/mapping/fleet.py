"""One 3D map from every drone's camera.

Each drone's frames are saved as they arrive. A build chooses photos per drone, pools them, and lets
COLMAP place them all together. COLMAP works from the pictures alone, so no shared positions are
needed, but it can only join two drones if their photos overlap. That is why every pair of photos is
matched rather than only neighbours in time, which would never link one drone to another.

Nothing here touches the flight loop. Capture and build run on their own threads, the heavy steps
run in child processes at low priority, and start(), stop(), status(), build() and points() return
straight away.

    root/<name>/drones/<id>/raw/       every frame as it arrived
    root/<name>/drones/<id>/images/    the sharp frames curate() kept
    root/<name>/images/<id>/           the photos COLMAP sees, one folder and one camera per drone
    root/<name>/colmap/                COLMAP's database and models
    root/<name>/cloud.ply, cloud.html, points.npy, result.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from mapping.curate import curate, frame_files, frame_time_ms, quiet_jpeg_warnings, raw_name
from mapping.pointcloud import read_points, write_ply
from mapping.reconstruct import (BRUSH, Model, StepFailed, ToolMissing, analyze, best_model, coverage_warnings, run, tool,
                                 train)

logger = logging.getLogger(__name__)

BACKEND = Path(__file__).resolve().parents[1]
MAX_PHOTOS = 240  # keeps a build under about 5 minutes on an M5 Pro, see README
MIN_PHOTOS = 10
PHOTO_RATE = 6.0  # photos a second kept before thinning; the S20 at 17 fps gives a window of 3
POLL_S = 0.02
MAX_FRAMES_PER_DRONE = 25000  # about 25 minutes of video, 900 MB per drone
THUMB_WIDTH = 160
TILT_DEG = 45.0  # estimate: the S20's fixed camera tilt
HFOV_DEG = 90.0  # estimate: a starting point that COLMAP refines
FLIGHT_HEIGHT_M = 1.0  # sets the map's scale, which is therefore only rough
SPLAT_STEPS = 5000
MIN_LINK_MATCHES = 30  # COLMAP's default of 15 lets false matches join drones that share no view, see README
MIN_ROOM_AGREEMENT = 0.8  # the share of photos that must sit near their tracked position. A folded map can have half
CLOUD_REACH_M = 5.0  # the camera sees about 3.5 m ahead at 1 m up, so points further out than this are errors
MIN_FIT_PHOTOS = 12  # fewer tracked photos than this can't say much about the map's shape
MIN_JOIN_PHOTOS = 20  # COLMAP leaves small side models with junk lenses; they stay out of the served map
ROOM_FIT_LIMIT_M = 0.5  # tracked positions and the map disagree by more than this: the map is probably warped
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SSH_ROOT = "zwarm-maps"


class MapError(ValueError):
    """The message is a sentence an operator can read."""


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not SAFE_NAME.match(name):
        raise MapError(f"{name!r} can't be a map name. Use letters, digits, dots, dashes and underscores.")
    return name


def drone_folder(drone: str) -> str:
    """A drone id as a folder name, with anything awkward replaced."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(drone)) or "drone"


# ---------------------------------------------------------------- choosing photos
def window_for(raw: Path) -> int:
    """How many frames to group so curate() keeps about PHOTO_RATE photos a second, whatever the video's rate."""
    paths = frame_files(raw)
    times = [frame_time_ms(p) for p in (paths[0], paths[-1])] if paths else [None, None]
    if None in times or times[1] <= times[0]:
        return 3
    return max(1, round((len(paths) - 1) / ((times[1] - times[0]) / 1000) / PHOTO_RATE))


def motion_scores(paths: list[Path]) -> list[float]:
    """How much the picture changed since the previous photo. The first photo scores 0."""
    scores, last = [], None
    with quiet_jpeg_warnings():
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if image is None:
                scores.append(0.0)
                continue
            h, w = image.shape
            thumb = cv2.GaussianBlur(cv2.resize(image, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))),
                                                interpolation=cv2.INTER_AREA), (0, 0), 1.5).astype(np.int16)
            scores.append(0.0 if last is None or last.shape != thumb.shape else float(np.abs(thumb - last).mean()))
            last = thumb
    return scores


def prepare_drone(folder: Path) -> str:
    """Runs in a child process: curate one drone's frames and score how much each photo moved."""
    raw = frame_files(folder / "raw")
    report = curate(folder, window=window_for(folder / "raw"))
    paths = sorted((folder / "images").glob("*.jpg"))
    (folder / "motion.json").write_text(json.dumps({"motion": motion_scores(paths), "raw": [raw[i].name for i in report.chosen]}))
    return report.summary()


def pick_by_motion(motion: list[float], budget: int) -> list[int]:
    """Spend `budget` photos evenly along the distance the picture moved, not along the clock.

    A hover earns few photos and a turn earns many, which is where overlap is otherwise lost.
    """
    count = len(motion)
    if budget <= 0 or count == 0:
        return []
    if count <= budget:
        return list(range(count))
    values = np.asarray(motion, float)
    floor = float(np.percentile(values, 10))  # sensor noise makes even a still picture "move" a little
    typical = float(np.median(values))
    change = np.maximum(values - floor, 0) + 0.1 * max(typical - floor, 1e-3)
    distance = np.cumsum(change)
    levels = np.linspace(distance[0], distance[-1], budget)
    chosen = sorted(set(np.minimum(np.searchsorted(distance, levels), count - 1).tolist()))
    if len(chosen) < budget:  # bursts of motion can land several levels on one photo
        spare = [i for i in range(count) if i not in set(chosen)]
        extra = np.linspace(0, len(spare) - 1, budget - len(chosen)).round().astype(int)
        chosen = sorted(chosen + [spare[i] for i in extra])
    return chosen


def share_budget(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Split the photo cap fairly. A drone with few photos leaves its unused share to the others."""
    budgets, left, waiting = {}, cap, sorted(counts, key=lambda d: counts[d])
    while waiting:
        drone = waiting.pop(0)
        budgets[drone] = min(counts[drone], left // (len(waiting) + 1))
        left -= budgets[drone]
    return budgets


def pool_photos(folder: Path, cap: int) -> dict[str, int]:
    """Copy each drone's chosen photos to folder/images/<drone>/. Returns photos per drone.

    folder/photos.json records which saved frame each photo came from.
    """
    scored = {}
    for path in sorted((folder / "drones").iterdir()):
        if (path / "motion.json").exists():
            scored[path.name] = json.loads((path / "motion.json").read_text())
    budgets = share_budget({d: len(m["motion"]) for d, m in scored.items()}, cap)
    images = folder / "images"
    shutil.rmtree(images, ignore_errors=True)
    pooled, sources = {}, {}
    for drone, score in scored.items():
        paths = sorted((folder / "drones" / drone / "images").glob("*.jpg"))
        chosen = pick_by_motion(score["motion"], budgets[drone])
        if not chosen:
            continue
        (images / drone).mkdir(parents=True)
        for n, i in enumerate(chosen, 1):
            shutil.copyfile(paths[i], images / drone / f"{n:05d}.jpg")
            sources[f"{drone}/{n:05d}.jpg"] = score["raw"][i]
        pooled[drone] = len(chosen)
    (folder / "photos.json").write_text(json.dumps(sources))
    return pooled


def static_mask(folder: Path, sample: int = 24) -> Path | None:
    """Black out whatever stays fixed in the picture: the simulator's label, or a leg or prop guard in view.

    Those pixels look identical in every photo, so COLMAP would "match" photos that show nothing in
    common. A real feature moves. Only a crisp edge that never changes counts, so a blank wall doesn't.
    """
    masks, size = [], None
    with quiet_jpeg_warnings():
        for drone in sorted((folder / "images").iterdir()):
            paths = sorted(drone.glob("*.jpg"))
            if len(paths) < 8:
                continue
            picks = np.linspace(0, len(paths) - 1, min(sample, len(paths))).round().astype(int)
            grays = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for i in picks]
            grays = [g for g in grays if g is not None]
            if len(grays) < 8 or len({g.shape for g in grays}) != 1:
                continue
            stack = np.stack(grays).astype(np.float32)
            size = grays[0].shape
            edges = np.abs(cv2.Laplacian(stack.mean(axis=0), cv2.CV_32F, ksize=3))
            fixed = ((stack.std(axis=0) < 12.0) & (edges > 25.0)).astype(np.uint8)
            grown = cv2.dilate(fixed, np.ones((15, 15), np.uint8))
            count, blobs = cv2.connectedComponents(grown)
            weight = np.bincount(blobs[fixed > 0], minlength=count)  # a few pixels can stay still by chance; a label can't
            keep = np.flatnonzero(weight >= fixed.size * 0.0002)
            grown = np.isin(blobs, keep[keep > 0]).astype(np.uint8)
            grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            if grown.mean() < 0.25:  # more than this means the drone barely moved, not an overlay
                masks.append(grown)
    if not masks or size is None or not any(m.any() for m in masks):
        return None
    keep = np.where(np.any(np.stack(masks), axis=0), 0, 255).astype(np.uint8)
    path = folder / "mask.png"
    cv2.imwrite(str(path), keep)
    return path


# ---------------------------------------------------------------- COLMAP
def colmap_steps(root: str, width: int, height: int, threads: int, gpu: bool, masked: bool, shared_camera: bool = False,
                 fix_lens: bool = False) -> list:
    """(log name, COLMAP arguments) for each step. Paths start at `root`, so they run here or on another machine."""
    gpu_flag = "1" if gpu else "0"  # brew's COLMAP has no GPU support and needs this off
    focal = width / 2 / math.tan(math.radians(HFOV_DEG) / 2)
    reader = ["--ImageReader.camera_model", "SIMPLE_RADIAL",
              "--ImageReader.camera_params", f"{focal:.1f},{width / 2:.1f},{height / 2:.1f},0"]
    # Three cheap lenses differ, so by default each drone gets its own camera.
    reader += ["--ImageReader.single_camera", "1"] if shared_camera else ["--ImageReader.single_camera_per_folder", "1"]
    if masked:
        reader += ["--ImageReader.camera_mask_path", f"{root}/mask.png"]
    mapper = ["--Mapper.min_num_matches", os.environ.get("MAP_MIN_LINK") or str(MIN_LINK_MATCHES)]
    if fix_lens:
        mapper += ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0"]
    return [
        ("features", ["feature_extractor", "--database_path", f"{root}/colmap/database.db", "--image_path", f"{root}/images",
                      *reader, "--FeatureExtraction.use_gpu", gpu_flag, "--FeatureExtraction.num_threads", str(threads)]),
        ("matching", ["exhaustive_matcher", "--database_path", f"{root}/colmap/database.db",
                      "--FeatureMatching.use_gpu", gpu_flag, "--FeatureMatching.num_threads", str(threads)]),
        ("mapper", ["mapper", "--database_path", f"{root}/colmap/database.db", "--image_path", f"{root}/images",
                    "--output_path", f"{root}/colmap/sparse", "--Mapper.num_threads", str(threads), *mapper]),
    ]


def build_threads() -> int:
    """Leave cores free: the flight loop may share this machine."""
    return max(2, (os.cpu_count() or 4) - 4)


class LocalRunner:
    """Runs COLMAP on this Mac, at low priority so a flight loop on the same machine keeps its share."""

    name = "local"

    def run(self, folder: Path, plan: Callable, on_step: Callable[[str], None]) -> None:
        colmap = tool("colmap")
        for log, args in plan(str(folder), build_threads(), False):
            on_step(log)
            run(["nice", "-n", "10", colmap, *args], folder / "logs" / f"{log}.log")


class SshRunner:
    """The same COLMAP steps on another machine, such as a Brev GPU instance. Photos go there, models come back."""

    def __init__(self, host: str):
        if not re.fullmatch(r"(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9][A-Za-z0-9._:-]*", host or ""):
            raise StepFailed(f"MAP_BUILD_SSH={host!r} doesn't look like user@host.")
        self.host = host
        self.name = f"ssh {host}"
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
                    "-o", "ServerAliveCountMax=4"]

    def _remote(self, command: str, log: Path) -> None:
        run([*self.ssh, self.host, command], log)

    def _gpu(self) -> bool:
        probe = subprocess.run([*self.ssh, self.host, "colmap -h 2>&1 | head -3"], capture_output=True, text=True, timeout=60)
        return "with CUDA" in probe.stdout or "with GPU" in probe.stdout

    def run(self, folder: Path, plan: Callable, on_step: Callable[[str], None]) -> None:
        logs = folder / "logs"
        rsync = ["rsync", "-a", "--delete", "-e", " ".join(self.ssh)]
        remote = f"{SSH_ROOT}/{folder.name}"
        on_step("sending photos")
        self._remote(f"mkdir -p {shlex.quote(remote)} && rm -rf {shlex.quote(remote + '/colmap')} && "
                     f"mkdir -p {shlex.quote(remote + '/colmap/sparse')}", logs / "ssh-setup.log")
        run([*rsync, f"{folder}/images/", f"{self.host}:{remote}/images/"], logs / "ssh-send.log")
        if (folder / "mask.png").exists():
            run([*rsync, str(folder / "mask.png"), f"{self.host}:{remote}/mask.png"], logs / "ssh-send-mask.log")
        for log, args in plan(remote, -1, self._gpu()):  # -1 is COLMAP's "use every core"
            on_step(log)
            self._remote("QT_QPA_PLATFORM=offscreen colmap " + shlex.join(args), logs / f"{log}.log")
        on_step("fetching the model")
        (folder / "colmap" / "sparse").mkdir(parents=True, exist_ok=True)
        run([*rsync, f"{self.host}:{remote}/colmap/sparse/", f"{folder}/colmap/sparse/"], logs / "ssh-fetch.log")


def pick_runner() -> LocalRunner | SshRunner:
    host = os.environ.get("MAP_BUILD_SSH", "").strip()
    return SshRunner(host) if host else LocalRunner()


# ---------------------------------------------------------------- reading a model
def to_text(model: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("colmap"), "model_converter", "--input_path", str(model), "--output_path", str(out),
                    "--output_type", "TXT"], capture_output=True, check=True)


def read_cameras(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Photo name -> (camera centre, world-to-camera rotation) from a COLMAP images.txt."""
    cameras = {}
    lines = [line for line in images_txt.read_text().splitlines() if not line.startswith("#")]
    for line in lines[::2]:  # each photo takes two lines, and the second may be empty
        fields = line.split()
        if len(fields) < 10:
            continue
        w, x, y, z = (float(v) for v in fields[1:5])
        t = np.array([float(v) for v in fields[5:8]])
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        cameras[fields[9]] = (-r.T @ t, r)
    return cameras


def photos_by_drone(names) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        drone = name.split("/")[0]
        counts[drone] = counts.get(drone, 0) + 1
    return counts


def level_map(cameras: dict, xyz: np.ndarray, tilt_deg: float = TILT_DEG, height_m: float = FLIGHT_HEIGHT_M) -> tuple:
    """COLMAP's axes are arbitrary. Turn the map so the floor is at z=0, up is +z and +x is the way drones faced at takeoff.

    Each camera is tilted the same way, so each one says where up is, and tilt errors cancel across headings.
    The scale comes from assuming drones fly height_m above the floor, so treat it as rough.
    Returns (rotation, scale, offset), where a level point is scale * rotation @ p + offset.
    """
    tilt = math.radians(tilt_deg)
    up_in_camera = np.array([0.0, -math.cos(tilt), -math.sin(tilt)])
    centres = np.array([c for c, _ in cameras.values()])
    up = np.mean([r.T @ up_in_camera for _, r in cameras.values()], axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    first: dict[str, np.ndarray] = {}
    for name in sorted(cameras):
        first.setdefault(name.split("/")[0], cameras[name][1].T @ np.array([0.0, 0.0, 1.0]))
    forward = np.mean(list(first.values()), axis=0)
    forward = forward - forward.dot(up) * up
    if np.linalg.norm(forward) < 1e-3:
        forward = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    rotation = np.array([forward, np.cross(up, forward), up])
    camera_height = float(np.median(centres @ up))
    heights = xyz @ up if len(xyz) else np.zeros(1)
    below = heights[heights < camera_height]
    floor = camera_height - 1.0
    if len(below) >= 16:  # the floor is the lowest strong layer of points
        low = float(np.percentile(below, 1))
        counts, edges = np.histogram(below, bins=64, range=(low, camera_height))
        strong = np.flatnonzero(counts >= 0.4 * counts.max())
        floor = float((edges[strong[0]] + edges[strong[0] + 1]) / 2)
    gap = camera_height - floor
    scale = height_m / gap if gap > 1e-6 else 1.0
    middle = scale * rotation @ np.median(centres, axis=0)
    offset = np.array([-middle[0], -middle[1], -scale * floor])
    return rotation, scale, offset


def _similarity(placed: np.ndarray, tracked: np.ndarray) -> tuple:
    """The least-squares scale, rotation and shift from one set of points to another (Umeyama)."""
    mean_p, mean_t = placed.mean(axis=0), tracked.mean(axis=0)
    a, b = placed - mean_p, tracked - mean_t
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    flip = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[2, 2] = -1
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(d) @ flip) / max((a * a).sum() / len(a), 1e-12))
    return rotation, scale, mean_t - scale * rotation @ mean_p


def fit_to_room(placed: np.ndarray, tracked: np.ndarray) -> tuple:
    """Carry map positions onto tracked ones, ignoring cameras far from the rest: one wild photo mustn't move the fit.

    Returns (rotation, scale, offset, distance of each camera from its tracked position, in metres).
    """
    keep = np.ones(len(placed), bool)
    for _ in range(6):
        rotation, scale, offset = _similarity(placed[keep], tracked[keep])
        errors = np.linalg.norm(scale * placed @ rotation.T + offset - tracked, axis=1)
        close = errors <= max(0.3, 2.0 * float(np.median(errors[keep])))
        if close.sum() < max(6, len(placed) // 2) or np.array_equal(close, keep):
            break
        keep = close
    return rotation, scale, offset, errors


def inside_flight(cloud: np.ndarray, centres: np.ndarray, reach_m: float = CLOUD_REACH_M) -> np.ndarray:
    """Which points lie within reach of where the cameras were. A wrongly placed photo scatters points far outside."""
    low, high = np.percentile(centres, 5, axis=0), np.percentile(centres, 95, axis=0)
    return ((cloud[:, :2] >= low[:2] - reach_m) & (cloud[:, :2] <= high[:2] + reach_m)).all(axis=1) \
        & (cloud[:, 2] >= -1.0) & (cloud[:, 2] <= high[2] + reach_m)


def distinct_maps(parts: list[dict]) -> list[dict]:
    """The models that each add real photos, largest first. COLMAP can place some photos twice, in a small side model."""
    distinct, seen = [], set()
    for part in parts:
        fresh = set(part["cameras"]) - seen
        if len(fresh) >= MIN_JOIN_PHOTOS:
            distinct.append(part)
            seen |= fresh
    return distinct


def read_tracked(folder: Path) -> dict[str, list[float]]:
    """Photo name -> where the tracker said its drone was, for the photos that have a tracked position."""
    sources_path = folder / "photos.json"
    if not sources_path.exists():
        return {}
    sources = json.loads(sources_path.read_text())
    tracked = {}
    for drone in {name.split("/")[0] for name in sources}:
        path = folder / "drones" / drone / "poses.jsonl"
        if not path.exists():
            continue
        known = {}
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
                known[entry["frame"]] = entry["p"]
            except (ValueError, KeyError):  # a line cut short by a crash
                continue
        tracked.update({name: known[raw] for name, raw in sources.items() if name.startswith(drone + "/") and raw in known})
    return tracked


def check_against_room(cameras: dict, tracked: dict) -> dict | None:
    """Compare the map's camera positions with the tracked ones. None when too few photos have both."""
    names = sorted(set(cameras) & set(tracked))
    if len(names) < MIN_FIT_PHOTOS:
        return None
    placed = np.array([cameras[n][0] for n in names])
    where = np.array([tracked[n] for n in names], float)
    if np.ptp(where[:, :2], axis=0).max() < 0.5:  # a hover can't fix a rotation
        return None
    rotation, scale, offset, errors = fit_to_room(placed, where)
    return {"photos": len(names), "median_error_m": round(float(np.median(errors)), 3), "agrees": round(float((errors <= ROOM_FIT_LIMIT_M).mean()), 3),
            "rotation": rotation, "scale": scale, "offset": offset}


# ---------------------------------------------------------------- the build
def build_map(folder: Path, cap: int, runner, on_step: Callable[[str], None] = lambda step: None, shared_camera: bool = False,
              fix_lens: bool = False) -> dict:
    """Everything from saved frames to a finished map. Raises StepFailed or ToolMissing with a readable message."""
    times, started = {}, time.perf_counter()
    (folder / "result.json").unlink(missing_ok=True)  # a restart mustn't serve the old map if this build fails

    def step(name: str) -> None:
        times[name] = time.perf_counter()
        on_step(name)

    drones = [p for p in sorted((folder / "drones").iterdir()) if p.is_dir() and frame_files(p / "raw")] \
        if (folder / "drones").exists() else []
    if not drones:
        raise StepFailed("No frames saved for this map. Start a capture while the drones are flying first.")
    step("choosing photos")
    with ThreadPoolExecutor(len(drones)) as workers:  # child processes, so they can't slow the flight loop's thread
        list(workers.map(lambda d: run(["nice", "-n", "10", sys.executable, "-m", "mapping.fleet", "prepare", d],
                                       d / "prepare.log", env={"PYTHONPATH": str(BACKEND)}), drones))
    pooled = pool_photos(folder, cap)
    total = sum(pooled.values())
    if total < MIN_PHOTOS:
        raise StepFailed(f"Only {total} usable photos: a map needs at least {MIN_PHOTOS}, and ideally 100 or more.")
    mask = static_mask(folder)
    first = next((folder / "images" / d / "00001.jpg" for d in pooled), None)
    with quiet_jpeg_warnings():
        shape = cv2.imread(str(first)).shape

    shutil.rmtree(folder / "colmap", ignore_errors=True)
    (folder / "colmap" / "sparse").mkdir(parents=True)

    def plan(root: str, threads: int, gpu: bool) -> list:
        return colmap_steps(root, shape[1], shape[0], threads, gpu, mask is not None, shared_camera, fix_lens)

    used = runner
    try:
        runner.run(folder, plan, step)
    except (StepFailed, ToolMissing, OSError, subprocess.SubprocessError) as error:
        if isinstance(runner, LocalRunner):
            raise
        logger.warning("[map] %s failed, building here instead: %s", runner.name, error)
        shutil.rmtree(folder / "colmap", ignore_errors=True)
        (folder / "colmap" / "sparse").mkdir(parents=True)
        used = LocalRunner()
        used.run(folder, plan, step)

    step("reading the model")
    models = [analyze(p) for p in sorted((folder / "colmap" / "sparse").iterdir()) if p.is_dir()]
    best = best_model(models)
    if best is None or best.registered < 3:
        raise StepFailed("COLMAP could not link the photos into a 3D model. Usually that means too little overlap or too "
                         "little detail (blank walls). Fly slower and keep textured surfaces in view.")
    tracked = read_tracked(folder)
    parts = []
    for model in sorted(models, key=lambda m: -m.registered):
        text = folder / "colmap" / "text" / model.path.name
        to_text(model.path, text)
        cameras = read_cameras(text / "images.txt")
        parts.append({"model": model, "cameras": cameras, "cloud": read_points(text / "points3D.txt"),
                      "fit": check_against_room(cameras, tracked)})
    summaries = [{"folder": part["model"].path.name, "registered": part["model"].registered, "points": part["model"].points,
                  "error_px": round(part["model"].error_px, 3), "by_drone": photos_by_drone(part["cameras"]),
                  "room_fit_m": part["fit"]["median_error_m"] if part["fit"] else None,
                  "room_agreement": part["fit"]["agrees"] if part["fit"] else None} for part in parts]
    distinct = distinct_maps(parts)
    joined = [part for part in distinct if part["fit"] and part["fit"]["agrees"] >= MIN_ROOM_AGREEMENT]
    main = parts[0]
    if joined:  # room coordinates, so sub-maps that share no view still line up
        clouds = [(part["fit"]["scale"] * part["cloud"][0] @ part["fit"]["rotation"].T + part["fit"]["offset"], part["cloud"][1])
                  for part in joined]
        level, rgb = np.vstack([c[0] for c in clouds]), np.vstack([c[1] for c in clouds])
        rotation, scale, offset = joined[0]["fit"]["rotation"], joined[0]["fit"]["scale"], joined[0]["fit"]["offset"]
        served = joined
        where = np.vstack([np.array([c for c, _ in part["cameras"].values()]) * part["fit"]["scale"] @ part["fit"]["rotation"].T
                           + part["fit"]["offset"] for part in joined])
    else:
        xyz, rgb = main["cloud"]
        rotation, scale, offset = level_map(main["cameras"], xyz)
        level = scale * xyz @ rotation.T + offset if len(xyz) else xyz
        served = [main]
        where = np.array([c for c, _ in main["cameras"].values()]) * scale @ rotation.T + offset
    keep = inside_flight(level, where) if len(level) else np.zeros(0, bool)
    level, rgb = level[keep], rgb[keep]
    write_ply(folder / "cloud.ply", level.astype(np.float32), rgb)
    np.save(folder / "points.npy", np.hstack([level, rgb]).astype(np.float32))
    try:
        from mapping.cloud_view import write_viewer
        write_viewer(folder / "cloud.ply", folder / "cloud.html", z_up=True)
    except Exception as error:
        logger.warning("[map] couldn't write the cloud viewer: %s", error)

    names = {name for part in served for name in part["cameras"]}
    placed = photos_by_drone(names)
    registered = len(names)
    warnings = coverage_warnings(Model(main["model"].path, registered, len(level), main["model"].error_px), total)
    for drone, count in pooled.items():
        if placed.get(drone, 0) < 0.3 * count:
            warnings.append(f"Drone {drone} barely joined the map: {placed.get(drone, 0)} of {count} photos placed. "
                            "Its view may not overlap the other drones'.")
    if len(joined) > 1:
        warnings.append(f"The drones' photos didn't link by sight, so {len(joined)} separate maps were joined using tracked "
                        "positions. Their edges only line up as well as the tracking does.")
    elif not joined and len(distinct) > 1:
        warnings.append(f"The photos split into {len(distinct)} separate maps that don't overlap, and this shows only the largest. "
                        "Fly so the drones see some of the same floor, or give the build tracked positions.")
    wrong = [part for part in distinct if part["fit"] and part["fit"]["agrees"] < MIN_ROOM_AGREEMENT]
    if wrong:
        share = wrong[0]["fit"]["agrees"]
        warnings.append(f"A map with {wrong[0]['model'].registered} photos disagrees with where the drones were tracked: only {share:.0%} of its "
                        f"photos are within {ROOM_FIT_LIMIT_M} m. It is probably warped or folded" + (" and was left out." if joined else ". Don't trust its shape.") +
                        " Repeating floor patterns or a drone that never turned can do this.")
    result = {"photos": total, "by_drone": pooled, "registered": registered, "registered_by_drone": placed,
              "models": summaries, "points": len(level), "error_px": round(main["model"].error_px, 3), "warnings": warnings,
              "masked": mask is not None, "runner": used.name, "in_room": bool(joined), "joined_maps": len(served),
              "room_fit_m": main["fit"]["median_error_m"] if main["fit"] else None,
              "level": {"rotation": rotation.tolist(), "scale": scale, "offset": offset.tolist()}}
    marks = sorted(times.items(), key=lambda item: item[1]) + [("end", time.perf_counter())]
    result["seconds"] = {name: round(next_t - t, 1) for (name, t), (_, next_t) in zip(marks, marks[1:])}
    result["seconds"]["total"] = round(time.perf_counter() - started, 1)
    (folder / "result.json").write_text(json.dumps(result, indent=1))
    return result


def build_splat(folder: Path, steps: int = SPLAT_STEPS) -> Path:
    """Train a Gaussian splat from a finished build. Raises ToolMissing when Brush isn't installed."""
    result = json.loads((folder / "result.json").read_text())
    best = max(result["models"], key=lambda m: m["registered"])
    model = Model(folder / "colmap" / "sparse" / best["folder"], best["registered"], best["points"], best["error_px"])
    return train(folder / "images", model, folder, steps)


def tracked_pose_of(hub, max_age_s: float = 0.5) -> Callable:
    """A pose_of for FleetCapture that reads the hub's tracker, and gives None once a drone has lost its floor tags."""
    def pose_of(drone: str):
        tracked = hub.co.drones[drone].tracked
        if getattr(tracked, "fix_age", 0.0) > max_age_s:
            return None
        return tracked.p
    return pose_of


# ---------------------------------------------------------------- the part everyone else calls
class FleetCapture:
    def __init__(self, frames, root, drones=None, max_photos: int | None = None, on_event=None, pose_of=None, flying=None):
        """pose_of(drone) -> tracked (x, y, z) or None, optional and called off the flight loop.

        With it, a finished map is checked against the tracked positions and placed in room coordinates.
        Without it, the map is levelled with a rough scale and no one can tell if its shape is wrong.

        flying(drone) -> bool, optional: frames from a drone on the floor aren't saved, or a landed one
        would keep using up its share of the photo cap on the same patch of floor.
        """
        self.frames = frames
        self.root = Path(root)
        self.drones = list(drones) if drones is not None else None
        self.max_photos = max_photos or int(os.environ.get("MAP_BUILD_PHOTOS") or MAX_PHOTOS)
        self.on_event = on_event
        self.pose_of = pose_of
        self.flying = flying
        self.lock = threading.Lock()
        self.name: str | None = None
        self.state = "idle"
        self.step: str | None = None
        self.error: str | None = None
        self.saved: dict[str, int] = {}
        self.result: dict | None = None
        self.splat: str | None = None
        self.stopping = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.build_thread: threading.Thread | None = None
        self._load_newest()

    def _newest(self) -> tuple[str, dict] | None:
        try:
            results = sorted(self.root.glob("*/result.json"), key=lambda p: p.stat().st_mtime)
            if results:
                return results[-1].parent.name, json.loads(results[-1].read_text())
        except (OSError, ValueError) as error:
            logger.warning("[map] couldn't read the last map: %s", error)
        return None

    def _load_newest(self) -> None:
        """A restarted backend still serves the last finished map."""
        newest = self._newest()
        if newest is not None:
            self.name, self.result, self.state = newest[0], newest[1], "done"

    def _drone_ids(self) -> list[str]:
        if self.drones is not None:
            return self.drones
        for source in (self.frames, getattr(self.frames, "fleet", None)):
            for attribute in ("backends", "drones"):
                known = getattr(source, attribute, None)
                if known:
                    if isinstance(known, dict):
                        return [str(d) for d in known]
                    return [str(getattr(d, "id", d)) for d in known]
        return []

    def start(self, name: str) -> dict:
        safe_name(name)
        with self.lock:
            if self.state == "capturing":
                raise MapError(f"Already capturing map {self.name!r}. Stop it first.")
            if self.state == "building":
                raise MapError(f"Map {self.name!r} is still building. Wait for it to finish.")
            self.name, self.state, self.step, self.error = name, "capturing", None, None
            self.saved, self.result, self.splat = {}, None, None
            self.stopping = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture, args=(name, self.stopping, self.capture_thread),
                                                   daemon=True, name="map-capture")
            self.capture_thread.start()
        return self.status()

    def _capture(self, name: str, stopping: threading.Event, previous: threading.Thread | None = None) -> None:
        """Save each new frame from each drone. Frames may be missed if the disk is slow; the flight never waits.

        Starting a map that already has frames carries on from them, so a battery swap doesn't lose a map.
        """
        if previous is not None:
            previous.join()  # it may still be finishing a write
        folder = self.root / name / "drones"
        seen: dict[str, int] = {}
        counts: dict[str, int] = {}
        carried_ms = 0  # later than every frame already saved
        if folder.exists():
            for path in folder.iterdir():
                earlier = frame_files(path / "raw")
                if earlier:
                    counts[path.name] = len(earlier)
                    carried_ms = max(carried_ms, (frame_time_ms(earlier[-1]) or 0) + 1000)
            with self.lock:
                self.saved = dict(counts)
        started = None  # the first frame's time: real and simulated frames run on different clocks
        while not stopping.wait(POLL_S):
            for drone in self._drone_ids():
                try:
                    frame = self.frames.latest(drone)
                    key = drone_folder(drone)
                    if frame is None or seen.get(drone) == frame.number or counts.get(key, 0) >= MAX_FRAMES_PER_DRONE:
                        continue
                    if self.flying is not None and not self.flying(drone):
                        continue
                    seen[drone] = frame.number
                    raw = folder / key / "raw"
                    raw.mkdir(parents=True, exist_ok=True)
                    index = counts.get(key, 0)
                    started = frame.t if started is None else started
                    t_ms = carried_ms + max(0, int((frame.t - started) * 1000))
                    name = raw_name(index, t_ms)
                    (raw / name).write_bytes(frame.jpeg)
                    self._save_pose(drone, raw.parent / "poses.jsonl", name)
                    counts[key] = index + 1
                    with self.lock:
                        self.saved[key] = index + 1
                except Exception as error:  # a full disk loses frames, nothing more
                    message = f"Couldn't save a frame from drone {drone}: {error}"
                    with self.lock:
                        repeated, self.error = self.error == message, message
                    if not repeated:  # this loop runs 50 times a second
                        logger.warning("[map] %s", message)

    def _save_pose(self, drone: str, path: Path, frame: str) -> None:
        """A tracker fault costs this frame its position and nothing else."""
        if self.pose_of is None:
            return
        try:
            where = self.pose_of(drone)
            if where is not None:
                with open(path, "a") as f:
                    f.write(json.dumps({"frame": frame, "p": [round(float(v), 4) for v in where[:3]]}) + "\n")
        except Exception as error:
            logger.debug("[map] no position for %s: %s", drone, error)

    def stop(self) -> dict:
        """Stop saving frames and return at once, so a mission can call this from the flight loop.

        The capture thread ends by itself after any write in progress. Safe when nothing is capturing.
        """
        with self.lock:
            if self.state == "capturing":
                self.stopping.set()
                self.state = "captured"
        return self.status()

    def status(self) -> dict:
        with self.lock:
            result = self.result or {}
            frames = sum(self.saved.values())
            return {"name": self.name, "state": self.state, "step": self.step,
                    "photos": result.get("photos", frames), "by_drone": dict(result.get("by_drone", self.saved)),
                    "frames": frames, "registered": result.get("registered", 0),
                    "registered_by_drone": dict(result.get("registered_by_drone", {})),
                    "models": [m["registered"] for m in result.get("models", [])], "points": result.get("points", 0),
                    "in_room": result.get("in_room", False), "room_fit_m": result.get("room_fit_m"),
                    "joined_maps": result.get("joined_maps", 0),
                    "warnings": list(result.get("warnings", [])), "seconds": dict(result.get("seconds", {})),
                    "runner": result.get("runner"), "splat": self.splat, "error": self.error}

    def build(self, name: str | None = None, splat: bool | None = None) -> dict:
        """Build a map in the background and return straight away. A failure shows up in status()['error'].

        Ends any capture of the same map first. Asking again during a build changes nothing.
        """
        name = safe_name(name if name is not None else (self.name or ""))
        if splat is None:
            splat = os.environ.get("MAP_SPLAT", "") == "1"
        if self.status()["state"] == "capturing" and self.name == name:
            self.stop()
        with self.lock:
            if self.state == "capturing":
                raise MapError(f"Map {self.name!r} is still capturing. Stop it before building {name!r}.")
            if self.state != "building":
                if self.name != name:
                    self.saved = {}
                self.name, self.state, self.step, self.error, self.result, self.splat = name, "building", None, None, None, None
                self.build_thread = threading.Thread(target=self._build, args=(name, splat, self.capture_thread),
                                                     daemon=True, name="map-build")
                self.build_thread.start()
        return self.status()

    def _on_step(self, step: str) -> None:
        with self.lock:
            self.step = step

    def _build(self, name: str, splat: bool, capture: threading.Thread | None = None) -> None:
        folder = self.root / name
        if capture is not None:
            capture.join()  # choose photos only once the last frame is on disk
        try:
            runner = pick_runner()
        except StepFailed as error:
            logger.warning("[map] %s Building here instead.", error)
            runner = LocalRunner()
        try:
            result = build_map(folder, self.max_photos, runner, self._on_step)
            with self.lock:
                self.result, self.state, self.step = result, "done", None
            self._report("map", f"Map {name!r} is built: {result['registered']} of {result['photos']} photos placed, "
                                f"{result['points']} points.", result)
        except Exception as error:
            logger.warning("[map] build of %s failed: %s", name, error)
            with self.lock:
                self.state, self.step, self.error = "failed", None, str(error)
            self._report("map", f"Map {name!r} failed: {error}", {"name": name})
            return
        if splat:
            self._train_splat(folder)

    def _train_splat(self, folder: Path) -> None:
        """Optional and after the fact: the point cloud is already being served while this runs."""
        with self.lock:
            self.splat = "training"
        try:
            path = build_splat(folder)
            with self.lock:
                self.splat = str(path)
        except ToolMissing:
            with self.lock:
                self.splat = "skipped: Brush isn't installed"
        except Exception as error:
            logger.warning("[map] splat failed: %s", error)
            with self.lock:
                self.splat = f"failed: {error}"

    def _report(self, kind: str, text: str, data: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, text, data)
        except Exception as error:
            logger.warning("[map] on_event failed: %s", error)

    def wait(self, timeout: float | None = None) -> dict:
        """Block until a running build ends. For benches and tests; the flight loop must never call this."""
        thread = self.build_thread
        if thread is not None:
            thread.join(timeout)
        return self.status()

    def points(self, limit: int = 50000) -> list[list[float]]:
        """A random sample of the map as [[x, y, z, r, g, b], ...], for the browser. Empty until a build has finished."""
        with self.lock:
            name, ready = self.name, self.result is not None
        if not name or not ready:
            return []
        try:
            cloud = np.load(self.root / name / "points.npy")
        except (OSError, ValueError) as error:
            logger.warning("[map] couldn't read points for %s: %s", name, error)
            return []
        if limit is not None and len(cloud) > limit:
            cloud = cloud[np.random.default_rng(0).choice(len(cloud), int(max(limit, 0)), replace=False)]
        out = np.round(cloud.astype(np.float64), 3).tolist()
        return [[x, y, z, int(r), int(g), int(b)] for x, y, z, r, g, b in out]


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        print(prepare_drone(Path(sys.argv[2])))
        return 0
    print("usage: python -m mapping.fleet prepare <drone folder>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
