"""Does a pooled map from three drones land in one model? Measured on the simulator, not on drones.

Renders three simulated S20 flights with the 45 degree camera, at the real frame rate and JPEG size,
hands the frames to FleetCapture and builds. Reports photos placed per drone, how many models came
out, how long it took, and how far the placed cameras are from where the simulator really put them.

    uv run python -m mapping.fleet_bench --seconds 180
    uv run python -m mapping.fleet_bench --layout apart --seconds 60    # the hard case: little overlap
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from mapping import fleet
from mapping.curate import raw_name
from mapping.fleet import read_cameras
from sim.camera import HEIGHT, SIZES, WIDTH, Camera, Pose, VideoEncoder
from sim.scenes import SCENES
from swarm.sim_cameras import DecalCamera, SimFrames, StillFleet, demo_world

FPS = 17.6  # measured from the S20, see sim/camera.py
TILT = math.radians(45.0)  # estimate: the S20's fixed camera
CRUISE = 1.0  # metres above the floor
SPEED, TURN = 0.4, 0.55  # m/s and rad/s, the same as swarm/search.py
DWELL_S = 2.0
ROOM_PADS = {"1": (-1.2, -0.8), "2": (-1.2, 0.0), "3": (-1.2, 0.8)}


def rotation(yaw: float, pitch_down: float, roll: float = 0.0) -> tuple:
    """Body-to-world, row by row, for a body with x forward, y left, z up."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch_down), math.sin(pitch_down)
    cr, sr = math.cos(roll), math.sin(roll)
    return (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr)


def turn_toward(yaw: float, goal: float, step: float) -> float:
    gap = (goal - yaw + math.pi) % (2 * math.pi) - math.pi
    return yaw + max(-step, min(step, gap))


def fly(pad: tuple, viewpoints: list, seconds: float, seed: int) -> list[tuple]:
    """(t, x, y, z, yaw, pitch, roll) at each camera frame: take off from the pad, then visit viewpoints in a loop."""
    rng = random.Random(seed)
    x, y = pad
    z, yaw = 0.05, 0.0
    dt = 1 / FPS
    phase = rng.uniform(0, 6.28)
    out, leg, dwell = [], 0, 0.0
    for i in range(int(seconds * FPS)):
        t = i * dt
        if z < CRUISE - 0.01:
            z += min(0.4 * dt, CRUISE - z)
        else:
            gx, gy, gyaw = viewpoints[leg % len(viewpoints)]
            gap = math.hypot(gx - x, gy - y)
            if gap > 0.01:
                step = min(gap, SPEED * dt)
                x += (gx - x) / gap * step
                y += (gy - y) / gap * step
            yaw = turn_toward(yaw, gyaw, TURN * dt)
            if gap <= 0.01 and abs((gyaw - yaw + math.pi) % (2 * math.pi) - math.pi) < 0.01:
                dwell += dt
                if dwell >= DWELL_S:
                    leg, dwell = leg + 1, 0.0
        # A toy drone never sits still: it bobs and rocks.
        bob = 0.04 * math.sin(0.9 * t + phase) + 0.015 * math.sin(3.1 * t)
        pitch = TILT + math.radians(2.5) * math.sin(1.7 * t + phase) + rng.gauss(0, math.radians(0.25))
        roll = math.radians(2.5) * math.sin(1.3 * t + 2 * phase) + rng.gauss(0, math.radians(0.25))
        out.append((t, x, y, z + bob, yaw, pitch, roll))
    return out


def layouts(name: str, world: str = "demo") -> dict[str, list]:
    """Viewpoints per drone as (x, y, yaw). Each drone keeps to its own strip of the room."""
    q = math.pi / 2
    if world == "demo":  # 8 x 6 m, swarm/sim_cameras.py
        if name == "mission":  # what the search planner does: stop, look, move on, any of eight headings
            return {
                "1": [(-2.5, -1.8, 0), (-0.8, -1.6, q / 2), (1.2, -2.0, 0), (3.0, -1.6, q), (2.6, -1.0, 2 * q), (0.4, -1.2, -q), (-1.8, -1.4, 3 * q / 2)],
                "2": [(-2.0, 0.0, 0), (0.0, 0.2, -q / 2), (2.0, 0.0, q / 2), (3.2, -0.2, 2 * q), (1.0, 0.2, q), (-1.0, -0.2, -q), (-3.0, 0.0, 0)],
                "3": [(-2.5, 1.8, 0), (-0.6, 1.2, -q / 2), (1.4, 1.8, 0), (3.2, 1.2, -q), (2.0, 1.0, 2 * q), (0.0, 1.6, -3 * q / 2), (-2.0, 1.0, -q)],
            }
        if name == "apart":  # the hard case: each drone only looks outward, away from the others
            return {
                "1": [(-2.5, -1.2, -q), (-0.5, -1.2, -q), (1.5, -1.2, -q), (3.0, -1.2, -q), (0.5, -0.8, -q)],
                "2": [(-1.0, 0.0, 0), (1.0, 0.0, 0), (2.8, 0.0, 0), (0.5, 0.0, 0)],
                "3": [(-2.5, 1.2, q), (-0.5, 1.2, q), (1.5, 1.2, q), (3.0, 1.2, q), (0.5, 0.8, q)],
            }
    elif world == "room":  # 6.6 x 5.2 m, sim/scenes.py, whose floor and walls repeat exactly
        if name == "mission":
            return {
                "1": [(-1.8, -1.6, 0), (0.0, -1.2, q / 2), (1.8, -1.6, q), (2.4, -0.9, 2 * q), (0.4, -1.0, -q), (-1.8, -1.0, 3 * q / 2)],
                "2": [(-1.2, 0.0, 0), (0.6, 0.2, -q / 2), (2.4, 0.0, q / 2), (1.2, -0.2, 2 * q), (-0.6, 0.2, q), (-2.2, 0.0, -q)],
                "3": [(-1.8, 1.4, 0), (0.0, 1.4, -q / 2), (1.4, 1.0, -q), (0.2, 0.9, 2 * q), (-1.0, 1.6, -3 * q / 2), (-2.2, 1.0, -q)],
            }
        if name == "apart":
            return {
                "1": [(-1.8, -1.4, -q), (0.0, -1.4, -q), (2.0, -1.4, -q), (0.0, -1.0, -q)],
                "2": [(0.0, 0.0, 0), (1.4, 0.0, 0), (2.2, 0.0, 0), (1.0, 0.0, 0)],
                "3": [(-1.8, 1.0, q), (-0.5, 1.0, q), (1.0, 1.0, q), (-0.5, 0.7, q)],
            }
    raise ValueError(f"unknown layout {name!r} for {world!r}")


def pads_for(world: str) -> dict:
    return dict(demo_world().pads) if world == "demo" else dict(ROOM_PADS)


def make_camera(world: str, walls: str, scale: float, seed: int, fov_deg: float = 90.0) -> Camera:
    fov = math.radians(fov_deg)
    if world == "demo":
        scene = demo_world().scene
        return DecalCamera(scene, scene.decals, render_scale=scale, fov=fov, seed=seed)
    return Camera(replace(SCENES[world](), wall_material=walls, ceiling_material=walls), render_scale=scale, fov=fov, seed=seed)


class Rough:
    """Faults a cheap camera has and the renderer doesn't. The amounts are guesses, not measurements of an S20."""

    LENS = {"1": (86.0, -0.10), "2": (90.0, -0.16), "3": (95.0, -0.06)}  # horizontal view and radial distortion, per drone
    EXPOSURE_S = 1 / 40

    def __init__(self, drone: str, width: int = WIDTH, height: int = HEIGHT):
        self.fov, k = self.LENS[drone]
        self.focal = width / 2 / math.tan(math.radians(self.fov) / 2)
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        x, y = (u - width / 2) / self.focal, (v - height / 2) / self.focal
        factor = 1 + k * (x * x + y * y)
        self.map_x = x * factor * self.focal + width / 2
        self.map_y = y * factor * self.focal + height / 2
        self.rng = random.Random(int(drone))
        self.gain = 1.0

    def apply(self, image: np.ndarray, moved_m: float, turned_rad: float, dt: float) -> np.ndarray:
        image = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sideways = abs(turned_rad) / dt * self.focal * self.EXPOSURE_S
        along = moved_m / dt / 1.4 * self.focal * self.EXPOSURE_S
        length = int(round(max(sideways, along)))
        if length >= 2:
            kernel = np.zeros((length, length), np.float32)
            if sideways >= along:
                kernel[length // 2, :] = 1.0 / length
            else:
                kernel[:, length // 2] = 1.0 / length
            image = cv2.filter2D(image, -1, kernel)
        self.gain = min(1.15, max(0.85, self.gain + self.rng.gauss(0, 0.01)))  # auto exposure hunting
        return cv2.convertScaleAbs(image, alpha=self.gain)


def _render_drone(job: tuple) -> dict:
    drone, track, folder, world, walls, scale, rough = job
    cv2.setNumThreads(1)
    raw = Path(folder) / "drones" / drone / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lens = Rough(drone) if rough else None
    camera = make_camera(world, walls, scale, seed=int(drone), fov_deg=lens.fov if lens else 90.0)
    encoder = VideoEncoder(seed=int(drone))
    started = time.perf_counter()
    sizes, truth = [], {}
    last = track[0]
    for i, (t, x, y, z, yaw, pitch, roll) in enumerate(track):
        image = camera.render(Pose(x, y, z, rotation(yaw, pitch, roll)))
        if lens:
            image = lens.apply(image, math.hypot(x - last[1], y - last[2]), yaw - last[4], 1 / FPS)
        last = (t, x, y, z, yaw, pitch, roll)
        jpeg = encoder.compress(image, SIZES[1])
        name = raw_name(i, int(t * 1000))
        (raw / name).write_bytes(jpeg)
        sizes.append(len(jpeg))
        truth[name] = [x, y, z, yaw]
    return {"drone": drone, "frames": len(track), "seconds": time.perf_counter() - started,
            "mean_jpeg_kb": sum(sizes) / max(len(sizes), 1) / 1000, "truth": truth}


def render_flights(folder: Path, seconds: float, layout: str, world: str = "demo", walls: str = "brick", scale: float = 0.54,
                   rough: bool = False) -> dict:
    """Write each drone's frames to folder/drones/<id>/raw, as FleetCapture would. Returns timing and truth."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    plan, pads = layouts(layout, world), pads_for(world)
    jobs = [(d, fly(pads[d], plan[d], seconds, seed=int(d) * 11), str(folder), world, walls, scale, rough) for d in plan]
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_render_drone, jobs)
    truth = {r["drone"]: r.pop("truth") for r in results}
    (folder / "truth.json").write_text(json.dumps(truth))
    return {"wall_s": time.perf_counter() - started, "drones": results}


def align(model: np.ndarray, truth: np.ndarray) -> tuple[float, np.ndarray]:
    """Scale, rotate and shift model points onto the truth (Umeyama). Returns (scale, aligned points)."""
    mean_m, mean_t = model.mean(axis=0), truth.mean(axis=0)
    a, b = model - mean_m, truth - mean_t
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    flip = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[2, 2] = -1
    rotation = u @ flip @ vt
    variance = (a * a).sum() / len(a)
    scale = float(np.trace(np.diag(d) @ flip) / max(variance, 1e-12))
    return scale, scale * a @ rotation.T + mean_t


def evaluate(folder: Path) -> dict:
    """How far the placed cameras are from where the simulator flew them, drone by drone, in metres."""
    result = json.loads((folder / "result.json").read_text())
    truth = json.loads((folder / "truth.json").read_text())
    sources = json.loads((folder / "photos.json").read_text())
    best = max(result["models"], key=lambda m: m["registered"])
    cameras = read_cameras(folder / "colmap" / "text" / best["folder"] / "images.txt")
    names = sorted(cameras)
    placed = np.array([cameras[n][0] for n in names])
    real = np.array([truth[n.split("/")[0]][sources[n]][:3] for n in names])
    _, aligned = align(placed, real)
    errors = np.linalg.norm(aligned - real, axis=1)
    per_drone = {}
    for drone in sorted({n.split("/")[0] for n in names}):
        mine = errors[[i for i, n in enumerate(names) if n.startswith(drone + "/")]]
        per_drone[drone] = {"placed": len(mine), "median_m": round(float(np.median(mine)), 3), "p95_m": round(float(np.percentile(mine, 95)), 3)}
    level = result["level"]
    rotation, scale, offset = np.array(level["rotation"]), level["scale"], np.array(level["offset"])
    in_level = scale * placed @ rotation.T + offset
    # The level map should differ from the room only by a shift: same way up, same heading, about the same size.
    a, b = in_level - in_level.mean(axis=0), real - real.mean(axis=0)
    turn = math.degrees(math.atan2(float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()), float((a[:, :2] * b[:, :2]).sum())))
    cloud = np.load(folder / "points.npy")
    return {"camera_error_m": {"median": round(float(np.median(errors)), 3), "p95": round(float(np.percentile(errors, 95)), 3),
                               "max": round(float(errors.max()), 3)},
            "by_drone": per_drone,
            "level_map": {"camera_height": round(float(np.median(in_level[:, 2])), 3), "true_camera_height": round(float(np.median(real[:, 2])), 3),
                          "heading_off_deg": round(turn, 1), "size_vs_truth": round(float(np.linalg.norm(a[:, :2]) / max(np.linalg.norm(b[:, :2]), 1e-9)), 3),
                          "points_z_p5_p50_p95": np.percentile(cloud[:, 2], [5, 50, 95]).round(3).tolist()}}


def live(root: Path, seconds: float, rate_hz: float, splat: bool) -> dict:
    """The whole path in real time: SimFrames video, FleetCapture saving it, a build in the background, points()."""
    world = demo_world()
    plan = layouts("mission", "demo")
    tracks = {d: fly(world.pads[d], plan[d], seconds + 5.0, seed=int(d) * 11) for d in plan}
    fleet_now = StillFleet(world.scene, {d: (*world.pads[d], 0.05, 0.0) for d in plan})
    frames = SimFrames(fleet_now, rate_hz=rate_hz, clock=time.monotonic)
    capture = fleet.FleetCapture(frames, root)
    stop = threading.Event()

    def move() -> None:
        started = time.monotonic()
        while not stop.wait(0.02):
            i = int((time.monotonic() - started) * FPS)
            for drone, track in tracks.items():
                _, x, y, z, yaw, _, _ = track[min(i, len(track) - 1)]
                hover = fleet_now.drones[drone]
                hover.x, hover.y, hover.z, hover.heading = x, y, z, yaw

    mover = threading.Thread(target=move, daemon=True)
    mover.start()
    frames.start()
    capture.start("live")
    time.sleep(seconds)
    print("sim: while capturing:", capture.status()["by_drone"], frames.stats(), flush=True)
    capture.build("live", splat=splat)
    time.sleep(1.0)
    print("sim: a second after asking for a build:", capture.status()["state"], capture.status()["step"], flush=True)
    stop.set()
    frames.stop()
    status = capture.wait()
    points = capture.points(limit=5000)
    status["sample"] = len(points)
    if points:
        cloud = np.array(points)
        status["level_z_percentiles"] = np.percentile(cloud[:, 2], [5, 50, 95]).round(3).tolist()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(prog="mapping.fleet_bench", description=__doc__)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--layout", default="mission", choices=["mission", "apart"])
    parser.add_argument("--world", default="demo", choices=["demo", *sorted(SCENES)], help="demo's floor never repeats; the others' textures do")
    parser.add_argument("--walls", default="brick")
    parser.add_argument("--scale", type=float, default=0.54, help="render scale; 0.54 is 640 x 360, what SimFrames uses")
    parser.add_argument("--root", default="/tmp/zwarm-fleet-bench")
    parser.add_argument("--name", default=None)
    parser.add_argument("--cap", type=int, default=fleet.MAX_PHOTOS, help="most photos to give COLMAP")
    parser.add_argument("--shared-camera", action="store_true", help="one camera model for all drones")
    parser.add_argument("--fix-lens", action="store_true", help="hold the lens at the --hfov guess instead of refining it")
    parser.add_argument("--hfov", type=float, default=fleet.HFOV_DEG, help="starting guess for the lens")
    parser.add_argument("--reuse", help="build from another run's frames instead of rendering, naming its folder under --root")
    parser.add_argument("--rough", action="store_true", help="different lens per drone, distortion, motion blur, exposure drift")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--live", action="store_true", help="run capture and build against SimFrames in real time")
    parser.add_argument("--rate", type=float, default=5.0, help="frames a second per drone in --live")
    parser.add_argument("--splat", action="store_true", help="in --live, also train a splat")
    args = parser.parse_args()
    if args.live:
        root = Path(args.root) / "live-root"
        shutil.rmtree(root, ignore_errors=True)
        print("sim:", json.dumps(live(root, args.seconds, args.rate, args.splat), indent=1))
        return 0
    name = args.name or f"{args.layout}-{int(args.seconds)}s"
    folder = Path(args.root) / name
    if args.reuse:
        source = Path(args.root) / args.reuse
        shutil.rmtree(folder, ignore_errors=True)
        (folder / "drones").mkdir(parents=True)
        shutil.copy(source / "truth.json", folder / "truth.json")
        for drone in (source / "drones").iterdir():
            (folder / "drones" / drone.name).mkdir()
            (folder / "drones" / drone.name / "raw").symlink_to(drone / "raw")
    else:
        timing = render_flights(folder, args.seconds, args.layout, args.world, args.walls, args.scale)
        print(f"sim: rendered {sum(d['frames'] for d in timing['drones'])} frames in {timing['wall_s']:.0f} s")
    if args.render_only:
        return 0
    fleet.HFOV_DEG = args.hfov
    started = time.perf_counter()
    result = fleet.build_map(folder, args.cap, fleet.LocalRunner(), lambda s: print(f"  {time.perf_counter() - started:6.1f} s  {s}", flush=True),
                             shared_camera=args.shared_camera)
    report = {key: result[key] for key in ("photos", "by_drone", "registered", "registered_by_drone", "points", "error_px",
                                           "masked", "seconds", "warnings")}
    report["models"] = [{k: m[k] for k in ("registered", "points", "by_drone")} for m in result["models"]]
    report["truth"] = evaluate(folder)
    print("sim:", json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
