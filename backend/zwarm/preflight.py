"""A read-only check before flying: `python -m zwarm preflight FILE`.

To a drone's backend this only ever sends GET /capabilities and GET /mjpeg. It opens no control
websocket and calls no flight route, so it can run beside a drone in the air, which is what --watch
is for. A test holds it to that (tests/test_zwarm_preflight.py).

A drone on the floor can't see the floor, so the tag and pose checks need each drone lifted by hand
to about 1 m over its pad. --on-pad skips them for a quick re-check.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import tempfile
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from swarm.frames import StreamFrames
from swarm.inspect import DATA_ROOT, MIN_SHARPNESS, sharpness
from swarm.tracking import CAMERA_HEIGHT_M, VIDEO_HEIGHT, VIDEO_WIDTH, AnchorReader, Lens, anchors_in_view, solve_fix
from zwarm.app import Readiness, check_setup, load_swarm_file

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
NOT_READY = 2

WATCH_S = 25.0  # time to lift three drones one after another
ON_PAD_WATCH_S = 6.0
ENOUGH_S = 5.0  # with every drone already showing a good pose, stop waiting after this long
ENOUGH_FIXES = 15
PRINT_EVERY_S = 3.0
RECENT_FIXES = 40  # what --watch and the live line report from, about 3 s of video

MIN_FPS = 4.0  # floor-tag flights still worked at 4 frames a second in the simulator
GOOD_FPS = 8.0  # the S20 has measured 11 to 19
VIDEO_STOPPED_S = 2.0
PAD_MISS_M = 0.3  # the same limit Coordinator.preflight refuses a takeoff at
HEADING_MISS_DEG = 25.0
TILT_MISS_DEG = 10.0
BAD_FIT_PX = 2.0  # MAX_REPROJECTION_PX rejects at 4
GOOD_SIGNAL = 55  # docs/research/S2x.md: about 90 up close and 50 carried far away
MIN_DISK_GB = 2.0
GOOD_DISK_GB = 10.0  # a map flight saves about 200 MB of photos and COLMAP adds to that
MIN_COVERED = 0.9


@dataclass
class Line:
    status: str
    name: str
    text: str


@dataclass
class Section:
    title: str
    lines: list[Line] = field(default_factory=list)

    def add(self, status: str, name: str, text: str) -> None:
        self.lines.append(Line(status, name, text))

    @property
    def failed(self) -> bool:
        return any(line.status == FAIL for line in self.lines)


def scrub(text: object, secret: str | None) -> str:
    """Nothing that reaches the screen may hold the key."""
    text = str(text)
    return text.replace(secret, "<key>") if secret else text


def only_drones(path, wanted: str | None) -> str:
    """A temporary copy of the swarm file with only these drones, for flying one or two at a time."""
    if not wanted:
        return str(path)
    data = json.loads(Path(path).read_text())
    ids = [part.strip() for part in wanted.split(",") if part.strip()]
    known = [str(d["id"]) for d in data["drones"]]
    missing = [i for i in ids if i not in known]
    if missing or not ids:
        raise ValueError(f"--drones {wanted}: the swarm file lists {', '.join(known)}.")
    data["drones"] = [d for d in data["drones"] if str(d["id"]) in ids]
    copy = Path(tempfile.mkdtemp(prefix="zwarm-drones-")) / Path(path).name
    copy.write_text(json.dumps(data, indent=1))
    return str(copy)


def read_capabilities(backend: str) -> dict:
    with urllib.request.urlopen(f"{backend.rstrip('/')}/capabilities", timeout=2.0) as reply:
        return json.load(reply)


def camera_down_deg(seen, lens: Lens, fix) -> float | None:
    """How far below level this frame says the camera points. Only means something with the drone level."""
    world = np.concatenate([anchor.corners() for anchor, _ in seen])
    pixels = np.concatenate([quad for _, quad in seen])
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(world, pixels, lens.matrix, lens.distortion, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best, best_gap = None, math.inf
    for i in range(count):
        to_world = cv2.Rodrigues(rvecs[i])[0].T
        position = -to_world @ tvecs[i].ravel()
        gap = float(np.linalg.norm(position - np.array([fix.p[0], fix.p[1], fix.p[2] + CAMERA_HEIGHT_M])))
        if gap < best_gap:
            best, best_gap = to_world, gap
    return None if best is None else math.degrees(math.asin(max(-1.0, min(1.0, -float(best[2, 2])))))


class DroneWatch:
    """What one drone's video shows over a few seconds. Runs on its own thread, so a slow frame only skips frames."""

    def __init__(self, entry, optics, anchors, fence, frames, clock=time.monotonic):
        self.entry, self.anchors, self.fence, self.frames, self.clock = entry, anchors, fence, frames, clock
        self.lens = optics.lens or Lens.from_fov()
        self.tilt = math.radians(optics.tilt_deg)
        self.yaw = optics.yaw
        self.lock = threading.Lock()
        self.first = self.newest = None
        self.size = None
        self.read = 0
        self.broken = 0
        self.with_tags = 0
        self.sharp: deque = deque(maxlen=200)
        self.jpeg_bytes: deque = deque(maxlen=200)
        self.fixes: list = []
        self.recent: deque = deque(maxlen=RECENT_FIXES)
        self.tags: dict[str, int] = {}
        self.seeing: tuple[str, ...] = ()
        self.seeing_at = -math.inf
        self.furthest = 0.0
        self.down_deg: deque = deque(maxlen=400)
        self.best_jpeg = None
        self.best_sharp = -1.0
        self.unknown: set[int] = set()
        self.error = None

    def run(self, stopping: threading.Event) -> None:
        try:
            reader = AnchorReader(self.anchors)
        except Exception as error:
            self.error = f"the tag detector won't start: {error}"
            return
        last = None
        while not stopping.is_set():
            frame = self.frames.latest(self.entry.id)
            if frame is None or frame.number == last:
                stopping.wait(0.005)
                continue
            last = frame.number
            try:
                self._look(frame, reader)
            except Exception as error:  # one bad frame shouldn't end the check
                self.error = str(error)

    def _look(self, frame, reader) -> None:
        with self.lock:
            self.first = self.first or (frame.number, frame.t)
            self.newest = (frame.number, frame.t)
            self.jpeg_bytes.append(len(frame.jpeg))
        gray = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.broken += 1
            return
        height, width = gray.shape
        lens = self.lens.for_size(width, height)
        sharp = sharpness(cv2.resize(gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA))
        seen = reader.find(gray)
        fix = solve_fix(seen, lens, self.tilt, frame.t, self.yaw, self.fence) if seen else None
        down = camera_down_deg(seen, lens, fix) if fix is not None else None
        with self.lock:
            self.size = (width, height)
            self.read += 1
            self.sharp.append(sharp)
            self.unknown |= reader.unknown
            if sharp > self.best_sharp:
                self.best_sharp, self.best_jpeg = sharp, frame.jpeg
            if seen:
                self.with_tags += 1
                self.seeing = tuple(anchor.id for anchor, _ in seen)
                self.seeing_at = self.clock()
                for anchor, _ in seen:
                    self.tags[anchor.id] = self.tags.get(anchor.id, 0) + 1
            if fix is not None:
                self.fixes.append(fix)
                self.recent.append(fix)
                for anchor, _ in seen:
                    self.furthest = max(self.furthest, math.dist((anchor.x, anchor.y, 0.0), (fix.p[0], fix.p[1], fix.p[2] + CAMERA_HEIGHT_M)))
            if down is not None:
                self.down_deg.append(down)

    def fps(self) -> float:
        with self.lock:
            if self.first is None or self.newest[1] <= self.first[1]:
                return 0.0
            return (self.newest[0] - self.first[0]) / (self.newest[1] - self.first[1])

    def video_age(self) -> float:
        with self.lock:
            return math.inf if self.newest is None else self.clock() - self.newest[1]

    def seeing_now(self) -> tuple[str, ...]:
        with self.lock:
            return self.seeing if self.clock() - self.seeing_at < 1.0 else ()

    def pad_miss(self, fixes) -> float:
        x = statistics.median(f.p[0] for f in fixes)
        y = statistics.median(f.p[1] for f in fixes)
        return math.hypot(x - self.entry.pad[0], y - self.entry.pad[1])

    def live(self) -> str:
        fps = self.fps()
        if self.newest is None:
            return f"{self.entry.id}: no video"
        seeing = self.seeing_now()
        with self.lock:
            fixes = list(self.recent)
        where = ""
        if seeing and fixes:
            where = (f", at ({statistics.median(f.p[0] for f in fixes):.2f}, {statistics.median(f.p[1] for f in fixes):.2f}) "
                     f"{statistics.median(f.p[2] for f in fixes):.2f} m up, {self.pad_miss(fixes):.2f} m from pad, "
                     f"heading {math.degrees(statistics.median(f.yaw for f in fixes)):.0f}")
        return f"{self.entry.id}: {fps:.1f} fps, " + (" ".join(seeing) if seeing else "no tag") + where


# ----------------------------------------------------------------------------- the checks
def check_file(config, path, section: Section) -> None:
    """Geometry a takeoff would be refused over anyway, found while there's still time to move a pad."""
    room, limits = config.room, config.limits
    size = room.hi - room.lo
    edge = next(iter(config.anchors.values())).size
    section.add(PASS, "swarm file", f"{Path(path).name}: {len(config.drones)} drone(s), {len(config.anchors)} tags, white square "
                f"{edge * 1000:.1f} mm, fence {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m. Measure the white square: every height scales with it.")
    for d in config.drones:
        pad = np.array([d.pad[0], d.pad[1], 1.0])
        if np.any(pad[:2] < room.lo[:2]) or np.any(pad[:2] > room.hi[:2]):
            section.add(FAIL, "pad", f"drone {d.id}'s pad ({d.pad[0]}, {d.pad[1]}) is outside the fence.")
        elif room.clearance(pad) < limits.surface_clearance:
            section.add(FAIL, "pad", f"drone {d.id}'s pad ({d.pad[0]}, {d.pad[1]}) is too close to a wall or obstacle to climb.")
    for i, a in enumerate(config.drones):
        for b in config.drones[i + 1:]:
            gap = math.dist(a.pad[:2], b.pad[:2])
            if gap < limits.separation:
                section.add(FAIL, "pads", f"pads {a.id} and {b.id} are {gap:.2f} m apart and need {limits.separation:.2f} m.")
    d = config.drones[0]
    lens, tilt, yaw = Lens.from_fov(), math.radians(d.camera_tilt_deg), math.radians(d.yaw_deg)
    counts = [len(anchors_in_view(config.anchors, lens, tilt, (x, y, 1.0), yaw))
              for x in np.arange(room.lo[0], room.hi[0] + 1e-6, 0.2) for y in np.arange(room.lo[1], room.hi[1] + 1e-6, 0.2)]
    one, two = float(np.mean(np.array(counts) >= 1)), float(np.mean(np.array(counts) >= 2))
    section.add(PASS if one >= MIN_COVERED else WARN, "tag coverage", f"from 1 m up, facing the pad heading: 1 or more tags readable from {one * 100:.0f}% of the "
                f"fence, 2 or more from {two * 100:.0f}%, by geometry alone." + ("" if one >= MIN_COVERED else " Shrink the fence or add tags."))


def check_setup_lines(config, need_model: bool, section: Section) -> None:
    readiness = Readiness(config)
    check_setup(config, need_model, readiness)
    for text in readiness.notes:
        section.add(PASS, "setup", text)
    for text in readiness.warnings:
        section.add(WARN, "setup", text)
    for text in readiness.problems:
        section.add(FAIL, "setup", text)


def check_model(client, section: Section) -> None:
    secret = None
    try:
        if client is None:
            from agent.nemotron import NemotronClient

            client = NemotronClient()
        secret = client._api_key()
        started = time.monotonic()
        reply = client.complete([{"role": "user", "content": "Reply with the word OK."}], max_tokens=16, budget_s=25.0)
        section.add(PASS, "nemotron", f"{str(reply.model).split('/')[-1]} answered in {time.monotonic() - started:.1f} s.")
    except Exception as error:
        section.add(FAIL, "nemotron", f"{scrub(error, secret)} Check the Mac's internet, or pass --no-model to fly on fixed phrases only.")


def check_disk(section: Section, root: Path = DATA_ROOT) -> None:
    where = Path(root)
    while not where.exists():
        where = where.parent
    free = shutil.disk_usage(where).free / 1e9
    status = FAIL if free < MIN_DISK_GB else WARN if free < GOOD_DISK_GB else PASS
    section.add(status, "disk", f"{free:.0f} GB free where maps and findings go ({where}).")


def check_vision(watches: list[DroneWatch], section: Section) -> None:
    jpeg = next((w.best_jpeg for w in watches if w.best_jpeg), None)
    if jpeg is None:
        section.add(SKIP, "vision", "no frame to send.")
        return
    from agent.vision import analyze

    started = time.monotonic()
    try:
        answer = analyze([jpeg])
        section.add(PASS, "vision", f"{str(answer.get('model')).split('/')[-1]} answered in {time.monotonic() - started:.0f} s about a real frame: "
                    f"damage {answer.get('damage')}, confidence {answer.get('confidence')}.")
    except Exception as error:
        section.add(WARN, "vision", f"no answer in {time.monotonic() - started:.0f} s: {str(error)[:200]}")


def judge_backend(entry, info: dict | None, error, section: Section) -> None:
    if info is None:
        section.add(FAIL, "backend", f"{entry.backend} doesn't answer ({error}). Is the tunnel up? `tools/pi/fleet.sh status`.")
        for name in ("signal", "flight state"):
            section.add(SKIP, name, "no backend.")
        return
    section.add(PASS, "backend", f"{entry.backend} answers, drone type {info.get('drone_type')}.")
    if info.get("busy"):
        section.add(FAIL, "backend", "a trajectory or plugin is running on it.")
    if not info.get("commands", {}).get("takeoff"):
        section.add(FAIL, "backend", f"drone type {info.get('drone_type')} has no takeoff command.")
    signal = info.get("link_quality")
    if "link_quality" not in info:
        section.add(SKIP, "signal", "this Pi runs older code that doesn't report the byte. Harmless. To get it, see Update the Pis in docs/runbook.md.")
    elif signal is None:
        section.add(WARN, "signal", "no reply from the drone in the last 3 s. Is it on? `tools/pi/fleet.sh status`.")
    else:
        section.add(PASS if signal >= GOOD_SIGNAL else WARN, "signal", f"{signal} (about 90 up close, 50 far away)." + ("" if signal >= GOOD_SIGNAL else " Move the Pi closer."))
    airborne = info.get("airborne")
    if airborne is None:
        section.add(WARN, "flight state", "this backend doesn't track one.")
    elif airborne:
        section.add(FAIL, "flight state", "its backend presumes it is IN THE AIR, so a takeoff would be ignored. If it's on the floor, press "
                    "\"mark landed\" on http://localhost:5173/fleet.")
    else:
        section.add(PASS, "flight state", "presumed on the ground.")


def judge_video(watch: DroneWatch, section: Section) -> None:
    fps = watch.fps()
    if watch.newest is None:
        section.add(FAIL, "video", "no frame from /mjpeg. The camera can take 5 s to start. If it stays dark, power the drone off and on.")
        return
    if watch.video_age() > VIDEO_STOPPED_S:
        section.add(FAIL, "video", f"stopped: the last frame came {watch.video_age():.1f} s ago.")
    elif fps < MIN_FPS:
        section.add(FAIL, "video", f"{fps:.1f} fps, too slow to track from.")
    else:
        status = PASS if fps >= GOOD_FPS else WARN
        section.add(status, "video", f"{fps:.1f} fps (S20 measured 11 to 19), {statistics.median(watch.jpeg_bytes) / 1000:.0f} KB a frame.")
    if watch.size is not None:
        status = PASS if watch.size == (VIDEO_WIDTH, VIDEO_HEIGHT) else WARN
        section.add(status, "resolution", f"{watch.size[0]} x {watch.size[1]}" + ("." if status == PASS else f", expected {VIDEO_WIDTH} x {VIDEO_HEIGHT}. The lens model rescales to match."))
    if watch.sharp:
        sharp = statistics.median(watch.sharp)
        status = PASS if sharp >= MIN_SHARPNESS else WARN
        section.add(status, "sharpness", f"{sharp:.0f} (variance of Laplacian at half size; the inspector drops frames under {MIN_SHARPNESS:.0f})."
                    + ("" if status == PASS else " Clean the lens and add light."))


def judge_tags(watch: DroneWatch, on_pad: bool, section: Section) -> None:
    entry = watch.entry
    expected = anchors_in_view(watch.anchors, watch.lens, watch.tilt, (entry.pad[0], entry.pad[1], 1.0), watch.yaw)
    seen = sorted(watch.tags, key=lambda t: int(t.partition(":")[2]))
    if watch.unknown:
        section.add(WARN, "tags", f"also read ids not in the swarm file: {sorted(watch.unknown)}. Check the file, or cover those tags.")
    if not seen:
        if on_pad:
            section.add(SKIP, "tags", "not checked: it's on its pad, and a drone on the floor can't see the floor.")
            section.add(SKIP, "pose", "not checked: no tag.")
        else:
            section.add(FAIL, "tags", f"none read. Over its pad at 1 m it should see {', '.join(expected) or 'nothing'}. Was it lifted? "
                        "Use --on-pad to skip this. If it was lifted: check the light, the lens and the camera angle.")
            section.add(FAIL, "pose", "no pose without a tag.")
        return
    share = watch.with_tags / max(watch.read, 1)
    section.add(PASS, "tags", f"read {', '.join(seen)}; now seeing {' '.join(watch.seeing_now()) or 'none'}; in {share * 100:.0f}% of frames; "
                f"furthest read {watch.furthest:.2f} m.")
    fixes = watch.fixes
    if not fixes:
        section.add(FAIL, "pose", "tags read but no pose believed. Check white_square_edge_m, camera_tilt_deg, each tag's rotation_deg and position in the file.")
        return
    miss = watch.pad_miss(fixes)
    height = statistics.median(f.p[2] for f in fixes)
    fit = statistics.median(f.reprojection_px for f in fixes)
    x, y = statistics.median(f.p[0] for f in fixes), statistics.median(f.p[1] for f in fixes)
    if on_pad:
        section.add(PASS, "pose", f"({x:.2f}, {y:.2f}), {height:.2f} m up, {miss:.2f} m from its pad.")
    elif miss > PAD_MISS_M:
        section.add(FAIL, "pose", f"({x:.2f}, {y:.2f}), {height:.2f} m up, {miss:.2f} m from its pad ({entry.pad[0]}, {entry.pad[1]}). Limit {PAD_MISS_M} m. "
                    "Check by tape: the pad, the tag grid, or where you held it is wrong.")
    else:
        section.add(PASS, "pose", f"({x:.2f}, {y:.2f}), {height:.2f} m up, {miss:.2f} m from its pad. Check height against a tape: off by more than 10 cm means the tag size or lens is wrong.")
    heading = math.degrees((statistics.median(f.yaw for f in fixes) - watch.yaw + math.pi) % (2 * math.pi) - math.pi)
    if abs(heading) > HEADING_MISS_DEG:
        section.add(WARN, "heading", f"{heading:+.0f} degrees from yaw_deg in the file. Hold the nose along it, or a tag's rotation_deg is wrong.")
    if fit > BAD_FIT_PX:
        section.add(WARN, "fit", f"tag corners miss by {fit:.1f} px (rejected at 4). A tag is out of place, or fit the lens: `python -m swarm fit-focal`.")
    if watch.down_deg:
        down = statistics.median(watch.down_deg)
        gap = down - math.degrees(watch.tilt)
        section.add(PASS if abs(gap) <= TILT_MISS_DEG else WARN, "camera angle", f"{down:.0f} degrees below level if the drone was held level; the file says {math.degrees(watch.tilt):.0f}."
                    + ("" if abs(gap) <= TILT_MISS_DEG else " Reset the camera or fix camera_tilt_deg."))


# ----------------------------------------------------------------------------- running it
def _capabilities(entry, fetch) -> tuple[dict | None, object]:
    try:
        return fetch(entry.backend), None
    except Exception as error:
        return None, error


def _start_watches(config, frames) -> tuple[list[DroneWatch], threading.Event, list[threading.Thread]]:
    optics = config.optics()
    fence = (config.room.lo, config.room.hi)
    watches = [DroneWatch(d, optics[d.id], config.anchors, fence, frames) for d in config.drones]
    stopping = threading.Event()
    threads = [threading.Thread(target=w.run, args=(stopping,), daemon=True, name=f"preflight-{w.entry.id}") for w in watches]
    for thread in threads:
        thread.start()
    return watches, stopping, threads


def _stop(frames, stopping, threads) -> None:
    stopping.set()
    for thread in threads:
        thread.join(timeout=2.0)
    threading.Thread(target=frames.stop, daemon=True).start()  # a stuck read has a 5 s timeout, and no one needs to wait for it


def watch(path, out=print, make_frames=StreamFrames, seconds: float | None = None) -> int:
    """Print each drone's tags and pose until Ctrl-C. Safe with a drone in the air: it only reads video."""
    config, problem = load_swarm_file(path)
    if config is None:
        out(f"FAIL {problem}")
        return NOT_READY
    frames = make_frames(config.backends())
    frames.start()
    watches, stopping, threads = _start_watches(config, frames)
    out("Watching video only. No flight command is sent. Ctrl-C ends this and doesn't touch the drones.")
    end = None if seconds is None else time.monotonic() + seconds
    try:
        while end is None or time.monotonic() < end:
            time.sleep(0.5)
            out(" | ".join(w.live() for w in watches))
    except KeyboardInterrupt:
        pass
    finally:
        _stop(frames, stopping, threads)
    return 0


def run(path, need_model: bool = True, seconds: float | None = None, on_pad: bool = False, save_frames=None, vision: bool = False, out=print,
        make_frames=StreamFrames, fetch=read_capabilities, client=None, setup=check_setup_lines) -> int:
    out(f"Checking {path}. This reads capabilities and video only and sends no flight command.")
    config, problem = load_swarm_file(path)
    if config is None:
        out(f"  {FAIL}  swarm file    {problem}")
        out("RESULT: NOT READY.")
        return NOT_READY
    shared = Section("Shared")
    check_file(config, path, shared)
    setup(config, need_model, shared)
    if need_model:
        check_model(client, shared)
    else:
        shared.add(SKIP, "nemotron", "--no-model given: the planner won't run and only fixed phrases will work.")
    check_disk(shared)

    # Read once now to clear old replies the backend's socket has been holding; the second read is the fresh one.
    for d in config.drones:
        _capabilities(d, fetch)
    seconds = (ON_PAD_WATCH_S if on_pad else WATCH_S) if seconds is None else seconds
    frames = make_frames(config.backends())
    frames.start()
    watches, stopping, threads = _start_watches(config, frames)
    if on_pad:
        out(f"\nWatching video for {seconds:.0f} s with the drones on their pads.")
    else:
        out(f"\nMotors off. Lift each drone in turn to about 1 m over its pad, nose along its heading. Watching for up to {seconds:.0f} s.")
    started = time.monotonic()
    next_print = started + PRINT_EVERY_S
    try:
        while True:
            now = time.monotonic()
            if now - started >= seconds:
                break
            if now - started >= ENOUGH_S:
                if on_pad or all(len(w.fixes) >= ENOUGH_FIXES for w in watches):
                    break
            if now >= next_print:
                out(f"  {now - started:>3.0f} s  " + " | ".join(w.live() for w in watches))
                next_print = now + PRINT_EVERY_S
            time.sleep(0.1)
    finally:
        _stop(frames, stopping, threads)

    sections = [shared]
    if save_frames:
        folder = Path(save_frames)
        folder.mkdir(parents=True, exist_ok=True)
    for w in watches:
        section = Section(f"Drone {w.entry.id}")
        info, error = _capabilities(w.entry, fetch)
        judge_backend(w.entry, info, error, section)
        judge_video(w, section)
        if w.newest is not None:
            judge_tags(w, on_pad, section)
        if w.error:
            section.add(WARN, "reader", w.error)
        if save_frames and w.best_jpeg:
            saved = folder / f"drone-{w.entry.id}.jpg"
            saved.write_bytes(w.best_jpeg)
            section.add(PASS, "photo", f"sharpest frame saved to {saved}.")
        sections.append(section)
    if vision:
        check_vision(watches, shared)

    for section in sections:
        verdict = "FAIL" if section.failed else "PASS"
        out(f"\n{section.title}: {verdict}")
        for line in section.lines:
            out(f"  {line.status}  {line.name:<13} {line.text}")
    ready = not any(section.failed for section in sections)
    out("\nRESULT: READY. Nothing has flown. Next is the staged flight test in docs/runbook.md." if ready else "\nRESULT: NOT READY. Fix every FAIL above, then run this again.")
    return 0 if ready else NOT_READY
