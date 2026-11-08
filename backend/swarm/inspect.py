"""Looks for damage during an inspect mission and keeps what it finds for a person to review.

Three threads' worth of work, so the flight loop only ever does the first:
  tick()    on the flight loop: notes each drone's pose and hands over its newest frame. No waits.
  watcher   decodes frames, keeps the sharpest of each second, skips views it already has.
  workers   send photos to the vision model and store what comes back. They are the slow part: about
            3 to 14 s an image. The queue is short, and when it is full the oldest photo is dropped.

A finding is always a possible issue. It is stored as pending whatever the model's confidence, and
only set_status, which only a person's request reaches, can change that. The model's summary is
data, so it never goes into an event's text, which an agent may read. It stays in the event's data.

Findings and photos live under ~/Developer/datasets/zwarm/, outside the repo.
Note this module is named like the standard library's inspect: run the package with `python -m swarm`
from backend/, never `python swarm/something.py`, which would put this file ahead of the real one.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from agent.vision import MAX_SUMMARY, SEVERITIES, VisionError, plain_text, prompt_for
from agent.vision import analyze as ask_vision
from swarm.coordinator import Phase
from swarm.eyes import DEFAULT_TILT

logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "Developer" / "datasets" / "zwarm"
STATUSES = ("pending", "confirmed", "dismissed")
LOW_CONFIDENCE = 0.5  # below this a finding is announced as a long shot. It is stored all the same

LOOK_EVERY_S = 0.2  # seconds between frames judged per drone; the camera sends 11 to 19 a second
WINDOW_S = 1.0  # the sharpest frame of each window of this long is the one considered
VIDEO_LATE_S = 0.15  # measured: video arrives 100 to 200 ms after the picture was taken
POSE_HISTORY = 60  # ticks of pose kept per drone, about 3 s

MIN_SHARPNESS = 5.0  # variance of Laplacian on a half-size frame. Below this it's a smear or a blank. A guess: real video is unmeasured
BLUR_SHARE = 0.4  # a frame this far below the drone's recent median is motion blur. Simulated frames lose 70% to a 9 pixel smear
RECENT_FRAMES = 15  # frames behind that median, about 3 s: short, since a plain floor is far less sharp than a patterned one
SAME_SPOT_M = 0.35  # a view from this close adds nothing new...
SAME_HEADING = math.radians(25.0)  # ...if it also faces this nearly the same way...
SAME_HEIGHT_M = 0.3  # ...from about the same height...
MUCH_SHARPER = 2.0  # ...unless it is this many times sharper than the view already sent
SAME_PICTURE = 5.0  # mean difference of thumbnails, out of 255, when there's no pose to compare
THUMB = (32, 18)
RECENT_PICTURES = 8
UNTRUSTED_SIGMA_M = 0.5  # a pose worse than this isn't used to compare views or merge findings
UNTRUSTED_FIX_AGE_S = 2.0
SAME_DAMAGE_M = 0.75  # looks whose picture centers land this close count as the same damage

QUEUE_SIZE = 6  # at 3 to 14 s an image and two workers, this is 10 to 40 s behind at worst
WORKERS = 2
MAX_PHOTOS = 200  # at most this many photos per mission: a limit on API use
FAULT_EVERY_S = 60.0  # at most one "vision is failing" event this often

SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    t REAL NOT NULL,
    drone TEXT NOT NULL,
    x REAL, y REAL, z REAL, yaw REAL,
    summary TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    target TEXT,
    photo TEXT,
    look_x REAL, look_y REAL,
    sigma REAL,
    views INTEGER NOT NULL DEFAULT 1
);
"""
COLUMNS = ("id", "t", "drone", "x", "y", "z", "yaw", "summary", "severity", "confidence", "status", "target", "photo",
           "look_x", "look_y", "sigma", "views")
SELECT = f"SELECT {', '.join(COLUMNS)} FROM findings"


def _finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _clean_confidence(value) -> float:
    number = _finite(value)
    return 0.0 if number is None else min(max(number, 0.0), 1.0)


def _clean_severity(value) -> str:
    return value if value in SEVERITIES and value != "none" else "low"


class FindingStore:
    """Findings in SQLite with one JPEG file each. Safe from any thread. Does disk waits, so never call it on the flight loop."""

    def __init__(self, path: str | Path = DATA_ROOT / "findings.sqlite"):
        path = Path(path).expanduser()
        if path.is_dir() or not path.suffix:
            path = path / "findings.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.photos = path.parent / "photos"
        self.photos.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def _save_photo(self, jpeg: bytes) -> str | None:
        if not jpeg:
            return None
        name = f"{uuid.uuid4().hex}.jpg"
        (self.photos / name).write_bytes(jpeg)
        return name

    def add(self, drone: str, x=None, y=None, z=None, yaw=None, summary: str = "", severity: str = "low", confidence: float = 0.0,
            jpeg: bytes = b"", target: str | None = None, t: float | None = None, look: tuple | None = None, sigma=None) -> dict:
        """Store a possible issue. Every new finding is pending: there is no way to add a confirmed one."""
        photo = self._save_photo(jpeg)
        look_x, look_y = look if look is not None else (None, None)
        with self.lock:
            cursor = self.db.execute(
                "INSERT INTO findings (t, drone, x, y, z, yaw, summary, severity, confidence, status, target, photo, look_x, look_y, sigma) "
                "VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?)",
                (time.time() if t is None else float(t), str(drone), _finite(x), _finite(y), _finite(z), _finite(yaw),
                 plain_text(summary)[:MAX_SUMMARY], _clean_severity(severity), _clean_confidence(confidence),
                 plain_text(target)[:80] if target else None, photo, _finite(look_x), _finite(look_y), _finite(sigma)))
            self.db.commit()
            return self._row(cursor.lastrowid)

    def add_view(self, finding_id: int, drone: str, x=None, y=None, z=None, yaw=None, summary: str = "", severity: str = "low",
                 confidence: float = 0.0, jpeg: bytes = b"", look: tuple | None = None, sigma=None) -> dict:
        """Another look at a finding already stored. Never changes its status.

        A pending finding takes the words and photo of its most confident look. One a person has
        already judged only counts the extra look: their decision stands.
        """
        confidence = _clean_confidence(confidence)
        with self.lock:
            row = self._row(finding_id)
            if row is None:
                raise KeyError(finding_id)
            better = row["status"] == "pending" and confidence > row["confidence"]
            photo = self._save_photo(jpeg) if better else None
            if better:
                look_x, look_y = look if look is not None else (None, None)
                self.db.execute(
                    "UPDATE findings SET drone=?, x=?, y=?, z=?, yaw=?, summary=?, severity=?, confidence=?, photo=?, look_x=?, look_y=?, sigma=?, "
                    "views=views+1 WHERE id=?",
                    (str(drone), _finite(x), _finite(y), _finite(z), _finite(yaw), plain_text(summary)[:MAX_SUMMARY], _clean_severity(severity),
                     confidence, photo or row["photo"], _finite(look_x), _finite(look_y), _finite(sigma), finding_id))
            else:
                self.db.execute("UPDATE findings SET views=views+1 WHERE id=?", (finding_id,))
            self.db.commit()
            updated = self._row(finding_id)
        if better and photo and row["photo"]:
            (self.photos / row["photo"]).unlink(missing_ok=True)
        return updated

    def _row(self, finding_id: int) -> dict | None:
        row = self.db.execute(f"{SELECT} WHERE id=?", (int(finding_id),)).fetchone()
        return dict(zip(COLUMNS, row)) if row else None

    def get(self, finding_id: int) -> dict | None:
        with self.lock:
            return self._row(finding_id)

    def list(self, status: str | None = None) -> list[dict]:
        """Every finding, oldest first."""
        query, values = f"{SELECT} ORDER BY id", ()
        if status is not None:
            query, values = f"{SELECT} WHERE status=? ORDER BY id", (status,)
        with self.lock:
            return [dict(zip(COLUMNS, row)) for row in self.db.execute(query, values).fetchall()]

    def set_status(self, finding_id: int, status: str) -> dict:
        """A person's decision on a finding. Raises ValueError on a bad status and KeyError on an unknown id."""
        if status not in STATUSES:
            raise ValueError(f"A finding's status must be one of {', '.join(STATUSES)}, not {status!r}.")
        with self.lock:
            if self.db.execute("UPDATE findings SET status=? WHERE id=?", (status, int(finding_id))).rowcount == 0:
                raise KeyError(finding_id)
            self.db.commit()
            return self._row(finding_id)

    def photo(self, finding_id: int) -> bytes:
        """The finding's JPEG. Raises KeyError if the finding or its photo isn't there."""
        row = self.get(finding_id)
        if row is None or not row["photo"]:
            raise KeyError(finding_id)
        try:
            return (self.photos / Path(row["photo"]).name).read_bytes()
        except OSError:
            raise KeyError(finding_id) from None

    def close(self) -> None:
        with self.lock:
            self.db.close()


# ------------------------------------------------------------------ judging frames
def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: edges are crisp when it's high and smeared when it's low."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def decode_small(jpeg: bytes) -> np.ndarray | None:
    """A half-size grayscale image, or None for a broken JPEG. Half size is faster and hides sensor noise."""
    try:
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    except cv2.error:
        return None


def look_point(p, yaw: float, tilt: float) -> tuple[float, float] | None:
    """Where the middle of the picture meets the floor, or None if it doesn't."""
    if p is None or p[2] <= 0.0 or tilt <= 0.05:
        return None
    ahead = p[2] / math.tan(tilt)
    return (p[0] + ahead * math.cos(yaw), p[1] + ahead * math.sin(yaw))


@dataclass
class Look:
    """One frame the watcher is considering, with where the drone was when it was taken."""

    drone: str
    now: float
    jpeg: bytes
    p: tuple[float, float, float] | None
    yaw: float
    sigma: float | None
    trusted: bool  # whether the pose is good enough to compare views by
    sharp: float = 0.0
    thumb: np.ndarray | None = None
    target: str | None = None
    tilt: float = DEFAULT_TILT


@dataclass
class _Window:
    start: float
    best: Look


@dataclass
class _Drone:
    poses: deque = field(default_factory=lambda: deque(maxlen=POSE_HISTORY))
    number: int = -1
    next_look: float = -math.inf
    window: _Window | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_FRAMES))
    pictures: deque = field(default_factory=lambda: deque(maxlen=RECENT_PICTURES))


class Inspector:
    """Watches every flying drone's video between start() and stop().

    frames only needs latest(drone). hub only needs co.drones, and optionally tilt_of(drone); pass
    poses to tick() instead to go without one. analyze(jpegs, prompt) -> {damage, summary, severity,
    confidence} defaults to the live model. on_event(kind, text, data) gets "finding" and "fault".
    """

    def __init__(self, store: FindingStore, frames, hub=None, analyze=None, on_event=None, *, workers: int = WORKERS,
                 queue_size: int = QUEUE_SIZE, max_photos: int = MAX_PHOTOS, min_sharpness: float = MIN_SHARPNESS,
                 window_s: float = WINDOW_S, look_every_s: float = LOOK_EVERY_S, tilt: float = DEFAULT_TILT):
        self.store, self.frames, self.hub = store, frames, hub
        self.analyze = analyze or ask_vision
        self.on_event = on_event
        self.workers, self.queue_size, self.max_photos = workers, queue_size, max_photos
        self.min_sharpness, self.window_s, self.look_every_s, self.tilt = min_sharpness, window_s, look_every_s, tilt

        self.cv = threading.Condition()  # guards everything below; held only for short list work, never for I/O
        self.watching = False
        self.closing = False
        self.target: str | None = None
        self.drones: dict[str, _Drone] = {}
        self.waiting: dict[str, Look] = {}  # the newest frame per drone for the watcher to judge
        self.queue: deque[Look] = deque()
        self.views: list[Look] = []  # what's been sent or is about to be
        self.spots: list[tuple[int, tuple[float, float]]] = []  # this mission's findings, for merging repeat sightings
        self.busy = 0
        self.counts = self._zero()
        self.last_error: str | None = None
        self.next_fault = -math.inf
        self.threads: list[threading.Thread] = []

    @staticmethod
    def _zero() -> dict[str, int]:
        return {"frames": 0, "broken": 0, "blurry": 0, "repeats": 0, "sent": 0, "dropped": 0, "checked": 0, "clear": 0,
                "unreadable": 0, "failed": 0, "findings": 0, "merged": 0, "tick_errors": 0}

    # ------------------------------------------------------------------ mission control
    def start(self, target: str | None = None) -> None:
        """Begin watching. A new mission can look at everything again. Findings already stored stay."""
        with self.cv:
            self.target = target
            self.watching = True
            self.drones = {}
            self.waiting = {}
            self.views = []
            self.spots = []
            self.counts = self._zero()
            if not self.threads:
                self.threads = [threading.Thread(target=self._watch, daemon=True, name="inspect-watch")]
                self.threads += [threading.Thread(target=self._work, daemon=True, name=f"inspect-vision-{i}") for i in range(self.workers)]
                for thread in self.threads:
                    thread.start()
            self.cv.notify_all()

    def stop(self) -> None:
        """Stop watching. Photos already queued still get checked, so nothing seen is lost."""
        with self.cv:
            self.watching = False
            self.waiting = {}

    def close(self, timeout: float = 2.0) -> None:
        """Stop for good. A vision call in progress is left to finish on its daemon thread."""
        with self.cv:
            self.watching = False
            self.closing = True
            self.queue.clear()
            self.cv.notify_all()
        for thread in self.threads:
            thread.join(timeout)
        self.threads = []

    def status(self) -> dict:
        with self.cv:
            return {"watching": self.watching, "target": self.target, "queued": len(self.queue), "busy": self.busy,
                    "last_error": self.last_error, **self.counts}

    # ------------------------------------------------------------------ the flight loop
    def tick(self, now: float, poses: dict | None = None) -> None:
        """Note poses and hand over new frames. Called on the flight loop: it does no I/O and never raises."""
        if not self.watching:
            return
        try:
            for drone_id, pose in (self._flying() if poses is None else poses.items()):
                self._note(drone_id, pose, now)
        except Exception:
            self.counts["tick_errors"] += 1
            if self.counts["tick_errors"] % 100 == 1:  # logging is a disk wait, so not on every flight tick
                logger.exception("[inspect] tick failed; flight is unaffected")

    def _flying(self) -> list[tuple[str, object]]:
        return [(drone_id, d.tracked) for drone_id, d in self.hub.co.drones.items() if d.phase == Phase.AIR] if self.hub else []

    def _note(self, drone_id: str, pose, now: float) -> None:
        with self.cv:
            state = self.drones.setdefault(drone_id, _Drone())
            state.poses.append((now, pose.p, pose.yaw, getattr(pose, "sigma", None), getattr(pose, "fix_age", 0.0)))
            if now < state.next_look:
                return
            frame = self.frames.latest(drone_id)
            if frame is None or frame.number == state.number:
                return
            state.number = frame.number
            state.next_look = now + self.look_every_s
            self.waiting[drone_id] = self._look(drone_id, frame.jpeg, now, state)
            self.cv.notify_all()

    def _look(self, drone_id: str, jpeg: bytes, now: float, state: _Drone) -> Look:
        then = now - VIDEO_LATE_S
        _, p, yaw, sigma, fix_age = min(state.poses, key=lambda entry: abs(entry[0] - then))
        sigma = _finite(sigma)
        where = tuple(float(v) for v in p)
        known = all(math.isfinite(v) for v in where) and math.isfinite(yaw)
        trusted = known and (sigma is None or sigma <= UNTRUSTED_SIGMA_M) and (fix_age or 0.0) <= UNTRUSTED_FIX_AGE_S
        return Look(drone_id, now, jpeg, where if known else None, float(yaw) if known else 0.0, sigma, trusted,
                    target=self.target, tilt=self._tilt(drone_id))

    def _tilt(self, drone_id: str) -> float:
        tilt_of = getattr(self.hub, "tilt_of", None)
        return tilt_of(drone_id) if tilt_of else self.tilt

    # ------------------------------------------------------------------ the watcher thread
    def _watch(self) -> None:
        while True:
            with self.cv:
                while not self.waiting and not self.closing:
                    self.cv.wait(0.5)
                if self.closing:
                    return
                batch, self.waiting = self.waiting, {}
            for look in batch.values():
                try:
                    self._judge(look)
                except Exception:
                    logger.exception("[inspect] judging a frame failed")

    def _judge(self, look: Look) -> None:
        """Keep the sharpest frame of each window. This is the thread's slow step, done outside the lock."""
        gray = decode_small(look.jpeg)
        if gray is None or gray.size == 0:
            with self.cv:
                self.counts["broken"] += 1
            return
        look.sharp = sharpness(gray)
        look.thumb = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
        with self.cv:
            self.counts["frames"] += 1
            state = self.drones.setdefault(look.drone, _Drone())
            state.recent.append(look.sharp)
            window = state.window
            if window is None or look.now < window.start:
                state.window = _Window(look.now, look)
            else:
                if look.sharp > window.best.sharp:
                    window.best = look
                if look.now - window.start >= self.window_s:
                    state.window = None
                    if self.watching:
                        self._consider(window.best, state)

    def _consider(self, look: Look, state: _Drone) -> None:
        """Decide whether this frame is worth a vision call. Holds the lock, so it stays quick."""
        if look.sharp < self.min_sharpness or look.sharp < BLUR_SHARE * statistics.median(state.recent):
            self.counts["blurry"] += 1
            return
        if self._adds_nothing(look, state):
            self.counts["repeats"] += 1
            return
        if self.counts["sent"] >= self.max_photos:
            return
        self.views.append(look)
        state.pictures.append(look.thumb)
        if len(self.queue) >= self.queue_size:
            oldest = self.queue.popleft()
            self.counts["dropped"] += 1
            self._forget_view(oldest)  # it was never checked, so a later pass may send that view again
        self.queue.append(look)
        self.counts["sent"] += 1
        self.cv.notify_all()

    def _adds_nothing(self, look: Look, state: _Drone) -> bool:
        if not look.trusted:
            return any(float(np.abs(look.thumb - old).mean()) < SAME_PICTURE for old in state.pictures)
        for old in self.views:
            if not old.trusted:
                continue
            turned = abs((look.yaw - old.yaw + math.pi) % (2 * math.pi) - math.pi)
            if (turned < SAME_HEADING and math.hypot(look.p[0] - old.p[0], look.p[1] - old.p[1]) < SAME_SPOT_M
                    and abs(look.p[2] - old.p[2]) < SAME_HEIGHT_M):
                if look.sharp >= MUCH_SHARPER * old.sharp:
                    old.sharp = look.sharp  # send this better one, and a third has to beat it to go too
                    continue
                return True
        return False

    def _forget_view(self, look: Look) -> None:
        self.views = [view for view in self.views if view is not look]

    # ------------------------------------------------------------------ the vision workers
    def _work(self) -> None:
        while True:
            with self.cv:
                while not self.queue and not self.closing:
                    self.cv.wait(0.5)
                if self.closing:
                    return
                look = self.queue.popleft()
                self.busy += 1
            try:
                self._check(look)
            except Exception:
                logger.exception("[inspect] checking a photo failed")
            finally:
                with self.cv:
                    self.busy -= 1

    def _check(self, look: Look) -> None:
        """Slow: the model call and the disk both happen here, on a worker thread."""
        try:
            verdict = self.analyze([look.jpeg], prompt_for(look.target))
            damage = verdict["damage"]
        except Exception as error:
            self._failed(look, error)
            return
        with self.cv:
            self.counts["checked"] += 1
        if not isinstance(damage, bool):
            with self.cv:
                self.counts["unreadable"] += 1
            return
        if not damage:
            with self.cv:
                self.counts["clear"] += 1
            return
        self._record(look, verdict)

    def _failed(self, look: Look, error: Exception) -> None:
        # A VisionError message is written to be safe to show. Anything else might quote who knows what.
        text = str(error) if isinstance(error, VisionError) else f"Vision failed with {type(error).__name__}."
        logger.warning("[inspect] %s", text)
        now = time.monotonic()
        with self.cv:
            self.counts["failed"] += 1
            self.last_error = text
            self._forget_view(look)  # never checked, so it may be tried again from there
            tell = now >= self.next_fault
            if tell:
                self.next_fault = now + FAULT_EVERY_S
        if tell:
            self._emit("fault", f"A photo couldn't be checked for damage. {text}", {"drone": look.drone})

    def _record(self, look: Look, verdict: dict) -> None:
        look_at = look_point(look.p, look.yaw, look.tilt) if look.trusted else None
        fields = dict(x=look.p[0] if look.p else None, y=look.p[1] if look.p else None, z=look.p[2] if look.p else None,
                      yaw=look.yaw if look.p else None, summary=verdict.get("summary", ""), severity=verdict.get("severity", "low"),
                      confidence=verdict.get("confidence", 0.0), jpeg=look.jpeg, look=look_at, sigma=look.sigma)
        same = self._same_damage(look_at)
        if same is not None:
            self.store.add_view(same, look.drone, **fields)
            with self.cv:
                self.counts["merged"] += 1
            return
        finding = self.store.add(look.drone, target=look.target, **fields)
        with self.cv:
            self.counts["findings"] += 1
            if look_at is not None:
                self.spots.append((finding["id"], look_at))
        self._announce(finding)

    def _same_damage(self, look_at: tuple[float, float] | None) -> int | None:
        if look_at is None:
            return None
        with self.cv:
            spots = list(self.spots)
        best, best_distance = None, SAME_DAMAGE_M
        for finding_id, spot in spots:
            distance = math.hypot(look_at[0] - spot[0], look_at[1] - spot[1])
            if distance < best_distance:
                best, best_distance = finding_id, distance
        return best

    def _announce(self, finding: dict) -> None:
        where = "" if finding["look_x"] is None else f" near ({finding['look_x']:.1f}, {finding['look_y']:.1f})"
        doubt = "A long shot: possible" if finding["confidence"] < LOW_CONFIDENCE else "Possible"
        self._emit("finding", f"{doubt} damage{where}, seen by {finding['drone']}. {finding['severity'].capitalize()} severity, "
                              f"{finding['confidence']:.0%} confidence. Finding {finding['id']} is waiting for a person to review it.",
                   {key: finding[key] for key in ("id", "drone", "x", "y", "z", "yaw", "summary", "severity", "confidence", "status", "target")})

    def _emit(self, kind: str, text: str, data: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, text, data)
        except Exception:
            logger.exception("[inspect] an event handler failed")
