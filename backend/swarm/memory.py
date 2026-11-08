"""What the fleet has seen, shared by every drone and kept in SQLite between flights.

Each camera report is a sighting. Sightings of the same kind of thing in about the same place merge
into one Thing, whose position is a weighted average that trusts close looks most. One sighting
proves little: cameras report things that aren't there, so a Thing only counts once it has been seen
a few times, and it is forgotten when drones look straight at it and it is gone.

Positions are in the room's coordinates, so a file is only meaningful for the room that wrote it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MOVERS = {"person", "dog", "cat"}  # these never become map obstacles and their position goes stale fast
KNOWN_SIGHTINGS = 3  # fewer than this could be a false detection
SURE_CONFIDENCE = 0.6  # the bar for telling someone "found it", above the bar for steering around it
MERGE_GATE = 0.7  # metres between sightings of the same still thing, at least
MOVER_GATE = 1.5
WEIGHT_CAP = 40.0  # keeps old looks from outvoting new ones forever
MOVER_BLEND = 0.4
MAX_MOVER_SPEED = 1.5
FORGET_AFTER_MISSES = 15  # looks straight at a known thing that found nothing
FORGET_UNPROVEN_AFTER = 3
KEEP_SIGHTINGS = 100_000  # the sightings table is history; the oldest rows are dropped past this

SCHEMA = """
CREATE TABLE IF NOT EXISTS things (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    sx REAL NOT NULL, sy REAL NOT NULL, sz REAL NOT NULL,
    vx REAL NOT NULL DEFAULT 0, vy REAL NOT NULL DEFAULT 0,
    weight REAL NOT NULL,
    sightings INTEGER NOT NULL,
    confidence REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    seen_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY,
    t REAL NOT NULL,
    drone TEXT NOT NULL,
    thing INTEGER NOT NULL,
    label TEXT NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY,
    t REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


@dataclass
class Thing:
    id: int
    label: str
    p: np.ndarray
    size: np.ndarray
    v: np.ndarray
    weight: float
    sightings: int
    confidence: float
    first_seen: float
    last_seen: float
    seen_by: set[str] = field(default_factory=set)
    misses: int = 0

    @property
    def moves(self) -> bool:
        return self.label in MOVERS

    @property
    def known(self) -> bool:
        return self.sightings >= KNOWN_SIGHTINGS

    @property
    def sure(self) -> bool:
        return self.known and self.confidence >= SURE_CONFIDENCE


class Memory:
    def __init__(self, path: str | Path = ":memory:"):
        # The flight loop and whatever takes commands may be different threads; the hub serializes them.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.things: dict[int, Thing] = {}
        self._dirty: set[int] = set()
        self._since_prune = 0
        for row in self.db.execute("SELECT id, label, x, y, z, sx, sy, sz, vx, vy, weight, sightings, confidence, "
                                   "first_seen, last_seen, seen_by FROM things"):
            self.things[row[0]] = Thing(row[0], row[1], np.array(row[2:5]), np.array(row[5:8]),
                                        np.array([row[8], row[9], 0.0]), row[10], row[11], row[12], row[13], row[14],
                                        set(json.loads(row[15])))

    # ------------------------------------------------------------------ writing
    def record(self, label: str, p, size, confidence: float, range_m: float, drone: str, t: float) -> Thing:
        """Add one sighting and return the Thing it belongs to."""
        p, size = np.asarray(p, float), np.asarray(size, float)
        thing = self._match(label, p)
        if thing is None:
            cursor = self.db.execute(
                "INSERT INTO things (label, x, y, z, sx, sy, sz, weight, sightings, confidence, first_seen, last_seen, seen_by) "
                "VALUES (?,?,?,?,?,?,?,0,0,0,?,?,'[]')", (label, *p.tolist(), *size.tolist(), t, t))
            thing = Thing(cursor.lastrowid, label, p.copy(), size.copy(), np.zeros(3), 0.0, 0, 0.0, t, t)
            self.things[thing.id] = thing
        weight = confidence / (range_m * range_m + 0.25)
        if thing.moves:
            self._track_mover(thing, p, t)
        else:
            total = thing.weight + weight
            thing.p += (p - thing.p) * (weight / total)
            thing.size += (size - thing.size) * (weight / total)
            thing.weight = min(total, WEIGHT_CAP)
        thing.confidence += (confidence - thing.confidence) / (thing.sightings + 1)
        thing.sightings += 1
        thing.last_seen = t
        thing.misses = 0
        thing.seen_by.add(drone)
        self._dirty.add(thing.id)
        self.db.execute("INSERT INTO sightings (t, drone, thing, label, x, y, z, confidence) VALUES (?,?,?,?,?,?,?,?)",
                        (t, drone, thing.id, label, *p.tolist(), confidence))
        self._since_prune += 1
        return thing

    def _match(self, label: str, p: np.ndarray) -> Thing | None:
        best, best_distance = None, math.inf
        for thing in self.things.values():
            if thing.label != label:
                continue
            gate = MOVER_GATE if thing.moves else max(MERGE_GATE, 0.75 * float(np.max(thing.size)))
            distance = float(np.linalg.norm(thing.p[:2] - p[:2]))
            if distance < gate and distance < best_distance:
                best, best_distance = thing, distance
        return best

    @staticmethod
    def _track_mover(thing: Thing, p: np.ndarray, t: float) -> None:
        dt = t - thing.last_seen
        if thing.sightings and dt > 1e-3:
            seen_v = (p - thing.p) / dt
            seen_v[2] = 0.0
            speed = float(np.linalg.norm(seen_v))
            if speed > MAX_MOVER_SPEED:
                seen_v *= MAX_MOVER_SPEED / speed
            thing.v += (seen_v - thing.v) * MOVER_BLEND
        thing.p = thing.p + (p - thing.p) * (MOVER_BLEND if thing.sightings else 1.0)
        thing.weight = 1.0

    def missed(self, thing_id: int) -> bool:
        """A drone looked right at it and saw nothing. True if that was enough to forget it."""
        thing = self.things.get(thing_id)
        if thing is None:
            return False
        thing.misses += 1
        if thing.misses >= (FORGET_AFTER_MISSES if thing.known else FORGET_UNPROVEN_AFTER):
            self.forget(thing_id)
            return True
        return False

    def forget(self, thing_id: int) -> bool:
        if self.things.pop(thing_id, None) is None:
            return False
        self._dirty.discard(thing_id)
        self.db.execute("DELETE FROM things WHERE id = ?", (thing_id,))
        return True

    def forget_label(self, label: str | None) -> int:
        """Forget every thing of this label, or everything when label is None."""
        gone = [t.id for t in self.things.values() if label is None or t.label == label]
        for thing_id in gone:
            self.forget(thing_id)
        return len(gone)

    def note(self, t: float, kind: str, text: str) -> None:
        self.db.execute("INSERT INTO log (t, kind, text) VALUES (?,?,?)", (t, kind, text))

    def commit(self) -> None:
        for thing_id in self._dirty:
            thing = self.things[thing_id]
            self.db.execute(
                "UPDATE things SET x=?, y=?, z=?, sx=?, sy=?, sz=?, vx=?, vy=?, weight=?, sightings=?, confidence=?, "
                "last_seen=?, seen_by=? WHERE id=?",
                (*thing.p.tolist(), *thing.size.tolist(), float(thing.v[0]), float(thing.v[1]), thing.weight,
                 thing.sightings, thing.confidence, thing.last_seen, json.dumps(sorted(thing.seen_by)), thing_id))
        self._dirty.clear()
        if self._since_prune > 5000:
            self._since_prune = 0
            self.db.execute("DELETE FROM sightings WHERE id <= (SELECT MAX(id) FROM sightings) - ?", (KEEP_SIGHTINGS,))
        self.db.commit()

    def close(self) -> None:
        self.commit()
        self.db.close()

    def __del__(self):
        self.db.close()

    # ------------------------------------------------------------------ reading
    def find(self, label: str) -> list[Thing]:
        """Things of this label that count, most recently seen first."""
        return sorted((t for t in self.things.values() if t.label == label and t.known),
                      key=lambda t: t.last_seen, reverse=True)

    def known(self) -> list[Thing]:
        return sorted((t for t in self.things.values() if t.known), key=lambda t: t.label)

    def obstacles(self) -> list[Thing]:
        """Still things drones must plan around. A doubtful label still gets avoided."""
        return [t for t in self.things.values() if t.known and not t.moves]

    def movers(self, now: float, within: float = 3.0) -> list[Thing]:
        return [t for t in self.things.values() if t.moves and t.sightings >= 2 and now - t.last_seen <= within]

    def log(self) -> list[tuple[float, str, str]]:
        return list(self.db.execute("SELECT t, kind, text FROM log ORDER BY id"))
