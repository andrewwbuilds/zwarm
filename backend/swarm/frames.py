"""The newest camera frame from each drone: one reader per drone, shared by everything that looks.

The tracker, the detector, inspections and map capture all want the same video. Opening /mjpeg once
for each would multiply the load on a Pi, so one thread per drone keeps only the newest frame and
everyone reads that. A stale frame is never queued: a slow reader just sees the latest one.

latest() never blocks and never raises, so it is safe to call from the flight loop's thread.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

SOI, EOI = b"\xff\xd8", b"\xff\xd9"
RETRY_S = 1.0
TIMEOUT_S = 5.0
MAX_BUFFER = 4 * 1024 * 1024  # a broken stream with no end marker must not grow without limit


@dataclass(frozen=True)
class Frame:
    drone: str
    t: float  # the clock's time when this frame arrived
    jpeg: bytes
    number: int  # goes up by one per frame from this drone, so a reader can tell a frame is new


class Frames(Protocol):
    def latest(self, drone_id: str) -> Frame | None: ...


def jpeg_chunks(stream, chunk: int = 16384):
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
                if len(buffer) > MAX_BUFFER:
                    buffer = b""
                break
            yield buffer[start:end + 2]
            buffer = buffer[end + 2:]


class StreamFrames:
    """Reads each real drone's /mjpeg in its own thread, reconnecting forever until stopped."""

    def __init__(self, backends: dict[str, str], clock: Callable[[], float] = time.monotonic, opener=None):
        self.backends = dict(backends)
        self.clock = clock
        self.opener = opener or (lambda url: urllib.request.urlopen(url, timeout=TIMEOUT_S))
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.frames: dict[str, Frame] = {}
        self.threads = [threading.Thread(target=self._read, args=(drone,), daemon=True, name=f"frames-{drone}")
                        for drone in self.backends]

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=TIMEOUT_S + 1.0)

    def latest(self, drone_id: str) -> Frame | None:
        with self.lock:
            return self.frames.get(drone_id)

    def _read(self, drone: str) -> None:
        url = self.backends[drone].rstrip("/") + "/mjpeg"
        number = 0
        while not self.stopping.is_set():
            try:
                with self.opener(url) as stream:
                    for jpeg in jpeg_chunks(stream):
                        number += 1
                        frame = Frame(drone, self.clock(), jpeg, number)
                        with self.lock:
                            self.frames[drone] = frame
                        if self.stopping.is_set():
                            return
            except Exception as error:
                logger.debug("[frames] %s: %s", drone, error)
            self.stopping.wait(RETRY_S)
