"""The coordinator's connection to one drone's backend, over that backend's control websocket.

Each link has its own thread, so a stalled Pi can't hold up the other drones. Only the newest stick
command is ever sent: a backlog of old movements must never be replayed after a stall. If the link
drops, the backend centres that drone's sticks by itself, and its stock flight controller holds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

logger = logging.getLogger(__name__)

RECONNECT_S = 0.5
PING_S = 1.0
ESTOP_HOLD_S = 5.0  # an E-stop that couldn't be delivered keeps trying for this long


class BackendLink:
    def __init__(self, url: str, name: str = ""):
        self.url = url.rstrip("/")
        self.ws_url = self.url.replace("http", "ws", 1) + "/ws"
        self.name = name or url
        self.connected = False
        self.latency: float | None = None
        self.problem: str | None = None  # something about the backend that should stop a takeoff
        self.condition = threading.Condition()
        self.sticks: tuple | None = None
        self.commands: list[str] = []
        self.estop_until = 0.0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"link-{self.name}")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)

    def send_sticks(self, throttle: float, yaw: float, pitch: float, roll: float, ttl: float) -> None:
        with self.condition:
            self.sticks = (throttle, yaw, pitch, roll, ttl)
            self.condition.notify_all()

    def command(self, name: str) -> bool:
        """Queue a takeoff, land or estop. False if it can't be delivered right now.

        Takeoff and land are one toggle on the drone, so a command that couldn't be sent is dropped
        rather than held for later, when it could mean the opposite. Only E-stop is held.
        """
        with self.condition:
            if name == "estop":
                self.estop_until = time.monotonic() + ESTOP_HOLD_S
                self.commands.insert(0, name)
                self.condition.notify_all()
                return True
            if not self.connected:
                return False
            self.commands.append(name)
            self.condition.notify_all()
            return True

    def inspect(self) -> None:
        """Ask the backend whether anything about it should stop a takeoff."""
        try:
            with urllib.request.urlopen(f"{self.url}/capabilities", timeout=2.0) as reply:
                info = json.load(reply)
        except (OSError, ValueError) as error:
            self.problem = f"could not read its capabilities: {error}"
            return
        problems = []
        if info.get("busy"):
            problems.append("a trajectory or plugin is running on it")
        if info.get("airborne"):
            problems.append("its backend presumes it is airborne, so a takeoff would be ignored")
        if not info.get("commands", {}).get("takeoff"):
            problems.append(f"drone type {info.get('drone_type')} has no takeoff command")
        self.problem = "; ".join(problems) or None

    def run(self) -> None:
        while not self.stopping.is_set():
            try:
                with connect(self.ws_url, open_timeout=2.0, close_timeout=1.0, ping_interval=None,
                             proxy=None, max_queue=1) as socket:
                    self.connected = True
                    self.inspect()
                    logger.info("[link] %s connected", self.name)
                    self.serve(socket)
            except (OSError, ConnectionClosed, TimeoutError) as error:
                logger.debug("[link] %s: %s", self.name, error)
            if self.connected:
                logger.warning("[link] %s lost", self.name)
            self.connected = False
            self.latency = None
            with self.condition:
                self.sticks = None
                # Whatever was queued was decided for a moment that has passed.
                self.commands = ["estop"] if time.monotonic() < self.estop_until else []
            self.stopping.wait(RECONNECT_S)

    def serve(self, socket) -> None:
        next_ping = 0.0
        pong = None
        ping_at = 0.0
        next_inspect = time.monotonic() + 2.0
        while not self.stopping.is_set():
            with self.condition:
                if self.sticks is None and not self.commands:
                    self.condition.wait(0.1)
                sticks, self.sticks = self.sticks, None
                commands, self.commands = self.commands, []
            for name in commands:
                socket.send(json.dumps({"type": name, "source": "swarm"}))
            if sticks is not None:
                throttle, yaw, pitch, roll, ttl = sticks
                socket.send(json.dumps({"type": "axes", "mode": "linear", "source": "swarm", "ttl": ttl,
                                        "throttle": throttle, "yaw": yaw, "pitch": pitch, "roll": roll}))
            now = time.monotonic()
            if pong is not None and pong.is_set():
                self.latency = round(now - ping_at, 4)
                pong = None
            if now >= next_ping:
                if pong is not None:
                    self.latency = round(now - ping_at, 4)  # still unanswered: at least this slow
                else:
                    pong = socket.ping()
                    ping_at = now
                next_ping = now + PING_S
            if now >= next_inspect and sticks is None:
                self.inspect()
                next_inspect = now + 2.0
