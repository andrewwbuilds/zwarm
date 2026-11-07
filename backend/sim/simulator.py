"""One simulated drone with its own clock. It has no sockets, so tests run faster than real time."""

from __future__ import annotations

import math
import random

from sim.config import DEFAULT_FIRMWARE, DroneSpec, FIRMWARES, FlightTuning
from sim.flight_controller import FlightController
from sim.link import RadioLink
from sim.physics import Quadcopter
from sim.scene import Scene
from sim.wifi_board import WifiBoard


class Wind:
    """A steady wind with gusts that wander around it."""

    def __init__(self, speed: float, direction_deg: float, gust: float, rng: random.Random):
        self.rng = rng
        self.set(speed, direction_deg, gust)
        self._gust_x = self._gust_y = 0.0

    def set(self, speed: float, direction_deg: float, gust: float) -> None:
        self.speed, self.direction_deg, self.gust = speed, direction_deg, gust
        self._x = speed * math.cos(math.radians(direction_deg))
        self._y = speed * math.sin(math.radians(direction_deg))

    def sample(self, dt: float) -> tuple[float, float, float]:
        if self.gust > 0.0:
            spread = self.gust * math.sqrt(2.0 * dt / 3.0)
            self._gust_x += -self._gust_x / 3.0 * dt + spread * self.rng.gauss(0.0, 1.0)
            self._gust_y += -self._gust_y / 3.0 * dt + spread * self.rng.gauss(0.0, 1.0)
        return self._x + self._gust_x, self._y + self._gust_y, 0.0


class Simulator:
    def __init__(self, firmware: str = DEFAULT_FIRMWARE, scene: Scene | None = None, seed: int = 1,
                 spec: DroneSpec | None = None, tuning: FlightTuning | None = None):
        self.spec = spec or DroneSpec.calibrated()
        self.tuning = tuning or FlightTuning.calibrated()
        self.firmware = FIRMWARES[firmware]
        self.scene = scene or Scene(name="open")
        self.rng = random.Random(seed)
        self.t = 0.0
        self.drone = Quadcopter(self.spec, self.scene, battery_soc=self.firmware.battery_soc)
        start_x, start_y, start_yaw = self.scene.start
        self.drone.place(start_x, start_y, self.spec.gear_height_m, start_yaw)
        self._start = (start_x, start_y)
        self.wind = Wind(self.scene.wind_mps, self.scene.wind_direction_deg, self.scene.gust_mps, self.rng)
        self.link = RadioLink(self.rng)
        self.fc = FlightController(self.spec, self.tuning, self.firmware, self.drone, self.scene, self.rng)
        self.board = WifiBoard(self.firmware, self.fc, self.drone, self.scene, self.tuning, self.link, self.rng)

    def place(self, x: float, y: float, yaw: float) -> None:
        """Put the landed drone somewhere else in the scene."""
        self.drone.place(x, y, self.spec.gear_height_m, yaw)
        self._start = (x, y)

    def step(self, dt: float) -> None:
        self.t += dt
        self.link.offset_m = math.hypot(self.drone.x - self._start[0], self.drone.y - self._start[1])
        self.board.update(self.t)
        commands = self.fc.update(self.t, dt)
        self.drone.step(dt, commands, self.wind.sample(dt))

    def receive_udp(self, source, data: bytes) -> list[bytes]:
        return self.board.on_udp(self.t, source, data)
