"""A scripted client that flies a Simulator directly, without any networking."""

from __future__ import annotations

from sim.rc_frames import CALIBRATE, FLY, HEADLESS, STOP, hy_packet  # noqa: F401
from sim.config import DEFAULT_FIRMWARE
from sim.simulator import Simulator

CLIENT = ("172.16.10.2", 50000)
CLIENT_IP = bytes([172, 16, 10, 2])


class Pilot:
    def __init__(self, firmware: str = DEFAULT_FIRMWARE, scene=None, rate_hz: float = 80.0, step_hz: float = 400.0,
                 seed: int = 1, spec=None, tuning=None):
        self.sim = Simulator(firmware=firmware, scene=scene, seed=seed, spec=spec, tuning=tuning)
        self.fc = self.sim.fc
        self.rate_hz = rate_hz
        self.step_s = 1.0 / step_hz
        self._next_packet = 0.0
        self._next_heartbeat = 0.0
        self._handshake_on = False

    def handshake(self) -> None:
        """The connect sequence that hiturbo-rs and the stock apps use."""
        for payload in (b"\x0f", b"\x28", b"\xff\x53\x54\x33\x16", b"\x2c", b"\x27"):
            self.sim.receive_udp(CLIENT, payload)
        self._handshake_on = True
        self._next_heartbeat = self.sim.t

    def _advance(self, seconds: float, builder) -> None:
        end = self.sim.t + seconds
        while self.sim.t < end:
            if self._handshake_on and self.sim.t >= self._next_heartbeat:
                self._next_heartbeat = self.sim.t + 1.0
                self.sim.receive_udp(CLIENT, b"\x09" + CLIENT_IP)
            if builder is not None:
                while self._next_packet <= self.sim.t:
                    self.sim.receive_udp(CLIENT, builder())
                    self._next_packet += 1.0 / self.rate_hz
            else:
                self._next_packet = self.sim.t
            self.sim.step(self.step_s)

    def fly(self, seconds: float, roll=128, pitch=128, throttle=128, yaw=128, flags=0, byte7=0x0A) -> None:
        packet = hy_packet(roll, pitch, throttle, yaw, flags, byte7)
        self._advance(seconds, lambda: packet)

    def hold(self, flag: int, seconds: float, **kwargs) -> None:
        self.fly(seconds, flags=flag, **kwargs)

    def stream(self, builder, seconds: float) -> None:
        self._advance(seconds, builder)

    def packets(self, count: int, **kwargs) -> None:
        """Send exactly this many packets at the current rate."""
        packet = hy_packet(**kwargs)
        for _ in range(count):
            self.sim.receive_udp(CLIENT, packet)
            self._next_packet = self.sim.t + 1.0 / self.rate_hz
            self._advance_without_packets(1.0 / self.rate_hz)

    def _advance_without_packets(self, seconds: float) -> None:
        end = self.sim.t + seconds
        while self.sim.t < end:
            self.sim.step(self.step_s)

    def silence(self, seconds: float) -> None:
        self._advance(seconds, None)
