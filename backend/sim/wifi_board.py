"""The XR872 WiFi and camera board: replies to clients and bridges to the flight controller.

Everything a client can observe here is taken from captures of the user's drone
(tests/fixtures/s20_capture.json) unless a comment says otherwise.
"""

from __future__ import annotations

import ipaddress
import math
import random
from collections import deque

DESCRIPTOR = bytes.fromhex("aaff0301010a0a011901010035")  # measured
FIRMWARE_VERSION = b"V4.6.1"  # measured
KEYFRAME_REPLY = b"forceI"  # measured
TCP_HEARTBEAT_REPLY = b"noact\r\n"  # measured
TCP_HEARTBEAT = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x25, 0x25])  # capture: hiturbo-rs, then measured

UART_DELAY_S = 0.017  # research: UDP in, UART out about 17 ms later
FLOW_PERIOD_S = 1 / 22  # research: the 77 flow report runs at about 22 Hz
FLOW_COUNTS_PER_RAD_S = 100.0  # estimate: research hasn't pinned the scale
VIDEO_TIMEOUT_S = 5.0  # estimate: must outlast upstream's working 2 s heartbeat
MESSAGE_TIMEOUT_S = 5.0  # estimate
TELEMETRY_PERIOD_S = 0.5  # estimate
FLOW_TEXTURE = {"white": 0.2, "black": 0.1, "metal": 0.4, "paint_red": 0.3, "tv": 0.3}  # estimate: plain tops


class WifiBoard:
    def __init__(self, firmware, fc, drone, scene, tuning, link, rng: random.Random):
        self.firmware = firmware
        self.fc = fc
        self.drone = drone
        self.scene = scene
        self.tuning = tuning
        self.link = link
        self.rng = rng
        self.t = 0.0
        self.camera_index = 0
        self.datagrams_received = 0
        self.control_packets_received = 0
        self.last_control_hex = ""
        self._video_address = None
        self._video_time = -math.inf
        self._message_address = None
        self._message_time = -math.inf
        self._seen_discovery = False
        self._seen_version = False
        self._uart: deque[tuple[float, bytes]] = deque()
        self._next_flow = 0.0
        self._next_telemetry = 0.0
        self._outbox: list[tuple[tuple[str, int], bytes]] = []
        self._status_wander = 0.0
        self.forward_blocked = 0

    # ---------------------------------------------------------- client traffic
    def on_udp(self, t: float, source, data: bytes) -> list[bytes]:
        """Handle a datagram on UDP 8080. Returns replies for the sender, in order."""
        self.datagrams_received += 1
        replies: list[bytes] = []
        if data == b"\x0f":
            self._seen_discovery = True
            replies.append(DESCRIPTOR)
        elif data == b"\x28":
            self._seen_version = True
            replies.append(FIRMWARE_VERSION)
        elif data == b"\x27":
            replies.append(KEYFRAME_REPLY)
        elif data == b"\x2c":
            pass  # measured: answered with only the status reply
        elif data == b"\xff\x53\x54\x33\x16":
            replies.append(data + bytes([self.camera_index]))  # measured
        elif len(data) == 5 and data[0] == 0x08:
            self._video_address = (str(ipaddress.IPv4Address(data[1:])), 8888)
            self._video_time = t
        elif len(data) == 5 and data[0] == 0x09:
            self._message_address = (str(ipaddress.IPv4Address(data[1:])), 8081)
            self._message_time = t
        elif len(data) == 2 and data[0] == 0x0B:
            self.camera_index = data[1]  # capture: hiturbo-rs's mock echoes this
            replies.append(data)
        elif data[:3] == b"\xff\x53\x54":
            pass  # other ST side-channel commands: not modelled
        else:
            self._forward(t, data)
        replies.append(self._status_reply())
        return replies

    def on_tcp(self, data: bytes) -> bytes:
        """The reply to a heartbeat on TCP 8888."""
        return TCP_HEARTBEAT_REPLY if data == TCP_HEARTBEAT else b""

    def video_target(self, t: float):
        if t - self._video_time > VIDEO_TIMEOUT_S:
            return None
        return self._video_address

    def handshake_complete(self, t: float) -> bool:
        return self._seen_discovery and self._seen_version and t - self._message_time < MESSAGE_TIMEOUT_S

    def take_outbox(self) -> list[tuple[tuple[str, int], bytes]]:
        outbox, self._outbox = self._outbox, []
        return outbox

    def _forward(self, t: float, data: bytes) -> None:
        # research: everything else goes to the flight controller byte for byte.
        self.control_packets_received += 1
        self.last_control_hex = data.hex()
        if self.firmware.requires_handshake and not self.handshake_complete(t):
            self.forward_blocked += 1
            return
        self._uart.append((t + UART_DELAY_S, data))

    def _status_reply(self) -> bytes:
        """measured: ff 53 54 08 <value> answers every datagram.

        The value's meaning is unknown. It sat near 103 at -38 dBm and near 88 at -47 dBm,
        so it is modelled as tracking signal strength, which may be a coincidence.
        """
        self._status_wander += self.rng.gauss(0.0, 0.25) - 0.02 * self._status_wander
        value = 103.5 + 1.67 * (self.link.rssi_dbm() + 38.0) + self._status_wander
        return b"\xff\x53\x54\x08" + bytes([max(0, min(255, round(value)))])

    # ---------------------------------------------------------------- the clock
    def update(self, t: float) -> None:
        self.t = t
        uart = self._uart
        while uart and uart[0][0] <= t:
            self.fc.on_uart(t, uart.popleft()[1])
        if t >= self._next_flow:
            self._next_flow = t + FLOW_PERIOD_S
            self.fc.on_uart(t, self._flow_frame())
        if self.firmware.status_frames and t >= self._next_telemetry:
            self._next_telemetry = t + TELEMETRY_PERIOD_S
            if self._message_address is not None and t - self._message_time < MESSAGE_TIMEOUT_S:
                self._outbox.append((self._message_address, self.fc.telemetry_frame()))
                del self._outbox[:-64]

    def _flow_frame(self) -> bytes:
        """research: 77 0E FXlo FXhi FYlo FYhi QQ 00 00 00 00 00 CK 99."""
        drone = self.drone
        flow_x = flow_y = quality = 0
        top, material = self.scene.ground(drone.x, drone.y, drone.z)
        height = max(drone.z - top, 0.05)
        texture = self.scene.flow_quality if top == 0.0 else FLOW_TEXTURE.get(material, 1.0)
        if not drone.on_ground and texture > 0.0 and height <= self.tuning.flow_max_height_m:
            yaw = drone.yaw()
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            forward = drone.vx * cos_yaw + drone.vy * sin_yaw
            left = -drone.vx * sin_yaw + drone.vy * cos_yaw
            flow_x = round(forward / height * FLOW_COUNTS_PER_RAD_S + self.rng.gauss(0.0, 1.5))
            flow_y = round(left / height * FLOW_COUNTS_PER_RAD_S + self.rng.gauss(0.0, 1.5))
            flow_x = max(-32768, min(32767, flow_x))
            flow_y = max(-32768, min(32767, flow_y))
            quality = round(255 * min(1.0, texture))
        body = flow_x.to_bytes(2, "little", signed=True) + flow_y.to_bytes(2, "little", signed=True)
        body += bytes([quality, 0, 0, 0, 0, 0])
        checksum = 0
        for value in body:
            checksum ^= value
        return b"\x77\x0e" + body + bytes([checksum, 0x99])
