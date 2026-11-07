"""The control packets a client can send, and how the flight controller reads them."""

from __future__ import annotations

from dataclasses import dataclass

# Byte 6 of the HY packet (research: docs/research/S2x.md)
FLY = 0x01
STOP = 0x02
CALIBRATE = 0x04
# Byte 7 of the HY packet (research)
HEADLESS = 0x01


@dataclass(slots=True)
class RcFrame:
    roll: float
    pitch: float
    throttle: float
    yaw: float
    fly: bool = False
    stop: bool = False
    calibrate: bool = False
    headless: bool = False


def _xor(data: bytes) -> int:
    checksum = 0
    for value in data:
        checksum ^= value
    return checksum


def _stick(value: int, center: int = 128) -> float:
    """Convert a raw stick byte to -1..+1."""
    return (value - center) / ((255 - center) if value >= center else center)


def hy_packet(roll=128, pitch=128, throttle=128, yaw=128, flags=0, byte7=0x0A) -> bytes:
    """The 20-byte HY packet: 66 14 RR PP TT YY F1 F2 <zeros> XOR 99."""
    body = bytes([roll, pitch, throttle, yaw, flags, byte7]) + bytes(10)
    return b"\x66\x14" + body + bytes([_xor(body), 0x99])


def short_packet(roll=128, pitch=128, throttle=128, yaw=128, fly=False, stop=False) -> bytes:
    """The legacy 8-byte Macrochip packet: 66 RR PP TT YY FF XOR 99."""
    flags = 0x40 | (0x01 if fly else 0) | (0x04 if stop else 0)
    body = bytes([roll, pitch, throttle, yaw, flags])
    return b"\x66" + body + bytes([_xor(body), 0x99])


def heliway_packet(fly=False, stop=False) -> bytes:
    """The 12-byte Heliway packet captured from an H120D."""
    flags = (0x01 if fly else 0) | (0x80 if stop else 0)
    body = bytes([0x08, 0x02, flags, 0x7F, 0x7F, 0x80, 0x80, 0x20, 0x20])
    return b"\x5a\x55" + body + bytes([_xor(body)])


def parse_frame(data: bytes, rc_format: str) -> tuple[RcFrame, int] | None:
    """Read a control packet in one format. Returns the frame and its mode byte."""
    if rc_format == "hy":
        if len(data) != 20 or data[0] != 0x66 or data[19] != 0x99 or _xor(data[2:18]) != data[18]:
            return None
        flags, mode = data[6], data[7]
        frame = RcFrame(_stick(data[2]), _stick(data[3]), _stick(data[4]), _stick(data[5]),
                        bool(flags & FLY), bool(flags & STOP), bool(flags & CALIBRATE), bool(mode & HEADLESS))
        return frame, mode
    if rc_format == "short":
        if len(data) != 8 or data[0] != 0x66 or data[7] != 0x99 or _xor(data[1:6]) != data[6]:
            return None
        flags = data[5]
        frame = RcFrame(_stick(data[1]), _stick(data[2]), _stick(data[3]), _stick(data[4]),
                        bool(flags & 0x01 or flags & 0x02), bool(flags & 0x04), bool(flags & 0x80), bool(flags & 0x10))
        return frame, flags
    if rc_format == "heliway":
        if len(data) != 12 or data[:4] != b"\x5a\x55\x08\x02" or _xor(data[2:11]) != data[11]:
            return None
        flags = data[4]
        frame = RcFrame(_stick(data[6], 127), _stick(data[5], 127), _stick(data[7]), _stick(data[8]),
                        bool(flags & 0x03), bool(flags & 0x80))
        return frame, flags
    raise ValueError(f"Unknown control format: {rc_format}")
