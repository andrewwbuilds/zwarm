"""No-fly link check for s2x drones such as the Hiturbo S20.

Confirms this Mac can reach the drone and that video arrives, without starting
the motors: it sends only the video heartbeat and centred-stick RC packets,
never a takeoff, land, or stop flag.

    uv run python preflight.py
"""

import argparse
import errno
import ipaddress
import os
import sys
import threading
import time

import cv2
import numpy as np

from utils.logging_config import bootstrap_runtime

bootstrap_runtime()

from models.s2x_rc import S2xDroneModel
from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from protocols.s2x_video_protocol import S2xVideoProtocolAdapter


def fail(message: str) -> int:
    print(f"FAILED: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="longest to wait for video (the camera can take ~5s to start)")
    parser.add_argument("--frames", type=int, default=30, help="stop early after this many frames")
    parser.add_argument("--save", help="write the first frame to this JPEG path")
    args = parser.parse_args()

    drone_ip = os.getenv("DRONE_IP", "172.16.10.1")
    control_port = int(os.getenv("CONTROL_PORT", 8080))
    video_port = int(os.getenv("VIDEO_PORT", 8888))

    try:
        video = S2xVideoProtocolAdapter(drone_ip, control_port, video_port)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return fail(f"UDP port {video_port} is busy. Stop the running backend first.")
        return fail(f"No network path to {drone_ip} ({e}). Is Wi-Fi on?")

    if ipaddress.ip_address(video.local_ip) not in ipaddress.ip_network(f"{drone_ip}/24", strict=False):
        local_ip = video.local_ip
        video.stop()
        return fail(f"Not on the drone's Wi-Fi (this Mac is {local_ip}). "
                    "Power on the drone and join its HITURBO_S20-xxxxxx network.")

    rc = S2xRCProtocolAdapter(drone_ip, control_port)
    model = S2xDroneModel()  # sticks centred, every command flag clear
    stop_rc = threading.Event()
    rc_sent = 0
    rc_error: OSError | None = None

    def rc_loop() -> None:
        nonlocal rc_sent, rc_error
        while not stop_rc.is_set():
            try:
                rc.send_control_packet(rc.build_control_packet(model))
                rc_sent += 1
                rc_error = None
            except OSError as e:
                rc_error = e
            time.sleep(0.05)

    rc_thread = threading.Thread(target=rc_loop, daemon=True)
    rc_thread.start()
    video.start()

    frames = packets = received_bytes = 0
    first_frame: bytes | None = None
    first_at = last_at = None
    worst_gap = 0.0
    started = time.monotonic()
    while time.monotonic() - started < args.seconds and frames < args.frames:
        frame = video.get_frame(timeout=0.25)
        for packet in video.get_packets():
            packets += 1
            received_bytes += len(packet)
        if frame is None:
            continue
        now = time.monotonic()
        frames += 1
        if first_frame is None:
            first_frame, first_at = frame.data, now
        else:
            worst_gap = max(worst_gap, now - last_at)
        last_at = now

    stop_rc.set()
    rc_thread.join(timeout=1.0)
    video.stop()
    rc.sock.close()

    print(f"Network: Mac {video.local_ip} -> drone {drone_ip}")
    print(f"RC:      {rc_sent} centred-stick packets sent" + (f", last error: {rc_error}" if rc_error else ""))
    print(f"Video:   {packets} packets, {received_bytes / max(time.monotonic() - started, 0.001) / 1024:.0f} KiB/s, {frames} frames")

    if rc_error is not None and rc_error.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH):
        return fail("macOS refused to send to the drone. Allow your terminal app under System Settings > "
                    "Privacy & Security > Local Network, then restart it.")
    if packets == 0:
        return fail("No video packets arrived. Power-cycle the drone and rejoin its Wi-Fi. If that does not help, "
                    "allow python3.13 under System Settings > Network > Firewall > Options.")
    if first_frame is None:
        return fail(f"{packets} packets arrived but none formed a complete frame.")

    image = cv2.imdecode(np.frombuffer(first_frame, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return fail("The first frame is not a valid JPEG.")
    span = last_at - first_at
    fps = (frames - 1) / span if span > 0 else 0.0
    print(f"Frames:  {image.shape[1]}x{image.shape[0]}, {fps:.1f} fps, worst gap {worst_gap * 1000:.0f} ms")
    if args.save:
        with open(args.save, "wb") as f:
            f.write(first_frame)
        print(f"Saved:   {args.save}")
    print("OK: link is good. The motors were not started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
