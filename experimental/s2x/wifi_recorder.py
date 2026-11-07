"""Record everything an S20 sends over WiFi while you move it by hand, to look for sensor data.

Motors stay off: only the app's read-only queries, heartbeats and centred sticks with
no command flags are sent. You hold the drone and follow the prompts (tilt, turn, lift).

    uv run --project backend --frozen --offline python experimental/s2x/wifi_recorder.py

Writes events.jsonl, rssi.txt and frames/ to ~/Developer/datasets/s20-wifi-capture/<time>/.
"""

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from takeoff_probe import Log, drone_wifi_name, join_wifi, restore_wifi, wifi_device, wifi_ip, wifi_router  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NEUTRAL = bytes([0x66, 0x14, 128, 128, 128, 128, 0x00, 0x0A] + [0] * 10 + [0x0A, 0x99])
assert NEUTRAL[6] == 0  # no takeoff, land, stop or calibrate bit, ever
TCP_HEARTBEAT = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x25, 0x25])
QUERIES = (b"\x28", b"\x2c", b"\xff\x53\x54\x33\x16", b"\x27")  # read-only, from hiturbo-rs
ENVELOPE = 44

PHASES = (
    ("still", "Put the drone flat and still on a table", 10),
    ("hold_level", "Pick it up and hold it level and still, camera pointing at something with detail", 8),
    ("pitch", "Tilt the nose down slowly, then back to level. Do it twice", 10),
    ("roll", "Tilt it left, then right, slowly. Do it twice", 10),
    ("yaw", "Turn it slowly about 90 degrees left, then back. Do it twice", 12),
    ("lift", "Raise it straight up about one meter, then lower it. Do it twice", 12),
    ("walk", "Walk forward about 3 meters holding it level, then back", 14),
    ("shake", "Shake it gently side to side", 6),
)


class Recorder:
    def __init__(self, drone_ip: str, out: Path):
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self.events = open(out / "events.jsonl", "w", buffering=1)
        self.drone = (drone_ip, 8080)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((drone_ip, 1))
            self.local_ip = probe.getsockname()[0]
        self.cmd = self._udp(0)
        self.video = self._udp(8888)
        self.status = self._udp(8081)
        self.video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self.tcp = None
        self.phase = "setup"
        self.stop = threading.Event()
        self.chunks: dict[int, dict[int, bytes]] = {}
        self.frame_count = 0
        self.packets = 0
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._send_loop, self._recv_loop)]

    @staticmethod
    def _udp(port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        return s

    def log(self, event: str, **fields) -> None:
        self.events.write(json.dumps({"t": round(time.time(), 4), "phase": self.phase, "event": event, **fields}) + "\n")

    def start(self) -> None:
        self.cmd.sendto(b"\x0f", self.drone)
        try:
            self.tcp = socket.create_connection((self.drone[0], 8888), timeout=1.0)
            self.tcp.setblocking(False)
        except OSError as e:
            self.log("tcp_failed", error=str(e))
        for t in self.threads:
            t.start()

    def _send_loop(self) -> None:
        ip = socket.inet_aton(self.local_ip)
        next_rc = next_hb = next_q = next_tcp = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            try:
                if now >= next_rc:
                    next_rc = now + 0.05
                    self.cmd.sendto(NEUTRAL, self.drone)
                if now >= next_hb:
                    next_hb = now + 1.0
                    self.cmd.sendto(b"\x09" + ip, self.drone)
                    self.cmd.sendto(b"\x08" + ip, self.drone)
                if now >= next_q:
                    next_q = now + 2.0
                    for q in QUERIES:
                        self.cmd.sendto(q, self.drone)
                if self.tcp is not None and now >= next_tcp:
                    next_tcp = now + 1.5
                    self.tcp.sendall(TCP_HEARTBEAT)
            except OSError as e:
                self.log("send_error", error=str(e))
            time.sleep(0.01)

    def _recv_loop(self) -> None:
        roles = {self.cmd: "command", self.video: "video", self.status: "status"}
        while not self.stop.is_set():
            socks = list(roles) + ([self.tcp] if self.tcp is not None else [])
            ready, _, _ = select.select(socks, [], [], 0.05)
            for s in ready:
                try:
                    if s is self.tcp:
                        data = s.recv(4096)
                        if not data:
                            self.tcp = None
                        else:
                            self.log("tcp", hex=data.hex())
                        continue
                    data, peer = s.recvfrom(4096)
                except OSError:
                    continue
                self.packets += 1
                if roles[s] == "video" and data[:2] == b"\x40\x40":
                    self._video(data)
                else:
                    self.log("rx", role=roles[s], port=peer[1], hex=data.hex())

    def _video(self, data: bytes) -> None:
        frame_id, total, index = int.from_bytes(data[2:4], "little"), data[4], data[5]
        parts = self.chunks.setdefault(frame_id, {})
        parts[index] = data[8:-2]
        for old in [k for k in self.chunks if k != frame_id and len(self.chunks) > 4]:
            del self.chunks[old]
        if len(parts) == total:
            blob = b"".join(parts[i] for i in range(total))
            del self.chunks[frame_id]
            length = int.from_bytes(blob[16:20], "little")
            jpeg = blob[ENVELOPE:ENVELOPE + length]
            self.frame_count += 1
            (self.out / "frames" / f"{self.frame_count:05d}.jpg").write_bytes(jpeg)
            self.log("frame", n=self.frame_count, id=frame_id, envelope=blob[:ENVELOPE].hex(), jpeg_bytes=len(jpeg))

    def close(self) -> None:
        self.stop.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)
        for s in (self.cmd, self.video, self.status, self.tcp):
            if s is not None:
                s.close()
        self.events.close()


def wait_key(prompt: str, auto: bool) -> None:
    print(prompt, end=" ", flush=True)
    if auto:
        print()
        return
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    while os.read(sys.stdin.fileno(), 1) not in (b"\n", b"\r", b" "):
        pass
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drone-ip")
    parser.add_argument("--no-wifi", action="store_true")
    parser.add_argument("--auto", action="store_true", help="no prompts, short phases (for testing)")
    parser.add_argument("--out", default=str(Path.home() / "Developer/datasets/s20-wifi-capture"))
    args = parser.parse_args()
    if not args.auto and not sys.stdin.isatty():
        print("Run this in a terminal: it prompts you as it goes.", file=sys.stderr)
        return 1

    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    device = ssid = None
    wifi_log = Log(out / "wifi.log")
    if not args.no_wifi:
        device = wifi_device()
        if not wifi_router(device).startswith("172.16."):
            scan = subprocess.run(["swift", str(REPO / "tools/scan_drone_wifi.swift")], capture_output=True, text=True).stdout.split()
            ssid = scan[0] if scan else drone_wifi_name()
            if not ssid or not join_wifi(device, ssid, wifi_log):
                print("Could not join the drone's WiFi. Is it powered on?", file=sys.stderr)
                if ssid:
                    restore_wifi(device, ssid)
                return 1
    drone_ip = args.drone_ip or (wifi_router(device) if device else "") or "127.0.0.1"
    print(f"Drone address: {drone_ip}. Saving to {out}")

    rssi = None
    if not args.no_wifi:
        rssi = subprocess.Popen(["swift", str(REPO / "tools/rssi_stream.swift")], stdout=open(out / "rssi.txt", "w"), stderr=subprocess.DEVNULL)
    rec = None
    saved = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    try:
        if saved:
            tty.setcbreak(sys.stdin.fileno())
        rec = Recorder(drone_ip, out)
        rec.start()
        time.sleep(4 if not args.auto else 1)
        for name, instruction, seconds in PHASES:
            seconds = 2 if args.auto else seconds
            rec.phase = "setup"
            print(f"\n{instruction}.")
            wait_key("Press Enter, then do it for the next %d seconds:" % seconds, args.auto)
            rec.phase = name
            rec.log("phase_start", seconds=seconds)
            time.sleep(seconds)
            rec.log("phase_end")
            print(f"  done ({rec.frame_count} frames so far)")
        rec.phase = "finish"
        print(f"\nRecorded {rec.frame_count} frames and {rec.packets} packets.")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if rec:
            rec.close()
        if rssi:
            rssi.terminate()
        if saved:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
        if ssid:
            restore_wifi(device, ssid)
    print(f"Saved to {out}. Tell Claude when you're back online.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
