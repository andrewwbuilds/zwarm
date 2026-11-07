"""Guided takeoff test for s2x drones whose video works but commands do nothing.

Tries each known way of sending takeoff, stops at the first that spins the motors,
and writes what happened to takeoff_probe.log. Takeoff is only ever sent after you
press t, and Esc sends an emergency stop at any prompt.

    uv run --project backend --frozen --offline python experimental/s2x/takeoff_probe.py

Sources for the variants: the 1-second hold, 20 Hz rate and connect handshake come
from hiturbo-rs and the H120D protocol notes. The other formats are in docs/research/S2x.md.
"""

import argparse
import errno
import json
import math
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

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / "takeoff_probe.log"
FLY = "fly"
STOP = "stop"
CENTER = (128, 128, 128, 128)  # roll, pitch, throttle, yaw
HOLD_SECONDS = 1.0


def hy(action=None, sticks=CENTER, byte7=0x0A) -> bytes:
    """The 20-byte HY packet: 66 14 RR PP TT YY F1 F2 <zeros> XOR 99."""
    packet = bytearray(20)
    packet[0:2] = b"\x66\x14"
    packet[2:6] = bytes(sticks)
    packet[6] = {None: 0x00, FLY: 0x01, STOP: 0x02}[action]
    packet[7] = byte7
    checksum = 0
    for value in packet[2:18]:
        checksum ^= value
    packet[18] = checksum
    packet[19] = 0x99
    return bytes(packet)


def hy_stock_byte7(action=None, sticks=CENTER) -> bytes:
    """HY with byte 7 = 0x02, the default in decompiled stock apps."""
    return hy(action, sticks, byte7=0x02)


def short(action=None, sticks=CENTER) -> bytes:
    """The legacy 8-byte Macrochip packet: 66 RR PP TT YY FF XOR 99."""
    flags = 0x40 | {None: 0x00, FLY: 0x01, STOP: 0x04}[action]
    body = bytes(sticks) + bytes([flags])
    checksum = 0
    for value in body:
        checksum ^= value
    return b"\x66" + body + bytes([checksum, 0x99])


def fei_sha(action=None, sticks=None) -> bytes:
    """The 12-byte Heliway packet captured from an H120D (its sticks stay centred)."""
    flags = {None: 0x00, FLY: 0x01, STOP: 0x80}[action]
    body = bytes([0x08, 0x02, flags, 0x7F, 0x7F, 0x80, 0x80, 0x20, 0x20])
    checksum = 0
    for value in body:
        checksum ^= value
    return b"\x5a\x55" + body + bytes([checksum])


class Log:
    def __init__(self, path: Path):
        self.file = open(path, "w", buffering=1)
        self.start = time.time()

    def write(self, event: str, **fields) -> None:
        self.file.write(json.dumps({"t": round(time.time() - self.start, 2), "event": event, **fields}) + "\n")


class Link:
    """Sockets and background threads that talk to the drone."""

    def __init__(self, drone_ip: str, log: Log):
        self.log = log
        self.target = (drone_ip, 8080)
        self.drone_ip = drone_ip
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((drone_ip, 1))
            self.local_ip = probe.getsockname()[0]
        self.rc = self._udp(0)
        self.video = self._udp(8888)
        self.status = self._udp(8081)
        self.tcp = None
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.descriptor = threading.Event()
        self.handshake_on = False
        self.build = None
        self.rate = 20.0
        self.sticks = CENTER
        self.action = None
        self.action_until = 0.0
        self.action_packets = None
        self.rc_sent = 0
        self.rc_error = None
        self.video_packets = 0
        self.video_frames = 0
        self.fc_frames = 0
        self.replies = 0
        self.sent_any_command = False
        self._last_frame_id = None
        self._last_status_value = None
        self._seen = set()
        self.threads = [threading.Thread(target=t, daemon=True) for t in (self._rc_loop, self._heartbeat_loop, self._rx_loop)]
        for thread in self.threads:
            thread.start()

    @staticmethod
    def _udp(port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        return sock

    def configure(self, build, rate: float) -> None:
        with self.lock:
            self.build = build
            self.rate = rate

    def set_sticks(self, sticks) -> None:
        with self.lock:
            self.sticks = sticks

    def fire(self, action: str, seconds: float | None = None, packets: int | None = None) -> None:
        """Send a command for a time, or for an exact number of packets."""
        with self.lock:
            self.sent_any_command = True
            self.action = action
            self.action_packets = packets
            self.action_until = math.inf if seconds is None else time.monotonic() + seconds

    def stop_all(self, seconds: float = 1.0) -> None:
        """Emergency stop in every format, with the normal stream paused."""
        with self.lock:
            saved = self.build
            self.build = None
            self.action = None
        self.log.write("estop")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            for packet in (hy(STOP), short(STOP), fei_sha(STOP)):
                try:
                    self.rc.sendto(packet, self.target)
                except OSError as e:
                    self.rc_error = e
            time.sleep(0.05)
        with self.lock:
            self.build = saved

    def start_handshake(self) -> None:
        """The connect sequence hiturbo-rs and the stock apps use."""
        self.rc.sendto(b"\x0f", self.target)
        if not self.descriptor.wait(3.0):
            self.rc.sendto(b"\x0f", self.target)
            self.descriptor.wait(3.0)
        for payload in (b"\x28", b"\xff\x53\x54\x33\x16", b"\x2c", b"\x27"):
            self.rc.sendto(payload, self.target)
        try:
            tcp = socket.create_connection((self.drone_ip, 8888), timeout=1.0)
            tcp.setblocking(False)
            self.tcp = tcp
        except OSError as e:
            print(f"  TCP 8888 unavailable: {e}")
            self.log.write("tcp_failed", error=str(e))
        self.handshake_on = True
        self.log.write("handshake_on", descriptor=self.descriptor.is_set())

    def _rc_loop(self) -> None:
        next_send = time.monotonic()
        while not self.closing.is_set():
            with self.lock:
                build, rate, sticks = self.build, self.rate, self.sticks
                now = time.monotonic()
                action = None
                if self.action is not None and now < self.action_until and self.action_packets != 0:
                    action = self.action
                    if self.action_packets is not None:
                        self.action_packets -= 1
                else:
                    self.action = None
            if build is not None:
                try:
                    self.rc.sendto(build(action, sticks), self.target)
                    self.rc_sent += 1
                    self.rc_error = None
                except OSError as e:
                    self.rc_error = e
            next_send += 1.0 / rate
            delay = next_send - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_send = time.monotonic()

    def _heartbeat_loop(self) -> None:
        tcp_sent = 0
        tcp_next = 0.0
        next_udp = 0.0
        heartbeat_tcp = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x25, 0x25])
        ip = socket.inet_aton(self.local_ip)
        while not self.closing.is_set():
            now = time.monotonic()
            if now >= next_udp:
                next_udp = now + 1.0
                try:
                    if self.handshake_on:
                        self.rc.sendto(b"\x09" + ip, self.target)
                        self.rc.sendto(b"\x08" + ip, self.target)
                    else:
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as once:
                            once.sendto(b"\x08" + ip, self.target)
                except OSError as e:
                    self.rc_error = e
            if self.tcp is not None and now >= tcp_next:
                try:
                    self.tcp.sendall(heartbeat_tcp)
                except OSError:
                    self.tcp = None
                tcp_sent += 1
                tcp_next = now + (0.1 if tcp_sent < 10 else 1.5)
            time.sleep(0.02)

    def _rx_loop(self) -> None:
        roles = {self.rc: "command", self.video: "video", self.status: "status"}
        while not self.closing.is_set():
            tcp = self.tcp
            sockets = list(roles) + ([tcp] if tcp is not None else [])
            try:
                ready, _, _ = select.select(sockets, [], [], 0.05)
            except (OSError, ValueError):
                continue
            for sock in ready:
                try:
                    if sock is tcp:
                        data = sock.recv(4096)
                        if not data:
                            self.tcp = None
                            continue
                        self._on_packet("tcp", data)
                    else:
                        data, _ = sock.recvfrom(4096)
                        self._on_packet(roles[sock], data)
                except OSError:
                    pass

    def _on_packet(self, role: str, data: bytes) -> None:
        if data[:2] == b"\x40\x40":
            self.video_packets += 1
            frame_id = data[2:4]
            if frame_id != self._last_frame_id:
                self._last_frame_id = frame_id
                self.video_frames += 1
            return
        if data[:4] == b"\xff\x53\x54\x08" and len(data) == 5:
            self.replies += 1
            if data[4] != self._last_status_value:
                self._last_status_value = data[4]
                self.log.write("board_status", value=data[4])
            return
        if data == b"noact\r\n":
            return
        if data[:2] == b"\xaa\xff":
            self.descriptor.set()
        is_fc_frame = len(data) == 10 and data[0] == 0x66
        if is_fc_frame:
            self.fc_frames += 1
        key = (role, data)
        if key in self._seen:
            return
        self._seen.add(key)
        self.log.write("drone_said", role=role, hex=data.hex(), fc_frame=is_fc_frame)
        printable = all(32 <= b < 127 or b in (10, 13) for b in data)
        text = repr(data.decode("ascii")) if printable else data.hex(" ")
        label = "flight controller says" if is_fc_frame else "drone says"
        print(f"\r  [{label}: {text}]")

    def close(self) -> None:
        self.closing.set()
        for thread in self.threads:
            thread.join(timeout=1.0)
        for sock in (self.rc, self.video, self.status, self.tcp):
            if sock is not None:
                sock.close()


class Done(Exception):
    """Motors spun and the drone is down again."""


def ask(prompt: str, allowed: str, link: Link) -> str:
    """Read one key. Esc is an emergency stop at every prompt."""
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)  # a stray earlier key must not fire takeoff
    print(prompt, end=" ", flush=True)
    while True:
        key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if not key:
            raise KeyboardInterrupt
        if key == "\x1b":
            print("\n  E-STOP sent")
            link.stop_all()
            print(prompt, end=" ", flush=True)
        elif key.lower() in allowed:
            print(key)
            return key.lower()


def lights_changed(link: Link, log: Log, step: str) -> bool:
    answer = ask("  Did the drone's lights change? y/n:", "yn", link)
    log.write("lights_changed", step=step, answer=answer)
    return answer == "y"


def flying(link: Link, log: Log, step: str, seconds, packets) -> None:
    """Motors are running. Keep sticks centred until the pilot lands or stops."""
    print("\n  The motors started! Sticks stay centred, so it should hold still.")
    while True:
        choice = ask("  Press l to land (Esc = E-STOP cuts the motors):", "l", link)
        link.fire(FLY, seconds=seconds, packets=packets)
        log.write("land_sent", step=step)
        time.sleep((seconds or 0) + 4)
        if ask("  Are the motors off? y/n:", "yn", link) == "y":
            log.write("down", step=step)
            raise Done
        print("  If it stays up, press l again, or Esc to cut the motors.")


def attempt(link: Link, log: Log, step: str, detail: str, build, rate: float,
            seconds: float | None = HOLD_SECONDS, packets: int | None = None) -> None:
    link.configure(build, rate)
    print(f"\nTest {step}: {detail}")
    if ask("  Press t to send takeoff, s to skip:", "ts", link) == "s":
        log.write("attempt_skipped", step=step)
        return
    time.sleep(1.0)  # let the new packet stream settle first
    link.fire(FLY, seconds=seconds, packets=packets)
    log.write("takeoff_sent", step=step, detail=detail, rate=rate, seconds=seconds, packets=packets)
    print("  Sending. Watch the drone...")
    time.sleep((seconds or 0) + 3)
    answer = ask("  Did the motors spin? y/n:", "yn", link)
    log.write("result", step=step, motors=answer, video_frames=link.video_frames,
              fc_frames=link.fc_frames, replies=link.replies, rc_sent=link.rc_sent, rc_error=str(link.rc_error))
    if answer == "y":
        flying(link, log, step, seconds, packets)


def stick_pattern(link: Link, log: Log, step: str, detail: str, positions) -> None:
    link.configure(hy, 20)
    print(f"\nTest {step}: {detail}")
    if ask("  Press t to send this stick pattern, s to skip:", "ts", link) == "s":
        log.write("attempt_skipped", step=step)
        return
    link.sent_any_command = True
    log.write("sticks_sent", step=step, positions=positions)
    for sticks in positions:
        link.set_sticks(sticks)
        time.sleep(1.5)
    link.set_sticks(CENTER)
    time.sleep(1.5)
    answer = ask("  Did the motors spin or the lights change? y/n:", "yn", link)
    log.write("result", step=step, motors_or_lights=answer)
    if answer == "y":
        print("  Noted. Now trying takeoff straight after that pattern.")
        attempt(link, log, f"{step}-takeoff", "Takeoff held for 1 second after that pattern", hy, 20)


def run_tests(link: Link, log: Log) -> None:
    print("\nFirst, a look at the drone. Nothing has been sent to it yet.")
    lights = ask("What are its lights doing? f=flashing  s=solid  o=off:", "fso", link)
    log.write("lights_before_any_packets", answer=lights)

    print("\nSending centred sticks for 5 seconds, as the browser does. Motors stay off.")
    link.configure(hy, 80)
    time.sleep(5)
    log.write("after_centered_sticks", video_frames=link.video_frames, replies=link.replies, rc_error=str(link.rc_error))
    print(f"  Video frames received: {link.video_frames}. Control replies: {link.replies}.")
    if link.rc_error is not None:
        print(f"  Sending failed: {link.rc_error}")
    lights_changed(link, log, "centered sticks")

    attempt(link, log, "A", "what the browser button sends today, 1 packet at 80 per second, to confirm the failure",
            hy, 80, seconds=None, packets=1)
    attempt(link, log, "B", "held for 1 second at 80 per second", hy, 80)
    attempt(link, log, "C", "held for 1 second at 20 per second, the stock app's rate", hy, 20)

    print("\nNow the app's connect handshake. Motors stay off.")
    link.start_handshake()
    time.sleep(3)
    lights_changed(link, log, "handshake")
    attempt(link, log, "D", "same as C, with the handshake running", hy, 20)
    attempt(link, log, "E", "same as D, with the stock app's default mode byte (0x02)", hy_stock_byte7, 20)

    print("\nOther packet formats, centred sticks only. Motors stay off.")
    for name, build in (("short", short), ("fei_sha", fei_sha)):
        link.configure(build, 20)
        print(f"  Sending the {name} format for 4 seconds.")
        time.sleep(4)
        lights_changed(link, log, f"format {name}")
    attempt(link, log, "F", "the legacy 8-byte packet format", short, 20)
    attempt(link, log, "G", "the 12-byte Heliway packet format", fei_sha, 50)

    print("\nLast, stick patterns that unlock some toy drones. If one works, the drone")
    print("could spin up or jump, so be ready to press Esc.")
    stick_pattern(link, log, "H", "throttle fully up, then fully down (pairing)", [(128, 128, 255, 128), (128, 128, 0, 128)])
    stick_pattern(link, log, "I", "both sticks down and inward (arming)", [(0, 0, 0, 255)])
    stick_pattern(link, log, "J", "both sticks down and outward (arming)", [(255, 0, 0, 0)])


def wifi_device() -> str:
    ports = subprocess.run(["networksetup", "-listallhardwareports"], capture_output=True, text=True).stdout.splitlines()
    for index, line in enumerate(ports):
        if "Wi-Fi" in line:
            return ports[index + 1].split()[-1]
    raise SystemExit("No Wi-Fi interface found.")


def wifi_ip(device: str) -> str:
    return subprocess.run(["ipconfig", "getifaddr", device], capture_output=True, text=True).stdout.strip()


def wifi_router(device: str) -> str:
    """The Wi-Fi gateway, which is the drone's address once joined to its network."""
    return subprocess.run(["ipconfig", "getoption", device, "router"], capture_output=True, text=True).stdout.strip()


def join_wifi(device: str, ssid: str, log: Log) -> bool:
    """Join the drone's Wi-Fi. It can take a while to appear after power-on."""
    print(f"Joining {ssid}. This Mac has no internet until the test ends.")
    before = wifi_ip(device)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        subprocess.run(["networksetup", "-setairportnetwork", device, ssid], capture_output=True)
        for _ in range(12):
            if wifi_ip(device) != before and wifi_router(device):
                log.write("joined_wifi", ip=wifi_ip(device), router=wifi_router(device))
                return True
            time.sleep(1)
    return False


def restore_wifi(device: str, ssid: str) -> None:
    print("Returning to your normal Wi-Fi...")
    subprocess.run(["networksetup", "-removepreferredwirelessnetwork", device, ssid], capture_output=True)
    subprocess.run(["networksetup", "-setairportpower", device, "off"], capture_output=True)
    time.sleep(1)
    subprocess.run(["networksetup", "-setairportpower", device, "on"], capture_output=True)


def drone_wifi_name() -> str:
    env = REPO / "backend" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DRONE_WIFI="):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drone-ip", help="default: the Wi-Fi gateway after joining the drone")
    parser.add_argument("--no-wifi", action="store_true", help="don't join or restore Wi-Fi")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("This test asks questions as it goes, so run it in a terminal.", file=sys.stderr)
        return 1

    log = Log(LOG_PATH)
    device = ssid = None
    if not args.no_wifi:
        device = wifi_device()
        if not wifi_router(device).startswith("172.16."):
            ssid = drone_wifi_name()
            if not ssid:
                print("Join the drone's Wi-Fi first, or add DRONE_WIFI=<its name> to backend/.env.", file=sys.stderr)
                return 1
            if not join_wifi(device, ssid, log):
                print(f"Could not join {ssid}. Is the drone powered on?", file=sys.stderr)
                restore_wifi(device, ssid)
                return 1

    if args.drone_ip is None:
        args.drone_ip = (wifi_router(device) if device else "") or "172.16.10.1"
    print(f"Drone address: {args.drone_ip}")

    link = None
    saved_terminal = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        try:
            link = Link(args.drone_ip, log)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                print("A UDP port is busy. Quit fly.sh first, then rerun this.", file=sys.stderr)
                return 1
            raise
        try:
            run_tests(link, log)
            log.write("finished", result="no test started the motors")
            print("\nNo test started the motors.")
        except Done:
            log.write("finished", result="motors started and stopped")
            print("\nThat one worked.")
    except KeyboardInterrupt:
        print("\nStopping.")
        log.write("interrupted")
    finally:
        if link is not None:
            if link.sent_any_command:
                link.stop_all()
            link.close()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved_terminal)
        if ssid:
            restore_wifi(device, ssid)
    print(f"Results saved to {LOG_PATH}. Tell Claude when you're back online.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
