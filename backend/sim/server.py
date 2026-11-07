"""Runs a Simulator in real time behind the S20's own sockets, so turbodrone flies it unchanged.

UDP 8080 takes commands and answers like the drone, video leaves UDP 4096 for the client's
port 8888, TCP 8888 answers heartbeats, and a small HTTP API reports state and resets.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from sim.camera import RATE, Camera, Pose, VideoEncoder, packetize
from sim.config import DEFAULT_FIRMWARE, FIRMWARES
from sim.scenes import SCENES
from sim.simulator import Simulator

STEP = 1 / 400


class Server:
    def __init__(self, scene: str = "room", firmware: str = DEFAULT_FIRMWARE, host: str = "127.0.0.1",
                 control_port: int = 8080, tcp_port: int = 8888, video_source_port: int = 4096,
                 client_video_port: int = 8888, api_port: int = 8090, hud: bool = True,
                 render_scale: float = 0.36, seed: int = 1, log=print):
        self.host = host
        self.hud = hud
        self.render_scale = render_scale
        self.client_video_port = client_video_port
        self.log = log
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.inbox: list = []
        self.order = itertools.count()
        self.rng = random.Random(seed)
        self.notes: list[tuple[float, str]] = []

        self.control = udp(host, control_port)
        self.video = udp(host, video_source_port, fallback=True)
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind((host, tcp_port))
        self.tcp.listen()
        self.tcp.settimeout(0.5)
        self.api = ThreadingHTTPServer((host, api_port), handler(self))
        self.api.daemon_threads = True
        self.control_port = self.control.getsockname()[1]
        self.tcp_port = self.tcp.getsockname()[1]
        self.api_port = self.api.server_address[1]
        self.threads: list[threading.Thread] = []
        self.reset(scene=scene, firmware=firmware, seed=seed)

    # ------------------------------------------------------------------ control
    def reset(self, scene: str | None = None, firmware: str | None = None, battery: float | None = None,
              seed: int | None = None, wind: dict | None = None) -> None:
        """Power-cycle the drone where it took off, optionally somewhere else or with another battery."""
        with self.lock:
            scene = scene or self.scene_name
            firmware = firmware or self.firmware_name
            if scene not in SCENES:
                raise ValueError(f"unknown scene {scene!r}")
            if firmware not in FIRMWARES:
                raise ValueError(f"unknown firmware {firmware!r}")
            seed = self.rng.randrange(1 << 30) if seed is None else seed
            sim = Simulator(firmware=firmware, scene=SCENES[scene](), seed=seed)
            if battery is not None:
                sim.drone.battery.soc = min(1.0, max(0.0, float(battery)))
            sim.fc.listener = self.event
            self.sim = sim
            self.scene_name, self.firmware_name = scene, firmware
            self.camera = Camera(sim.scene, render_scale=self.render_scale, seed=seed,
                                 fov=math.radians(sim.spec.camera_fov_deg))
            self.encoder = VideoEncoder(seed=seed)
            self.frame_id = self.rng.randrange(1 << 16)
            self.origin = time.monotonic()
            self.inbox.clear()
            self.notes.clear()
            if wind:
                self.set_wind(**wind)
        self.log(f"[sim] Drone powered on in the {scene} (firmware {firmware})")

    def set_wind(self, speed: float | None = None, direction: float | None = None, gust: float | None = None) -> None:
        with self.lock:
            wind = self.sim.wind
            wind.set(wind.speed if speed is None else float(speed),
                     wind.direction_deg if direction is None else float(direction),
                     wind.gust if gust is None else float(gust))

    def event(self, t: float, message: str) -> None:
        self.notes.append((t, message))
        del self.notes[:-8]
        self.log(f"[sim {t:7.2f}s] {message}")

    def state(self) -> dict:
        with self.lock:
            sim = self.sim
            d, fc, wind = sim.drone, sim.fc, sim.wind
            return {
                "t": round(sim.t, 2),
                "scene": self.scene_name,
                "firmware": self.firmware_name,
                "state": fc.state,
                "leds": fc.leds(),
                "position": [round(d.x, 3), round(d.y, 3), round(d.z, 3)],
                "velocity": [round(d.vx, 3), round(d.vy, 3), round(d.vz, 3)],
                "attitude": [round(math.degrees(a), 1) for a in (d.roll(), d.pitch(), d.yaw())],
                "height": round(d.z - sim.spec.gear_height_m, 3),
                "speed": round(math.hypot(d.vx, d.vy), 3),
                "battery": round(d.battery.soc, 4),
                "voltage": round(d.voltage, 3),
                "current": round(d.current_a, 2),
                "link": fc.link_alive,
                "rc_rate": round(fc.rc_rate_hz, 1),
                "accepted": fc.frames_accepted,
                "rejected": fc.frames_rejected,
                "flow": fc.flow_valid,
                "rssi": round(sim.link.rssi_dbm(), 1),
                "distance": round(sim.link.distance_m(), 2),
                "video": sim.board.video_target(sim.t) is not None,
                "wind": {"speed": wind.speed, "direction": wind.direction_deg, "gust": wind.gust},
                "events": [[round(t, 2), m] for t, m in list(fc.events)[-12:]],
                "scenes": list(SCENES),
                "firmwares": {name: f.description for name, f in FIRMWARES.items()},
            }

    # ------------------------------------------------------------------ threads
    def start(self) -> None:
        for target in (self.physics, self.receive, self.stream, self.accept, self.api.serve_forever):
            thread = threading.Thread(target=target, daemon=True, name=f"sim-{target.__name__}")
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stopping.set()
        self.api.shutdown()
        for thread in self.threads:
            thread.join(timeout=2.0)
        for sock in (self.control, self.video, self.tcp):
            sock.close()
        self.api.server_close()

    def clock(self) -> float:
        return time.monotonic() - self.origin

    def physics(self) -> None:
        while not self.stopping.is_set():
            with self.lock:
                sim = self.sim
                now = self.clock()
                steps = 0
                while sim.t + STEP <= now and steps < 400:
                    self.deliver(sim)
                    sim.step(STEP)
                    steps += 1
                if steps == 400:
                    self.origin = time.monotonic() - sim.t
                outbox = sim.board.take_outbox()
            for address, data in outbox:
                self.send(self.control, data, address, "control")
            time.sleep(0.002)

    def deliver(self, sim: Simulator) -> None:
        inbox = self.inbox
        while inbox and inbox[0][0] <= sim.t:
            _, _, source, data = heapq.heappop(inbox)
            for reply in sim.receive_udp(source, data):
                self.send(self.control, reply, source, "control")

    def send(self, sock: socket.socket, data: bytes, address, kind: str) -> None:
        if self.rng.random() < self.sim.link.loss_probability(kind):
            return
        try:
            sock.sendto(data, address)
        except OSError:
            pass

    def receive(self) -> None:
        self.control.settimeout(0.5)
        while not self.stopping.is_set():
            try:
                data, source = self.control.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                delay = self.sim.link.transmit("control")
                if delay is not None:
                    heapq.heappush(self.inbox, (self.clock() + delay, next(self.order), source, data))

    def stream(self) -> None:
        due = time.monotonic()
        while not self.stopping.is_set():
            wait = due - time.monotonic()
            if wait > 0:
                self.stopping.wait(wait)
                continue
            with self.lock:
                sim, camera, encoder = self.sim, self.camera, self.encoder
                target = sim.board.video_target(sim.t)
                d = sim.drone
                pose = Pose(d.x, d.y, d.z, d.rotation())
                t = sim.t
                status = self.state() if self.hud else None
            if target is None:
                due = time.monotonic() + 1 / RATE
                continue
            image = camera.render(pose)
            if status is not None:
                overlay(image, status, list(self.notes), t)
            envelope, _ = encoder.encode(image, t)
            with self.lock:
                frame_id = self.frame_id
                self.frame_id = (frame_id + 1) & 0xFFFF
            for chunk in packetize(frame_id, envelope):
                self.send(self.video, chunk, (target[0], self.client_video_port), "video")
            due = max(due + encoder.gap, time.monotonic() - 0.2)

    def accept(self) -> None:
        while not self.stopping.is_set():
            try:
                conn, _ = self.tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self.heartbeat, args=(conn,), daemon=True).start()

    def heartbeat(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        with conn:
            while not self.stopping.is_set():
                try:
                    data = conn.recv(256)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not data:
                    return
                for i in range(0, len(data), 12):
                    reply = self.sim.board.on_tcp(data[i:i + 12])
                    if reply:
                        conn.sendall(reply)


def udp(host: str, port: int, fallback: bool = False) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        if not fallback:
            sock.close()
            raise
        sock.bind((host, 0))
    return sock


def overlay(image, status: dict, notes, t: float) -> None:
    """Flight data the real drone never shows, drawn small in the corner of the simulated feed."""
    lines = [
        f"{status['state']}  {status['height']:.1f} m  {status['speed']:.1f} m/s",
        f"battery {status['battery'] * 100:.0f}%  {status['voltage']:.2f} V",
        f"link {'ok' if status['link'] else 'LOST'}  flow {'ok' if status['flow'] else 'none'}",
    ]
    lines += [message for at, message in notes if t - at < 4.0][-2:]
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = max(cv2.getTextSize(line, font, 0.6, 1)[0][0] for line in lines)
    box = image[6:14 + 24 * len(lines), 6:min(image.shape[1], 22 + width)]
    box[:] = box // 3
    for i, line in enumerate(lines):
        cv2.putText(image, line, (14, 28 + 24 * i), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.headers_cors()
            self.end_headers()
            self.wfile.write(data)

        def headers_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.headers_cors()
            self.end_headers()

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/state"):
                self.reply(200, server.state())
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                path = self.path.rstrip("/")
                if path == "/reset":
                    server.reset(**{k: body[k] for k in ("scene", "firmware", "battery", "seed", "wind") if k in body})
                elif path == "/wind":
                    server.set_wind(**{k: body[k] for k in ("speed", "direction", "gust") if k in body})
                else:
                    self.reply(404, {"error": "not found"})
                    return
            except (ValueError, TypeError) as e:
                self.reply(400, {"error": str(e)})
                return
            self.reply(200, server.state())

    return Handler
