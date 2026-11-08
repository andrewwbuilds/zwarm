"""The read-only preflight, the example swarm file, printable tags, and the drone's link byte.

Frames come from the simulator's floor camera. No drone, no model and no internet.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from sim.scene import Box, Scene
from swarm.frames import Frame
from swarm.server import SwarmFile
from swarm.sim_anchors import FloorCamera, VideoConditions
from swarm.tracking import AnchorReader, Lens, anchors_in_view, load_anchors, solve_fix
from zwarm import markers, preflight

BACKEND = Path(__file__).resolve().parents[1]
EXAMPLE = BACKEND / "zwarm" / "real.example.json"
LEVEL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
GROUNDED = {"drone_type": "s2x", "commands": {"takeoff": True}, "busy": False, "airborne": False, "link_quality": 88}


def no_setup(config, need_model, section) -> None:
    """Skips loading YOLO, which the e2e tests cover."""


class Answer:
    model = "nvidia/some-model"


class FakeClient:
    def _api_key(self):
        return "nvapi-test-secret-value"

    def complete(self, *args, **kwargs):
        return Answer()


def render_jpeg(config, drone, z=1.0, tilt=None, seed=0) -> bytes:
    scene = Scene(name="floor", bounds=Box((-6.0, -6.0, 0.0), (6.0, 6.0, 3.0), "wall"))
    camera = FloorCamera(scene, config.anchors, drone.camera_tilt_deg if tilt is None else tilt, VideoConditions(render_scale=1.0), seed=seed)
    gray = camera.render(drone.pad[0], drone.pad[1], z, LEVEL)
    ok, jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return jpeg.tobytes()


class FakeFrames:
    """The same picture again and again, numbered as a 15 fps stream."""

    def __init__(self, jpegs: dict, stream_fps: float = 15.0):
        self.jpegs, self.fps = jpegs, stream_fps
        self.started = time.monotonic()

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self, drone_id):
        jpeg = self.jpegs.get(drone_id)
        if jpeg is None:
            return None
        now = time.monotonic()
        number = int((now - self.started) * self.fps) + 1
        return Frame(drone_id, self.started + (number - 1) / self.fps, jpeg, number)


def run(config_path, jpegs, fetch=lambda url: GROUNDED, **kwargs):
    lines: list[str] = []
    kwargs.setdefault("seconds", 1.2)
    kwargs.setdefault("client", FakeClient())
    kwargs.setdefault("setup", no_setup)
    code = preflight.run(config_path, out=lines.append, make_frames=lambda backends: FakeFrames(jpegs), fetch=fetch, **kwargs)
    return code, "\n".join(lines)


class Rendered(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SwarmFile.load(EXAMPLE)
        cls.lifted = {d.id: render_jpeg(cls.config, d, seed=i) for i, d in enumerate(cls.config.drones)}


class TestVerdicts(Rendered):
    def test_three_lifted_drones_pass(self):
        code, text = run(EXAMPLE, self.lifted)
        self.assertEqual(code, 0, text)
        self.assertIn("RESULT: READY", text)
        for drone in ("1", "2", "3"):
            self.assertIn(f"Drone {drone}: PASS", text)
        self.assertIn("1184 x 656", text)
        self.assertIn("furthest read", text)
        self.assertRegex(text, r"\b88\b")
        self.assertIn("presumed on the ground", text)

    def test_pose_is_near_the_pad_it_was_rendered_over(self):
        code, text = run(EXAMPLE, {k: v for k, v in self.lifted.items()})
        self.assertRegex(text, r"\(-2\.\d\d, 1\.\d\d\), 0\.9\d m up|\(-2\.\d\d, 1\.\d\d\), 1\.0\d m up", text)

    def test_backend_down_fails_and_names_the_drone(self):
        def down(url):
            if "8002" in url:
                raise OSError("connection refused")
            return GROUNDED

        code, text = run(EXAMPLE, self.lifted, fetch=down)
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("Drone 2: FAIL", text)
        self.assertIn("Drone 1: PASS", text)
        self.assertIn("RESULT: NOT READY", text)

    def test_no_video_fails(self):
        code, text = run(EXAMPLE, {"1": self.lifted["1"], "3": self.lifted["3"]})
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("Drone 2: FAIL", text)
        self.assertIn("no frame from /mjpeg", text)

    def test_presumed_airborne_fails_and_says_how_to_clear_it(self):
        code, text = run(EXAMPLE, self.lifted, fetch=lambda url: {**GROUNDED, "airborne": True})
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("IN THE AIR", text)
        self.assertIn("mark landed", text)

    def test_older_pi_without_the_link_byte_still_passes(self):
        old = {k: v for k, v in GROUNDED.items() if k != "link_quality"}
        code, text = run(EXAMPLE, self.lifted, fetch=lambda url: old)
        self.assertEqual(code, 0, text)
        self.assertIn("older code", text)

    def test_quiet_drone_warns_but_passes(self):
        code, text = run(EXAMPLE, self.lifted, fetch=lambda url: {**GROUNDED, "link_quality": None})
        self.assertEqual(code, 0, text)
        self.assertIn("no reply from the drone", text)

    def test_drones_on_the_floor_fail_unless_on_pad_is_given(self):
        floor = {d.id: render_jpeg(self.config, d, z=0.05) for d in self.config.drones}
        code, text = run(EXAMPLE, floor)
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("Was it lifted?", text)
        code, text = run(EXAMPLE, floor, on_pad=True, seconds=0.8)
        self.assertEqual(code, 0, text)
        self.assertIn("SKIP  tags", text)

    def test_pose_far_from_the_pad_fails(self):
        data = json.loads(EXAMPLE.read_text())
        data["drones"][0]["pad"] = [-2.5, 2.2]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "moved.json"
            path.write_text(json.dumps(data))
            code, text = run(path, self.lifted)
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("from its pad", text)
        self.assertIn("Drone 1: FAIL", text)

    def test_wrong_tag_size_fails_and_points_at_it(self):
        # Corners that don't fit the tag grid mean no pose gets believed, which is the tracker's own rule.
        data = json.loads(EXAMPLE.read_text())
        data["anchors"]["white_square_edge_m"] = 0.1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "big.json"
            path.write_text(json.dumps(data))
            code, text = run(path, self.lifted)
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("no pose believed", text)
        self.assertIn("white_square_edge_m", text)

    def test_camera_angle_is_measured(self):
        steep = render_jpeg(self.config, self.config.drones[1], tilt=30.0)
        _, text = run(EXAMPLE, {**self.lifted, "2": steep})
        self.assertRegex(text, r"camera angle\s+(29|30|31|32) degrees below level")
        self.assertIn("Reset the camera", text)

    def test_key_never_printed(self):
        secret = "nvapi-test-secret-value"

        class Broken(FakeClient):
            def complete(self, *args, **kwargs):
                raise RuntimeError(f"401 for {secret}")

        code, text = run(EXAMPLE, self.lifted, client=Broken())
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("nemotron", text)
        self.assertNotIn(secret, text)

    def test_no_model_skips_the_model(self):
        client = mock.MagicMock()
        code, text = run(EXAMPLE, self.lifted, need_model=False, client=client)
        self.assertEqual(code, 0, text)
        client.complete.assert_not_called()

    def test_missing_file_says_so(self):
        code, text = run("/nonexistent/real.json", {})
        self.assertEqual(code, preflight.NOT_READY)
        self.assertIn("no swarm file", text)

    def test_saves_the_sharpest_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = run(EXAMPLE, self.lifted, save_frames=folder)
            self.assertEqual(code, 0, text)
            self.assertGreater((Path(folder) / "drone-2.jpg").stat().st_size, 1000)

    def test_only_drones(self):
        path = preflight.only_drones(EXAMPLE, "1,3")
        self.assertEqual([d.id for d in SwarmFile.load(path).drones], ["1", "3"])
        self.assertEqual(preflight.only_drones(EXAMPLE, None), str(EXAMPLE))
        with self.assertRaises(ValueError):
            preflight.only_drones(EXAMPLE, "9")


class Recording(BaseHTTPRequestHandler):
    """A backend that writes down everything it's sent."""

    seen: list = []
    jpeg = b""

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).seen.append(("GET", self.path))
        if self.path == "/capabilities":
            body = json.dumps(GROUNDED).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + type(self).jpeg + b"\r\n")
                    time.sleep(1 / 15)
            except OSError:
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        type(self).seen.append(("POST", self.path))
        self.send_error(404)

    do_PUT = do_DELETE = do_PATCH = do_POST


class TestOnlyReads(Rendered):
    def test_only_reads_capabilities_and_video(self):
        Recording.seen = []
        Recording.jpeg = self.lifted["1"]
        server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            data = json.loads(EXAMPLE.read_text())
            data["drones"] = data["drones"][:1]
            data["drones"][0]["backend"] = f"http://127.0.0.1:{server.server_address[1]}"
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "one.json"
                path.write_text(json.dumps(data))
                lines: list[str] = []
                code = preflight.run(path, seconds=1.5, setup=no_setup, client=FakeClient(), out=lines.append)
                preflight.watch(path, out=lines.append, seconds=0.6)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue(Recording.seen)
        self.assertEqual({entry for entry in Recording.seen}, {("GET", "/capabilities"), ("GET", "/mjpeg")})


class TestExampleFile(unittest.TestCase):
    def test_loads_with_what_fly_reads(self):
        config = SwarmFile.load(EXAMPLE)
        self.assertEqual([d.backend for d in config.drones], [f"http://localhost:800{i}" for i in (1, 2, 3)])
        self.assertTrue(all(d.camera_tilt_deg == 45 for d in config.drones))
        self.assertEqual(len(config.anchors), 25)
        self.assertEqual(config.limits.max_speed, 0.3)
        raw = json.loads(EXAMPLE.read_text())
        self.assertIn("north_yaw_deg", raw)
        self.assertIn("flight_limit_s", raw)

    def test_every_pad_can_see_a_tag_at_one_metre(self):
        config = SwarmFile.load(EXAMPLE)
        for d in config.drones:
            self.assertTrue(anchors_in_view(config.anchors, Lens.from_fov(), np.radians(45.0), (d.pad[0], d.pad[1], 1.0), 0.0), d.id)


class TestMarkers(unittest.TestCase):
    def test_a_printed_page_reads_back_as_its_own_id(self):
        with tempfile.TemporaryDirectory() as folder:
            pages, section, real_m = markers.write(folder, only=[7])
            self.assertEqual([p.name for p in pages], ["tag-07.png"])
            self.assertAlmostEqual(real_m, 0.138, delta=0.001)
            raw = pages[0].read_bytes()
            self.assertIn(b"pHYs", raw[:64])
            sheet = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(sheet.shape, (markers.PAGE[1], markers.PAGE[0]))
            anchors = load_anchors(json.loads(section.read_text())["anchors"])
            self.assertEqual(len(anchors), 25)
            seen = AnchorReader(anchors).find(cv2.resize(sheet, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
            self.assertEqual([anchor.id for anchor, _ in seen], ["tag:7"])

    def test_refuses_a_tag_too_big_for_the_page(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            markers.write(folder, width_m=0.5, only=[0])


class TestLinkQuality(unittest.TestCase):
    def setUp(self):
        self.drone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.drone.bind(("127.0.0.1", 0))
        self.drone.settimeout(1.0)
        self.adapter = S2xRCProtocolAdapter("127.0.0.1", self.drone.getsockname()[1])
        self.addCleanup(self.drone.close)
        self.addCleanup(self.adapter.sock.close)

    def answer(self, *values: int) -> None:
        for value in values:
            self.adapter.send_control_packet(b"x" * 20)
            _, source = self.drone.recvfrom(64)
            self.drone.sendto(b"\xff\x53\x54\x08" + bytes([value]), source)

    def read(self):
        deadline = time.monotonic() + 1.0
        value = self.adapter.read_link_quality()
        while value is None and time.monotonic() < deadline:
            time.sleep(0.01)
            value = self.adapter.read_link_quality()
        return value

    def test_nothing_yet_returns_none_without_waiting(self):
        started = time.monotonic()
        self.assertIsNone(self.adapter.read_link_quality())
        self.assertLess(time.monotonic() - started, 0.05)

    def test_reads_the_newest_byte(self):
        self.answer(70, 71, 85)
        time.sleep(0.05)
        self.assertEqual(self.read(), 85)

    def test_goes_quiet(self):
        self.answer(80)
        self.assertEqual(self.read(), 80)
        self.adapter.link_quality_at -= 10.0
        self.assertIsNone(self.adapter.read_link_quality())

    def test_ignores_other_datagrams(self):
        self.adapter.send_control_packet(b"x" * 20)
        _, source = self.drone.recvfrom(64)
        self.drone.sendto(b"\xff\x53\x54\x33\x16", source)
        time.sleep(0.05)
        self.assertIsNone(self.adapter.read_link_quality())

    def test_a_closed_socket_doesnt_raise(self):
        self.adapter.sock.close()
        self.assertIsNone(self.adapter.read_link_quality())


class TestCameraAngle(Rendered):
    def test_reads_back_what_was_rendered(self):
        d = self.config.drones[0]
        for tilt in (35.0, 45.0, 55.0):
            scene = Scene(name="floor", bounds=Box((-6.0, -6.0, 0.0), (6.0, 6.0, 3.0), "wall"))
            gray = FloorCamera(scene, self.config.anchors, tilt, VideoConditions(render_scale=1.0)).render(d.pad[0], d.pad[1], 1.0, LEVEL)
            seen = AnchorReader(self.config.anchors).find(gray)
            self.assertTrue(seen, tilt)
            fix = solve_fix(seen, Lens.from_fov(), np.radians(45.0), 0.0, 0.0, None)
            self.assertIsNotNone(fix, tilt)
            self.assertAlmostEqual(preflight.camera_down_deg(seen, Lens.from_fov(), fix), tilt, delta=3.0)


if __name__ == "__main__":
    unittest.main()
