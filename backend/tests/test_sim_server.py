import json
import socket
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from models.s2x_rc import S2xDroneModel
from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from protocols.s2x_video_protocol import S2xVideoProtocolAdapter
from services.flight_controller import FlightController
from sim.server import Server

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "s20_capture.json").read_text())


class ServerTests(unittest.TestCase):
    """The real turbodrone S2x stack against the simulator over loopback sockets."""

    def setUp(self):
        self.server = Server(control_port=0, tcp_port=0, video_source_port=0, api_port=0, log=lambda _: None)
        self.server.start()
        self.addCleanup(self.server.stop)

    def api(self, path="/state", body=None):
        url = f"http://127.0.0.1:{self.server.api_port}{path}"
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=2) as reply:
            return json.load(reply)

    def wait(self, check, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if check(self.api()):
                return True
            time.sleep(0.05)
        return False

    def test_video_adapter_receives_full_frames(self):
        video = S2xVideoProtocolAdapter("127.0.0.1", self.server.control_port, 0)
        self.addCleanup(video.stop)
        self.server.client_video_port = video.get_receiver_socket().getsockname()[1]
        video.start()
        frames = []
        end = time.monotonic() + 6
        while len(frames) < 5 and time.monotonic() < end:
            frame = video.get_frame(timeout=0.5)
            if frame is not None:
                frames.append(frame)
        self.assertEqual(len(frames), 5)
        image = cv2.imdecode(np.frombuffer(frames[-1].data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((image.shape[1], image.shape[0]), tuple(CAPTURE["video"]["jpeg_size"]))

    def test_backend_takes_off_and_stops_the_simulated_drone(self):
        model = S2xDroneModel()
        rc = S2xRCProtocolAdapter("127.0.0.1", self.server.control_port)
        self.addCleanup(rc.sock.close)
        flight = FlightController(model, rc, 80.0)
        flight.start()
        self.addCleanup(flight.stop)
        self.assertTrue(self.wait(lambda s: s["link"] and s["leds"] == "solid", 4))
        model.takeoff()
        self.assertTrue(self.wait(lambda s: s["state"] == "FLYING", 5))
        self.assertAlmostEqual(self.api()["height"], 1.2, delta=0.25)
        model.emergency_stop()
        self.assertTrue(self.wait(lambda s: s["state"] == "READY", 2))

    def test_board_answers_with_the_captured_replies(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        sock.settimeout(2)
        sock.sendto(b"\x0f", ("127.0.0.1", self.server.control_port))
        self.assertEqual(sock.recv(64), bytes.fromhex(CAPTURE["udp_8080_replies"]["0f"]))
        self.assertEqual(sock.recv(64)[:4], bytes.fromhex(CAPTURE["status_reply_prefix"]))

    def test_tcp_heartbeat_is_answered(self):
        with socket.create_connection(("127.0.0.1", self.server.tcp_port), timeout=2) as conn:
            conn.sendall(bytes.fromhex(CAPTURE["tcp_8888_heartbeat"]))
            self.assertEqual(conn.recv(64), bytes.fromhex(CAPTURE["tcp_8888_reply"]))

    def test_api_resets_into_another_scene_and_rejects_unknown_ones(self):
        state = self.api("/reset", {"scene": "gym", "battery": 0.5})
        self.assertEqual(state["scene"], "gym")
        self.assertAlmostEqual(state["battery"], 0.5, delta=0.01)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.api("/reset", {"scene": "moon"})
        self.assertEqual(caught.exception.code, 400)

    def test_api_sets_the_wind(self):
        state = self.api("/wind", {"speed": 3.0, "direction": 45})
        self.assertEqual(state["wind"]["speed"], 3.0)
        self.assertEqual(state["wind"]["direction"], 45.0)


if __name__ == "__main__":
    unittest.main()
