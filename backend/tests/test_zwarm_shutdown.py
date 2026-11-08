"""Quitting with drones in the air, and the routes around the operator. Everything here is the simulator."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from swarm.coordinator import Phase
from swarm.server import sim_runtime
from tests.test_zwarm_e2e import Flight
from zwarm.app import EventLog, Readiness, fly

BACKEND = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QuittingInTheAir(unittest.TestCase):
    """A real `python -m zwarm sim` process and real signals."""

    def start_flying(self):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log")
        self.addCleanup(self.log.close)
        self.process = subprocess.Popen([sys.executable, "-m", "zwarm", "sim", "--port", str(self.port)], cwd=BACKEND, stdout=self.log, stderr=subprocess.STDOUT)
        self.addCleanup(self.process.kill)
        self.wait_for(lambda: self.get("/zwarm/state"), 30.0, "the backend to start")
        self.wait_for(lambda: self.post("/takeoff") == 200, 10.0, "takeoff to be accepted")
        self.wait_for(lambda: all(d["phase"] == "air" for d in self.get("/zwarm/state")["drones"]), 40.0, "every drone in the air")

    def wait_for(self, condition, seconds, what):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                if condition():
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        self.fail(f"gave up waiting for {what}")

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=2.0) as reply:
            return json.load(reply)

    def post(self, path):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", b"{}", {"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=2.0) as reply:
                return reply.status
        except urllib.error.HTTPError as error:
            return error.code

    def logged(self):
        self.log.flush()
        return Path(self.log.name).read_text()

    def test_a_second_ctrl_c_doesnt_skip_the_landing(self):
        self.start_flying()
        self.process.send_signal(signal.SIGINT)
        time.sleep(0.05)
        self.process.send_signal(signal.SIGINT)
        self.assertEqual(self.process.wait(40.0), 0)
        self.assertIn("landing them where they are", self.logged())

    def test_closing_the_terminal_lands_too(self):
        self.start_flying()
        self.process.send_signal(signal.SIGHUP)
        self.assertEqual(self.process.wait(40.0), 0)
        self.assertIn("landing them where they are", self.logged())

    def test_e_stop_gets_through_during_the_landing(self):
        self.start_flying()
        self.process.send_signal(signal.SIGINT)
        time.sleep(1.0)
        self.assertEqual(self.post("/estop"), 200)
        self.assertEqual(self.process.wait(40.0), 0)


class ShutDownDuringATakeoff(unittest.TestCase):
    def test_a_drone_still_climbing_is_landed(self):
        runtime = sim_runtime(seed=1)
        runtime.start()
        self.addCleanup(runtime.shut_down)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with runtime.lock:
                if not runtime.co.takeoff():
                    break
            time.sleep(0.1)
        time.sleep(0.5)
        self.assertEqual(runtime.co.drones["1"].phase, Phase.TAKEOFF)
        runtime.shut_down()
        self.assertTrue(runtime.co.all_phase(Phase.GROUND), [d.phase for d in runtime.co.drones.values()])
        self.assertEqual([name for _, name in runtime.co.drones["1"].link.commands], ["takeoff", "land"])
        runtime.shut_down()  # only the first call does anything


class WhoCanUseTheRoutes(unittest.TestCase):
    def test_a_public_page_cant_act_and_the_console_and_curl_can(self):
        f = Flight(self)
        for origin in ("https://example.com", "http://localhost.evil.com", "null"):
            self.assertEqual(f.http.post("/hold", headers={"Origin": origin}).status_code, 403, origin)
            self.assertEqual(f.http.post("/takeoff", headers={"Origin": origin}).status_code, 403, origin)
        self.assertFalse(f.zw.fleet.drones["1"].sim.fc.state == "FLYING")
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.1.20:5173", "http://andrews-mac.local:5173"):
            self.assertEqual(f.http.post("/hold", headers={"Origin": origin}).status_code, 200, origin)
        self.assertEqual(f.http.post("/hold").status_code, 200)  # curl and the runbook's emergency commands send no Origin
        self.assertEqual(f.http.get("/state", headers={"Origin": "https://example.com"}).headers.get("access-control-allow-origin"), None)


class BuildMapRoute(unittest.TestCase):
    def test_nothing_captured_gets_a_plain_refusal(self):
        f = Flight(self)
        answer = f.http.post("/zwarm/map/build")
        self.assertEqual(answer.status_code, 409)
        self.assertIn("no photos", answer.json()["detail"])

    def test_it_cant_cut_a_running_capture_short_or_throw_away_a_finished_map(self):
        f = Flight(self)
        for state, words in (("capturing", "still being captured"), ("done", "already built")):
            f.capture.state, f.capture.name = state, "m"
            answer = f.http.post("/zwarm/map/build")
            self.assertEqual(answer.status_code, 409, state)
            self.assertIn(words, answer.json()["detail"])
        self.assertNotIn("build", f.capture.calls)


class RestartTests(unittest.TestCase):
    def test_event_numbers_keep_rising_across_a_restart(self):
        with mock.patch("zwarm.app.time.time", return_value=1000.0):
            first = EventLog()
        for _ in range(500):
            first.add("mission", "x")
        with mock.patch("zwarm.app.time.time", return_value=1002.0):  # a restart takes longer than this
            second = EventLog()
        second.add("mission", "y")
        self.assertGreater(second.since(first.count)[0]["n"], first.count)  # so a tab left open sees it, whatever it saw before

    def test_fly_keeps_maps_where_a_restart_will_find_them(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        swarm_file = Path(d.name) / "swarm.json"
        swarm_file.write_text("{}")
        config = mock.Mock()
        config.drones = []
        config.backends.return_value = {}
        with mock.patch("zwarm.app.check_ready", return_value=Readiness(config=config)), mock.patch("zwarm.app.real_runtime"), \
                mock.patch("swarm.yolo_eyes.YoloEyes"), mock.patch("zwarm.app.compose") as compose:
            fly(swarm_file, Path(d.name))
        self.assertEqual(compose.call_args.kwargs["maps_path"], Path(d.name) / "maps")


if __name__ == "__main__":
    unittest.main()
