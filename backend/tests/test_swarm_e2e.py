"""The real path from coordinator to packets: real backends over real websockets and UDP.

Only the drones and camera are simulated. Slow (about a minute), so run it on purpose:
    SWARM_E2E=1 uv run python -m unittest tests.test_swarm_e2e
"""

import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from swarm.control import Pose
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.links import BackendLink
from swarm.room import Room

BACKEND = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PADS = {"1": (-1.2, 0.0), "2": (1.2, 0.0), "3": (0.0, 1.0)}
SIM = """
import sys, time
from sim.server import Server
control, tcp, video_source, video, api = map(int, sys.argv[1:6])
server = Server(scene="room", control_port=control, tcp_port=tcp, video_source_port=video_source,
                client_video_port=video, api_port=api, hud=False, render_scale=0.1, log=lambda _: None)
server.start()
while True:
    time.sleep(1)
"""


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url, timeout=1.0):
    with OPENER.open(url, timeout=timeout) as reply:
        return json.load(reply)


class SimDroneProcess:
    """One simulated drone and its real backend, as a Pi would have."""

    def __init__(self):
        control, tcp, source, self.video, self.api, self.port = (
            free_port(socket.SOCK_DGRAM), free_port(), free_port(socket.SOCK_DGRAM),
            free_port(socket.SOCK_DGRAM), free_port(), free_port())
        self.sim = subprocess.Popen([sys.executable, "-c", SIM, str(control), str(tcp), str(source), str(self.video), str(self.api)],
                                    cwd=BACKEND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env = {**os.environ, "DRONE_TYPE": "s2x", "DRONE_IP": "127.0.0.1", "CONTROL_PORT": str(control),
               "VIDEO_PORT": str(self.video), "PLUGINS_ENABLED": "false"}
        self.backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "web_server:app", "--host", "127.0.0.1",
                                         "--port", str(self.port), "--log-level", "warning"],
                                        cwd=BACKEND, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{self.port}"

    def wait_ready(self):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                get(f"{self.url}/capabilities")
                get(f"http://127.0.0.1:{self.api}/state")
                return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("backend or simulator did not come up")

    def state(self):
        return get(f"http://127.0.0.1:{self.api}/state")

    def stop(self):
        for process in (self.backend, self.sim):
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class StatePoses:
    """Stands in for the camera: reads each simulator's true state as if a tracker had seen it."""

    def __init__(self, drones, offsets):
        self.drones, self.offsets = drones, offsets
        self.lock = threading.Lock()
        self.fresh = {}
        self.stopping = threading.Event()
        self.threads = [threading.Thread(target=self.watch, args=(k,), daemon=True) for k in drones]
        for thread in self.threads:
            thread.start()

    def watch(self, drone_id):
        dx, dy = self.offsets[drone_id]
        while not self.stopping.wait(1 / 30):
            try:
                state = self.drones[drone_id].state()
            except OSError:
                continue
            x, y, _ = state["position"]
            pose = Pose(time.monotonic(), (x + dx, y + dy, state["height"]), math.radians(state["attitude"][2]))
            with self.lock:
                self.fresh[drone_id] = pose

    def poses(self):
        with self.lock:
            fresh, self.fresh = self.fresh, {}
            return fresh

    def stop(self):
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=2)


@unittest.skipUnless(os.getenv("SWARM_E2E"), "slow: set SWARM_E2E=1 to run")
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.drones = {k: SimDroneProcess() for k in PADS}
        self.addCleanup(lambda: [d.stop() for d in self.drones.values()])
        for drone in self.drones.values():
            drone.wait_ready()
        self.links = {k: BackendLink(d.url, k) for k, d in self.drones.items()}
        for link in self.links.values():
            link.start()
        self.addCleanup(lambda: [link.stop() for link in self.links.values()])
        self.poses = StatePoses(self.drones, PADS)
        self.addCleanup(self.poses.stop)
        configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in PADS.items()]
        self.co = Coordinator(Room((-3, -2.5, 0), (3, 2.5, 2.4)), configs, self.links)

    def run_until(self, test, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            self.co.tick(now, self.poses.poses())
            if test():
                return
            time.sleep(0.05)
        faults = {d.id: (d.phase.value, d.faults) for d in self.co.drones.values()}
        self.fail(f"timed out waiting for {what}: {faults}")

    def true_positions(self):
        out = {}
        for k, drone in self.drones.items():
            state = drone.state()
            x, y, _ = state["position"]
            out[k] = (x + PADS[k][0], y + PADS[k][1], state["height"])
        return out

    def test_three_backends_take_off_swap_places_and_land(self):
        time.sleep(3.0)  # gyro calibration
        self.run_until(lambda: not self.co.preflight(time.monotonic()), 15.0, "preflight")
        self.assertEqual(self.co.takeoff(time.monotonic()), [])
        self.run_until(lambda: self.co.all_phase(Phase.AIR), 40.0, "takeoff and checks")
        self.assertEqual(self.co.goto({"1": (1.2, 0.0, 1.0), "2": (-1.2, 0.0, 1.0), "3": (0.0, -0.6, 1.0)}), [])
        closest = [9.0]

        def arrived():
            spots = list(self.true_positions().values())
            for i, a in enumerate(spots):
                for b in spots[i + 1:]:
                    closest[0] = min(closest[0], sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5)
            return self.co.all_arrived()

        self.run_until(arrived, 45.0, "the swap")
        self.assertGreater(closest[0], 0.24, "they touched")
        self.assertEqual(get(f"{self.drones['1'].url}/capabilities")["airborne"], True)
        self.co.land(time.monotonic())
        self.run_until(lambda: self.co.all_phase(Phase.GROUND), 40.0, "landing")
        # The tracker sees it down a moment before the flight controller finishes its landing.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and any(d.state()["state"] != "READY" for d in self.drones.values()):
            time.sleep(0.2)
        for drone in self.drones.values():
            self.assertEqual(drone.state()["state"], "READY")
            self.assertEqual(get(f"{drone.url}/capabilities")["airborne"], False)
        print(f"\nclosest approach through the real stack: {closest[0]:.2f} m")

    def test_losing_the_coordinator_centres_the_sticks(self):
        time.sleep(3.0)
        self.run_until(lambda: not self.co.preflight(time.monotonic()), 15.0, "preflight")
        self.co.takeoff(time.monotonic())
        self.run_until(lambda: self.co.all_phase(Phase.AIR), 40.0, "takeoff and checks")
        # A stick held forward, then silence, with the websocket still open: the coordinator hung.
        link = self.links["2"]
        for _ in range(40):
            link.send_sticks(0.0, 0.0, 0.1, 0.0, 0.3)
            time.sleep(0.05)
        moving = self.drones["2"].state()["speed"]
        self.assertGreater(moving, 0.2)
        time.sleep(2.5)
        self.assertLess(self.drones["2"].state()["speed"], 0.15, "the sticks did not expire")
        # Then the link goes away altogether while the drone is being flown.
        for _ in range(20):
            link.send_sticks(0.0, 0.0, 0.1, 0.0, 30.0)
            time.sleep(0.05)
        self.assertGreater(self.drones["2"].state()["speed"], 0.2)
        link.stop()
        time.sleep(2.5)
        self.assertLess(self.drones["2"].state()["speed"], 0.15, "disconnecting did not centre the sticks")
        self.assertEqual(self.drones["2"].state()["state"], "FLYING")


if __name__ == "__main__":
    unittest.main()
