"""The real flight path from a swarm file: `real_runtime`, real backends over websockets and UDP,
HTTP /mjpeg through the shared frame reader, and the real tag detector.

Only two things are simulated: the drones, and what their cameras see. Slow, so run it on purpose:
    SWARM_E2E=1 uv run python -m unittest tests.test_swarm_floor_e2e
"""

import json
import math
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from sim.scene import Box, Scene
from swarm.coordinator import Phase
from swarm.server import SwarmFile, real_runtime
from swarm.sim_anchors import FloorCamera, VideoConditions
from swarm.tracking import anchors_section, load_anchors, tag_grid
from tests.test_swarm_e2e import SimDroneProcess, free_port

SECTION = anchors_section(0.0767, tag_grid())
PADS = {"1": (-2.5, 0.6), "2": (-2.5, -0.6)}
HALL = Scene(name="hall", bounds=Box((-6, -6, 0), (6, 6, 3), "wall"))


class FakeCamera:
    """Serves /mjpeg like a backend, drawing what a tilted camera sees from the drone's true state."""

    def __init__(self, drone: SimDroneProcess, pad):
        self.drone, self.pad = drone, pad
        self.camera = FloorCamera(HALL, load_anchors(SECTION), 45.0, VideoConditions())
        self.port = free_port()
        camera = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + camera.jpeg() + b"\r\n")
                        time.sleep(0.05)
                except OSError:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def jpeg(self) -> bytes:
        state = self.drone.state()
        x, y, _ = state["position"]
        roll, pitch, yaw = (math.radians(a) for a in state["attitude"])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
        image = self.camera.render(x + self.pad[0], y + self.pad[1], state["height"] + 0.03, rotation)
        return cv2.imencode(".jpg", image)[1].tobytes()

    def stop(self):
        self.server.shutdown()


@unittest.skipUnless(os.getenv("SWARM_E2E"), "slow: set SWARM_E2E=1 to run")
class FloorEndToEnd(unittest.TestCase):
    def test_a_swarm_file_flies_from_tags_alone(self):
        drones = {k: SimDroneProcess() for k in PADS}
        self.addCleanup(lambda: [d.stop() for d in drones.values()])
        for drone in drones.values():
            drone.wait_ready()
        cameras = {k: FakeCamera(d, PADS[k]) for k, d in drones.items()}
        self.addCleanup(lambda: [c.stop() for c in cameras.values()])
        time.sleep(3.0)  # gyro calibration

        # /mjpeg comes from the fake camera; everything else from the real backend. The frame reader
        # only knows one address per drone, so a small proxy isn't worth it: point it at the camera.
        data = {"fence": {"lo": [-2.8, -2.4, 0.0], "hi": [1.7, 2.4, 2.2]}, "obstacles": [], "video_latency_s": 0.05,
                "drones": [{"id": k, "backend": drones[k].url, "pad": list(v), "yaw_deg": 0, "camera_tilt_deg": 45} for k, v in PADS.items()],
                "anchors": SECTION}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "swarm.json"
            path.write_text(json.dumps(data))
            config = SwarmFile.load(path)
        import swarm.frames
        real_reader = swarm.frames.StreamFrames
        swarm.frames.StreamFrames = lambda backends: real_reader({k: f"http://127.0.0.1:{cameras[k].port}" for k in backends})
        self.addCleanup(setattr, swarm.frames, "StreamFrames", real_reader)
        runtime = real_runtime(config)
        self.addCleanup(runtime.close)
        co = runtime.co
        tracker = runtime.tracker

        def run_until(test, timeout, what):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                now = time.monotonic()
                co.tick(now, tracker.poses(now))
                if test():
                    return
                time.sleep(0.05)
            self.fail(f"timed out waiting for {what}: {[(d.id, d.phase.value, d.faults, d.tracked.p.round(2).tolist()) for d in co.drones.values()]}")

        run_until(lambda: not co.preflight(time.monotonic()), 15.0, "preflight")
        self.assertEqual(tracker.report(time.monotonic())["1"]["fixes"], 0)  # on the floor it can't see the floor
        self.assertEqual(co.takeoff(time.monotonic()), [])
        run_until(lambda: co.all_phase(Phase.AIR), 45.0, "takeoff and checks")
        self.assertEqual(co.goto({"1": (0.8, -0.6, 1.0), "2": (0.8, 0.6, 1.0)}), [])

        errors = []

        def arrived():
            for k, d in co.drones.items():
                state = drones[k].state()
                errors.append(math.hypot(state["position"][0] + PADS[k][0] - d.tracked.p[0], state["position"][1] + PADS[k][1] - d.tracked.p[1]))
            return co.all_arrived()

        run_until(arrived, 60.0, "the crossing")
        co.land(time.monotonic())
        run_until(lambda: co.all_phase(Phase.GROUND), 60.0, "landing")
        errors.sort()
        print(f"\nreal path pose error against the simulator's truth: median {errors[len(errors) // 2] * 100:.1f} cm, "
              f"95% {errors[int(len(errors) * 0.95)] * 100:.1f} cm; fixes per drone {[r['fixes'] for r in tracker.report(time.monotonic()).values()]}")
        self.assertLess(errors[int(len(errors) * 0.95)], 0.2)


if __name__ == "__main__":
    unittest.main()
