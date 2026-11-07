import math
import unittest
from dataclasses import asdict, replace

import cv2
import numpy as np

from control import trajectory
from sim.calibrate import Run, fit, observe, replay
from sim import video
from sim.camera import Camera, Pose, VideoEncoder
from sim.config import FlightTuning
from sim.pilot import FLY, Pilot
from sim.scenes import SCENES

DEFAULT = FlightTuning()
REAL = replace(DEFAULT, takeoff_height_m=0.95, max_speed_mps=3.4, max_climb_mps=1.1, max_yaw_rate_rad_s=1.8)
PROGRAMS = {
    "hover": ("takeoff\nhold 3\nland", ("hover_height",)),
    "climb": ("takeoff\nhold 2\nclimb 0.5\nmove 0 0 1\nhold 2\nland", ("final_height",)),
    "forward": ("takeoff\nspeed 1\nmove 2.5 0\nhold 1\nland", ("landed_x", "landed_y")),
    "spin": ("takeoff\nrate 90\nturn 420\nland", ("yaw_deg",)),
}


def flown(name: str) -> Run:
    """A calibration program planned on the default model, flown by a drone that differs from it."""
    text, keys = PROGRAMS[name]
    p, _ = trajectory.refine(trajectory.plan(text, DEFAULT), tuning=DEFAULT)
    commands, fired, t = [], set(), 0.0
    while t < p.duration:
        index, seg = p.find(t)
        action = seg.action if seg.action and index not in fired else None
        fired.add(index)
        commands.append({"t": round(t, 4), "i": index, "sticks": seg.sticks, "action": action})
        t += trajectory.TICK
    run = Run(name, [asdict(s) for s in p.segments], commands, {})
    real = observe(run, replay(run, REAL, seed=7))
    run.measured = {k: real[k] for k in keys}
    return run


class CalibrationTests(unittest.TestCase):
    def test_fitting_recovers_how_the_real_drone_differs(self):
        runs = [flown(name) for name in PROGRAMS]
        tuned = fit(runs, DEFAULT, log=lambda _: None)
        for name in ("takeoff_height_m", "max_speed_mps", "max_climb_mps", "max_yaw_rate_rad_s"):
            self.assertAlmostEqual(getattr(tuned, name), getattr(REAL, name), delta=getattr(REAL, name) * 0.07, msg=name)

    def test_replay_of_a_plan_matches_its_prediction(self):
        run = flown("forward")
        mine = observe(run, replay(run, DEFAULT))
        p, prediction = trajectory.refine(trajectory.plan(PROGRAMS["forward"][0], DEFAULT), tuning=DEFAULT)
        self.assertAlmostEqual(mine["landed_x"], prediction["track"][-1][1], delta=0.1)


def filmed_spin(scene_name: str, fov: float):
    """JPEG frames of the simulated drone spinning, as the backend would record them, and its true headings."""
    scene = SCENES[scene_name]()
    pilot = Pilot(scene=scene)
    pilot.fly(2.5)
    pilot.hold(FLY, 0.3)
    pilot.fly(4.0)
    camera, encoder = Camera(scene, fov=math.radians(fov)), VideoEncoder(seed=2)
    images, times, headings = [], [], []
    for i in range(int(7.5 * 17.6)):
        d = pilot.sim.drone
        _, jpeg = encoder.encode(camera.render(Pose(d.x, d.y, d.z, d.rotation())), pilot.sim.t)
        images.append(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        times.append(pilot.sim.t)
        headings.append(math.degrees(d.yaw()))
        pilot.fly(1 / 17.6, yaw=60 if i > 17 else 128)
    return images, times, headings


class VideoTests(unittest.TestCase):
    def check(self, scene_name, fov):
        images, times, headings = filmed_spin(scene_name, fov)
        result = video.spin(images, times)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["fov_deg"], fov, delta=4.0)
        start = times.index(result["start"])
        truth = sum(trajectory.wrap(b - a) for a, b in zip(headings[start:], headings[start + 1:]))
        self.assertAlmostEqual(result["yaw_deg"], truth, delta=abs(truth) * 0.03)

    def test_spin_in_a_room_gives_the_lens_and_the_turn(self):
        self.check("room", 90.0)

    def test_plain_walls_and_matching_ends_do_not_fool_it(self):
        self.check("garage", 75.0)


if __name__ == "__main__":
    unittest.main()
