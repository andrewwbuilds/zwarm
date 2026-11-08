"""The real detector's plumbing, with a stand-in model: no torch or weights needed."""

import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from swarm.eyes import DEFAULT_TILT, locate
from swarm.frames import Frame
from swarm.yolo_eyes import YoloEyes, detection_from_box


def jpeg(width=1280, height=720) -> bytes:
    return cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))[1].tobytes()


class FakeFrames:
    def __init__(self, frames=None, fail=None):
        self.frames, self.fail = frames or {}, fail

    def latest(self, drone_id):
        if self.fail == drone_id:
            raise OSError("stream lost")
        return self.frames.get(drone_id)


class FakeTensor(list):
    def tolist(self):
        return list(self)


class FakeModel:
    names = {0: "person", 1: "backpack", 2: "airplane"}

    def __init__(self, boxes=()):
        self.boxes = boxes
        self.calls = 0

    def predict(self, image, **kwargs):
        self.calls += 1
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=FakeTensor(b[0] for b in self.boxes), cls=FakeTensor(b[1] for b in self.boxes),
                                                      conf=FakeTensor(b[2] for b in self.boxes)))]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class BoxGeometryTests(unittest.TestCase):
    def test_the_middle_of_the_picture_is_the_camera_axis(self):
        d = detection_from_box("backpack", 0.9, (600, 320, 680, 400), 1280, 720)
        self.assertAlmostEqual(d.bearing, 0.0)
        self.assertAlmostEqual(d.elevation, 0.0)

    def test_left_and_up_are_positive(self):
        d = detection_from_box("backpack", 0.9, (100, 50, 180, 130), 1280, 720)
        self.assertGreater(d.bearing, 0.0)
        self.assertGreater(d.elevation, 0.0)
        right = detection_from_box("backpack", 0.9, (1100, 600, 1180, 680), 1280, 720)
        self.assertLess(right.bearing, 0.0)
        self.assertLess(right.elevation, 0.0)

    def test_the_edge_of_the_picture_is_half_the_field_of_view(self):
        d = detection_from_box("backpack", 0.9, (0, 340, 2, 380), 1280, 720)
        self.assertAlmostEqual(d.bearing, math.radians(45.0), delta=math.radians(0.2))

    def test_a_smaller_box_is_further_away(self):
        near = detection_from_box("person", 0.9, (600, 100, 700, 600), 1280, 720)
        far = detection_from_box("person", 0.9, (620, 200, 680, 450), 1280, 720)
        self.assertAlmostEqual(far.range / near.range, 2.0, delta=0.1)

    def test_a_floor_thing_gets_a_foot_unless_its_bottom_is_cut_off(self):
        whole = detection_from_box("backpack", 0.9, (600, 300, 680, 500), 1280, 720)
        self.assertLess(whole.foot[1], whole.elevation)
        self.assertIsNone(detection_from_box("backpack", 0.9, (600, 500, 680, 719), 1280, 720).foot)
        self.assertIsNone(detection_from_box("cup", 0.9, (600, 300, 680, 500), 1280, 720).foot)

    def test_legs_only_still_give_a_position_from_the_feet(self):
        d = detection_from_box("person", 0.9, (590, 0, 690, 360), 1280, 720)
        self.assertIsNotNone(d.foot)
        where = locate((0.0, 0.0, 1.0), 0.0, d, DEFAULT_TILT)
        self.assertAlmostEqual(where[0], 1.0 + 0.175, delta=0.01)
        self.assertAlmostEqual(where[1], 0.0, delta=0.01)


class EyesTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.frames = FakeFrames({"a": Frame("a", 100.0, jpeg(), 1)})
        self.model = FakeModel([((600, 300, 680, 500), 1, 0.8)])
        self.eyes = YoloEyes(["a", "b"], self.frames, model=self.model, clock=self.clock)

    def test_nothing_new_is_not_the_same_as_nothing_there(self):
        self.assertIsNone(self.eyes.look("a", 0.0))
        self.model.boxes = []
        self.assertTrue(self.eyes.step())
        self.assertEqual(self.eyes.look("a", 0.0), [])
        self.assertIsNone(self.eyes.look("a", 0.0), "a result is handed over once")

    def test_a_frame_is_only_looked_at_once(self):
        self.eyes.step()
        self.assertFalse(self.eyes.step())
        self.assertEqual(self.model.calls, 1)

    def test_a_detection_says_how_old_its_picture_is(self):
        self.eyes.step()
        self.clock.t += 0.1
        found = self.eyes.look("a", 0.0)
        self.assertAlmostEqual(found[0].age, 0.1 + self.eyes.latency)
        self.assertEqual(found[0].label, "backpack")

    def test_a_stale_result_is_dropped(self):
        self.eyes.step()
        self.clock.t += 2.0
        self.assertIsNone(self.eyes.look("a", 0.0))

    def test_only_things_that_can_be_in_a_room_count(self):
        self.model.boxes = [((10, 10, 200, 200), 2, 0.9)]
        self.eyes.step()
        self.assertEqual(self.eyes.look("a", 0.0), [])

    def test_a_tiny_box_is_ignored(self):
        self.model.boxes = [((10, 10, 15, 300), 0, 0.9)]
        self.eyes.step()
        self.assertEqual(self.eyes.look("a", 0.0), [])

    def test_a_missing_frame_is_just_nothing_new(self):
        self.assertFalse(YoloEyes(["z"], FakeFrames(), model=self.model).step())

    def test_a_new_frame_number_is_looked_at_again(self):
        self.eyes.step()
        self.frames.frames["a"] = Frame("a", 100.1, jpeg(), 2)
        self.assertTrue(self.eyes.step())
        self.assertEqual(self.model.calls, 2)

    def test_one_broken_camera_does_not_blind_the_others(self):
        frames = FakeFrames(self.frames.frames, fail="b")
        eyes = YoloEyes(["b", "a"], frames, model=self.model, clock=self.clock)
        self.assertTrue(eyes.step())
        self.assertIsNotNone(eyes.look("a", 0.0))
        self.assertIn("stream lost", eyes.health()["b"]["error"])

    def test_a_corrupt_frame_is_reported_and_survived(self):
        eyes = YoloEyes(["a"], FakeFrames({"a": Frame("a", 1.0, b"not a jpeg", 1)}), model=self.model, clock=self.clock)
        eyes.step()
        self.assertIn("decoded", eyes.health()["a"]["error"])

    def test_health_shows_pictures_arriving(self):
        self.eyes.step()
        health = self.eyes.health()
        self.assertEqual(health["a"]["pictures"], 1)
        self.assertIsNone(health["b"]["seconds_since_picture"])

    def test_it_runs_on_its_own_thread(self):
        self.eyes.start()
        self.addCleanup(self.eyes.stop)
        for _ in range(200):
            if self.model.calls:
                break
            cv2.waitKey(5)
        self.assertGreaterEqual(self.model.calls, 1)


if __name__ == "__main__":
    unittest.main()
