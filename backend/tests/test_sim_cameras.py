"""Simulated camera feeds and the demo room. Everything here is simulator truth, never drone results."""

import math
import statistics
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from sim.camera import HEIGHT, SIZES, WIDTH, Pose
from sim.scene import Box, Scene
from swarm.coordinator import Coordinator, DroneConfig
from swarm.eyes import SimEyes
from swarm.sim_cameras import (
    DecalCamera,
    DecalScene,
    SimFrames,
    StillFleet,
    anchor_decals,
    demo_world,
    flat_decal,
    level,
    paper_width,
    side_decal,
    tilted,
    view_of,
)
from swarm.sim_fleet import SimFleet
from swarm.tracking import Anchor, AnchorReader, DroneOptics, InsideOutTracker, Lens, anchors_in_view, load_anchors, solve_fix

TILT = math.radians(45.0)
ROOM = Box((-3.0, -2.5, 0.0), (3.0, 2.5, 2.6), "wall")
WORLD = None


def setUpModule():
    global WORLD
    WORLD = demo_world()


def world() -> "demo_world":
    return WORLD


def color(bgr, alpha=1.0, pixels=32) -> np.ndarray:
    return np.dstack([np.full((pixels, pixels), v, np.float32) for v in (*bgr, alpha)])


def gray(frame) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)


def picture(scene, decals, x, y, z, yaw=0.0) -> np.ndarray:
    camera = DecalCamera(scene, decals, render_scale=0.54, seed=3)
    return camera.shade(Pose(x, y, z, tilted(level(yaw), TILT)))


def share_changed(scene, decals, x, y, z, yaw=0.0, threshold=0.05) -> float:
    """How much of the picture a set of decals changes. The grain isn't in shade(), so nothing else differs."""
    bare, painted = picture(scene, (), x, y, z, yaw), picture(scene, decals, x, y, z, yaw)
    return float(np.mean(np.abs(painted - bare).max(axis=2) > threshold))


class TiltTests(unittest.TestCase):
    def test_the_middle_of_the_picture_is_45_degrees_down(self):
        camera = DecalCamera(Scene("plain", bounds=ROOM), render_scale=0.54)
        for yaw in (0.0, 1.1, -2.4):
            dist, _, _ = camera.trace(Pose(0.0, 0.0, 1.0, tilted(level(yaw), TILT)))
            self.assertAlmostEqual(float(dist[camera.height // 2, camera.width // 2]), math.sqrt(2.0), delta=0.02)

    def test_the_camera_can_not_see_above_the_horizon(self):
        camera = DecalCamera(Scene("plain", bounds=ROOM), render_scale=0.54)
        _, objects, _ = camera.trace(Pose(0.0, 0.0, 1.0, tilted(level(0.0), TILT)))
        self.assertFalse((objects == DecalCamera.CEILING).any())
        self.assertTrue((objects[-1] == DecalCamera.FLOOR).all())
        # The top of the view is about 16 degrees below level, so the floor ends about 3.5 m ahead.
        dist, _, _ = camera.trace(Pose(0.0, 0.0, 2.2, tilted(level(0.0), TILT)))
        self.assertTrue((objects[0] != DecalCamera.CEILING).all())
        self.assertGreater(float(dist[0, camera.width // 2]), 0.0)

    def test_a_pitched_body_tips_the_picture(self):
        # Nose 10 degrees down: the middle ray is 55 degrees below level and hits the floor closer.
        camera = DecalCamera(Scene("plain", bounds=ROOM), render_scale=0.54)
        c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
        pitched = (c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
        dist, _, _ = camera.trace(Pose(0.0, 0.0, 1.0, tilted(pitched, TILT)))
        self.assertAlmostEqual(float(dist[camera.height // 2, camera.width // 2]), 1 / math.sin(math.radians(55)), delta=0.02)


class DecalTests(unittest.TestCase):
    def test_a_floor_decal_is_where_it_says_and_nowhere_else(self):
        scene = Scene("plain", bounds=ROOM, floor_material="concrete")
        decal = flat_decal(color((0.0, 0.0, 1.0)), 1.0, 0.0, 0.4, 0.4)  # hovering at 1 m, this is straight down the axis
        camera = DecalCamera(scene, (decal,), render_scale=0.54)
        out = camera.shade(Pose(0.0, 0.0, 1.0, tilted(level(0.0), TILT)))
        middle = out[camera.height // 2, camera.width // 2]
        self.assertGreater(middle[2], 0.8)
        self.assertLess(middle[0], 0.1)
        self.assertLess(out[camera.height // 2, 8, 2] - out[camera.height // 2, 8, 0], 0.2)

    def test_a_box_in_the_way_hides_a_decal_on_the_floor_behind_it(self):
        scene = Scene("blocked", bounds=ROOM, boxes=(Box((0.5, -1.0, 0.0), (0.7, 1.0, 2.0), "wood"),))
        decal = flat_decal(color((0.0, 0.0, 1.0)), 1.5, 0.0, 0.4, 0.4)
        self.assertEqual(share_changed(scene, (decal,), 0.0, 0.0, 1.0), 0.0)
        clear = Scene("clear", bounds=ROOM)
        self.assertGreater(share_changed(clear, (decal,), 0.0, 0.0, 1.0), 0.005)

    def test_a_decal_on_top_of_a_box_is_on_the_box_and_not_on_the_floor(self):
        box = Box((0.5, -0.5, 0.0), (1.5, 0.5, 0.5), "metal")
        scene = Scene("box", bounds=ROOM, boxes=(box,))
        on_top = flat_decal(color((0.0, 0.0, 1.0)), 1.0, 0.0, 0.4, 0.4, z=0.5)
        self.assertGreater(share_changed(scene, (on_top,), 0.5, 0.0, 1.0), 0.01)
        floating = flat_decal(color((0.0, 0.0, 1.0)), 1.0, 0.0, 0.4, 0.4, z=0.25)  # inside the box: no surface there
        self.assertEqual(share_changed(scene, (floating,), 0.5, 0.0, 1.0), 0.0)

    def test_a_decal_on_a_side_shows_only_from_that_side(self):
        box = Box((1.0, -0.5, 0.0), (2.0, 0.5, 0.6), "wood")
        scene = Scene("box", bounds=ROOM, boxes=(box,))
        mark = side_decal(color((0.0, 0.0, 1.0)), box, "-x", 0.0, 0.3, 0.4, 0.3)
        self.assertGreater(share_changed(scene, (mark,), 0.3, 0.0, 1.0, 0.0), 0.005)
        self.assertEqual(share_changed(scene, (mark,), 2.8, 0.0, 1.0, math.pi), 0.0)

    def test_the_side_of_a_picture_stays_the_right_way_round(self):
        # Left half red, right half blue, seen looking +x: blue must be on the viewer's right, -y.
        image = np.zeros((32, 32, 4), np.float32)
        image[..., 3] = 1.0
        image[:, :16, 2] = 1.0
        image[:, 16:, 0] = 1.0
        box = Box((1.0, -0.5, 0.0), (2.0, 0.5, 0.6), "wood")
        mark = side_decal(image, box, "-x", 0.0, 0.3, 0.6, 0.4)
        scene = Scene("box", bounds=ROOM, boxes=(box,))
        camera = DecalCamera(scene, (mark,), render_scale=0.54)
        out = camera.shade(Pose(0.3, 0.0, 1.0, tilted(level(0.0), TILT)))
        row = np.argmax(np.abs(out[:, camera.width // 2 - 20, 2] - out[:, camera.width // 2 - 20, 0]))
        left, right = out[row, camera.width // 2 - 20], out[row, camera.width // 2 + 20]
        self.assertGreater(left[2], left[0])
        self.assertGreater(right[0], right[2])


class FramesTests(unittest.TestCase):
    def still(self, **kwargs):
        fleet = StillFleet(world().scene, {"a": (-1.0, 0.0, 1.0, 0.0), "b": (1.0, 0.0, 1.2, 0.7)})
        return fleet, SimFrames(fleet, **kwargs)

    def test_nothing_until_a_frame_exists_and_then_a_real_jpeg(self):
        fleet, frames = self.still()
        self.assertIsNone(frames.latest("a"))
        self.assertIsNone(frames.latest("nobody"))
        frame = frames.capture("a")
        self.assertIs(frames.latest("a"), frame)
        self.assertEqual((frame.drone, frame.number), ("a", 1))
        self.assertEqual(frame.truth, (-1.0, 0.0, 1.0, 0.0))
        image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
        self.assertLess(abs(len(frame.jpeg) - SIZES[1]), 2000)

    def test_a_frame_arrives_as_late_as_real_video_does(self):
        fleet, frames = self.still(latency_s=0.15)
        fleet.t = 10.0
        frames.capture("a", delay=True)
        self.assertIsNone(frames.latest("a"))
        fleet.t = 10.1
        self.assertIsNone(frames.latest("a"))
        fleet.t = 10.16
        frame = frames.latest("a")
        self.assertEqual((frame.taken, frame.t), (10.0, 10.15))

    def test_latest_does_not_wait_for_a_render(self):
        fleet, frames = self.still()
        frames.capture("a")
        rendering = threading.Event()
        release = threading.Event()

        def slow(pose):
            rendering.set()
            release.wait(5.0)
            return np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        with mock.patch.object(frames.camera, "render", slow):
            worker = threading.Thread(target=frames.capture, args=("b",))
            worker.start()
            self.assertTrue(rendering.wait(2.0))
            started = time.perf_counter()
            self.assertEqual(frames.latest("a").number, 1)
            took = time.perf_counter() - started
            release.set()
            worker.join(5.0)
        self.assertLess(took, 0.05)

    def _spin(self, frames, fleet, seconds, step=0.02):
        """A clock that keeps real time, as the flight loop's does."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            fleet.t += step
            time.sleep(step)

    def test_the_worker_serves_every_drone_at_the_rate_asked(self):
        fleet, frames = self.still(rate_hz=4.0, render_scale=0.2, latency_s=0.0, max_busy=0.9)
        frames.start()
        try:
            self._spin(frames, fleet, 1.2)
        finally:
            frames.stop()
        for key in ("a", "b"):
            frame = frames.latest(key)
            self.assertIsNotNone(frame)
            self.assertGreaterEqual(frame.number, 3)
            self.assertLessEqual(frame.number, 7)
        self.assertEqual(frames.errors, 0)
        self.assertIsNone(frames.thread)

    def test_the_busy_limit_slows_the_frame_rate_instead_of_the_cpu(self):
        fleet, frames = self.still(rate_hz=100.0, max_busy=0.25, latency_s=0.0)
        frames.start()
        started = time.monotonic()
        try:
            self._spin(frames, fleet, 1.0)
        finally:
            frames.stop()
        elapsed = time.monotonic() - started
        spent = sum(frames.render_ms) / 1000 + sum(frames.encode_ms) / 1000
        self.assertGreater(frames.sent, 0)
        self.assertLess(spent / elapsed, 0.4)

    def test_a_failed_render_is_counted_and_the_worker_carries_on(self):
        fleet, frames = self.still(rate_hz=10.0, render_scale=0.2, latency_s=0.0, max_busy=0.9)
        real = frames.camera.render
        calls = {"n": 0}

        def flaky(pose):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("render failed on purpose")
            return real(pose)

        with mock.patch.object(frames.camera, "render", flaky), self.assertLogs("swarm.sim_cameras", "ERROR"):
            frames.start()
            try:
                self._spin(frames, fleet, 0.8)
            finally:
                frames.stop()
        self.assertEqual(frames.errors, 1)
        self.assertGreater(frames.sent, 1)

    def test_blind_cameras_and_lost_links_stop_sending(self):
        fleet = StillFleet(world().scene, {"a": (-1.0, 0.0, 1.0, 0.0), "b": (1.0, 0.0, 1.2, 0.7)})

        class Link:
            connected = False

        fleet.links = {"b": Link()}
        frames = SimFrames(fleet, rate_hz=10.0, render_scale=0.2, latency_s=0.0, max_busy=0.9)
        frames.blind.add("a")
        frames.start()
        try:
            self._spin(frames, fleet, 0.5)
        finally:
            frames.stop()
        self.assertIsNone(frames.latest("a"))
        self.assertIsNone(frames.latest("b"))
        self.assertEqual(frames.sent, 0)


class DemoWorldTests(unittest.TestCase):
    def test_pads_are_inside_the_fence_and_clear(self):
        w = world()
        lo, hi = w.fence
        for x, y in w.pads.values():
            self.assertTrue(lo[0] < x < hi[0] and lo[1] < y < hi[1])
            self.assertIsNone(w.scene.collide(x, y, 1.0, 0.3))

    def test_furniture_is_on_the_map_and_hidden_items_are_not(self):
        w = world()
        room = w.room()
        names = set(room.obstacles)
        self.assertEqual(names, {"generator", "crate", "cabinet", "workbench"})
        known = [(o.lo, o.hi) for o in room.obstacles.values()]
        boxes = [(b.lo, b.hi) for b in w.scene.boxes]
        for item in w.items:
            self.assertIn((item.lo, item.hi), boxes)
            self.assertNotIn((item.lo, item.hi), known)

    def test_a_coordinator_accepts_the_room_and_the_pads(self):
        w = world()
        fleet = SimFleet(w.scene, w.pads, 1)
        configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in w.pads.items()]
        co = Coordinator(w.room(), configs, fleet.links)
        fleet.advance(0.5)
        co.tick(fleet.t, fleet.poses(), [])
        self.assertEqual(co.preflight(fleet.t), [])

    def test_anchors_round_trip_through_the_swarm_file_format(self):
        w = world()
        self.assertEqual(load_anchors(w.anchors_section()), w.anchors)
        self.assertGreaterEqual(len(w.anchors), 10)

    def test_no_anchor_is_under_or_touching_a_box_or_on_damage(self):
        w = world()
        self.assertEqual(len(w.anchors), 20)  # none of the grid is lost to furniture or damage
        marks = w.scene.decals[1 + len(w.anchors):]
        for a in w.anchors.values():
            reach = paper_width(a) / 2
            for thing in (*w.scene.boxes, *(m for m in marks if m.center[2] == 0.0)):
                self.assertFalse(thing.lo[0] - reach < a.x < thing.hi[0] + reach and thing.lo[1] - reach < a.y < thing.hi[1] + reach, a.id)

    def test_tags_cover_the_room_from_a_hover(self):
        w = world()
        lens = Lens.from_fov()
        spots = [(x, y) for x in np.arange(-2.5, 1.6, 0.25) for y in np.arange(-2.0, 2.01, 0.25)]
        seen = sum(bool(anchors_in_view(w.anchors, lens, TILT, (x, y, 1.0), 0.0)) for x, y in spots)
        self.assertGreater(seen / len(spots), 0.75)  # all heading north; a turn finds a tag for the rest

    def test_every_pad_should_see_an_anchor_from_a_hover(self):
        w = world()
        for x, y in w.pads.values():
            self.assertTrue(anchors_in_view(w.anchors, Lens.from_fov(), TILT, (x, y, 1.0), 0.0), (x, y))

    def test_hidden_items_can_be_found_by_simulated_eyes(self):
        w = world()
        for item in w.items:
            x, y, z, yaw = view_of(item.center, 1.5)
            fleet = StillFleet(w.scene, {"a": (x, y, z, yaw)})
            hover = fleet.drones["a"]
            hover.pitch = lambda: 0.0
            hover.spec = mock.Mock(gear_height_m=0.0)
            fleet.truth = lambda _: np.array([x, y, z])
            eyes = SimEyes(fleet, w.items, false_rate=0.0, seed=2)
            seen = [d.label for _ in range(40) for d in eyes.look("a", 0.0)]
            self.assertIn(item.label, seen)

    def test_staged_damage_is_low_enough_for_the_camera_and_shows_up_in_frames(self):
        w = world()
        bare = DecalScene(name="bare", bounds=w.scene.bounds, boxes=w.scene.boxes, floor_material=w.scene.floor_material)
        floor_and_markers = w.scene.decals[:1 + len(w.anchors)]
        marks = w.scene.decals[1 + len(w.anchors):]
        self.assertEqual(len(marks), len(w.damage))
        for d, mark in zip(w.damage, marks):
            self.assertLessEqual(d.where[2], 0.7, d.name)
            yaw = math.atan2(-d.facing[1], -d.facing[0]) if any(d.facing) else 0.0
            x, y, z, yaw = view_of(d.where, 1.0, yaw)
            with_mark = picture(bare, (*floor_and_markers, mark), x, y, z, yaw)
            without = picture(bare, floor_and_markers, x, y, z, yaw)
            changed = np.abs(with_mark - without).max(axis=2) > 0.08
            self.assertGreater(float(changed.mean()), 0.002, d.name)
            rows, cols = changed.shape
            middle = changed[rows // 4:3 * rows // 4, cols // 4:3 * cols // 4]
            self.assertGreater(float(middle.sum()) / max(1, int(changed.sum())), 0.5, f"{d.name} isn't near the middle of its view")

    def test_the_same_seed_gives_the_same_room(self):
        again = demo_world()
        self.assertEqual(again.anchors, world().anchors)
        for a, b in zip(again.scene.decals, world().scene.decals):
            self.assertTrue(np.array_equal(a.levels[0], b.levels[0]))


class TrackerTests(unittest.TestCase):
    """The anchors, as the tracker's own code reads them through a JPEG. Simulator numbers."""

    def test_markers_on_any_heading_give_the_right_pose(self):
        anchors = {a.id: a for a in (Anchor("tag:7", 1.0, 0.0, 0.0778, 0.0), Anchor("tag:8", 1.0, 0.5, 0.0778, 37.0),
                                     Anchor("tag:9", 1.0, -0.5, 0.0778, -120.0))}
        scene = DecalScene(name="plain", bounds=ROOM, floor_material="concrete", decals=anchor_decals(anchors))
        fleet = StillFleet(scene, {"a": (0.0, 0.0, 1.0, 0.0)})
        frames = SimFrames(fleet)
        seen = AnchorReader(anchors).find(gray(frames.capture("a")))
        self.assertEqual({a.id for a, _ in seen}, set(anchors))
        fix = solve_fix(seen, Lens.from_fov(), TILT, 0.0, 0.0)
        self.assertLess(math.hypot(fix.p[0], fix.p[1]), 0.05)
        self.assertLess(abs(fix.p[2] + 0.03 - 1.0), 0.05)
        self.assertLess(abs(fix.yaw), math.radians(3.0))

    def test_hovering_in_the_demo_room_reads_anchors_accurately(self):
        w = world()
        spots = {f"s{i}": (x, y, z, yaw) for i, (x, y, z, yaw) in enumerate((
            (-2.2, -1.2, 1.0, 0.0), (-2.2, 0.0, 1.0, 0.0), (-2.2, 1.2, 1.0, 0.0), (-1.0, 0.5, 1.0, 0.4),
            (0.5, -0.5, 1.0, -0.8), (1.5, 0.0, 1.0, 2.5), (-0.5, 1.0, 1.0, -2.0), (2.0, -0.2, 0.9, 1.7)))}
        frames = SimFrames(StillFleet(w.scene, spots))
        reader = AnchorReader(w.anchors)
        errors = []
        for name, (x, y, z, yaw) in spots.items():
            fix = solve_fix(reader.find(gray(frames.capture(name))), Lens.from_fov(), TILT, 0.0, yaw, w.fence)
            if fix is not None:
                errors.append(math.hypot(fix.p[0] - x, fix.p[1] - y))
        self.assertGreaterEqual(len(errors), 6)
        self.assertLess(statistics.median(errors), 0.08)

    def test_the_tracker_can_read_from_sim_frames_on_its_own_thread(self):
        w = world()
        spot = (-2.2, 0.0, 1.0, 0.0)
        fleet = StillFleet(w.scene, {"2": spot})
        frames = SimFrames(fleet, rate_hz=10.0, latency_s=0.0, max_busy=0.9)
        tracker = InsideOutTracker({"2": DroneOptics(spot[0] + 0.1, spot[1] - 0.05, 0.0)}, w.anchors, w.fence, video_latency=0.0)
        frames.start()
        tracker.start(frames)
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and tracker.report(fleet.t)["2"]["fixes"] < 3:
                fleet.t += 0.02
                time.sleep(0.02)
        finally:
            tracker.stop()
            frames.stop()
        report = tracker.report(fleet.t)["2"]
        self.assertGreaterEqual(report["fixes"], 1, report)
        pose = tracker.poses(fleet.t)["2"]
        self.assertLess(math.hypot(pose.p[0] - spot[0], pose.p[1] - spot[1]), 0.1)


class SimFleetTests(unittest.TestCase):
    def test_frames_follow_a_real_simulated_flight(self):
        w = world()
        fleet = SimFleet(w.scene, w.pads, 3)
        frames = SimFrames(fleet, latency_s=0.0)
        on_the_ground = frames.capture("2")
        self.assertLess(on_the_ground.truth[2], 0.2)
        for link in fleet.links.values():
            link.command("takeoff")
        fleet.advance(6.0)
        flying = {k: frames.capture(k) for k in w.pads}
        reader = AnchorReader(w.anchors)
        read = 0
        for key, frame in flying.items():
            self.assertGreater(frame.truth[2], 0.6, key)
            read += bool(reader.find(gray(frame)))
        self.assertGreaterEqual(read, 2)
        self.assertNotEqual(on_the_ground.jpeg, flying["2"].jpeg)


if __name__ == "__main__":
    unittest.main()
