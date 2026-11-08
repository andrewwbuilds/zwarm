import json
import os
import shlex
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from mapping import fleet
from mapping.curate import raw_name
from mapping.fleet import (FleetCapture, LocalRunner, MapError, SshRunner, build_map, colmap_steps, level_map, motion_scores,
                           photos_by_drone, pick_by_motion, pick_runner, pool_photos, read_cameras, safe_name, share_budget,
                           static_mask, window_for)
from mapping.reconstruct import StepFailed
from swarm.frames import Frame


def jpeg(seed: int = 0, stamp: bool = False, size=(160, 90)) -> bytes:
    """A picture of random blobs that differs per seed. With stamp, a fixed label in the corner."""
    rng = np.random.default_rng(seed)
    w, h = size
    image = cv2.resize(rng.integers(0, 255, (h // 6, w // 6, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    if stamp:
        cv2.putText(image, "SIMULATED", (w - 110, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "SIMULATED", (w - 110, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


class FakeFrames:
    def __init__(self, drones=("1", "2")):
        self.backends = {d: f"http://{d}" for d in drones}
        self.current: dict[str, Frame] = {}
        self.broken = False

    def show(self, drone: str, number: int, t: float = 0.0, data: bytes = b"\xff\xd8x\xff\xd9") -> None:
        self.current[drone] = Frame(drone, t, data, number)

    def latest(self, drone: str):
        if self.broken:
            raise RuntimeError("camera exploded")
        return self.current.get(drone)


def until(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class NameTests(unittest.TestCase):
    def test_paths_cant_escape_the_root(self):
        for bad in ("../up", "a/b", "", ".hidden", "..", "/etc", "a b", None):
            with self.assertRaises(MapError, msg=repr(bad)):
                safe_name(bad)

    def test_a_normal_name_passes(self):
        self.assertEqual(safe_name("north-side_2.map"), "north-side_2.map")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frames = FakeFrames()
        self.capture = FleetCapture(self.frames, self.root)
        self.addCleanup(self.capture.stop)

    def saved(self, drone: str) -> list[str]:
        return sorted(p.name for p in (self.root / "m" / "drones" / drone / "raw").glob("*.jpg"))

    def test_saves_each_new_frame_once_per_drone(self):
        self.capture.start("m")
        self.frames.show("1", 1)
        self.assertTrue(until(lambda: self.capture.status()["by_drone"].get("1") == 1))
        time.sleep(0.06)  # the same frame, polled again, isn't saved again
        self.frames.show("1", 2)
        self.frames.show("2", 7)
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 3))
        self.assertEqual(len(self.saved("1")), 2)
        self.assertEqual(len(self.saved("2")), 1)

    def test_status_while_capturing_and_after_stop(self):
        status = self.capture.start("m")
        self.assertEqual((status["name"], status["state"]), ("m", "capturing"))
        self.assertEqual(self.capture.stop()["state"], "captured")

    def test_stop_never_waits_for_a_slow_disk(self):
        release, writing = threading.Event(), threading.Event()
        real = Path.write_bytes

        def slow(path, data):
            writing.set()
            release.wait(5)
            return real(path, data)
        self.addCleanup(release.set)
        with mock.patch.object(Path, "write_bytes", slow):
            self.capture.start("m")
            self.frames.show("1", 1)
            self.assertTrue(writing.wait(3))
            started = time.perf_counter()
            self.assertEqual(self.capture.stop()["state"], "captured")
            self.assertLess(time.perf_counter() - started, 0.1)
            release.set()
            self.capture.capture_thread.join(3)

    def test_stop_with_nothing_running_is_harmless(self):
        self.assertEqual(self.capture.stop()["state"], "idle")

    def test_a_second_start_is_refused(self):
        self.capture.start("m")
        with self.assertRaisesRegex(MapError, "Already capturing"):
            self.capture.start("other")

    def test_a_bad_name_is_refused_before_any_folder_exists(self):
        with self.assertRaises(MapError):
            self.capture.start("../escape")
        self.assertFalse((self.root.parent / "escape").exists())

    def test_a_camera_failure_doesnt_end_the_capture(self):
        self.frames.broken = True
        self.capture.start("m")
        time.sleep(0.05)
        self.frames.broken = False
        self.frames.show("2", 1)
        self.assertTrue(until(lambda: self.capture.status()["by_drone"].get("2") == 1))

    def test_a_write_failure_is_reported_and_capture_carries_on(self):
        self.capture.start("m")
        with mock.patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
            self.frames.show("1", 1)
            self.assertTrue(until(lambda: "disk full" in (self.capture.status()["error"] or "")))
        self.frames.show("1", 2)
        self.assertTrue(until(lambda: self.capture.status()["by_drone"].get("1") == 1))

    def test_frame_times_count_from_the_first_frame_on_any_clock(self):
        self.capture.start("t")
        self.frames.show("1", 1, t=5000.0)
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 1))
        self.frames.show("2", 1, t=5002.5)
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 2))
        self.assertEqual([p.name for p in (self.root / "t" / "drones" / "1" / "raw").glob("*.jpg")], [raw_name(0, 0)])
        self.assertEqual([p.name for p in (self.root / "t" / "drones" / "2" / "raw").glob("*.jpg")], [raw_name(0, 2500)])

    def test_starting_the_same_map_again_carries_on_after_a_battery_swap(self):
        self.capture.start("m")
        self.frames.show("1", 1, t=10.0)
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 1))
        self.frames.show("1", 2, t=12.0)
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 2))
        self.capture.stop()
        self.capture.start("m")
        self.frames.show("1", 1, t=99.0)  # the same drone's video starts counting again
        self.assertTrue(until(lambda: self.capture.status()["photos"] == 3))
        self.capture.stop()
        names = self.saved("1")
        self.assertEqual(names, [raw_name(0, 0), raw_name(1, 2000), raw_name(2, 3000)])

    def test_a_drone_id_with_odd_characters_stays_inside_its_folder(self):
        frames = FakeFrames(drones=("../evil",))
        capture = FleetCapture(frames, self.root)
        self.addCleanup(capture.stop)
        capture.start("m")
        frames.show("../evil", 1)
        self.assertTrue(until(lambda: capture.status()["photos"] == 1))
        self.assertEqual([p.name for p in (self.root / "m" / "drones").iterdir()], [".._evil"])

    def test_drone_ids_come_from_whatever_the_frames_object_offers(self):
        sim = SimpleNamespace(fleet=SimpleNamespace(drones={"a": object(), "b": object()}), latest=lambda d: None)
        self.assertEqual(FleetCapture(sim, self.root)._drone_ids(), ["a", "b"])
        self.assertEqual(FleetCapture(sim, self.root, drones=["z"])._drone_ids(), ["z"])
        self.assertEqual(FleetCapture(self.frames, self.root)._drone_ids(), ["1", "2"])
        self.assertEqual(FleetCapture(SimpleNamespace(latest=lambda d: None), self.root)._drone_ids(), [])


class ChoosingTests(unittest.TestCase):
    def test_window_follows_the_video_rate(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for i in range(171):  # 17 fps for 10 s
                (raw / raw_name(i, i * 1000 // 17)).write_bytes(b"x")
            self.assertEqual(window_for(raw), 3)
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for i in range(51):  # the simulator's 5 fps
                (raw / raw_name(i, i * 200)).write_bytes(b"x")
            self.assertEqual(window_for(raw), 1)

    def test_a_hover_earns_few_photos_and_movement_earns_many(self):
        motion = [1.0] * 100 + [9.0] * 100 + [1.0] * 100  # noise, then a turn, then noise
        chosen = pick_by_motion(motion, 40)
        self.assertEqual(len(chosen), 40)
        moving = sum(100 <= i < 200 for i in chosen)
        self.assertGreater(moving, 28)

    def test_a_small_flight_keeps_every_photo(self):
        self.assertEqual(pick_by_motion([1.0, 2.0, 3.0], 10), [0, 1, 2])

    def test_no_budget_or_no_photos_gives_nothing(self):
        self.assertEqual(pick_by_motion([1.0, 2.0], 0), [])
        self.assertEqual(pick_by_motion([], 5), [])

    def test_a_still_picture_still_spends_the_whole_budget_evenly(self):
        chosen = pick_by_motion([1.0] * 100, 10)
        self.assertEqual(len(chosen), 10)
        self.assertLess(max(np.diff(chosen)), 20)

    def test_the_budget_is_met_even_when_motion_comes_in_one_burst(self):
        motion = [0.0] * 50 + [500.0] + [0.0] * 50
        self.assertEqual(len(pick_by_motion(motion, 30)), 30)

    def test_a_short_flight_leaves_its_share_to_the_others(self):
        self.assertEqual(share_budget({"1": 500, "2": 500, "3": 10}, 240), {"3": 10, "1": 115, "2": 115})

    def test_the_cap_is_never_exceeded(self):
        self.assertEqual(sum(share_budget({"1": 500, "2": 500, "3": 500}, 100).values()), 100)

    def test_motion_notices_a_changed_picture(self):
        with TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.jpg" for i in range(3)]
            for path, seed in zip(paths, (1, 1, 2)):
                path.write_bytes(jpeg(seed, size=(640, 360)))
            scores = motion_scores(paths)
        self.assertEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 0.0, places=3)
        self.assertGreater(scores[2], 5.0)


class PoolingTests(unittest.TestCase):
    def make_drone(self, folder: Path, drone: str, count: int) -> None:
        images = folder / "drones" / drone / "images"
        images.mkdir(parents=True)
        for i in range(count):
            (images / f"{i + 1:05d}.jpg").write_bytes(f"{drone}-{i}".encode())
        (folder / "drones" / drone / "motion.json").write_text(json.dumps(
            {"motion": [1.0] * count, "raw": [raw_name(i, i * 100) for i in range(count)]}))

    def test_photos_land_in_one_folder_per_drone_within_the_cap(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_drone(folder, "1", 50)
            self.make_drone(folder, "2", 50)
            self.make_drone(folder, "3", 4)
            pooled = pool_photos(folder, 30)
            self.assertEqual(pooled, {"1": 13, "2": 13, "3": 4})
            self.assertEqual(len(list((folder / "images" / "1").glob("*.jpg"))), 13)
            self.assertEqual((folder / "images" / "3" / "00002.jpg").read_bytes(), b"3-1")
            sources = json.loads((folder / "photos.json").read_text())
            self.assertEqual(sources["3/00002.jpg"], raw_name(1, 100))
            self.assertEqual(len(sources), 30)

    def test_a_rebuild_replaces_old_photos(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_drone(folder, "1", 20)
            pool_photos(folder, 20)
            pool_photos(folder, 5)
            self.assertEqual(len(list((folder / "images" / "1").glob("*.jpg"))), 5)


class MaskTests(unittest.TestCase):
    def write(self, folder: Path, stamp: bool, drones=("1", "2"), count: int = 12) -> None:
        for drone in drones:
            (folder / "images" / drone).mkdir(parents=True)
            for i in range(count):
                (folder / "images" / drone / f"{i:05d}.jpg").write_bytes(jpeg(int(drone) * 100 + i, stamp, size=(640, 360)))

    def test_a_fixed_label_is_masked_and_the_rest_is_kept(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, stamp=True)
            path = static_mask(folder)
            self.assertIsNotNone(path)
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (360, 640))
            self.assertEqual(int(mask[352, 580]), 0)  # on the label
            self.assertEqual(int(mask[100, 100]), 255)
            self.assertLess((mask == 0).mean(), 0.1)

    def test_a_moving_picture_needs_no_mask(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, stamp=False)
            self.assertIsNone(static_mask(folder))

    def test_a_drone_that_never_moved_doesnt_black_out_the_picture(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "images" / "1").mkdir(parents=True)
            still = jpeg(5, size=(640, 360))
            for i in range(12):
                (folder / "images" / "1" / f"{i:05d}.jpg").write_bytes(still)
            self.assertIsNone(static_mask(folder))


class StepsTests(unittest.TestCase):
    def flags(self, **kwargs) -> dict:
        settings = dict(root="/r", width=1184, height=656, threads=8, gpu=False, masked=False)
        settings.update(kwargs)
        return {name: args for name, args in colmap_steps(**settings)}

    def test_gpu_stays_off_for_brews_colmap(self):
        steps = self.flags()
        self.assertEqual(steps["features"][steps["features"].index("--FeatureExtraction.use_gpu") + 1], "0")
        self.assertEqual(steps["matching"][steps["matching"].index("--FeatureMatching.use_gpu") + 1], "0")

    def test_every_pair_is_matched_so_drones_can_link(self):
        self.assertEqual(self.flags()["matching"][0], "exhaustive_matcher")

    def test_each_drone_gets_its_own_camera_unless_told_otherwise(self):
        self.assertIn("--ImageReader.single_camera_per_folder", self.flags()["features"])
        shared = self.flags(shared_camera=True)["features"]
        self.assertIn("--ImageReader.single_camera", shared)
        self.assertNotIn("--ImageReader.single_camera_per_folder", shared)

    def test_the_lens_guess_is_90_degrees_across_the_width(self):
        features = self.flags()["features"]
        self.assertEqual(features[features.index("--ImageReader.camera_params") + 1], "592.0,592.0,328.0,0")

    def test_the_mask_is_only_used_when_there_is_one(self):
        self.assertNotIn("--ImageReader.camera_mask_path", self.flags()["features"])
        masked = self.flags(masked=True)["features"]
        self.assertEqual(masked[masked.index("--ImageReader.camera_mask_path") + 1], "/r/mask.png")


class ReadingTests(unittest.TestCase):
    def test_reads_pose_and_name_and_survives_an_empty_points_line(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            # A half turn about z, so the world-to-camera rotation is diag(-1, -1, 1).
            path.write_text("# header\n"
                            "1 0 0 0 1 1 2 3 1 1/00001.jpg\n"
                            "\n"
                            "2 1 0 0 0 0 0 5 1 2/00001.jpg\n"
                            "10 20 -1\n")
            cameras = read_cameras(path)
        self.assertEqual(sorted(cameras), ["1/00001.jpg", "2/00001.jpg"])
        centre, rotation = cameras["1/00001.jpg"]
        np.testing.assert_allclose(rotation, np.diag([-1.0, -1.0, 1.0]))
        np.testing.assert_allclose(centre, [1.0, 2.0, -3.0])
        np.testing.assert_allclose(cameras["2/00001.jpg"][0], [0.0, 0.0, -5.0])

    def test_photos_are_counted_by_their_folder(self):
        self.assertEqual(photos_by_drone(["1/a.jpg", "1/b.jpg", "2/a.jpg"]), {"1": 2, "2": 1})


class LevelTests(unittest.TestCase):
    def world_to_camera(self, yaw: float, tilt_deg: float = 45.0) -> np.ndarray:
        t = np.radians(tilt_deg)
        forward = np.array([np.cos(yaw) * np.cos(t), np.sin(yaw) * np.cos(t), -np.sin(t)])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        down = np.cross(forward, right)
        return np.array([right, down, forward])

    def test_undoes_an_arbitrary_rotation_scale_and_shift(self):
        rng = np.random.default_rng(3)
        # COLMAP's frame: world scaled by 0.2, then rotated and shifted.
        angle = 0.7
        spin = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        def to_model(p):
            return 0.2 * spin @ np.asarray(p) + np.array([3.0, -1.0, 2.0])
        cameras = {}
        for i, yaw in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False)):
            where = [rng.uniform(-2, 2), rng.uniform(-2, 2), 1.0]
            name = f"{1 + i % 3}/{i:05d}.jpg"
            cameras[name] = (to_model(where), self.world_to_camera(0.0 if i < 3 else yaw) @ spin.T)
        floor = np.array([to_model([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0]) for _ in range(2000)])
        rotation, scale, offset = level_map(cameras, floor)
        level = scale * floor @ rotation.T + offset
        self.assertLess(np.abs(level[:, 2]).max(), 0.05)  # the floor is flat at z=0
        self.assertAlmostEqual(scale, 5.0, delta=0.2)
        centres = np.array([c for c, _ in cameras.values()])
        np.testing.assert_allclose((scale * centres @ rotation.T + offset)[:, 2], 1.0, atol=0.05)
        forward = self.world_to_camera(0.0) @ spin.T
        self.assertGreater((rotation @ forward.T @ np.array([0, 0, 1.0]))[0], 0.6)  # +x is the way drones faced at takeoff

    def test_a_cloud_with_no_points_still_gives_a_frame(self):
        cameras = {f"1/{i}.jpg": (np.array([float(i), 0.0, 0.0]), self.world_to_camera(0.0)) for i in range(4)}
        rotation, scale, offset = level_map(cameras, np.zeros((0, 3)))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        self.assertGreater(scale, 0)


class BuildStateTests(unittest.TestCase):
    RESULT = {"photos": 3, "by_drone": {"1": 3}, "registered": 3, "registered_by_drone": {"1": 3}, "points": 2,
              "models": [{"folder": "0", "registered": 3, "points": 2, "by_drone": {"1": 3}}], "warnings": [], "seconds": {}, "runner": "local"}

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = []
        self.capture = FleetCapture(FakeFrames(), self.root, on_event=lambda *a: self.events.append(a))

    def finish(self, result=None):
        folder = self.root / "m"
        folder.mkdir(exist_ok=True)
        np.save(folder / "points.npy", np.array([[1.23456, 2, 3, 10, 20, 30], [4, 5, 6, 40, 50, 60], [7, 8, 9, 70, 80, 90]], np.float32))
        return result or self.RESULT

    def test_a_finished_build_reports_its_numbers_and_tells_the_agent(self):
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()):
            self.capture.build("m", splat=False)
            status = self.capture.wait(5)
        self.assertEqual(status["state"], "done")
        self.assertEqual((status["photos"], status["registered"], status["models"]), (3, 3, [3]))
        self.assertEqual(self.events[0][0], "map")
        self.assertIn("3 of 3", self.events[0][1])

    def test_a_failed_build_says_why_and_never_raises(self):
        with mock.patch("mapping.fleet.build_map", side_effect=StepFailed("too little overlap")):
            self.capture.build("m", splat=False)
            status = self.capture.wait(5)
        self.assertEqual(status["state"], "failed")
        self.assertIn("too little overlap", status["error"])
        self.assertEqual(self.capture.points(), [])

    def test_no_saved_frames_is_a_readable_failure(self):
        self.capture.build("empty", splat=False)
        status = self.capture.wait(5)
        self.assertEqual(status["state"], "failed")
        self.assertIn("No frames saved", status["error"])

    def test_an_exploding_event_listener_cant_break_a_build(self):
        def explode(*_):
            raise RuntimeError("listener bug")
        self.capture.on_event = explode
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()):
            self.capture.build("m", splat=False)
            self.assertEqual(self.capture.wait(5)["state"], "done")

    def test_asking_again_during_a_build_starts_nothing_new(self):
        gate, calls = threading.Event(), []
        self.finish()

        def slow(*args, **kwargs):
            calls.append(1)
            gate.wait(5)
            return self.RESULT
        with mock.patch("mapping.fleet.build_map", side_effect=slow):
            self.capture.build("m", splat=False)
            self.assertTrue(until(lambda: calls))
            self.assertEqual(self.capture.build("m", splat=False)["state"], "building")
            with self.assertRaisesRegex(MapError, "still building"):
                self.capture.start("next")
            gate.set()
            self.assertEqual(self.capture.wait(5)["state"], "done")
        self.assertEqual(len(calls), 1)

    def test_a_build_waits_for_the_last_frame_to_reach_the_disk(self):
        release, writing, seen = threading.Event(), threading.Event(), []
        real = Path.write_bytes

        def slow(path, data):
            writing.set()
            release.wait(5)
            return real(path, data)

        def check(folder, *args, **kwargs):
            seen.append(len(list((folder / "drones" / "1" / "raw").glob("*.jpg"))))
            return self.RESULT
        self.addCleanup(release.set)
        self.finish()
        frames = self.capture.frames
        with mock.patch.object(Path, "write_bytes", slow), mock.patch("mapping.fleet.build_map", side_effect=check):
            self.capture.start("m")
            frames.show("1", 1)
            self.assertTrue(writing.wait(3))
            self.capture.stop()
            self.capture.build("m", splat=False)
            time.sleep(0.05)
            self.assertEqual(seen, [])  # still held back by the write
            release.set()
            self.capture.wait(5)
        self.assertEqual(seen, [1])

    def test_build_ends_a_running_capture_of_the_same_map(self):
        self.capture.start("m")
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()):
            self.capture.build("m", splat=False)
            self.assertEqual(self.capture.wait(5)["state"], "done")
        self.assertFalse(self.capture.capture_thread.is_alive())

    def test_building_another_map_during_a_capture_is_refused(self):
        self.capture.start("m")
        self.addCleanup(self.capture.stop)
        with self.assertRaisesRegex(MapError, "still capturing"):
            self.capture.build("other")

    def test_points_are_empty_until_built_then_sampled_as_plain_numbers(self):
        self.assertEqual(self.capture.points(), [])
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()):
            self.capture.build("m", splat=False)
            self.capture.wait(5)
        points = self.capture.points()
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], [1.235, 2.0, 3.0, 10, 20, 30])
        self.assertIsInstance(points[0][3], int)
        json.dumps(points)
        self.assertEqual(len(self.capture.points(limit=2)), 2)

    def test_a_restart_still_serves_the_last_map(self):
        folder = self.root / "m"
        self.finish()
        (folder / "result.json").write_text(json.dumps(self.RESULT))
        fresh = FleetCapture(FakeFrames(), self.root)
        self.assertEqual(fresh.status()["state"], "done")
        self.assertEqual(fresh.status()["name"], "m")
        self.assertEqual(len(fresh.points()), 3)

    def test_splat_is_optional_and_skips_cleanly_without_brush(self):
        from mapping.reconstruct import ToolMissing
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()), \
                mock.patch("mapping.fleet.build_splat", side_effect=ToolMissing("no brush")):
            self.capture.build("m", splat=True)
            status = self.capture.wait(5)
        self.assertEqual(status["state"], "done")
        self.assertIn("skipped", status["splat"])

    def test_a_failed_splat_leaves_the_map_usable(self):
        with mock.patch("mapping.fleet.build_map", return_value=self.finish()), \
                mock.patch("mapping.fleet.build_splat", side_effect=StepFailed("gpu sad")):
            self.capture.build("m", splat=True)
            status = self.capture.wait(5)
        self.assertEqual(status["state"], "done")
        self.assertIn("failed", status["splat"])
        self.assertEqual(len(self.capture.points()), 3)


class RunnerTests(unittest.TestCase):
    def test_the_env_var_picks_the_remote_runner(self):
        with mock.patch.dict(os.environ, {"MAP_BUILD_SSH": "ubuntu@gpu-box"}):
            self.assertIsInstance(pick_runner(), SshRunner)
        with mock.patch.dict(os.environ, {"MAP_BUILD_SSH": ""}):
            self.assertIsInstance(pick_runner(), LocalRunner)

    def test_a_host_that_looks_like_an_ssh_option_is_refused(self):
        for bad in ("-oProxyCommand=sh", "a b", "x;rm -rf /", ""):
            with self.assertRaises(StepFailed, msg=bad):
                SshRunner(bad)

    def test_the_same_steps_run_remotely_with_safe_quoting(self):
        commands = []
        with TemporaryDirectory() as tmp, \
                mock.patch("mapping.fleet.run", side_effect=lambda command, log, env=None: commands.append(command)), \
                mock.patch("mapping.fleet.subprocess.run", return_value=SimpleNamespace(stdout="COLMAP 3.11 with CUDA")):
            folder = Path(tmp) / "my map"
            folder.mkdir()
            seen = []

            def plan(root, threads, gpu):
                seen.append((root, threads, gpu))
                return colmap_steps(root, 640, 360, threads, gpu, False)
            SshRunner("ubuntu@gpu-box").run(folder, plan, lambda step: None)
        self.assertEqual(seen, [("zwarm-maps/my map", -1, True)])
        joined = [" ".join(c) for c in commands]
        self.assertTrue(any(c[0] == "rsync" and c[-1] == "ubuntu@gpu-box:zwarm-maps/my map/images/" for c in commands))
        self.assertTrue(any(c[0] == "rsync" and c[-2] == "ubuntu@gpu-box:zwarm-maps/my map/colmap/sparse/" for c in commands))
        remote = [c[-1] for c in commands if c[0] == "ssh" and "colmap feature_extractor" in c[-1]]
        self.assertEqual(len(remote), 1)
        words = shlex.split(remote[0])
        self.assertEqual(words[words.index("--image_path") + 1], "zwarm-maps/my map/images")
        self.assertEqual(words[words.index("--FeatureExtraction.use_gpu") + 1], "1")
        self.assertTrue(any("exhaustive_matcher" in c for c in joined))

    def test_a_remote_failure_falls_back_to_a_local_build(self):
        class Broken:
            name = "ssh nowhere"

            def run(self, folder, plan, on_step):
                raise StepFailed("no route to host")
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image_dir = folder / "drones" / "1"
            for sub in ("raw", "images"):
                (image_dir / sub).mkdir(parents=True)
            for i in range(12):
                data = jpeg(i, size=(320, 180))
                (image_dir / "raw" / raw_name(i, i * 100)).write_bytes(data)
                (image_dir / "images" / f"{i + 1:05d}.jpg").write_bytes(data)
            (image_dir / "motion.json").write_text(json.dumps({"motion": [1.0] * 12, "raw": [raw_name(i, i * 100) for i in range(12)]}))
            local = mock.Mock()
            with mock.patch("mapping.fleet.run"), mock.patch.object(LocalRunner, "run", local):
                # No model comes back from the mocked local run, so the build stops there, having tried it.
                with self.assertRaisesRegex(StepFailed, "could not link"):
                    build_map(folder, 240, Broken())
            local.assert_called_once()


class ConstantsTests(unittest.TestCase):
    def test_a_build_leaves_cores_for_the_flight_loop(self):
        with mock.patch("mapping.fleet.os.cpu_count", return_value=18):
            self.assertEqual(fleet.build_threads(), 14)
        with mock.patch("mapping.fleet.os.cpu_count", return_value=2):
            self.assertEqual(fleet.build_threads(), 2)


if __name__ == "__main__":
    unittest.main()
