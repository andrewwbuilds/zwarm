"""The floor-tag tracker against rendered frames, where the true poses are known exactly.

These run the same detector real flights use, on simulated frames. They bound the maths. They say
nothing about what the real camera will do.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.scene import Box, Scene
from swarm.control import Pose
from swarm.coordinator import Coordinator, DroneConfig, Phase, SIGMA_HOLD, SIGMA_LAND
from swarm.room import Room
from swarm.server import SwarmFile
from swarm.sim_anchors import FloorCamera, VideoConditions
from swarm.tracking import (
    Anchor,
    AnchorReader,
    Estimator,
    Fix,
    InsideOutTracker,
    DroneOptics,
    Lens,
    SwarmFileError,
    anchors_in_view,
    anchors_section,
    fit_focal,
    grid_layout,
    load_anchors,
    solve_fix,
    tag_grid,
)

WHITE = 0.0767
TILT = 45.0
HALL = Scene(name="hall", bounds=Box((-6, -6, 0), (6, 6, 3), "wall"))
LEVEL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def yawed(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


def grid():
    return load_anchors(anchors_section(WHITE, tag_grid()))


def read(anchors, x, y, z, yaw, seed=0, tilt=TILT, video=None, lens=None, assumed_tilt=None):
    image = FloorCamera(HALL, anchors, tilt, video or VideoConditions(), seed).render(x, y, z, yawed(yaw))
    seen = AnchorReader(anchors).find(image)
    return seen, solve_fix(seen, lens or Lens.from_fov(), math.radians(assumed_tilt or tilt), 0.0, yaw)


class LayoutTests(unittest.TestCase):
    def test_the_default_grid_matches_how_the_tags_are_laid_out(self):
        tags = {t["tag"]: t for t in tag_grid()}
        self.assertEqual(len(tags), 25)
        self.assertEqual((tags[0]["x"], tags[0]["y"]), (-2.0, 2.0))  # near left, seen from the south facing +x
        self.assertEqual((tags[4]["x"], tags[4]["y"]), (-2.0, -2.0))  # rising to the right along the row
        self.assertEqual((tags[5]["x"], tags[5]["y"]), (-1.0, 2.0))  # next row further away
        self.assertEqual((tags[24]["x"], tags[24]["y"]), (2.0, -2.0))  # far right
        self.assertTrue(all(t["rotation_deg"] == 0.0 for t in tags.values()))

    def test_ids_can_start_elsewhere(self):
        self.assertEqual(tag_grid(start_id=30)[0]["tag"], 30)

    def test_a_simulated_room_cannot_ask_for_more_tags_than_exist(self):
        with self.assertRaises(ValueError):
            grid_layout((-5, -5), (5, 5), 1.0, WHITE)


class SwarmFileTests(unittest.TestCase):
    def section(self, **changes):
        return {**anchors_section(WHITE, tag_grid()), **changes}

    def test_a_missing_size_is_refused_rather_than_guessed(self):
        for section in (self.section(white_square_edge_m=None), {k: v for k, v in self.section().items() if k != "white_square_edge_m"}):
            with self.assertRaisesRegex(SwarmFileError, "white_square_edge_m is missing"):
                load_anchors(section)

    def test_a_size_in_the_wrong_units_is_refused(self):
        with self.assertRaises(SwarmFileError):
            load_anchors(self.section(white_square_edge_m=76.7))

    def test_the_wrong_family_is_refused(self):
        with self.assertRaises(SwarmFileError):
            load_anchors(self.section(tag_family="tag36h11"))

    def test_the_example_file_loads_with_the_real_tags(self):
        config = SwarmFile.load(Path(__file__).parents[1] / "swarm" / "example.json")
        self.assertEqual(len(config.anchors), 25)
        self.assertEqual(config.anchors["tag:0"].size, WHITE)
        self.assertEqual(config.drones[0].camera_tilt_deg, 45.0)
        self.assertEqual(config.optics()["1"].tilt_deg, 45.0)

    def test_a_swarm_file_without_the_size_will_not_load(self):
        data = json.loads((Path(__file__).parents[1] / "swarm" / "example.json").read_text())
        data["anchors"]["white_square_edge_m"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "swarm.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(SwarmFileError):
                SwarmFile.load(path)


class TagReadingTests(unittest.TestCase):
    """Poses across the flying volume, over the real layout, against known truth."""

    POSES = [(x, y, z, yaw) for x, y in ((-1.3, 0.2), (0.3, -0.6), (-0.5, 1.1), (0.7, 0.7)) for z in (0.8, 1.0, 1.3)
             for yaw in (0.0, 0.5, -0.8)]

    def test_position_height_and_heading_across_the_volume(self):
        anchors = grid()
        lens = Lens.from_fov()
        errors_xy, errors_z, errors_yaw, read_count = [], [], [], 0
        for i, (x, y, z, yaw) in enumerate(self.POSES):
            if not anchors_in_view(anchors, lens, math.radians(TILT), (x, y, z), yaw):
                continue
            seen, fix = read(anchors, x, y, z, yaw, i)
            self.assertIsNotNone(fix, (x, y, z, yaw))
            read_count += 1
            errors_xy.append(math.hypot(fix.p[0] - x, fix.p[1] - y))
            errors_z.append(abs(fix.p[2] + 0.03 - z))
            errors_yaw.append(abs((fix.yaw - yaw + math.pi) % (2 * math.pi) - math.pi))
        self.assertGreater(read_count, 25)
        self.assertLess(max(errors_xy), 0.1, "horizontal")
        self.assertLess(max(errors_z), 0.1, "height")
        self.assertLess(max(errors_yaw), math.radians(4), "heading")

    def test_tag_rotation_on_the_floor_is_read_correctly(self):
        for rotation in (0.0, 90.0, 180.0, -135.0):
            tag = Anchor("tag:7", 0.8, 0.1, WHITE, rotation)
            seen, fix = read({tag.id: tag}, 0.0, 0.0, 1.0, 0.0)
            self.assertIsNotNone(fix, rotation)
            self.assertAlmostEqual(fix.p[0], 0.0, delta=0.08, msg=str(rotation))
            self.assertAlmostEqual(fix.yaw, 0.0, delta=math.radians(6), msg=str(rotation))

    def test_a_wrong_size_scales_position_and_height(self):
        # Why the size is required: what happens if the whole tag's width were entered instead.
        tag = Anchor("tag:7", 0.8, 0.0, WHITE)
        seen, good = read({tag.id: tag}, 0.0, 0.0, 1.0, 0.0)
        wrong = Anchor("tag:7", 0.8, 0.0, WHITE * 9 / 5)
        bad = solve_fix([(wrong, seen[0][1])], Lens.from_fov(), math.radians(TILT), 0.0, 0.0)
        self.assertAlmostEqual((bad.p[2] + 0.03) / (good.p[2] + 0.03), 1.8, delta=0.05)

    def test_a_tag_not_in_the_file_gives_no_position(self):
        tag = Anchor("tag:7", 0.8, 0.0, WHITE)
        image = FloorCamera(HALL, {tag.id: tag}, TILT, VideoConditions()).render(0.0, 0.0, 1.0, LEVEL)
        reader = AnchorReader({"tag:3": Anchor("tag:3", 0.0, 0.0, WHITE)})
        self.assertEqual(reader.find(image), [])
        self.assertEqual(reader.unknown, {7})

    def test_tilt_is_undone_when_working_out_heading(self):
        for tilt in (45.0, 60.0, 80.0):
            ahead = 1.0 / math.tan(math.radians(tilt))  # where the middle of the view meets the floor
            tag = Anchor("tag:7", ahead * math.cos(0.3), ahead * math.sin(0.3), WHITE)
            seen, fix = read({tag.id: tag}, 0.0, 0.0, 1.0, 0.3, tilt=tilt)
            self.assertIsNotNone(fix, tilt)
            self.assertAlmostEqual(fix.yaw, 0.3, delta=math.radians(6), msg=str(tilt))

    def test_a_drone_on_the_floor_cannot_see_the_floor(self):
        self.assertEqual(anchors_in_view(grid(), Lens.from_fov(), math.radians(TILT), (-2.5, 0.0, 0.05), 0.0), [])

    def test_from_a_metre_up_it_can(self):
        self.assertTrue(anchors_in_view(grid(), Lens.from_fov(), math.radians(TILT), (-2.5, 0.0, 1.0), 0.0))


class FocalFitTests(unittest.TestCase):
    def frames(self, fov, count=6):
        anchors = grid()
        out = []
        for i in range(count):
            video = VideoConditions(render_scale=1.0, fov_deg=fov)
            image = FloorCamera(HALL, anchors, TILT, video, i).render(-1.4 + 0.1 * i, 0.3, 1.1, yawed(0.1 * i))
            out.append(AnchorReader(anchors).find(image))
        return out

    def test_it_recovers_a_lens_narrower_than_the_guess(self):
        frames = self.frames(80.0)
        self.assertTrue(any(len(seen) >= 2 for seen in frames))
        result = fit_focal(frames)
        truth = (1184 / 2) / math.tan(math.radians(40.0))
        self.assertAlmostEqual(result["focal_px"] / truth, 1.0, delta=0.04)
        self.assertLess(result["error_px"], result["error_at_guess_px"])

    def test_one_tag_at_a_time_is_not_enough(self):
        tag = Anchor("tag:7", 0.8, 0.0, WHITE)
        seen = AnchorReader({tag.id: tag}).find(FloorCamera(HALL, {tag.id: tag}, TILT, VideoConditions()).render(0.0, 0.0, 1.0, LEVEL))
        with self.assertRaises(ValueError):
            fit_focal([seen])


def fix_at(t, x, y, z=1.0, yaw=0.0, sigma=0.02):
    return Fix(t, (x, y, z), yaw, ("tag:1",), 1.2, 0.5, sigma, 0.03, math.radians(2))


class EstimatorTests(unittest.TestCase):
    def test_on_the_ground_it_stays_at_its_pad_and_stays_sure(self):
        e = Estimator(-2.5, 1.0, 0.0)
        start = e.sigma_xy
        for i in range(100):
            e.advance(i * 0.05)
        self.assertEqual(tuple(e.p[:2]), (-2.5, 1.0))
        self.assertEqual(e.sigma_xy, start)

    def test_uncertainty_grows_without_sightings_and_faster_when_moving(self):
        hover, moving = Estimator(0, 0, 0), Estimator(0, 0, 0)
        hover.note_command(np.zeros(3), 0.0, True, False)
        moving.note_command(np.array([0.5, 0, 0]), 0.0, True, False)
        for i in range(100):
            hover.advance(i * 0.05)
            moving.advance(i * 0.05)
        self.assertGreater(hover.sigma_xy, 0.1)
        self.assertGreater(moving.sigma_xy, hover.sigma_xy + 0.2)
        self.assertGreater(moving.p[0], 1.5)  # and it carried the commanded motion forward

    def test_a_sighting_shrinks_uncertainty_and_moves_the_estimate(self):
        e = Estimator(0, 0, 0)
        e.note_command(np.zeros(3), 0.0, True, False)
        for i in range(60):
            e.advance(i * 0.05)
        before = e.sigma_xy
        self.assertTrue(e.add_fix(fix_at(2.9, 0.1, 0.0)))
        self.assertLess(e.sigma_xy, before / 2)
        self.assertAlmostEqual(e.p[0], 0.1, delta=0.03)

    def test_a_late_sighting_is_compared_with_where_it_was_then(self):
        e = Estimator(0, 0, 0)
        e.note_command(np.array([0.5, 0, 0]), 0.0, True, False)
        for i in range(61):
            e.advance(i * 0.05)
        at_three = e.p[0]
        then = [p for t, p, _ in e.history if t <= 2.81][-1][0]
        e.add_fix(fix_at(2.81, then, 0.0))  # exactly where we thought it was 0.2 s ago
        self.assertAlmostEqual(e.p[0], at_three, delta=0.02)

    def test_one_wild_sighting_is_ignored_and_three_agreeing_ones_are_believed(self):
        e = Estimator(0, 0, 0)
        e.note_command(np.zeros(3), 0.0, True, False)
        e.advance(0.0)
        e.add_fix(fix_at(0.0, 0.0, 0.0))
        e.advance(0.1)
        self.assertFalse(e.add_fix(fix_at(0.1, 3.0, 0.0)))
        self.assertLess(abs(e.p[0]), 0.1)
        e.advance(0.2)
        self.assertFalse(e.add_fix(fix_at(0.2, 3.0, 0.0)))
        e.advance(0.3)
        self.assertTrue(e.add_fix(fix_at(0.3, 3.0, 0.0)))
        self.assertGreater(e.p[0], 2.0)

    def test_a_late_toggle_does_not_invent_a_climb(self):
        e = Estimator(0, 0, 0)
        e.note_command(np.zeros(3), 0.0, True, False)
        for i in range(100):
            e.advance(i * 0.05)
        self.assertEqual(e.p[2], 0.0)  # height only ever comes from a sighting, so a failed takeoff stays visible

    def test_landing_is_modelled_down_to_the_floor(self):
        e = Estimator(0, 0, 0)
        e.note_command(np.zeros(3), 0.0, True, False)
        e.advance(0.0)
        e.add_fix(fix_at(0.0, 0.0, 0.0, z=1.0, sigma=0.02))
        e.note_command(np.zeros(3), 0.0, True, True)
        for i in range(1, 120):
            e.advance(i * 0.05)
        self.assertEqual(e.p[2], 0.0)


class FakeLink:
    def __init__(self):
        self.connected = True
        self.commands = []
        self.sticks = []

    def send_sticks(self, *values):
        self.sticks.append(values)

    def command(self, name):
        self.commands.append(name)
        return True


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.links = {"a": FakeLink(), "b": FakeLink()}
        self.co = Coordinator(Room((-4, -3, 0), (4, 3, 2.6)), [DroneConfig("a", (-1, 0, 0)), DroneConfig("b", (1, 0, 0))], self.links)
        self.co.tick(0.0, self.poses(0.0))
        self.co.drones["a"].phase = self.co.drones["b"].phase = Phase.AIR
        self.co.drones["a"].goal = np.array([-1.0, 0.0, 1.0])
        self.co.drones["b"].goal = np.array([1.0, 0.0, 1.0])

    def poses(self, t, sigma=0.03):
        return {"a": Pose(t, (-1.0, 0.0, 1.0), 0.0, sigma=sigma, v=np.zeros(3), fix_age=0.0),
                "b": Pose(t, (1.0, 0.0, 1.0), 0.0, sigma=0.03, v=np.zeros(3), fix_age=0.0)}

    def test_a_steady_pose_is_steered(self):
        self.co.tick(0.05, self.poses(0.05))
        self.assertTrue(self.co._steerable(self.co.drones["a"], 0.05))

    def test_past_the_hold_limit_it_stops_being_steered_and_the_sticks_centre(self):
        self.co.tick(0.05, self.poses(0.05, SIGMA_HOLD + 0.05))
        self.assertFalse(self.co._steerable(self.co.drones["a"], 0.05))
        self.assertEqual(self.links["a"].sticks[-1][:4], (0.0, 0.0, 0.0, 0.0))
        self.assertIn("tracking", self.co.drones["a"].faults)
        self.assertEqual(self.links["a"].commands, [])

    def test_a_sighting_clears_the_hold(self):
        self.co.tick(0.05, self.poses(0.05, SIGMA_HOLD + 0.05))
        self.co.tick(0.10, self.poses(0.10, 0.03))
        self.assertNotIn("tracking", self.co.drones["a"].faults)
        self.assertTrue(self.co._steerable(self.co.drones["a"], 0.10))

    def test_past_the_land_limit_it_lands_once(self):
        for i in range(10):
            self.co.tick(0.05 * (i + 1), self.poses(0.05 * (i + 1), SIGMA_LAND + 0.05))
        self.assertEqual(self.links["a"].commands, ["land"])
        self.assertEqual(self.co.drones["a"].phase, Phase.LANDING)

    def test_a_neighbour_keeps_further_away_from_an_uncertain_drone(self):
        def run(sigma):
            co = Coordinator(Room((-4, -3, 0), (4, 3, 2.6)), [DroneConfig("a", (0, 0, 0)), DroneConfig("b", (0.7, 0, 0))],
                             {"a": FakeLink(), "b": FakeLink()})
            poses = {"a": Pose(0.0, (0.0, 0.0, 1.0), 0.0, sigma=sigma, v=np.zeros(3)), "b": Pose(0.0, (0.7, 0.0, 1.0), 0.0, sigma=0.03, v=np.zeros(3))}
            co.tick(0.0, poses)
            for d in co.drones.values():
                d.phase = Phase.AIR
            co.drones["a"].goal = np.array([0.0, 0.0, 1.0])
            co.drones["b"].goal = np.array([0.7, 0.0, 1.0])
            co.tick(0.05, {k: Pose(0.05, p.p, 0.0, sigma=p.sigma, v=np.zeros(3)) for k, p in poses.items()})
            return co.drones["b"].commanded[0]
        self.assertGreater(run(0.25), run(0.03) + 0.05)


class HeadingLimitTests(unittest.TestCase):
    def test_a_heading_that_sees_nothing_is_turned_to_one_that_does(self):
        tracker = InsideOutTracker({"a": DroneOptics(0.0, 0.0)}, grid())
        at_far_edge = (1.9, 0.0, 1.0)
        allowed = tracker.allowed_yaw("a", at_far_edge, 0.0)  # facing off the end of the grid
        self.assertNotEqual(allowed, 0.0)
        self.assertTrue(anchors_in_view(tracker.anchors, Lens.from_fov(), math.radians(45), at_far_edge, allowed))

    def test_a_heading_that_sees_tags_is_left_alone(self):
        tracker = InsideOutTracker({"a": DroneOptics(0.0, 0.0)}, grid())
        self.assertEqual(tracker.allowed_yaw("a", (-1.0, 0.0, 1.0), 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
