"""On floor tags, furniture costs a drone its position: from above it hides the tags, and from the side it hides those behind it.

Geometry and planning only, so these say nothing about how a real drone reads tags.
"""

import math
import unittest

import numpy as np

from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.eyes import DEFAULT_TILT
from swarm.hub import Hub
from swarm.memory import Memory
from swarm.mission import MissionManager, RegionCoverage, Regions, _marker_filter, views_of
from swarm.planner import plan_path
from swarm.room import Obstacle, Room
from swarm.safety import Body, Limits, filter_velocities
from swarm.tracking import DroneOptics, InsideOutTracker, Lens, anchors_in_view, anchors_section, load_anchors, tag_grid


LOW = Obstacle("couch", (-0.5, -1.0, 0.0), (0.5, 1.0, 0.5))
FENCE = ((-2.5, -2.5, 0.0), (2.5, 2.5, 2.4))
CLEARANCE = 0.33


class FakeLink:
    connected = True

    def command(self, name):
        return True

    def send_sticks(self, *values):
        pass


class FakeEyes:
    def look(self, *args, **kwargs):
        return []

    def knows(self, label):
        return True


def tracked_room(*obstacles):
    room = Room(*FENCE, obstacles or (LOW,))
    optics = {"a": DroneOptics(-2.0, 0.0)}
    tracker = InsideOutTracker(optics, load_anchors(anchors_section(0.0767, tag_grid())), FENCE)
    coordinator = Coordinator(room, [DroneConfig("a", (-2.0, 0.0, 0.0))], {"a": FakeLink()}, tracker=tracker)
    return room, tracker, coordinator


class FullHeightTests(unittest.TestCase):
    def over(self, p, obstacle=LOW):
        return obstacle.lo[0] <= p[0] <= obstacle.hi[0] and obstacle.lo[1] <= p[1] <= obstacle.hi[1]

    def test_with_a_tracker_a_path_goes_around_a_low_obstacle(self):
        room, _, _ = tracked_room()
        path = plan_path(room, (-1.5, 0.0, 1.5), (1.5, 0.0, 1.5), CLEARANCE)
        for a, b in zip(path, path[1:]):
            for share in np.linspace(0.0, 1.0, 40):
                self.assertFalse(self.over(a + (b - a) * share), f"flew over it: {path}")

    def test_without_a_tracker_it_can_still_go_over(self):
        path = plan_path(Room(*FENCE, (LOW,)), (-1.5, 0.0, 1.5), (1.5, 0.0, 1.5), CLEARANCE)
        self.assertEqual(len(path), 2)

    def test_a_goal_over_it_is_refused(self):
        room, _, coordinator = tracked_room()
        coordinator.drones["a"].phase = Phase.AIR
        self.assertTrue(any("too close" in problem for problem in coordinator.check_goals({"a": (0.0, 0.0, 1.6)})))

    def test_the_filter_stops_a_drone_flying_in_over_the_top(self):
        room, _, _ = tracked_room()
        moving_in = Body("a", np.array([-0.8, 0.0, 1.5]), np.zeros(3), np.array([0.5, 0.0, 0.0]))
        result = filter_velocities([moving_in], room, Limits())
        self.assertLess(result.velocities["a"][0], 0.5)

    def test_a_drone_pushed_inside_leaves_sideways(self):
        room, _, _ = tracked_room()
        normal, distance = room.surfaces(np.array([-0.4, 0.0, 1.5]), 0.5)[0]
        self.assertLess(distance, 0.0)
        self.assertEqual(normal[2], 0.0)

    def test_the_map_keeps_its_true_height(self):
        room, _, _ = tracked_room()
        self.assertEqual(room.json()["obstacles"][0]["hi"][2], 0.5)
        self.assertEqual(room.obstacles["couch"].hi[2], 0.5)

    def test_what_a_camera_found_keeps_its_height_since_it_may_be_false(self):
        room, _, _ = tracked_room(LOW, Obstacle("seen-cup-1", (1.5, 1.5, 0.0), (1.7, 1.7, 0.3)))
        self.assertGreater(room.clearance((1.6, 1.6, 1.5)), 0.8)
        self.assertLess(room.clearance((0.0, 0.0, 1.5)), 0.0)
        self.assertEqual([o.name for o in room.listed()], ["couch"])


class HiddenTagTests(unittest.TestCase):
    def test_a_tag_behind_furniture_does_not_count(self):
        lens = Lens.from_fov()
        tilt = math.radians(45)
        anchors = load_anchors(anchors_section(0.0767, tag_grid()))
        # Facing north from south of the couch, the tags beyond it are in the frame.
        pose = ((-1.0, 0.0, 1.0), 0.0)
        anyway = anchors_in_view(anchors, lens, tilt, *pose)
        self.assertTrue(anyway)
        wall = (np.array([-0.7, -3.0, 0.0]), np.array([-0.5, 3.0, 2.0]))
        self.assertEqual(anchors_in_view(anchors, lens, tilt, *pose, blockers=[wall]), [])

    def test_a_tag_under_furniture_does_not_count(self):
        lens = Lens.from_fov()
        tilt = math.radians(45)
        anchors = load_anchors(anchors_section(0.0767, tag_grid()))
        pose = ((-1.0, 0.0, 1.0), 0.0)
        seen = anchors_in_view(anchors, lens, tilt, *pose)
        first = anchors[seen[0]]
        over_one = (np.array([first.x - 0.2, first.y - 0.2, 0.0]), np.array([first.x + 0.2, first.y + 0.2, 0.05]))
        self.assertNotIn(seen[0], anchors_in_view(anchors, lens, tilt, *pose, blockers=[over_one]))

    def test_a_heading_turns_away_from_a_view_with_only_hidden_tags(self):
        room = Obstacle("wall", (-0.7, -3.0, 0.0), (-0.5, 3.0, 2.0))
        _, tracker, _ = tracked_room(room)
        turned = tracker.allowed_yaw("a", (-1.0, 0.0, 1.0), 0.0)
        self.assertNotEqual(turned, 0.0)
        self.assertTrue(anchors_in_view(tracker.anchors, Lens.from_fov(), math.radians(45), (-1.0, 0.0, 1.0), turned, blockers=tracker.blockers()))

    def test_no_hold_point_faces_nothing_but_hidden_tags(self):
        room, tracker, _ = tracked_room()
        usable = _marker_filter(tracker)
        coverage = RegionCoverage(room, {DEFAULT_TILT}, Regions(room), usable)
        coverage.remaining()
        self.assertGreater(len(coverage.all_candidates), 0)
        blockers = tracker.blockers()
        lens, tilt = Lens.from_fov(), math.radians(45)
        for p, yaw in zip(coverage.all_candidates, coverage.all_yaws):
            self.assertFalse(LOW.lo[0] <= p[0] <= LOW.hi[0] and LOW.lo[1] <= p[1] <= LOW.hi[1], f"a viewpoint over the couch: {p}")
            self.assertTrue(anchors_in_view(tracker.anchors, lens, tilt, p, yaw, blockers=blockers))

    def test_a_newly_listed_obstacle_takes_back_viewpoints_it_hides_and_a_sighting_doesnt(self):
        room, tracker, _ = tracked_room()
        coverage = RegionCoverage(room, {DEFAULT_TILT}, Regions(room), _marker_filter(tracker))
        coverage.remaining()
        answers = coverage.usable_known
        room.add(Obstacle("seen-cup-1", (1.5, 1.5, 0.0), (1.7, 1.7, 0.3)))
        coverage.remaining()
        self.assertIs(coverage.usable_known, answers)
        room.add(Obstacle("crate", (1.0, 1.0, 0.0), (1.5, 1.5, 0.5)))
        coverage.remaining()
        self.assertIsNot(coverage.usable_known, answers)

    def test_views_of_furniture_stay_off_it_and_keep_a_tag(self):
        room, tracker, _ = tracked_room()
        coverage = RegionCoverage(room, {DEFAULT_TILT}, Regions(room), _marker_filter(tracker))
        coverage.remaining()
        lo, hi = np.array(LOW.lo), np.array(LOW.hi)
        manager = MissionManager(Hub(Coordinator(room, [DroneConfig("a", (-2.0, 0.0, 0.0))], {"a": FakeLink()}, tracker=tracker), Memory(), FakeEyes()))
        thing = manager._furniture()[0]
        self.assertEqual(thing.size[2], 0.5, "an inspection still aims at its true height")
        views = views_of(thing, coverage, DEFAULT_TILT, 4)
        self.assertTrue(views)
        for view in views:
            self.assertFalse(np.all(view.p[:2] >= lo[:2]) and np.all(view.p[:2] <= hi[:2]), f"a view over it: {view.p}")
            self.assertTrue(anchors_in_view(tracker.anchors, Lens.from_fov(), math.radians(45), view.p, view.yaw, blockers=tracker.blockers()))


if __name__ == "__main__":
    unittest.main()
