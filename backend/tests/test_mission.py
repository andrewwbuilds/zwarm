"""The mission manager: regions, what it refuses, losing drones, the flight limit, and whole simulated missions."""

import logging
import math
import threading
import time
import unittest
from unittest import mock

import numpy as np

from mapping.fleet import safe_name
from swarm import mission_bench
from swarm.control import Pose
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.eyes import DEFAULT_TILT, in_view
from swarm.frames import Frame
from swarm.hub import Hub
from swarm.inspect import FindingStore, Inspector
from swarm.memory import Memory
from swarm.mission import (REGION_NAMES, LOOK_EVERY, MissionError, MissionManager, RegionCoverage, Regions, Survey, Sweep,
                           views_of)
from swarm.room import Obstacle, Room
from swarm.search import LOOKS_NEEDED, SEARCH_Z


def room():
    return Room((-4, -3, 0), (4, 3, 2.6))


class FakeLink:
    connected = True

    def __init__(self):
        self.commands = []

    def send_sticks(self, *sticks):
        pass

    def command(self, name):
        self.commands.append(name)
        return True


class FakeEyes:
    def knows(self, label):
        return label != "unicorn"

    def look(self, drone_id, now):
        return []


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.regions = Regions(room())

    def inside(self, name, x, y):
        return bool(self.regions.mask(name, np.array([[x, y, 0.1]]))[0])

    def test_the_four_halves(self):
        self.assertTrue(self.inside("north", 2, 0) and not self.inside("north", -2, 0))
        self.assertTrue(self.inside("south", -2, 0) and not self.inside("south", 2, 0))
        self.assertTrue(self.inside("east", 0, -2) and not self.inside("east", 0, 2))
        self.assertTrue(self.inside("west", 0, 2) and not self.inside("west", 0, -2))

    def test_quarters_are_both_halves(self):
        self.assertTrue(self.inside("north-east", 2, -2))
        for x, y in ((-2, -2), (2, 2), (-2, 2)):
            self.assertFalse(self.inside("north-east", x, y))
        self.assertTrue(self.inside("south-west", -2, 2))

    def test_center_and_all(self):
        self.assertTrue(self.inside("center", 0.5, 0.5))
        self.assertFalse(self.inside("center", 3.5, 0.5))
        self.assertFalse(self.inside("center", 0.5, 2.5))
        self.assertTrue(self.inside("all", 3.9, -2.9))

    def test_every_point_is_in_exactly_one_half_of_each_pair(self):
        points = np.random.default_rng(1).uniform((-4, -3, 0), (4, 3, 1), (200, 3))
        self.assertTrue(np.all(self.regions.mask("north", points) ^ self.regions.mask("south", points)))
        self.assertTrue(np.all(self.regions.mask("east", points) ^ self.regions.mask("west", points)))

    def test_north_can_be_turned(self):
        regions = Regions(room(), north_yaw_deg=90.0)  # north is +y, so facing it, east is +x
        self.assertTrue(regions.mask("north", np.array([[0.0, 2.0, 0.0]]))[0])
        self.assertTrue(regions.mask("east", np.array([[2.0, 0.0, 0.0]]))[0])
        self.assertFalse(regions.mask("north", np.array([[2.0, -1.0, 0.0]]))[0])

    def test_a_fence_off_center_counts_from_its_own_middle(self):
        regions = Regions(Room((1, 1, 0), (5, 3, 2)))
        self.assertTrue(regions.mask("north", np.array([[3.5, 2.0, 0.0]]))[0])
        self.assertFalse(regions.mask("north", np.array([[2.5, 2.0, 0.0]]))[0])

    def test_things_people_say(self):
        for said, name in (("the north side", "north"), ("North", "north"), ("north east", "north-east"), ("northeast", "north-east"),
                           ("NE", "north-east"), ("south-west corner", "south-west"), ("east north", "north-east"), ("the middle", "center"),
                           ("centre of the room", "center"), ("everything", "all"), ("the entire area", "all"), ("whole room", "all"),
                           (None, "all"), ("", "all"), ("west half", "west")):
            self.assertEqual(self.regions.resolve(said), name, said)

    def test_an_unknown_region_is_a_sentence_listing_the_real_ones(self):
        with self.assertRaises(MissionError) as caught:
            self.regions.resolve("the moon")
        self.assertIn("the moon", str(caught.exception))
        self.assertIn("north-east", str(caught.exception))


class RegionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.room = room()
        self.coverage = RegionCoverage(self.room, {DEFAULT_TILT}, Regions(self.room))

    def see_everything(self, coverage=None):
        coverage = coverage or self.coverage
        coverage.counts[:] = LOOKS_NEEDED

    def test_only_the_regions_floor_is_asked_for(self):
        self.coverage.set_region("north")
        self.assertTrue(np.all(self.coverage.cells[self.coverage.remaining()][:, 0] > 0.0))
        self.assertGreater(self.coverage.remaining().sum(), 40)

    def test_viewpoints_stay_inside_the_region(self):
        self.coverage.set_region("west")
        view = self.coverage.best(np.array([0.0, -2.0, 1.0]), 0.0, np.zeros(len(self.coverage.cells), bool))
        self.assertGreaterEqual(view.p[1], 0.0)

    def test_share_counts_only_the_region(self):
        self.coverage.set_region("north")
        self.coverage.counts[self.coverage.cells[:, 0] < 0.0] = LOOKS_NEEDED
        self.assertEqual(self.coverage.share_seen(), 0.0)
        self.coverage.counts[self.coverage.cells[:, 0] > 0.0] = LOOKS_NEEDED
        self.assertEqual(self.coverage.share_seen(), 1.0)

    def test_what_was_seen_survives_a_change_of_region(self):
        self.coverage.set_region("north")
        self.see_everything()
        self.coverage.set_region("south")
        self.assertEqual(self.coverage.share_seen(), 1.0)
        self.assertEqual(self.coverage.stats("north")["share_seen"], 1.0)

    def test_looking_again_keeps_the_record_that_it_was_seen(self):
        self.coverage.set_region("north")
        self.see_everything()
        self.coverage.forget_region()
        self.assertEqual(self.coverage.share_seen(), 0.0)
        self.assertEqual(self.coverage.stats("north")["share_seen"], 1.0)

    def test_a_new_obstacle_reopens_only_the_floor_near_it(self):
        self.see_everything()
        self.coverage.obstacle_appeared((1.0, 1.0, 0.0), (1.5, 1.5, 0.5))
        self.assertLess(self.coverage.stats("all")["share_seen"], 1.0)
        self.assertEqual(self.coverage.stats("south")["share_seen"], 1.0)

    def test_unusable_viewpoints_are_not_offered_and_their_floor_is_not_expected(self):
        coverage = RegionCoverage(self.room, {DEFAULT_TILT}, Regions(self.room), usable=lambda p, yaw: p[0] < 0.0)
        coverage.remaining()
        self.assertTrue(np.all(coverage.all_candidates[:, 0] < 0.0))
        self.assertGreater(coverage.stats("all")["out_of_view_m2"], 0.0)

    def test_a_broken_viewpoint_check_doesnt_lose_the_viewpoints(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        def broken(p, yaw):
            raise RuntimeError("no")
        coverage = RegionCoverage(self.room, {DEFAULT_TILT}, Regions(self.room), usable=broken)
        coverage.remaining()
        self.assertGreater(len(coverage.all_candidates), 100)

    def test_suggestion_shows_unseen_floor_in_that_region(self):
        view = self.coverage.suggest("east")
        self.assertLess(view.p[1], 0.0)
        self.assertGreater(len(view.cells), 0)
        self.see_everything()
        self.assertIsNone(self.coverage.suggest("east"))

    def test_the_last_patch_still_gets_a_viewpoint(self):
        self.see_everything()
        self.coverage.counts[10] = 0
        claimed = np.zeros(len(self.coverage.cells), bool)
        self.assertIsNone(self.coverage.best(np.zeros(3), 0.0, claimed), "best() alone leaves one patch")
        view = self.coverage.leftovers(np.zeros(3), claimed, (), 0.7, (), DEFAULT_TILT)
        self.assertEqual(view.cells.tolist(), [10])

    def test_a_few_views_cover_a_region(self):
        views = self.coverage.cover("north", 12)
        self.assertLessEqual(len(views), 12)
        shown = np.zeros(len(self.coverage.cells), bool)
        for view in views:
            shown[view.cells] = True
        north = self.coverage.regions.mask("north", self.coverage.cells)
        self.assertGreater(shown[north].mean(), 0.9)
        self.assertTrue(all(view.p[0] >= 0.0 for view in views))


class ViewsOfTests(unittest.TestCase):
    def setUp(self):
        self.room = room()
        self.coverage = RegionCoverage(self.room, {DEFAULT_TILT}, Regions(self.room))
        self.coverage.remaining()

    def thing(self, x=0.0, y=0.0, size=(0.4, 0.4, 0.5), base=0.0):
        memory = Memory()
        for i in range(3):
            memory.record("chair", (x, y, base + size[2] / 2), size, 0.9, 1.0, "a", float(i))
        return next(iter(memory.things.values()))

    def test_views_put_it_in_the_picture_from_different_sides(self):
        thing = self.thing()
        views = views_of(thing, self.coverage, DEFAULT_TILT, 3)
        self.assertEqual(len(views), 3)
        for view in views:
            self.assertTrue(in_view(view.p, view.yaw, thing.p, tilt=DEFAULT_TILT)[0])
            self.assertGreater(float(np.linalg.norm(view.p[:2] - thing.p[:2])), 0.4)
        sides = sorted(math.atan2(*(view.p[:2] - thing.p[:2])[::-1]) for view in views)
        self.assertGreater(min(b - a for a, b in zip(sides, sides[1:])), math.radians(60))

    def test_no_view_looks_up_at_it(self):
        """Anything above the camera's horizon can't be in the picture, whatever the distance."""
        for view in views_of(self.thing(size=(0.4, 0.4, 1.0)), self.coverage, DEFAULT_TILT, 3):
            self.assertGreater(view.p[2], 0.5)

    def test_something_up_at_the_ceiling_has_no_view(self):
        low_room = Room((-4, -3, 0), (4, 3, 2.0))
        coverage = RegionCoverage(low_room, {DEFAULT_TILT}, Regions(low_room))
        coverage.remaining()
        self.assertEqual(views_of(self.thing(size=(0.4, 0.4, 0.3), base=1.65), coverage, DEFAULT_TILT, 3), [])

    def test_stays_clear_of_walls_and_things_in_the_way(self):
        self.room.add(Obstacle("wall", (-0.9, -1.5, 0.0), (-0.7, 1.5, 2.6)))
        thing = self.thing()
        for view in views_of(thing, self.coverage, DEFAULT_TILT, 3):
            self.assertGreater(view.p[0], -0.7, "that view is hidden behind the wall")
            self.assertGreaterEqual(self.room.clearance(view.p), self.coverage.clearance_needed)


class Fleet:
    """Two drones on scripted poses: nothing here simulates flight, so it's quick and exact."""

    def __init__(self, phase=Phase.AIR, **kwargs):
        self.room = room()
        self.links = {"a": FakeLink(), "b": FakeLink()}
        configs = [DroneConfig("a", (-2.0, -1.0, 0.0)), DroneConfig("b", (-2.0, 1.0, 0.0))]
        self.co = Coordinator(self.room, configs, self.links)
        self.memory = Memory()
        self.hub = Hub(self.co, self.memory, FakeEyes())
        self.events = []
        self.missions = MissionManager(self.hub, on_event=lambda kind, text, data: self.events.append((kind, text, data)), **kwargs)
        self.t = 0.0
        self.poses = {"a": (-2.0, -1.0, 1.0), "b": (-2.0, 1.0, 1.0)}
        for d in self.co.drones.values():
            d.phase = phase

    def run(self, seconds=0.5):
        for _ in range(max(1, round(seconds / 0.05))):
            self.t += 0.05
            self.hub.tick(self.t, {k: Pose(self.t, v, 0.0) for k, v in self.poses.items()})
            self.missions.tick(self.t)

    def told(self, event):
        self.missions.settle()
        return [data for _, _, data in self.events if data.get("event") == event]

    def seen_thing(self, label="chair", x=2.0, y=0.0):
        for i in range(3):
            self.memory.record(label, (x, y, 0.25), (0.4, 0.4, 0.5), 0.9, 1.0, "a", float(i))


class RefusalTests(unittest.TestCase):
    def assert_refused(self, words, call, *args):
        with self.assertRaises(MissionError) as caught:
            call(*args)
        self.assertIn(words, str(caught.exception))

    def test_nothing_flying(self):
        f = Fleet(Phase.GROUND)
        self.assert_refused("Take off first", f.missions.start, "explore")
        self.assertEqual(f.missions.status()["state"], "idle")
        self.assertIsNone(f.hub.search)

    def test_taking_off_is_enough_to_start(self):
        f = Fleet(Phase.GROUND)
        f.co.takeoff_queue = ["a", "b"]
        self.assertEqual(f.missions.start("explore")["state"], "starting")

    def test_a_mission_never_takes_off_by_itself(self):
        f = Fleet(Phase.GROUND)
        with self.assertRaises(MissionError):
            f.missions.start("map")
        f.run(1.0)
        for link in f.links.values():
            self.assertEqual(link.commands, [])

    def test_unknown_kind_and_region(self):
        f = Fleet()
        f.run()
        self.assert_refused("explore", f.missions.start, "backflip")
        self.assert_refused("the moon", f.missions.start, "explore", "the moon")

    def test_map_needs_a_capture(self):
        f = Fleet()
        f.run()
        self.assert_refused("explore instead", f.missions.start, "map")

    def test_search_needs_something_the_camera_knows(self):
        f = Fleet()
        f.run()
        self.assert_refused("what to search for", f.missions.start, "search")
        self.assert_refused("unicorn", f.missions.start, "search", "all", "unicorn")

    def test_inspect_needs_something_seen(self):
        f = Fleet()
        f.run()
        self.assert_refused("haven't seen", f.missions.start, "inspect", "all", "chair")
        f.seen_thing(x=2.0)
        self.assert_refused("haven't seen a chair in", f.missions.start, "inspect", "south", "chair")
        self.assertEqual(f.missions.start("inspect", "north", "chair")["target"], "chair")

    def test_inspect_says_so_when_a_thing_is_too_high_to_see(self):
        f = Fleet()
        f.room.hi[2] = 2.0
        f.run()
        for i in range(3):
            f.memory.record("clock", (2.0, 0.0, 1.8), (0.3, 0.3, 0.3), 0.9, 1.0, "a", float(i))
        self.assert_refused("can't look up", f.missions.start, "inspect", "all", "clock")

    def test_too_little_flight_time_left(self):
        f = Fleet(flight_limit_s=20.0)
        f.run(8.0)
        self.assert_refused("flight limit", f.missions.start, "explore")

    def test_focus_needs_a_mission_and_a_real_region(self):
        f = Fleet()
        f.run()
        self.assert_refused("no mission", f.missions.focus, "north")
        f.missions.start("explore")
        self.assert_refused("the moon", f.missions.focus, "the moon")
        self.assertEqual(f.missions.status()["region"], "all")

    def test_a_refused_change_leaves_the_running_mission_alone(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "north")
        work = f.hub.search
        self.assert_refused("haven't seen", f.missions.start, "inspect", "all", "chair")
        self.assertIs(f.hub.search, work)
        self.assertEqual((f.missions.status()["kind"], f.missions.status()["state"]), ("explore", "running"))


class StatusTests(unittest.TestCase):
    def test_status_has_every_field_and_can_be_sent_as_json(self):
        import json
        f = Fleet()
        f.run()
        status = f.missions.start("explore", "north")
        for key in ("kind", "state", "region", "share_seen", "target", "elapsed_s", "tasks", "flight_s"):
            self.assertIn(key, status)
        self.assertEqual(status["kind"], "explore")
        self.assertEqual(status["region"], "north")
        self.assertEqual(set(status["flight_s"]), {"a", "b"})
        json.dumps(status)
        json.dumps(f.missions.gaps())

    def test_idle_status(self):
        status = Fleet(Phase.GROUND).missions.status()
        self.assertEqual((status["kind"], status["state"], status["elapsed_s"]), (None, "idle", 0.0))
        self.assertEqual(set(status["tasks"].values()), {"on the ground"})

    def test_gaps_list_every_region_with_a_place_to_look_next(self):
        f = Fleet()
        f.run()
        gaps = f.missions.gaps()
        self.assertEqual([gap["name"] for gap in gaps], list(REGION_NAMES))
        for gap in gaps:
            self.assertGreaterEqual(f.room.clearance((gap["viewpoint"]["x"], gap["viewpoint"]["y"], gap["viewpoint"]["z"])), 0.3)
            self.assertEqual(gap["viewpoint"]["z"], SEARCH_Z)
        north = next(gap for gap in gaps if gap["name"] == "north")
        self.assertGreaterEqual(north["viewpoint"]["x"], 0.0)

    def test_gaps_follow_what_has_been_seen(self):
        f = Fleet()
        f.run()
        f.missions.coverage.counts[f.missions.coverage.cells[:, 0] > 0.0] = LOOKS_NEEDED
        gaps = {gap["name"]: gap for gap in f.missions.gaps()}
        self.assertEqual(gaps["north"]["share_seen"], 1.0)
        self.assertIsNone(gaps["north"]["viewpoint"])
        self.assertIsNotNone(gaps["south"]["viewpoint"])
        self.assertLess(gaps["south"]["share_seen"], 0.5)
        self.assertAlmostEqual(gaps["all"]["share_seen"], 0.5, delta=0.2)

    def test_floor_seen_with_no_mission_running_still_counts(self):
        f = Fleet()
        f.run(1.5)
        self.assertGreater(f.missions.coverage.stats("south")["share_seen"], 0.0)


class FlightLimitTests(unittest.TestCase):
    def test_a_drone_turns_back_early_enough_to_land_by_the_limit(self):
        f = Fleet(flight_limit_s=40.0)
        f.poses["a"] = (2.0, -1.0, 1.0)  # 4 m out: 10 s back at the cautious speed, 18 s with the landing
        f.run(40.0 - 18.0 - 1.0)
        self.assertFalse(f.co.drones["a"].landing_requested)
        f.run(1.5)
        a = f.co.drones["a"]
        self.assertTrue(a.landing_requested)
        self.assertEqual(a.goal[:2].tolist(), [-2.0, -1.0], "it should head for its pad")
        told = f.told("flight_limit")
        self.assertEqual([data["drone"] for data in told], ["a"])

    def test_only_that_drone_lands(self):
        f = Fleet(flight_limit_s=40.0)
        f.poses["a"] = (2.0, -1.0, 1.0)
        f.run(25.0)
        self.assertTrue(f.co.drones["a"].landing_requested)
        self.assertFalse(f.co.drones["b"].landing_requested)

    def test_a_warning_comes_first(self):
        f = Fleet(flight_limit_s=60.0)
        f.run(30.0)
        warnings = f.told("flight_warning")
        self.assertEqual(sorted(data["drone"] for data in warnings), ["a", "b"])
        self.assertEqual(f.told("flight_limit"), [])

    def test_a_hold_cant_keep_a_drone_past_its_limit(self):
        f = Fleet(flight_limit_s=30.0)
        f.poses = {"a": (2.0, -1.0, 1.0), "b": (2.0, 1.0, 1.0)}  # too far out to have started down yet
        f.run(14.0)
        self.assertEqual(f.co.drones["a"].phase, Phase.AIR)
        self.assertTrue(f.co.drones["a"].landing_requested)
        f.hub.hold()
        self.assertFalse(f.co.drones["a"].landing_requested)
        f.run(0.2)
        self.assertTrue(f.co.drones["a"].landing_requested)
        self.assertEqual(len(f.told("flight_limit")), 2, "one report per drone, not one per tick")

    def test_a_landing_that_wasnt_seen_is_never_sent_again(self):
        """Takeoff and land share one toggle on the S20, so a second one could take off again."""
        f = Fleet(flight_limit_s=20.0)
        f.run(13.0)
        self.assertEqual(f.co.drones["a"].phase, Phase.LANDING)
        f.run(20.0)  # the pose never comes down, so the coordinator gives up on it
        self.assertEqual(f.co.drones["a"].phase, Phase.AIR)
        self.assertIn("land", f.co.drones["a"].faults)
        f.run(5.0)
        self.assertEqual(f.links["a"].commands, ["land"])

    def test_time_counts_from_takeoff_and_resets_on_the_ground(self):
        f = Fleet(Phase.GROUND)
        f.run(5.0)
        self.assertEqual(f.missions.status()["flight_s"]["a"], 0.0)
        f.co.drones["a"].phase = Phase.TAKEOFF
        f.co.drones["a"].toggled_at = f.t
        f.poses["a"] = (-2.0, -1.0, 1.0)
        f.co.drones["a"].phase = Phase.AIR
        f.run(3.0)
        self.assertAlmostEqual(f.missions.status()["flight_s"]["a"], 3.0, delta=0.2)
        self.assertEqual(f.missions.status()["flight_s"]["b"], 0.0)
        f.co.drones["a"].phase = Phase.GROUND
        f.run(0.2)
        self.assertEqual(f.missions.status()["flight_s"]["a"], 0.0)

    def test_many_short_flights_on_one_battery_get_a_warning(self):
        f = Fleet(flight_limit_s=30.0)
        for _ in range(4):
            f.co.drones["a"].phase = Phase.AIR
            f.co.drones["a"].landing_requested = False
            f.run(9.0)
            f.co.drones["a"].phase = Phase.GROUND
            f.run(0.2)
        f.co.drones["a"].phase = Phase.AIR
        f.run(1.0)
        self.assertIn("a", [data["drone"] for data in f.told("flight_warning") if "battery" in str(data) or data["flight_s"] >= 30.0])


class LosingDroneTests(unittest.TestCase):
    def test_a_short_tracking_gap_isnt_a_lost_drone(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "all")
        poses, f.poses = f.poses, {"a": f.poses["a"]}
        f.run(0.7)
        self.assertEqual(f.told("drone_lost"), [])
        f.poses = poses
        f.run(0.5)
        self.assertEqual(f.told("drone_lost"), [])

    def test_losing_tracking_is_reported_and_frees_its_work(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "all")
        f.run(0.5)
        self.assertIn("b", f.hub.orders)
        del f.poses["b"]
        f.run(2.0)
        lost = f.told("drone_lost")
        self.assertEqual([data["drone"] for data in lost], ["b"])
        self.assertEqual(lost[0]["others"], ["a"])
        self.assertNotIn("b", f.hub.orders)
        self.assertNotIn("b", f.hub.search.views)

    def test_a_lost_link_is_reported_as_that(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "all")
        f.links["b"].connected = False
        f.run(2.0)
        self.assertIn("control link", f.told("drone_lost")[0]["reason"])

    def test_a_drone_that_comes_back_takes_work_again(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "all")
        f.links["b"].connected = False
        f.run(2.0)
        f.links["b"].connected = True
        f.run(1.0)
        self.assertEqual([data["drone"] for data in f.told("drone_back")], ["b"])
        self.assertIn("b", f.hub.orders)

    def test_no_drones_left_ends_the_mission_and_says_so(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "all")
        f.links["a"].connected = f.links["b"].connected = False
        f.run(2.5)
        self.assertEqual(f.missions.state, "failed")
        self.assertIn("No drones are left", f.missions.reason)

    def test_a_drone_that_never_flew_this_mission_isnt_lost(self):
        f = Fleet()
        f.co.drones["b"].phase = Phase.GROUND
        f.run()
        f.missions.start("explore", "all")
        f.run(2.0)
        self.assertEqual(f.told("drone_lost"), [])


class OverrideTests(unittest.TestCase):
    """The hub's plain actions must end a mission by themselves, with no model and no call to this file."""

    def running(self, **kwargs):
        f = Fleet(**kwargs)
        f.run()
        f.missions.start("explore", "all")
        f.run(0.5)
        self.assertTrue(f.hub.orders)
        return f

    def test_hold(self):
        f = self.running()
        f.hub.hold()
        self.assertEqual(f.hub.orders, {})
        f.run(0.1)
        self.assertEqual(f.missions.state, "stopped")
        f.run(1.0)
        self.assertEqual(f.hub.orders, {})

    def test_land_goes_on_landing(self):
        f = self.running()
        f.hub.land()
        f.run(0.2)
        self.assertEqual(f.missions.state, "stopped")
        self.assertTrue(all(d.landing_requested for d in f.co.drones.values()))
        self.assertEqual(f.hub.orders, {})

    def test_estop(self):
        f = self.running()
        f.hub.estop()
        f.run(0.2)
        self.assertEqual(f.missions.state, "stopped")
        self.assertEqual(f.links["a"].commands, ["estop"])

    def test_a_plain_search_replaces_a_mission(self):
        f = self.running()
        f.hub.find("backpack")
        f.run(0.2)
        self.assertEqual(f.missions.state, "stopped")
        self.assertEqual(f.hub.search.label, "backpack")

    def test_stop_holds_but_leaves_a_landing_alone(self):
        f = self.running()
        d = f.co.drones["a"]
        d.landing_requested = True
        f.missions.stop()
        self.assertTrue(d.landing_requested)
        self.assertEqual(f.missions.state, "stopped")
        self.assertEqual(f.hub.orders, {})
        np.testing.assert_allclose(f.co.drones["b"].goal, f.poses["b"], atol=0.01)

    def test_stop_with_nothing_running_is_fine(self):
        f = Fleet()
        self.assertEqual(f.missions.stop()["state"], "idle")

    def test_stopping_cancels_a_takeoff_still_waiting(self):
        f = Fleet(Phase.GROUND)
        f.co.takeoff_queue = ["a", "b"]
        f.missions.start("explore")
        f.missions.stop()
        self.assertEqual(f.co.takeoff_queue, [])

    def test_a_new_mission_replaces_a_running_one(self):
        f = self.running()
        f.seen_thing()
        status = f.missions.start("inspect", "north", "chair")
        self.assertEqual((status["kind"], status["state"]), ("inspect", "running"))
        self.assertIsInstance(f.hub.search, Survey)


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_tick_never_raises(self):
        f = Fleet()
        f.run()
        f.missions.start("explore")
        with mock.patch.object(f.missions, "_watch_work", side_effect=RuntimeError("no")), \
                mock.patch.object(f.missions, "_watch_flight_time", side_effect=RuntimeError("no")):
            f.missions.tick(f.t)
        self.assertEqual(f.missions.errors, 2)

    def test_a_failure_watching_the_mission_doesnt_stop_the_flight_timer(self):
        f = Fleet(flight_limit_s=20.0)
        with mock.patch.object(f.missions, "_count_looks", side_effect=RuntimeError("no")):
            f.run(14.0)
        self.assertTrue(f.co.drones["a"].landing_requested)

    def test_a_broken_listener_doesnt_touch_the_flight_loop(self):
        f = Fleet()
        f.missions.on_event = mock.Mock(side_effect=RuntimeError("no"))
        f.run()
        f.missions.start("explore")
        f.run(0.5)
        self.assertTrue(f.missions.settle())
        self.assertEqual(f.missions.errors, 0)
        self.assertTrue(f.missions.on_event.called)

    def test_a_slow_listener_doesnt_slow_the_flight_loop(self):
        release = threading.Event()
        self.addCleanup(release.set)
        f = Fleet()
        f.missions.on_event = lambda kind, text, data: release.wait(10.0)
        f.run()
        started = time.monotonic()
        f.missions.start("explore")
        f.run(1.0)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_a_capture_that_fails_is_reported_not_raised(self):
        capture = mock.Mock()
        capture.start.side_effect = RuntimeError("disk full")
        f = Fleet(capture=capture)
        f.run()
        f.missions.start("map")
        f.run(0.5)
        f.missions.settle()
        self.assertEqual(f.missions.errors, 0)
        self.assertIn("disk full", next(text for kind, text, _ in f.events if kind == "fault"))
        self.assertEqual(f.missions.state, "running")

    def test_a_slow_capture_doesnt_hold_the_lock(self):
        release = threading.Event()
        self.addCleanup(release.set)
        capture = mock.Mock()
        capture.start.side_effect = lambda name: release.wait(10.0)
        f = Fleet(capture=capture)
        f.run()
        started = time.monotonic()
        f.missions.start("map")
        f.run(1.0)
        self.assertLess(time.monotonic() - started, 2.0)


class MapTests(unittest.TestCase):
    def test_the_map_name_is_one_the_capture_accepts(self):
        capture = mock.Mock()
        f = Fleet(capture=capture)
        f.run()
        f.missions.start("map")
        f.missions.settle()
        safe_name(capture.start.call_args.args[0])

    def test_a_map_flies_the_floor_again_even_if_seen_before(self):
        capture = mock.Mock()
        f = Fleet(capture=capture)
        f.run()
        f.missions.coverage.counts[:] = LOOKS_NEEDED
        status = f.missions.start("map", "north")
        self.assertEqual(status["share_seen"], 0.0)

    def test_explore_carries_on_from_what_was_seen(self):
        f = Fleet()
        f.run()
        coverage = f.missions.coverage
        coverage.counts[coverage.cells[:, 1] > 0.0] = LOOKS_NEEDED
        self.assertGreater(f.missions.start("explore", "all")["share_seen"], 0.4)

    def test_build_starts_when_the_mission_is_stopped(self):
        capture = mock.Mock()
        f = Fleet(capture=capture)
        f.run()
        f.missions.start("map")
        f.missions.settle()
        name = capture.start.call_args.args[0]
        capture.build.assert_not_called()
        f.missions.stop()
        f.missions.settle()
        capture.stop.assert_called_once()
        capture.build.assert_called_once_with(name)

    def test_an_explore_touches_no_capture(self):
        capture = mock.Mock()
        f = Fleet(capture=capture)
        f.run()
        f.missions.start("explore")
        f.missions.stop()
        f.missions.settle()
        capture.start.assert_not_called()
        capture.build.assert_not_called()


class SimpleFrames:
    def __init__(self, jpeg=b"not really a jpeg"):
        self.jpeg = jpeg
        self.number = 0
        self.dead = set()

    def latest(self, drone_id):
        if drone_id in self.dead:
            return None
        self.number += 1
        return Frame(drone_id, 0.0, self.jpeg, self.number)


class FramesCountTests(unittest.TestCase):
    def test_a_look_is_a_new_picture_even_when_the_detector_has_nothing(self):
        f = Fleet(frames=SimpleFrames())
        f.hub.eyes.look = lambda drone_id, now: None
        f.run(1.5)
        self.assertGreater(f.missions.coverage.stats("south")["share_seen"], 0.0)

    def test_a_dead_camera_sees_nothing(self):
        frames = SimpleFrames()
        frames.dead = {"a", "b"}
        f = Fleet(frames=frames)
        f.run(1.5)
        self.assertEqual(f.missions.coverage.stats("all")["share_seen"], 0.0)

    def test_the_hub_doesnt_count_the_same_pictures_again(self):
        f = Fleet(frames=SimpleFrames())
        f.run()
        f.missions.start("explore")
        before = f.missions.coverage.counts.copy()
        for _ in range(8):
            f.t += 0.05
            f.hub.tick(f.t, {k: Pose(f.t, v, 0.0) for k, v in f.poses.items()})
        np.testing.assert_array_equal(f.missions.coverage.counts, before)
        self.assertGreater(LOOK_EVERY, 0.0)


class FakeTracker:
    def __init__(self):
        from swarm.tracking import DroneOptics
        self.anchors = {}
        self.drones = {"a": DroneOptics(-2.0, -1.0), "b": DroneOptics(-2.0, 1.0)}
        self.turned = []

    def allowed_yaw(self, drone_id, p, yaw):
        self.turned.append(drone_id)
        return yaw

    def expected(self, drone_id):
        return ["x"]

    def note_command(self, *args):
        pass


class TagFilterTests(unittest.TestCase):
    def test_with_a_tracker_a_viewpoint_must_show_a_tag(self):
        tracker = FakeTracker()
        f = Fleet()
        f.co.tracker = tracker
        with mock.patch("swarm.tracking.anchors_in_view", side_effect=lambda anchors, lens, tilt, p, yaw, **kw: ["t"] if p[0] < 0 else []):
            missions = MissionManager(f.hub)
        self.assertGreater(len(missions.coverage.all_candidates), 50)
        self.assertTrue(np.all(missions.coverage.all_candidates[:, 0] < 0.0))
        f.hub.aim(f.co.drones["a"], 1.0)
        self.assertEqual(tracker.turned, ["a"])

    def test_with_no_tag_anywhere_a_mission_is_refused(self):
        f = Fleet()
        f.co.tracker = FakeTracker()
        with mock.patch("swarm.tracking.anchors_in_view", return_value=[]):
            missions = MissionManager(f.hub)
            f.run()
            with self.assertRaises(MissionError) as caught:
                missions.start("explore")
        self.assertIn("No place", str(caught.exception))


class InspectorHookTests(unittest.TestCase):
    def inspect_flight(self):
        f = Fleet(inspector=mission_bench.FakeInspector())
        f.seen_thing()
        f.run()
        return f

    def test_inspector_runs_for_an_inspection_only(self):
        f = self.inspect_flight()
        f.missions.start("explore")
        self.assertEqual(f.missions.inspector.started, [])
        f.missions.start("inspect", "all", "chair")
        self.assertEqual(f.missions.inspector.started, ["chair"])
        self.assertTrue(f.missions.inspector.watching)
        f.missions.stop()
        self.assertFalse(f.missions.inspector.watching)

    def test_vague_targets_mean_everything_in_the_region(self):
        f = self.inspect_flight()
        status = f.missions.start("inspect", "north", "these things")
        self.assertIsNone(status["target"])
        self.assertGreater(status["views"]["total"], 3)

    def test_a_region_given_as_the_target_is_a_region(self):
        f = self.inspect_flight()
        status = f.missions.start("inspect", "all", "the north side")
        self.assertEqual((status["region"], status["target"]), ("north", None))

    def test_aliases_are_understood(self):
        f = self.inspect_flight()
        f.seen_thing("couch", x=-1.0, y=-1.5)
        self.assertEqual(f.missions.start("inspect", "all", "the sofa")["target"], "couch")

    def test_people_arent_inspected(self):
        f = self.inspect_flight()
        f.seen_thing("person", x=1.0, y=1.0)
        with self.assertRaises(MissionError):
            f.missions.start("inspect", "all", "person")

    def test_a_forgotten_thing_is_skipped(self):
        f = self.inspect_flight()
        f.missions.start("inspect", "all", "chair")
        f.run(0.2)
        f.memory.forget_label("chair")
        f.run(0.2)
        self.assertEqual(f.missions.state, "done")
        self.assertEqual(f.missions.status()["views"]["done"], 0)

    def test_analysis_keeps_the_mission_open_until_it_finishes(self):
        f = self.inspect_flight()
        inspector = f.missions.inspector
        inspector.status = lambda: {"queued": 1, "busy": 1, "checked": 3, "findings": 1}
        f.missions.start("inspect", "all", "chair")
        for task in f.hub.search.tasks:
            task.state = "done"
        f.run(0.2)
        self.assertEqual(f.missions.state, "analyzing")
        self.assertFalse(inspector.watching, "no new photos while the last ones are checked")
        inspector.status = lambda: {"queued": 0, "busy": 0, "checked": 5, "findings": 2}
        f.run(0.2)
        self.assertEqual(f.missions.state, "done")
        self.assertIn("2 possible issues", f.missions.reason)

    def test_a_real_inspector_stores_a_pending_finding_from_a_drone_at_its_view(self):
        import cv2
        import tempfile
        rng = np.random.default_rng(1)
        ok, jpeg = cv2.imencode(".jpg", rng.integers(0, 255, (360, 640), dtype=np.uint8))
        self.assertTrue(ok)
        frames = SimpleFrames(jpeg.tobytes())
        with tempfile.TemporaryDirectory() as folder:
            store = FindingStore(folder)
            calls = []
            def analyze(jpegs, prompt):
                calls.append(prompt)
                return {"damage": True, "summary": "a crack", "severity": "high", "confidence": 0.8}
            inspector = Inspector(store, frames, analyze=analyze)
            self.addCleanup(inspector.close)
            f = Fleet(frames=frames, findings=store, inspector=inspector)
            f.seen_thing()
            f.run()
            f.missions.start("inspect", "all", "chair")
            # On the way there it gets no frames, however long that takes.
            f.run(0.5)
            self.assertEqual(calls, [])
            task = f.hub.search.tasks[0]
            task.state, task.holder, task.arrived = "looking", "a", f.t
            f.hub.orders["a"].stage = "done"
            for _ in range(60):
                f.run(0.1)
                time.sleep(0.01)
                if store.list():
                    break
            found = store.list()
            self.assertEqual(len(found), 1)
            self.assertEqual((found[0]["drone"], found[0]["status"], found[0]["target"]), ("a", "pending", "chair"))


def fly(name, seed):
    logging.disable(logging.WARNING)
    try:
        return mission_bench.fly(name, seed)
    finally:
        logging.disable(logging.NOTSET)


class SimulatedMissionTests(unittest.TestCase):
    """Whole missions on the simulator. These are simulator results, not drone results."""

    def test_every_mission(self):
        for name in mission_bench.FAST:
            with self.subTest(name):
                outcome, numbers = fly(name, 3)
                self.assertTrue(outcome.passed, outcome.notes)
                self.assertEqual(numbers["errors"], 0)
                self.assertEqual(numbers["hub_errors"], 0)

    def test_three_drones_cover_a_region_on_more_seeds(self):
        for seed in (7, 8):
            outcome, numbers = fly("explore_region", seed)
            self.assertTrue(outcome.passed, outcome.notes)
            self.assertGreaterEqual(numbers["share"], mission_bench.COVERED)
            self.assertLess(numbers["mission_s"], 120.0)

    def test_a_lost_drone_and_a_focus_on_more_seeds(self):
        for name, seed in (("lost_drone", 9), ("focus_moves_all", 10)):
            outcome, numbers = fly(name, seed)
            self.assertTrue(outcome.passed, outcome.notes)
            self.assertGreaterEqual(numbers["share"], mission_bench.COVERED)

    def test_sweep_and_survey_fit_what_the_hub_expects(self):
        f = Fleet()
        f.run()
        f.missions.start("explore", "north")
        self.assertIsInstance(f.hub.search, Sweep)
        self.assertIn("Search for unseen floor in the north side: searching", f.hub.status())


if __name__ == "__main__":
    unittest.main()
