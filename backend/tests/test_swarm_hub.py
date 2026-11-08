"""The shared memory, the camera geometry, the command phrases, the search and the hub."""

import logging
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from swarm import hub_bench
from swarm.commands import parse
from swarm.control import Pose
from swarm.coordinator import Coordinator, DroneConfig, Phase
from swarm.eyes import Detection, blocked, in_view, locate
from swarm.hub import Hub
from swarm.memory import FORGET_AFTER_MISSES, Memory
from swarm.room import Obstacle, Room
from swarm.search import LOOKS_NEEDED, Coverage


def sight(memory, label="box", p=(1.0, 1.0, 0.5), size=(0.5, 0.5, 1.0), confidence=0.9, range_m=2.0, drone="a", t=0.0):
    return memory.record(label, p, size, confidence, range_m, drone, t)


class MemoryTests(unittest.TestCase):
    def test_one_sighting_does_not_count(self):
        memory = Memory()
        sight(memory)
        self.assertEqual(memory.known(), [])
        self.assertEqual(memory.obstacles(), [])

    def test_repeat_sightings_merge_and_then_count(self):
        memory = Memory()
        for i in range(3):
            sight(memory, p=(1.0 + 0.1 * i, 1.0, 0.5), t=float(i), drone="ab"[i % 2])
        self.assertEqual(len(memory.things), 1)
        thing = memory.known()[0]
        self.assertEqual(thing.sightings, 3)
        self.assertEqual(thing.seen_by, {"a", "b"})

    def test_different_things_stay_apart(self):
        memory = Memory()
        sight(memory, label="box", p=(1.0, 1.0, 0.5))
        sight(memory, label="chair", p=(1.0, 1.0, 0.5))
        sight(memory, label="box", p=(3.0, 1.0, 0.5))
        self.assertEqual(len(memory.things), 3)

    def test_a_close_look_outweighs_a_far_one(self):
        memory = Memory()
        sight(memory, p=(2.0, 0.0, 0.5), range_m=3.5)
        thing = sight(memory, p=(1.6, 0.0, 0.5), range_m=0.8)
        self.assertLess(thing.p[0], 1.72)

    def test_looking_at_nothing_repeatedly_forgets_it(self):
        memory = Memory()
        for i in range(4):
            thing = sight(memory, t=float(i))
        for _ in range(FORGET_AFTER_MISSES - 1):
            self.assertFalse(memory.missed(thing.id))
        self.assertTrue(memory.missed(thing.id))
        self.assertEqual(memory.things, {})

    def test_a_new_sighting_clears_the_misses(self):
        memory = Memory()
        for i in range(4):
            thing = sight(memory, t=float(i))
        for _ in range(FORGET_AFTER_MISSES - 1):
            memory.missed(thing.id)
        sight(memory, t=9.0)
        self.assertFalse(memory.missed(thing.id))

    def test_a_low_confidence_thing_is_avoided_but_not_called_found(self):
        memory = Memory()
        for i in range(4):
            thing = sight(memory, confidence=0.5, t=float(i))
        self.assertIn(thing, memory.obstacles())
        self.assertFalse(thing.sure)

    def test_people_are_not_map_obstacles_and_go_stale(self):
        memory = Memory()
        for i in range(4):
            person = sight(memory, label="person", p=(0.5 * i, 0.0, 0.9), t=float(i))
        self.assertEqual(memory.obstacles(), [])
        self.assertEqual(memory.movers(3.5), [person])
        self.assertEqual(memory.movers(10.0), [])
        self.assertGreater(person.v[0], 0.1)

    def test_it_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.db"
            first = Memory(path)
            for i in range(3):
                sight(first, label="chair", p=(1.5, -0.5, 0.4), t=float(i))
            first.note(2.0, "said", "hello")
            first.close()
            second = Memory(path)
            thing = second.find("chair")[0]
            self.assertAlmostEqual(thing.p[0], 1.5)
            self.assertEqual(thing.sightings, 3)
            self.assertEqual(second.log()[0][2], "hello")
            for i in range(3, 5):
                sight(second, label="chair", p=(1.5, -0.5, 0.4), t=float(i))
            self.assertEqual(len(second.things), 1)
            second.close()


class CameraGeometryTests(unittest.TestCase):
    def test_locate_goes_along_the_heading(self):
        ahead = locate((1.0, 1.0, 1.0), math.pi / 2, Detection("x", 0.0, 0.0, 2.0, (0, 0, 0), 0.9))
        np.testing.assert_allclose(ahead, (1.0, 3.0, 1.0), atol=1e-9)
        left = locate((0.0, 0.0, 1.0), 0.0, Detection("x", math.pi / 2, 0.0, 1.0, (0, 0, 0), 0.9))
        np.testing.assert_allclose(left, (0.0, 1.0, 1.0), atol=1e-9)

    def test_what_a_level_camera_sees(self):
        p = (0.0, 0.0, 1.0)
        self.assertTrue(in_view(p, 0.0, (2.0, 0.0, 1.0))[0])
        self.assertFalse(in_view(p, 0.0, (-2.0, 0.0, 1.0))[0], "behind")
        self.assertFalse(in_view(p, 0.0, (5.0, 0.0, 1.0))[0], "too far")
        self.assertFalse(in_view(p, 0.0, (0.5, 0.0, 0.0))[0], "under the bottom of the picture")
        self.assertTrue(in_view(p, 0.0, (2.5, 0.0, 0.1))[0], "floor further off")

    def test_blocked_lines(self):
        lo, hi = (1.0, -0.5, 0.0), (2.0, 0.5, 1.0)
        ends = np.array([[3.0, 0.0, 0.5], [3.0, 2.0, 0.5], [0.5, 0.0, 0.5], [3.0, 0.0, 1.5]])
        np.testing.assert_array_equal(blocked((0.0, 0.0, 0.5), ends, lo, hi)[:3], [True, False, False])
        self.assertTrue(blocked((0.0, 0.0, 0.5), ends[3][None, :], lo, hi)[0], "still clips the box on the way up")
        self.assertFalse(blocked((0.0, 0.0, 1.2), np.array([[3.0, 0.0, 1.2]]), lo, hi)[0], "over the top, parallel")


class CommandTests(unittest.TestCase):
    """The phrase table. These were written alongside the parser, so they guard against regressions, nothing more."""

    TABLE = [
        ("take off", "takeoff"), ("Takeoff!", "takeoff"), ("please take off", "takeoff"), ("launch", "takeoff"),
        ("land", "land"), ("come back", "land"), ("go home", "land"), ("return to base", "land"),
        ("land here", "land_here"), ("land now", "land_here"), ("land where you are", "land_here"),
        ("stop", "hold"), ("STOP.", "hold"), ("freeze", "hold"), ("hold position", "hold"), ("cancel", "hold"),
        ("stop searching", "hold"), ("ok, stop", "hold"),
        ("emergency stop", "estop"), ("e-stop", "estop"), ("estop", "estop"), ("kill the motors", "estop"),
        ("status", "status"), ("what are you doing?", "status"),
        ("what have you seen?", "list"), ("what's in the room", "list"), ("what do you remember", "list"),
        ("help", "help"), ("line up", "form"), ("form a ring", "form"), ("make a circle", "form"), ("spread out", "form"),
        ("forget everything", "forget"),
    ]

    def test_verbs(self):
        for text, verb in self.TABLE:
            with self.subTest(text):
                self.assertEqual(parse(text, ("a", "b")).verb, verb)

    def test_things_named(self):
        for text, verb, label in [
            ("find the backpack", "find", "backpack"), ("Find my bag please", "find", "backpack"),
            ("look for a chair", "find", "chair"), ("search for the potted plant", "find", "potted plant"),
            ("locate someone", "find", "person"), ("go find the sofa", "find", "couch"),
            ("where is the chair?", "where", "chair"), ("where's my phone", "where", "cell phone"),
            ("have you seen a bottle", "where", "bottle"), ("is there a dog", "where", "dog"),
            ("forget the box", "forget", "box"),
        ]:
            with self.subTest(text):
                command = parse(text, ("a", "b"))
                self.assertEqual((command.verb, command.label), (verb, label))

    def test_places(self):
        for text, drone, point in [
            ("go to 1 2", None, (1.0, 2.0)), ("fly to 1.5, -2", None, (1.5, -2.0)), ("go to -1 -2 1.2", None, (-1.0, -2.0, 1.2)),
            ("drone a go to 3 0", "a", (3.0, 0.0)), ("a, go to 3 0", "a", (3.0, 0.0)), ("drone b move to 0.5 0.5", "b", (0.5, 0.5)),
        ]:
            with self.subTest(text):
                command = parse(text, ("a", "b"))
                self.assertEqual((command.verb, command.drone, command.point), ("goto", drone, point))
        command = parse("drone a go to the chair", ("a", "b"))
        self.assertEqual((command.verb, command.drone, command.label), ("goto", "a", "chair"))

    def test_anything_else_does_nothing(self):
        for text in ["", "   ", "do a backflip", "land on the table", "stop the music", "hello", "take off your hat", "kill", "emergency",
                     "go", "find", "fly away", "cut", "motors"]:
            with self.subTest(text):
                self.assertEqual(parse(text, ("a", "b")).verb, "unknown")

    def test_stop_never_cuts_the_motors(self):
        for text in ["stop", "stop!", "stop stop", "please stop", "freeze", "abort", "hold it"]:
            with self.subTest(text):
                self.assertNotEqual(parse(text, ("a", "b")).verb, "estop")

    def test_fleet_commands_refuse_to_be_aimed_at_one_drone(self):
        for text in ["drone a land", "drone b take off", "a, land here", "drone a emergency stop", "drone a stop", "a, find the bag"]:
            with self.subTest(text):
                self.assertEqual(parse(text, ("a", "b")).verb, "refused")


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.room = Room((-4, -3, 0), (4, 3, 2.6))

    def test_looking_needs_several_frames(self):
        coverage = Coverage(self.room)
        start = int(coverage.remaining().sum())
        coverage.look((0.0, 0.0, 1.0), 0.0)
        self.assertEqual(int(coverage.remaining().sum()), start)
        for _ in range(LOOKS_NEEDED):
            coverage.look((0.0, 0.0, 1.0), 0.0)
        self.assertLess(int(coverage.remaining().sum()), start)

    def test_a_wall_of_furniture_hides_the_floor_behind_it(self):
        self.room.add(Obstacle("wall", (1.0, -3.0, 0.0), (1.2, 3.0, 2.6)))
        coverage = Coverage(self.room)
        for _ in range(LOOKS_NEEDED):
            coverage.look((0.0, 0.0, 1.0), 0.0)
        behind = coverage.cells[:, 0] > 1.2
        self.assertEqual(int(coverage.counts[behind].sum()), 0)

    def test_a_second_drone_goes_somewhere_else(self):
        coverage = Coverage(self.room)
        claimed = np.zeros(len(coverage.cells), bool)
        first = coverage.best((0.0, 0.0, 1.0), 0.0, claimed)
        claimed[first.cells] = True
        second = coverage.best((0.0, 0.0, 1.0), 0.0, claimed)
        overlap = np.intersect1d(first.cells, second.cells)
        self.assertEqual(len(overlap), 0)

    def test_a_new_obstacle_sends_drones_back_to_look(self):
        coverage = Coverage(self.room)
        for _ in range(LOOKS_NEEDED):
            coverage.look((0.0, 0.0, 1.0), 0.0)
        before = int(coverage.remaining().sum())
        coverage.obstacle_appeared((2.0, -0.5, 0.0), (2.5, 0.5, 1.0))
        self.assertGreater(int(coverage.remaining().sum()), before)

    def test_it_stops_when_nothing_is_left(self):
        coverage = Coverage(self.room)
        coverage.counts[:] = LOOKS_NEEDED
        self.assertIsNone(coverage.best((0.0, 0.0, 1.0), 0.0, np.zeros(len(coverage.cells), bool)))
        self.assertEqual(coverage.share_seen(), 1.0)


class FakeLink:
    connected = True

    def __init__(self):
        self.commands = []
        self.sticks = []

    def send_sticks(self, *sticks):
        self.sticks.append(sticks)

    def command(self, name):
        self.commands.append(name)
        return True


class FakeEyes:
    def __init__(self):
        self.next = {}
        self.broken = False

    def knows(self, label):
        return label != "unicorn"

    def look(self, drone_id, now):
        if self.broken:
            raise RuntimeError("camera exploded")
        return self.next.get(drone_id, [])


class HubRuleTests(unittest.TestCase):
    def setUp(self):
        self.room = Room((-4, -3, 0), (4, 3, 2.6))
        self.links = {"a": FakeLink(), "b": FakeLink()}
        configs = [DroneConfig("a", (-2.0, -1.0, 0.0)), DroneConfig("b", (-2.0, 1.0, 0.0))]
        self.co = Coordinator(self.room, configs, self.links)
        self.eyes = FakeEyes()
        self.memory = Memory()
        self.hub = Hub(self.co, self.memory, self.eyes)
        self.t = 0.0
        self.poses = {"a": (-2.0, -1.0, 1.0), "b": (-2.0, 1.0, 1.0)}
        for d in self.co.drones.values():
            d.phase = Phase.AIR

    def run_ticks(self, count=10):
        for _ in range(count):
            self.t += 0.05
            self.hub.tick(self.t, {k: Pose(self.t, v, 0.0) for k, v in self.poses.items()})

    def box_ahead_of_a(self):
        return [Detection("couch", 0.0, 0.0, 2.0, (0.6, 0.6, 1.0), 0.9)]

    def test_what_one_drone_sees_goes_on_everyones_map(self):
        self.eyes.next["a"] = self.box_ahead_of_a()
        self.co.drones["b"].goal = np.array([2.0, -1.0, 1.0])
        self.run_ticks(20)
        self.assertEqual(len(self.room.obstacles), 1)
        lo, hi = next(iter(self.room.obstacles.values())).lo, next(iter(self.room.obstacles.values())).hi
        self.assertLess(lo[0], 0.0)
        self.assertGreater(hi[0], 0.0)
        self.assertIn("a", self.memory.known()[0].seen_by)

    def test_a_single_false_detection_changes_no_map(self):
        self.eyes.next["a"] = self.box_ahead_of_a()
        self.run_ticks(1)
        self.eyes.next["a"] = []
        self.run_ticks(20)
        self.assertEqual(self.room.obstacles, {})

    def test_an_obstacle_comes_off_the_map_when_it_is_gone(self):
        self.eyes.next["a"] = self.box_ahead_of_a()
        self.run_ticks(20)
        self.assertEqual(len(self.room.obstacles), 1)
        self.eyes.next["a"] = []
        self.run_ticks(100)
        self.assertEqual(self.room.obstacles, {})

    def test_a_new_flight_starts_with_what_the_last_one_learned(self):
        self.eyes.next["a"] = self.box_ahead_of_a()
        self.run_ticks(20)
        co = Coordinator(Room((-4, -3, 0), (4, 3, 2.6)), [DroneConfig("a", (-2.0, -1.0, 0.0))], {"a": FakeLink()})
        Hub(co, self.memory, FakeEyes())
        self.assertEqual(len(co.room.obstacles), 1)

    def test_a_person_is_avoided_as_a_moving_thing_not_mapped(self):
        self.eyes.next["a"] = [Detection("person", 0.0, 0.0, 2.0, (0.5, 0.5, 1.7), 0.9)]
        self.run_ticks(20)
        self.assertEqual(self.room.obstacles, {})
        self.assertEqual([m.id.split("-")[0] for m in self.co.moving], ["seen"])

    def test_a_broken_camera_does_not_stop_the_sticks(self):
        self.eyes.broken = True
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        before = len(self.links["a"].sticks)
        self.run_ticks(10)
        self.assertGreater(len(self.links["a"].sticks), before)
        self.assertGreater(self.hub.think_errors, 0)

    def test_answers(self):
        self.run_ticks(1)
        self.assertIn("haven't seen", self.hub.command("where is the chair"))
        self.assertIn("Nothing yet", self.hub.command("what have you seen"))
        self.assertIn("didn't understand", self.hub.command("do a flip"))
        self.assertIn("can't recognise", self.hub.command("find the unicorn"))
        self.assertIn("no drone called z", self.hub.command("drone z go to 1 1"))
        self.assertIn("too close", self.hub.command("go to 3.95 0"))
        self.eyes.next["a"] = self.box_ahead_of_a()
        self.run_ticks(20)
        self.assertIn("(0.0, -1.0)", self.hub.command("where is the sofa"))
        self.assertIn("seen just now by a", self.hub.command("where is the couch"))
        self.assertIn("Forgot 1 couch", self.hub.command("forget the couch"))
        self.assertEqual(self.room.obstacles, {})

    def test_emergency_stop_reaches_every_drone(self):
        self.run_ticks(1)
        self.hub.command("emergency stop")
        for link in self.links.values():
            self.assertEqual(link.commands, ["estop"])

    def test_commands_that_do_nothing_send_nothing(self):
        self.run_ticks(1)
        for text in ["do a flip", "drone a land", "where is the chair", "status", "what have you seen"]:
            self.hub.command(text)
        for link in self.links.values():
            self.assertEqual(link.commands, [])

    def test_actions_are_plain_methods_taking_plain_values(self):
        self.run_ticks(1)
        self.assertIn("a is going to", self.hub.go_to(-1.0, -1.0, drone="a"))
        self.assertIn("Looking for a chair", self.hub.find("chair"))
        self.assertEqual(self.hub.search.label, "chair")
        self.assertIn("Holding", self.hub.hold())
        self.assertEqual(self.hub.search.state, "stopped")
        self.assertIn("haven't seen", self.hub.where("chair"))
        self.assertIn("Nothing yet", self.hub.seen())
        self.assertIn("nothing about", self.hub.forget("chair"))
        self.assertIn("a air", self.hub.status())

    def test_a_failed_action_still_answers(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        with mock.patch.object(self.co, "hold", side_effect=RuntimeError("no")):
            self.assertIn("failed inside the hub", self.hub.hold())

    def test_a_tracker_can_overrule_a_turn(self):
        self.hub.limit_heading = lambda drone, yaw: 0.25
        self.hub.aim(self.co.drones["a"], 2.0)
        self.assertEqual(self.co.drones["a"].config.yaw, 0.25)

    def test_detections_use_the_pose_from_when_the_picture_was_taken(self):
        self.run_ticks(10)
        self.poses["a"] = (2.0, -1.0, 1.0)
        self.eyes.next["a"] = [Detection("couch", 0.0, 0.0, 1.0, (0.2, 0.2, 0.2), 0.9, age=20.0)]
        self.run_ticks(5)
        self.eyes.next["a"] = []
        thing = next(iter(self.memory.things.values()))
        self.assertLess(thing.p[0], 1.0, "placed from the old position, not the new one")

    def test_everything_said_is_logged(self):
        self.run_ticks(1)
        self.hub.command("status")
        self.assertEqual([kind for _, kind, _ in self.memory.log()], ["said", "action", "reply"])


class HubMissionTests(unittest.TestCase):
    """One flight of each, so a broken change shows up without the whole bench."""

    def setUp(self):
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_every_mission_flies(self):
        for name in hub_bench.MISSIONS:
            with self.subTest(name):
                outcome = hub_bench.fly(name, 11)
                self.assertTrue(outcome.passed, outcome)

    def test_the_bench_can_tell_when_sharing_is_missing(self):
        """With nothing passed to the map, the blind drone flies into the box only the other one saw."""
        with mock.patch("swarm.hub.Hub._sync_obstacles", lambda self: None):
            outcomes = [hub_bench.fly("told_by_another", seed) for seed in range(1, 6)]
        self.assertGreaterEqual(sum(bool(o.contacts) or not o.passed for o in outcomes), 4, outcomes)

    def test_the_bench_can_tell_when_a_search_invents_a_find(self):
        with mock.patch("swarm.memory.KNOWN_SIGHTINGS", 1), mock.patch("swarm.memory.SURE_CONFIDENCE", 0.0), \
             mock.patch("swarm.hub.CLOSE_SIGHTINGS", 0), mock.patch("swarm.memory.FORGET_UNPROVEN_AFTER", 10**6), \
             mock.patch("swarm.memory.FORGET_AFTER_MISSES", 10**6):
            outcomes = [hub_bench.fly("search_decoy", seed) for seed in range(1, 6)]
        self.assertGreaterEqual(sum(not o.passed for o in outcomes), 3, outcomes)


if __name__ == "__main__":
    unittest.main()
