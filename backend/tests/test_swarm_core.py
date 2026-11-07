import math
import unittest
from unittest import mock

import numpy as np

from sim.scene import Box, Scene
from swarm import bench
from swarm.control import Airframe, Pose, StickMapper, Tracked
from swarm.coordinator import Coordinator, DroneConfig, Phase, formation_points
from swarm.planner import PlanError, plan_path
from swarm.room import Obstacle, Room
from swarm.safety import Body, Limits, Moving, Result, filter_velocities
from swarm.sim_fleet import SimFleet

LIMITS = Limits()


def open_room():
    return Room((-4, -3, 0), (4, 3, 2.6))


def body(name, x, y, want=(0, 0, 0), v=(0, 0, 0), z=1.0, controllable=True):
    return Body(name, np.array([x, y, z], float), np.array(v, float), np.array(want, float), controllable)


class RoomTests(unittest.TestCase):
    def test_clearance_measures_walls_and_obstacles_but_not_the_floor(self):
        room = Room((0, 0, 0), (10, 10, 3), [Obstacle("box", (4, 4, 0), (6, 6, 1))])
        self.assertAlmostEqual(room.clearance((1, 5, 0.1)), 1.0)
        self.assertAlmostEqual(room.clearance((3, 5, 0.5)), 1.0)
        self.assertAlmostEqual(room.clearance((5, 5, 2.5)), 0.5)
        self.assertLess(room.clearance((5, 5, 0.5)), 0)

    def test_a_map_change_is_visible_to_anyone_holding_a_path(self):
        room = open_room()
        before = room.version
        room.add(Obstacle("chair", (0, 0, 0), (1, 1, 1)))
        self.assertGreater(room.version, before)
        self.assertTrue(room.remove("chair"))
        self.assertFalse(room.remove("chair"))

    def test_an_obstacle_with_no_volume_is_refused(self):
        with self.assertRaises(ValueError):
            open_room().add(Obstacle("flat", (0, 0, 0), (1, 1, 0)))


class PlannerTests(unittest.TestCase):
    def assert_clear(self, room, path, clearance):
        for a, b in zip(path, path[1:]):
            for f in np.linspace(0, 1, 40):
                self.assertGreater(room.clearance(a + (b - a) * f), clearance - 0.12, msg=f"{a}->{b} at {f}")

    def test_a_clear_line_needs_no_search(self):
        path = plan_path(open_room(), (-2, 0, 1), (2, 0, 1), 0.3)
        self.assertEqual(len(path), 2)

    def test_it_goes_around_a_floor_to_ceiling_wall(self):
        room = Room((-4, -3, 0), (4, 3, 2.6), [Obstacle("wall", (-0.1, -3, 0), (0.1, 2, 2.6))])
        path = plan_path(room, (-2, -2, 1), (2, -2, 1), 0.35)
        self.assertGreater(len(path), 2)
        self.assert_clear(room, path, 0.35)
        self.assertTrue(any(p[1] > 2 for p in path))

    def test_it_flies_over_something_low_rather_than_a_long_way_round(self):
        room = Room((-4, -3, 0), (4, 3, 2.6), [Obstacle("table", (-0.5, -3, 0), (0.5, 3, 0.5))])
        path = plan_path(room, (-2, 0, 1), (2, 0, 1), 0.3)
        self.assertEqual(len(path), 2)

    def test_it_says_so_when_there_is_no_way_through(self):
        room = Room((-4, -3, 0), (4, 3, 2.6), [Obstacle("wall", (-0.1, -3, 0), (0.1, 3, 2.6))])
        with self.assertRaises(PlanError):
            plan_path(room, (-2, 0, 1), (2, 0, 1), 0.3)

    def test_a_gap_narrower_than_the_clearance_counts_as_closed(self):
        room = Room((-4, -3, 0), (4, 3, 2.6), [Obstacle("a", (-0.1, -3, 0), (0.1, -0.2, 2.6)),
                                               Obstacle("b", (-0.1, 0.2, 0), (0.1, 3, 2.6))])
        with self.assertRaises(PlanError):
            plan_path(room, (-2, 0, 1), (2, 0, 1), 0.35)


class SafetyFilterTests(unittest.TestCase):
    def test_a_request_that_breaks_no_rule_passes_unchanged(self):
        result = filter_velocities([body("a", -2, 0, (0.3, 0, 0)), body("b", 2, 2, (0, -0.2, 0))], open_room(), LIMITS)
        self.assertAlmostEqual(result.velocities["a"][0], 0.3)
        self.assertEqual(result.limited, set())

    def test_two_drones_closing_at_the_keep_out_distance_are_stopped(self):
        gap = LIMITS.separation
        result = filter_velocities([body("a", -gap / 2, 0, (0.5, 0, 0)), body("b", gap / 2, 0, (-0.5, 0, 0))], open_room(), LIMITS)
        closing = result.velocities["a"][0] - result.velocities["b"][0]
        self.assertLessEqual(closing, 0.01)

    def test_inside_the_keep_out_distance_they_are_pushed_apart(self):
        result = filter_velocities([body("a", -0.1, 0, (0, 0, 0)), body("b", 0.1, 0, (0, 0, 0))], open_room(), LIMITS)
        self.assertLess(result.velocities["a"][0], -0.05)
        self.assertGreater(result.velocities["b"][0], 0.05)

    def test_a_drone_that_cannot_be_steered_gets_no_share_of_the_work(self):
        rogue = body("rogue", -1.0, 0, v=(0.5, 0, 0), controllable=False)
        target = body("target", 0.0, 0, (0, 0, 0))
        both = filter_velocities([body("rogue", -1.0, 0, v=(0.5, 0, 0)), body("target", 0.0, 0)], open_room(), LIMITS)
        alone = filter_velocities([rogue, target], open_room(), LIMITS)
        self.assertNotIn("rogue", alone.velocities)
        self.assertGreater(alone.velocities["target"][0], both.velocities["target"][0])

    def test_it_stops_short_of_a_wall(self):
        keep = LIMITS.surface_clearance
        result = filter_velocities([body("a", 4 - keep, 0, (0.5, 0, 0))], open_room(), LIMITS)
        self.assertLessEqual(result.velocities["a"][0], 0.01)

    def test_it_is_harder_to_pass_under_a_drone_than_beside_one(self):
        gap = 0.7
        beside = filter_velocities([body("a", -gap / 2, 0, (0.1, 0, 0)), body("b", gap / 2, 0)], open_room(), LIMITS)
        over = filter_velocities([body("a", 0, 0, (0, 0, 0.1), z=1.0), body("b", 0, 0, z=1.0 + gap)], open_room(), LIMITS)
        self.assertEqual(beside.limited, set())
        self.assertIn("a", over.limited)

    def test_it_keeps_clear_of_a_tracked_person(self):
        person = Moving("person", np.array([1.0, 0.0, 0.9]), np.zeros(3), 0.3)
        result = filter_velocities([body("a", 0.2, 0, (0.5, 0, 0))], open_room(), LIMITS, [person])
        self.assertLess(result.velocities["a"][0], 0.5)

    def test_speed_stays_within_the_limit_even_when_pushed(self):
        result = filter_velocities([body("a", 0.0, 0), body("b", 0.05, 0)], open_room(), LIMITS)
        self.assertLessEqual(np.abs(result.velocities["a"][:2]).max(), LIMITS.max_speed + 1e-9)


class ControlTests(unittest.TestCase):
    def test_velocity_comes_out_of_noisy_poses(self):
        tracked = Tracked()
        rng = np.random.default_rng(1)
        for i in range(90):
            t = i / 30
            tracked.update(Pose(t, (0.5 * t + rng.normal(0, 0.01), 0.0, 1.0), 0.0))
        self.assertAlmostEqual(tracked.v[0], 0.5, delta=0.1)

    def test_sticks_follow_the_drones_own_frame(self):
        mapper = StickMapper(Airframe())
        _, _, pitch, roll = mapper.sticks((0.5, 0, 0), 0.0, 0.0, 0.05)
        self.assertAlmostEqual(pitch, 0.1)
        self.assertAlmostEqual(roll, 0.0)
        # Facing left, flying along world +x means sliding to the drone's right: positive roll.
        _, _, pitch, roll = mapper.sticks((0.5, 0, 0), math.pi / 2, math.pi / 2, 0.05)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(roll, 0.1)

    def test_a_small_request_is_pulsed_past_the_deadband(self):
        mapper = StickMapper(Airframe())
        sticks = [mapper.sticks((0.15, 0, 0), 0.0, 0.0, 0.05)[2] for _ in range(100)]
        self.assertTrue(all(s == 0.0 or abs(s) >= Airframe().deadband for s in sticks))
        self.assertAlmostEqual(sum(sticks) / len(sticks), 0.15 / 5.0, delta=0.005)

    def test_a_tiny_request_leaves_the_sticks_centred(self):
        mapper = StickMapper(Airframe())
        self.assertEqual(mapper.sticks((0.01, 0, 0), 0.0, 0.0, 0.05)[2], 0.0)

    def test_it_learns_a_faster_drone_from_steady_flight(self):
        mapper = StickMapper(Airframe())
        for i in range(200):
            mapper.learn(i * 0.05, 0.4, 0.0, 0.8, 0.0, 0.05)
        self.assertGreater(mapper.scale, 1.7)

    def test_it_learns_nothing_while_the_command_is_changing(self):
        mapper = StickMapper(Airframe())
        for i in range(200):
            mapper.learn(i * 0.05, 0.4 * (-1) ** (i // 10), 0.0, 0.8, 0.0, 0.05)
        self.assertEqual(mapper.scale, 1.0)


class FormationTests(unittest.TestCase):
    def test_neighbours_in_a_ring_are_the_spacing_apart(self):
        points = formation_points("ring", 3, (0, 0), 1.2, 1.0)
        self.assertAlmostEqual(np.linalg.norm(points[0] - points[1]), 1.2)

    def test_an_unknown_shape_is_refused(self):
        with self.assertRaises(ValueError):
            formation_points("heart", 3, (0, 0), 1.0, 1.0)


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


class CoordinatorRuleTests(unittest.TestCase):
    def setUp(self):
        self.links = {"a": FakeLink(), "b": FakeLink()}
        pads = [DroneConfig("a", (-1.0, 0.0, 0.0)), DroneConfig("b", (1.0, 0.0, 0.0))]
        self.co = Coordinator(open_room(), pads, self.links)

    def poses(self, t, a=(-1.0, 0.0, 0.0), b=(1.0, 0.0, 0.0)):
        return {"a": Pose(t, a, 0.0), "b": Pose(t, b, 0.0)}

    def test_no_takeoff_without_tracking(self):
        problems = self.co.takeoff(0.0)
        self.assertTrue(any("not being tracked" in p for p in problems))
        self.assertEqual(self.links["a"].commands, [])

    def test_no_takeoff_when_a_drone_is_off_its_pad(self):
        self.co.tick(0.0, self.poses(0.0, a=(0.0, 0.0, 0.0)))
        self.assertTrue(any("from its pad" in p for p in self.co.takeoff(0.0)))

    def test_no_takeoff_without_a_link(self):
        self.links["b"].connected = False
        self.co.tick(0.0, self.poses(0.0))
        self.assertTrue(any("no control link" in p for p in self.co.takeoff(0.0)))

    def test_pads_too_close_together_are_refused(self):
        pads = [DroneConfig("a", (0.0, 0.0, 0.0)), DroneConfig("b", (0.2, 0.0, 0.0))]
        co = Coordinator(open_room(), pads, self.links)
        co.tick(0.0, self.poses(0.0, a=(0.0, 0.0, 0.0), b=(0.2, 0.0, 0.0)))
        self.assertTrue(any("apart" in p for p in co.preflight(0.0)))

    def test_a_takeoff_that_never_climbs_is_never_sent_again(self):
        t = 0.0
        self.co.tick(t, self.poses(t))
        self.assertEqual(self.co.takeoff(t), [])
        for _ in range(400):
            t += 0.05
            self.co.tick(t, self.poses(t))
        self.assertEqual(self.links["a"].commands, ["takeoff"])
        self.assertEqual(self.links["b"].commands, [])  # the second waits to see the first climb
        self.assertEqual(self.co.drones["a"].phase, Phase.GROUND)
        self.assertIn("takeoff", self.co.drones["a"].faults)

    def test_goals_too_close_to_each_other_or_a_wall_are_refused(self):
        self.co.drones["a"].phase = self.co.drones["b"].phase = Phase.AIR
        self.assertTrue(self.co.check_goals({"a": (0, 0, 1), "b": (0.2, 0, 1)}))
        self.assertTrue(self.co.check_goals({"a": (3.99, 0, 1)}))
        self.assertFalse(self.co.check_goals({"a": (-1, 0, 1), "b": (1, 0, 1)}))

    def test_estop_reaches_every_drone(self):
        self.co.estop()
        self.assertEqual(self.links["a"].commands, ["estop"])
        self.assertEqual(self.links["b"].commands, ["estop"])
        self.assertTrue(self.co.all_phase(Phase.STOPPED))

    def test_a_drone_that_loses_its_pose_is_centred_and_then_landed_once(self):
        a = self.co.drones["a"]
        t = 0.0
        self.co.tick(t, self.poses(t, a=(-1.0, 0.0, 1.0), b=(1.0, 0.0, 1.0)))
        a.phase = Phase.AIR
        self.co.drones["b"].phase = Phase.AIR
        a.goal = np.array([-1.0, 0.0, 1.0])
        for _ in range(100):
            t += 0.05
            self.co.tick(t, {"b": Pose(t, (1.0, 0.0, 1.0), 0.0)})
        self.assertEqual(self.links["a"].commands, ["land"])
        self.assertEqual(self.links["a"].sticks[-1][:4], (0.0, 0.0, 0.0, 0.0))


class MissionTests(unittest.TestCase):
    """One flight of each mission, so a broken change shows up without the whole bench."""

    def test_every_mission_flies_safely(self):
        for name in bench.MISSIONS:
            if name == "blank_floor":
                continue
            with self.subTest(name):
                outcome = bench.fly(name, 7)
                self.assertTrue(outcome.passed, outcome)

    def test_the_bench_can_tell_when_the_filter_is_missing(self):
        def passthrough(bodies, room, limits, moving=()):
            return Result({b.id: b.want for b in bodies if b.controllable}, set(), False, 9.0)

        with mock.patch("swarm.coordinator.filter_velocities", passthrough):
            outcomes = [bench.fly("swap_two", seed) for seed in range(1, 6)]
        self.assertGreaterEqual(sum(bool(o.collisions) for o in outcomes), 4, outcomes)

    def test_simulated_drones_really_fall_when_stopped(self):
        scene = Scene(name="hall", bounds=Box((-4, -3, 0), (4, 3, 2.6), "wall"))
        fleet = SimFleet(scene, {"a": (0.0, 0.0)}, seed=1)
        fleet.links["a"].command("takeoff")
        fleet.advance(4.0)
        self.assertGreater(fleet.truth("a")[2], 0.5)
        fleet.links["a"].command("estop")
        fleet.advance(2.0)
        self.assertLess(fleet.truth("a")[2], 0.05)


if __name__ == "__main__":
    unittest.main()
