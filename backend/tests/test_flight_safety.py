"""The land toggle is the takeoff toggle, so these rules decide whether motors start next to someone.

Rule tests use scripted poses. The radio outage runs the real tracker on rendered frames in the simulator.
"""

import unittest

import numpy as np

from swarm.control import Pose
from swarm.coordinator import DOWN_SETTLED_S, Coordinator, DroneConfig, Phase, ResponseCheck
from swarm.room import Room
from swarm.safety import Limits


class Link:
    def __init__(self):
        self.connected = True
        self.commands = []
        self.sticks = []

    def send_sticks(self, *values):
        self.sticks.append(values)

    def command(self, name):
        if not self.connected:
            return False
        self.commands.append(name)
        return True

    def assume_flying(self):
        self.commands.append("assume_flying")

    def assume_grounded(self):
        self.commands.append("assume_grounded")


class Tracker:
    """Only what the coordinator asks of an InsideOutTracker."""

    def __init__(self):
        self.still = False
        self.video_at = 0.0

    def expected(self, drone_id):
        return ["tag"]

    def video_age(self, drone_id, now):
        return now - self.video_at

    def camera_still(self, drone_id, now):
        return self.still

    def note_command(self, *args):
        pass

    def reset(self, drone_id):
        pass


class Flight(unittest.TestCase):
    def setUp(self):
        self.links = {"a": Link(), "b": Link()}
        self.tracker = Tracker()
        pads = [DroneConfig("a", (-1.0, 0.0, 0.0)), DroneConfig("b", (1.0, 0.0, 0.0))]
        self.room = Room((-4.0, -4.0, 0.0), (4.0, 4.0, 2.5))
        self.co = Coordinator(self.room, pads, self.links, Limits(), tracker=self.tracker)
        self.t = 0.0
        self.sigma = 0.0
        self.video = True
        self.a, self.b = self.co.drones["a"], self.co.drones["b"]

    def tick(self, a=None, b=None, seconds=0.05):
        """a and b are (x, y, z, seconds since the last sighting, z of that sighting), or None for a drone seen level at 1 m."""
        for _ in range(max(1, round(seconds / 0.05))):
            self.t += 0.05
            if self.video:
                self.tracker.video_at = self.t
            self.co.tick(self.t, {"a": self.pose(a, -1.0), "b": self.pose(b, 1.0)})

    def pose(self, spec, x):
        spec = spec or (x, 0.0, 1.0, 0.0, 1.0)
        return Pose(self.t, spec[:3], 0.0, sigma=self.sigma if x < 0 else 0.0, v=np.zeros(3), fix_age=spec[3], seen_p=(spec[0], spec[1], spec[4]))

    def flying(self):
        self.tick(a=(-1.0, 0.0, 0.0, 0.0, 0.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.co.takeoff(self.t), [])
        for _ in range(6):
            self.tick()
        for d in (self.a, self.b):
            d.check = None
            d.phase = Phase.AIR
            self.co._set_goal(d, np.array([d.config.pad[0], 0.0, 1.0]))
        self.tick()
        self.assertEqual(self.a.phase, Phase.AIR)
        self.links["a"].commands.clear()
        self.links["b"].commands.clear()

    def lose_sightings(self, seconds, last_z=1.0, est_z=1.0):
        """Sightings stop. The estimate carries on at est_z, as dead reckoning does."""
        start = self.t
        while self.t < start + seconds:
            self.tick(a=(-1.0, 0.0, est_z, self.t - start, last_z))


class NoToggleToAFloorDrone(Flight):
    def test_a_low_last_sighting_means_down_and_no_toggle(self):
        self.flying()
        self.tick(a=(-1.0, 0.0, 0.3, 0.0, 0.3))
        self.lose_sightings(20.0, last_z=0.3, est_z=0.85)
        self.assertNotIn("land", self.links["a"].commands)
        self.assertEqual(self.a.phase, Phase.GROUND)
        self.assertTrue(self.a.unknown_air)
        self.assertIn("motors", self.a.faults["down"])
        self.assertEqual(self.links["a"].commands, ["assume_grounded"])

    def test_sightings_that_were_falling_mean_down(self):
        self.flying()
        for z in (1.0, 0.9, 0.75, 0.6):
            self.tick(a=(-1.0, 0.0, z, 0.0, z), seconds=0.2)
        self.lose_sightings(20.0, last_z=0.6, est_z=0.9)
        self.assertNotIn("land", self.links["a"].commands)
        self.assertEqual(self.a.phase, Phase.GROUND)

    def test_a_still_camera_with_no_sightings_means_down(self):
        self.flying()
        self.tracker.still = True
        self.lose_sightings(20.0)
        self.assertNotIn("land", self.links["a"].commands)
        self.assertEqual(self.a.phase, Phase.GROUND)

    def test_a_moving_camera_at_hover_height_still_gets_landed_blind_once(self):
        self.flying()
        self.sigma = 0.7
        self.lose_sightings(10.0)
        self.assertEqual(self.links["a"].commands, ["land"])
        self.assertEqual(self.a.phase, Phase.LANDING)

    def test_a_stop_button_gets_no_toggle_through_to_a_floor_drone(self):
        self.flying()
        self.tick(a=(-1.0, 0.0, 0.3, 0.0, 0.3))
        for _ in range(3):
            self.t += 0.05
            self.co.drones["a"].tracked.seen_log[-1] = (self.t - 2.0, 0.3)
            self.co.land_here(self.t)
        self.assertNotIn("land", self.links["a"].commands)
        self.assertEqual(self.a.phase, Phase.GROUND)

    def test_seen_flying_again_it_holds_and_waits_for_the_operator(self):
        self.flying()
        self.tick(a=(-1.0, 0.0, 0.3, 0.0, 0.3))
        self.lose_sightings(3.0, last_z=0.3, est_z=0.3)
        self.assertEqual(self.a.phase, Phase.GROUND)
        self.tick(a=(-1.0, 0.0, 1.1, 0.0, 1.1))
        self.assertEqual(self.a.phase, Phase.AIR)
        self.assertFalse(self.a.unknown_air)
        self.assertNotIn("down", self.a.faults)
        # It looked down once, so losing its sightings again doesn't earn it a blind toggle.
        self.a.tracked.sigma = 0.7
        self.co._toggle_land(self.a, self.t, blind=True)
        self.assertNotIn("land", self.links["a"].commands)
        # The operator can. The backend wrongly thinks it's down, so it's told first.
        self.co.land_here(self.t)
        self.assertEqual(self.links["a"].commands[-2:], ["assume_flying", "land"])


class NothingRaisesAndEverythingIsRetried(Flight):
    def test_land_here_during_the_check_with_the_link_down(self):
        self.flying()
        self.a.phase = Phase.CHECK
        self.a.check = ResponseCheck([np.array([1.0, 0.0, 0.0])], self.a.tracked.p.copy(), 0, self.t)
        self.links["a"].connected = False
        self.co.land_here(self.t)
        self.assertIn("land_link", self.a.faults)
        self.links["a"].connected = True
        sticks = len(self.links["b"].sticks)
        for _ in range(100):
            self.tick()  # every tick used to raise here, and no drone got sticks
        self.assertGreater(len(self.links["b"].sticks), sticks + 90)
        self.assertEqual(self.links["a"].commands, ["land"])
        self.assertEqual(self.a.phase, Phase.LANDING)
        self.assertNotIn("land_link", self.a.faults)

    def test_a_check_with_nothing_in_it_only_ends_the_check(self):
        self.flying()
        self.a.phase, self.a.check = Phase.CHECK, None
        self.tick()
        self.assertEqual(self.a.phase, Phase.AIR)

    def test_a_failure_for_one_drone_leaves_the_others_their_sticks(self):
        self.flying()
        real = self.co._want
        self.co._want = lambda d, now: (_ for _ in ()).throw(RuntimeError("boom")) if d.id == "a" else real(d, now)
        sticks = len(self.links["b"].sticks)
        with self.assertLogs("swarm.coordinator", "ERROR"):
            self.tick()
        self.assertEqual(len(self.links["b"].sticks), sticks + 1)
        self.assertIn("internal", self.a.faults)


class LandingDuringTakeoff(Flight):
    def climbing(self):
        self.tick(a=(-1.0, 0.0, 0.0, 0.0, 0.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.co.takeoff(self.t), [])
        self.tick(a=(-1.0, 0.0, 0.1, 0.0, 0.1), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.a.phase, Phase.TAKEOFF)

    def test_land_is_kept_until_it_is_up(self):
        self.climbing()
        self.co.land(self.t)
        self.assertTrue(self.a.landing_requested)
        self.assertEqual(self.co.takeoff_queue, [])
        self.assertEqual(self.links["a"].commands, ["takeoff"])  # no land toggle while it climbs
        self.tick(a=(-1.0, 0.0, 1.0, 0.0, 1.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))  # up, and over its pad
        self.tick(a=(-1.0, 0.0, 1.0, 0.0, 1.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.links["a"].commands, ["takeoff", "land"])
        self.assertEqual(self.a.phase, Phase.LANDING)
        self.assertEqual(self.links["b"].commands, [])

    def test_land_here_is_kept_until_it_is_up(self):
        self.climbing()
        self.co.land_here(self.t)
        self.assertEqual(self.links["a"].commands, ["takeoff"])
        for _ in range(3):
            self.tick(a=(-1.0, 0.0, 1.0, 0.0, 1.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.links["a"].commands, ["takeoff", "land"])

    def test_hold_cancels_a_landing_asked_for_during_the_climb(self):
        self.climbing()
        self.co.land(self.t)
        self.co.hold()
        self.assertFalse(self.a.landing_requested)


class FailedLandings(Flight):
    def blind_landing(self):
        self.flying()
        self.a.tracked.sigma = 0.7
        self.co._toggle_land(self.a, self.t, blind=True)
        self.assertEqual(self.a.phase, Phase.LANDING)
        self.links["a"].commands.clear()

    def test_a_blind_landing_that_times_out_with_it_in_view_goes_back_to_air(self):
        self.blind_landing()
        self.tick(a=(-1.0, 0.0, 1.2, 0.0, 1.2), seconds=16.0)
        self.assertEqual(self.a.phase, Phase.AIR)
        self.assertFalse(self.a.unknown_air)
        # LAND reaches it, past the backend's belief that it landed.
        self.co.land_here(self.t)
        self.assertEqual(self.links["a"].commands, ["assume_flying", "land"])

    def test_a_blind_landing_that_times_out_unseen_can_still_come_back(self):
        self.blind_landing()
        start = self.t
        while self.t < start + 16.0:
            self.tick(a=(-1.0, 0.0, 1.0, self.t - start + 5.0, 1.0))
        self.assertEqual(self.a.phase, Phase.GROUND)
        self.assertTrue(self.a.unknown_air)
        self.tick(a=(-1.0, 0.0, 1.2, 0.0, 1.2))
        self.assertEqual(self.a.phase, Phase.AIR)

    def test_a_touchdown_only_the_estimate_saw_can_still_come_back(self):
        self.blind_landing()
        start = self.t
        while self.a.phase == Phase.LANDING and self.t < start + 5.0:
            self.tick(a=(-1.0, 0.0, 0.05, 3.0, 1.0))
        self.assertEqual(self.a.phase, Phase.GROUND)
        self.assertTrue(self.a.unknown_air)
        self.assertIn("land", self.a.faults)
        self.tick(a=(-1.0, 0.0, 1.2, 0.0, 1.2))
        self.assertEqual(self.a.phase, Phase.AIR)

    def test_a_lost_land_packet_doesnt_silence_the_next_land(self):
        self.flying()
        self.co.land_here(self.t)
        self.assertEqual(self.links["a"].commands, ["land"])
        self.tick(seconds=16.0)  # still seen at 1 m: the packet never arrived
        self.assertEqual(self.a.phase, Phase.AIR)
        self.assertIn("still in the air", self.a.faults["land"])
        self.links["a"].commands.clear()
        self.co.land_here(self.t)
        self.assertEqual(self.links["a"].commands, ["assume_flying", "land"])


class Preflight(Flight):
    def test_no_takeoff_with_dead_video(self):
        self.tick(a=(-1.0, 0.0, 0.0, 0.0, 0.0), b=(1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.co.preflight(self.t), [])
        self.video = False
        self.tick(a=(-1.0, 0.0, 0.0, 0.0, 0.0), b=(1.0, 0.0, 0.0, 0.0, 0.0), seconds=3.0)
        self.assertTrue(any("no video" in p for p in self.co.preflight(self.t)))
        self.assertTrue(self.co.takeoff(self.t))
        self.assertEqual(self.links["a"].commands, [])


class HoldDuringTheCheck(Flight):
    def in_check(self, drone):
        drone.phase = Phase.CHECK
        drone.check = ResponseCheck([np.array([1.0, 0.0, 0.0])], drone.tracked.p.copy(), 0, self.t)

    def test_hold_stops_the_test_moves(self):
        self.flying()
        self.in_check(self.a)
        self.tick()
        self.assertGreater(np.linalg.norm(self.a.commanded), 0.1)
        self.co.hold()
        self.tick(seconds=5.0)
        self.assertEqual(np.linalg.norm(self.a.commanded), 0.0)
        self.assertEqual(self.a.phase, Phase.CHECK)

    def test_land_still_works_and_work_lets_the_check_carry_on(self):
        self.flying()
        self.in_check(self.a)
        self.co.hold()
        self.tick()
        self.co.resume_checks()
        self.tick()
        self.assertGreater(np.linalg.norm(self.a.commanded), 0.1)
        self.co.hold()
        self.co.land(self.t)
        self.assertEqual(self.a.phase, Phase.AIR)
        self.assertTrue(self.a.landing_requested)


class AFreshStart(Flight):
    def test_reset_pads_forgets_the_last_flight(self):
        self.flying()
        self.co.land(self.t)
        self.a.sticks_wrong = True
        self.co.estop()
        self.assertFalse(self.a.landing_requested)
        self.a.landing_requested = True  # however it got there, reset clears it
        self.assertEqual(self.co.reset_to_pads(), [])
        for d in (self.a, self.b):
            self.assertFalse(d.landing_requested or d.sticks_wrong or d.land_here_wanted or d.check_held or d.unknown_air)


class ADownedDroneStopsFencingOffTheRoom(Flight):
    def test_the_others_get_the_room_back_once_it_has_finished_coming_down(self):
        self.flying()
        self.co.takeoff_queue.clear()
        self.sigma = 0.71  # what a real flight froze at, which kept everyone 1.5 m away for the rest of it
        down = (-1.0, 0.0, 0.0, 5.0, 1.0)
        near = (0.2, 0.0, 1.0, 0.0, 1.0)
        self.tick(a=down, b=near)
        self.assertIn("land", self.links["a"].commands)
        self.co._set_goal(self.b, np.array([-0.2, 0.0, 1.0]))
        self.tick(a=down, b=near, seconds=3.0)
        self.assertEqual(self.a.phase, Phase.GROUND)
        self.assertTrue(self.a.came_down)
        self.assertGreaterEqual(self.b.commanded[0], 0.0, "it may still be falling, so b is held well away at first")
        self.tick(a=down, b=near, seconds=DOWN_SETTLED_S)
        self.assertLess(self.b.commanded[0], -0.1, "b should be free to fly past now")
        self.assertTrue(self.a.unknown_air, "it still needs checking by eye")
        self.assertIn("land", self.a.faults)

    def test_it_is_only_for_a_drone_that_came_down(self):
        self.flying()
        self.co.takeoff_queue.clear()
        self.a.phase, self.a.unknown_air = Phase.GROUND, True  # a takeoff nobody saw may be hovering unseen
        self.sigma = 0.71
        near = (0.2, 0.0, 1.0, 0.0, 1.0)
        self.co._set_goal(self.b, np.array([-0.2, 0.0, 1.0]))
        self.tick(a=(-1.0, 0.0, 1.0, self.t, 1.0), b=near, seconds=DOWN_SETTLED_S + 1.0)
        self.assertGreaterEqual(self.b.commanded[0], 0.0)


class RadioOutageInTheSimulator(unittest.TestCase):
    """Simulator numbers: its flight controller lands itself after losing packets for too long."""

    def fly(self, outage):
        import logging

        from swarm import floor_bench
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        f = floor_bench.FloorFlight(1, {"a": (-2.5, 0.0)}, limits=Limits())

        def script():
            yield from f.takeoff()
            yield from f.go({"a": (-1.0, 0.0, 1.0)}, 40.0)
        f.run(script())
        while f.fleet.t < 20.0:
            f.tick()
        original, start = f.fleet._send_packet, f.fleet.t
        f.fleet._send_packet = lambda x: None if f.fleet.t < start + outage else original(x)
        before = len(f.fleet.links["a"].commands)
        peak = 0.0
        while f.fleet.t < start + 20.0:
            f.tick()
            if f.fleet.drones["a"].sim.fc.state == "READY":
                peak = max(peak, float(f.fleet.truth("a")[2]))
        return f, f.fleet.links["a"].commands[before:], peak

    def test_landed_by_its_own_failsafe_it_gets_no_toggle(self):
        f, commands, peak = self.fly(3.6)
        self.assertEqual(commands, [])
        self.assertEqual(f.co.drones["a"].phase, Phase.GROUND)
        self.assertEqual(f.fleet.drones["a"].sim.fc.state, "READY")
        self.assertLess(float(f.fleet.truth("a")[2]), 0.05)
        self.assertFalse(f.fleet.drones["a"].model.airborne)  # /fleet's Land all can't restart it

    def test_a_shorter_outage_changes_nothing(self):
        f, commands, _ = self.fly(2.5)
        self.assertEqual(commands, [])
        self.assertEqual(f.co.drones["a"].phase, Phase.AIR)
        self.assertGreater(float(f.fleet.truth("a")[2]), 0.5)


if __name__ == "__main__":
    unittest.main()
