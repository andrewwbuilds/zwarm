import json
import math
import unittest
from pathlib import Path

from sim.config import DEFAULT_FIRMWARE, DroneSpec, FlightTuning
from sim.pilot import FLY, STOP, CALIBRATE, HEADLESS, Pilot
from sim.rc_frames import heliway_packet, short_packet
from sim.scene import Box, Scene

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "s20_capture.json").read_text())
TUNE = FlightTuning.calibrated()
SPEC = DroneSpec.calibrated()
OPEN = Scene(name="open")
ROOM = Scene(name="room", bounds=Box((-5, -5, 0), (5, 5, 3), "wall"))


def airborne(firmware=DEFAULT_FIRMWARE, scene=OPEN, **kwargs) -> Pilot:
    pilot = Pilot(firmware=firmware, scene=scene, **kwargs)
    pilot.fly(2.5)  # gyro calibration after power-on
    pilot.hold(FLY, 1.0)
    pilot.fly(5.0)
    return pilot


class TakeoffSignalTests(unittest.TestCase):
    def test_one_packet_takes_off_on_the_default_firmware(self):
        # Measured: takeoff_probe step A, one packet in an 80 Hz stream.
        self.assertTrue(CAPTURE["takeoff"]["probe_single_packet_at_80hz_started_motors"])
        pilot = Pilot()
        pilot.fly(2.5)
        pilot.packets(1, flags=FLY)
        pilot.fly(0.2)
        self.assertEqual(pilot.fc.state, "SPINUP")

    def test_one_packet_lands_on_the_default_firmware(self):
        self.assertTrue(CAPTURE["takeoff"]["probe_single_packet_landed"])
        pilot = airborne()
        pilot.packets(1, flags=FLY)
        pilot.fly(0.1)
        self.assertEqual(pilot.fc.state, "LANDING")
        pilot.fly(6.0)
        self.assertEqual(pilot.fc.state, "READY")

    def test_one_packet_does_nothing_on_the_hold_firmware(self):
        pilot = Pilot(firmware="hold")
        pilot.fly(2.5)
        pilot.packets(1, flags=FLY)
        pilot.fly(3.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertEqual(pilot.sim.drone.thrust, [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(any("ignored" in message for _, message in pilot.fc.events))

    def test_holding_the_signal_for_a_second_starts_the_motors(self):
        pilot = Pilot()
        pilot.fly(2.5)
        pilot.hold(FLY, 1.0)
        self.assertIn(pilot.fc.state, ("SPINUP", "CLIMB"))
        self.assertGreater(sum(pilot.sim.drone.thrust), 0.0)

    def test_upstream_firmware_takes_off_on_one_packet(self):
        pilot = Pilot(firmware="upstream")
        pilot.fly(2.5)
        pilot.packets(1, flags=FLY)
        pilot.fly(0.2)
        self.assertEqual(pilot.fc.state, "SPINUP")

    def test_a_long_hold_only_toggles_once(self):
        pilot = Pilot()
        pilot.fly(2.5)
        pilot.hold(FLY, 2.0)
        pilot.fly(5.0)
        self.assertEqual(pilot.fc.state, "FLYING")

    def test_takeoff_before_gyro_calibration_finishes_is_refused(self):
        pilot = Pilot()
        pilot.hold(FLY, 0.5)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertTrue(any("calibrat" in message.lower() for _, message in pilot.fc.events))


class TakeoffAndLandingTests(unittest.TestCase):
    def test_neutral_sticks_never_start_the_motors(self):
        pilot = Pilot()
        pilot.fly(5.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertEqual(sum(pilot.sim.drone.thrust), 0.0)

    def test_climbs_to_hover_height(self):
        pilot = airborne()
        self.assertEqual(pilot.fc.state, "FLYING")
        self.assertAlmostEqual(pilot.sim.drone.z, TUNE.takeoff_height_m, delta=0.2)

    def test_holds_altitude_and_position_for_a_long_hover(self):
        pilot = airborne()
        drone = pilot.sim.drone
        x, y = drone.x, drone.y
        lowest = highest = drone.z
        for _ in range(30):
            pilot.fly(1.0)
            lowest, highest = min(lowest, drone.z), max(highest, drone.z)
        self.assertGreater(lowest, 0.85)
        self.assertLess(highest, 1.55)
        self.assertLess(math.hypot(drone.x - x, drone.y - y), 0.5)

    def test_hover_is_level(self):
        pilot = airborne()
        pilot.fly(3.0)
        self.assertLess(abs(pilot.sim.drone.roll()), math.radians(4))
        self.assertLess(abs(pilot.sim.drone.pitch()), math.radians(4))

    def test_second_signal_lands_and_stops_the_motors(self):
        pilot = airborne()
        pilot.hold(FLY, 1.0)
        self.assertEqual(pilot.fc.state, "LANDING")
        pilot.fly(6.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertTrue(pilot.sim.drone.on_ground)
        pilot.fly(0.5)
        self.assertAlmostEqual(sum(pilot.sim.drone.thrust), 0.0, places=3)

    def test_lands_on_a_table(self):
        table = Scene(name="table", boxes=(Box((1.0, -0.5, 0.0), (2.0, 0.5, 0.45), "wood"),))
        pilot = airborne(scene=table)
        pilot.fly(2.5, pitch=150)
        pilot.fly(2.0)
        self.assertTrue(1.0 < pilot.sim.drone.x < 2.0)
        pilot.hold(FLY, 0.3)
        pilot.fly(6.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertGreater(pilot.sim.drone.z, 0.4)

    def test_landing_is_gentle(self):
        pilot = airborne()
        pilot.hold(FLY, 1.0)
        pilot.sim.drone.impacts.clear()
        pilot.fly(6.0)
        self.assertTrue(all(impact.speed < 1.0 for impact in pilot.sim.drone.impacts))


class EmergencyStopTests(unittest.TestCase):
    def test_stop_cuts_the_motors_and_the_drone_falls(self):
        pilot = airborne()
        pilot.hold(STOP, 1.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertAlmostEqual(sum(pilot.sim.drone.thrust), 0.0, places=2)
        self.assertTrue(pilot.sim.drone.on_ground)

    def test_stop_acts_as_soon_as_the_firmware_accepts_it(self):
        pilot = airborne()
        pilot.hold(STOP, 0.3)
        self.assertEqual(pilot.fc.state, "READY")

    def test_a_drop_from_hover_height_does_not_break_it(self):
        pilot = airborne()
        pilot.hold(STOP, 1.0)
        pilot.fly(1.0)
        self.assertEqual(pilot.fc.state, "READY")
        pilot.hold(FLY, 1.0)
        pilot.fly(5.0)
        self.assertEqual(pilot.fc.state, "FLYING")

    def test_a_drop_from_high_up_breaks_it(self):
        pilot = airborne()
        pilot.fly(4.0, throttle=255)
        pilot.hold(STOP, 1.0)
        pilot.fly(1.5)
        self.assertEqual(pilot.fc.state, "CRASHED")


class StickTests(unittest.TestCase):
    def test_pitch_stick_flies_forward_at_the_estimated_top_speed(self):
        pilot = airborne()
        x = pilot.sim.drone.x
        pilot.fly(4.0, pitch=255)
        self.assertGreater(pilot.sim.drone.x, x + TUNE.max_speed_mps)
        self.assertAlmostEqual(pilot.sim.drone.vx, TUNE.max_speed_mps, delta=TUNE.max_speed_mps * 0.2)
        self.assertLess(pilot.sim.drone.pitch(), math.radians(28))

    def test_releasing_the_stick_stops_the_drone(self):
        pilot = airborne()
        pilot.fly(3.0, pitch=255)
        x = pilot.sim.drone.x
        pilot.fly(3.0)
        self.assertLess(abs(pilot.sim.drone.vx), 0.3)
        self.assertLess(pilot.sim.drone.x - x, 4.0)

    def test_roll_stick_flies_right(self):
        pilot = airborne()
        y = pilot.sim.drone.y
        pilot.fly(2.0, roll=255)
        self.assertLess(pilot.sim.drone.y, y - 1.0)

    def test_half_stick_flies_about_half_speed(self):
        pilot = airborne()
        pilot.fly(4.0, pitch=192)
        self.assertAlmostEqual(pilot.sim.drone.vx, TUNE.max_speed_mps / 2, delta=TUNE.max_speed_mps * 0.12)

    def test_throttle_stick_climbs_and_descends(self):
        pilot = airborne()
        z = pilot.sim.drone.z
        pilot.fly(1.0, throttle=255)
        self.assertGreater(pilot.sim.drone.z, z + 0.7)
        z = pilot.sim.drone.z
        pilot.fly(1.0, throttle=0)
        self.assertLess(pilot.sim.drone.z, z - 0.25)  # reversing a 1.5 m/s climb takes a moment

    def test_altitude_holds_where_the_throttle_stick_was_released(self):
        pilot = airborne()
        pilot.fly(1.0, throttle=255)
        pilot.fly(1.0)
        z = pilot.sim.drone.z
        pilot.fly(4.0)
        self.assertAlmostEqual(pilot.sim.drone.z, z, delta=0.2)

    def test_yaw_stick_turns_right_at_the_estimated_rate(self):
        pilot = airborne()
        start = pilot.sim.drone.yaw()
        pilot.fly(0.5, yaw=255)
        pilot.fly(0.1, yaw=255)
        self.assertAlmostEqual(pilot.sim.drone.r, -TUNE.max_yaw_rate_rad_s, delta=TUNE.max_yaw_rate_rad_s * 0.2)
        self.assertLess(pilot.sim.drone.yaw(), start)

    def test_forward_means_where_the_nose_points(self):
        pilot = airborne()
        pilot.fly(0.6, yaw=0)  # about 90 degrees left
        while abs(pilot.sim.drone.yaw() - math.pi / 2) > 0.15:
            pilot.fly(0.05, yaw=0)
        pilot.fly(1.0)
        y = pilot.sim.drone.y
        pilot.fly(2.0, pitch=255)
        self.assertGreater(pilot.sim.drone.y, y + 1.0)

    def test_headless_mode_keeps_forward_along_the_takeoff_heading(self):
        pilot = airborne()
        while abs(pilot.sim.drone.yaw() - math.pi / 2) > 0.15:
            pilot.fly(0.05, yaw=0, byte7=0x0A | HEADLESS)
        pilot.fly(1.0, byte7=0x0A | HEADLESS)
        x = pilot.sim.drone.x
        pilot.fly(2.0, pitch=255, byte7=0x0A | HEADLESS)
        self.assertGreater(pilot.sim.drone.x, x + 1.0)


class LinkLossTests(unittest.TestCase):
    def test_losing_the_link_hovers_then_lands(self):
        pilot = airborne()
        pilot.silence(1.5)
        self.assertEqual(pilot.fc.state, "FLYING")
        self.assertFalse(pilot.fc.link_alive)
        pilot.silence(10.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertTrue(pilot.sim.drone.on_ground)

    def test_held_sticks_are_dropped_when_the_link_is_lost(self):
        pilot = airborne()
        pilot.fly(2.0, pitch=255)
        pilot.silence(2.0)
        self.assertLess(abs(pilot.sim.drone.vx), 1.0)

    def test_the_link_coming_back_during_the_hover_resumes_control(self):
        pilot = airborne()
        pilot.silence(1.0)
        pilot.fly(1.0)
        self.assertTrue(pilot.fc.link_alive)
        pilot.fly(5.0)
        self.assertEqual(pilot.fc.state, "FLYING")


class BatteryTests(unittest.TestCase):
    def test_low_battery_refuses_takeoff(self):
        pilot = Pilot(firmware="low_battery")
        pilot.fly(2.5)
        pilot.hold(FLY, 1.0)
        self.assertEqual(pilot.fc.state, "READY")
        self.assertTrue(any("battery" in message.lower() for _, message in pilot.fc.events))

    def test_flight_time_matches_the_rated_twelve_minutes(self):
        pilot = airborne(step_hz=100)
        started = pilot.sim.t
        while pilot.fc.state == "FLYING" and pilot.sim.t - started < 1200:
            pilot.fly(5.0)
        minutes = (pilot.sim.t - started) / 60
        self.assertAlmostEqual(minutes, SPEC.rated_flight_min, delta=SPEC.rated_flight_min / 6)
        self.assertTrue(any("battery" in message.lower() for _, message in pilot.fc.events))


class CrashTests(unittest.TestCase):
    def test_a_fast_wall_hit_crashes(self):
        pilot = airborne(scene=ROOM)
        pilot.fly(4.0, pitch=255)
        self.assertEqual(pilot.fc.state, "CRASHED")
        pilot.fly(1.5)
        self.assertAlmostEqual(sum(pilot.sim.drone.thrust), 0.0, places=2)
        self.assertTrue(pilot.sim.drone.on_ground)

    def test_a_crashed_drone_refuses_takeoff(self):
        pilot = airborne(scene=ROOM)
        pilot.fly(4.0, pitch=255)
        pilot.fly(2.0)
        pilot.hold(FLY, 1.0)
        self.assertEqual(pilot.fc.state, "CRASHED")

    def test_a_gentle_bump_keeps_flying(self):
        pilot = airborne(scene=ROOM)
        pilot.sim.drone.x = 4.0
        pilot.fly(4.0, pitch=150)
        self.assertEqual(pilot.fc.state, "FLYING")
        self.assertLess(pilot.sim.drone.x, 5.0)


class OpticalFlowTests(unittest.TestCase):
    def test_wind_is_held_off_while_the_flow_sensor_works(self):
        windy = Scene(name="windy", wind_mps=2.0)
        pilot = airborne(scene=windy)
        x = pilot.sim.drone.x
        pilot.fly(15.0)
        self.assertLess(abs(pilot.sim.drone.x - x), 2.0)

    def test_blank_floor_loses_position_hold_and_drifts_in_wind(self):
        blank = Scene(name="blank", wind_mps=2.0, flow_quality=0.0)
        pilot = airborne(scene=blank)
        x = pilot.sim.drone.x
        pilot.fly(10.0)
        self.assertGreater(abs(pilot.sim.drone.x - x), 5.0)

    def test_too_high_for_the_flow_sensor_loses_position_hold(self):
        windy = Scene(name="windy", wind_mps=2.0)
        pilot = airborne(scene=windy)
        pilot.fly(5.0, throttle=255)
        self.assertGreater(pilot.sim.drone.z, 4.5)
        self.assertFalse(pilot.fc.flow_valid)


class FirmwareHypothesisTests(unittest.TestCase):
    def takes_off(self, pilot: Pilot, **kwargs) -> bool:
        pilot.hold(FLY, 1.0, **kwargs)
        pilot.fly(0.5, **kwargs)
        return pilot.fc.state != "READY"

    def test_rate20_rejects_a_fast_stream(self):
        pilot = Pilot(firmware="rate20", rate_hz=80)
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))

    def test_rate20_accepts_the_stock_rate(self):
        pilot = Pilot(firmware="rate20", rate_hz=20)
        pilot.fly(3.0)
        self.assertTrue(self.takes_off(pilot))

    def test_handshake_firmware_needs_the_handshake(self):
        pilot = Pilot(firmware="handshake")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.handshake()
        pilot.fly(1.0)
        self.assertTrue(self.takes_off(pilot))

    def test_byte7_firmware_needs_the_stock_mode_byte(self):
        pilot = Pilot(firmware="byte7")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        self.assertTrue(self.takes_off(pilot, byte7=0x02))

    def test_short_firmware_only_reads_the_short_packet(self):
        pilot = Pilot(firmware="short")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.stream(short_packet, 1.0)
        pilot.stream(lambda: short_packet(fly=True), 1.0)
        pilot.stream(short_packet, 0.5)
        self.assertNotEqual(pilot.fc.state, "READY")

    def test_heliway_firmware_only_reads_the_heliway_packet(self):
        pilot = Pilot(firmware="heliway")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.stream(heliway_packet, 1.0)
        pilot.stream(lambda: heliway_packet(fly=True), 1.0)
        pilot.stream(heliway_packet, 0.5)
        self.assertNotEqual(pilot.fc.state, "READY")

    def test_pairing_firmware_needs_throttle_up_then_down(self):
        pilot = Pilot(firmware="pairing")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.fly(1.5, throttle=255)
        pilot.fly(1.5, throttle=0)
        pilot.fly(1.0)
        self.assertTrue(self.takes_off(pilot))

    def test_arming_firmware_needs_sticks_down_and_inward(self):
        pilot = Pilot(firmware="arm_inward")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.fly(1.5, roll=0, pitch=0, throttle=0, yaw=255)
        pilot.fly(1.0)
        self.assertTrue(self.takes_off(pilot))

    def test_arming_firmware_rejects_the_wrong_pattern(self):
        pilot = Pilot(firmware="arm_outward")
        pilot.fly(3.0)
        pilot.fly(1.5, roll=0, pitch=0, throttle=0, yaw=255)
        pilot.fly(1.0)
        self.assertFalse(self.takes_off(pilot))
        pilot.fly(1.5, roll=255, pitch=0, throttle=0, yaw=0)
        pilot.fly(1.0)
        self.assertTrue(self.takes_off(pilot))

    def test_remote_firmware_never_takes_off_over_wifi(self):
        pilot = Pilot(firmware="remote")
        pilot.fly(3.0)
        self.assertFalse(self.takes_off(pilot))
        self.assertTrue(any("remote" in message.lower() for _, message in pilot.fc.events))


class CalibrationTests(unittest.TestCase):
    def test_calibrate_on_the_ground_reduces_level_error_and_flashes_the_lights(self):
        pilot = Pilot()
        pilot.fly(2.5)
        self.assertEqual(pilot.fc.leds(), "solid")
        before = abs(pilot.fc.trim_roll) + abs(pilot.fc.trim_pitch)
        pilot.hold(CALIBRATE, 0.5)
        self.assertEqual(pilot.fc.leds(), "fast flash")
        pilot.fly(3.0)
        self.assertEqual(pilot.fc.leds(), "solid")
        self.assertLess(abs(pilot.fc.trim_roll) + abs(pilot.fc.trim_pitch), before)


class LedTests(unittest.TestCase):
    def test_lights_are_solid_before_any_packets_and_stay_solid(self):
        lights = CAPTURE["lights"]
        pilot = Pilot()
        pilot.silence(1.0)
        self.assertEqual(pilot.fc.leds(), "flashing")
        pilot.silence(2.0)
        self.assertEqual(pilot.fc.leds(), lights["before_any_packets"])
        pilot.fly(0.5)
        self.assertEqual(pilot.fc.leds() != lights["before_any_packets"], lights["changed_by_centered_sticks"])

    def test_lights_flash_after_a_crash(self):
        pilot = airborne(scene=ROOM)
        pilot.fly(4.0, pitch=255)
        self.assertEqual(pilot.fc.leds(), "flashing")


if __name__ == "__main__":
    unittest.main()
