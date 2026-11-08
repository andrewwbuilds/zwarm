import math
import unittest

import cv2
import numpy as np

from control.stabilizer import ACTIVE_STICK, Gains, Stabilizer, decode_gray
from control.strategies import IncrementalStrategy, LinearStrategy
from control.visual_motion import MotionTracker, flow_basis, solve_flow, unturn
from models.s2x_rc import S2xDroneModel
from services.flight_controller import FlightController
from sim.stability import Case, fly


def texture(width=296, height=164, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (height // 4, width // 4), np.uint8)
    return cv2.GaussianBlur(cv2.resize(noise, (width, height), interpolation=cv2.INTER_NEAREST), (3, 3), 0)


class FlowFitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.x, self.y = rng.uniform(-1, 1, 150), rng.uniform(-0.55, 0.55, 150)

    def fit(self, truth, **kwargs):
        flow = flow_basis(self.x, self.y) @ np.asarray(truth)
        du, dv = np.split(flow, 2)
        return solve_flow(self.x, self.y, du, dv, **kwargs)[0]

    def test_tells_a_turn_from_a_slide(self):
        turn = self.fit([0, 0.05, 0, 0, 0, 0])
        slide = self.fit([0, 0, 0, 0.05, 0, 0])
        self.assertAlmostEqual(turn[1], 0.05, places=4)
        self.assertAlmostEqual(turn[3], 0.0, places=4)
        self.assertAlmostEqual(slide[1], 0.0, places=4)
        self.assertAlmostEqual(slide[3], 0.05, places=4)

    def test_reads_forward_motion_as_expansion(self):
        self.assertAlmostEqual(self.fit([0, 0, 0, 0, 0, 0.03])[5], 0.03, places=4)

    def test_a_moving_object_does_not_become_drone_motion(self):
        flow = flow_basis(self.x, self.y) @ np.array([0, 0, 0, 0.02, 0, 0])
        du, dv = np.split(flow, 2)
        du[:30] += 0.2  # a fifth of the view is something crossing the frame
        params, mask = solve_flow(self.x, self.y, du, dv)
        self.assertAlmostEqual(params[3], 0.02, places=3)
        self.assertFalse(mask[:30].any())


class UnturnTests(unittest.TestCase):
    def test_takes_out_an_exact_turn(self):
        rng = np.random.default_rng(2)
        x, y = rng.uniform(-1, 1, 50), rng.uniform(-0.5, 0.5, 50)
        angle = 0.1  # a right turn moves every point left, by this exact rotation
        z = np.cos(angle) + x * np.sin(angle)
        x1, y1 = (x * np.cos(angle) - np.sin(angle)) / z, y / z
        du, dv = unturn(x, y, x1, y1, angle)
        self.assertLess(np.abs(du).max(), 1e-9)
        self.assertLess(np.abs(dv).max(), 1e-9)


class TrackerTests(unittest.TestCase):
    def test_sees_the_camera_slide_right_when_the_image_moves_left(self):
        image = texture(400, 200)
        tracker = MotionTracker(90.0)
        crop = lambda shift: image[18:182, shift:shift + 296].copy()
        tracker.update(crop(20), 0.0)
        motion = tracker.update(crop(26), 0.1)
        self.assertTrue(motion.valid)
        self.assertAlmostEqual(motion.lateral, 6 / 148 / 0.1, delta=0.05)
        self.assertAlmostEqual(motion.forward, 0.0, delta=0.05)

    def test_a_blank_view_is_not_a_measurement(self):
        blank = np.full((164, 296), 128, np.uint8)
        tracker = MotionTracker(90.0)
        tracker.update(blank, 0.0)
        self.assertFalse(tracker.update(blank, 0.1).valid)

    def test_a_repeated_frame_is_ignored(self):
        image = texture()
        tracker = MotionTracker(90.0)
        tracker.update(image, 0.0)
        self.assertFalse(tracker.update(image, 0.001).valid)

    def test_decodes_a_large_jpeg_small(self):
        image = cv2.resize(texture(), (1184, 656))
        _, jpeg = cv2.imencode(".jpg", image)
        self.assertEqual(decode_gray(jpeg.tobytes()).shape, (164, 296))


class FakeMotion:
    """Drives the control law without a camera."""

    def __init__(self, stabilizer: Stabilizer):
        self.s = stabilizer
        self.t = 0.0

    def frame(self, forward=0.0, lateral=0.0, yaw=0.0, dt=0.05):
        from control.visual_motion import Motion
        self.t += dt
        motion = Motion(self.t, dt, True, 100, 100, lateral=lateral, forward=forward, yaw_rate=yaw)
        turning = abs(self.s.sticks[1]) > ACTIVE_STICK
        self.s.last_good = self.t
        self.s._control(motion, self.t, turning)

    def out(self, throttle=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        return self.s.correction(self.t, throttle, yaw, pitch, roll)


class ControlLawTests(unittest.TestCase):
    def setUp(self):
        self.s = Stabilizer()
        self.f = FakeMotion(self.s)

    def test_pushes_against_the_slide_on_both_axes(self):
        self.f.frame(forward=0.5, lateral=-0.5)
        pitch, roll = self.f.out()
        self.assertLess(pitch, 0.0)
        self.assertGreater(roll, 0.0)

    def test_a_still_drone_gets_exactly_centred_sticks(self):
        # Any pulse past the deadband would clear the flight controller's own position hold.
        for i in range(100):
            noise = 0.005 if i % 2 else -0.005
            self.f.frame(forward=noise, lateral=-noise)
            self.assertEqual(self.f.out(), (0.0, 0.0))

    def test_holds_the_spot_after_stopping(self):
        self.f.frame()
        for _ in range(20):
            self.f.frame(lateral=0.2)
            self.f.out()
        for _ in range(20):
            self.f.frame()
        self.assertTrue(self.s.lateral.held)
        self.assertLess(self.s.lateral.command, 0.0)

    def test_a_stop_ends_when_the_speed_crosses_zero(self):
        self.f.frame(lateral=0.8)
        self.assertLess(self.s.lateral.command, -self.s.gains.limit / 2)
        for _ in range(6):
            self.f.frame(lateral=-0.8)
        self.assertTrue(self.s.lateral.held)
        self.assertLessEqual(abs(self.s.lateral.command), self.s.gains.limit)

    def test_the_pilots_stick_wins_on_its_axis_at_once(self):
        self.f.frame(forward=0.5, lateral=0.5)
        pitch, roll = self.f.out(pitch=0.5)
        self.assertEqual(pitch, 0.0)
        self.assertNotEqual(roll, 0.0)

    def test_letting_go_of_a_stick_forgets_the_old_spot(self):
        self.f.frame()
        self.f.frame()
        self.assertTrue(self.s.forward.held)
        self.f.out(pitch=0.5)
        self.f.frame(forward=0.4)
        self.assertFalse(self.s.forward.held)
        self.assertEqual(self.s.forward.error, 0.0)

    def test_no_correction_while_turning(self):
        self.f.frame()
        self.f.frame(lateral=0.3)
        self.f.out(yaw=0.5)
        self.f.frame(forward=0.5, lateral=0.5)
        self.assertEqual(self.s.forward.command, 0.0)
        self.assertEqual(self.s.lateral.command, 0.0)
        self.assertFalse(self.s.lateral.held)

    def test_small_corrections_pulse_to_the_same_average(self):
        self.s.forward.command = -0.012  # below the deadband, and would do nothing as a steady stick
        self.s.last_good = self.f.t
        out = [self.f.out()[0] for _ in range(1000)]
        self.assertTrue(all(v == 0.0 or abs(v) >= self.s.gains.deadband for v in out))
        self.assertAlmostEqual(sum(out) / len(out), -0.012, places=3)

    def test_corrections_stop_when_the_video_does(self):
        self.f.frame(forward=0.5)
        self.assertNotEqual(self.f.out()[0], 0.0)
        gains = self.s.gains
        self.f.t += gains.stale_s + gains.fade_s + 0.01
        self.assertEqual(self.f.out(), (0.0, 0.0))
        self.assertEqual(self.s.state, "lost")

    def test_corrections_fade_rather_than_step(self):
        self.f.frame(forward=0.8)
        full = self.f.out()[0]
        self.f.t += self.s.gains.stale_s + self.s.gains.fade_s / 2
        self.assertAlmostEqual(self.f.out()[0], full / 2, delta=0.01)

    def test_correction_never_exceeds_the_limit(self):
        self.s.forward.held = self.s.lateral.held = True
        self.f.frame(forward=3.0, lateral=-3.0)
        pitch, roll = self.f.out()
        self.assertLessEqual(abs(pitch), self.s.gains.limit)
        self.assertLessEqual(abs(roll), self.s.gains.limit)

    def test_gives_up_if_the_drone_keeps_running_away(self):
        for _ in range(40):
            self.f.frame(forward=2.0)
        self.assertIsNotNone(self.s.fault)
        self.assertEqual(self.s.state, "fault")
        self.assertEqual(self.f.out(), (0.0, 0.0))

    def test_oscillation_lowers_the_gain(self):
        self.s.forward.held = True
        for i in range(30):
            self.f.frame(forward=0.6 if i % 2 else -0.6)
        self.assertLess(self.s.gain_scale, 1.0)


class FlightStateTests(unittest.TestCase):
    def test_off_means_off(self):
        s = Stabilizer(enabled=False)
        s.forward.command = 0.3
        self.assertEqual(s.correction(1.0, 0, 0, 0, 0), (0.0, 0.0))

    def test_waits_out_the_climb(self):
        s = Stabilizer()
        s.on_takeoff(10.0)
        s.forward.command = 0.3
        s.last_good = 11.0
        self.assertEqual(s.correction(11.0, 0, 0, 0, 0), (0.0, 0.0))

    def test_stops_on_landing(self):
        s = Stabilizer()
        s.forward.command = 0.3
        s.last_good = 1.0
        s.on_land()
        self.assertEqual(s.correction(1.0, 0, 0, 0, 0), (0.0, 0.0))

    def test_turning_it_off_and_on_clears_a_fault(self):
        s = Stabilizer()
        s.fault = "something"
        s.set_enabled(False)
        s.set_enabled(True)
        self.assertIsNone(s.fault)


class Recorder:
    def __init__(self):
        self.seen = []

    def build_control_packet(self, model):
        self.seen.append((model.pitch, model.roll))
        return b""

    def send_control_packet(self, packet):
        pass


class FlightControllerTests(unittest.TestCase):
    def setUp(self):
        self.model = S2xDroneModel()
        self.model.airborne = True
        self.stabilizer = Stabilizer()
        self.link = Recorder()
        self.flight = FlightController(self.model, self.link, stabilizer=self.stabilizer)
        self.flight.last_control_source = "frontend"
        self.stabilizer.last_good = 1.0
        self.stabilizer.forward.command = 0.25

    def test_correction_reaches_the_packet_but_not_the_model(self):
        self.model.set_strategy(LinearStrategy())
        self.flight.tick(1 / 80, 1.0)
        sent_pitch, _ = self.link.seen[-1]
        self.assertAlmostEqual(sent_pitch, self.model._scale_normalised(0.25))
        self.assertEqual(self.model.pitch, self.model.center_value)

    def test_the_keyboard_ramp_does_not_absorb_the_correction(self):
        self.model.set_strategy(IncrementalStrategy())
        for _ in range(40):
            self.flight.tick(1 / 80, 1.0)
        self.assertEqual(self.model.pitch, self.model.center_value)

    def test_a_plugin_or_the_swarm_flies_uncorrected(self):
        self.model.set_strategy(LinearStrategy())
        for source in ("follow", "swarm"):
            self.flight.last_control_source = source
            self.flight.tick(1 / 80, 1.0)
            self.assertEqual(self.link.seen[-1][0], self.model.center_value)

    def test_no_stabilizer_behaves_as_before(self):
        self.flight.stabilizer = None
        self.model.set_strategy(LinearStrategy())
        self.flight.tick(1 / 80, 1.0)
        self.assertEqual(self.link.seen[-1][0], self.model.center_value)

    def test_a_stabilizer_bug_never_stops_the_control_stream(self):
        def broken(*args):
            raise RuntimeError("bug")
        self.stabilizer.correction = broken
        self.model.set_strategy(LinearStrategy())
        with self.assertLogs("services.flight_controller", level="ERROR"):
            self.flight.tick(1 / 80, 1.0)
        self.assertEqual(self.link.seen[-1][0], self.model.center_value)
        self.assertIsNone(self.flight.stabilizer)

    def test_a_flight_ending_stops_corrections(self):
        self.model.set_strategy(LinearStrategy())
        self.model.airborne = False
        self.flight.tick(1 / 80, 1.0)
        self.assertFalse(self.stabilizer.airborne)
        self.assertEqual(self.link.seen[-1][0], self.model.center_value)


class ClosedLoopTests(unittest.TestCase):
    """Real rendered frames and the real packet path, against the simulator's flight controller."""

    def test_holds_position_when_the_flow_sensor_is_lost(self):
        case = Case("hover", "room", flow=False, wind=0.5, seed=1)
        before, after = fly(case, False), fly(case, True)
        self.assertLess(after["drift"], before["drift"] / 2, f"{before} {after}")
        self.assertIsNone(after["fault"])

    def test_leaves_a_working_flow_sensor_alone(self):
        case = Case("hover", "room", flow=True, wind=0.0, seed=1)
        before, after = fly(case, False), fly(case, True)
        self.assertLess(after["drift"], before["drift"] + 0.1, f"{before} {after}")


if __name__ == "__main__":
    unittest.main()
