import unittest

from slam.runner import run
from slam.sequences import SimFlight, hall


class SimulatedFlightTests(unittest.TestCase):
    """Fly the simulator and score the tracker against the exact truth. Slow: renders and tracks real frames."""

    @classmethod
    def setUpClass(cls):
        flight = SimFlight(hall())
        flight.forward(2.0, 0.5).arc(180, speed=0.4, turn=0.4)
        cls.flight = flight
        cls.result = run(flight.frames, flight.intrinsics)

    def test_tracks_nearly_every_frame(self):
        self.assertLess(self.result.lost_frames, 0.2 * len(self.flight.frames))

    def test_position_error_is_small_after_alignment(self):
        report = self.result.report
        self.assertIsNotNone(report)
        self.assertLess(report.drift_percent, 4.0, str(report))
        self.assertLess(report.ate_rmse, 0.35, str(report))

    def test_keeps_up_with_the_cameras_frame_rate(self):
        self.assertLess(self.result.mean_ms, 1000 / 17.6)


if __name__ == "__main__":
    unittest.main()
