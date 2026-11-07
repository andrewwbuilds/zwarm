import unittest

from models.s2x_rc import S2xDroneModel


class S2xFlightStateTest(unittest.TestCase):
    """Takeoff and land share one toggle bit, so a repeat must not reverse the flight."""

    def test_takeoff_then_land_both_go_out(self):
        model = S2xDroneModel()
        self.assertTrue(model.takeoff())
        self.assertTrue(model.takeoff_flag)
        self.assertTrue(model.airborne)
        self.assertTrue(model.land())
        self.assertTrue(model.land_flag)
        self.assertFalse(model.airborne)

    def test_second_takeoff_is_dropped(self):
        model = S2xDroneModel()
        model.takeoff()
        model.takeoff_flag = False  # the adapter consumed the first one
        self.assertFalse(model.takeoff())
        self.assertFalse(model.takeoff_flag)
        self.assertTrue(model.airborne)

    def test_land_on_the_ground_is_dropped(self):
        model = S2xDroneModel()
        self.assertFalse(model.land())
        self.assertFalse(model.land_flag)
        self.assertFalse(model.airborne)

    def test_estop_always_goes_out_and_grounds(self):
        model = S2xDroneModel()
        model.takeoff()
        model.emergency_stop()
        self.assertTrue(model.stop_flag)
        self.assertFalse(model.airborne)
        model.emergency_stop()  # also on the ground
        self.assertTrue(model.stop_flag)
        self.assertTrue(model.takeoff())


if __name__ == "__main__":
    unittest.main()
