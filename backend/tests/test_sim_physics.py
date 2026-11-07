import math
import unittest

from sim.config import DroneSpec
from sim.physics import Battery, Quadcopter
from sim.scene import Box, Scene

DT = 1 / 400
SPEC = DroneSpec()
OPEN = Scene(name="open")


def hover_command(drone: Quadcopter) -> float:
    """Motor command that balances gravity at the current battery voltage."""
    return SPEC.mass_kg * 9.81 / 4 / drone.max_thrust_now()


def flying_drone(z: float = 2.0, scene: Scene = OPEN) -> Quadcopter:
    drone = Quadcopter(SPEC, scene)
    drone.place(0.0, 0.0, z, 0.0)
    drone.set_motor_thrust([SPEC.mass_kg * 9.81 / 4] * 4)
    return drone


def run(drone: Quadcopter, seconds: float, command=None, wind=(0.0, 0.0, 0.0)) -> None:
    for _ in range(int(seconds / DT)):
        drone.step(DT, command if command is not None else [hover_command(drone)] * 4, wind)


class FreeFallTests(unittest.TestCase):
    def test_a_drop_from_takeoff_height_lands_when_gravity_says(self):
        drone = Quadcopter(SPEC, OPEN)
        drone.place(0.0, 0.0, 1.2 + SPEC.gear_height_m, 0.0)
        elapsed = 0.0
        while not drone.impacts and elapsed < 2.0:
            drone.step(DT, [0.0] * 4, (0.0, 0.0, 0.0))
            elapsed += DT
        self.assertAlmostEqual(elapsed, math.sqrt(2 * 1.2 / 9.81), delta=0.02)
        # Exact solution for a fall with quadratic drag.
        terminal = math.sqrt(SPEC.mass_kg * 9.81 / SPEC.body_drag)
        expected = terminal * math.sqrt(1 - math.exp(-2 * 9.81 * 1.2 / terminal**2))
        self.assertAlmostEqual(drone.impacts[0].speed, expected, delta=0.05)


class ForceDirectionTests(unittest.TestCase):
    def test_balanced_thrust_holds_altitude(self):
        drone = flying_drone()
        run(drone, 1.0)
        self.assertAlmostEqual(drone.z, 2.0, delta=0.03)

    def test_more_thrust_on_the_left_rolls_right_and_flies_right(self):
        drone = flying_drone()
        hover = hover_command(drone)
        run(drone, 0.3, [hover * 1.05, hover * 0.95, hover * 0.95, hover * 1.05])
        run(drone, 0.3)
        self.assertGreater(drone.roll(), 0.0)
        self.assertLess(drone.vy, 0.0)  # +y is left

    def test_more_thrust_at_the_rear_pitches_down_and_flies_forward(self):
        drone = flying_drone()
        hover = hover_command(drone)
        run(drone, 0.3, [hover * 0.95, hover * 0.95, hover * 1.05, hover * 1.05])
        run(drone, 0.3)
        self.assertGreater(drone.pitch(), 0.0)
        self.assertGreater(drone.vx, 0.0)

    def test_reaction_torque_turns_the_drone(self):
        drone = flying_drone()
        hover = hover_command(drone)
        run(drone, 0.5, [hover * 1.1, hover * 0.9, hover * 1.1, hover * 0.9])
        self.assertGreater(drone.r, 0.0)  # counter-clockwise from above

    def test_yaw_rotates_thrust_tilt_into_world_axes(self):
        drone = flying_drone()
        drone.place(0.0, 0.0, 2.0, math.pi / 2)  # facing +y
        hover = hover_command(drone)
        run(drone, 0.3, [hover * 0.95, hover * 0.95, hover * 1.05, hover * 1.05])
        run(drone, 0.3)
        self.assertGreater(drone.vy, abs(drone.vx))


class AerodynamicsTests(unittest.TestCase):
    def test_drag_limits_speed_at_full_tilt_near_the_estimate(self):
        drone = flying_drone(z=50.0)
        drone.set_attitude(0.0, 0.436, 0.0)
        speed = 0.0
        for _ in range(int(20 / DT)):
            drone.set_attitude(0.0, 0.436, 0.0)
            thrust = SPEC.mass_kg * 9.81 / math.cos(0.436) / 4 / drone.max_thrust_now()
            drone.step(DT, [thrust] * 4, (0.0, 0.0, 0.0))
            speed = drone.vx
        self.assertAlmostEqual(speed, 5.0, delta=0.8)

    def test_wind_pushes_a_drone_downwind(self):
        drone = flying_drone(z=10.0)
        run(drone, 3.0, wind=(2.0, 0.0, 0.0))
        self.assertGreater(drone.vx, 0.3)

    def test_ground_effect_adds_lift_near_the_floor(self):
        high = flying_drone(z=2.0)
        low = flying_drone(z=SPEC.gear_height_m + 0.02)
        command = [hover_command(high)] * 4
        high.step(DT, command, (0.0, 0.0, 0.0))
        low.step(DT, command, (0.0, 0.0, 0.0))
        self.assertGreater(low.vz, high.vz)


class BatteryTests(unittest.TestCase):
    def test_voltage_sags_under_load(self):
        battery = Battery(SPEC.capacity_ah, SPEC.resistance_ohm, soc=1.0)
        self.assertAlmostEqual(battery.ocv(), 4.2, delta=0.01)
        self.assertLess(battery.loaded_voltage(6.0), battery.ocv())

    def test_rated_current_empties_the_usable_charge_in_the_rated_time(self):
        drone = flying_drone()
        run(drone, 2.0)
        rated = SPEC.capacity_ah * SPEC.usable_fraction / (SPEC.rated_flight_min / 60)
        self.assertAlmostEqual(drone.current_a, rated, delta=rated * 0.15)

    def test_idle_electronics_still_drain_the_battery(self):
        drone = Quadcopter(SPEC, OPEN)
        start = drone.battery.soc
        for _ in range(int(60 / 0.01)):
            drone.step(0.01, [0.0] * 4, (0.0, 0.0, 0.0))
        used = (start - drone.battery.soc) * SPEC.capacity_ah * 3600
        self.assertAlmostEqual(used, SPEC.electronics_current_a * 60, delta=2.0)

    def test_low_voltage_reduces_available_thrust(self):
        full = Quadcopter(SPEC, OPEN)
        empty = Quadcopter(SPEC, OPEN, battery_soc=0.1)
        self.assertLess(empty.max_thrust_now(), full.max_thrust_now())


class ContactTests(unittest.TestCase):
    def test_landed_drone_rests_level_on_its_gear(self):
        drone = Quadcopter(SPEC, OPEN)
        drone.place(0.0, 0.0, 0.5, 0.0)
        drone.set_attitude(0.2, 0.1, 0.0)
        for _ in range(int(3 / DT)):
            drone.step(DT, [0.0] * 4, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(drone.z, SPEC.gear_height_m, delta=0.005)
        self.assertTrue(drone.on_ground)
        self.assertAlmostEqual(drone.roll(), 0.0, delta=0.02)
        self.assertAlmostEqual(drone.pitch(), 0.0, delta=0.02)

    def test_settling_on_the_ground_keeps_the_heading_after_a_full_turn(self):
        drone = Quadcopter(SPEC, OPEN)
        drone.place(0.0, 0.0, 0.3, 0.0)
        drone.set_attitude(0.1, 0.0, math.radians(200))
        drone.qw, drone.qx, drone.qy, drone.qz = -drone.qw, -drone.qx, -drone.qy, -drone.qz
        for _ in range(int(2 / DT)):
            drone.step(DT, [0.0] * 4, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(math.degrees(drone.yaw()) % 360, 200, delta=1.0)

    def test_a_wall_stops_the_drone_and_reports_the_impact(self):
        room = Scene(name="room", bounds=Box((-2, -2, 0), (2, 2, 3), "wall"))
        drone = flying_drone(z=1.0, scene=room)
        drone.vx = 3.0
        run(drone, 1.0)
        self.assertLessEqual(drone.x, 2.0 - SPEC.collision_radius_m + 0.01)
        hits = [i for i in drone.impacts if i.normal[0] < -0.5]
        self.assertTrue(hits)
        self.assertGreater(hits[0].speed, 1.5)

    def test_a_box_blocks_the_drone(self):
        scene = Scene(name="box", boxes=(Box((3, -1, 0), (4, 1, 2), "wood"),))
        drone = flying_drone(z=1.0, scene=scene)
        drone.vx = 3.0
        run(drone, 2.0)
        self.assertLess(drone.x, 3.0)


class SceneTests(unittest.TestCase):
    def test_collide_reports_the_nearest_wall_of_a_room(self):
        room = Scene(name="room", bounds=Box((-2, -2, 0), (2, 2, 3), "wall"))
        self.assertIsNone(room.collide(0.0, 0.0, 1.0, 0.12))
        normal, depth = room.collide(1.95, 0.0, 1.0, 0.12)
        self.assertEqual(normal, (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(depth, 0.07, places=5)

    def test_collide_finds_the_closest_face_of_a_box(self):
        scene = Scene(name="box", boxes=(Box((1, -1, 0), (2, 1, 1), "wood"),))
        normal, depth = scene.collide(0.95, 0.0, 0.5, 0.12)
        self.assertEqual(normal, (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(depth, 0.07, places=5)
        normal, _ = scene.collide(1.5, 0.0, 1.05, 0.12)
        self.assertEqual(normal, (0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
