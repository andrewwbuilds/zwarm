"""Rigid-body model of the S20: motors, battery, aerodynamics, and contact.

Axes: world x forward at takeoff, y left, z up. Body axes are forward, left, up.
Motor order is front-left, front-right, rear-right, rear-left. Front-left and
rear-right react counter-clockwise.

Written with plain floats because it runs hundreds of times per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim.config import DroneSpec

GRAVITY = 9.81
AIR_DENSITY = 1.225
_OCV_TABLE = (
    (0.00, 3.30), (0.05, 3.50), (0.10, 3.62), (0.20, 3.70), (0.30, 3.74), (0.40, 3.77),
    (0.50, 3.80), (0.60, 3.84), (0.70, 3.90), (0.80, 3.98), (0.90, 4.08), (1.00, 4.20),
)  # estimate: typical Li-ion discharge curve


class Battery:
    def __init__(self, capacity_ah: float, resistance_ohm: float, soc: float = 1.0):
        self.capacity_ah = capacity_ah
        self.resistance_ohm = resistance_ohm
        self.soc = soc

    def ocv(self) -> float:
        soc = self.soc
        for (s0, v0), (s1, v1) in zip(_OCV_TABLE, _OCV_TABLE[1:]):
            if soc <= s1:
                return v0 + (v1 - v0) * (soc - s0) / (s1 - s0)
        return _OCV_TABLE[-1][1]

    def loaded_voltage(self, current_a: float) -> float:
        return max(self.ocv() - current_a * self.resistance_ohm, 2.5)

    def drain(self, current_a: float, dt: float) -> None:
        self.soc = max(0.0, self.soc - current_a * dt / (3600.0 * self.capacity_ah))


@dataclass(frozen=True)
class Impact:
    speed: float
    normal: tuple[float, float, float]
    kind: str  # "ground" or "obstacle"


class Quadcopter:
    def __init__(self, spec: DroneSpec, scene, battery_soc: float = 1.0):
        self.spec = spec
        self.scene = scene
        self.battery = Battery(spec.capacity_ah, spec.resistance_ohm, battery_soc)
        self.x = self.y = 0.0
        self.z = spec.gear_height_m
        self.vx = self.vy = self.vz = 0.0
        self.qw, self.qx, self.qy, self.qz = 1.0, 0.0, 0.0, 0.0
        self.p = self.q = self.r = 0.0  # body rates: roll, pitch, yaw
        self.thrust = [0.0, 0.0, 0.0, 0.0]
        self.current_a = spec.electronics_current_a
        self.voltage = self.battery.loaded_voltage(self.current_a)
        self.on_ground = True
        self.impacts: list[Impact] = []
        self._steps = 0
        self._resting = False  # on top of furniture at the last contact check
        # Scale momentum-theory rotor power so hovering draws the rated-flight-time current.
        hover_thrust = spec.mass_kg * GRAVITY / 4.0
        rated_a = spec.capacity_ah * spec.usable_fraction / (spec.rated_flight_min / 60.0)
        average_voltage = 3.8  # estimate: mid-discharge
        self._power_per_thrust15 = (
            (rated_a - spec.electronics_current_a) * average_voltage / (4.0 * hover_thrust ** 1.5)
        )

    # ------------------------------------------------------------- pose helpers
    def place(self, x: float, y: float, z: float, yaw: float) -> None:
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.set_attitude(0.0, 0.0, yaw)
        self.on_ground = z <= self.spec.gear_height_m

    def set_attitude(self, roll: float, pitch: float, yaw: float) -> None:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        self.qw = cy * cp * cr + sy * sp * sr
        self.qx = cy * cp * sr - sy * sp * cr
        self.qy = cy * sp * cr + sy * cp * sr
        self.qz = sy * cp * cr - cy * sp * sr
        self.p = self.q = self.r = 0.0

    def set_motor_thrust(self, thrust: list[float]) -> None:
        self.thrust = list(thrust)

    def rotation(self) -> tuple[float, ...]:
        """Body-to-world rotation matrix, row by row."""
        w, a, b, c = self.qw, self.qx, self.qy, self.qz
        return (
            1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b),
            2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a),
            2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b),
        )

    def roll(self) -> float:
        w, a, b, c = self.qw, self.qx, self.qy, self.qz
        return math.atan2(2 * (b * c + w * a), 1 - 2 * (a * a + b * b))

    def pitch(self) -> float:
        w, a, b, c = self.qw, self.qx, self.qy, self.qz
        return math.asin(max(-1.0, min(1.0, 2 * (w * b - a * c))))

    def yaw(self) -> float:
        w, a, b, c = self.qw, self.qx, self.qy, self.qz
        return math.atan2(2 * (a * b + w * c), 1 - 2 * (b * b + c * c))

    def max_thrust_now(self) -> float:
        volts = self.battery.loaded_voltage(self.current_a)
        return self.spec.max_thrust_per_motor_n * (volts / self.spec.nominal_voltage) ** 2

    def pop_impacts(self) -> list[Impact]:
        impacts, self.impacts = self.impacts, []
        return impacts

    # -------------------------------------------------------------- simulation
    def step(self, dt: float, commands, wind=(0.0, 0.0, 0.0)) -> None:
        spec = self.spec
        mass = spec.mass_kg

        # Battery and motors. Thrust follows the command with a lag, and sags with voltage.
        volts = self.battery.loaded_voltage(self.current_a)
        self.voltage = volts
        max_thrust = spec.max_thrust_per_motor_n * (volts / spec.nominal_voltage) ** 2
        alpha = min(1.0, dt / spec.motor_lag_s)
        thrust = self.thrust
        for i in range(4):
            command = commands[i]
            command = 0.0 if command < 0.0 else 1.0 if command > 1.0 else command
            thrust[i] += (command * max_thrust - thrust[i]) * alpha
        t0, t1, t2, t3 = thrust
        power = self._power_per_thrust15 * (t0 * math.sqrt(t0) + t1 * math.sqrt(t1)
                                            + t2 * math.sqrt(t2) + t3 * math.sqrt(t3))
        self.current_a = spec.electronics_current_a + power / volts
        self.battery.drain(self.current_a, dt)

        w, a, b, c = self.qw, self.qx, self.qy, self.qz
        r00 = 1 - 2 * (b * b + c * c)
        r10 = 2 * (a * b + w * c)
        r02 = 2 * (a * c + w * b)
        r12 = 2 * (b * c - w * a)
        r22 = 1 - 2 * (a * a + b * b)

        # Lift is stronger close to the floor (ground effect).
        total = t0 + t1 + t2 + t3
        x, y, z = self.x, self.y, self.z
        height = z if z > 0.03 else 0.03
        ratio = spec.prop_radius_m / (4.0 * height)
        ratio *= ratio
        total_effective = total / (1.0 - (ratio if ratio < 0.25 else 0.25))

        # Translation: thrust along body-up, drag against the air, gravity.
        vx, vy, vz = self.vx, self.vy, self.vz
        air_x, air_y, air_z = vx - wind[0], vy - wind[1], vz - wind[2]
        air_speed = math.sqrt(air_x * air_x + air_y * air_y + air_z * air_z)
        rotor_drag = spec.rotor_drag_at_hover * total / (mass * GRAVITY)
        drag = (rotor_drag + spec.body_drag * air_speed) / mass
        lift = total_effective / mass
        vx += (lift * r02 - drag * air_x) * dt
        vy += (lift * r12 - drag * air_y) * dt
        vz += (lift * r22 - drag * air_z - GRAVITY) * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # Rotation: differential thrust, reaction torque, and Euler's equations.
        p, q, r = self.p, self.q, self.r
        arm = spec.arm_offset_m
        ix, iy, iz = spec.inertia_xx, spec.inertia_yy, spec.inertia_zz
        damping = spec.angular_damping
        torque_x = arm * (t0 + t3 - t1 - t2) - damping * p
        torque_y = arm * (t2 + t3 - t0 - t1) - damping * q
        torque_z = spec.yaw_torque_per_newton_m * (t0 + t2 - t1 - t3) - damping * r
        p += (torque_x + (iy - iz) * q * r) / ix * dt
        q += (torque_y + (iz - ix) * p * r) / iy * dt
        r += (torque_z + (ix - iy) * p * q) / iz * dt

        half = 0.5 * dt
        dw = -half * (a * p + b * q + c * r)
        da = half * (w * p + b * r - c * q)
        db = half * (w * q + c * p - a * r)
        dc = half * (w * r + a * q - b * p)
        w += dw
        a += da
        b += db
        c += dc

        # Contact with the floor.
        on_ground = False
        floor = spec.gear_height_m
        if z <= floor:
            on_ground = True
            z = floor
            if vz < 0.0:
                if vz < -0.3:
                    self._record_impact(-vz, (0.0, 0.0, 1.0), "ground")
                vz = -0.2 * vz if vz < -1.0 else 0.0

        # Contact with walls and furniture, checked at 100 Hz or so.
        self._steps += 1
        if self._steps % 4 == 0:
            self._resting = False
            hit = self.scene.collide(x, y, z, spec.collision_radius_m)
            if hit is not None:
                (nx, ny, nz), depth = hit
                x += nx * depth
                y += ny * depth
                z += nz * depth
                into = vx * nx + vy * ny + vz * nz
                if into < 0.0:
                    speed = -into
                    if speed > 0.3:
                        self._record_impact(speed, (nx, ny, nz), "obstacle")
                    bounce = 1.3 * into
                    vx -= bounce * nx
                    vy -= bounce * ny
                    vz -= bounce * nz
                    vx, vy, vz = vx * 0.8, vy * 0.8, vz * 0.8
                    # The drone is hit off-centre, so it knocks the attitude too.
                    kick = 0.5 * speed
                    wx, wy = -ny * kick, nx * kick
                    p += r00 * wx + r10 * wy
                    q += (2 * (a * b - w * c)) * wx + (1 - 2 * (a * a + c * c)) * wy
                if nz > 0.7:
                    self._resting = True
        on_ground = on_ground or self._resting

        # Resting on the gear: friction stops it sliding and it settles level.
        if on_ground and total_effective < mass * GRAVITY * 1.02:
            horizontal = math.hypot(vx, vy)
            if horizontal > 0.0:
                scale = max(0.0, horizontal - 0.8 * GRAVITY * dt) / horizontal
                vx *= scale
                vy *= scale
            blend = 1.0 - math.exp(-15.0 * dt)
            yaw = math.atan2(2 * (a * b + w * c), 1 - 2 * (b * b + c * c))
            level_w, level_c = math.cos(yaw / 2), math.sin(yaw / 2)
            if level_w * w + level_c * c < 0.0:  # q and -q are the same attitude: blend toward the near one
                level_w, level_c = -level_w, -level_c
            w += (level_w - w) * blend
            a -= a * blend
            b -= b * blend
            c += (level_c - c) * blend
            settle = math.exp(-30.0 * dt)
            p *= settle
            q *= settle
            r *= math.exp(-5.0 * dt)

        norm = 1.0 / math.sqrt(w * w + a * a + b * b + c * c)
        self.qw, self.qx, self.qy, self.qz = w * norm, a * norm, b * norm, c * norm
        self.p, self.q, self.r = p, q, r
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.on_ground = on_ground

    def _record_impact(self, speed: float, normal, kind: str) -> None:
        self.impacts.append(Impact(speed, normal, kind))
        del self.impacts[:-32]
