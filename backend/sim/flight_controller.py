"""The flight controller: reads control packets, runs flight modes, and drives the motors.

The real firmware is closed, and nothing on this page has been measured on the real
drone. It's a typical toy-drone controller shaped to the published behaviour: one-key
takeoff, altitude hold from the barometer, position hold from the optical-flow reports.
"""

from __future__ import annotations

import math
import random
from collections import deque

from sim.physics import GRAVITY
from sim.rc_frames import parse_frame

_MOTORS_OFF = [0.0, 0.0, 0.0, 0.0]
_SPINUP_COMMAND = 0.12  # estimate
_ATTITUDE_GAIN = 5.0  # estimate
_RATE_GAIN = 14.0  # estimate
_YAW_RATE_GAIN = 6.0  # estimate
_VELOCITY_GAIN = 0.20  # estimate: radians of tilt per m/s of error
_VELOCITY_INTEGRAL = 0.08  # estimate
_HOLD_GAIN = 0.8  # estimate
_HOLD_SPEED_LIMIT = 0.6  # estimate
_FLOW_MIN_QUALITY = 40  # estimate


def _clamp(value: float, limit: float) -> float:
    return -limit if value < -limit else limit if value > limit else value


class FlightController:
    def __init__(self, spec, tuning, firmware, drone, scene, rng: random.Random):
        self.spec = spec
        self.tuning = tuning
        self.firmware = firmware
        self.drone = drone
        self.scene = scene
        self.rng = rng
        self.t = 0.0
        self.state = "READY"
        self.events: deque[tuple[float, str]] = deque(maxlen=200)
        self.listener = None
        self.unlocked = firmware.unlock is None
        self.link_alive = False
        self.flow_valid = False
        self.last_frame_time = None
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.low_battery = False
        self.rc_rate_hz = 0.0
        self.roll_stick = self.pitch_stick = self.throttle_stick = self.yaw_stick = 0.0
        self.headless = False
        self.trim_roll = rng.uniform(-0.007, 0.007)  # estimate: about 0.4 degrees of level error
        self.trim_pitch = rng.uniform(-0.007, 0.007)
        self._state_since = 0.0
        self._link_lost_since = None
        self._ground_since = None
        self._tilt_since = None
        self._battery_low_since = None
        self._calibrate_until = 0.0
        self._flag_since = {"fly": None, "stop": None, "calibrate": None}
        self._flag_fired = {"fly": False, "stop": False, "calibrate": False}
        self._last_rate_time = None
        self._rate_interval = 0.05
        self._last_rejection_note = -math.inf
        self._unlock_seen_at = None
        self._pair_up_at = None
        self._z_est = drone.z
        self._vz_est = 0.0
        self._baro_bias = 0.0
        self._z_target = drone.z
        self._z_held = True
        self._z_integral = 0.0
        self._flow_time = -math.inf
        self.flow_forward = self.flow_left = 0.0
        self._est_x = self._est_y = 0.0
        self._hold = None
        self._forward_integral = self._left_integral = 0.0
        self._headless_reference = 0.0

    # ------------------------------------------------------------------- events
    def _event(self, message: str) -> None:
        self.events.append((round(self.t, 2), message))
        if self.listener is not None:
            self.listener(self.t, message)

    def _set_state(self, state: str, message: str | None = None) -> None:
        self.state = state
        self._state_since = self.t
        if message:
            self._event(message)

    # ------------------------------------------------------------ incoming UART
    def on_uart(self, t: float, data: bytes) -> None:
        if data[:2] == b"\x77\x0e":
            self._on_flow_frame(t, data)
            return
        parsed = parse_frame(data, self.firmware.rc_format)
        if parsed is None:
            self._reject(t, f"it isn't a valid {self.firmware.rc_format} packet")
            return
        frame, mode_byte = parsed
        if self.firmware.required_byte7 is not None and mode_byte != self.firmware.required_byte7:
            self._reject(t, f"mode byte is {mode_byte:#04x}, this firmware needs {self.firmware.required_byte7:#04x}")
            return
        if self._last_rate_time is not None:
            self._rate_interval += (t - self._last_rate_time - self._rate_interval) * 0.1
        self._last_rate_time = t
        if self._rate_interval > 0.0:
            self.rc_rate_hz = 1.0 / self._rate_interval
        limit = self.firmware.max_rc_rate_hz
        if limit is not None and self.rc_rate_hz > limit:
            self._reject(t, f"packets arrive at {self.rc_rate_hz:.0f} per second, this firmware handles {limit:.0f}")
            return

        self.frames_accepted += 1
        self.last_frame_time = t
        if not self.link_alive:
            self.link_alive = True
            self._link_lost_since = None
            self._event("Control link up")
        self.roll_stick = self._deadband(frame.roll)
        self.pitch_stick = self._deadband(frame.pitch)
        self.throttle_stick = self._deadband(frame.throttle)
        self.yaw_stick = self._deadband(frame.yaw)
        self.headless = frame.headless
        self._watch_unlock(t, frame)
        self._flag("stop", frame.stop, t, self._on_stop)
        self._flag("fly", frame.fly and not frame.stop, t, self._on_fly)
        self._flag("calibrate", frame.calibrate, t, self._on_calibrate)

    def _reject(self, t: float, reason: str) -> None:
        self.frames_rejected += 1
        if t - self._last_rejection_note > 5.0:
            self._last_rejection_note = t
            self._event(f"Flight controller ignored a control packet: {reason}")

    def _deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self.tuning.stick_deadband else value

    def _flag(self, name: str, high: bool, t: float, action) -> None:
        """A command fires once, after its bit has been set for the firmware's minimum time."""
        since = self._flag_since[name]
        if high:
            if since is None:
                self._flag_since[name] = since = t
                self._flag_fired[name] = False
            if not self._flag_fired[name] and t - since >= self.firmware.fly_min_hold_s:
                self._flag_fired[name] = True
                action()
        elif since is not None:
            if not self._flag_fired[name]:
                held_ms = (t - since) * 1000
                needed_ms = self.firmware.fly_min_hold_s * 1000
                self._event(f"{name.capitalize()} signal lasted {held_ms:.0f} ms, this firmware needs "
                            f"{needed_ms:.0f} ms: ignored")
            self._flag_since[name] = None

    def _on_flow_frame(self, t: float, data: bytes) -> None:
        checksum = 0
        for value in data[2:12]:
            checksum ^= value
        if len(data) != 14 or data[13] != 0x99 or checksum != data[12]:
            return
        self._flow_time = t
        quality = data[6]
        self.flow_valid = quality >= _FLOW_MIN_QUALITY
        height = max(self._z_est, 0.05)
        self.flow_forward = int.from_bytes(data[2:4], "little", signed=True) / 100.0 * height
        self.flow_left = int.from_bytes(data[4:6], "little", signed=True) / 100.0 * height

    def _watch_unlock(self, t: float, frame) -> None:
        unlock = self.firmware.unlock
        if self.unlocked or unlock is None or unlock == "remote":
            return
        if unlock == "pairing":
            if frame.throttle > 0.8:
                self._pair_up_at = t
            elif frame.throttle < -0.8 and self._pair_up_at is not None and t - self._pair_up_at < 3.0:
                self.unlocked = True
                self._event("Paired: throttle up then down")
            return
        inward = frame.throttle < -0.8 and frame.pitch < -0.8 and frame.yaw > 0.8 and frame.roll < -0.8
        outward = frame.throttle < -0.8 and frame.pitch < -0.8 and frame.yaw < -0.8 and frame.roll > 0.8
        pattern = inward if unlock == "arm_inward" else outward
        if not pattern:
            self._unlock_seen_at = None
        elif self._unlock_seen_at is None:
            self._unlock_seen_at = t
        elif t - self._unlock_seen_at >= 0.5:
            self.unlocked = True
            self._event("Armed: sticks held down")

    # ----------------------------------------------------------------- commands
    def _on_fly(self) -> None:
        if self.state == "READY":
            self._try_takeoff()
        elif self.state in ("SPINUP", "CLIMB", "FLYING"):
            self._set_state("LANDING", "Landing")

    def _try_takeoff(self) -> None:
        tuning = self.tuning
        if self.t < tuning.boot_s:
            self._event("Takeoff refused: still calibrating after power-on")
        elif not self.unlocked:
            if self.firmware.unlock == "remote":
                self._event("Takeoff refused: only the physical remote can arm the motors")
            else:
                self._event(f"Takeoff refused: locked until the {self.firmware.unlock} stick pattern")
        elif self.drone.battery.ocv() < tuning.min_takeoff_ocv_v:
            self._event("Takeoff refused: battery too low")
        else:
            self._z_target = self._z_est
            self._z_held = True
            self._z_integral = 0.0
            self._hold = None
            self._forward_integral = self._left_integral = 0.0
            self._headless_reference = self.drone.yaw()
            self._set_state("SPINUP", "Takeoff: motors spinning up")

    def _on_stop(self) -> None:
        if self.state in ("SPINUP", "CLIMB", "FLYING", "LANDING"):
            self._set_state("READY", "Emergency stop: motors cut")

    def _on_calibrate(self) -> None:
        if self.state == "READY":
            self._calibrate_until = self.t + 2.0
            self.trim_roll *= 0.1
            self.trim_pitch *= 0.1
            self._event("Gyro calibration")

    # ----------------------------------------------------------------- the loop
    def update(self, t: float, dt: float) -> list[float]:
        self.t = t
        self._update_estimates(dt)
        self._watch_impacts()
        self._watch_link(t)
        self._watch_battery(t)
        state = self.state
        if state == "READY" or state == "CRASHED":
            return _MOTORS_OFF
        if state == "SPINUP":
            if t - self._state_since < self.tuning.spinup_s:
                return [_SPINUP_COMMAND] * 4
            self._set_state("CLIMB", "Climbing")
        self._watch_tilt(t)
        self._watch_touchdown(t)
        if self.state not in ("CLIMB", "FLYING", "LANDING"):
            return _MOTORS_OFF
        return self._control(dt)

    def _update_estimates(self, dt: float) -> None:
        drone = self.drone
        rng = self.rng
        self._baro_bias += (-self._baro_bias / 25.0 * dt) + 0.06 * math.sqrt(2.0 * dt / 25.0) * rng.gauss(0.0, 1.0)
        noisy = drone.z + self._baro_bias + rng.gauss(0.0, 0.02)
        self._z_est += (noisy - self._z_est) * min(1.0, dt / 0.03)
        self._vz_est += (drone.vz + rng.gauss(0.0, 0.03) - self._vz_est) * min(1.0, dt / 0.02)
        if self.t - self._flow_time > 0.25:
            self.flow_valid = False

    def _watch_impacts(self) -> None:
        tuning = self.tuning
        for impact in self.drone.pop_impacts():
            if self.state == "CRASHED":
                continue
            if impact.kind == "ground":
                broke = impact.speed > tuning.drop_crash_speed_mps
            else:
                broke = impact.speed > tuning.side_crash_speed_mps
            if broke:
                self._set_state("CRASHED", f"Crashed: hit the {impact.kind} at {impact.speed:.1f} m/s")
            elif impact.speed > 1.0:
                self._event(f"Bumped the {impact.kind} at {impact.speed:.1f} m/s")

    def _watch_link(self, t: float) -> None:
        if self.link_alive and t - self.last_frame_time > self.tuning.rc_timeout_s:
            self.link_alive = False
            self._link_lost_since = t
            self.roll_stick = self.pitch_stick = self.throttle_stick = self.yaw_stick = 0.0
            self._event("Control link lost")
        if (not self.link_alive and self._link_lost_since is not None and self.state in ("CLIMB", "FLYING")
                and t - self._link_lost_since > self.tuning.failsafe_hover_s):
            self._set_state("LANDING", "Link lost for too long: landing")

    def _watch_battery(self, t: float) -> None:
        volts = self.drone.voltage
        self.low_battery = volts < self.tuning.low_battery_warn_v
        if self.state in ("CLIMB", "FLYING") and volts < self.tuning.low_battery_land_v:
            if self._battery_low_since is None:
                self._battery_low_since = t
            elif t - self._battery_low_since > 0.5:
                self._set_state("LANDING", "Battery empty: landing")
        else:
            self._battery_low_since = None

    def _watch_tilt(self, t: float) -> None:
        drone = self.drone
        upright = 1 - 2 * (drone.qx * drone.qx + drone.qy * drone.qy)
        if upright < 0.17:
            if self._tilt_since is None:
                self._tilt_since = t
            elif t - self._tilt_since > 0.4:
                self._set_state("CRASHED", "Crashed: tipped over in the air")
        else:
            self._tilt_since = None

    def _watch_touchdown(self, t: float) -> None:
        landing = self.state == "LANDING"
        forced_down = self.state == "FLYING" and self.throttle_stick < -0.5
        # estimate: like most flight controllers, call it landed once a descent stalls near the floor.
        stalled = landing and self._z_est < 0.25 and abs(self._vz_est) < 0.1
        if (self.drone.on_ground or stalled) and (landing or forced_down):
            if self._ground_since is None:
                self._ground_since = t
            elif t - self._ground_since > 0.3:
                self._set_state("READY", "Landed: motors off")
        else:
            self._ground_since = None

    def _control(self, dt: float) -> list[float]:
        spec, tuning, drone = self.spec, self.tuning, self.drone
        roll_stick, pitch_stick = self.roll_stick, self.pitch_stick
        roll, pitch, yaw = drone.roll() + self.trim_roll, drone.pitch() + self.trim_pitch, drone.yaw()
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        if self.headless:
            delta = yaw - self._headless_reference
            forward = pitch_stick * math.cos(delta) - roll_stick * math.sin(delta)
            left = -pitch_stick * math.sin(delta) - roll_stick * math.cos(delta)
            pitch_stick, roll_stick = forward, -left

        # Horizontal: with flow, sticks set velocity. Without it, they set tilt.
        tilt_limit = tuning.max_tilt_rad
        if self.flow_valid:
            est_forward, est_left = self.flow_forward, self.flow_left
            self._est_x += (est_forward * cos_yaw - est_left * sin_yaw) * dt
            self._est_y += (est_forward * sin_yaw + est_left * cos_yaw) * dt
            command_forward = pitch_stick * tuning.max_speed_mps
            command_left = -roll_stick * tuning.max_speed_mps
            if roll_stick == 0.0 and pitch_stick == 0.0:
                if self._hold is None and math.hypot(est_forward, est_left) < 0.3:
                    self._hold = (self._est_x, self._est_y)
                if self._hold is not None:
                    error_x, error_y = self._hold[0] - self._est_x, self._hold[1] - self._est_y
                    command_forward = _clamp(_HOLD_GAIN * (error_x * cos_yaw + error_y * sin_yaw), _HOLD_SPEED_LIMIT)
                    command_left = _clamp(_HOLD_GAIN * (-error_x * sin_yaw + error_y * cos_yaw), _HOLD_SPEED_LIMIT)
            else:
                self._hold = None
            error_forward = command_forward - est_forward
            error_left = command_left - est_left
            desired_pitch = _VELOCITY_GAIN * error_forward + self._forward_integral
            desired_roll = -(_VELOCITY_GAIN * error_left + self._left_integral)
            if abs(desired_pitch) < tilt_limit:
                self._forward_integral = _clamp(self._forward_integral + _VELOCITY_INTEGRAL * error_forward * dt, 0.25)
            if abs(desired_roll) < tilt_limit:
                self._left_integral = _clamp(self._left_integral + _VELOCITY_INTEGRAL * error_left * dt, 0.25)
            desired_pitch = _clamp(desired_pitch, tilt_limit)
            desired_roll = _clamp(desired_roll, tilt_limit)
        else:
            self._hold = None
            self._forward_integral = self._left_integral = 0.0
            desired_pitch = pitch_stick * tilt_limit
            desired_roll = roll_stick * tilt_limit

        # Vertical: hold the target altitude; the throttle stick sets climb rate.
        if self.state == "CLIMB":
            self._z_target = min(tuning.takeoff_height_m, self._z_target + tuning.takeoff_climb_mps * dt)
            command_vz = _clamp(1.5 * (self._z_target - self._z_est) + tuning.takeoff_climb_mps, tuning.max_climb_mps)
            if self._z_est >= tuning.takeoff_height_m - 0.05:
                self._set_state("FLYING", "Hovering")
        elif self.state == "LANDING":
            command_vz = -tuning.landing_descent_mps if self._z_est > 0.3 else -0.25
            self._z_target = self._z_est
        elif self.throttle_stick != 0.0:
            limit = tuning.max_climb_mps if self.throttle_stick > 0.0 else tuning.max_descent_mps
            command_vz = self.throttle_stick * limit
            self._z_held = False
        else:
            # Latch the new altitude once the climb or descent has actually stopped.
            if not self._z_held and abs(self._vz_est) < 0.15:
                self._z_target = self._z_est
                self._z_held = True
            command_vz = _clamp(1.2 * (self._z_target - self._z_est), 1.0) if self._z_held else 0.0
        error_vz = command_vz - self._vz_est
        self._z_integral = _clamp(self._z_integral + 2.0 * error_vz * dt, 5.0)
        vertical = 4.0 * error_vz + self._z_integral
        tilt_cos = max(0.5, math.cos(roll) * math.cos(pitch))
        total = spec.mass_kg * (GRAVITY + vertical) / tilt_cos
        total_limit = 4.0 * spec.max_thrust_per_motor_n
        total = 0.0 if total < 0.0 else total_limit if total > total_limit else total

        # Attitude and yaw rate, then mix to the four motors.
        torque_roll = spec.inertia_xx * _RATE_GAIN * (_ATTITUDE_GAIN * (desired_roll - roll) - drone.p)
        torque_pitch = spec.inertia_yy * _RATE_GAIN * (_ATTITUDE_GAIN * (desired_pitch - pitch) - drone.q)
        desired_yaw_rate = -self.yaw_stick * tuning.max_yaw_rate_rad_s
        torque_yaw = spec.inertia_zz * _YAW_RATE_GAIN * (desired_yaw_rate - drone.r)
        share = total / 4.0
        mix_roll = torque_roll / (4.0 * spec.arm_offset_m)
        mix_pitch = torque_pitch / (4.0 * spec.arm_offset_m)
        mix_yaw = torque_yaw / (4.0 * spec.yaw_torque_per_newton_m)
        scale = 1.0 / spec.max_thrust_per_motor_n
        return [
            (share + mix_roll - mix_pitch + mix_yaw) * scale,
            (share - mix_roll - mix_pitch - mix_yaw) * scale,
            (share - mix_roll + mix_pitch + mix_yaw) * scale,
            (share + mix_roll + mix_pitch - mix_yaw) * scale,
        ]

    # ------------------------------------------------------------------ outputs
    def leds(self) -> str:
        """measured: solid once booted, before any packets, and unchanged by the control stream."""
        if self.t < self.tuning.boot_s or self.state == "CRASHED":
            return "flashing"
        if self.t < self._calibrate_until:
            return "fast flash"
        return "slow flash" if self.low_battery else "solid"

    def telemetry_frame(self) -> bytes:
        """capture: hiturbo-rs saw 66 <battery> <flags> ... <xor>, with flag 0x04 when airborne."""
        percent = round(self.drone.battery.soc * 100)
        flags = 0x04 if self.state in ("SPINUP", "CLIMB", "FLYING", "LANDING") else 0x00
        body = bytes([percent, flags, 0, 0, 0, 0, 0, 0])
        checksum = 0
        for value in body:
            checksum ^= value
        return b"\x66" + body + bytes([checksum])
