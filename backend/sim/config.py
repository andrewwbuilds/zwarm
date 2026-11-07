"""Everything the simulator knows about the S20, and where each value came from.

Source tags used in the comments:
  measured  captured from the user's own drone (tests/fixtures/s20_capture.json, takeoff_probe.log)
  fitted    fitted to the user's calibration flights by sim/calibrate.py; stored in sim/calibration.json
            and applied by calibrated(), over the value written here
  capture   captured by other projects from drones in this app family
  research  reverse engineered in docs/research/S2x.md
  spec      Hiturbo's published specifications
  estimate  an engineering estimate that nothing has measured
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

CALIBRATION = Path(__file__).with_name("calibration.json")


def fitted() -> dict:
    try:
        return json.loads(CALIBRATION.read_text()).get("values", {})
    except (OSError, ValueError):
        return {}


class Calibrated:
    @classmethod
    def calibrated(cls):
        """These values with the user's calibration flights applied."""
        names = {f.name for f in fields(cls)}
        return replace(cls(), **{k: v for k, v in fitted().items() if k in names})


@dataclass(frozen=True)
class DroneSpec(Calibrated):
    # Airframe
    mass_kg: float = 0.146  # spec
    arm_offset_m: float = 0.088  # estimate: motor offset along each body axis
    prop_radius_m: float = 0.065  # estimate
    inertia_xx: float = 4.2e-4  # estimate: from mass distribution
    inertia_yy: float = 4.2e-4  # estimate
    inertia_zz: float = 9.0e-4  # estimate
    gear_height_m: float = 0.03  # estimate: centre of mass above the floor when landed
    collision_radius_m: float = 0.12  # estimate: reach of the prop tips
    camera_fov_deg: float = 90.0  # estimate: horizontal

    # Motors and aerodynamics
    max_thrust_per_motor_n: float = 0.72  # estimate: thrust to weight of about 2
    motor_lag_s: float = 0.035  # estimate: small brushed motors
    yaw_torque_per_newton_m: float = 0.016  # estimate
    rotor_drag_at_hover: float = 0.06  # estimate: N per m/s, scales with thrust
    body_drag: float = 0.0144  # estimate: N per (m/s)^2; both together give about 5 m/s at full tilt
    angular_damping: float = 2.0e-4  # estimate

    # Battery
    nominal_voltage: float = 3.7  # estimate: one Li-ion cell
    capacity_ah: float = 1.2  # spec
    resistance_ohm: float = 0.06  # estimate
    electronics_current_a: float = 0.35  # estimate: camera, WiFi, flight controller
    rated_flight_min: float = 12.0  # spec: "2 x 12 mins"
    usable_fraction: float = 0.85  # estimate: charge used before low-voltage landing


@dataclass(frozen=True)
class FlightTuning(Calibrated):
    """Flight controller behaviour. Nothing here has been measured on the real drone."""

    takeoff_height_m: float = 1.2  # estimate
    takeoff_climb_mps: float = 0.8  # estimate
    spinup_s: float = 0.6  # estimate
    landing_descent_mps: float = 0.45  # estimate
    max_tilt_rad: float = 0.436  # estimate: 25 degrees
    max_speed_mps: float = 5.0  # estimate
    max_climb_mps: float = 1.5  # estimate
    max_descent_mps: float = 1.0  # estimate
    max_yaw_rate_rad_s: float = 2.6  # estimate: 150 degrees per second
    stick_deadband: float = 0.05  # estimate
    flow_max_height_m: float = 4.0  # estimate
    rc_timeout_s: float = 0.5  # estimate
    failsafe_hover_s: float = 3.0  # estimate: research says this lives in the FC; behaviour unknown
    low_battery_warn_v: float = 3.45  # estimate
    low_battery_land_v: float = 3.25  # estimate
    min_takeoff_ocv_v: float = 3.60  # estimate: the manual lists charge first for "won't take off"
    side_crash_speed_mps: float = 2.5  # estimate
    drop_crash_speed_mps: float = 6.0  # estimate
    boot_s: float = 2.0  # estimate: gyro calibration after power-on


@dataclass(frozen=True)
class Firmware:
    """One hypothesis for how the real drone's flight controller reads commands."""

    name: str
    description: str
    fly_min_hold_s: float = 0.2  # estimate: anywhere between 12.5 ms and 1 s fits the evidence
    max_rc_rate_hz: float | None = None
    requires_handshake: bool = False
    required_byte7: int | None = None
    rc_format: str = "hy"
    unlock: str | None = None
    status_frames: bool = False  # measured: this drone sent none in 28 s
    battery_soc: float = 1.0


FIRMWARES = {
    firmware.name: firmware
    for firmware in (
        Firmware("s20", "Your S20 as measured: one takeoff packet at 80 per second starts the motors", fly_min_hold_s=0.0),
        Firmware("hold", "Takeoff must be held for a moment, as the stock app does (capture); ruled out for your S20"),
        Firmware("upstream", "What turbodrone's author reports: one packet is enough", fly_min_hold_s=0.0),
        Firmware("rate20", "Ignores control packets faster than about 30 per second", max_rc_rate_hz=30.0),
        Firmware("handshake", "Forwards controls only after the app's connect handshake", requires_handshake=True),
        Firmware("byte7", "Needs mode byte 0x02, the default in decompiled stock apps", required_byte7=0x02),
        Firmware("short", "Only understands the legacy 8-byte control packet", rc_format="short"),
        Firmware("heliway", "Only understands the 12-byte Heliway control packet", rc_format="heliway"),
        Firmware("pairing", "Needs throttle fully up then fully down first", unlock="pairing"),
        Firmware("arm_inward", "Needs both sticks held down and inward first", unlock="arm_inward"),
        Firmware("arm_outward", "Needs both sticks held down and outward first", unlock="arm_outward"),
        Firmware("remote", "Only the physical remote can arm the motors, as on the H120D (capture)", unlock="remote"),
        Firmware("low_battery", "Battery too low to take off", battery_soc=0.05),
        Firmware("telemetry", "Sends battery and flight status frames, as hiturbo-rs's drone does (capture)", status_frames=True),
    )
}
DEFAULT_FIRMWARE = "s20"  # measured: takeoff_probe.log step A
