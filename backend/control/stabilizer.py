"""Hold the drone still with its camera when its own position hold has nothing to work with.

The stock flight controller holds position from a downward optical-flow sensor. Over a plain floor,
in dim light or high up it loses the floor, and then a centred stick only means "stay level": the
drone keeps whatever speed it had, and trim error and moving air push it around. This is the outer
loop that fills the gap. It reads motion from the front camera (control/visual_motion.py) and adds a
small correction to the pitch and roll sticks.

Each horizontal axis, every time a measurement arrives:

    pilot is on this stick       no correction, and forget the held position
    pilot is on another stick    resist sideways motion only, and only what's clearly above the noise
    pilot is turning             no correction at all: brake and hold again once the turn ends
    nobody is flying             brake until nearly still, then hold that spot

Height is left to the flight controller's barometer, and heading to its gyro. The camera can't tell a
climb from a nose-up tilt, so it has no business correcting height.

Distances are in units of the distance to the scene, because that's all one camera can see. The same
stick correction therefore moves the drone more when it's close to a wall, which is the case the gain
backoff is there for. Measurement stops and the corrections fade out when the view stops being
trustworthy: a blank wall, a stalled stream, or motion the corrections aren't calming.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from control.visual_motion import Motion, MotionTracker

logger = logging.getLogger(__name__)

TAKEOFF_SETTLE_S = 4.0  # estimate: spin-up and climb, when the view changes too much to mean anything
TURN_COOLDOWN_S = 0.6  # the drone keeps turning briefly after the stick centres
ACTIVE_STICK = 0.06  # just above the flight controller's estimated deadband
PILOT_SOURCES = ("frontend", "trajectory", "disconnect", "unknown", "init")  # a plugin flies by its own view


@dataclass(frozen=True)
class Gains:
    hold: float = 0.35  # stick per unit of position error
    damp: float = 0.55  # stick per 1/s of speed
    integral: float = 0.08  # stick per unit of error, per second: finds trim error and steady wind
    integral_limit: float = 0.12
    limit: float = 0.30  # largest correction: about 7.5 degrees of tilt with no flow
    brake_damp: float = 0.8  # stick per 1/s while stopping after the pilot lets go
    brake_limit: float = 0.45  # about 11 degrees of tilt with no flow, briefly
    quiet_error: float = 0.02  # inside this and quiet_speed the sticks stay exactly centred: any pulse
    quiet_speed: float = 0.03  # past the deadband clears the flight controller's own position hold
    cross_damp: float = 0.6  # share of damp used while the pilot flies another axis
    settle: float = 0.05  # 1/s: slow enough to call the drone stopped
    cross_noise: float = 0.12  # 1/s: speed readings during other motion are noise up to about here
    smoothing: float = 0.45  # share of each new speed measurement kept
    stale_s: float = 0.25  # no good frame for this long: fade out
    fade_s: float = 0.10
    deadband: float = 0.06  # sticks smaller than this do nothing, so smaller corrections are pulsed
    runaway_speed: float = 1.5  # 1/s with no pilot: the corrections aren't helping
    runaway_s: float = 1.2
    max_error: float = 0.8  # further than this from the held spot, drop it and brake again
    max_yaw_rate: float = 2.6  # rad/s at full stick, for the tracker's first guess
    backoff_after: int = 4  # direction changes within backoff_window
    backoff_window_s: float = 3.0
    backoff_floor: float = 0.35
    backoff_recovery: float = 0.04  # per second


@dataclass
class Axis:
    speed: float = 0.0
    error: float = 0.0
    integral: float = 0.0
    held: bool = False
    command: float = 0.0
    residue: float = 0.0
    brake_sign: int = 0  # which way the drone was moving when the brake came on
    flips: list = field(default_factory=list)
    sign: int = 0

    def clear_hold(self) -> None:
        self.error = 0.0
        self.held = False
        self.brake_sign = 0


class Stabilizer:
    """The control law, with its clock passed in, so a simulated flight can run it faster than real time."""

    def __init__(self, gains: Gains | None = None, hfov_deg: float = 90.0, enabled: bool = True):
        self.gains = gains or Gains()
        self.tracker = MotionTracker(hfov_deg)
        self.enabled = enabled
        self.airborne = True
        self.lock = threading.RLock()
        self.forward = Axis()
        self.lateral = Axis()
        self.sticks = (0.0, 0.0, 0.0, 0.0)  # throttle, yaw, pitch, roll
        self.settle_until = 0.0
        self.last_turn = -math.inf
        self.last_good = -math.inf
        self.fast_since: float | None = None
        self.gain_scale = 1.0
        self.fault: str | None = None
        self.motion: Motion | None = None
        self.frames = 0
        self.lost = 0
        self.state = "off" if not enabled else "waiting"

    # ------------------------------------------------------------------ commands
    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.enabled = enabled
            self.fault = None
            self.gain_scale = 1.0
            self._reset()
            self.state = "waiting" if enabled else "off"

    def on_takeoff(self, t: float) -> None:
        with self.lock:
            self.airborne = True
            self.settle_until = t + TAKEOFF_SETTLE_S
            self._reset()

    def on_land(self) -> None:
        with self.lock:
            self.airborne = False
            self._reset()

    def _reset(self) -> None:
        for axis in (self.forward, self.lateral):
            axis.__init__()
        self.tracker.reset()
        self.fast_since = None

    @property
    def active(self) -> bool:
        return self.enabled and self.airborne and self.fault is None

    # --------------------------------------------------------------- measurement
    def observe(self, gray: np.ndarray, t: float) -> None:
        """Run on every video frame."""
        with self.lock:
            self.frames += 1
            if not self.active or t < self.settle_until:
                self.tracker.update(gray, t)
                self.state = "off" if not self.enabled else self.fault and "fault" or "waiting"
                return
            _, yaw, _, _ = self.sticks
            if abs(yaw) > ACTIVE_STICK:
                self.last_turn = t
            turning = t - self.last_turn < TURN_COOLDOWN_S
            motion = self.tracker.update(gray, t, yaw * self.gains.max_yaw_rate, turning)
            self.motion = motion
            if not motion.valid:
                self.lost += 1
                return
            self.last_good = t
            self._control(motion, t, turning)

    def _control(self, motion: Motion, t: float, turning: bool) -> None:
        gains = self.gains
        throttle, _, pitch, roll = self.sticks
        keep = gains.smoothing
        self.forward.speed += (motion.forward - self.forward.speed) * keep
        self.lateral.speed += (motion.lateral - self.lateral.speed) * keep
        forward, lateral = self.forward, self.lateral

        flying = (abs(pitch) > ACTIVE_STICK, abs(roll) > ACTIVE_STICK)
        elsewhere = turning or abs(throttle) > ACTIVE_STICK
        idle = not (any(flying) or elsewhere)
        stopped = abs(forward.speed) < gains.settle and abs(lateral.speed) < gains.settle
        scale = self.gain_scale

        for axis, moved, pilot, other in ((forward, motion.forward, flying[0], flying[1] or elsewhere),
                                          (lateral, motion.lateral, flying[1], flying[0] or elsewhere)):
            if pilot:
                axis.clear_hold()
                axis.command = 0.0
                continue
            if other:
                axis.clear_hold()
                if turning:
                    # The sim flew every damping share here and none was a clear win: a sideways reading
                    # looks too much like the turn, and learned trim stops pointing the right way.
                    axis.command = 0.0
                    continue
                # A tiny stick offset is real velocity to a flight controller with flow, so don't chase noise.
                slide = math.copysign(max(abs(axis.speed) - gains.cross_noise, 0.0), axis.speed)
                axis.command = -(gains.damp * gains.cross_damp * scale * slide + axis.integral)
                continue
            sign = (axis.speed > 0) - (axis.speed < 0)
            # A stop ends when the speed crosses zero. Braking on past that overshoots: to a flight
            # controller with flow, this much stick is a command to fly back at about 2 m/s.
            crossed = idle and axis.brake_sign and sign == -axis.brake_sign
            if not axis.held and idle and (stopped or crossed):
                axis.held = True
                axis.error = 0.0
                axis.brake_sign = 0
            if not axis.held:
                axis.brake_sign = axis.brake_sign or sign
                brake = gains.brake_damp * scale * axis.speed + axis.integral
                axis.command = -max(-gains.brake_limit, min(gains.brake_limit, brake))
                continue
            axis.error += moved * motion.dt
            if abs(axis.error) > gains.max_error:
                axis.clear_hold()
                continue
            if abs(axis.error) < gains.quiet_error and abs(axis.speed) < gains.quiet_speed:
                axis.command = -axis.integral
                continue
            command = gains.hold * scale * axis.error + gains.damp * scale * axis.speed
            if abs(command + axis.integral) < gains.limit:
                axis.integral += gains.integral * axis.error * motion.dt
                axis.integral = max(-gains.integral_limit, min(gains.integral_limit, axis.integral))
            axis.command = -(command + axis.integral)

        for axis in (forward, lateral):
            if axis.held:
                axis.command = max(-gains.limit, min(gains.limit, axis.command))
            self._watch_oscillation(axis, t)
        self.gain_scale = min(1.0, self.gain_scale + gains.backoff_recovery * motion.dt)
        self._watch_runaway(idle, t)
        if self.fault is None:
            self.state = "holding" if forward.held and lateral.held else "damping" if not idle else "braking"

    def _watch_oscillation(self, axis: Axis, t: float) -> None:
        gains = self.gains
        if abs(axis.command) < 0.06:
            return
        sign = 1 if axis.command > 0 else -1
        if axis.sign and sign != axis.sign:
            axis.flips.append(t)
        axis.sign = sign
        axis.flips = [at for at in axis.flips if t - at < gains.backoff_window_s]
        if len(axis.flips) >= gains.backoff_after:
            self.gain_scale = max(gains.backoff_floor, self.gain_scale * 0.7)
            axis.flips.clear()
            logger.info("[stabilizer] Oscillating: gain now %.2f", self.gain_scale)

    def _watch_runaway(self, idle: bool, t: float) -> None:
        gains = self.gains
        fast = max(abs(self.forward.speed), abs(self.lateral.speed)) > gains.runaway_speed
        if not (idle and fast):
            self.fast_since = None
        elif self.fast_since is None:
            self.fast_since = t
        elif t - self.fast_since > gains.runaway_s:
            # Wrong stick signs or a misread view would feed on themselves. Hand back to the pilot.
            self.fault = "still moving fast with no pilot input: corrections stopped"
            self.state = "fault"
            self._reset()
            logger.warning("[stabilizer] %s", self.fault)

    # -------------------------------------------------------------------- output
    def correction(self, t: float, throttle: float, yaw: float, pitch: float, roll: float) -> tuple[float, float]:
        """Run on every control packet. The current sticks go in; pitch and roll to add come out."""
        gains = self.gains
        with self.lock:
            self.sticks = (throttle, yaw, pitch, roll)
            if not self.active or t < self.settle_until:
                return 0.0, 0.0
            age = t - self.last_good
            if age > gains.stale_s:
                if age > gains.stale_s + gains.fade_s:
                    for axis in (self.forward, self.lateral):
                        axis.clear_hold()
                        axis.speed = 0.0
                        axis.command = 0.0
                    self.state = "lost"
                    return 0.0, 0.0
                fade = 1.0 - (age - gains.stale_s) / gains.fade_s
            else:
                fade = 1.0
            return (self._shape(self.forward, fade, abs(pitch) > ACTIVE_STICK),
                    self._shape(self.lateral, fade, abs(roll) > ACTIVE_STICK))

    def _shape(self, axis: Axis, fade: float, pilot: bool) -> float:
        """A pilot's stick wins at once. Corrections below the deadband become pulses of the same average."""
        if pilot:
            axis.residue = 0.0
            return 0.0
        value = axis.command * fade
        deadband = self.gains.deadband
        if abs(value) >= deadband:
            axis.residue = 0.0
            return value
        axis.residue += value
        if abs(axis.residue) >= deadband:
            pulse = math.copysign(deadband, axis.residue)
            axis.residue -= pulse
            return pulse
        return 0.0

    def status(self) -> dict:
        with self.lock:
            motion = self.motion
            return {
                "enabled": self.enabled,
                "state": self.state,
                "fault": self.fault,
                "frames": self.frames,
                "lost": self.lost,
                "gain": round(self.gain_scale, 2),
                "forward": {"speed": round(self.forward.speed, 3), "error": round(self.forward.error, 3),
                            "trim": round(self.forward.integral, 3), "stick": round(self.forward.command, 3)},
                "lateral": {"speed": round(self.lateral.speed, 3), "error": round(self.lateral.error, 3),
                            "trim": round(self.lateral.integral, 3), "stick": round(self.lateral.command, 3)},
                "inliers": motion.inliers if motion else 0,
                "residual_px": round(motion.residual, 2) if motion else 0.0,
            }


def decode_gray(jpeg: bytes, target_width: int = 300) -> np.ndarray | None:
    """A small grayscale copy of a JPEG. The reduced decode is what makes this cheap on a Raspberry Pi."""
    data = np.frombuffer(jpeg, np.uint8)
    width = 0
    # The header is enough to choose a reduction, but the first bytes vary by camera, so try the cheap one first.
    for flag, factor in ((cv2.IMREAD_REDUCED_GRAYSCALE_4, 4), (cv2.IMREAD_REDUCED_GRAYSCALE_2, 2),
                         (cv2.IMREAD_GRAYSCALE, 1)):
        gray = cv2.imdecode(data, flag)
        if gray is None:
            return None
        width = gray.shape[1]
        if width >= target_width * 0.7 or factor == 1:
            return gray
    return gray if width else None


class StabilizerService:
    """Runs a Stabilizer on the live video without ever holding up the stream or the control loop."""

    def __init__(self, stabilizer: Stabilizer, clock=time.monotonic):
        self.stabilizer = stabilizer
        self.clock = clock
        self.slot: tuple[float, bytes] | None = None
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.seconds = 0.0

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True, name="stabilizer")
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    def feed(self, frame) -> None:
        """Keep only the newest frame: an old one would steer by where the drone used to be."""
        if not self.stabilizer.enabled or getattr(frame, "format", "jpeg") != "jpeg" or not frame.data:
            return
        self.slot = (self.clock(), bytes(frame.data))
        self.wake.set()

    def run(self) -> None:
        while not self.stopping.is_set():
            self.wake.wait(0.5)
            self.wake.clear()
            slot, self.slot = self.slot, None
            if slot is None:
                continue
            arrived, jpeg = slot
            began = time.perf_counter()
            try:
                gray = decode_gray(jpeg)
                if gray is not None:
                    self.stabilizer.observe(gray, arrived)
            except Exception:
                logger.exception("[stabilizer] Frame failed")
            self.seconds += (time.perf_counter() - began - self.seconds) * 0.05

    def status(self) -> dict:
        return {**self.stabilizer.status(), "frame_ms": round(self.seconds * 1000, 1)}


def from_env() -> Stabilizer:
    return Stabilizer(hfov_deg=float(os.getenv("STABILIZER_HFOV", "90")),
                      enabled=os.getenv("STABILIZER_ENABLED", "false").lower() in ("1", "true", "yes", "on"))
