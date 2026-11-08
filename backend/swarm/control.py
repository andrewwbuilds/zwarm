"""Turning tracker poses into a velocity estimate, and a world velocity into stick positions.

The S20's flight controller already holds position and turns the pitch and roll sticks into a
ground speed using its optical-flow sensor. So the swarm only ever asks for a velocity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    t: float  # when the frame was captured, on the coordinator's clock
    p: tuple[float, float, float]
    yaw: float  # radians, positive to the left
    sigma: float = 0.0  # how far off this may be, in metres, when the tracker knows
    v: object = None  # velocity, when the tracker worked it out itself
    fix_age: float = 0.0  # seconds since a real sighting, when poses can be carried forward between them
    seen_p: object = None  # where the last real sighting put it, free of anything assumed from commands


class Tracked:
    """Smooths one drone's poses (an alpha-beta filter) and predicts where it is now."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.12):
        self.alpha, self.beta = alpha, beta
        self.t: float | None = None
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.yaw = 0.0
        self.sigma = 0.0
        self.fix_age = 0.0
        self.seen_p = np.zeros(3)
        self.seen_log: deque = deque(maxlen=60)  # (when, height) of the latest real sightings

    def update(self, pose: Pose) -> None:
        measured = np.array(pose.p, float)
        self.sigma, self.fix_age = pose.sigma, pose.fix_age
        self.seen_p = measured if pose.seen_p is None else np.asarray(pose.seen_p, float)
        if pose.seen_p is not None or pose.fix_age == 0.0:
            when = pose.t - pose.fix_age
            if not self.seen_log or when > self.seen_log[-1][0] + 1e-6:
                self.seen_log.append((when, float(self.seen_p[2])))
        if pose.v is not None:
            self.t, self.p, self.v, self.yaw = pose.t, measured, np.asarray(pose.v, float), pose.yaw
            return
        if self.t is None or pose.t - self.t > 1.0:
            self.t, self.p, self.v, self.yaw = pose.t, measured, np.zeros(3), pose.yaw
            return
        dt = pose.t - self.t
        if dt <= 0.0:
            return
        predicted = self.p + self.v * dt
        error = measured - predicted
        self.p = predicted + self.alpha * error
        self.v = self.v + self.beta * error / dt
        self.t = pose.t
        self.yaw = pose.yaw

    def age(self, now: float) -> float:
        return math.inf if self.t is None else now - self.t

    def seen_drop(self, window: float) -> float:
        """How far the last sightings came down, over this many seconds before they stopped. Estimates don't count."""
        if not self.seen_log:
            return 0.0
        last_t, last_z = self.seen_log[-1]
        return max(z for t, z in self.seen_log if last_t - t <= window) - last_z

    def now(self, now: float) -> np.ndarray:
        """Position extrapolated to this moment, at most a short way."""
        return self.p + self.v * min(max(self.age(now), 0.0), 0.3)


@dataclass(frozen=True)
class Airframe:
    """How sticks translate to motion. The defaults are the simulator's estimates, not measurements."""

    speed_per_stick: float = 5.0  # m/s at full pitch or roll
    climb_per_stick: float = 1.5  # m/s at full throttle up
    descent_per_stick: float = 1.0  # m/s at full throttle down
    yaw_rate_per_stick: float = 2.6  # rad/s at full yaw, stick positive turns right
    deadband: float = 0.06  # smallest stick the flight controller acts on, with a little spare


class _Pulse:
    """Small requests fall inside the stick's deadband and would do nothing, so they are averaged:
    a stick just past the deadband, held for the right fraction of the time."""

    def __init__(self):
        self.owed = 0.0

    def apply(self, stick: float, deadband: float, dt: float, period: float = 0.4) -> float:
        if abs(stick) >= deadband or stick == 0.0:
            self.owed = 0.0
            return stick
        self.owed += stick * dt / period
        if abs(self.owed) >= deadband * dt / period:
            pulse = math.copysign(deadband, self.owed)
            self.owed -= pulse * dt / period
            return pulse
        return 0.0


class StickMapper:
    """World velocity to throttle, yaw, pitch and roll, in the drone's own frame.

    Airframes differ and these values are unmeasured, so the speed per stick is also learned in
    flight by comparing what was asked for with what the tracker saw.
    """

    STOPPED = 0.03  # m/s below which the sticks stay centred and the stock hold does the work
    LEARN_RATE = 0.5  # share of the way to the newest estimate per second
    DOWN_RATE_SHARE = 0.1  # lowering the gain goes this much slower than raising it
    GAIN_RANGE = (0.3, 2.5)  # how far the learned speed may stray from the airframe's value
    STEADY_S = 1.5  # a command must have held this long before its result is believed

    def __init__(self, airframe: Airframe):
        self.airframe = airframe
        self.scale = 1.0
        self.pulses = {"pitch": _Pulse(), "roll": _Pulse(), "throttle": _Pulse()}
        self.history: list[tuple[float, float, float]] = []

    @property
    def speed_per_stick(self) -> float:
        return self.airframe.speed_per_stick * self.scale

    def learn(self, now: float, forward: float, left: float, measured_forward: float, measured_left: float, dt: float) -> None:
        """Nudge the speed-per-stick estimate from how fast a steady command is actually flying."""
        self.history.append((now, forward, left))
        while len(self.history) > 2 and self.history[0][0] < now - self.STEADY_S:
            self.history.pop(0)
        if now - self.history[0][0] < self.STEADY_S * 0.9:
            return
        # Only learn from a command held steady for a while. While the command is changing, the
        # speed lags it, and treating that lag as a gain error makes the loop oscillate.
        asked = math.hypot(forward, left)
        if asked < 0.15:  # small requests are mostly pulses and noise
            return
        if any(math.hypot(f - forward, l - left) > 0.1 for _, f, l in self.history):
            return
        along = (measured_forward * forward + measured_left * left) / asked
        if along <= 0.0:
            return
        estimate = max(self.GAIN_RANGE[0], min(self.GAIN_RANGE[1], self.scale * along / asked))
        # Believing a drone is weaker makes it fly faster, and a few seconds of being slowed by
        # anything would otherwise double its real speed. Too high a gain only means a slow drone.
        rate = self.LEARN_RATE if estimate > self.scale else self.LEARN_RATE * self.DOWN_RATE_SHARE
        self.scale += (estimate - self.scale) * min(1.0, rate * dt)

    def sticks(self, velocity, yaw: float, yaw_target: float, dt: float) -> tuple[float, float, float, float]:
        frame = self.airframe
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        forward = velocity[0] * cos_yaw + velocity[1] * sin_yaw
        left = -velocity[0] * sin_yaw + velocity[1] * cos_yaw
        speed = self.speed_per_stick
        pitch = 0.0 if abs(forward) < self.STOPPED else forward / speed
        roll = 0.0 if abs(left) < self.STOPPED else -left / speed
        climb = velocity[2]
        if abs(climb) < self.STOPPED:
            throttle = 0.0
        else:
            throttle = climb / (frame.climb_per_stick if climb > 0 else frame.descent_per_stick)
        pitch = self.pulses["pitch"].apply(pitch, frame.deadband, dt)
        roll = self.pulses["roll"].apply(roll, frame.deadband, dt)
        throttle = self.pulses["throttle"].apply(throttle, frame.deadband, dt)
        turn = (yaw_target - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_stick = 0.0
        if abs(turn) > math.radians(8):
            yaw_stick = -max(-0.25, min(0.25, 0.8 * turn / frame.yaw_rate_per_stick))
            if abs(yaw_stick) < frame.deadband:
                yaw_stick = math.copysign(frame.deadband, yaw_stick)
        return (max(-1.0, min(1.0, throttle)), yaw_stick, max(-1.0, min(1.0, pitch)), max(-1.0, min(1.0, roll)))
