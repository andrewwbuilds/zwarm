# 1. Correct drift from the video, off by default

Date: 2026-11-08. Status: accepted, not yet flown on a real drone.

## Context

Drones were reported drifting when they turn or move, and getting hit as a result. The firmware
is closed, so the stock flight controller can't be retuned. The simulator holds within about
0.1 m while its optical-flow sensor can see the floor, and drifts 1.3 to 8 m in 15 s when it can't.
That points at flow loss, but the simulator's flight controller is an estimate, so this is a
hypothesis about the real drones and not a measurement of them.

## Decision

Add an outer loop on the host, `control/stabilizer.py`. It measures motion from the front camera
(`control/visual_motion.py`) and adds a correction to the pitch and roll sticks. It needs no drone
changes and works on any drone whose video arrives as JPEG.

Choices the benchmark (`python -m sim.stability`) settled:

- **Centred sticks when nothing is moving.** A pulse past the deadband clears the flight
  controller's own position hold, which is far better than this loop when it has flow.
- **No corrections during a turn.** Damping mid-turn was swept at several strengths, and each one
  traded a better median for regressions elsewhere. Sideways speed is hard to tell from the turn
  itself, and trim learned before the turn stops pointing the right way.
- **Stronger braking than holding.** Stopping after the pilot lets go gets its own, higher limit.
- **Corrections are added to the packet, not the model.** The keyboard's stick ramp starts each
  update from the model's last value and would absorb them.

SLAM (`backend/slam`) was rejected as the sensor. It can't initialise while hovering, because a
hover shows no parallax, and it costs far more per frame than tracking does.

## Safety

Off unless `STABILIZER_ENABLED=true` or the console button turns it on, because no real flight has
tested it. When on, the pilot's stick wins on its axis at once. Corrections are capped, and they fade
out within 0.35 s of the last good frame. Gain drops if corrections oscillate. It stops itself if
the drone keeps moving fast with no pilot input. An exception in it turns it off and never costs a
control packet. Plugins and the swarm coordinator fly uncorrected.

## Limits

- Height and heading are left to the flight controller.
- In-place turns with no flow improve little. See the README for the numbers.
- It can't see within about 0.6 m of a plain surface, or a blank view at any range. It goes
  hands-off there rather than correcting, so it doesn't stop a drone that's already close to a wall.
- Speed is known only relative to the distance to the scene, so correction strength varies with it.
- Motion blur, rolling shutter, real video latency and the real stick deadband aren't modelled.
