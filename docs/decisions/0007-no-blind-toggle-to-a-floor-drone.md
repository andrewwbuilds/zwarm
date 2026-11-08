# 7. A land toggle only goes to a drone that isn't on the floor

The S20's land bit is its takeoff bit. A land toggle sent to a drone on the floor starts its motors, and someone may be walking over to pick it up. The drone reports nothing, and with no tag in view the pose estimate only echoes commands: in the simulator it rose to 0.89 m while the drone lay on the floor.

## Decision

Before any land toggle, and on every tick for a flying drone, the coordinator looks for signs it has already come down. It goes by real sightings and the camera, never the estimate:

- the last sighting was under 0.4 m, with nothing asking for that;
- the sightings fell more than 0.25 m in their last 1.5 s, with nothing asking for that;
- no sighting for 2.5 s and the picture has stopped changing.

If so, the drone is marked down with no toggle, and its Pi is told, so /fleet's Land all can't start it either. A later sighting at flight height puts it back to holding. A drone that looked down once gets no more automatic toggles that flight, only the operator's.

With no sign of a fall, a drone that loses its tags is still landed blind, as before. Most tag loss is a hover over the wrong patch of floor, and it shouldn't wander until the battery goes.

## What this costs

A hovering drone whose picture barely changes can be called down. It then holds on the stock hold, unlanded, with a fault on the console, until a tag comes back or someone uses E-STOP or /fleet. That was chosen over starting motors next to a person.

`STILL_CHANGE` is an estimate. Simulator hover is about 0.9 grey levels per frame. It hasn't been measured on a drone, on the floor or in the air.

A fast fall with a moving camera, such as a drone bouncing or being carried, isn't caught until it comes to rest. If someone picks a drone up within 2.5 s of it falling with no low sighting, the camera moves, so the blind toggle could still fire later. On the S20 nothing reports whether it landed, so this can't be closed from here.

## Also

A landing that's asked for and can't go out yet stays asked for: a drone still climbing gets it once it's up, and one with its link down gets it when the link returns. A toggle that went out and didn't work is never retried by itself, but LAND can send it again, telling the Pi first that the drone is flying, or the Pi would drop it.
