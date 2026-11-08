# 2. One shared memory on the ground, fed by each drone's camera

The S20 only sends video and takes sticks, so drones can't tell each other anything. A hub on the Mac holds what every camera has seen, and the coordinator plans every drone around it.

- **SQLite, not a server.** One computer in the room needs no more, and it survives a restart.
- **Three sightings before anything counts**, because detectors report things that aren't there. Doubtful things are still avoided, but only confident ones are called found, and only after a close look. Looking straight at something and not seeing it, repeatedly, removes it.
- **Floor position from the ray to a thing's base.** With the camera 45 degrees down this beats guessing distance from size. The ray is ignored when it disagrees with the size guess by more than double, which is what a cup on a table does, unless the picture cuts the box off and the size guess is the unreliable one.
- **Fixed phrases, not a guess.** `commands.py` does nothing with a sentence it doesn't know. "stop" holds position, and only an explicit phrase cuts the motors. Actions are plain hub methods, so an agent can call them without going through text.
- **The gain falls slower than it rises.** A few seconds of being slowed cut a drone's learned gain in half, and it then flew at twice the commanded speed, which caused drone collisions on the bench. Too high a gain only means a slow drone.
