# Track each drone from AprilTags on the floor, seen by its own camera

Status: accepted, not yet flown.

The S20 reports no position, so the swarm needs one from somewhere. The first version used fixed room cameras reading markers on the drones. There will be no room cameras and no markers on the drones, so each drone now works out its own position from its own video.

The camera is tilted 45 degrees down and 25 printed `tagStandard41h12` tags lie flat on the floor in a 5 x 5 grid, 1 m apart. Tags on the floor beat tags on walls because a tilted camera almost always has some in view, and they also give height, which nothing else measures.

## How it works

`swarm/tracking.py` reads each drone's frames through `swarm/frames.py`, finds tags with `pupil-apriltags`, and solves for the camera's pose. An estimator blends those fixes, which arrive about 150 ms late, with what the drone was commanded to do. Its uncertainty grows between fixes. The safety filter widens its margins by that uncertainty. At 0.3 m the drone stops being steered and rests on its stock position hold, waiting for a tag. At 0.6 m it lands where it is.

## Consequences

- **A drone on the floor can't see the floor.** From 5 cm up, the top of the view reaches about 17 cm ahead. So a flight starts from the pad's position in the swarm file, placed by hand. Nothing checks that placement until the first fix during the climb. `swarm check`, with each drone lifted by hand to 1 m, is how to see the tags before flying.
- **A takeoff that shows no fix can't be told from one that didn't happen.** It becomes a fault for the operator to check by eye, and the toggle is never sent again.
- **Touchdown is modelled, not seen,** because tags leave the view below about 0.2 m. If the video stops at the wrong moment, a drone can be marked landed while it is still in the air.
- **The size must be measured.** `white_square_edge_m` is the white square inside the tag's black ring, 5/9 of the tag's width, and the detector's corners sit on it. Every position and height scales with it. The swarm refuses to load without it.
- **Lens error is the main unknown.** The 90 degree field of view is an estimate, and nothing models distortion. `swarm fit-focal` fits a focal length per drone from two or more tags in view.

## Rejected

- Tags on walls: a tilted camera can't see them, and a level one would have to turn to find them.
- QR codes: on the sim camera at this video size they barely read beyond 0.5 m even at 25 cm across.
- ArUco: it reads well, but the tags already printed are `tagStandard41h12`, and OpenCV's detector doesn't have that family.
- `control/visual_motion.py` for position between fixes: it measures speed as a share of an unknown depth. With tags in view most of the time, the commanded velocity carries the short gaps.
