# Flying the swarm from floor tags

Nothing here has flown yet. Every number in this repo about floor-tag tracking comes from the simulator, so treat the first flight as a test of the method. Work through these steps in order, and don't skip ahead if one fails.

## Before you start

The Pi backends must be sending video. Tracking is the video: with no frames, no drone knows where it is. When last checked, `/mjpeg` on ports 8001-8003 gave no frame in 4 s. Fix that first.

On the Mac:

```sh
cd backend && uv sync --inexact
```

## 1. Measure a tag

Measure the white square inside the black ring on a printed tag. It should read about 3 in (76.7 mm). The whole tag is about 138 mm across. Ignore the 184 mm printed on the sheet, because the sheets printed at 75%.

Every position and height scales with this number, and the swarm refuses to fly without it.

## 2. Lay out the tags

Stand at the south edge, facing north (+x). Lay the grid out 1 m apart, centre to centre:

- ID 0 is the near-left corner, at (-2, 2). IDs rise to your right along each row.
- The next row (ID 5 onward) is 1 m further away. ID 24 is the far-right corner, at (2, -2).
- The top of each tag is the side opposite its printed label, pointing away from you.

Tape them flat, and keep the spacing accurate. A tag in the wrong place pulls the drone by the same amount.

Then generate the section and copy it into your swarm file:

```sh
uv run python -m swarm layout --white-square-m 0.0767 --out anchors.json
```

Start from `swarm/example.json`. Each drone needs its `backend`, `pad` and `camera_tilt_deg`. For a different layout, see `--origin`, `--spacing`, `--across-deg` and `--start-id`.

## 3. Pads and fence

Put pads behind the first row, as in the example: (-2.5, 1.2), (-2.5, 0) and (-2.5, -1.2), each nose toward +x. A drone on the floor can't see any tags, so **its starting position is wherever the file says its pad is**. Measure the pads and set the drones down carefully. Nothing checks this until the first sighting in the climb.

Keep the fence inside the tag area: a drone facing +x past the last row sees nothing. List anything solid under `obstacles`. The swarm only avoids what's in that list and what the other drones report, and it can't see a wall the file doesn't mention.

## 4. Check each drone without flying

Motors off. Lift each drone by hand to about 1 m over its pad, nose along its heading, while this runs:

```sh
uv run python -m swarm check my-swarm.json
```

For each drone it prints the link, the frames read per second, which tags it should see, which it did see, and its position from the tags. Compare that position with a tape measure. **If it's off by more than about 10 cm, stop:** the tag size, layout or lens is wrong.

## 5. Fit each drone's lens

Each drone's field of view is a guess until this is done. In simulation an 8 degree error put positions 17 cm off, while the tracker claimed to be far more certain than that. Hold a drone about 1 m up with several tags in view:

```sh
uv run python -m swarm fit-focal my-swarm.json --drone 1 --write
```

Repeat for each drone, then run `check` again, because positions should now be closer to the tape measure.

This fits only a focal length. A strongly distorted lens will still be wrong, and shows up as a large "miss" in the output, particularly near the edges of the picture.

## 6. Fly one drone

Use a swarm file listing **one drone**, and keep the speed limit at the example's 0.3 m/s.

```sh
uv run python -m swarm fly one-drone.json
```

Then open the console and, with someone ready on E-STOP, go through these in order:

1. **Take off.** After climbing, it makes two small test moves on its own, about 0.4 m each. It lands itself if it moves the wrong way, doesn't respond, or won't stop. If it does, read the fault before trying again.
2. **Hold** for a minute. Watch the position on the console against the real drone. The uncertainty should stay under about 0.1 m.
3. **Move** 1 m and back.
4. **Land.**

## 7. Add drones

Go to two drones, in separate parts of the grid, and then have them pass each other. Only then fly three. The closest the filter aims to let them get is 0.5 m.

## What it does when things go wrong

| What happens | What it does |
|---|---|
| No tag for a moment | Carries on from the commanded motion, with wider margins around it |
| Uncertainty reaches 0.3 m | Stops steering that drone, which rests on its stock hold until it sees a tag |
| Uncertainty reaches 0.6 m | Lands that drone where it is |
| Takeoff sent, no tag seen after | Reports a fault and never resends, because the toggle would reverse it. **It may be hovering: look at it** |
| Link lost, or the coordinator hangs | The backend centres the sticks within 0.3 s |
| Ctrl-C | Lands anything in the air, then quits |

## Known risks

- **Everything real is unmeasured:** lens, sharpness, exposure, video delay, stick response, hover height, and how well the stock hold copes with the tilted camera.
- **Dim light hurts.** Blur shortens the read range. Light the floor well.
- **Touchdown is modelled.** Tags leave the view below about 0.2 m, so "landed" on the console is an estimate. Confirm by eye before touching a drone.
- **Floors the flow sensor can't read.** In simulation on the old tracker, the post-takeoff check caught most of these but not all, and 2 of 40 flights still collided. Don't fly over a plain, shiny floor.
- **Batteries aren't monitored.** Nothing reports charge to the swarm.
- **E-STOP drops the drones.** Use Land unless something is about to go wrong.
