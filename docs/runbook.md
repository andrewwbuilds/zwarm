# Zwarm flight runbook

Nothing here has flown yet. Every number marked **sim** comes from the simulator, and real drones will do worse. Do the steps in order. Don't skip a stage because the last one went well.

Commands run from the repo root unless they start with `cd backend`.

## If something goes wrong in the air

Go down this list until the drones stop.

1. **Console at http://localhost:5173/swarm.** LAND is the normal stop. HOLD stops moving and stays up. E-STOP cuts the motors and the drones fall, so use it only if they're about to hit something or someone.
2. **Console stuck?** http://localhost:5173/fleet has Land all and E-STOP all, and talks straight to each Pi.
3. **No browser?** This cuts all three motors:
   ```sh
   for p in 8001 8002 8003; do curl -s -m 1 -X POST localhost:$p/flight/estop & done
   ```
4. **Tunnels gone?** `ssh zwarm-pi-01 'curl -s -X POST 127.0.0.1:8000/flight/estop'`, and the same for `zwarm-pi-02` and `zwarm-pi-03`.
5. **Nothing else works:** catch it or wait. With no link its sticks centre and it hovers until the battery gets low.

Ctrl-C on `fly` lands anything in the air and then quits, in up to 25 s. The console stays up through the landing, so E-STOP still works. A second Ctrl-C doesn't skip the landing, and closing the terminal lands them too.

Someone else's hand near E-STOP during every flight if you can get one. Nobody stands inside the fence.

## Setup, once

### 1. Set the camera angle

All three cameras at 45 degrees below level. Lay a phone level app against the camera's face, and aim for the same angle on each. `camera_tilt_deg` in the swarm file has to match. Step 6 reads the angle back from real video and warns if it's more than 10 degrees off.

The camera can't see above the horizon. At 1 m up it sees the floor from about 0.3 m to 3.5 m ahead (geometry, 58 degree view). Anything you want inspected has to be below the drone or low down ahead of it.

### 2. Lay the tags and tape them

Use the 25 sheets you already printed. If you've been calling them QR codes, they're AprilTags, family `tagStandard41h12`, IDs 0 to 24, and that's the only kind the tracker reads. Details are in [The tags](#the-tags) below. Don't print anything new unless a sheet is torn. Then: `cd backend && uv run --frozen --offline python -m zwarm markers --out markers --ids 7` prints a spare of tag 7.

Stand at the south edge facing north, which is +x. "North" in a voice command means this direction, wherever real north is.

```
              north (+x), far side of the grid
  x=+2    20   21   22   23   24
  x=+1    15   16   17   18   19
  x= 0    10   11   12   13   14
  x=-1     5    6    7    8    9
  x=-2     0    1    2    3    4
         y=+2 y=+1  y=0 y=-1 y=-2      y=+2 is on your left

  x=-2.5  pad 1 at y=+1.2   pad 2 at y=0   pad 3 at y=-1.2
              you stand here, south, facing north
```

- 1 m centre to centre. Use a tape, not your feet. A tag in the wrong place moves the drone by the same amount.
- The top of each tag is the side away from its printed label. Every top points north.
- Tape flat, at the paper's edges. No tape on the tag. No wrinkles, no glare from lights.
- Light the floor well. Dim light means blur, and blur cuts the read range to under 1 m (**sim**).
- The floor needs texture for the drone's own flow sensor. A plain shiny floor makes it drift. Put down a rug or strips of tape if you have one.

**Room too small?** Write the section for a smaller grid and paste it into your swarm file. Pads go 0.5 m south of the first row, so move them to match, and preflight checks the result. Coverage below is geometry only.

| Grid | Command (`cd backend`, `uv run --frozen --offline python -m swarm layout ... --white-square-m 0.0767 --out anchors.json`) | Fence it suggests | 1 tag / 2 tags in view |
|---|---|---|---|
| 5 x 5 at 1 m | defaults | x -3 to 1.7, y -2.5 to 2.5 | 100% / 76% |
| 5 x 5 at 0.8 m | `--spacing 0.8 --origin -1.6 1.6` | x -2.6 to 1.3, y -2.1 to 2.1 | 98% / 89% |
| 4 x 4 at 1 m | `--columns 4 --rows 4 --origin -1.5 1.5` | x -2.5 to 1.2, y -2 to 2 | 97% / 68% |
| 3 x 3 at 1 m | `--columns 3 --rows 3 --origin -1 1` | x -2 to 0.7, y -1.5 to 1.5 | 96% / 62% |

### 3. Write your swarm file and measure the fence

```sh
cp backend/zwarm/real.example.json real.json
```

Edit `real.json`:

- **`white_square_edge_m`:** measure the white square inside a tag's black ring. It should be about 0.0767. Every position and height scales with it. Ignore the "184mm" printed on the sheets, since they printed at 75%.
- **`fence`:** measure the room. Keep `lo` and `hi` at least 0.5 m inside every wall and 0.5 m below the ceiling. Keep x `hi` no further than the last tag row, because a drone facing north past it sees nothing. Drones stay another 0.28 m inside this box.
- **`pad`:** mark each with tape and measure from the grid. The tracker takes the drone's start position from this number, because a drone on the floor can't see the floor. Pads must be at least 0.5 m apart.
- **`yaw_deg`:** the way the nose points on the pad. 0 is north. Leave it at 0 and set every drone down nose north.
- **`obstacles`:** anything solid inside the fence, for example `{"name": "couch", "lo": [0.8, 1.3, 0], "hi": [1.4, 2.4, 0.9]}`. The drones can't avoid what isn't listed until a camera spots it. On this flight each listed thing counts as floor to ceiling, so drones go around it and never over it, and don't count a tag it hides. Furniture hides floor tags, and a drone over it loses its position (**sim**). Leave at least 0.7 m between it and the fence or anything else, or there's no way past.
- Leave `limits.max_speed` at 0.3 for now.

### 4. Charge everything

No battery report exists. A battery lasts about 6 minutes, and the flight limit lands drones at 270 s. Charge every battery, label them, and write down the flight count on each. A weak battery is the likely cause of a sudden sink.

Also charge the Mac. It has to stay on its normal Wi-Fi, because Nemotron needs the internet and the drones talk to the Pis, not the Mac.

### 5. Start the fleet

Turn on all three drones on the floor. Wait 20 s. Then, in its own terminal that stays open:

```sh
tools/pi/fleet.sh up
```

Pass: `3/3 drones ready`. If one says NOT READY, the reason is above it. `tools/pi/fleet.sh status` shows the link, backend and tunnel for each Pi. A drone that isn't broadcasting needs power-cycling. Pairing: Pi 1 and drone 1 are `HITURBO_S20-8be411`, Pi 2 and drone 2 `-d06410`, Pi 3 and drone 3 `-de9b28`. Label the drones so they don't get swapped: a swapped drone flies from the wrong pad.

### 6. Run preflight

This sends only `GET /capabilities` and `GET /mjpeg`. It can't move anything.

```sh
./fly.sh --swarm preflight real.json
```

When it says so, lift each drone in turn, motors off, to about 1 m over its pad with the nose north. It watches for up to 25 s and stops early once all three have a pose. To re-check without lifting: add `--on-pad`. For one drone: add `--drones 1`.

Pass: `RESULT: READY`, and each drone says PASS. What each line means:

| Line | Fail means | Do this |
|---|---|---|
| backend | Tunnel or Pi is down | `tools/pi/fleet.sh status`, rerun `up` |
| video | No frames, or under 4 fps | Power-cycle the drone. Wait 20 s |
| flight state | The Pi thinks it's flying | On the floor? "mark landed" on /fleet. It sends nothing to the drone |
| tags | None read | Lift to 1 m. Nose north. More light. Check the camera angle |
| pose | Over 0.3 m from the pad, or no pose believed | Tape-measure the pad and grid. Check `white_square_edge_m`, and each tag's position and direction |
| nemotron | No answer | Mac's internet. Or `--no-model` to fly on fixed phrases |

`signal` says SKIP on a Pi running older code. That's fine. See [Update the Pis](#update-the-pis).

Check the printed height against a tape measure. Over 10 cm off means the tag size or lens is wrong, so stop and do step 7.

### 7. Fit each drone's lens

The field of view is a guess until you do this. In the **sim**, an 8 degree error put positions 17 cm off while the tracker claimed far better. Hold a drone about 1 m up with several tags in view, motors off:

```sh
cd backend && uv run --frozen --offline python -m swarm fit-focal ../real.json --drone 1 --write
```

Repeat for 2 and 3, then rerun step 6. The height should now match the tape. If it says "large miss", a tag is out of place.

## Staged flight test

Each stage fits on one battery. Keep this running in a spare terminal. It prints each drone's phase, position, uncertainty and faults every second:

```sh
while sleep 1; do curl -s localhost:8010/zwarm/state | jq -c '{pair_m, pair_lowest_m, tick_max_ms: .loop.max_ms, drones: [.drones[] | {id, phase, p, sigma, fix_age, flight_s, faults}]}'; done
```

Between flights: land, power off, swap the battery, power on, set it on its pad with the nose north, wait about 35 s until `tools/pi/fleet.sh status` shows it answering, and then:

```sh
curl -s -X POST localhost:8010/reset-pads
```

That tells the tracker the drones are back on their pads. **Skipping it means the next takeoff is refused or starts from the wrong place.**

Constants load at start. To change one: land, Ctrl-C `fly`, edit, start it again. Symptoms and fixes are in the [tuning table](#tuning-table).

### Stage 0: props off, sticks and E-STOP

Take the props off drone 1. Hold it or set it down. Nobody's fingers near the motors.

1. Open http://localhost:5173/?backend=8001. Click the window: only a focused window sends sticks.
2. Press **takeoff**. Motors spin up.
3. Press W, S, A, D, the up and down arrows, and the left and right arrows. The motor sound changes each time.
4. Press **land**. Motors stop. Press **takeoff**, then Esc (E-STOP). Motors stop within a second.
5. Close that tab. In a terminal: `./fly.sh --swarm fly real.json --drones 1`. Open /swarm. Type "take off" and press Approve. Motors spin up.
6. After 8 s the console shows "takeoff was sent but it was not seen to climb". That's correct, since it can't climb. Press E-STOP. Motors stop.
7. `curl -s -X POST localhost:8010/reset-pads`. Ctrl-C `fly`.

**Pass:** motors answered every step. **If not:** no response to takeoff is usually a stale presumed state, so check /fleet. A drone that answers nothing needs power-cycling and `fleet.sh up`. Repeat steps 2 to 4 on drones 2 and 3. Put the props back on the right motors.

### Stage 1a: one drone hover, flown by hand, tracker only watching

This is the first real measurement of hover height and tag reads. No swarm code flies. In one terminal:

```sh
cd backend && uv run --frozen --offline python -m zwarm preflight ../real.json --drones 1 --watch
```

On http://localhost:5173/?backend=8001, press **takeoff** and leave the sticks alone for 30 s. Watch the terminal, then press **land**.

**Pass:** hover height 0.6 to 1.4 m by tape, tags named on most lines, position within 0.1 m of what you see, and drift under 0.5 m in 30 s.

| What you see | Do this |
|---|---|
| Drifts over 0.5 m | Its flow sensor can't read the floor. Add a rug or tape, more light |
| Hovers under 0.6 m | Few tags are readable that low. See `AIRBORNE_HEIGHT` in the tuning table |
| "no tag" most of the time | Light, lens smudge, camera angle. Save a frame with `--save-frames frames` in plain preflight and look at it |
| Position jumps around | Tag out of place, or fit the lens (step 7) |

**Write down the hover height.**

### Stage 1b: one drone hover, flown by the swarm

```sh
./fly.sh --swarm fly real.json --drones 1
```

Type "take off", press Approve. It climbs, then makes two small test moves on its own of about 0.36 m each, then returns. Leave 1.6 m of clear space around the pad. Then hold for 60 s and press LAND.

**Pass:** phase reaches `air` with no fault, `sigma` under 0.1 for the whole minute, `fix_age` mostly under 1 s, real drone within 0.3 m of the console's position, and it lands within 0.3 m of the pad.

Read `video_change` for this drone from `curl -s localhost:8010/state | jq '.anchors.drones'` in the hover and again on the floor. The first has to be well above `STILL_CHANGE` (0.5, an estimate) and the second below it. That's what tells a hover with no tags from a drone that has fallen. If they don't straddle it, change `STILL_CHANGE` before going on: see the tuning table.

| Fault | Cause and fix |
|---|---|
| "probably on the floor. No landing was sent" | It went low or fell, or its camera stopped moving, with no tag in view. A land toggle would start its motors, so none is sent, and its Pi is told it's down. **LOOK AT IT BEFORE TOUCHING IT.** On the floor: reset-pads. If it's in the air after all, a sighting puts it back to holding by itself. If it stays unseen: "mark flying" then Land all on /fleet, or E-STOP |
| "landing was sent but it's still in the air" | The one land packet was probably lost. It's holding and won't try again by itself. Press LAND: its Pi is told it's still flying first, or it would drop the land. On /fleet, the same thing by hand is "mark flying", then Land all |
| "takeoff was sent but it was not seen to climb" | It may be hovering unseen. **LOOK AT IT. The console's LAND ignores a drone it thinks is on the ground.** In the air: Land all on /fleet. On the floor with motors stopped: "mark landed" on /fleet. Then reset-pads. See the tuning table |
| "moved the wrong way" | Nose wasn't north, or `rotation_deg` is wrong on a tag. Reset the drone and rerun step 6 |
| "barely responded to the sticks" | See `speed_per_stick` in the tuning table. This is the most likely real-world mismatch: the value is an estimate |
| "kept going after being told to stop", "drifted" | Flow sensor can't read the floor. Fix the floor first |
| "position uncertain, holding", then lands | Lost its tags. Check the light and hover height. See `SIGMA_HOLD` |
| "landed blind and touchdown was never seen" | Same, and it came down where it was. **LOOK AT IT.** Anyone else flying stays 1.5 m away for 10 s, then 0.75 m (`DOWN_SETTLED_S`, `DOWN_GAP`), and carries on. reset-pads refuses until everything is down, so finish or land, then fetch it |

### Stage 2: one drone short move

With drone 1 in the air and holding, go 1 m north and back, using its real height. Replace the numbers with your pad's x and y, and your hover height from the terminal:

```sh
curl -s -X POST localhost:8010/goto -H 'content-type: application/json' -d '{"goals":{"1":[-1.5,1.2,1.0]}}'
# wait until it stops
curl -s -X POST localhost:8010/goto -H 'content-type: application/json' -d '{"goals":{"1":[-2.5,1.2,1.0]}}'
```

**Pass:** arrives within 0.2 m both ways by tape, overshoots under 0.3 m, stays level, and `sigma` stays under 0.15. **If not:** overshoot and slow-to-arrive rows in the tuning table. Do this move sideways too, along y.

### Stage 3: one drone mission

Same one drone. In the console, by text or voice:

1. "Explore the north side." Approve if it asks.
2. "Which areas haven't been explored yet?"
3. "Stop and land."

**Pass:** seen share on the map rises, the drone stays in the fence, the model's answer names real gaps, and it lands on command. **If not:** "Nemotron" and "coverage" rows below. If a mission is refused, its answer says why.

### Stage 4: two drones

`./fly.sh --swarm fly real.json --drones 1,2`. "Take off", Approve. "Map the north half." After 60 s, "Stop and land."

**Pass:** both reach `air`, `pair_lowest_m` in the terminal stays at or above 0.35 m, and both land. That's the closest any two drones came, by tracked position, since `fly` started. Contact is 0.24 m (**sim**). **If not:** `margin` row below. Any contact, go back to stage 2 on each drone.

`closest` in `/zwarm/state` isn't a distance, so don't judge by it. It's the gap between where two drones would stop, with height gaps stretched, minus the separation being kept. It's often below 0 in a good flight: it read as low as -0.5 in a **sim** flight whose drones never came within 0.44 m.

### Stage 5: three drones

`./fly.sh --swarm fly real.json`. "Take off", Approve. Wait for all three at `air`. "Map this entire area in 3D." After 90 s, "Stop mapping and focus on the north side." After 60 s, "Stop and land."

**Pass:** all three up, all moved north after the focus line, `pair_lowest_m` at or above 0.35 m, all three landed, and `loop.overruns` barely moved. **If not:** fly two.

### Stage 6: full demo

Run the [demo script](#demo-script) start to finish, on fresh batteries and at the real pace. Press Build map at the end and time it. **Do it twice.** The first one finds the problem you don't expect.

**Pass:** both runs end with all drones landed by command, at least one finding on screen, and a map with no warning. **If not:** see [When tracking isn't good enough](#when-tracking-isnt-good-enough).

## Demo script

Stage the damage before anyone arrives: dark marks or cracks in tape on low surfaces and on the floor, in well-lit spots. The camera can't see anything above its own height, so nothing on a wall. Put a box or couch inside the fence and list it in `obstacles`. Drones inspect it from the side and fly around it.

1. Say: **"Map this entire area in 3D."** Press Approve. Point at the three tiles and the shared map.
2. Say: **"Stop mapping and focus on the north side."** No approval is needed in flight.
3. Say: **"Which areas haven't been explored yet?"** It names the gaps.
4. Say: **"Inspect these things for damage."** Findings arrive with a photo, 3 to 40 s later. Confirm one on screen. Findings are "potential" and a person decides.
5. Say: **"Stop and land."** This never waits for the model.
6. Press **Build map.** It took 153 to 550 s in the **sim**, so pre-build a map from an earlier flight to show. A new map is refused while one is building.

The flight limit lands them at 270 s. Keep steps 1 to 5 under about 4 minutes.

## When tracking isn't good enough

Decide by 08:30. Don't discover this at 09:55.

**A. It works on fewer drones.** Fly those: `./fly.sh --swarm fly real.json --drones 1,2`. Say that you're flying two today. Everything else runs the same.

**B. Tracking won't hold in flight at all.** Use the simulator for the fleet, and be open about it. The console shows a SIMULATED badge the whole time, and it stays on screen.

```sh
./fly.sh --swarm
```

Run through the demo script once beforehand. Nemotron and vision are live on this path. Show the real drones separately: fly one or more by hand from /fleet and show the real video on the fleet page.

Say something like: *"The real drones are here and their video is live. Their position tracking didn't hold up in this room this morning, so the fleet on this map is our simulator running the same flight code. The planner and the vision model are live."*

Don't quote a simulator number as a drone result. Don't let anyone think the map is real if it isn't.

**C. Record what does work.** Whenever a stage goes well, film it on a phone and note the time. A real 30 s clip of one drone flying a mission beats a live failure. Say that it's a recording.

## Demo-day checklist

- [ ] Every battery charged and labelled, with a spare set
- [ ] Props checked after any crash. Bent props drift
- [ ] Tags flat, no scuffs, no tape on them. Nobody moved one since step 6
- [ ] Floor lit, no fresh glare or shadows from the venue's lights
- [ ] Pads marked. Drones labelled 1, 2, 3
- [ ] `tools/pi/fleet.sh up` running in its own terminal. Mac plugged in, on the internet
- [ ] `./fly.sh --swarm preflight real.json` says READY, run in this room this morning
- [ ] `/swarm` open. E-STOP in reach. A second person on /fleet if you have one
- [ ] Nobody inside the fence. Audience behind the south edge
- [ ] Staged damage in place, low down, and in the swarm file's `obstacles`
- [ ] A pre-built map, and a phone clip of a good flight
- [ ] Fallback B rehearsed once, and you know the sentence that goes with it
- [ ] Decided who presses E-STOP, and said it out loud

After the demo: land everything, then `tools/pi/fleet.sh down`.

## The tags

The printed sheets are AprilTags, family `tagStandard41h12`, IDs 0 to 24, about 138 mm across with a 76.7 mm white square. That's the only family `AnchorReader` reads. Actual QR codes and ArUco markers won't work, and the swarm file refuses any other `tag_family`. `python -m zwarm markers` prints this family, one tag per page at 300 dpi. Print at 100%, then measure.

**Expected read range (sim).** The real detector on rendered floor frames compressed to the S20's 34 KB JPEG, 1 m up, 45 degrees down. Frames read out of 12, by metres ahead on the floor. Tags under about 0.3 m ahead are below the bottom of the view.

| Video | 0.5 | 0.8 | 1.0 | 1.25 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|---|
| Sharp, full detail | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| 640 x 360 detail, still | 12 | 12 | 12 | 12 | 10 | 6 | 0 |
| Soft (blur sigma 2) | 12 | 12 | 12 | 10 | 0 | 0 | 0 |
| 640 x 360, 5 px blur (0.5 m/s, 1/60 s) | 12 | 12 | 12 | 11 | 12 | 1 | 0 |
| 640 x 360, 11 px blur (dim light or a turn) | 12 | 11 | 5 | 1 | 0 | 0 | 0 |

So plan on about 1.5 m ahead in good light (1.8 m in a straight line from the lens), and under 1 m in dim light or fast turns. To rerun: `cd backend && uv run python -m swarm.read_range`.

**Real QR codes for comparison (sim).** OpenCV's QR detectors on the same frames, out of 8. At 640 x 360 detail a 138 mm QR read 3 of 8 at 0.5 m and nothing further. A 250 mm one read 7 of 8 at 0.8 m and 4 of 8 at 1 m. A tag this size reads more than twice as far, which is why the tracker uses tags.

**Real frames.** The tunnels were down when this was written, so no real frame has been measured. Preflight prints the real numbers: `furthest read`, `sharpness`, `resolution`, `fps`. With a drone lifted 1 m up, `--watch` names the tags it sees as you walk it around. Add `--save-frames frames` to plain preflight to keep each drone's sharpest frame. Replace the table above with what you measure.

## Update the Pis

Optional: the only thing this adds is the signal byte in preflight. Skip it if you're short of time. Only with every drone on the floor, since restarting a backend cuts its control link:

```sh
for n in 01 02 03; do
  scp backend/web_server.py zwarm-pi-$n:zwarm/backend/
  scp backend/protocols/s2x_rc_protocol_adapter.py zwarm-pi-$n:zwarm/backend/protocols/
  ssh zwarm-pi-$n 'sudo systemctl restart zwarm-backend'
done
```

## Tuning table

Change one thing, then repeat the stage. Write down what you changed. Numbers in `real.json` need no code edit. The rest are in the code, and it's fine to leave those uncommitted tonight. If a line number has moved, search for the name.

Rows marked **safety** give up a margin. Change those last, and by a small step.

### Tag reads and position

| Symptom | Likely cause | Change | Direction |
|---|---|---|---|
| Sharp video, tags visible, "no tag" | Detector too strict for this compression | `backend/swarm/tracking.py:210` `MIN_MARGIN` | Lower, 15 to 8. Keep `MAX_HAMMING` at 1 |
| Tags read when still, lost when moving | Motion blur | `limits.max_speed` in `real.json` | Lower, 0.3 to 0.2. And add light |
| "no pose believed", or corners "miss" over 2 px | Tag out of place, or lens unfitted. Check both first | `backend/swarm/tracking.py:40` `MAX_REPROJECTION_PX` | Raise, 4 to 6 |
| Height wrong by a fixed share | White square measured wrong | `white_square_edge_m` in `real.json` | Measure again. Too high a reading means the file's size is too big |
| Height wrong by a fixed few cm | Lens height above the floor | `backend/swarm/tracking.py:37` `CAMERA_HEIGHT_M` | Set it to what you measure |
| Position right when still, off during moves | Video later than assumed | `video_latency_s` in `real.json` | Raise, 0.15 to 0.25 |
| Pose ignored after a gap, then snaps back | True sightings rejected | `backend/swarm/tracking.py:371` `GATE_M` | Raise, 0.45 to 0.7 |
| Sudden false jumps | Bad read let through | `backend/swarm/tracking.py:371` `GATE_M` | Lower, 0.45 to 0.3 |
| `sigma` climbs while the drone holds fine | Hold better than assumed | `backend/swarm/tracking.py:363` `HOLD_DRIFT` | Lower, 0.04 to 0.02 |
| Turns away from its tags, then loses them | Too few tags at that heading | Grid spacing in step 2 | Use the 0.8 m grid |

### Takeoff and the test moves

| Symptom | Likely cause | Change | Direction |
|---|---|---|---|
| "not seen to climb", but it's hovering with tags in view | Slow climb or first read | `backend/swarm/coordinator.py:35` `TAKEOFF_TIMEOUT_S` | Raise, 8 to 12 |
| Same, and it hovers low | Lower than the height it counts as flying | `backend/swarm/coordinator.py:38` `AIRBORNE_HEIGHT` | Lower, 0.4 to 0.3. Must stay above `GROUNDED_HEIGHT` |
| Next drone waits too long to go | Last one not read high enough yet | `backend/swarm/coordinator.py:56` `NEXT_TAKEOFF_HEIGHT` | Lower, 0.25 to 0.15 |
| "barely responded to the sticks" | Real speed per stick is lower than the estimate | `backend/swarm/control.py:84` `speed_per_stick` | Lower, 5.0 to 2.5, which means bigger sticks for the same speed |
| Same, still | Small sticks fall in the drone's deadband | `backend/swarm/control.py:88` `deadband` | Raise, 0.06 to 0.1 |
| Same, and it visibly moved a little | Bar set too high for this drone | `backend/swarm/coordinator.py:61` `CHECK_MIN_SPEED` | Lower, 0.08 to 0.05 |
| "kept going after being told to stop" on a good floor | Coasts longer than assumed | `backend/swarm/coordinator.py:63` `CHECK_STILL_SPEED` **safety** | Raise, 0.13 to 0.2 |
| "drifted X m after being told to stop" on a good floor | Same | `backend/swarm/coordinator.py:65` `CHECK_MAX_COAST` **safety** | Raise, 0.75 to 1.0, only with room to spare |
| "moved the wrong way" with the nose right and tags right | A gust. Or a reversed axis, if it repeats | `backend/swarm/coordinator.py:62` `CHECK_WRONG_WAY` **safety** | Raise, 0.2 to 0.3. If it repeats, check the signs at `backend/swarm/control.py:156` |

### Flying

| Symptom | Likely cause | Change | Direction |
|---|---|---|---|
| Overshoots goals, swings around them | Sticks too big | `backend/swarm/control.py:84` `speed_per_stick` | Raise, 5.0 to 7.0 |
| Same, but only when slowing down at the end | Starts braking too late for this hold | The `1.0 * horizontal` slowdown at `backend/swarm/coordinator.py:789` | Lower, 1.0 to 0.7 |
| Crawls, or never quite arrives | Goal tolerance too tight for this hold | `backend/swarm/coordinator.py:40` `ARRIVED` | Raise, 0.15 to 0.25 |
| Too slow for the demo | Set for the first flights | `limits.max_speed` in `real.json` | Raise by 0.05 at a time, at most 0.5. Blur grows with it |
| "position uncertain, holding" in the middle of the grid | Tags lost briefly. Fix the light and hover height first | `backend/swarm/coordinator.py:46` `SIGMA_HOLD` **safety** | Raise, 0.3 to 0.4 |
| Lands itself for uncertainty with room to spare | Same | `backend/swarm/coordinator.py:47` `SIGMA_LAND` **safety** | Raise, 0.6 to 0.8 |
| "no pose", lands within a blink of a video stall | Video gaps longer than allowed | `backend/swarm/coordinator.py:32` `LOST_LAND_S` **safety** | Raise, 2.5 to 4 |
| "flying at X m/s" while plainly slow | Noisy speed from the tags | `backend/swarm/coordinator.py:42` `OVERSPEED_FACTOR` **safety** | Raise, 3 to 4 |
| Goals near a wall refused | Margin bigger than the room allows | `limits.surface_margin` in `real.json` **safety** | Lower, 0.1 to 0.05. Or widen the fence |
| `closest` under 0.4 m | Tracking worse than assumed | `limits.pose_error`, `limits.margin` in `real.json` | Raise, 0.06 to 0.1 and 0.14 to 0.25 |
| Drones stall each other, nothing moves | Margin too big for this room | Same | Lower, and spread the pads |
| Climbs past its height, then comes back | Climb too fast for this hold | `limits.max_climb` in `real.json` | Lower, 0.3 to 0.2 |

### Landing

| Symptom | Likely cause | Change | Direction |
|---|---|---|---|
| "probably on the floor" while it's hovering fine | The picture barely changes in a hover, over a plain floor. Read `video_change` for each drone in `/state` under `anchors.drones`, on the floor and in a hover. Set the limit between them. Sim hover is about 0.9, and it hasn't been measured on a drone | `backend/swarm/tracking.py:359` `STILL_CHANGE` | Lower, 0.5 to 0.2 |
| Same, when it was only coming down slowly | Sightings fell further than a fall would | `backend/swarm/coordinator.py:51` `FALL_DROP` and `LOW_SIGHTING` **safety** | Raise, 0.25 to 0.4, and lower 0.4 to 0.3 |
| "touchdown was not seen" and it's down | Tags leave the view below about 0.2 m | `backend/swarm/coordinator.py:39` `GROUNDED_HEIGHT` | Raise, 0.15 to 0.3. Always check by eye before touching |
| Same, and it takes a while | Slow stock landing | `backend/swarm/tracking.py:358` `LANDING_DESCENT` and `backend/swarm/coordinator.py:36` `LAND_TIMEOUT_S` | Lower to its real speed, raise to 20 |
| "could not reach its pad in time" | Slow to get back | `backend/swarm/coordinator.py:37` `PAD_APPROACH_TIMEOUT_S` | Raise, 45 to 70 |
| Never settles over the pad to land | Tolerance too tight | The `0.25` over-pad test at `backend/swarm/coordinator.py:497` | Raise to 0.35 |

### Missions, vision and the map

| Symptom | Likely cause | Change | Direction |
|---|---|---|---|
| Lands too early, or battery sags first | Real battery differs | `flight_limit_s` in `real.json` | Match your battery, minus a minute. Weak ones sink at the end |
| Coverage too slow | Long dwell | `backend/swarm/mission.py:57` `DWELL_S` and `backend/swarm/search.py:21` `LOOKS_NEEDED` | Lower, 1.0 to 0.5, and 5 to 3 |
| Mission fails after a tracking or link gap the drones would have come back from | Gap longer than a mission waits with nobody able to work | `backend/swarm/mission.py:55` `ALL_LOST_S` | Raise, 15 to 30 |
| Inspection too slow | Too many views | `backend/swarm/mission.py:68` `MAX_THINGS` and `backend/swarm/mission.py:58` `VIEW_DWELL_S` | Lower, 8 to 4, and 4 to 2.5 |
| No findings, and frames marked blurry | Real video softer than the guess | `backend/swarm/inspect.py:50` `MIN_SHARPNESS` | Lower, using the sharpness preflight printed |
| False alarms | Model reading compression as damage | `backend/swarm/inspect.py:43` `LOW_CONFIDENCE` | Raise to 0.7, so more are marked long shots |
| Vision answers slowly | Slow model | `NEMOTRON_VISION_MODEL` in `backend/.env` | Name the 3 s model first |
| Planner slow or hanging | Network or model | `backend/agent/nemotron.py:40` `TIMEOUT_S`, or `NEMOTRON_MODEL` in `backend/.env` | Lower to 10 s, or name another model. Stop buttons never wait for it |
| Same object listed many times | Position noise | `backend/swarm/memory.py:24` `MERGE_GATE` | Raise, 0.7 to 1.0 |
| False objects blocking paths | Detector too keen | `backend/swarm/yolo_eyes.py:30` `CONFIDENCE` and `backend/swarm/memory.py:22` `KNOWN_SIGHTINGS` | Raise, 0.45 to 0.6, and 3 to 5 |
| Real objects missed | Detector too strict, or too small in frame | `backend/swarm/yolo_eyes.py:30` `CONFIDENCE` and `backend/swarm/yolo_eyes.py:31` `IMAGE_SIZE` | Lower to 0.35, and raise to 640 |
| Map build too slow | Too many photos | `backend/mapping/fleet.py:47` `MAX_PHOTOS` | Lower, 240 to 120 |
| Map warning: photos don't agree with the tracker | Drones too far apart, or blur | Fly with more overlap and slower. `backend/mapping/fleet.py:57` `MIN_LINK_MATCHES` | Raise, 30 to 50, if drones with no shared view got joined |
| `loop` in `/zwarm/state`: `max_ms` over 50, overruns climbing | Too much on this Mac | `backend/swarm/hub.py:32` `LOOK_EVERY` and `backend/swarm/yolo_eyes.py:31` `IMAGE_SIZE` | Raise to 0.4, and lower to 320. Close other apps |
