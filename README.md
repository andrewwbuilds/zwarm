# Turbodrone
Reverse-engineered API and client for controlling some of the best-selling ~$50 "toy" drones on Amazon from a computer replacing the closed-source mobile apps they come with.

![S20 Drone Short Clip](docs/images/s20-drone-short-clip-small.gif)

## Introduction
Nowadays, there are incredibly cheap "toy" drones available on Amazon that are basically pared-down clones of the early DJI mavic. Only ~$50 to have a 1080p camera for FPV and recording, tiny downward-facing optical flow sensor for position and altitude hold, and a well-tuned flight profile out-of-the-box. The only problem with drones like this is that they run closed-source firmware and are locked to only being controlled by a custom mobile app. I thought it would be cool to free some of these drones from their "jail" and write an API and client for accessing the video feed and sending control commands down to the drone by reverse-engineering how the mobile apps work. That way you can turn a highly capable $50 "toy" drone into something that can be programmatically controlled and used for all sorts of applications and experiments.

## Hardware
* WiFi Camera Drones (ranked in order of recommendation):

    | Brand      | Model Number(s) | Compatibility | Implementation | Motor      | Purchase Link                                               | Notes |
    |------------|-----------------|---------------|----------------|------------|-------------------------------------------------------------|-------|
    | Generic        | M10           | Tested    | `cooingdv` | Brushless | [Aliexpress](https://www.aliexpress.us/item/3256809708895605.html) | My current favorite. RTSP video stream is very reliable. |
    | Plegble        | PL-515        | Tested | `s2x`      | Brushless | [Amazon](https://www.amazon.com/dp/B0D2B25YQQ) | Same airframe as the S2x models and takes the same batteries, but upgraded to brushless motors. |
    | Karuisrc       | K417          | Tested    | `wifi_uav` | Brushless | [Amazon](https://www.amazon.com/Electric-Adjustable-AIdrones-Quadcopter-Beginners/dp/B0CYPSJ34H/) | First model to be supported with brushless motors. Build quality is great. |
    | Loiley         | S29           | Tested    | `s2x`      | Brushed   | Can't find link anymore | Great build quality, has servo for tilting camera (_not implemented in API yet_) |
    | Hiturbo        | S20           | Tested    | `s2x`      | Brushed   | [Amazon](https://www.amazon.com/dp/B0BBVZ849G), [Alternate Amazon Listing](https://www.amazon.com/Beginners-Foldable-Quadcopter-Gestures-Batteries/dp/B0D8LK1KJ3) | Original test platform, great build quality |
    | Loiley         | X69           | Tested  | `x69_lg`   | Brushed   | [Amazon](https://www.amazon.com/Altitude-Adjustment-Headless-Batteries-Beginner/dp/B0F8QTBGNP) | New LG/X69 app family; RC, camera tilt, and experimental FFmpeg-backed H.265 video implemented. |
    | FlyVista       | V88           | Tested    | `wifi_uav` | Brushed   | [Amazon](https://www.amazon.com/dp/B0D5CXY6X8) | |
    | ?              | D16/GT3/V66   | Tested    | `wifi_uav` | Brushed   | cheapest on [Aliexpress](https://www.aliexpress.us/item/3256808590663347.html), [Amazon](https://www.amazon.com/AUHIFVAX-Intelligent-Avoidance-Christmas-Thanksgiving/dp/B0FJRVH76T) | 20% smaller DJI Neo clone. Only really good for indoor flight. |
    | Several Brands | E58           | Tested    | Unknown    | Brushed   | [Amazon](https://www.amazon.com/Foldable-Quadcopter-Beginners-Batteries-Waypoints/dp/B09KV8L7WN/) | |
    | Several Brands | E88           | Suspected | `cooingdv` | Brushed   | [Amazon](https://www.amazon.com/Beginners-Foldable-Quadcopter-Real-Time-Rechargable/dp/B0FKNH6Q4T) | |
    | Several Brands | E88 Pro       | Suspected | Unknown    | Brushed   | [Amazon](https://www.amazon.com/Beginners-Foldable-Quadcopter-Real-Time-Rechargable/dp/B0FKNH6Q4T) | |
    | Several Brands | E99           | Suspected | `cooingdv` | Brushed   | [Amazon](https://www.amazon.com/LJN53-Foldable-Drone-Dual-Cameras/dp/B0DRH9C6RF) | |
    | Several Brands | E99 Pro       | Suspected | Unknown    | Brushed   | [Amazon](https://www.amazon.com/LJN53-Foldable-Drone-Dual-Cameras/dp/B0DRH9C6RF) | |
    | Swifsen        | A35           | Suspected | Unknown    | Brushed   | [Amazon](https://a.co/d/bqKvloz) | Very small "toy" drone |
    | Unknown        | LSRC-S1S      | Tested | `wifi_uav` | Brushed   | | mentioned in another reverse-engineering effort for the WiFi UAV app |
    | Velcase        | S101          | TODO      | Unknown      | Brushed   | [Amazon](https://www.amazon.com/Foldable-Beginners-Quadcopter-Carrying-Positioning/dp/B0CH341G5F/) | lower quality build, smaller battery and props than S29 & S20 |
    | Redrie         | X29           | Tested      | `s2x`    | Brushed   | [Amazon](https://www.amazon.com/Adults-1080P-Foldable-Altitude-Auto-Follow-Batteries/dp/B0CZQKNYL5) | Average build quality |

    _**Tested** means the drone has been physically run with turbodrone to ensure its compatibility._

  _**Suspected** means the APK for the drone appears to use the exact same packages and libraries as one of the tested drones._

  _**Untested** means the APK has been reverse engineered and implementation code exists, but the drone has not yet been physically flown with TurboDrone._

  _**TODO** means the APK operates with different byte packets and protocols and will have to be added as a new implementation in the API._

* WiFi Dongle ([recommend ALFA Network AWUS036ACM](https://www.amazon.com/Network-AWUS036ACM-Long-Range-Wide-Coverage-High-Sensitivity/dp/B08BJS8FXD) or similar) 
  * Drone broadcasts its own WiFi network so your computer will have to connect to it.
  * Not strictly necessary because you can use your computer's built-in WiFi radio to connect to the drone's network, but nice to have that way you can stay connected to the internet while flying the drone.


## Raspberry Pi gateways

For one Raspberry Pi per drone, see the [Pi installation and operation guide](tools/pi/README.md). It includes the backend service installer, a loopback simulator check, and the USB setup used for the three-Pi installation.

## Quick start: Hiturbo S20 on a Mac
You need [uv](https://docs.astral.sh/uv/) and Node.js 20+. Install once while you have internet, because the drone's Wi-Fi has none:
```
(cd backend && uv sync)
(cd frontend && npm ci)
```

Create `backend/.env` with your drone's Wi-Fi name:
```
DRONE_TYPE=s2x
DRONE_WIFI=HITURBO_S20-xxxxxx
```

Put the drone on the ground, power it on, and run:
```
./fly.sh
```

This joins the drone's Wi-Fi, runs a no-fly link check, starts the backend and web client, and opens the browser. Ctrl-C stops everything and returns the Mac to its normal Wi-Fi. If you leave out `DRONE_WIFI`, join the network yourself first.

| Key | Action |
|-----|--------|
| W / S | forward / back |
| A / D | left / right |
| Up / Down | climb / descend |
| Left / Right | turn |
| Esc | E-STOP |

Before you fly:

* Takeoff and Land send the same toggle, so Land on the ground takes off and Takeoff in the air lands. Press once and wait.
* E-STOP cuts the motors at once, and the drone drops.
* Land before pressing Ctrl-C. Quitting or losing Wi-Fi cuts the control link, and what happens next is up to the drone's flight controller.
* Only the focused browser window flies. Switching away centers the sticks.

To check the link without starting the motors, run `uv run python preflight.py` in `backend`.

The person-following plugin needs the heavy YOLO dependencies, which are optional: run `uv sync --extra follow` in `backend`, then set `PLUGINS_ENABLED=true`.


## Drift stabilizer
Over a plain floor, in dim light or high up, the drone's optical-flow sensor loses the floor, and a centred stick then only means "stay level". The stabilizer reads motion from the front camera and adds small pitch and roll corrections: it brakes when you let go of the sticks, then holds that spot. It's off by default. Turn it on with the `stabilizer` button in the console, or set `STABILIZER_ENABLED=true` in `backend/.env`. If your camera's field of view isn't 90 degrees, set `STABILIZER_HFOV`.

**It has not flown on a real drone yet.** Try it low, in open space, with a hand near E-stop. Your sticks always win on their axis. It doesn't touch height or heading, and it goes hands-off on a blank view, within about 0.6 m of a plain surface, and if the video stalls.

Measured in the simulator only (`cd backend && uv run python -m sim.stability`, 384 flight pairs). The simulated flight controller is an estimate, so these numbers don't predict a real drone's improvement.

| Flight | Median drift, off → on | Worst, off → on | Target |
|---|---|---|---|
| Hover 20 s, no flow | 2.78 → 0.44 m | 11.05 → 1.99 m | met (70% less) |
| Stop after moving, no flow | 3.06 → 1.45 m | 8.38 → 4.15 m | met (50% less) |
| Hold near a wall, no flow | 1.49 → 0.07 m | 3.39 → 1.92 m | met (50% less) |
| Turn right after takeoff, no flow | 0.57 → 0.39 m | 3.64 → 3.06 m | **missed** (50% less) |
| Turn after 5 s of hover, no flow | 0.56 → 0.74 m | 4.28 → 2.58 m | **missed: median got worse** |
| All flights, flow working | 0.10 → 0.22 m | 1.67 → 1.59 m | **missed: 13 of 192 over 0.25 m worse** |

Bumps and crashes across all 384 flights fell from 19 and 1 to 3 and 0. So leave it off if your drone already holds well, and don't expect it to fix drift during an in-place turn. To check whether flow loss is your cause, fly over a busy patterned rug: if the drift mostly goes away, it is. See [the decision record](docs/decisions/0001-visual-stabilizer.md).

## Localization (visual SLAM)
The S20 sends no position data, so the only source is its camera. `backend/slam` tracks the video and builds a map: it follows corners between frames, finds the camera pose from the 3D points it has triangulated, refines recent keyframes (bundle adjustment), and closes loops when the drone returns to a place it has seen.

With `./fly.sh` running, in another terminal:
```
cd backend && uv run python -m slam live
```
It shows the video beside a top-down path (q quits, r restarts). Positions are in map units until you set a scale: measure a distance you flew and pass `--scale <metres per map unit>`. If your camera's field of view isn't 90 degrees, pass `--fov`.

Test it without the drone: `uv run python -m slam sim --flight circle --save map.png` flies the simulator and scores the path against the exact truth.

What to expect, measured in the simulator: smooth flights drift about 1% of the path. Turning on the spot loses scale (about 4% in the square test), and the camera alone cannot fix that, because a turn shows no depth. It needs a textured scene: blank walls and floors stop it tracking, and it will say so by showing "lost".

## 3D mapping (photogrammetry)
Turns the drone's camera into a 3D map of a room, three ways from the same photos: a **point cloud** (COLMAP's own 3D points, fast and accurate), a **Gaussian splat** (Brush, realistic and quick to view), and a **NeRF** (nerfstudio, slow). No localization is needed: COLMAP works out where each photo was taken.

One-time setup: `brew install colmap`; unpack [Brush](https://github.com/ArthurBrussee/brush/releases) (`brush-app-aarch64-apple-darwin`) into `~/.local/share/brush-splat/v0.3.0/`; and run `tools/setup_nerf.sh` for the NeRF.

With `./fly.sh` running, in `backend` in another terminal:
```
uv run python -m mapping capture living-room      # records EVERY frame; fly slowly, circle each area; q to finish
uv run python -m mapping select living-room       # optional: choose photos now and look at them (images/)
uv run python -m mapping build living-room        # chooses photos, then makes all three; add --only cloud,splat to skip the NeRF
uv run python -m mapping view living-room cloud   # or splat, or nerf
```
Photos and results go in `~/Developer/datasets/room-maps/<name>/`. Try it without the drone: `uv run python -m mapping demo`.

**How photos are chosen.** Capture keeps every frame untouched (about 17 a second, around 35 KB each), in `raw/`. Choosing happens afterwards, with the whole flight in view: frames are split into short groups and the sharpest of each group is kept, so a blurry moment costs one photo, not a gap. Blur is judged against neighboring frames, so a uniformly soft camera still gives photos, and similar-looking frames are all kept: that overlap is how COLMAP links photos. Only frames that are truly identical (the camera repeating itself, or a held-still hover) are dropped. `--window 2` keeps more photos, `--window 4` fewer. You can rerun `select` or `build` with different settings any time, without flying again.

Timing on an M2 Max: the point cloud is seconds, the splat about 3 minutes, and the NeRF about 5 minutes per 1,000 steps (default 5,000), plus a short fly-through video. The NeRF is the weakest of the three on a toy camera's frames, since it needs many sharp views to resolve detail.

Tips: fly slowly and smoothly, keep textured surfaces in view (blank walls cannot be mapped), and look at each area from several angles. The build step reports how many photos it could place and warns about gaps.

## Swarm flight (simulator tested, not yet flown)
`backend/swarm/` flies several drones at once, keeps them apart and steers them around mapped obstacles. A coordinator on the Mac plans each drone's path, then a safety filter adjusts every drone's velocity together. Each drone works out its own position by reading the 25 printed `tagStandard41h12` tags on the floor through its own camera, tilted 45 degrees down.

Follow [the first-flight guide](docs/swarm-first-flight.md) for real drones. To try it without them:
```
cd backend
uv run python -m swarm sim --inside-out     # then open /swarm in the frontend
uv run python -m swarm floor-bench          # the fixed test set, real tag detector on simulated frames
uv run python -m swarm bench                # avoidance alone, with a simulated tracker
uv run python -m swarm.read_range           # how far a tag reads
SWARM_E2E=1 uv run python -m unittest tests.test_swarm_floor_e2e   # through real backends
```

Limits: it only avoids obstacles listed in the swarm file or reported by the fleet. Lens, sharpness, exposure and stick response on the real drone are all unmeasured, so simulated accuracy may not carry over.

### One map from every drone (simulator tested)
`backend/mapping/fleet.py` saves every drone's video, chooses up to 240 photos across them, and places them all with COLMAP. A map mission starts the capture and the build by itself. The build runs in the background at low priority and never touches the flight loop.

```
cd backend
uv run python -m unittest tests.test_mapping_fleet     # no COLMAP or network needed
uv run python -m mapping.fleet_bench --seconds 180     # three simulated flights, checked against the simulator's truth
uv run python -m mapping.fleet_bench --live            # real time: SimFrames, capture, build, points
```

Pass `pose_of=tracked_pose_of(hub)` to `FleetCapture`. Without it the map is only levelled, with a rough scale from assuming a 1 m flight height, and nothing can tell whether its shape is right. With it, the build checks the map against the floor-tag tracker and puts it in room coordinates.

Results on the simulator, from three drones flying 3 minutes each (9504 frames, 240 photos). Frames match the S20's size and JPEG bytes, but a render has no rolling shutter or real lens, so **sim results are not drone results**:

| Flight | Placed | Cameras off by (median) | Result |
|---|---|---|---|
| Normal mission, floor that never repeats | 234 | 1 mm | one map, in the room |
| Same, different rough lens per drone, with blur and distortion | 231 | 3 mm | one map |
| Dull floor with tags and furniture only | 223 | 3 mm, about 9% of photos over 0.5 m | one map, in the room |
| Drones looking away from each other | 228 | each map fits the tracker to 0.08, 0.26 and 0.08 m | 3 maps joined by tracked position, with a warning |
| Floor and walls that repeat exactly | 228 | 1.4 m | **folded**, flagged as untrustworthy, not fixed |

A 3-minute flight builds in about 2 minutes on an M5 Pro (114 s with other jobs running). A Gaussian splat is optional (`MAP_SPLAT=1`) and takes another 100 s for 5000 steps. It stays in COLMAP's own axes rather than room coordinates.

Environment:
- `MAP_BUILD_PHOTOS`: the photo cap, default 240. Exhaustive matching grows with the square of this: 360 photos took 466 s under heavy load.
- `MAP_MIN_LINK`: matches needed to link two photos, default 30. Lower it if real flights split into too many maps.
- `MAP_SPLAT=1`: also train a splat after the cloud is ready.
- `MAP_BUILD_SSH=user@host`: run COLMAP on another machine, such as a Brev GPU instance. It needs COLMAP 4 there, because older versions name their flags differently. A failure falls back to building here. Tested through stand-in `ssh` and `rsync` scripts and a real failed connection, but never against a real host.

For real flights, keep texture on the floor and let the drones turn. A drone flying straight at one heading can't pin down its own lens. See [the decision record](docs/decisions/0006-fleet-map.md).

## Shared memory, search and commands (simulator tested)
`backend/swarm/hub.py` gives the fleet one shared memory. Drones can't talk to each other, so whatever one camera sees goes into a SQLite database on the Mac, and the coordinator flies every drone around it. Say `find the backpack` and the drones share out the floor, confirm any find with a close look, and report back.

Try it without drones, from `backend`:
```
uv run python -m swarm.talk --speed 4         # type "take off", then "find the backpack"
uv run python -m swarm.hub_bench              # the fixed test set
uv run python -m unittest tests.test_swarm_hub tests.test_swarm_yolo_eyes
```
Real cameras use `swarm/yolo_eyes.py`, which needs `uv sync --inexact --extra follow` and reads pictures through `swarm/frames.py`. A camera that isn't 45 degrees down sets `camera_tilt_deg` on its drone in the swarm file, read with `eyes.camera_tilts`.

Limits: YOLO only recognises the labels in `eyes.LABELS`, so a plain box or a wall is invisible to it. It hasn't been measured on real footage from a tilted camera. In simulation, a person known only from the cameras can still reach a drone (30 of 40 flights clean), so keep people out of the flying space. Only `talk` and the bench run the hub today: `swarm/server.py` still calls the coordinator directly.

## Missions (simulator tested)
`backend/swarm/mission.py` holds what the swarm is trying to do and who does which part: `start(kind, region, target)`, `focus(region)`, `stop()`, `status()` and `gaps()`. Explore shares out a region's unseen floor, map adds photo capture and starts the build when it ends, inspect visits viewpoints around things and floor, and search hands over to the hub. A drone that drops out frees its work for the others, and each drone heads back in time to land by its flight limit. Hold, land and E-stop through the hub end a mission, with or without the model. Why it works this way: `docs/decisions/0005-mission-manager.md`.

From `backend`:
```
uv run python -m swarm.mission_bench            # the fixed test set, about a minute
uv run python -m swarm.mission_bench --floor    # plus flights on simulated floor tags, a few minutes each
uv run python -m unittest tests.test_mission
```
Limits: nothing here has flown. Coverage means the floor was in the picture for about a second, worked out from pose, not that a map of it came out well. The flight limit counts one flight, so a second flight on the same battery only gets a warning. Regions come from the fence, so north is +x unless `north_yaw_deg` says otherwise.

## Damage inspection (simulator tested)
`backend/swarm/inspect.py` watches each flying drone's video during an inspect mission, sends sharp new views to a vision model on worker threads, and stores what it reports under `~/Developer/datasets/zwarm/` with a photo. Every finding stays pending until a person confirms or dismisses it. Why it works this way: `docs/decisions/0003-damage-inspection.md`.

From `backend`, with `NVIDIA_API_KEY` in `backend/.env` for the last two:
```
uv run python -m unittest tests.test_inspect    # no network
uv run python -m swarm.inspect_eval             # labelled test photos through the live model
uv run python -m swarm.inspect_eval --demo      # the whole inspector over the simulated demo room
```
Limits: all scores so far are on simulated images (85% on the 20 test photos with Nano Omni, 2 false alarms and 1 miss), and the prompt was tuned on them. Real video is unscored. Sharpness limits are guesses until they are measured on real frames: `MIN_SHARPNESS` and `BLUR_SHARE` in `swarm/inspect.py`.

## Operator console
`frontend/src/pages/SwarmPage.tsx`, at `/swarm`, is one screen for running a mission: command box with voice, the conversation with Nemotron's tool calls, plan approval, a live room map, each drone's video and flight time, findings to review and a 3D point cloud. HOLD, LAND and E-STOP call the fixed endpoints directly and never go through the chat.

```
cd frontend && npm run dev
# http://localhost:5173/swarm                 the backend on http://localhost:8010
# http://localhost:5173/swarm?api=8011        another port, or ?api=http://host:port
# http://localhost:5173/swarm?mock=1          a pretend backend, nothing real or simulated
```
Voice uses the browser's speech recognition. Chrome and Edge need internet for it.

## Setup
Move to the `backend` directory
```
cd backend
```

Add venv
```
python -m venv venv
source venv/bin/activate
```

Install the dependencies
```
pip install -r requirements.txt
```

_If_ you are on Windows, you will need to manually install the `curses` library.
```
pip install windows-curses
```

Open a new terminal window and install the dependencies for the frontend.
_Make sure you have Node.js 20+ installed._
```
cd frontend
npm install
```

Make sure WiFi Dongle is plugged in, drone is turned on, connect to the "BRAND-MODEL-XXXXXX" network before proceeding.

Create a `.env` file in the `backend` directory. Add a DRONE_TYPE based on which drone you have:
```
# For "com.vison.macrochip" (s2x) based drones like S20 and S29:
DRONE_TYPE=s2x
# For WiFi UAV-based drones like V88 and D16:
# DRONE_TYPE=wifi_uav
# For cooingdv publisher drone apps like RC UFO and KY UFO:
# DRONE_TYPE=cooingdv
# For KY FPV / Jieli CTP control with first-pass RTP/JPEG video:
# DRONE_TYPE=cooingdv_jieli
# For WiFi_CAM native UDP apps:
# DRONE_TYPE=wifi_cam
# For Loiley X69 / com.lg.drone apps:
# DRONE_TYPE=x69_lg
```

`x69_lg` supports three video modes via `X69_LG_VIDEO_MODE`:

- **`rtsp`** (default): OpenCV pulls `rtsp://172.16.11.1/live/ch00_1` (HEVC 1280×720).
  Set `X69_LG_RTSP_JPEG_QUALITY` (1–100, higher is better).
- **`h265`**: H.265 on UDP port `1234`, reassembled and transcoded with **`ffmpeg`** on
  `PATH`. Tune `X69_LG_OUTPUT_WIDTH`, `X69_LG_OUTPUT_FPS`, and `X69_LG_JPEG_QUALITY`
  (FFmpeg `q:v`: lower = sharper). Alias: `udp`.
- **`jpeg`** (experimental): Legacy UDP JPEG on ports `7070`/`7080`; often unavailable
  on current firmware. Probe: `python turbodrone/experimental/x69/jpeg_probe.py`

Launch the backend: 
```
uvicorn web_server:app
```

In a separate terminal, launch the frontend web client:
```
npm run dev
```

Open the web client which will be at `http://localhost:5173` and you should see the drone video feed and be able to control it.

To control via a gaming controller, plug it in and move the sticks around for it to be detected and then push the toggle button to switch between keyboard and controller control.

Make sure to fly in a safe area, preferably outdoors with little wind. And note that the "Land" button _currently_ is more of an E-stop button that will stop the drone motors immediately.


## Status
Reconnection logic was solved recently.

Video feed: solid.

Controls: improved greatly via the web client. The implementation for WiFi UAV-based drones could use some fine-tuning.

Web Client: support for various inputs like keyboard, gamepad controllers, and ThinkPad TrackPoint mouse(lol).

Working on adding support for more drones from [Amazon's best-selling drone list](https://www.amazon.com/best-selling-drones/s?k=best+selling+drones).


## Contribute
To contribute support for a new "toy" drone, download the APK the drone uses on a mirror site and start reverse engineering it by decompiling to Java files with [jadx](https://github.com/skylot/jadx).
From there, look at the `AndroidManifest.xml` and see if you can find the classes that are entry points for the app. Look for port usage or protocol usage explicitly mentioned like TCP or UDP. Most of these apps will do the actual communication and video feed processing in native C++ libraries that will be embedded inside the APK. You can use a tool like Ghidra to decompile the native libraries and see if you can discover anything useful. For video feed processing you want to figure out what format it uses e.g. JPEG, YUV, etc. and also if it uses compression and what the byte structure looks like when its reforming an image frame from packets.
Additionally, Wireshark is your friend for understanding the raw data packets being sent and received by the app. Watch this [video](https://x.com/marshallrichrds/status/1923165437698670818) for an overview into the reverse engineering process used for adding support for the Hiturbo S20 drone.

Once you have the protocols and processing for RC and video figured out, make a small test program and add it to the `experimental` directory at that point if you'd like others to be able to try it out.
After that, you can work on an implementation that is compatible with the existing back-end architecture; examples of this are the `s2x` and `wifi_uav` reverse-engineered implementations.


## Experimental Support
For drones and apps with limited support which are not yet fully integrated into Turbodrone, see the `experimental` directory.
# dronetesting
