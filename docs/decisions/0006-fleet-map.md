# 6. One map from every drone's photos

Status: accepted. Every number here is from the simulator. No real footage has been through it.

## Context

Three drones each fly part of a room, and the pitch needs one shared 3D map. COLMAP places photos from the pictures alone, so it needs no shared positions, but it can only join drones whose photos overlap.

## Decisions

**Match every pair of photos, and cap the photos at 240.** The single-drone pipeline switches to matching neighbours in time above 250 photos, which can never link one drone to another. Exhaustive matching grows with the square of the photo count: 240 photos built in 114 s and 360 in 466 s, the second under heavier machine load. `MAP_BUILD_PHOTOS` changes the cap.

**Spend photos along motion, not time.** A hover earns few photos and a turn earns many.

**Give each drone its own camera model.** With three simulated lenses differing by up to 9 degrees, per-drone cameras gave a median camera error of 4 mm and a shared camera 91 mm. Each camera is seeded with the 90 degree estimate and refined. A guess of 75 degrees on a true 90 still came out right.

**Require 30 matches for a link, not COLMAP's default of 15.** Drones with no real overlap showed false links of 20 to 24 inliers, probably from the simulator's repeating wall texture. Those joined accurate sub-maps in the wrong place: every photo placed, in one model, with cameras about 1 m off. At 30 they split into separate accurate models instead. Real links on the normal flights reached 150 inliers. `MAP_MIN_LINK` changes it, and this is the first thing to tune on real video.

**Check the map against tracked positions, because the photo count can't show a bad map.** Both bad maps above placed over 90% of their photos in one model. With `pose_of`, a build compares placed cameras with the floor-tag tracker. At least 80% of photos must sit within 0.5 m, using a fit that ignores wild cameras. A map that passes goes into room coordinates. One that fails is reported as warped or folded. Separate maps that each pass are joined by position, which gives one map even when the drones share no view.

## What this doesn't solve

A floor that repeats exactly still folds the map. The check catches it, but nothing fixes it. Restricting matching to photos the tracker says overlap would be the next step.

A drone flying straight at one heading can't pin down its own lens. In the simulator that cost 0.25 m on that drone's sub-map.
