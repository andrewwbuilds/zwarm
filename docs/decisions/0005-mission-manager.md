# 5. Missions sit in the hub's search slot

Explore, map, inspect and search all need the same things from the hub: choose goals each tick, stop when told. So `swarm/mission.py` puts its work where a search goes.

- **`hub.search` is the plug.** It's anything with `step(now)`, `state`, `label` and `coverage`. Hub actions already stop whatever is "searching", so hold, land and E-stop end a mission without the hub knowing missions exist, and they work with the model down.
- **Work is a pool, not a plan per drone.** A drone holds a piece only while flying to it or looking from it. When a drone drops out its piece is free, and whoever is next free takes it. There's no reassignment step to get wrong.
- **One coverage record for the session.** What's been seen survives a change of region or mission, which is what lets "which areas haven't been explored?" answer at any time. Only a map flies seen floor again on purpose, since those earlier looks saved no photos.
- **Viewpoints stay inside the region.** A drone could often see a region from outside it, but "focus on the north side" has to visibly send everyone north. A drone with nothing left to look at still moves in.
- **On floor tags, a viewpoint must show a tag.** The tracker holds a drone at 0.3 m of position doubt and lands it at 0.6 m, so a view with no tag in it would cost the drone. Floor only visible from such a view is left out of the share, and shows up as `out_of_view_m2` in `gaps()`.
- **With real video, a look is a new picture.** The hub only counts a look when its detector answers. A map shouldn't fail because YOLO did, so `frames` takes over the counting when it's there.
- **The inspector only sees drones settled at a view.** Vision takes 3 to 14 s an image and its queue is short. Frames from the way there would push out the ones that matter.
- **The flight limit is per flight.** There's no battery reading and no way to tell a swap from a relanding. Time across flights only raises a warning, so a second flight on the same battery is on the operator.
- **A landing that wasn't seen isn't sent again.** On the S20 a second toggle could take off instead.
