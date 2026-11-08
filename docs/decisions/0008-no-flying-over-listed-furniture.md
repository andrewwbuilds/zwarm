# 8. No flying over listed furniture on floor tags

**Status:** accepted.

On a tag-tracked flight a drone's only position source is floor tags in its own camera. Furniture hides them, from above and from the side. In the simulator, a drone holding behind a generator or crossing a couch lost its tags, held, landed blind, and in 3 of 4 flights came down on the couch with the tracker reading z 0. The room's fence and the 3D planner didn't care, because 1 m over a 0.7 m couch is clear air.

So on any flight with a tracker:

- `Room.flight_box` treats a listed obstacle as floor to ceiling, for paths, the safety filter and viewpoints. The map keeps its true height, so inspections still aim at the real thing and the console draws the real thing.
- `anchors_in_view` doesn't count a tag a listed obstacle hides. Viewpoints and headings need a tag that can really be seen.
- Things a camera found (`seen-` names) keep their true height and hide no tags. A detector's false sighting mustn't wall off a room, and these come and go in flight, when working out every viewpoint again would stall the flight loop.

This costs room: a small room with big furniture has fewer places to stand, and an unlisted couch still causes the original problem. Flights with outside-in positions, or no tracker, still fly over low things.

Sim results. Nothing here has flown.
