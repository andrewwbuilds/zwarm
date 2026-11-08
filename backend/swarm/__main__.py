"""Command line for the swarm. Run `python -m swarm --help`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PORT = 8010


def cmd_sim(args) -> int:
    import uvicorn

    from swarm.server import build_app, sim_runtime

    print(f"Simulated swarm at http://localhost:{args.port}. Open http://localhost:5173/swarm to fly it.")
    uvicorn.run(build_app(sim_runtime(args.seed, inside_out=args.inside_out)), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_fly(args) -> int:
    import uvicorn

    from swarm.server import SwarmFile, build_app, real_runtime
    from swarm.tracking import SwarmFileError

    try:
        config = SwarmFile.load(args.config)
    except SwarmFileError as error:
        print(f"Refusing to fly: {error}")
        return 2
    print(f"Swarm ready at http://localhost:{args.port}. Open http://localhost:5173/swarm to fly it.")
    print("Nothing moves until you press Take off. Ctrl-C lands anything in the air, then quits.")
    uvicorn.run(build_app(real_runtime(config)), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_layout(args) -> int:
    """Write the swarm file's anchors section for a grid of already printed tags."""
    import numpy as np

    from swarm.tracking import Lens, anchors_in_view, anchors_section, load_anchors, tag_grid

    if args.tag_width_m is not None and args.white_square_m is not None:
        raise SystemExit("give --white-square-m or --tag-width-m, not both")
    edge = args.white_square_m if args.tag_width_m is None else args.tag_width_m * 5 / 9
    tags = tag_grid(tuple(args.origin), args.spacing, args.columns, args.rows, args.start_id, args.across_deg, args.rotation_deg)
    section = anchors_section(None if edge is None else round(edge, 5), tags)
    Path(args.out).write_text(json.dumps({"anchors": section}, indent=1))
    xs, ys = [t["x"] for t in tags], [t["y"] for t in tags]
    fence = {"lo": [min(xs) - 0.8, min(ys) - 0.4, 0.0], "hi": [max(xs) - 0.3, max(ys) + 0.4, 2.2]}
    print(f"Wrote {args.out}: tags {tags[0]['tag']} to {tags[-1]['tag']}.")
    print(f"  tag {tags[0]['tag']} at ({tags[0]['x']}, {tags[0]['y']}), tag {tags[-1]['tag']} at ({tags[-1]['x']}, {tags[-1]['y']}), "
          f"tops pointing {args.rotation_deg:g} degrees from +x. The top is the side opposite the printed label.")
    print(f"  Suggested fence, for drones facing +x: {json.dumps(fence)}")
    if edge is None:
        print("  white_square_edge_m is null: fill it in from a ruler. The swarm refuses to fly without it.")
        return 0
    anchors = load_anchors(section)
    lens = Lens.from_fov()
    counts = [len(anchors_in_view(anchors, lens, np.radians(45.0), (x, y, 1.0), 0.0))
              for x in np.arange(fence["lo"][0], fence["hi"][0] + 1e-6, 0.1) for y in np.arange(fence["lo"][1], fence["hi"][1] + 1e-6, 0.1)]
    print(f"  From 1 m up, facing +x, inside that fence: at least 1 tag readable from {np.mean(np.array(counts) >= 1) * 100:.0f}% of "
          f"spots, at least 2 from {np.mean(np.array(counts) >= 2) * 100:.0f}%, by geometry alone.")
    return 0


def _detections(config, drone_id: str, seconds: float):
    """What one drone's camera reads over some seconds, frame by frame."""
    import cv2
    import numpy as np

    from swarm.frames import StreamFrames
    from swarm.tracking import AnchorReader

    frames = StreamFrames(config.backends())
    frames.start()
    reader = AnchorReader(config.anchors)
    out, last = [], None
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            frame = frames.latest(drone_id)
            if frame is None or frame.number == last:
                time.sleep(0.01)
                continue
            last = frame.number
            gray = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                out.append((gray, reader.find(gray)))
    finally:
        frames.stop()
    return out, reader


def cmd_fit_focal(args) -> int:
    """Fit a drone's focal length from two or more tags in view, and optionally store it."""
    from swarm.server import SwarmFile
    from swarm.tracking import fit_focal

    config = SwarmFile.load(args.config)
    print(f"Hold drone {args.drone}, motors off, about 1 m up over the grid, with several tags in view, for {args.seconds:.0f} s.")
    captured, _ = _detections(config, args.drone, args.seconds)
    if not captured:
        raise SystemExit("no frames arrived. Is its backend running and sending video?")
    height, width = captured[0][0].shape
    result = fit_focal([seen for _, seen in captured], width, height)
    scaled = result["focal_px"] * 1184 / width
    print(f"{len(captured)} frames, {result['frames']} with 2 or more tags. Focal length {scaled:.1f} px at 1184 wide "
          f"(90 degrees would be 592). Corners miss by {result['error_px']:.2f} px, against {result['error_at_guess_px']:.2f} px at the guess.")
    if result["error_px"] > 2.5:
        print("That's a large miss: check the tag positions, spacing and white_square_edge_m. Not storing it.")
        return 1
    if args.write:
        data = json.loads(Path(args.config).read_text())
        for drone in data["drones"]:
            if str(drone["id"]) == args.drone:
                drone["focal_px"] = round(scaled, 1)
        Path(args.config).write_text(json.dumps(data, indent=1))
        print(f"Stored focal_px for drone {args.drone} in {args.config}.")
    return 0


def cmd_check(args) -> int:
    """Everything except flying: links, video, and which tags each drone can see. Sends no flight command."""
    import numpy as np

    from swarm.frames import StreamFrames
    from swarm.links import BackendLink
    from swarm.server import SwarmFile
    from swarm.tracking import InsideOutTracker, SwarmFileError

    try:
        config = SwarmFile.load(args.config)
    except SwarmFileError as error:
        print(f"Not ready: {error}")
        return 2
    links = {d.id: BackendLink(d.backend, d.id) for d in config.drones}
    frames = StreamFrames(config.backends())
    tracker = InsideOutTracker(config.optics(), config.anchors, (config.room.lo, config.room.hi), config.video_latency_s)
    for link in links.values():
        link.start()
    frames.start()
    tracker.start(frames)
    print(f"Watching for {args.seconds:.0f} s. A drone on the floor can't see the floor: its camera looks too low.")
    print("Lift each drone by hand, motors off, to about 1 m over its pad, nose along its heading, to see its tags.\n")
    seen: dict[str, dict[str, int]] = {d.id: {} for d in config.drones}
    last: dict[str, list] = {d.id: [] for d in config.drones}
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        time.sleep(0.1)
        now = time.monotonic()
        report = tracker.report(now)
        poses = tracker.poses(now)
        for drone_id, row in report.items():
            for tag in row["seeing"]:
                seen[drone_id][tag] = seen[drone_id].get(tag, 0) + 1
            if poses[drone_id].fix_age < 0.5:
                last[drone_id].append(poses[drone_id].p)
    good = True
    report = tracker.report(time.monotonic())
    for d in config.drones:
        link, row = links[d.id], report[d.id]
        print(f"Drone {d.id}: link {'up' if link.connected else 'DOWN'}" + (f" ({link.latency * 1000:.0f} ms)" if link.latency else "")
              + (f", {link.problem}" if link.problem else "") + f"; {row['frames'] / args.seconds:.1f} frames/s read")
        print(f"  should see over its pad at 1 m: {', '.join(tracker.expected(d.id)) or 'NOTHING'}")
        print(f"  actually saw: {', '.join(sorted(seen[d.id], key=lambda t: int(t[4:]))) or 'nothing'}")
        if last[d.id]:
            p = np.mean(np.array(last[d.id]), axis=0)
            print(f"  position from tags: ({p[0]:.2f}, {p[1]:.2f}), {p[2]:.2f} m up. Pad: ({d.pad[0]}, {d.pad[1]}). Check this against a tape measure.")
        unknown = tracker.readers[d.id].unknown
        if unknown:
            print(f"  also saw tags not in the file: {sorted(unknown)}")
        good = good and link.connected and not link.problem and bool(tracker.expected(d.id)) and bool(seen[d.id])
    tracker.stop()
    frames.stop()
    for link in links.values():
        link.stop()
    print("\nReady." if good else "\nNot ready: fix what's listed above before flying.")
    return 0 if good else 1


def cmd_bench(args) -> int:
    from swarm.bench import main

    return main(args.rest)


def cmd_floor_bench(args) -> int:
    from swarm.floor_bench import main

    return main(args.rest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("sim", help="fly a simulated swarm from the /swarm page")
    sub.add_argument("--port", type=int, default=PORT)
    sub.add_argument("--seed", type=int, default=1)
    sub.add_argument("--inside-out", action="store_true", help="track from rendered camera frames, as real flights do")
    sub.set_defaults(run=cmd_sim)

    sub = commands.add_parser("fly", help="run the real swarm from a swarm file")
    sub.add_argument("config")
    sub.add_argument("--port", type=int, default=PORT)
    sub.set_defaults(run=cmd_fly)

    sub = commands.add_parser("check", help="links, video and tags in view, without sending a flight command")
    sub.add_argument("config")
    sub.add_argument("--seconds", type=float, default=15.0)
    sub.set_defaults(run=cmd_check)

    sub = commands.add_parser("layout", help="write the anchors section for a grid of printed tags")
    sub.add_argument("--out", default="anchors.json")
    sub.add_argument("--start-id", type=int, default=0)
    sub.add_argument("--spacing", type=float, default=1.0, help="metres between tag centres")
    sub.add_argument("--origin", type=float, nargs=2, default=(-2.0, 2.0), metavar=("X", "Y"), help="the centre of the first tag")
    sub.add_argument("--columns", type=int, default=5)
    sub.add_argument("--rows", type=int, default=5)
    sub.add_argument("--across-deg", type=float, default=-90.0, help="direction ids rise along a row; the next row is 90 degrees left of it")
    sub.add_argument("--rotation-deg", type=float, default=0.0, help="the direction every tag's top points")
    sub.add_argument("--white-square-m", type=float, help="measured edge of the white square inside the black ring")
    sub.add_argument("--tag-width-m", type=float, help="or the measured width of the whole printed tag; 5/9 of it is used")
    sub.set_defaults(run=cmd_layout)

    sub = commands.add_parser("fit-focal", help="fit a drone's focal length from tags at known spacing in its video")
    sub.add_argument("config")
    sub.add_argument("--drone", required=True)
    sub.add_argument("--seconds", type=float, default=8.0)
    sub.add_argument("--write", action="store_true", help="store the result in the swarm file")
    sub.set_defaults(run=cmd_fit_focal)

    sub = commands.add_parser("bench", help="score collision avoidance in simulation, with a simulated tracker", add_help=False)
    sub.add_argument("rest", nargs=argparse.REMAINDER)
    sub.set_defaults(run=cmd_bench)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
