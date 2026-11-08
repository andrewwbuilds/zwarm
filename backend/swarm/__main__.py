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
    from swarm.tracking import CameraModel, CameraWorker, PoseStore

    config = SwarmFile.load(args.config)
    links = {d.id: BackendLink(d.backend, d.id) for d in config.drones}
    store = PoseStore({d.marker: d.id for d in config.drones}, config.mount_height_m)
    workers = [CameraWorker(c["source"], CameraModel.load(c["calibration"]), config.marker_m, store) for c in config.cameras]
    for item in (*links.values(), *workers):
        item.start()
    seen = {d.id: [] for d in config.drones}
    print(f"Listening for {args.seconds:.0f} s. No commands are sent to any drone.")
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        time.sleep(0.02)
        for drone_id, pose in store.poses().items():
            seen[drone_id].append((pose.t, *pose.p, pose.yaw))
    good = True
    for z in (0.5, 1.0, 1.5):
        xs = np.linspace(config.room.lo[0], config.room.hi[0], 25)
        ys = np.linspace(config.room.lo[1], config.room.hi[1], 25)
        seen_share = np.mean([any(w.sees((x, y, z)) for w in workers) for x in xs for y in ys])
        print(f"Cameras see {seen_share * 100:.0f}% of the fence at {z:.1f} m up.")
        if seen_share < 0.99:
            print("  Goals outside that are refused, and a drone pushed out of view gets landed.")
    for d in config.drones:
        rows = np.array(seen[d.id]) if seen[d.id] else None
        link = links[d.id]
        link_text = f"link {'up' if link.connected else 'DOWN'}, {('%.0f ms' % (link.latency * 1000)) if link.latency else 'latency unknown'}"
        if link.problem:
            link_text += f", {link.problem}"
        if rows is None:
            print(f"{d.id}: {link_text}; marker {d.marker} NOT SEEN")
            good = False
            continue
        rate = len(rows) / args.seconds
        p, spread = rows[:, 1:4].mean(0), rows[:, 1:4].std(0)
        miss = float(np.linalg.norm(p[:2] - np.array(d.pad[:2])))
        print(f"{d.id}: {link_text}; {rate:.0f} poses/s, at {p.round(2).tolist()} m, jitter "
              f"{(spread * 100).round(1).tolist()} cm, heading {np.degrees(rows[:, 4].mean()):.0f} deg, {miss:.2f} m from pad")
        good = good and link.connected and not link.problem and rate >= 15 and miss < 0.3
    for item in (*links.values(), *workers):
        item.stop()
    print("Ready." if good else "Not ready: fix what's listed above before flying.")
    return 0 if good else 1


def cmd_bench(args) -> int:
    from swarm.bench import main

    return main(args.rest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("sim", help="fly a simulated swarm from the /swarm page")
    sub.add_argument("--port", type=int, default=PORT)
    sub.add_argument("--seed", type=int, default=1)
    sub.set_defaults(run=cmd_sim)

    sub = commands.add_parser("fly", help="run the real swarm from a swarm file")
    sub.add_argument("config")
    sub.add_argument("--port", type=int, default=PORT)
    sub.set_defaults(run=cmd_fly)

    sub = commands.add_parser("check", help="check links and tracking without sending any command")
    sub.add_argument("config")
    sub.add_argument("--seconds", type=float, default=10.0)
    sub.set_defaults(run=cmd_check)

    sub = commands.add_parser("markers", help="write printable ArUco markers")
    sub.add_argument("--out", default="markers")
    sub.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 30, 40, 41, 42, 43])
    sub.set_defaults(run=cmd_markers)

    sub = commands.add_parser("calibrate-lens", help="measure a camera's lens with a checkerboard")
    sub.add_argument("camera")
    sub.add_argument("--out", required=True)
    sub.add_argument("--columns", type=int, default=9, help="inner corners across")
    sub.add_argument("--rows", type=int, default=6, help="inner corners down")
    sub.add_argument("--square", type=float, required=True, help="square size in metres")
    sub.add_argument("--seconds", type=float, default=30.0)
    sub.set_defaults(run=cmd_calibrate_lens)

    sub = commands.add_parser("locate-camera", help="find where a camera is from the anchors it sees")
    sub.add_argument("camera")
    sub.add_argument("--config", required=True)
    sub.add_argument("--lens", required=True)
    sub.add_argument("--out", required=True)
    sub.set_defaults(run=cmd_locate_camera)

    sub = commands.add_parser("bench", help="score the swarm on the fixed set of simulated missions", add_help=False)
    sub.add_argument("rest", nargs=argparse.REMAINDER)
    sub.set_defaults(run=cmd_bench)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
