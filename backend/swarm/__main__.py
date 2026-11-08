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


def cmd_markers(args) -> int:
    """Printable marker images for drones, things to avoid, and anchors."""
    import cv2

    from swarm.tracking import marker_image

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    for marker in args.ids:
        cv2.imwrite(str(folder / f"marker-{marker:02d}.png"), cv2.copyMakeBorder(
            marker_image(marker), 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255))
    print(f"Wrote {len(args.ids)} markers to {folder}. Print them without scaling, then measure the black")
    print("square: that size goes in the swarm file. The white border must stay. Mount drone markers")
    print("flat with the top of the page toward the nose.")
    return 0


def _frames(source, count: int, gap_s: float):
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {source}")
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"camera {source} stopped delivering frames")
            yield frame
            time.sleep(gap_s)
    finally:
        capture.release()


def _source(text: str):
    return int(text) if text.isdigit() else text


def cmd_calibrate_lens(args) -> int:
    from swarm.tracking import lens_from_checkerboards

    print(f"Move a printed checkerboard around in front of camera {args.camera}, tilting it, for {args.seconds:.0f} s.")
    frames = list(_frames(_source(args.camera), int(args.seconds / 0.5), 0.5))
    model = lens_from_checkerboards(frames, (args.columns, args.rows), args.square)
    model.save(args.out)
    print(f"Saved {args.out}: reprojection error {model.reprojection_px:.2f} px. Above about 1 px, try again.")
    return 0


def cmd_locate_camera(args) -> int:
    from swarm.server import SwarmFile
    from swarm.tracking import CameraModel, camera_position, locate_camera

    config = SwarmFile.load(args.config)
    frame = next(_frames(_source(args.camera), 1, 0.0))
    model = locate_camera(frame, CameraModel.load(args.lens), config.anchors, config.anchor_m)
    model.save(args.out)
    print(f"Saved {args.out}: camera at {camera_position(model).round(2).tolist()} m, "
          f"reprojection error {model.reprojection_px:.2f} px.")
    return 0


def cmd_check(args) -> int:
    """Everything except flying: is each drone linked, is it tracked, and how well?"""
    import numpy as np

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
