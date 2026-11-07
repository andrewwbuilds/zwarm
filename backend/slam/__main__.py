"""python -m slam live | recording PATH | sim [--scene NAME]"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slam.live import run_live


def main() -> int:
    parser = argparse.ArgumentParser(prog="slam", description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live", help="localize from the running backend's video")
    live.add_argument("--url", default="http://localhost:8000/mjpeg")
    live.add_argument("--scale", type=float, help="metres per map unit, once known")
    live.add_argument("--fov", type=float, default=90.0, help="the camera's horizontal field of view in degrees")
    live.add_argument("--no-window", action="store_true")
    live.add_argument("--save", help="folder for latest.png and trajectory.csv")
    live.add_argument("--seconds", type=float)
    rec = sub.add_parser("recording", help="run on a wifi_recorder capture folder")
    rec.add_argument("path")
    rec.add_argument("--fov", type=float, default=90.0)
    rec.add_argument("--save", help="write the map image here")
    sim = sub.add_parser("sim", help="fly the simulator and score against the truth")
    sim.add_argument("--flight", choices=["square", "circle"], default="circle")
    sim.add_argument("--save", help="write the map image here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.mode == "live":
        run_live(args.url, args.scale, args.fov, not args.no_window, Path(args.save) if args.save else None, args.seconds)
        return 0

    import cv2
    import numpy as np
    from slam.camera import Intrinsics
    from slam.plot import topdown
    from slam.runner import run
    from slam.sequences import SimFlight, hall, load_recording

    if args.mode == "recording":
        frames = load_recording(Path(args.path))
        result = run(frames, Intrinsics.from_fov(1184, 656, args.fov))
        print(result.summary())
    else:
        flight = SimFlight(hall()).forward(0.6)
        if args.flight == "square":
            flight.square(2.0).square(2.0)
        else:
            flight.arc(360, speed=0.4, turn=0.4).arc(360, speed=0.4, turn=0.4)
        frames = flight.frames
        result = run(frames, flight.intrinsics)
        print(result.summary())
    if args.save:
        trajectory = {i: p for i, _, p in result.slam.trajectory()}
        used = sorted(trajectory)
        est = np.array([trajectory[i][:3, 3] for i in used])
        truth = np.array([frames[i].truth[:3, 3] for i in used]) if frames[0].truth is not None else None
        cv2.imwrite(args.save, topdown(est, truth, points=result.slam.map_points()))
        print(f"Map saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
