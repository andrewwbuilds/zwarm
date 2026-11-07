"""python -m mapping capture NAME | select NAME | build NAME | demo | view NAME"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path.home() / "Developer/datasets/room-maps"


def main() -> int:
    parser = argparse.ArgumentParser(prog="mapping", description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture", help="record photos from the running backend's video")
    cap.add_argument("name", help="a name for this map, e.g. living-room")
    cap.add_argument("--url", default="http://localhost:8000/mjpeg")
    cap.add_argument("--seconds", type=float)
    cap.add_argument("--no-window", action="store_true")
    cap.add_argument("--build", action="store_true", help="build the map straight after capturing")
    sel = sub.add_parser("select", help="choose photos from the recorded frames, without building")
    sel.add_argument("folder")
    sel.add_argument("--window", type=int, default=3, help="frames per group, of which the sharpest is kept (smaller keeps more)")
    bld = sub.add_parser("build", help="turn recorded frames or photos into 3D maps")
    bld.add_argument("--window", type=int, default=3, help="frames per group, of which the sharpest is kept (smaller keeps more)")
    bld.add_argument("folder", help="a map name, or a folder containing images/")
    bld.add_argument("--steps", type=int, default=7000, help="Gaussian splat training steps")
    bld.add_argument("--nerf-steps", type=int, default=5000, help="NeRF training steps: slow, about 5 minutes per 1000")
    bld.add_argument("--only", help="comma-separated outputs to make: cloud, splat, nerf (default: all three)")
    dem = sub.add_parser("demo", help="make photos in the simulator's hall and build a map from them")
    dem.add_argument("--steps", type=int, default=6000)
    dem.add_argument("--nerf-steps", type=int, default=1000)
    dem.add_argument("--only", help="comma-separated outputs to make: cloud, splat, nerf")
    view = sub.add_parser("view", help="open a finished map in Brush's viewer")
    view.add_argument("folder")
    view.add_argument("kind", nargs="?", default="splat", choices=["cloud", "splat", "nerf"])
    args = parser.parse_args()

    from mapping.reconstruct import BRUSH, OUTPUTS, StepFailed, ToolMissing, build

    def outputs_from(text: str | None) -> tuple:
        if not text:
            return OUTPUTS
        chosen = tuple(n.strip() for n in text.split(","))
        unknown = [n for n in chosen if n not in OUTPUTS]
        if unknown:
            parser.error(f"unknown output {unknown[0]!r}: choose from {', '.join(OUTPUTS)}")
        return chosen

    def where(text: str) -> Path:
        path = Path(text)
        return path if path.exists() else ROOT / text

    try:
        if args.mode == "capture":
            from mapping.capture import capture
            folder = ROOT / args.name
            kept = capture(args.url, folder, args.seconds, not args.no_window)
            if args.build and kept:
                build(folder)
        elif args.mode == "select":
            from mapping.curate import curate
            report = curate(where(args.folder), window=args.window)
            print(report.summary() + f"\nPhotos are in {where(args.folder) / 'images'}")
        elif args.mode == "build":
            build(where(args.folder), args.steps, args.nerf_steps, outputs_from(args.only), window=args.window)
        elif args.mode == "view":
            folder = where(args.folder)
            if args.kind == "splat":
                subprocess.run([str(BRUSH), "--with-viewer", str(folder / "map.ply")])
            elif args.kind == "cloud":
                subprocess.run(["open", str(folder / "cloud.html")])
            else:
                from mapping.nerf import TRUSTED_CHECKPOINTS, viewer_command
                print("Opening the NeRF viewer: go to the address it prints (usually http://localhost:7007). Ctrl-C to stop.")
                import os
                subprocess.run([str(c) for c in viewer_command(folder)], env={**os.environ, **TRUSTED_CHECKPOINTS})
        else:
            import cv2
            import shutil
            from mapping.curate import raw_name
            from slam.sequences import SimFlight, hall
            folder = ROOT / "demo-hall"
            shutil.rmtree(folder / "raw", ignore_errors=True)
            (folder / "raw").mkdir(parents=True)
            flight = SimFlight(hall(), jpeg=False).forward(0.6)
            flight.arc(360, speed=0.35, turn=0.35).turn(-90).forward(1.0).arc(360, speed=0.35, turn=0.35)
            for i, frame in enumerate(flight.frames):  # every frame, as a real capture would
                cv2.imwrite(str(folder / "raw" / raw_name(i, int(frame.t * 1000))), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            build(folder, args.steps, args.nerf_steps, outputs_from(args.only))
    except (ToolMissing, StepFailed) as error:
        print(f"\n{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
