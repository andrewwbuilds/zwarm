"""Run Zwarm. `python -m zwarm sim`, `python -m zwarm fly swarm.json`, `python -m zwarm preflight swarm.json` or `python -m zwarm markers`."""

from __future__ import annotations

import argparse
import logging
import sys

from zwarm.app import PORT


def _serve(zw, app, port: int) -> None:
    from swarm.server import serve

    serve(app, zw.runtime, port, log_level="warning", timeout_graceful_shutdown=2)


def cmd_sim(args) -> int:
    from zwarm.app import build, sim

    zw = sim(args.seed, args.data, args.rate, args.inside_out)
    print(f"SIMULATED Zwarm at http://localhost:{args.port}. Nothing here talks to a real drone. Console: http://localhost:5173/swarm", flush=True)
    _serve(build(zw), args.port)
    return 0


def _print_ready(readiness) -> None:
    for line in readiness.notes:
        print(f"  ok: {line}")
    for line in readiness.warnings:
        print(f"  WARNING: {line}")
    for line in readiness.problems:
        print(f"  NOT READY: {line}")


def _subset(args) -> str | None:
    """The swarm file with only --drones in it. Prints the reason and returns None if that can't be done."""
    from zwarm.preflight import only_drones

    try:
        return only_drones(args.config, args.drones)
    except FileNotFoundError:
        print(f"There's no swarm file at {args.config}.")
    except (OSError, ValueError, KeyError) as error:
        print(f"Can't read the swarm file {args.config}: {error}")
    return None


def cmd_preflight(args) -> int:
    from zwarm.preflight import NOT_READY, run, watch

    path = _subset(args)
    if path is None:
        return NOT_READY
    if args.watch:
        return watch(path)
    return run(path, not args.no_model, args.seconds, args.on_pad, args.save_frames, args.vision, out=lambda text: print(text, flush=True))


def cmd_markers(args) -> int:
    from zwarm.markers import write

    only = [int(part) for part in args.ids.split(",")] if args.ids else None
    pages, section, real_m = write(args.out, args.tag_width_mm / 1000, args.origin, args.spacing, args.columns, args.rows, args.start_id, only)
    print(f"Wrote {len(pages)} page(s) and {section} in {args.out}. Each tag is {real_m * 1000:.1f} mm wide, with a {real_m * 5 / 9 * 1000:.1f} mm white square.")
    print("Print at 100%, no fit to page, then measure the white square. Copy the \"anchors\" section into your swarm file and set white_square_edge_m to what you measured.")
    return 0


def cmd_fly(args) -> int:
    from zwarm.app import build, fly

    path = _subset(args)
    if path is None:
        return 2
    print(f"Checking {args.config} before starting. Nothing flies until you approve it.", flush=True)
    zw, readiness = fly(path, args.data, not args.no_model)
    _print_ready(readiness)
    if zw is None:
        print("Refusing to fly: fix what's listed above. `python -m zwarm preflight FILE` repeats this check.")
        return 2
    print(f"REAL DRONES on http://localhost:{args.port}. Console: http://localhost:5173/swarm. Ctrl-C lands anything in the air, then quits.", flush=True)
    _serve(build(zw), args.port)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m zwarm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("sim", help="the demo room with simulated drones and cameras")
    sub.add_argument("--port", type=int, default=PORT)
    sub.add_argument("--seed", type=int, default=1)
    sub.add_argument("--data", help="where findings and maps go; a new temporary folder if left out")
    sub.add_argument("--rate", type=float, default=5.0, help="camera frames a second per simulated drone")
    sub.add_argument("--inside-out", action="store_true", help="positions from reading floor tags in rendered frames, as real flights do")
    sub.set_defaults(run=cmd_sim)

    sub = commands.add_parser("fly", help="real drones from a swarm file")
    sub.add_argument("config")
    sub.add_argument("--port", type=int, default=PORT)
    sub.add_argument("--data", help="where findings and maps go; ~/Developer/datasets/zwarm if left out")
    sub.add_argument("--no-model", action="store_true", help="fly on fixed phrases if there's no NVIDIA_API_KEY")
    sub.add_argument("--drones", help="fly only these, for example 1 or 1,2. The others stay on the floor")
    sub.set_defaults(run=cmd_fly)

    sub = commands.add_parser("preflight", help="read-only pass or fail for each drone. Sends no flight command")
    sub.add_argument("config")
    sub.add_argument("--no-model", action="store_true", help="don't require NVIDIA_API_KEY or try the model")
    sub.add_argument("--drones", help="check only these, for example 1 or 1,2")
    sub.add_argument("--seconds", type=float, help="how long to watch the video; 25 s, or 6 s with --on-pad")
    sub.add_argument("--on-pad", action="store_true", help="the drones stay on their pads, so skip the tag and pose checks")
    sub.add_argument("--save-frames", metavar="DIR", help="save each drone's sharpest frame here")
    sub.add_argument("--vision", action="store_true", help="also send one real frame to the vision model, which takes up to 40 s")
    sub.add_argument("--watch", action="store_true", help="print tags and pose until Ctrl-C, safe with a drone in the air")
    sub.set_defaults(run=cmd_preflight)

    sub = commands.add_parser("markers", help="write printable floor tags and their swarm file section")
    sub.add_argument("--out", default="markers")
    sub.add_argument("--tag-width-mm", type=float, default=138.0, help="the whole tag; 138 matches the sheets already printed")
    sub.add_argument("--ids", help="print only these, for example 3,7; anchors.json still lists the whole grid")
    sub.add_argument("--origin", type=float, nargs=2, default=(-2.0, 2.0), metavar=("X", "Y"), help="the centre of the first tag")
    sub.add_argument("--spacing", type=float, default=1.0, help="metres between tag centres")
    sub.add_argument("--columns", type=int, default=5)
    sub.add_argument("--rows", type=int, default=5)
    sub.add_argument("--start-id", type=int, default=0)
    sub.set_defaults(run=cmd_markers)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
