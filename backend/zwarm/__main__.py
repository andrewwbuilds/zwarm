"""Run Zwarm. `python -m zwarm sim`, `python -m zwarm fly swarm.json` or `python -m zwarm preflight swarm.json`."""

from __future__ import annotations

import argparse
import logging
import sys

from zwarm.app import PORT


def _serve(app, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=2)


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


def cmd_preflight(args) -> int:
    from zwarm.app import check_ready

    print(f"Checking {args.config}. This reads capabilities and video and sends no flight command.", flush=True)
    readiness = check_ready(args.config, not args.no_model)
    _print_ready(readiness)
    print("Ready to fly." if readiness.ready else "Not ready: fix what's listed above before flying.")
    return 0 if readiness.ready else 2


def cmd_fly(args) -> int:
    from zwarm.app import build, fly

    print(f"Checking {args.config} before starting. Nothing flies until you approve it.", flush=True)
    zw, readiness = fly(args.config, args.data, not args.no_model)
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
    sub.set_defaults(run=cmd_fly)

    sub = commands.add_parser("preflight", help="check a swarm file, the backends, video and detector, without flying")
    sub.add_argument("config")
    sub.add_argument("--no-model", action="store_true", help="don't require NVIDIA_API_KEY")
    sub.set_defaults(run=cmd_preflight)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
