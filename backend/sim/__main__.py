"""Fly a simulated S20 with the real turbodrone stack.

    uv run python -m sim --scene room

Then start the backend with DRONE_IP=127.0.0.1, or run ./fly.sh --sim from the repo root.
"""

import argparse
import signal
import sys
import threading

from sim.config import DEFAULT_FIRMWARE, FIRMWARES
from sim.scenes import SCENES
from sim.server import Server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="room", choices=SCENES)
    parser.add_argument("--firmware", default=DEFAULT_FIRMWARE, choices=FIRMWARES)
    parser.add_argument("--host", default="127.0.0.1", help="address the drone listens on")
    parser.add_argument("--api-port", type=int, default=8090)
    parser.add_argument("--battery", type=float, help="starting charge, 0 to 1")
    parser.add_argument("--wind", type=float, help="wind speed in m/s, replacing the scene's")
    parser.add_argument("--no-hud", action="store_true", help="leave flight data off the video")
    parser.add_argument("--list", action="store_true", help="list scenes and firmwares, then exit")
    args = parser.parse_args()

    if args.list:
        for name, build in SCENES.items():
            print(f"scene     {name:10s} {build.__doc__}")
        for name, firmware in FIRMWARES.items():
            print(f"firmware  {name:10s} {firmware.description}")
        return 0

    try:
        server = Server(scene=args.scene, firmware=args.firmware, host=args.host,
                        api_port=args.api_port, hud=not args.no_hud)
    except OSError as e:
        print(f"Could not open the drone's ports on {args.host}: {e}. Is another simulator or probe running?",
              file=sys.stderr)
        return 1
    if args.battery is not None or args.wind is not None:
        server.reset(battery=args.battery, wind=None if args.wind is None else {"speed": args.wind})
    server.start()
    print(f"[sim] Simulated S20 at {args.host}: UDP {server.control_port}, video to port 8888, "
          f"state at http://{args.host}:{server.api_port}/state", flush=True)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    try:
        done.wait()
    except KeyboardInterrupt:
        pass
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
