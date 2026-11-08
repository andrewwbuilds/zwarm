"""Talk to a simulated fleet: `python -m swarm.talk`, then type commands such as "find the backpack".

Only the simulator is wired up. Real drones also need the room tracker, which isn't built yet.
A line "wait 5" pauses a piped script for that many simulated seconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from swarm import bench
from swarm.bench import hall, solid
from swarm.coordinator import Coordinator, DroneConfig
from swarm.eyes import Item, SimEyes
from swarm.hub import Hub
from swarm.memory import Memory
from swarm.room import Room
from swarm.sim_fleet import SimFleet

PADS = {"a": (-3.0, -1.5), "b": (-3.0, 0.0), "c": (-3.0, 1.5)}
FURNITURE = [("couch", solid((0.0, -2.6, 0.0), (0.9, -1.0, 0.9))), ("bed", solid((1.2, 0.6, 0.0), (2.6, 2.4, 0.7)))]
HIDDEN = [Item("backpack", (2.6, -2.2, 0.0), (2.9, -1.9, 0.4)), Item("chair", (-0.8, 1.6, 0.0), (-0.3, 2.1, 0.9))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm.talk")
    parser.add_argument("--memory", default=":memory:", help="SQLite file, to remember between runs")
    parser.add_argument("--speed", type=float, default=1.0, help="how many times faster than real time")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)

    fleet = SimFleet(hall(*(box for _, box in FURNITURE)), PADS, seed=1)
    room = Room(fleet.scene.bounds.lo, fleet.scene.bounds.hi)  # the map starts empty: they find the furniture themselves
    configs = [DroneConfig(k, (v[0], v[1], 0.0)) for k, v in PADS.items()]
    coordinator = Coordinator(room, configs, fleet.links)
    items = [Item(label, box.lo, box.hi) for label, box in FURNITURE] + HIDDEN
    memory = Memory(args.memory)
    hub = Hub(coordinator, memory, SimEyes(fleet, items), report=lambda text: print(f"[{fleet.t:5.1f}] {text}", flush=True))
    lock = threading.Lock()
    running = threading.Event()
    running.set()

    def fly() -> None:
        while running.is_set():
            with lock:
                fleet.advance(bench.TICK)
                hub.tick(fleet.t, fleet.poses())
            time.sleep(bench.TICK / args.speed)

    worker = threading.Thread(target=fly, daemon=True)
    worker.start()
    print('Three simulated drones in an unmapped room. Try "take off", then "find the backpack". Ctrl-D quits.', flush=True)
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text.startswith("wait "):
                with lock:
                    until = fleet.t + float(text.split()[1])
                while fleet.t < until:
                    time.sleep(0.01)
                continue
            with lock:
                print(f"[{fleet.t:5.1f}] > {text}\n[{fleet.t:5.1f}] {hub.command(text)}", flush=True)
    except KeyboardInterrupt:
        pass
    running.clear()
    worker.join(timeout=2.0)
    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
