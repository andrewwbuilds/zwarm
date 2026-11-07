"""python -m swarm bench: score the swarm in simulation."""

import sys

from swarm.bench import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["bench"]:
        args = args[1:]
    sys.exit(main(args))
