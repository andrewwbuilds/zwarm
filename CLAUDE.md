# ZWARM

One person talks to three cheap Wi-Fi camera drones with closed firmware, and NVIDIA Nemotron turns the sentences into missions (fork of turbodrone). Python backend, React console, a physics simulator of the S20, visual SLAM, 3D mapping, and a fleet of Raspberry Pi gateways. `README.md` says what it does, how to run it and what's broken today. `docs/decisions/` says why, starting with `0002-zwarm-architecture.md`.

## Layout

- `backend/web_server.py`: FastAPI app. The one flight controller, video pump, WebSocket and REST API.
- `backend/services/flight_controller.py`: the control loop. `tick()` is one cycle, and the simulator benches call it directly.
- `backend/control/`: stick strategies, trajectory plans, the visual stabilizer (`stabilizer.py`, `visual_motion.py`).
- `backend/models/`, `backend/protocols/`: one RC model and protocol adapter per drone family.
- `backend/sim/`: simulated S20. `config.py` tags every value with its source. `stability.py` is the drift benchmark.
- `backend/mapping/fleet.py`: one map from every drone. `uv run python -m mapping.fleet_bench` checks it against simulator truth. Placing every photo doesn't prove a map is right.
- `backend/swarm/`: multi-drone coordination. `hub.py` is the shared memory and the actions (plain methods; `commands.py` only parses fixed phrases into them), `memory.py` is SQLite, `eyes.py` and `yolo_eyes.py` are the cameras, `search.py` shares out the floor. `backend/slam/`, `backend/mapping/`: localization and photogrammetry.
- `backend/swarm/mission.py`: explore, map, inspect and search on top of the hub. `inspect.py` finds damage, `sim_cameras.py` renders the demo room.
- `backend/agent/`: the Nemotron side. `agent.py` is `handle`, `approve` and `observe`, `tools.py` the twelve checked tools, `nemotron.py` the client and model chain, `vision.py` the image calls, `eval.py` the live test set.
- `backend/zwarm/`: the whole app on port 8010. `app.py` composes every part. Its `Front` is what `swarm/server.py`'s Runtime and routes hold as `co`, so a tick is hub then missions, and a stop button ends the mission too. The `/zwarm/*` routes are in `app.py`, `preflight.py` is read-only and `markers.py` prints floor tags.
- `docs/runbook.md`: flight day, staged test and tuning table, with file:line for each constant. `backend/zwarm/real.example.json` is the swarm file to copy.
- `frontend/`: Vite and React console. `?backend=` picks the backend per tab. `/swarm` (`src/pages/SwarmPage.tsx`, `src/swarm/`) is the operator console; `?api=` picks its backend and `?mock=1` fakes one.
- `tools/pi/`: Pi install, fleet scripts and watchdogs. `docs/research/`: reverse-engineering notes.

## Commands

```sh
./fly.sh --sim                                      # fly the simulator in the browser
./fly.sh --swarm                                    # the whole app in the simulator, console at /swarm
./fly.sh --swarm preflight real.json                # read-only pass or fail per drone, sends no flight command
./fly.sh --swarm fly real.json [--drones 1,2]        # the whole app on real drones
cd backend
uv sync
uv run python -m unittest discover tests            # all backend tests, about 10 minutes
uv run python -m unittest tests.test_stabilizer     # one module
uv run python -m sim.stability                      # drift benchmark, about 5 minutes
uv run python -m sim.stability --quick --no-robust  # about 30 seconds
uv run python -m zwarm sim [--inside-out]           # the whole app, no browser, on port 8010
uv run python -m unittest tests.test_zwarm_e2e      # the pitch sentences with a scripted model, no network
SWARM_E2E=1 uv run python -m unittest tests.test_swarm_e2e tests.test_swarm_floor_e2e  # real backends, simulated drones, 2 minutes
uv run python -m swarm bench --seeds 3              # avoidance on a simulated tracker
uv run python -m swarm.mission_bench                # missions, about a minute
uv run python -m agent.eval                         # the live model, about 240 requests on the user's key
cd ../frontend && npx tsc -b --noEmit && npm run lint
npm run dev                                         # /swarm on port 5173, for a backend on 8010
```

## Gotchas

- Sim results are not drone results. The simulated flight controller is an estimate, so say which one a number came from.
- The S20 sends back only video and a link-quality byte. No attitude, height, flow or battery (`docs/research/S2x.md`).
- Takeoff and land share one toggle bit on S2x, so `S2xDroneModel` guards against reversing a flight.
- On tag-tracked flights a listed obstacle counts as floor to ceiling for flying (`Room.flight_box`), and hides the tags behind it, because a drone over furniture loses its position (`docs/decisions/0008-no-flying-over-listed-furniture.md`). Read `o.lo` and `o.hi` for what it looks like, and `flight_box` for where a drone may go.
- Anything that raises in `FlightController.tick` drops a control packet. Keep helper failures from reaching it.
- `backend/services/flight_controller.py` uses CRLF line endings. Preserve them, or the whole file shows as changed.
- `backend/runs/` isn't ignored and fills with flight recordings. Stage explicit paths.
- Camera geometry assumes a camera tilted 45 degrees down (`eyes.DEFAULT_TILT`). Pass `tilt=0.0` for a level one.
- Real positions come from 25 printed AprilTags, family `tagStandard41h12`, IDs 0-24, flat on the floor in a grid and seen by each drone's own camera. Don't ask the user to print ArUco or QR codes. OpenCV can't read this family, so use `pupil-apriltags`. Its `tag_size` is the edge of the 5-cell white square inside the black ring: 0.0767 m on these sheets. They printed at 75%, so ignore the "184mm" printed on them.
- `backend/swarm/inspect.py` shares a name with the standard library. Run swarm code with `python -m swarm...` from `backend/`. `python swarm/x.py` puts that folder first on the path and breaks `import inspect`.
- Several agents may share this checkout. Stage by path or hunk, never `git add -A`.
