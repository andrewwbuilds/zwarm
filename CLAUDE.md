# ZWARM

Host software for cheap Wi-Fi camera drones with closed firmware (fork of turbodrone). Python backend, React console, a physics simulator of the S20, visual SLAM, 3D mapping, and a fleet of Raspberry Pi gateways. See `README.md` for what it does and `docs/decisions/` for why.

## Layout

- `backend/web_server.py`: FastAPI app. The one flight controller, video pump, WebSocket and REST API.
- `backend/services/flight_controller.py`: the control loop. `tick()` is one cycle, and the simulator benches call it directly.
- `backend/control/`: stick strategies, trajectory plans, the visual stabilizer (`stabilizer.py`, `visual_motion.py`).
- `backend/models/`, `backend/protocols/`: one RC model and protocol adapter per drone family.
- `backend/sim/`: simulated S20. `config.py` tags every value with its source. `stability.py` is the drift benchmark.
- `backend/swarm/`: multi-drone coordination. `hub.py` is the shared memory and the actions (plain methods; `commands.py` only parses fixed phrases into them), `memory.py` is SQLite, `eyes.py` and `yolo_eyes.py` are the cameras, `search.py` shares out the floor. `backend/slam/`, `backend/mapping/`: localization and photogrammetry.
- `frontend/`: Vite and React console. `?backend=` picks the backend per tab.
- `tools/pi/`: Pi install, fleet scripts and watchdogs. `docs/research/`: reverse-engineering notes.

## Commands

```sh
./fly.sh --sim                                      # fly the simulator in the browser
cd backend
uv sync
uv run python -m unittest discover tests            # all backend tests
uv run python -m unittest tests.test_stabilizer     # one module
uv run python -m sim.stability                      # drift benchmark, about 5 minutes
uv run python -m sim.stability --quick --no-robust  # about 30 seconds
cd ../frontend && npx tsc -b --noEmit && npm run lint
```

## Gotchas

- Sim results are not drone results. The simulated flight controller is an estimate, so say which one a number came from.
- The S20 sends back only video and a link-quality byte. No attitude, height, flow or battery (`docs/research/S2x.md`).
- Takeoff and land share one toggle bit on S2x, so `S2xDroneModel` guards against reversing a flight.
- Anything that raises in `FlightController.tick` drops a control packet. Keep helper failures from reaching it.
- `backend/services/flight_controller.py` uses CRLF line endings. Preserve them, or the whole file shows as changed.
- `backend/runs/` isn't ignored and fills with flight recordings. Stage explicit paths.
- Camera geometry assumes a camera tilted 45 degrees down (`eyes.DEFAULT_TILT`). Pass `tilt=0.0` for a level one.
- Real positions come from 25 printed AprilTags, family `tagStandard41h12`, IDs 0-24, flat on the floor in a grid and seen by each drone's own camera. Don't ask the user to print ArUco or QR codes. OpenCV can't read this family, so use `pupil-apriltags`. Its `tag_size` is the 5-cell black square, not the 9-cell tag.
- Several agents may share this checkout. Stage by path or hunk, never `git add -A`.
