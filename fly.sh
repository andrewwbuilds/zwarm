#!/bin/zsh
# Fly the S20 from this Mac. Put the drone on the ground, power it on, then:
#
#   ./fly.sh
#
# Joins the drone's Wi-Fi if needed, checks the link, starts the backend and web
# client, and opens the browser. Ctrl-C stops everything and returns to your
# normal Wi-Fi. Land first: quitting cuts the drone's control link.
#
# No drone? Fly the simulated one with the same backend and web client:
#
#   ./fly.sh --sim [--scene room|garage|backyard|field|gym|course] [--firmware s20]
#
# The swarm and its operator console, on port 8010 and at /swarm:
#
#   ./fly.sh --swarm                 the simulator
#   ./fly.sh --swarm fly swarm.json  real drones, once tools/pi/fleet.sh has the tunnels up
setopt pipefail
start_dir=$PWD
cd "${0:A:h}" || exit 1

LOG=fly.log
joined_wifi=0
pids=()
sim=0
if [[ $1 == --sim ]]; then
  sim=1
  shift
fi
swarm=0
if [[ $1 == --swarm ]]; then
  swarm=1
  shift
fi

wifi_router() { ipconfig getoption "$WIFI_DEVICE" router 2>/dev/null; }
# The drone's address is whatever gateway its Wi-Fi hands out, so read it
# instead of assuming one. Already joined by hand if the gateway is 172.16.x.x.
on_drone_wifi() { [[ $(wifi_router) == 172.16.* ]]; }

cleanup() {
  trap - EXIT INT TERM
  (( ${#pids} )) && kill $pids 2>/dev/null
  for pid in $pids; do
    # Up to 30 s: Zwarm lands anything in the air before it quits, and must not be killed part way down.
    for i in {1..60}; do
      kill -0 $pid 2>/dev/null || break
      sleep 0.5
    done
    kill -9 $pid 2>/dev/null
  done
  if (( joined_wifi )); then
    echo "Returning to your normal Wi-Fi..."
    # Forget the drone first so auto-join can't pick it again.
    networksetup -removepreferredwirelessnetwork "$WIFI_DEVICE" "$DRONE_WIFI" >/dev/null
    networksetup -setairportpower "$WIFI_DEVICE" off
    sleep 1
    networksetup -setairportpower "$WIFI_DEVICE" on
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

: > "$LOG"
if (( swarm )); then
  mode=${1:-sim}
  (( $# )) && shift
  args=("$@")
  if [[ $mode == fly || $mode == preflight ]]; then
    if [[ -z $args[1] ]]; then
      echo "Which swarm file? ./fly.sh --swarm $mode swarm.json" >&2
      exit 1
    fi
    args[1]=$(cd "$start_dir" && print -r -- ${args[1]:A})
  fi
  if curl -sf -m 1 -o /dev/null http://localhost:8010/zwarm/state; then
    echo "Something is already answering on port 8010. Stop it first." >&2
    exit 1
  fi
  if [[ $mode == preflight ]]; then
    cd backend && exec uv run --frozen --offline python -m zwarm preflight "${args[@]}"
  fi
  (cd backend && exec uv run --frozen --offline python -m zwarm "$mode" "${args[@]}") >> "$LOG" 2>&1 &
  pids+=($!)
  zwarm_pid=$!
  # Takes a few seconds in the simulator and longer for real drones, which get checked first.
  for i in {1..120}; do
    curl -sf -m 1 -o /dev/null http://localhost:8010/zwarm/state && break
    kill -0 $zwarm_pid 2>/dev/null || break
    sleep 0.5
  done
  if ! curl -sf -m 1 -o /dev/null http://localhost:8010/zwarm/state; then
    cat "$LOG" >&2
    echo "Zwarm did not start. The reason is above, and in $LOG." >&2
    exit 1
  fi
  if ! curl -sf -m 1 -o /dev/null http://localhost:5173; then
    (cd frontend && exec node_modules/.bin/vite --strictPort) >> "$LOG" 2>&1 &
    pids+=($!)
    for i in {1..40}; do
      curl -sf -m 1 -o /dev/null http://localhost:5173 && break
      sleep 0.5
    done
  fi
  if ! curl -sf -m 1 -o /dev/null http://localhost:5173; then
    echo "The web client did not start. See $LOG." >&2
    exit 1
  fi
  command -v open >/dev/null && open http://localhost:5173/swarm
  grep -E "SIMULATED|REAL DRONES|WARNING" "$LOG"
  echo "Console: http://localhost:5173/swarm. Land first, then press Ctrl-C here to quit."
  while kill -0 $zwarm_pid 2>/dev/null; do
    sleep 1
  done
  echo "Zwarm stopped. See $LOG." >&2
  exit 1
fi
if (( sim )); then
  export DRONE_TYPE=s2x DRONE_IP=127.0.0.1 SIM_URL=http://localhost:8090
  echo "Starting the simulated S20. Nothing here talks to a real drone."
  (cd backend && exec uv run --frozen --offline python -m sim --no-hud "$@") >> "$LOG" 2>&1 &
  pids+=($!)
  for i in {1..40}; do
    curl -sf -m 1 -o /dev/null $SIM_URL/state && break
    kill -0 $pids[-1] 2>/dev/null || break
    sleep 0.5
  done
  if ! curl -sf -m 1 -o /dev/null $SIM_URL/state; then
    echo "The simulator did not start. See $LOG." >&2
    exit 1
  fi
else
  WIFI_DEVICE=$(networksetup -listallhardwareports | awk '/Wi-Fi/ {getline; print $2}')
  DRONE_WIFI=$(sed -n 's/^DRONE_WIFI=//p' backend/.env 2>/dev/null)
fi

if (( ! sim )) && ! on_drone_wifi; then
  if [[ -z $DRONE_WIFI ]]; then
    echo "Join the drone's Wi-Fi first, or add DRONE_WIFI=<its name> to backend/.env." >&2
    exit 1
  fi
  # Use whichever S20 network is actually in range: the drone's name can change.
  # A drone that was just powered on takes a while to start broadcasting.
  for i in {1..20}; do
    visible=("${(@f)$(swift tools/scan_drone_wifi.swift 2>/dev/null)}")
    [[ -n $visible[1] ]] && break
    (( i == 1 )) && echo "Waiting for a drone's Wi-Fi. If its lights are off or steady, turn it off and on."
    sleep 2
  done
  if [[ -n $visible[1] && ${visible[(Ie)$DRONE_WIFI]} -eq 0 ]]; then
    echo "$DRONE_WIFI is not in range; using $visible[1] instead."
    DRONE_WIFI=$visible[1]
  fi
  echo "Joining $DRONE_WIFI. This Mac has no internet until you quit."
  before=$(ipconfig getifaddr "$WIFI_DEVICE")
  networksetup -setairportnetwork "$WIFI_DEVICE" "$DRONE_WIFI"
  for i in {1..30}; do
    now=$(ipconfig getifaddr "$WIFI_DEVICE")
    [[ -n $now && $now != $before && -n $(wifi_router) ]] && break
    sleep 1
  done
  if [[ -z $(wifi_router) || $(ipconfig getifaddr "$WIFI_DEVICE") == $before ]]; then
    echo "Could not join $DRONE_WIFI. Is the drone powered on?" >&2
    exit 1
  fi
  joined_wifi=1
fi
if (( ! sim )); then
  export DRONE_IP=$(wifi_router)
  echo "Drone address: $DRONE_IP"
fi

# Everything is already installed, and the drone's Wi-Fi has no internet, so
# nothing below may touch the network.
echo "Checking the link. The motors stay off."
(cd backend && uv run --frozen --offline python preflight.py 2>&1) | grep -v "Start command sent" || exit 1

# Without the shutdown limit, an open browser tab's video stream keeps the
# backend alive, still sending stick packets, after you quit.
(cd backend && exec uv run --frozen --offline uvicorn web_server:app --timeout-graceful-shutdown 2) >> "$LOG" 2>&1 &
pids+=($!)
(cd frontend && exec node_modules/.bin/vite --strictPort) >> "$LOG" 2>&1 &
pids+=($!)

for i in {1..40}; do
  curl -sf -m 1 -o /dev/null http://localhost:8000/capabilities \
    && curl -sf -m 1 -o /dev/null http://localhost:5173 && break
  sleep 0.5
done
if ! curl -sf -m 1 -o /dev/null http://localhost:5173; then
  echo "The web client did not start. See $LOG." >&2
  exit 1
fi
if command -v open >/dev/null; then
  open http://localhost:5173
else
  xdg-open http://localhost:5173 >/dev/null 2>&1
fi

cat <<'HELP'

Ready: http://localhost:5173

  W / S         forward / back
  A / D         left / right
  Up / Down     climb / descend
  Left / Right  turn
  Esc           E-STOP: motors cut at once, the drone drops

  Takeoff and Land send the same toggle: press once and wait. Land on the
  ground takes off, and Takeoff in the air lands.

Land first, then press Ctrl-C here to quit.
HELP
(( sim )) && echo "\nSimulator: state, scene, wind and power cycle are in the panel at the top right."

while kill -0 $pids 2>/dev/null; do
  sleep 1
done
echo "A server stopped unexpectedly. See $LOG." >&2
exit 1
