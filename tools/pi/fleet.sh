#!/usr/bin/env bash
# One command for the three-drone setup. Run on the Mac with the Pis on USB.
#
#   tools/pi/fleet.sh up [--no-preflight]
#       Join each Pi to its drone, run preflight, start the backends, open the tunnels
#       (localhost:8001-8003) and the app, then open the fleet monitor. Stays in the
#       foreground and keeps the Mac awake. Ctrl-C closes the tunnels and the app; the
#       backends keep running and a disconnected console centres the sticks.
#       A running backend is left running (no preflight), so re-running is safe.
#
#   tools/pi/fleet.sh status
#       Drone link, backend and tunnel for each Pi.
#
#   tools/pi/fleet.sh down [--yes]
#       Stop the backends and close the tunnels. Land every drone first: a stopped
#       backend sends no more control packets.
set -euo pipefail

PIS=(zwarm-pi-01 zwarm-pi-02 zwarm-pi-03)
PORTS=(8001 8002 8003)
APP_PORT=5173
DRONE_IP=172.16.10.1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SSH=(ssh -n -o ConnectTimeout=5 -o BatchMode=yes)
CURL=(curl -s --noproxy '*' -m 3)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
remote() { "${SSH[@]}" "$1" "$2"; }
port_up() { "${CURL[@]}" -o /dev/null "http://localhost:$1/capabilities"; }
app_up() { "${CURL[@]}" -o /dev/null "http://localhost:$APP_PORT/"; }

# Join wlan0 to the saved drone-link profile unless it is already on it and the drone answers.
connect_drone() {
  remote "$1" '
    if [ "$(nmcli -g GENERAL.CONNECTION device show wlan0)" = drone-link ] &&
       ping -c 2 -W 2 '"$DRONE_IP"' >/dev/null 2>&1; then
      echo "drone link: already up ($(nmcli -g 802-11-wireless.ssid connection show drone-link))"
      exit 0
    fi
    ssid=$(nmcli -g 802-11-wireless.ssid connection show drone-link) || { echo "no drone-link profile"; exit 1; }
    seen=no
    for _ in 1 2 3 4 5 6; do
      sudo nmcli device wifi rescan ifname wlan0 >/dev/null 2>&1 || true
      sleep 3
      if nmcli -g SSID device wifi list ifname wlan0 | grep -qxF "$ssid"; then seen=yes; break; fi
    done
    [ $seen = yes ] || { echo "drone $ssid is not broadcasting. Is it on?"; exit 1; }
    sudo nmcli connection up drone-link >/dev/null || { echo "could not join $ssid"; exit 1; }
    ping -c 3 -W 2 '"$DRONE_IP"' >/dev/null || { echo "joined $ssid but the drone does not answer"; exit 1; }
    echo "drone link: joined $ssid"
  '
}

preflight() {
  local out
  out=$(remote "$1" 'cd ~/zwarm/backend && timeout 120 .venv/bin/python preflight.py 2>&1 | tail -4')
  printf '%s\n' "$out" | grep -E '^(Video|Frames):' | sed 's/^/preflight: /'
  printf '%s\n' "$out" | grep -q '^OK: link is good' || { printf '%s\n' "$out"; echo "preflight FAILED"; return 1; }
  echo "preflight: OK, motors not started"
}

start_backend() {
  remote "$1" '
    sudo systemctl start zwarm-backend
    for _ in $(seq 20); do
      curl -s -m 2 -o /dev/null 127.0.0.1:8000/capabilities && { echo "backend: running"; exit 0; }
      sleep 1
    done
    echo "backend: did not answer after 20 s"; sudo journalctl -u zwarm-backend -n 15 --no-pager; exit 1'
}

prepare_pi() {
  local host=$1 run_preflight=$2
  remote "$host" true || { echo "unreachable over USB"; return 1; }
  if remote "$host" 'systemctl is-active -q zwarm-backend'; then
    echo "backend: already running, drone link left as is"
    return 0
  fi
  connect_drone "$host" || return 1
  if [ "$run_preflight" = yes ]; then preflight "$host" || return 1; fi
  start_backend "$host" || return 1
}

open_tunnel() {
  local host=$1 port=$2
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
    -L "$port:127.0.0.1:8000" "$host" &
  CHILDREN+=("$!")
}

cmd_up() {
  local run_preflight=yes
  [ "${1:-}" = --no-preflight ] && run_preflight=no
  [ "$(uname -s)" = Darwin ] || die "Run this on the Mac."

  local tmp; tmp=$(mktemp -d)
  echo "Preparing ${#PIS[@]} Pis in parallel (drone link, preflight, backend)..."
  for i in "${!PIS[@]}"; do
    ( set +e; prepare_pi "${PIS[$i]}" "$run_preflight" >"$tmp/$i.log" 2>&1; echo $? >"$tmp/$i.rc" ) &
  done
  wait

  local ready=()
  for i in "${!PIS[@]}"; do
    echo "== ${PIS[$i]} (drone $((i + 1)))"
    sed 's/^/   /' "$tmp/$i.log"
    if [ "$(cat "$tmp/$i.rc")" = 0 ]; then ready+=("$i"); else echo "   NOT READY"; fi
  done
  rm -rf "$tmp"
  [ ${#ready[@]} -gt 0 ] || die "No Pi is ready."

  CHILDREN=()
  trap 'for p in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill "$p" 2>/dev/null || true; done' EXIT
  trap 'exit 130' INT TERM

  for i in "${ready[@]}"; do
    port_up "${PORTS[$i]}" || open_tunnel "${PIS[$i]}" "${PORTS[$i]}"
  done
  for i in "${ready[@]}"; do
    for _ in $(seq 15); do port_up "${PORTS[$i]}" && break; sleep 1; done
    port_up "${PORTS[$i]}" && echo "tunnel: ${PIS[$i]} -> localhost:${PORTS[$i]}" \
      || echo "tunnel: localhost:${PORTS[$i]} NOT answering"
  done

  if app_up; then
    echo "app: already running on localhost:$APP_PORT"
  else
    [ -d "$ROOT/frontend/node_modules" ] || (cd "$ROOT/frontend" && npm ci --no-audit --no-fund >/dev/null)
    (cd "$ROOT/frontend" && exec npx vite --port "$APP_PORT" --strictPort >/tmp/zwarm-fleet-app.log 2>&1) &
    CHILDREN+=("$!")
    for _ in $(seq 30); do app_up && break; sleep 1; done
    app_up || die "app did not start; see /tmp/zwarm-fleet-app.log"
    echo "app: running on localhost:$APP_PORT"
  fi

  open "http://localhost:$APP_PORT/fleet"
  echo
  echo "${#ready[@]}/${#PIS[@]} drones ready. Fleet monitor: http://localhost:$APP_PORT/fleet"
  echo "Consoles: http://localhost:$APP_PORT/?backend=8001 (8002, 8003)"

  if [ ${#CHILDREN[@]} -eq 0 ]; then
    echo "Tunnels and app were already running in another terminal; nothing to hold open."
    return 0
  fi
  echo "Keep this terminal open. Ctrl-C closes the tunnels and app; backends keep running."
  echo "After landing: tools/pi/fleet.sh down"
  caffeinate -dis -w $$ &
  CHILDREN+=("$!")
  wait
}

cmd_status() {
  for i in "${!PIS[@]}"; do
    local host=${PIS[$i]} port=${PORTS[$i]} line
    line=$(remote "$host" '
      printf "wlan0=%s drone=%s backend=%s" \
        "$(nmcli -g GENERAL.CONNECTION device show wlan0)" \
        "$(ping -c 2 -W 2 '"$DRONE_IP"' >/dev/null 2>&1 && echo answering || echo no-reply)" \
        "$(systemctl is-active zwarm-backend)"' 2>/dev/null) || line="unreachable over USB"
    printf '%s  %s tunnel:%s=%s\n' "$host" "$line" "$port" "$(port_up "$port" && echo up || echo down)"
  done
  printf 'app localhost:%s=%s\n' "$APP_PORT" "$(app_up && echo up || echo down)"
}

cmd_down() {
  if [ "${1:-}" != --yes ]; then
    printf 'Stopping the backends cuts control of every drone. Are all drones landed? [y/N] '
    read -r answer
    [ "$answer" = y ] || [ "$answer" = Y ] || die "Cancelled."
  fi
  for host in "${PIS[@]}"; do
    remote "$host" 'sudo systemctl stop zwarm-backend; echo "$(hostname): backend $(systemctl is-active zwarm-backend)"' \
      || echo "$host: unreachable"
  done
  pkill -f 'ssh -N .*-L 800[1-3]:127.0.0.1:8000' && echo "tunnels closed" || true
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  status) cmd_status ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
