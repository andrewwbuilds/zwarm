#!/usr/bin/env bash
# Open one SSH tunnel per Pi so the Mac reaches every backend at once:
#   localhost:8001 -> zwarm-pi-01, 8002 -> zwarm-pi-02, 8003 -> zwarm-pi-03
# The tunnels stay up until Ctrl-C. This does not start or stop any backend.
set -euo pipefail

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for n in 1 2 3; do
  host="zwarm-pi-0$n"
  port="800$n"
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=3 \
    -L "$port:127.0.0.1:8000" "$host" &
  pids+=("$!")
  echo "$host -> localhost:$port"
done

echo "Tunnels open. Fleet monitor: http://localhost:5173/fleet  (Ctrl-C to close)"
wait
