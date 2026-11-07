#!/usr/bin/env bash
# Install the locked backend and a manually started systemd service on a Pi.
set -euo pipefail

if [[ $(uname -s) != Linux || $(id -u) == 0 ]]; then
  echo "Run this as your regular user on Raspberry Pi OS." >&2
  exit 1
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
backend="$repo/backend"
uv_bin=${UV_BIN:-$HOME/.local/bin/uv}
if [[ ! -x "$uv_bin" ]]; then
  echo "Install uv first, or set UV_BIN to its executable path." >&2
  exit 1
fi

cd "$backend"
UV_PYTHON_DOWNLOADS=never "$uv_bin" sync --locked --python /usr/bin/python3
.venv/bin/python -c 'import cv2, fastapi, numpy, scipy; print("Backend dependencies OK")'

if [[ ! -e .env ]]; then
  (umask 077; cat > .env <<'EOF'
DRONE_TYPE=s2x
DRONE_IP=172.16.10.1
PLUGINS_ENABLED=false
EOF
  )
fi

unit_dir=$(mktemp -d)
unit="$unit_dir/zwarm-backend.service"
trap 'rm -f "$unit"; rmdir "$unit_dir"' EXIT
python3 - "$backend" "$(id -un)" "$(id -gn)" "$unit" <<'PY'
import json
import sys
from pathlib import Path

backend, user, group, destination = sys.argv[1:]
if any(char in backend for char in '\r\n'):
    raise SystemExit('The repository path cannot contain line breaks')
def quote(value):
    return json.dumps(value.replace('%', '%%'), ensure_ascii=False)

Path(destination).write_text(f'''[Unit]
Description=ZWARM drone backend
After=network.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={backend.replace('%', '%%')}
Environment=PYTHONUNBUFFERED=1
ExecStart={quote(backend + '/.venv/bin/python')} -m uvicorn web_server:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 2
Restart=no
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
''')
PY

sudo systemd-analyze verify "$unit"
destination=/etc/systemd/system/zwarm-backend.service
if sudo test -f "$destination"; then
  sudo cp -a "$destination" "$destination.backup-$(date +%Y%m%d-%H%M%S)"
fi
sudo install -m 644 "$unit" "$destination"
sudo systemctl daemon-reload
echo "Installed zwarm-backend.service; this installer does not start or enable it."
echo "After joining the drone Wi-Fi and passing preflight: sudo systemctl start zwarm-backend"
