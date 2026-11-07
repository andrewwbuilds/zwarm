#!/usr/bin/env bash
# Install and start the drone Wi-Fi keeper timer on a Pi.
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
  echo "Run this on the Raspberry Pi." >&2
  exit 1
fi

here=$(cd "$(dirname "$0")" && pwd)
sudo install -m 755 "$here/zwarm-drone-link-keeper" /usr/local/sbin/zwarm-drone-link-keeper
sudo install -m 644 "$here/zwarm-drone-link-keeper.service" /etc/systemd/system/
sudo install -m 644 "$here/zwarm-drone-link-keeper.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zwarm-drone-link-keeper.timer
systemctl list-timers zwarm-drone-link-keeper.timer --no-pager
