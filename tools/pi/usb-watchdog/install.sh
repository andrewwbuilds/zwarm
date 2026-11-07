#!/usr/bin/env bash
# Install and start the USB receive-path watchdog timer on a Pi.
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
  echo "Run this on the Raspberry Pi." >&2
  exit 1
fi

here=$(cd "$(dirname "$0")" && pwd)
sudo install -m 755 "$here/zwarm-usb-watchdog" /usr/local/sbin/zwarm-usb-watchdog
sudo install -m 644 "$here/zwarm-usb-watchdog.service" /etc/systemd/system/
sudo install -m 644 "$here/zwarm-usb-watchdog.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zwarm-usb-watchdog.timer
systemctl list-timers zwarm-usb-watchdog.timer --no-pager
