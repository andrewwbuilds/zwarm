# Raspberry Pi drone gateways

Verified on three Raspberry Pi 4s running Raspberry Pi OS Lite 64-bit (Debian 13
Trixie), Python 3.13.5, with one independent drone connection per Pi.

## Current installation

- Pi 1: `molt`, USB address `10.12.194.1/28`, Mac address `10.12.194.2/28`.
- Pi 2: `clawdbot-pi`, USB address `10.12.195.1/28`, Mac address `10.12.195.2/28`.
- Pi 3: `zwarm-pi-03`, USB address `10.12.196.1/28`, Mac address `10.12.196.2/28`.
- Mac SSH aliases: `zwarm-pi-01`, `zwarm-pi-02` and `zwarm-pi-03`, login user `andrew`.
- Repository on each Pi: `/home/andrew/zwarm`, backend environment: `backend/.venv`.
- Each backend listens on `127.0.0.1:8000` on its Pi; use an SSH tunnel from the Mac.
- `zwarm-backend.service` is installed for manual operation. It is not started or
  enabled at boot by the installer. A drone Wi-Fi connection is still required.

USB management works independently of the Mac's current internet connection.
These links do not give the Pis internet access. Installing/updating packages
requires an internet route or a temporary proxy carried over SSH. No proxy
settings are stored in the backend service.

The current Mac has static USB addresses with no router or DNS configured. The
Pis have distinct, fixed USB MAC addresses, `USB Gadget (shared)` profiles with
the addresses above, and the automatic ICS watcher disabled. A local dnsmasq
snippet suppresses router and DNS advertisements. Wi-Fi is available for drones.
On a different Mac, recreate the USB IP settings and authorize its SSH public key.

Pis 1 and 3 also have a `ZWARM Ethernet direct` profile using IPv4/IPv6 link-local
addresses. On Pi 1 the original Ethernet profile is retained. To use Ethernet with a
DHCP router again, explicitly activate that original profile. Link-local IPv6
addresses change between boots; find the current one from the Mac with
`ping6 -c 2 -I <ethernet-interface> ff02::1`.

All three Pis currently join the home Wi-Fi on `wlan0` for internet access
(`netplan-wlan0-...` on Pis 1 and 2, `home-wifi` on Pi 3). Connecting a drone takes
over `wlan0`, so the Pi loses internet while it is on the drone network. While a Pi
is on the home Wi-Fi, you can SSH to its home address even when USB is down.

## Install or update the backend

With internet available on the Pi, install the system libraries and
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0t64
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/andrewwbuilds/zwarm.git ~/zwarm
cd ~/zwarm
bash tools/pi/install-backend.sh
backend/.venv/bin/python tools/pi/check-backend.py
```

The installer uses the system Python (3.13 or newer required), creates the locked
virtual environment, preserves an existing `backend/.env`, and installs a systemd
service for the user running the installer. It backs up an existing service file.
It does not start, enable, or restart the service. The simulator check verifies
the actual FastAPI capabilities endpoint and a complete JPEG through the video
pipeline, using only loopback sockets and temporary ports; it cleans up afterwards.

For subsequent updates, stop the backend after landing, run `git pull --ff-only`,
then rerun the installer and simulator check. Existing environments can be synced
offline if all required packages are already cached.

## Connect a drone

Power on one drone at a time to identify its SSID. On its assigned Pi:

```sh
sudo nmcli device wifi list ifname wlan0
sudo nmcli --ask device wifi connect 'ACTUAL-DRONE-SSID' ifname wlan0 name drone-link
sudo nmcli connection modify drone-link \
  ipv4.never-default yes ipv4.ignore-auto-dns yes \
  ipv6.never-default yes ipv6.ignore-auto-dns yes
sudo nmcli connection up drone-link
```

Check the assigned network and verify that the drone IP is routed over `wlan0`:

```sh
ip -4 address show wlan0
ip -4 route get 172.16.10.1
cd ~/zwarm/backend
.venv/bin/python preflight.py
```

Current pairing (HITURBO S20, open networks). Each Pi saves the link as `drone-link` with
autoconnect priority 100 and unlimited retries. If a drone powers off, the Pi falls back to
home Wi-Fi and stays there: NetworkManager does not leave a working connection when the drone
returns. Bring the link back by hand with `sudo nmcli connection up drone-link` on that Pi:

| Pi | Drone | SSID |
|---|---|---|
| `zwarm-pi-01` | 1 | `HITURBO_S20-8be411` |
| `zwarm-pi-02` | 2 | `HITURBO_S20-d06410` |
| `zwarm-pi-03` | 3 | `HITURBO_S20-de9b28` |

Every drone hands out `172.16.10.2` to its client, so all three Pis show the same `wlan0`
address on separate networks. An S20 blinks its lights while no controller is sending
packets, which is normal after preflight ends.

The default `backend/.env` selects S20/S2x at `172.16.10.1` with plugins disabled.
Change the type/IP if the actual airframe differs. Stop any running backend before
preflight because both receive video on the same UDP port. Preflight sends centered
controls and requests video without issuing a takeoff command.

## Operate from the Mac

After preflight passes, confirm each Pi is on its drone and start the backends:

```sh
for n in 1 2 3; do ssh zwarm-pi-0$n 'nmcli -g GENERAL.CONNECTION device show wlan0'; done
for n in 1 2 3; do ssh zwarm-pi-0$n 'sudo systemctl start zwarm-backend'; done
```

Open one tunnel per Pi (`localhost:8001` to Pi 1, `8002` to Pi 2, `8003` to Pi 3) and keep
the terminal open:

```sh
tools/pi/fleet-tunnels.sh
```

In a second terminal, run the frontend:

```sh
cd frontend
npm ci
npm run dev
```

- `http://localhost:5173/fleet` shows all three video feeds and backend status. It is view
  only: it never opens the control websocket, so it cannot move a drone.
- `http://localhost:5173/?backend=8001` (or `8002`, `8003`) is the console for one drone.
  Each tile on the fleet page has an "open console" link. The tab title names the drone.
- Only the focused console window sends stick input. An unfocused console centres its
  sticks once and goes quiet, so its drone holds position. Click a window before flying it.
- Without `?backend=`, the console uses `VITE_BACKEND_URL`, then `localhost:8000`, so the
  single-tunnel setup (`ssh -N -L 8000:127.0.0.1:8000 zwarm-pi-01`) still works.

Fleet coordination and shared position feedback are separate work.

Inspect and stop the backend with:

```sh
ssh zwarm-pi-01 'sudo systemctl status zwarm-backend --no-pager'
ssh zwarm-pi-01 'sudo journalctl -u zwarm-backend -n 50 --no-pager'
ssh zwarm-pi-01 'sudo systemctl stop zwarm-backend'
```

Land before stopping the backend or shutting down the Pi. On S20, takeoff and land
share a toggle; a repeated command can reverse the intended action.

## USB recovery and shutdown

All three Pis have `zwarm-usb-startup-recovery.service`. It checks for the
observed receive-path stall at startup and attempts at most two `g_ether` reloads.
It is a workaround, not a fix for the underlying driver. A reboot test passed on
Pis 1 and 2; ongoing reliability beyond that test has not been established.

The link also stalls after the Mac sleeps: macOS marks every gadget interface
inactive and they stay inactive after wake, while each Pi still reports carrier.
`zwarm-usb-watchdog.timer` covers this. Every minute it pings the Mac's `.2`
address over `usb0`, and if there is no reply and `rx_packets` did not move, it
reloads `g_ether`. Reloads are limited to one every three minutes, because a
sleeping Mac looks the same as a stall. Expect USB back within about four minutes
of the Mac waking. Install or update it on a Pi with:

```sh
bash ~/zwarm/tools/pi/usb-watchdog/install.sh
sudo journalctl -u zwarm-usb-watchdog -n 20 --no-pager
```

The forced reload path and rate limit were tested on Pi 3. Recovery after an
actual Mac sleep has not yet been observed with the timer installed. Upstream
reports describe similar symptoms: [issue 27](https://github.com/raspberrypi/rpi-usb-gadget/issues/27)
and [proposed recovery](https://github.com/raspberrypi/rpi-usb-gadget/pull/31).

If USB fails, use Ethernet or home Wi-Fi to inspect `ip -s link show usb0` and the service logs.
With no backend running, `sudo modprobe -r g_ether; sudo modprobe g_ether` reloads
the USB network driver. This disconnects USB temporarily. Ethernet, home Wi-Fi or a
local console is needed to issue it when USB is already unreachable.

Before disconnecting USB power:

```sh
ssh zwarm-pi-01 'sudo poweroff'
ssh zwarm-pi-02 'sudo poweroff'
ssh zwarm-pi-03 'sudo poweroff'
```

## Writing a new card on macOS

Raspberry Pi Imager v2.0.11 in `--cli` mode hung at "Unmounting drive" on this
Mac, with or without administrator rights. Writing `/dev/rdiskN` with `dd` as root
also fails (`Operation not permitted`) unless the terminal app has Full Disk Access.
What worked for Pi 3:

1. Decompress the image and check its SHA-256 against the release's `extract_sha256`.
2. Open the raw disk through `/usr/libexec/authopen -stdoutpipe -o 2 /dev/rdiskN`,
   which shows a system authorization prompt and passes back a file descriptor,
   then write the image in 4 MiB blocks and read it back to compare the hash.
3. Copy `user-data` and `network-config` onto the `bootfs` partition with `cat`
   (plain `cp` leaves `._` AppleDouble files), keep the image's `meta-data`, eject.

On first boot, set the Wi-Fi country (`sudo raspi-config nonint do_wifi_country US`)
before `wlan0` will come up, and let NTP sync the clock before running `apt-get`.
On the Mac, give the new gadget interface a manual `.2` address with no router.

Private keys, passwords, local environment files, and recordings are not stored
in this repository. Configuration backups remain on the Mac and on each Pi.
