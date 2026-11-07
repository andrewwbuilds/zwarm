"""Verify the installed API and JPEG stream against a loopback-only simulated S20.

Run with backend/.venv/bin/python tools/pi/check-backend.py from the repo root.
No real drone connection or takeoff command is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from sim.server import Server


def unused_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    api_port = unused_port(socket.SOCK_STREAM)
    video_port = unused_port(socket.SOCK_DGRAM)
    simulator = Server(
        host="127.0.0.1", control_port=0, tcp_port=0,
        video_source_port=0, client_video_port=video_port, api_port=0,
        hud=False, render_scale=0.12, log=lambda _: None,
    )
    process = None
    # Ignore inherited proxies for local health requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile(mode="w+") as log:
        try:
            simulator.start()
            env = {
                **os.environ,
                "DRONE_TYPE": "s2x", "DRONE_IP": "127.0.0.1",
                "CONTROL_PORT": str(simulator.control_port),
                "VIDEO_PORT": str(video_port), "PLUGINS_ENABLED": "false",
                "SIM_URL": f"http://127.0.0.1:{simulator.api_port}",
            }
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "web_server:app",
                 "--host", "127.0.0.1", "--port", str(api_port),
                 "--timeout-graceful-shutdown", "2"],
                cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            url = f"http://127.0.0.1:{api_port}"
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Backend exited before becoming ready")
                try:
                    with opener.open(url + "/capabilities", timeout=2) as reply:
                        capabilities = json.load(reply)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Backend did not become ready within 45 seconds")
                    time.sleep(0.25)
            if capabilities["drone_type"] != "s2x":
                raise RuntimeError(f"Unexpected capabilities: {capabilities}")
            with opener.open(url + "/mjpeg", timeout=20) as stream:
                data = bytearray()
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and len(data) < 1024 * 1024:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    data.extend(chunk)
                    start = data.find(b"\xff\xd8")
                    if start >= 0 and data.find(b"\xff\xd9", start + 2) >= 0:
                        print("PASS: backend API and complete simulated JPEG received over loopback")
                        return
            raise RuntimeError("No complete simulated JPEG received")
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-12000:], file=sys.stderr)
            raise
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            simulator.stop()


if __name__ == "__main__":
    main()
