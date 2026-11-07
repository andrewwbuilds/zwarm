"""Record every frame the drone's camera sends. Choosing which to use happens later (mapping.curate)."""

from __future__ import annotations

import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from mapping.curate import quiet_jpeg_warnings, raw_name, sharpness
from slam.live import jpeg_frames


def capture(url: str, out_dir: Path, seconds: float | None = None, window: bool = True) -> int:
    """Save every frame of the MJPEG stream at url, byte for byte, to out_dir/raw. Returns how many were saved.

    Nothing is thrown away here, so a bad moment cannot cost a good view.
    """
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = time.time()
    count = len(list(raw.glob("*.jpg")))
    recent: deque[float] = deque(maxlen=60)
    print(f"Recording every frame to {raw}. Fly slowly and circle each area. Press q in the window or Ctrl-C to finish.")
    try:
        with urllib.request.urlopen(url, timeout=10) as stream:
            for jpeg in jpeg_frames(stream):
                t = time.time() - started
                # The camera's own JPEG bytes, untouched: re-encoding would add another round of compression.
                (raw / raw_name(count, int(t * 1000))).write_bytes(jpeg)
                count += 1
                if window or count % 17 == 0:
                    with quiet_jpeg_warnings():
                        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    recent.append(sharpness(image))
                    soft = len(recent) > 10 and recent[-1] < 0.6 * float(np.median(recent))
                    if window:
                        view = cv2.resize(image, (640, 355))
                        color = (0, 140, 255) if soft else (0, 200, 0)
                        cv2.rectangle(view, (0, 0), (639, 354), color, 4)
                        cv2.putText(view, f"{count} frames  {t:4.0f}s" + ("  blurry" if soft else ""), (12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.imshow("Mapping capture", view)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        print(f"\r{count} frames, {t:.0f}s" + ("  (blurry right now)" if soft else "").ljust(30), end="", flush=True)
                if seconds and t > seconds:
                    break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    print(f"\nDone: {count} frames saved. Next: uv run python -m mapping build {out_dir.name}")
    return count
