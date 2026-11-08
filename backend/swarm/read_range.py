"""How far ahead a floor tag reads, through the simulator's camera at the S20's video size and
compression, at the angle a tilted camera really sees: `python -m swarm.read_range`.

Simulator results. Real sharpness, exposure and lens are unmeasured, so a real frame may read less far.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from sim.scene import Box, Scene
from swarm.sim_anchors import FloorCamera, VideoConditions
from swarm.tracking import Anchor, AnchorReader, Lens, solve_fix

TILT = 45.0
LEVEL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
ROOM = Box((-1.0, -8.0, 0.0), (12.0, 8.0, 3.0), "wall")
AHEAD = (0.3, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
HEIGHT = 1.0


def trial(white: float, ahead: float, video: VideoConditions, blur_px: float, decimate: float, count: int, rng):
    found, errors_xy, errors_z, elapsed = 0, [], [], 0.0
    lens = Lens.from_fov()
    scene = Scene(name="floor", bounds=ROOM)
    for i in range(count):
        anchor = Anchor("tag:7", ahead, float(rng.uniform(-0.15, 0.15)) * ahead, white, float(rng.uniform(-180, 180)))
        anchors = {anchor.id: anchor}
        image = FloorCamera(scene, anchors, TILT, video, seed=i).render(0.0, 0.0, HEIGHT, LEVEL, (blur_px / video.exposure_s, 0.0))
        reader = AnchorReader(anchors, decimate)
        started = time.perf_counter()
        seen = reader.find(image)
        elapsed += time.perf_counter() - started
        if seen:
            found += 1
            fix = solve_fix(seen, lens, math.radians(TILT), 0.0, 0.0)
            if fix is not None:
                errors_xy.append(math.hypot(fix.p[0], fix.p[1]))
                errors_z.append(abs(fix.p[2] + 0.03 - HEIGHT))
    return found, (float(np.median(errors_xy)) if errors_xy else math.nan), (float(np.median(errors_z)) if errors_z else math.nan), elapsed / count * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--tag-width", type=float, default=0.138, help="whole printed tag, metres")
    args = parser.parse_args()
    rng = np.random.default_rng(3)
    white = args.tag_width * 5 / 9
    print(f"tagStandard41h12, {args.tag_width * 100:.0f} cm wide ({white * 100:.2f} cm white square), {HEIGHT} m up, 45 deg down, "
          "1184 x 656 at about 34 KB.\nFrames read out of each count, by metres ahead on the floor; then horizontal error (cm) on each read.\n")
    print(f"{'video':<34}{'':<3}" + "".join(f"{d:>7.2f}" for d in AHEAD) + f"{'ms':>6}")
    for label, scale, sigma, blur, decimate in (("sharp, still", 1.0, 0.0, 0, 1.0), ("640x360 detail, still", 0.54, 0.0, 0, 1.0),
                                                ("soft (sigma 2), still", 1.0, 2.0, 0, 1.0), ("640x360, 5 px blur (0.5 m/s, 1/60)", 0.54, 0.0, 5, 1.0),
                                                ("640x360, 11 px blur (dim or turn)", 0.54, 0.0, 11, 1.0), ("sharp, still, decimate 2", 1.0, 0.0, 0, 2.0),
                                                ("640x360, still, decimate 2", 0.54, 0.0, 0, 2.0), ("soft (sigma 2), decimate 2", 1.0, 2.0, 0, 2.0)):
        video = VideoConditions(render_scale=scale, blur_sigma=sigma)
        rows = [trial(white, d, video, blur, decimate, args.count, rng) for d in AHEAD]
        print(f"{label:<34}{'n':<3}" + "".join(f"{r[0]:>7}" for r in rows) + f"{float(np.mean([r[3] for r in rows])):>6.1f}")
        print(f"{'':<34}{'cm':<3}" + "".join(f"{r[1] * 100:>7.1f}" if r[0] else f"{'-':>7}" for r in rows))


if __name__ == "__main__":
    main()
