"""Printable floor tags and the swarm file section that goes with them: `python -m zwarm markers`.

These are tagStandard41h12, the only family the tracker reads. It can't read ArUco or QR codes.
One tag per Letter page at 300 dpi, so a reprint is a spare for a sheet that got torn or lost.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from swarm.tag41h12 import TOTAL_CELLS
from swarm.tracking import anchors_section, tag_grid, tag_image

DPI = 300
PAGE = (2550, 3300)  # Letter at 300 dpi, width by height
QUIET_CELLS = 2  # white paper around the tag; the detector needs it


def _with_dpi(png: bytes, dpi: int = DPI) -> bytes:
    """Mark a PNG with its print size. OpenCV leaves this out, and without it a printer scales to fit."""
    per_metre = round(dpi / 0.0254)
    payload = b"pHYs" + struct.pack(">IIB", per_metre, per_metre, 1)
    chunk = struct.pack(">I", 9) + payload + struct.pack(">I", zlib.crc32(payload))
    end_of_header = 8 + 4 + 4 + 13 + 4
    return png[:end_of_header] + chunk + png[end_of_header:]


def page(tag: int, width_m: float, where: tuple[float, float] | None = None) -> tuple[np.ndarray, float]:
    """One printable page, and the whole tag's real width in metres once printed at 300 dpi."""
    cell = int(round(width_m / 0.0254 * DPI / TOTAL_CELLS))
    image = tag_image(tag, cell, QUIET_CELLS)
    if image.shape[0] > PAGE[0]:
        raise ValueError(f"a {width_m * 1000:.0f} mm tag doesn't fit on a Letter page.")
    sheet = np.full((PAGE[1], PAGE[0]), 255, np.uint8)
    top = (PAGE[1] - image.shape[0]) // 2
    left = (PAGE[0] - image.shape[1]) // 2
    sheet[top:top + image.shape[0], left:left + image.shape[1]] = image
    real_m = cell * TOTAL_CELLS / DPI * 0.0254
    below = top + image.shape[0] + 70
    text = [f"tag {tag}   tagStandard41h12" + (f"   goes at ({where[0]:g}, {where[1]:g})" if where else ""),
            f"print at 100%, no fit to page. Whole tag {real_m * 1000:.1f} mm, white square {real_m * 5 / 9 * 1000:.1f} mm.",
            "The top of the tag is the side away from this text. Measure the white square after printing."]
    for i, line in enumerate(text):
        cv2.putText(sheet, line, (left, below + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 90, 3, cv2.LINE_AA)
    return sheet, real_m


def write(out_dir, width_m: float = 0.138, origin=(-2.0, 2.0), spacing: float = 1.0, columns: int = 5, rows: int = 5, start_id: int = 0,
          only: list[int] | None = None) -> tuple[list[Path], Path, float]:
    """The PNGs and anchors.json. `only` limits which pages are printed; the section still lists the whole grid."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tags = tag_grid(tuple(origin), spacing, columns, rows, start_id)
    real_m = width_m
    pages = []
    for entry in tags:
        if only is not None and entry["tag"] not in only:
            continue
        sheet, real_m = page(entry["tag"], width_m, (entry["x"], entry["y"]))
        ok, png = cv2.imencode(".png", sheet)
        if not ok:
            raise RuntimeError(f"couldn't encode tag {entry['tag']}")
        path = out / f"tag-{entry['tag']:02d}.png"
        path.write_bytes(_with_dpi(png.tobytes()))
        pages.append(path)
    section = out / "anchors.json"
    section.write_text(json.dumps({"anchors": anchors_section(round(real_m * 5 / 9, 5), tags)}, indent=1))
    return pages, section, real_m
