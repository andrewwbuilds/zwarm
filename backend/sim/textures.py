"""Procedural surface textures, generated once. Each is 256x256 RGB with its tile size in metres."""

from __future__ import annotations

import cv2
import numpy as np

SIZE = 256


def noise(rng, cells: int, rows: int | None = None) -> np.ndarray:
    """Smooth noise that wraps at the edges, so tiles join without seams."""
    cells = max(2, cells)
    rows = cells if rows is None else max(2, rows)
    grid = rng.random((rows, cells)).astype(np.float32)
    big = cv2.resize(np.tile(grid, (3, 3)), (SIZE * 3, SIZE * 3), interpolation=cv2.INTER_CUBIC)
    return big[SIZE:2 * SIZE, SIZE:2 * SIZE] - 0.5


def fbm(rng, base: int = 4, octaves: int = 5) -> np.ndarray:
    out = np.zeros((SIZE, SIZE), np.float32)
    amp, cells, total = 1.0, base, 0.0
    for _ in range(octaves):
        out += amp * noise(rng, min(cells, SIZE))
        total += amp
        amp *= 0.5
        cells *= 2
    return out / total


def streaks(rng, cells: int, along: int) -> np.ndarray:
    """Noise stretched along u: wood grain, bark."""
    return noise(rng, cells // along, cells)


def tint(color, shade) -> np.ndarray:
    return np.clip(np.asarray(color, np.float32) * shade[..., None], 0.0, 1.0)


def grid():
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    return u, v


def woodfloor(rng):
    u, v = grid()
    planks = 8
    row = np.floor(v * planks).astype(int)
    tones = rng.uniform(0.8, 1.15, planks)
    offsets = rng.uniform(0, 1, planks)
    grain = streaks(rng, 32, 8)
    shade = tones[row] * (1 + 0.35 * grain) * (1 + 0.08 * fbm(rng, 8))
    seam = (np.abs(v * planks - np.round(v * planks)) < 0.03) | (np.abs(((u + offsets[row]) % 1.0) - 0.5) < 0.004)
    shade = np.where(seam, 0.45, shade)
    return tint((0.62, 0.42, 0.24), shade)


def paint(rng, color, amount=0.05):
    return tint(color, 1 + amount * fbm(rng, 8) + 0.6 * amount * noise(rng, 128))


def grass(rng):
    shade = 0.75 + 0.5 * fbm(rng, 16) + 0.5 * noise(rng, 256)
    hue = fbm(rng, 4)[..., None]
    base = np.asarray((0.24, 0.46, 0.14), np.float32) + hue * np.asarray((0.25, 0.1, 0.0), np.float32)
    return np.clip(base * shade[..., None], 0, 1)


def concrete(rng):
    shade = 1 + 0.18 * fbm(rng, 6) + 0.12 * noise(rng, 256)
    stains = np.clip(fbm(rng, 3) * 3, 0, 1) * 0.25
    return tint((0.56, 0.56, 0.54), shade - stains)


def brick(rng):
    u, v = grid()
    rows = 14
    row = np.floor(v * rows).astype(int)
    shift = (row % 2) * 0.5 / 4
    col = np.floor((u + shift) * 4).astype(int)
    tones = rng.uniform(0.75, 1.15, (rows, 5))
    shade = tones[row, col % 5] * (1 + 0.2 * noise(rng, 64))
    mortar = (np.abs(v * rows - np.round(v * rows)) < 0.08) | (np.abs((u + shift) * 4 - np.round((u + shift) * 4)) < 0.03)
    color = tint((0.62, 0.27, 0.18), shade)
    color[mortar] = (0.72, 0.70, 0.66)
    return color


def wood(rng):
    grain = streaks(rng, 24, 10)
    return tint((0.50, 0.32, 0.18), 1 + 0.4 * grain + 0.1 * fbm(rng, 8))


def fabric(rng, color):
    u, v = grid()
    weave = 0.06 * np.sin(u * SIZE * np.pi / 2) * np.sin(v * SIZE * np.pi / 2)
    return tint(color, 1 + weave + 0.15 * fbm(rng, 8) + 0.1 * noise(rng, 256))


def foliage(rng):
    shade = 0.55 + 0.9 * np.clip(fbm(rng, 8) + 0.5, 0, 1) + 0.3 * noise(rng, 128)
    return tint((0.16, 0.34, 0.12), shade)


def bark(rng):
    grain = np.rot90(streaks(rng, 32, 12))
    return tint((0.36, 0.25, 0.16), 1 + 0.6 * grain)


def tiles(rng, color, count, gap=0.04, gapcolor=(0.35, 0.35, 0.33)):
    u, v = grid()
    tone = rng.uniform(0.85, 1.12, (count, count))
    shade = tone[np.floor(v * count).astype(int), np.floor(u * count).astype(int)] * (1 + 0.12 * noise(rng, 128))
    out = tint(color, shade)
    edge = (np.abs(u * count - np.round(u * count)) < gap) | (np.abs(v * count - np.round(v * count)) < gap)
    out[edge] = gapcolor
    return out


def boards(rng, color, count):
    u, _ = grid()
    tone = rng.uniform(0.8, 1.15, count)[np.floor(u * count).astype(int)]
    out = tint(color, tone * (1 + 0.3 * np.rot90(streaks(rng, 32, 8))))
    out[np.abs(u * count - np.round(u * count)) < 0.05] *= 0.4
    return out


def stripes(rng, a, b, count):
    _, v = grid()
    band = (np.floor(v * count) % 2).astype(bool)
    out = np.where(band[..., None], np.asarray(a, np.float32), np.asarray(b, np.float32))
    return np.clip(out * (1 + 0.05 * noise(rng, 64))[..., None], 0, 1)


def frame(img, width, color):
    w = int(SIZE * width)
    img[:w], img[-w:], img[:, :w], img[:, -w:] = color, color, color, color
    return img


def rug(rng):
    u, v = grid()
    out = paint(rng, (0.55, 0.12, 0.12), 0.1)
    ring = (np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) > 0.38) & (np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) < 0.44)
    out[ring] = (0.85, 0.72, 0.4)
    diamond = np.abs(u - 0.5) + np.abs(v - 0.5) < 0.22
    out[diamond] = (0.15, 0.2, 0.45)
    return out


def screen(rng):
    u, v = grid()
    out = np.zeros((SIZE, SIZE, 3), np.float32)
    out[..., 0] = 0.15 + 0.5 * v
    out[..., 1] = 0.3 + 0.4 * np.clip(np.sin(u * 7 + v * 3), 0, 1)
    out[..., 2] = 0.6 - 0.3 * v
    return frame(out, 0.04, (0.03, 0.03, 0.03))


def books(rng):
    u, v = grid()
    out = np.zeros((SIZE, SIZE, 3), np.float32)
    shelves = 5
    shelf = np.floor(v * shelves).astype(int)
    colors = rng.uniform(0.15, 0.85, (shelves, 40, 3)).astype(np.float32)
    index = np.floor(u * 40 + rng.uniform(0, 1, shelves)[shelf]).astype(int) % 40
    out[:] = colors[shelf, index]
    top = (v * shelves) % 1.0
    out[top < 0.15] = (0.12, 0.08, 0.05)
    out[top > 0.92] = (0.45, 0.3, 0.18)
    return frame(out, 0.03, (0.45, 0.3, 0.18))


def door(rng):
    u, v = grid()
    out = wood(rng) * 1.2
    for x0, x1, y0, y1 in ((0.15, 0.85, 0.08, 0.45), (0.15, 0.85, 0.55, 0.92)):
        edge = ((np.abs(u - x0) < 0.015) | (np.abs(u - x1) < 0.015)) & (v > y0) & (v < y1)
        edge |= ((np.abs(v - y0) < 0.01) | (np.abs(v - y1) < 0.01)) & (u > x0) & (u < x1)
        out[edge] *= 0.55
    knob = (u - 0.85) ** 2 + ((v - 0.5) * 0.45) ** 2 < 0.0012
    out[knob] = (0.85, 0.75, 0.3)
    return np.clip(out, 0, 1)


def window(rng):
    u, v = grid()
    out = np.zeros((SIZE, SIZE, 3), np.float32)
    out[..., 0] = 0.55 + 0.3 * v
    out[..., 1] = 0.72 + 0.2 * v
    out[..., 2] = 0.92
    tree = (fbm(rng, 4) > 0.05) & (v > 0.55)
    out[tree] = (0.25, 0.45, 0.2)
    bars = (np.abs(u - 0.5) < 0.02) | (np.abs(v - 0.5) < 0.02)
    out[bars] = (0.95, 0.95, 0.95)
    return frame(out, 0.05, (0.95, 0.95, 0.95))


def painting(rng):
    out = np.zeros((SIZE, SIZE, 3), np.float32)
    for channel in range(3):
        out[..., channel] = 0.5 + 1.4 * fbm(rng, 3, 3)
    return frame(np.clip(out, 0, 1), 0.06, (0.55, 0.4, 0.12))


def hoop(rng):
    u, v = grid()
    out = np.full((SIZE, SIZE, 3), 0.95, np.float32)
    box = (np.maximum(np.abs(u - 0.5) / 0.22, np.abs(v - 0.62) / 0.2) < 1) & (np.maximum(np.abs(u - 0.5) / 0.22, np.abs(v - 0.62) / 0.2) > 0.85)
    out[box] = (0.85, 0.15, 0.1)
    return frame(out, 0.04, (0.85, 0.15, 0.1))


def shutter(rng):
    _, v = grid()
    out = paint(rng, (0.78, 0.78, 0.76), 0.04)
    out[np.abs(v * 16 - np.round(v * 16)) < 0.08] *= 0.6
    return out


def build(seed: int = 7) -> dict[str, tuple[np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    items = {
        "wood_floor": (woodfloor(rng), 1.2),
        "wall": (paint(rng, (0.86, 0.84, 0.79)), 2.0),
        "ceiling": (paint(rng, (0.93, 0.93, 0.92), 0.02), 2.0),
        "grass": (grass(rng), 1.0),
        "concrete": (concrete(rng), 2.0),
        "brick": (brick(rng), 1.0),
        "wood": (wood(rng), 1.0),
        "fabric": (fabric(rng, (0.32, 0.38, 0.48)), 0.5),
        "leather": (fabric(rng, (0.35, 0.2, 0.12)), 0.5),
        "metal": (paint(rng, (0.6, 0.62, 0.65), 0.08), 1.0),
        "foliage": (foliage(rng), 1.5),
        "bark": (bark(rng), 1.0),
        "court": (tiles(rng, (0.82, 0.62, 0.38), 2, 0.004, (0.7, 0.5, 0.3)), 4.0),
        "patio": (tiles(rng, (0.62, 0.6, 0.56), 2), 1.0),
        "fence": (boards(rng, (0.58, 0.45, 0.32), 7), 1.0),
        "cardboard": (paint(rng, (0.66, 0.5, 0.32), 0.08), 1.0),
        "paint_red": (paint(rng, (0.7, 0.08, 0.06), 0.04), 2.0),
        "orange": (paint(rng, (0.95, 0.45, 0.08), 0.04), 1.0),
        "stripes": (stripes(rng, (0.9, 0.1, 0.08), (0.95, 0.95, 0.95), 4), 1.0),
        "white": (paint(rng, (0.94, 0.94, 0.94), 0.03), 1.0),
        "black": (paint(rng, (0.06, 0.06, 0.07), 0.2), 1.0),
        "rug": (rug(rng), 1.0),
        "tv": (screen(rng), 1.0),
        "books": (books(rng), 1.0),
        "door": (door(rng), 1.0),
        "window": (window(rng), 1.0),
        "painting": (painting(rng), 1.0),
        "hoop": (hoop(rng), 1.0),
        "shutter": (shutter(rng), 1.0),
        "clouds": (np.clip(fbm(rng, 4, 4)[..., None] * 2.2 + 0.1, 0, 1).repeat(3, axis=2), 900.0),
    }
    return {name: ((np.clip(img, 0, 1) * 255).astype(np.uint8), tile) for name, (img, tile) in items.items()}
