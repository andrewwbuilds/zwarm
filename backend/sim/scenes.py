"""Places to fly. Each builder returns a fresh Scene; x is forward from the takeoff spot, y left, z up."""

from __future__ import annotations

from sim.scene import Box, Scene


def deco(lo, hi, material, fit=True) -> Box:
    return Box(lo, hi, material, fit=fit, solid=False)


def tree(x, y, height=4.2, crown=2.3) -> tuple[Box, Box]:
    r = crown / 2
    return (Box((x - 0.15, y - 0.15, 0), (x + 0.15, y + 0.15, height * 0.55), "bark"),
            Box((x - r, y - r, height * 0.5), (x + r, y + r, height), "foliage"))


def gate(x, y, width=1.8, height=2.2) -> tuple[Box, ...]:
    w = width / 2
    return (Box((x - 0.08, y + w, 0), (x + 0.08, y + w + 0.15, height), "stripes"),
            Box((x - 0.08, y - w - 0.15, 0), (x + 0.08, y - w, height), "stripes"),
            Box((x - 0.08, y - w - 0.15, height), (x + 0.08, y + w + 0.15, height + 0.15), "stripes"))


def cone(x, y) -> Box:
    return Box((x - 0.15, y - 0.15, 0), (x + 0.15, y + 0.15, 0.45), "orange")


def room() -> Scene:
    """A living room, about the size of where most first flights happen."""
    return Scene(
        name="room",
        bounds=Box((-3.0, -2.6, 0.0), (3.6, 2.6, 2.6), "wall"),
        boxes=(
            Box((3.15, -0.9, 0.0), (3.6, 0.9, 0.5), "wood"),
            Box((3.28, -0.68, 0.55), (3.36, 0.68, 1.32), "tv", fit=True),
            Box((-1.5, 2.22, 0.0), (-0.3, 2.6, 2.0), "books", fit=True),
            Box((-1.2, -2.6, 0.0), (1.0, -1.75, 0.45), "fabric"),
            Box((-1.2, -2.6, 0.45), (1.0, -2.32, 0.92), "fabric"),
            Box((-1.45, -2.6, 0.0), (-1.2, -1.75, 0.62), "fabric"),
            Box((1.0, -2.6, 0.0), (1.25, -1.75, 0.62), "fabric"),
            Box((0.9, -0.45, 0.0), (1.6, 0.45, 0.45), "wood"),
            Box((2.0, 1.55, 0.0), (2.8, 2.45, 0.85), "leather"),
            Box((-0.2, -0.2, 2.45), (0.2, 0.2, 2.6), "white"),
            deco((-0.6, -1.1, 0.0), (1.9, 1.1, 0.004), "rug"),
            deco((-3.0, 0.6, 0.0), (-2.97, 1.5, 2.05), "door"),
            deco((0.5, -2.6, 1.0), (2.0, -2.58, 2.0), "window"),
            deco((0.8, 2.58, 1.2), (1.8, 2.6, 1.9), "painting"),
        ),
        floor_material="wood_floor",
    )


def garage() -> Scene:
    """A garage: bare concrete gives the flow sensor less to lock on to."""
    return Scene(
        name="garage",
        bounds=Box((-3.0, -3.5, 0.0), (6.0, 3.5, 2.8), "wall"),
        boxes=(
            Box((1.5, -3.2, 0.15), (5.5, -1.4, 1.0), "paint_red"),
            Box((2.5, -3.1, 1.0), (4.3, -1.5, 1.45), "paint_red"),
            Box((-1.0, 2.9, 0.0), (3.0, 3.5, 2.2), "metal"),
            Box((-2.5, 2.2, 0.0), (-1.8, 2.9, 0.5), "cardboard"),
            Box((-2.4, 2.3, 0.5), (-1.9, 2.8, 0.9), "cardboard"),
            Box((-2.9, -3.4, 0.0), (-2.2, -2.7, 0.6), "cardboard"),
            Box((5.0, 0.5, 0.0), (6.0, 2.5, 0.9), "wood"),
            deco((5.98, -3.0, 0.0), (6.0, -0.3, 2.3), "shutter"),
            deco((-3.0, -0.5, 0.0), (-2.98, 0.4, 2.0), "door"),
        ),
        floor_material="concrete",
        ceiling_material="concrete",
        flow_quality=0.7,
    )


def backyard() -> Scene:
    """A fenced back garden with a light breeze."""
    return Scene(
        name="backyard",
        boxes=(
            Box((8.0, -8.0, 0.0), (9.0, 6.0, 4.0), "brick"),
            deco((7.98, 0.5, 0.0), (8.0, 1.5, 2.1), "door"),
            deco((7.98, -4.0, 1.0), (8.0, -2.5, 2.0), "window"),
            deco((7.98, 2.5, 1.0), (8.0, 4.0, 2.0), "window"),
            Box((-8.0, 7.95, 0.0), (8.0, 8.05, 1.6), "fence"),
            Box((-8.0, -8.05, 0.0), (8.0, -7.95, 1.6), "fence"),
            Box((-8.05, -8.0, 0.0), (-7.95, 8.0, 1.6), "fence"),
            *tree(-3.0, 4.2),
            *tree(3.2, -4.8, 4.6),
            deco((5.5, -3.0, 0.0), (8.0, 3.0, 0.01), "patio", fit=False),
            Box((5.8, -1.0, 0.0), (7.2, 1.0, 0.75), "wood"),
        ),
        floor_material="grass",
        wind_mps=1.2,
        wind_direction_deg=30.0,
        gust_mps=0.8,
        fog_distance_m=80.0,
    )


def field() -> Scene:
    """Open grass with a steady wind: room to practise speed, and to lose the drone."""
    return Scene(
        name="field",
        boxes=(*tree(25, 8, 6), *tree(32, -14, 7, 3), *tree(-18, 20, 5), *tree(45, 30, 8, 4)),
        floor_material="grass",
        wind_mps=2.5,
        wind_direction_deg=90.0,
        gust_mps=1.5,
        fog_distance_m=150.0,
    )


def gym() -> Scene:
    """A sports hall. The plain polished floor starves the flow sensor, so the drone drifts."""
    return Scene(
        name="gym",
        bounds=Box((-10.0, -7.0, 0.0), (14.0, 7.0, 7.0), "wall"),
        boxes=(
            Box((13.5, -0.9, 2.9), (13.6, 0.9, 3.95), "hoop", fit=True),
            Box((-9.6, -0.9, 2.9), (-9.5, 0.9, 3.95), "hoop", fit=True),
            Box((-6.0, 5.5, 0.0), (10.0, 7.0, 0.5), "wood"),
            Box((-6.0, 6.0, 0.5), (10.0, 7.0, 1.0), "wood"),
            Box((-6.0, 6.5, 1.0), (10.0, 7.0, 1.5), "wood"),
            Box((-9.0, -6.5, 0.0), (-7.0, -4.0, 0.1), "fabric"),
            deco((-4.0, -7.0, 4.0), (0.0, -6.98, 6.0), "window"),
            deco((4.0, -7.0, 4.0), (8.0, -6.98, 6.0), "window"),
            deco((-10.0, -0.6, 0.0), (-9.98, 0.6, 2.2), "door"),
        ),
        floor_material="court",
        ceiling_material="metal",
        flow_quality=0.25,
    )


def course() -> Scene:
    """Gates and cones on a calm field."""
    return Scene(
        name="course",
        boxes=(
            *gate(5.0, 0.0),
            *gate(12.0, 3.0),
            *gate(20.0, -2.0),
            cone(8.5, 1.5), cone(16.0, 0.5), cone(24.0, -2.0),
            *tree(30, 10, 6), *tree(-10, -12, 5),
        ),
        floor_material="grass",
        wind_mps=0.8,
        wind_direction_deg=200.0,
        gust_mps=0.5,
        fog_distance_m=120.0,
    )


SCENES = {build.__name__: build for build in (room, garage, backyard, field, gym, course)}
