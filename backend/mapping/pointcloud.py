"""A coloured point cloud from COLMAP's 3D points: the plain, accurate version of the map."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def read_points(points_txt: Path, max_error: float = 1.5, min_track: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Positions (N x 3) and colours (N x 3, 0-255) of the reliable points in a COLMAP points3D.txt.

    A point is reliable when its reprojection error is small and several photos see it.
    """
    xyz, rgb = [], []
    for line in points_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        error, track = float(fields[7]), (len(fields) - 8) // 2
        if error <= max_error and track >= min_track:
            xyz.append([float(v) for v in fields[1:4]])
            rgb.append([int(v) for v in fields[4:7]])
    return np.array(xyz, np.float32).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """A binary little-endian PLY with colour, which MeshLab, Blender and CloudCompare all open."""
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    record = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    record["p"], record["c"] = xyz, rgb
    path.write_bytes(header.encode() + record.tobytes())


def point_cloud(model: Path, out: Path, colmap: str) -> int:
    """Write out/cloud.ply from a COLMAP model folder. Returns the number of points."""
    text = out / "colmap_text"
    text.mkdir(parents=True, exist_ok=True)
    subprocess.run([colmap, "model_converter", "--input_path", str(model), "--output_path", str(text), "--output_type", "TXT"],
                   capture_output=True, check=True)
    xyz, rgb = read_points(text / "points3D.txt")
    write_ply(out / "cloud.ply", xyz, rgb)
    return len(xyz)
