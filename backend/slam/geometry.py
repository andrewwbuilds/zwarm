"""Poses, similarity transforms and trajectory alignment.

Poses are 4x4 camera-to-world matrices in the OpenCV camera frame: x right, y down,
z forward. The world frame is whatever the first keyframe defined, in arbitrary units.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# A body frame (x forward, y left, z up) seen from the OpenCV camera frame.
BODY_FROM_CAMERA = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def invert(pose: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def camera_pose_from_body(position, body_rotation) -> np.ndarray:
    """Ground truth: the drone's position and body-to-world rotation (3x3 or 9 numbers) as a camera pose."""
    rotation = np.asarray(body_rotation, float).reshape(3, 3) @ BODY_FROM_CAMERA
    return make_pose(rotation, np.asarray(position, float))


def log_so3(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_rotvec()


def exp_so3(vector: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(vector).as_matrix()


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True):
    """Best similarity transform (scale, rotation, translation) taking source points onto target.

    Returns (s, R, t) with target ~= s * R @ source + t. Points are rows.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    centred_s, centred_t = source - mean_s, target - mean_t
    covariance = centred_t.T @ centred_s / len(source)
    u, d, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    variance = (centred_s ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(d) @ sign) / variance) if with_scale and variance > 0 else 1.0
    translation = mean_t - scale * rotation @ mean_s
    return scale, rotation, translation


def apply_sim3(transform, points: np.ndarray) -> np.ndarray:
    scale, rotation, translation = transform
    return scale * np.asarray(points) @ rotation.T + translation


def triangulate(k: np.ndarray, pose_a: np.ndarray, pose_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray):
    """Points seen in two views (pixels). Returns (points in world, depth in a, depth in b)."""
    proj_a = k @ invert(pose_a)[:3]
    proj_b = k @ invert(pose_b)[:3]
    homogeneous = __import__("cv2").triangulatePoints(proj_a, proj_b, pts_a.T.astype(float), pts_b.T.astype(float))
    world = (homogeneous[:3] / homogeneous[3]).T
    depth_a = (invert(pose_a)[:3, :3] @ world.T + invert(pose_a)[:3, 3:4])[2]
    depth_b = (invert(pose_b)[:3, :3] @ world.T + invert(pose_b)[:3, 3:4])[2]
    return world, depth_a, depth_b
