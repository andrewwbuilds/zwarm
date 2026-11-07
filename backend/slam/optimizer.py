"""Bundle adjustment and similarity pose-graph optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from slam.geometry import invert, make_pose


@dataclass
class BundleResult:
    poses: dict  # keyframe id -> 4x4 camera-to-world
    points: dict  # point id -> xyz
    residuals: np.ndarray  # pixel error of each observation, in the order given
    error_before: float
    error_after: float


def _project(k: np.ndarray, rotvec: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    camera = Rotation.from_rotvec(rotvec).apply(points) + t
    z = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
    return np.stack([k[0, 0] * camera[:, 0] / z + k[0, 2], k[1, 1] * camera[:, 1] / z + k[1, 2]], axis=1)


def bundle_adjust(k: np.ndarray, poses: dict, fixed: set, points: dict, observations: list,
                  max_nfev: int = 20, huber_px: float = 1.5) -> BundleResult:
    """Refine free keyframe poses and all points to minimise reprojection error.

    poses: keyframe id -> camera-to-world. observations: (keyframe id, point id, pixel x, pixel y).
    """
    cam_ids = sorted(poses)
    free_ids = [c for c in cam_ids if c not in fixed]
    cam_row = {c: i for i, c in enumerate(cam_ids)}
    free_row = {c: i for i, c in enumerate(free_ids)}
    point_ids = sorted(points)
    point_row = {p: i for i, p in enumerate(point_ids)}
    obs = [o for o in observations if o[0] in cam_row and o[1] in point_row]
    cam_index = np.array([cam_row[o[0]] for o in obs])
    point_index = np.array([point_row[o[1]] for o in obs])
    pixels = np.array([[o[2], o[3]] for o in obs], float)

    cw = np.array([invert(poses[c]) for c in cam_ids])
    cam_rot = Rotation.from_matrix(cw[:, :3, :3]).as_rotvec()
    cam_t = cw[:, :3, 3].copy()
    xyz0 = np.array([points[p] for p in point_ids], float)
    free_rows = np.array([cam_row[c] for c in free_ids], int)
    n_free, n_pts = len(free_ids), len(point_ids)

    def unpack(x):
        rot, t = cam_rot.copy(), cam_t.copy()
        if n_free:
            block = x[: n_free * 6].reshape(-1, 6)
            rot[free_rows], t[free_rows] = block[:, :3], block[:, 3:]
        return rot, t, x[n_free * 6:].reshape(-1, 3)

    def residual(x):
        rot, t, xyz = unpack(x)
        out = np.empty((len(obs), 2))
        # Project per camera so each camera's rotation is built once.
        matrices = Rotation.from_rotvec(rot).as_matrix()
        camera = np.einsum("nij,nj->ni", matrices[cam_index], xyz[point_index]) + t[cam_index]
        z = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
        out[:, 0] = k[0, 0] * camera[:, 0] / z + k[0, 2]
        out[:, 1] = k[1, 1] * camera[:, 1] / z + k[1, 2]
        return (out - pixels).ravel()

    x0 = np.concatenate([np.concatenate([cam_rot[free_rows], cam_t[free_rows]], axis=1).ravel() if n_free else [],
                         xyz0.ravel()])
    before = residual(x0).reshape(-1, 2)
    error_before = float(np.sqrt((before ** 2).sum(axis=1)).mean()) if len(obs) else 0.0

    sparsity = lil_matrix((len(obs) * 2, len(x0)), dtype=int)
    rows = np.arange(len(obs))
    for axis in range(2):
        for c in range(6):
            mask = np.isin(cam_index, free_rows)
            local = np.array([free_row[cam_ids[i]] for i in cam_index[mask]], int) if mask.any() else np.array([], int)
            sparsity[2 * rows[mask] + axis, local * 6 + c] = 1
        for c in range(3):
            sparsity[2 * rows + axis, n_free * 6 + point_index * 3 + c] = 1

    # A tight robust loss alone stalls when the start is a few pixels off. So: pull the solution into
    # the right basin with plain least squares on the observations that start plausibly close, then
    # let Huber discount real outliers across all of them.
    start_error = np.sqrt((before ** 2).sum(axis=1))
    plausible = np.repeat(start_error < max(25.0, 5.0 * float(np.median(start_error))), 2).astype(float)
    common = dict(jac_sparsity=sparsity, x_scale="jac", method="trf", ftol=1e-4, xtol=1e-6)
    warm = least_squares(lambda x: residual(x) * plausible, x0, loss="linear", max_nfev=max_nfev, **common)
    solution = least_squares(residual, warm.x, loss="huber", f_scale=huber_px, max_nfev=max_nfev, **common)
    rot, t, xyz = unpack(solution.x)
    after = residual(solution.x).reshape(-1, 2)
    new_poses = dict(poses)
    for c in free_ids:
        i = cam_row[c]
        new_poses[c] = invert(make_pose(Rotation.from_rotvec(rot[i]).as_matrix(), t[i]))
    new_points = {p: xyz[i] for p, i in point_row.items()}
    norms = np.sqrt((after ** 2).sum(axis=1))
    return BundleResult(new_poses, new_points, norms, error_before, float(norms.mean()) if len(obs) else 0.0)


# ------------------------------------------------------------------- pose graph
def _sim3_parts(x: np.ndarray):
    rotvec, t, log_s = x[:, :3], x[:, 3:6], x[:, 6]
    return Rotation.from_rotvec(rotvec), t, np.exp(log_s)


def relative_sim3(rot_i, t_i, s_i, rot_j, t_j, s_j):
    """S_i^-1 * S_j for similarity transforms x_w = s R x_c + t."""
    rot = rot_i.inv() * rot_j
    t = rot_i.inv().apply(t_j - t_i) / s_i[:, None] if np.ndim(s_i) else rot_i.inv().apply(t_j - t_i) / s_i
    return rot, t, s_j / s_i


def optimize_pose_graph(poses: dict, edges: list, fixed: set, max_nfev: int = 60,
                        rotation_weight: float = 10.0, translation_weight: float = 10.0, scale_weight: float = 10.0):
    """Adjust keyframe poses so relative-motion constraints hold.

    poses: keyframe id -> 4x4 camera-to-world. edges: (i, j, R_ij, t_ij, s_ij, weight), the measured
    S_i^-1 * S_j. Returns id -> (R, t, s): corrected similarity, where scale s says how much that
    keyframe's local map must be stretched.
    """
    ids = sorted(poses)
    row = {k: i for i, k in enumerate(ids)}
    free = [k for k in ids if k not in fixed]
    free_row = {k: i for i, k in enumerate(free)}
    start = np.zeros((len(ids), 7))
    for k in ids:
        start[row[k], :3] = Rotation.from_matrix(poses[k][:3, :3]).as_rotvec()
        start[row[k], 3:6] = poses[k][:3, 3]
    ei = np.array([row[e[0]] for e in edges])
    ej = np.array([row[e[1]] for e in edges])
    measured_rot = Rotation.from_matrix(np.array([e[2] for e in edges]))
    measured_t = np.array([e[3] for e in edges])
    measured_log_s = np.log(np.array([e[4] for e in edges]))
    weights = np.array([e[5] for e in edges])
    free_index = np.array([row[k] for k in free], int)

    def build(x):
        full = start.copy()
        if len(free):
            full[free_index] = x.reshape(-1, 7)
        return full

    def residual(x):
        full = build(x)
        rot, t, s = _sim3_parts(full)
        rot_ij, t_ij, s_ij = relative_sim3(rot[ei], t[ei], s[ei], rot[ej], t[ej], s[ej])
        r_rot = (measured_rot.inv() * rot_ij).as_rotvec() * rotation_weight
        r_t = (t_ij - measured_t) * translation_weight
        r_s = (np.log(s_ij) - measured_log_s) * scale_weight
        return (np.concatenate([r_rot, r_t, r_s[:, None]], axis=1) * weights[:, None]).ravel()

    x0 = start[free_index].ravel() if len(free) else np.zeros(0)
    if len(free):
        sparsity = lil_matrix((len(edges) * 7, len(x0)), dtype=int)
        for n in range(len(edges)):
            for node in (ei[n], ej[n]):
                if ids[node] in free_row:
                    base = free_row[ids[node]] * 7
                    sparsity[n * 7:(n + 1) * 7, base:base + 7] = 1
        solution = least_squares(residual, x0, jac_sparsity=sparsity, method="trf", loss="huber", f_scale=1.0,
                                 max_nfev=max_nfev, ftol=1e-6, xtol=1e-8)
        full = build(solution.x)
    else:
        full = start
    rot, t, s = _sim3_parts(full)
    return {k: (rot[row[k]].as_matrix(), t[row[k]], float(s[row[k]])) for k in ids}
