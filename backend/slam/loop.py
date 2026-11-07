"""Recognising places the drone has been before.

Every keyframe's binary descriptors go into one big database. A new keyframe's descriptors vote
for the older keyframes they match, the best candidates are checked for a rigid 3D fit, and a
fit that holds gives the similarity transform that closes the loop. Using the 3D points makes
the fit pin down scale drift as well as position, which a monocular tracker accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from slam.geometry import apply_sim3, umeyama


@dataclass
class LoopCandidate:
    keyframe: int  # the older keyframe that was recognised
    transform: tuple  # (s, R, t): maps the current map's points onto the old map's
    pairs: list  # (current point id, old point id) of the inliers
    votes: int


class DescriptorDatabase:
    def __init__(self):
        self.descriptors = np.zeros((0, 32), np.uint8)
        self.owner = np.zeros(0, np.int64)
        self.point = np.zeros(0, np.int64)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __len__(self) -> int:
        return len(self.owner)

    def add(self, keyframe: int, descriptors: np.ndarray, point_ids: np.ndarray) -> None:
        if len(descriptors) == 0:
            return
        self.descriptors = np.concatenate([self.descriptors, descriptors])
        self.owner = np.concatenate([self.owner, np.full(len(descriptors), keyframe, np.int64)])
        self.point = np.concatenate([self.point, point_ids.astype(np.int64)])

    def drop_point(self, point_id: int) -> None:
        keep = self.point != point_id
        if not keep.all():
            self.descriptors, self.owner, self.point = self.descriptors[keep], self.owner[keep], self.point[keep]

    def match(self, descriptors: np.ndarray, newest_allowed: int, ratio: float = 0.75):
        """Matches against keyframes up to newest_allowed. Returns (query index, database row) arrays."""
        rows = np.where(self.owner <= newest_allowed)[0]
        if len(rows) < 2 or len(descriptors) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        pairs = self._matcher.knnMatch(descriptors, self.descriptors[rows], k=2)
        query, row = [], []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance and pair[0].distance < 64:
                query.append(pair[0].queryIdx)
                row.append(rows[pair[0].trainIdx])
        return np.array(query, int), np.array(row, int)


def ransac_sim3(source: np.ndarray, target: np.ndarray, threshold: float, rng: np.random.Generator, iterations: int = 150):
    """Robust similarity fit. Returns (transform, inlier mask) or None."""
    n = len(source)
    if n < 6:
        return None
    best, best_mask = None, None
    for _ in range(iterations):
        pick = rng.choice(n, 3, replace=False)
        a, b = source[pick], target[pick]
        if np.linalg.matrix_rank(a - a.mean(0), tol=1e-6 * max(1.0, np.abs(a).max())) < 2:
            continue
        s, rotation, t = umeyama(a, b)
        if not 0.3 < s < 3.3:
            continue
        mask = np.linalg.norm(apply_sim3((s, rotation, t), source) - target, axis=1) < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best, best_mask = (s, rotation, t), mask
    if best_mask is None or best_mask.sum() < 6:
        return None
    s, rotation, t = umeyama(source[best_mask], target[best_mask])
    mask = np.linalg.norm(apply_sim3((s, rotation, t), source) - target, axis=1) < threshold
    if mask.sum() >= 6:
        s, rotation, t = umeyama(source[mask], target[mask])
    return (s, rotation, t), mask


def find_loop(database: DescriptorDatabase, descriptors: np.ndarray, point_ids: np.ndarray, points_xyz: dict,
              newest_allowed: int, min_votes: int = 25, min_inliers: int = 20, rng: np.random.Generator | None = None,
              candidates: int = 3) -> LoopCandidate | None:
    """Look for a place seen before. points_xyz maps point id -> xyz for both current and old points."""
    rng = rng or np.random.default_rng(0)
    query, rows = database.match(descriptors, newest_allowed)
    if len(query) < min_votes:
        return None
    owners = database.owner[rows]
    votes = np.bincount(owners)
    best: LoopCandidate | None = None
    for keyframe in np.argsort(votes)[::-1][:candidates]:
        if votes[keyframe] < min_votes:
            break
        chosen = owners == keyframe
        pairs = {}
        for q, row in zip(query[chosen], rows[chosen]):
            cur, old = int(point_ids[q]), int(database.point[row])
            if cur != old and cur in points_xyz and old in points_xyz:
                pairs[cur] = old
        if len(pairs) < min_inliers:
            continue
        cur_ids, old_ids = list(pairs), [pairs[c] for c in pairs]
        source = np.array([points_xyz[c] for c in cur_ids])
        target = np.array([points_xyz[o] for o in old_ids])
        spread = np.linalg.norm(target - target.mean(0), axis=1).mean()
        found = ransac_sim3(source, target, threshold=0.06 * max(spread, 1e-6), rng=rng)
        if found is None:
            continue
        transform, mask = found
        if mask.sum() < min_inliers or mask.sum() < 0.5 * len(mask):
            continue
        candidate = LoopCandidate(int(keyframe), transform, [(cur_ids[i], old_ids[i]) for i in np.where(mask)[0]],
                                  int(votes[keyframe]))
        if best is None or len(candidate.pairs) > len(best.pairs):
            best = candidate
    return best
