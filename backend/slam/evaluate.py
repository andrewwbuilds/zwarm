"""How wrong is an estimated trajectory? Metrics against ground truth.

A monocular trajectory is only known up to a similarity transform, so it is aligned to the
truth first (Umeyama). The leftover error after alignment is what the tracker is responsible for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.geometry import apply_sim3, umeyama


@dataclass
class Report:
    frames: int
    ate_rmse: float  # metres, after the best-fit alignment
    ate_max: float
    scale: float  # how many metres one map unit is, from the alignment
    path_length: float  # true path length in metres
    drift_percent: float  # ate_rmse as a percentage of the path
    rpe_percent: float  # median error of 1 s of motion, as a percentage of the true motion
    end_error: float  # distance between the estimated and true final position after alignment

    def __str__(self) -> str:
        return (f"{self.frames} frames, path {self.path_length:.1f} m: position error {self.ate_rmse * 100:.1f} cm rms "
                f"(worst {self.ate_max * 100:.1f} cm, end {self.end_error * 100:.1f} cm), {self.drift_percent:.1f}% of the path; "
                f"1 s of motion off by {self.rpe_percent:.1f}%; 1 map unit = {self.scale:.3f} m")


def evaluate(estimated: np.ndarray, truth: np.ndarray, times: np.ndarray | None = None, align_scale: bool = True) -> Report:
    """Compare N estimated and true positions (N x 3), already matched frame to frame."""
    estimated, truth = np.asarray(estimated, float), np.asarray(truth, float)
    transform = umeyama(estimated, truth, align_scale)
    aligned = apply_sim3(transform, estimated)
    errors = np.linalg.norm(aligned - truth, axis=1)
    steps = np.linalg.norm(np.diff(truth, axis=0), axis=1)
    path = float(steps.sum())
    window = 17 if times is None else max(1, int(np.searchsorted(times - times[0], 1.0)))
    window = min(window, len(truth) - 1)
    if window >= 1:
        moved_true = np.linalg.norm(truth[window:] - truth[:-window], axis=1)
        moved_est = np.linalg.norm(aligned[window:] - aligned[:-window], axis=1)
        enough = moved_true > 0.1
        rpe = float(np.median(np.abs(moved_est[enough] - moved_true[enough]) / moved_true[enough]) * 100) if enough.any() else 0.0
    else:
        rpe = 0.0
    rmse = float(np.sqrt((errors ** 2).mean()))
    return Report(len(truth), rmse, float(errors.max()), float(transform[0]), path,
                  rmse / path * 100 if path > 0 else 0.0, rpe, float(errors[-1]))
