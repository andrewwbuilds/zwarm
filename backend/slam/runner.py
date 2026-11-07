"""Run the SLAM system over a sequence and score it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from slam.camera import Intrinsics
from slam.evaluate import Report, evaluate
from slam.sequences import Frame
from slam.system import Config, Slam


@dataclass
class Run:
    slam: Slam
    estimates: list
    report: Report | None
    lost_frames: int
    mean_ms: float

    def summary(self) -> str:
        head = (f"{len(self.estimates)} frames, {self.lost_frames} without a pose, {self.mean_ms:.0f} ms/frame, "
                f"{len(self.slam.keyframes)} keyframes, {len(self.slam.points)} points, {len(self.slam.loops)} loops closed")
        return head + ("\n" + str(self.report) if self.report else "")


def run(frames: list[Frame], intrinsics: Intrinsics, config: Config | None = None, align_scale: bool = True,
        progress=None) -> Run:
    slam = Slam(intrinsics, config)
    estimates = []
    for n, frame in enumerate(frames):
        estimates.append(slam.process(frame.image, frame.t, frame.command))
        if progress and n % 100 == 0:
            progress(n, len(frames), estimates[-1])
    return finish(slam, frames, estimates, align_scale)


def finish(slam: Slam, frames: list[Frame], estimates: list, align_scale: bool = True) -> Run:
    trajectory = slam.trajectory()
    report = None
    have_truth = frames and frames[0].truth is not None
    if have_truth and len(trajectory) > 5:
        index = {i: p for i, _, p in trajectory}
        used = [i for i in range(len(frames)) if i in index]
        est = np.array([index[i][:3, 3] for i in used])
        truth = np.array([frames[i].truth[:3, 3] for i in used])
        times = np.array([frames[i].t for i in used])
        report = evaluate(est, truth, times, align_scale)
    lost = sum(1 for e in estimates if e.pose is None)
    mean_ms = 1000 * float(np.mean([e.seconds for e in estimates])) if estimates else 0.0
    return Run(slam, estimates, report, lost, mean_ms)
