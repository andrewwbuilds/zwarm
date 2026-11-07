import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from mapping.curate import curate, score_frames

RNG = np.random.default_rng(8)
WORLD = cv2.resize(RNG.integers(0, 255, (225, 500, 3), dtype=np.uint8), (2000, 900), interpolation=cv2.INTER_CUBIC)


def frame(dx=0, blur=0.0):
    image = WORLD[100:756, 200 + dx:1384 + dx].copy()
    return cv2.GaussianBlur(image, (0, 0), blur) if blur else image


def write_raw(folder: Path, frames: list) -> Path:
    raw = folder / "raw"
    raw.mkdir(parents=True)
    for i, image in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        (raw / f"{i:06d}_{i * 57:08d}.jpg").write_bytes(buf.tobytes())
    return raw


class CurateTests(unittest.TestCase):
    def run_curate(self, frames, **kwargs):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        write_raw(root, frames)
        report = curate(root, **kwargs)
        return report, sorted((root / "images").glob("*.jpg")), root

    def test_keeps_the_sharpest_frame_of_each_window(self):
        frames = [frame(dx=i * 6, blur=b) for i, b in enumerate([3, 0, 2, 4, 2, 0, 3, 3, 0, 4, 3, 2])]
        report, kept, _ = self.run_curate(frames, window=3)
        self.assertEqual(len(kept), 4)  # one per window of three
        self.assertEqual(report.chosen, [1, 5, 8, 11])  # the sharpest of each, by original frame number

    def test_a_uniformly_soft_camera_still_gives_photos(self):
        # The S20's frames are all soft. Blur is judged against the neighbours, not against a fixed bar.
        frames = [frame(dx=i * 8, blur=2.5) for i in range(30)]
        report, kept, _ = self.run_curate(frames, window=3)
        self.assertGreaterEqual(len(kept), 9)

    def test_a_motion_smeared_stretch_is_rejected_next_to_sharp_ones(self):
        frames = [frame(dx=i * 8) for i in range(24)]
        for i in range(9, 15):
            frames[i] = frame(dx=i * 8, blur=6)
        report, kept, _ = self.run_curate(frames, window=3)
        self.assertTrue(set(report.chosen).isdisjoint(range(9, 15)))
        self.assertGreater(report.rejected["blurry"], 0)

    def test_similar_but_different_views_are_all_kept(self):
        # Neighbouring frames of a slow flight look alike. That is overlap, not duplication.
        frames = [frame(dx=i * 3) for i in range(36)]
        _, kept, _ = self.run_curate(frames, window=3)
        self.assertEqual(len(kept), 12)

    def test_only_genuinely_identical_frames_are_dropped(self):
        frames = [frame(dx=0)] * 9 + [frame(dx=40 + i * 5) for i in range(9)]
        report, kept, _ = self.run_curate(frames, window=3)
        self.assertEqual(report.rejected["identical"], 2)
        self.assertEqual(len(kept), 4)

    def test_kept_photos_are_the_original_files_in_flight_order(self):
        frames = [frame(dx=i * 6) for i in range(9)]
        report, kept, root = self.run_curate(frames, window=3)
        self.assertEqual([p.name for p in kept], ["00001.jpg", "00002.jpg", "00003.jpg"])
        originals = sorted((root / "raw").glob("*.jpg"))
        self.assertEqual(kept[0].read_bytes(), originals[report.chosen[0]].read_bytes())

    def test_scores_are_remembered_so_a_second_run_is_quick_and_the_same(self):
        frames = [frame(dx=i * 6, blur=(i % 3)) for i in range(12)]
        report, _, root = self.run_curate(frames, window=3)
        self.assertTrue((root / "raw" / "scores.json").exists())
        again = curate(root, window=3)
        self.assertEqual(again.chosen, report.chosen)

    def test_a_looser_window_keeps_more(self):
        frames = [frame(dx=i * 5) for i in range(30)]
        _, tight, _ = self.run_curate(frames, window=5)
        _, loose, _ = self.run_curate(frames, window=2)
        self.assertGreater(len(loose), len(tight))

    def test_no_raw_frames_is_a_clear_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "raw"):
                curate(Path(tmp))


class ScoreTests(unittest.TestCase):
    def test_a_blank_frame_scores_far_below_a_detailed_one(self):
        with TemporaryDirectory() as tmp:
            raw = write_raw(Path(tmp), [frame(), np.full((656, 1184, 3), 100, np.uint8)])
            scores = score_frames(sorted(raw.glob("*.jpg")))
        self.assertGreater(scores[0][0], 50 * scores[1][0] + 1)


if __name__ == "__main__":
    unittest.main()
