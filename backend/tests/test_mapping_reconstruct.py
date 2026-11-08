import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mapping.reconstruct import Model, StepFailed, best_model, coverage_warnings, parse_analysis, reconstruct

SAMPLE = """Rigs: 1
Cameras: 1
Frames: 108
Registered images: 108
Points: 3019
Observations: 18557
Mean track length: 6.146737
Mean reprojection error: 0.885723px
"""


class AnalysisTests(unittest.TestCase):
    def test_reads_colmaps_summary(self):
        self.assertEqual(parse_analysis(SAMPLE), (108, 3019, 0.885723))

    def test_missing_fields_read_as_zero(self):
        self.assertEqual(parse_analysis("nothing here"), (0, 0, 0.0))


class ModelChoiceTests(unittest.TestCase):
    def test_picks_the_model_that_placed_the_most_photos(self):
        small, big = Model(Path("0"), 3, 312, 0.8), Model(Path("1"), 108, 3019, 0.9)
        self.assertIs(best_model([small, big]), big)

    def test_no_models_is_none(self):
        self.assertIsNone(best_model([]))


class CoverageWarningTests(unittest.TestCase):
    def test_a_good_model_has_no_warnings(self):
        self.assertEqual(coverage_warnings(Model(Path("0"), 100, 5000, 0.8), 105), [])

    def test_unlinked_photos_are_called_out(self):
        text = " ".join(coverage_warnings(Model(Path("0"), 40, 5000, 0.8), 100))
        self.assertIn("40%", text)
        self.assertIn("gaps", text)

    def test_a_plain_scene_is_called_out(self):
        self.assertTrue(any("plain" in w for w in coverage_warnings(Model(Path("0"), 100, 300, 0.8), 100)))

    def test_high_error_is_called_out(self):
        self.assertTrue(any("error" in w.lower() for w in coverage_warnings(Model(Path("0"), 100, 5000, 2.4), 100)))


class GuardTests(unittest.TestCase):
    def test_too_few_photos_gives_a_clear_message_before_any_work(self):
        with TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir()
            for i in range(4):
                (images / f"{i}.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(StepFailed, "at least 10"):
                reconstruct(images, Path(tmp) / "colmap")


if __name__ == "__main__":
    unittest.main()
