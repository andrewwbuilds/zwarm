import unittest
from pathlib import Path

from mapping.nerf import render_command, train_command


class CommandTests(unittest.TestCase):
    def test_trains_nerfacto_on_the_mac_gpu_without_asking_questions(self):
        command = [str(c) for c in train_command(Path("/ds"), Path("/out"), 4000)]
        text = " ".join(command)
        self.assertIn("nerfacto", command)
        self.assertIn("--machine.device-type mps", text)
        self.assertIn("--max-num-iterations 4000", text)
        self.assertIn("--viewer.quit-on-train-completion True", text)
        self.assertLess(command.index("colmap"), command.index("--downscale-factor"))  # dataparser options come last
        self.assertEqual(command[command.index("--downscale-factor") + 1], "1")  # 1 avoids a prompt about missing downscaled images

    def test_renders_a_video_through_the_training_cameras(self):
        command = [str(c) for c in render_command(Path("/out/config.yml"), Path("/out/nerf.mp4"))]
        text = " ".join(command)
        self.assertIn("interpolate", command)
        self.assertIn("--pose-source eval", text)  # held-out photos, so the video is an honest test and short to render
        self.assertIn("--output-format video", text)


if __name__ == "__main__":
    unittest.main()
