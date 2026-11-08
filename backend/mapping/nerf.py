"""A NeRF (neural radiance field) of the room, trained with nerfstudio on the Mac's GPU."""

from __future__ import annotations

import shutil
from pathlib import Path

from mapping.reconstruct import StepFailed, ToolMissing, run

NERF_ENV = Path.home() / ".local/share/nerf-env/.venv/bin"


# nerfstudio saves checkpoints with NumPy values that PyTorch's safe loader refuses. These are files this tool
# just wrote itself, so loading them normally is fine.
TRUSTED_CHECKPOINTS = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}


def _bin(name: str) -> Path:
    path = NERF_ENV / name
    if not path.exists():
        raise ToolMissing(f"nerfstudio is not installed. Run: tools/setup_nerf.sh")
    return path


def train_command(dataset: Path, out: Path, iterations: int) -> list:
    return [_bin("ns-train"), "nerfacto", "--data", dataset, "--output-dir", out, "--max-num-iterations", iterations,
            "--vis", "tensorboard", "--machine.device-type", "mps", "--pipeline.datamanager.train-num-rays-per-batch", 2048,
            "--viewer.quit-on-train-completion", "True",
            # Everything after 'colmap' is for the COLMAP data parser. Factor 1 stops it asking about missing downscaled images.
            "colmap", "--downscale-factor", 1]


def render_command(config: Path, video: Path) -> list:
    return [_bin("ns-render"), "interpolate", "--load-config", config, "--output-path", video, "--pose-source", "eval",
            "--interpolation-steps", 2, "--downscale-factor", 4, "--frame-rate", 8, "--output-format", "video",
            "--rendered-output-names", "rgb"]


def train_nerf(dataset: Path, out: Path, iterations: int = 5000) -> Path:
    """Train a NeRF and render a fly-through video. Returns the video; the trained model stays in out/nerf."""
    nerf = out / "nerf"
    shutil.rmtree(nerf, ignore_errors=True)
    nerf.mkdir(parents=True)
    run(train_command(dataset, nerf, iterations), out / "nerf_training.log", timeout_s=None)
    configs = sorted(nerf.rglob("config.yml"))
    if not configs:
        raise StepFailed(f"nerfstudio finished but wrote no model (log: {out / 'nerf_training.log'})")
    (nerf / "config_path.txt").write_text(str(configs[-1]))
    video = out / "nerf.mp4"
    run(render_command(configs[-1], video), out / "nerf_render.log", TRUSTED_CHECKPOINTS, timeout_s=None)
    if not video.exists():
        raise StepFailed(f"The NeRF trained but the fly-through video was not made (log: {out / 'nerf_render.log'})")
    return video


def viewer_command(folder: Path) -> list:  # run with TRUSTED_CHECKPOINTS in the environment
    config = Path((folder / "nerf" / "config_path.txt").read_text())
    return [_bin("ns-viewer"), "--load-config", config]
