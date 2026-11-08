"""Photos to a 3D splat: COLMAP finds where each photo was taken, Brush trains the splat on the Mac's GPU."""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

BRUSH = Path.home() / ".local/share/brush-splat/v0.3.0/brush-app-aarch64-apple-darwin/brush_app"
EXHAUSTIVE_LIMIT = 250  # matching every pair of more photos than this takes too long
STEP_TIMEOUT_S = float(os.environ.get("MAP_STEP_TIMEOUT_S") or 1800.0)  # one COLMAP step. Without a limit a stuck one blocks every later map


class ToolMissing(RuntimeError):
    pass


class StepFailed(RuntimeError):
    pass


@dataclass
class Model:
    path: Path
    registered: int
    points: int
    error_px: float


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise ToolMissing(f"{name} is not installed. Run: brew install {name}")
    return found


def run(command: list, log: Path, env: dict | None = None, timeout_s: float | None = STEP_TIMEOUT_S) -> None:
    """Run a command, sending its output to a log file. Raises with the log's tail on failure or on running out of time."""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "w") as f:
            result = subprocess.run([str(c) for c in command], stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise StepFailed(f"{Path(str(command[0])).name} {command[1] if len(command) > 1 else ''} was stopped after {timeout_s:.0f} s (log: {log})") from None
    if result.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
        raise StepFailed(f"{Path(str(command[0])).name} {command[1] if len(command) > 1 else ''} failed (log: {log}):\n{tail}")


def parse_analysis(text: str) -> tuple[int, int, float]:
    """(registered images, points, mean reprojection error) from `colmap model_analyzer` output."""
    def number(label: str) -> float:
        match = re.search(rf"{label}:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else 0.0
    return int(number("Registered images")), int(number("Points")), number("Mean reprojection error")


def analyze(model: Path) -> Model:
    result = subprocess.run([tool("colmap"), "model_analyzer", "--path", str(model)], capture_output=True, text=True)
    registered, points, error = parse_analysis(result.stdout + result.stderr)
    return Model(model, registered, points, error)


def best_model(models: list[Model]) -> Model | None:
    return max(models, key=lambda m: (m.registered, m.points), default=None)


def registered_names(model: Path, scratch: Path) -> set[str]:
    """File names of the photos a model placed."""
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("colmap"), "model_converter", "--input_path", str(model), "--output_path", str(scratch),
                    "--output_type", "TXT"], capture_output=True)
    names = set()
    images = scratch / "images.txt"
    if images.exists():
        lines = [l for l in images.read_text().splitlines() if l and not l.startswith("#")]
        names = {l.split()[-1] for l in lines[::2]}
    return names


def reconstruct(images: Path, work: Path) -> Model:
    """Find the camera position of every photo. Returns the model that placed the most."""
    colmap = tool("colmap")
    count = len(list(images.glob("*.jpg")))
    if count < 10:
        raise StepFailed(f"Only {count} photos in {images}: a map needs at least 10, and ideally 100 or more.")
    shutil.rmtree(work, ignore_errors=True)
    (work / "sparse").mkdir(parents=True)
    database = work / "database.db"
    run([colmap, "feature_extractor", "--database_path", database, "--image_path", images,
         "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
         "--FeatureExtraction.use_gpu", 0, "--FeatureExtraction.num_threads", 8], work / "features.log")
    if count <= EXHAUSTIVE_LIMIT:
        run([colmap, "exhaustive_matcher", "--database_path", database, "--FeatureMatching.use_gpu", 0], work / "matching.log")
    else:  # photos are in flight order, so neighbours in the list overlap
        run([colmap, "sequential_matcher", "--database_path", database, "--SequentialMatching.overlap", 20,
             "--FeatureMatching.use_gpu", 0], work / "matching.log")
    run([colmap, "mapper", "--database_path", database, "--image_path", images, "--output_path", work / "sparse"],
        work / "mapper.log")
    best = best_model([analyze(p) for p in sorted((work / "sparse").iterdir()) if p.is_dir()])
    if best is None or best.registered < 3:
        raise StepFailed("COLMAP could not link the photos into a 3D model. Usually that means too little overlap or too "
                         "little detail (blank walls). Fly slower and keep textured surfaces in view.")
    return best


def train(images: Path, model: Model, out: Path, steps: int = 7000) -> Path:
    """Train a Gaussian splat on the Mac's GPU. Returns the .ply file."""
    if not BRUSH.exists():
        raise ToolMissing(f"Brush not found at {BRUSH}. Download brush-app-aarch64-apple-darwin from "
                          "https://github.com/ArthurBrussee/brush/releases and unpack it there.")
    dataset = out / "dataset"
    shutil.rmtree(dataset, ignore_errors=True)
    (dataset / "sparse").mkdir(parents=True)
    (dataset / "images").symlink_to(images.resolve())
    shutil.copytree(model.path, dataset / "sparse" / "0")
    splat = out / "splat"
    shutil.rmtree(splat, ignore_errors=True)
    splat.mkdir(parents=True)
    run([BRUSH, dataset, "--total-steps", steps, "--export-every", steps, "--export-path", splat,
         "--export-name", "map_{iter}.ply"], out / "training.log", timeout_s=None)
    exports = sorted(splat.glob("map_*.ply"), key=lambda p: int(re.search(r"map_(\d+)", p.name).group(1)))
    if not exports:
        raise StepFailed(f"Brush finished but wrote no splat (log: {out / 'training.log'})")
    final = out / "map.ply"
    shutil.copy(exports[-1], final)
    return final


OUTPUTS = ("cloud", "splat", "nerf")


def build(folder: Path, steps: int = 7000, nerf_steps: int = 5000, outputs=OUTPUTS, say=print, window: int = 3) -> dict:
    """Run the pipeline and make the chosen outputs. Returns output name -> file.

    If folder/raw holds recorded frames, photos are chosen from them first; otherwise folder/images is used as it is.
    """
    images = folder / "images"
    if (folder / "raw").exists():
        from mapping.curate import curate
        say("Choosing photos from the recorded frames...")
        say("  " + curate(folder, window=window).summary())
    total = len(list(images.glob("*.jpg")))
    say(f"Linking {total} photos (COLMAP, a few minutes)...")
    model = reconstruct(images, folder / "colmap")
    say(f"  placed {model.registered} of {total} photos, {model.points} 3D points, error {model.error_px:.2f} px")
    for warning in coverage_warnings(model, total):
        say("  ! " + warning)
    made = {}
    if "cloud" in outputs:
        from mapping.cloud_view import write_viewer
        from mapping.pointcloud import point_cloud
        count = point_cloud(model.path, folder, tool("colmap"))
        write_viewer(folder / "cloud.ply", folder / "cloud.html")
        made["cloud"] = folder / "cloud.ply"
        say(f"Point cloud: {count} points -> {made['cloud']}")
    if "splat" in outputs:
        say(f"Training the Gaussian splat ({steps} steps, Mac GPU, a few minutes)...")
        made["splat"] = train(images, model, folder, steps)
        say(f"Gaussian splat -> {made['splat']}")
    if "nerf" in outputs:
        from mapping.nerf import train_nerf
        say(f"Training the NeRF ({nerf_steps} steps; about 5 minutes per 1000 steps)...")
        made["nerf"] = train_nerf(folder / "dataset", folder, nerf_steps)
        say(f"NeRF fly-through video -> {made['nerf']}")
    say("View them with: uv run python -m mapping view " + folder.name + " cloud|splat|nerf")
    return made


def coverage_warnings(model: Model, total: int) -> list[str]:
    warnings = []
    share = model.registered / max(total, 1)
    if share < 0.6:
        warnings.append(f"Only {share:.0%} of photos could be linked, so the map will have large gaps. Fly slower, keep "
                        "textured surfaces in view, and overlap your path.")
    elif share < 0.85:
        warnings.append(f"{1 - share:.0%} of photos could not be linked; parts of the room may be missing.")
    if model.points < 8 * model.registered:
        warnings.append("Few 3D points per photo: the scene is probably too plain (blank walls and floors).")
    if model.error_px > 1.5:
        warnings.append(f"High error ({model.error_px:.1f} px): blurry photos or a wobbly camera mount may be hurting accuracy.")
    return warnings
