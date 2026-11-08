"""Runs the labelled test photos through the live vision model and prints how it did.

    cd backend && uv run python -m swarm.inspect_eval
    uv run python -m swarm.inspect_eval --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

Needs NVIDIA_API_KEY and the network, so it isn't part of the unit tests. The photos are simulated
(see swarm.inspect_fixtures): these numbers don't show how the model does on real drone video.
Pass --folder with a folder holding its own labels.json to score real frames the same way.

--demo runs the whole inspector instead: three simulated cameras hover over swarm.sim_cameras'
demo room, and the score says which photos really had staged damage in view. It also needs the key.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.vision import VisionError, analyze, prompt_for
from swarm.inspect_fixtures import FOLDER


def score(rows: list[dict]) -> dict:
    """Counts over rows of {damage, answer}. A row whose answer is None got no answer at all."""
    answered = [r for r in rows if r["answer"] is not None]
    right = sum(r["answer"] == r["damage"] for r in answered)
    return {"images": len(rows), "answered": len(answered), "no_answer": len(rows) - len(answered),
            "correct": right, "accuracy": right / len(rows) if rows else 0.0,
            "false_alarms": sum(r["answer"] is True and not r["damage"] for r in answered),
            "misses": sum(r["answer"] is False and r["damage"] for r in answered)}


def _one(row: dict, folder: Path, model: str | None) -> dict:
    start = time.monotonic()
    out = {**row, "answer": None, "severity": "", "confidence": 0.0, "summary": "", "model": "", "parsed": ""}
    try:
        verdict = analyze([(folder / row["file"]).read_bytes()], prompt_for(), model_list=[model] if model else None)
        out.update(answer=verdict["damage"], severity=verdict["severity"], confidence=verdict["confidence"],
                   summary=verdict["summary"], model=verdict["model"], parsed=verdict["parsed"])
    except VisionError as error:
        out["summary"] = str(error)
    out["seconds"] = time.monotonic() - start
    return out


def run(folder: Path, model: str | None, workers: int, group: str | None = None) -> list[dict]:
    labels = json.loads((folder / "labels.json").read_text())
    if group:
        labels = [row for row in labels if row.get("group") == group]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda row: _one(row, folder, model), labels))


def report(rows: list[dict]) -> None:
    for row in rows:
        mark = "no answer" if row["answer"] is None else "ok" if row["answer"] == row["damage"] else "WRONG"
        print(f"{mark:9s} {row['file']:32s} truth={'damage' if row['damage'] else 'clean':6s} "
              f"said={row['answer']!s:5s} {row['severity']:6s} conf={row['confidence']:.2f} {row['seconds']:5.1f}s "
              f"{row['parsed']:5s} {row['summary'][:90]}")
    for group in sorted({row.get("group", "main") for row in rows}):
        part = [row for row in rows if row.get("group", "main") == group]
        numbers = score(part)
        print(f"\nSimulated images, group {group}: {numbers['correct']}/{numbers['images']} correct, accuracy {numbers['accuracy']:.0%}, "
              f"false alarms {numbers['false_alarms']}, misses {numbers['misses']}, no answer {numbers['no_answer']}")
    seconds = sorted(row["seconds"] for row in rows)
    used = sorted({row["model"] for row in rows if row["model"]})
    print(f"Median {seconds[len(seconds) // 2]:.1f} s per image, slowest {seconds[-1]:.1f} s. Models that answered: {', '.join(used) or 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--model", help="use only this model, with no fallback")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--group", help="only this group of photos: main or words")
    parser.add_argument("--json", type=Path, help="also write every row to this file")
    args = parser.parse_args()
    rows = run(args.folder, args.model, args.workers, args.group)
    report(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
