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
import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

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


class _Spy:
    """Reads frames for the inspector and remembers where each one was taken."""

    def __init__(self, frames):
        self.frames = frames
        self.where: dict[bytes, tuple] = {}

    def latest(self, drone: str):
        frame = self.frames.latest(drone)
        if frame is not None:
            self.where.setdefault(frame.jpeg, frame.truth)
        return frame


def demo(model: str | None, dwell_s: float = 4.0, wait_s: float = 180.0) -> dict:
    """Fly no one: hover three simulated cameras over the demo room and score the whole inspector."""
    from agent.vision import analyze as ask
    from swarm.eyes import DEFAULT_TILT, in_view
    from swarm.inspect import FindingStore, Inspector
    from swarm.sim_cameras import SimFrames, StillFleet, demo_world, view_of

    world = demo_world()
    views = [view_of(d.where, 1.0, math.atan2(-d.facing[1], -d.facing[0]) if any(d.facing) else 0.0) for d in world.damage]
    views += [view_of(spot, 1.0, yaw) for spot, yaw in (((-3.0, 1.5, 0.0), 0.0), ((-2.5, -0.5, 0.0), 2.4), ((0.8, -0.8, 0.0), -1.0),
                                                        ((3.3, -1.8, 0.0), -0.6), ((-2.8, 2.4, 0.0), 1.57), ((1.2, 2.5, 0.0), 1.2))]
    names = ("a", "b", "c")
    fleet = StillFleet(world.scene, {name: views[i] for i, name in enumerate(names)})
    frames = SimFrames(fleet, rate_hz=5.0)
    spy = _Spy(frames)
    answers: list[tuple[bytes, dict]] = []
    lock = threading.Lock()

    def analyze(jpegs, prompt):
        verdict = ask(jpegs, prompt, model_list=[model] if model else None)
        with lock:
            answers.append((jpegs[0], verdict))
        return verdict

    events = []
    with tempfile.TemporaryDirectory() as folder:
        store = FindingStore(Path(folder) / "findings.sqlite")
        inspector = Inspector(store, spy, analyze=analyze, on_event=lambda *event: events.append(event))
        frames.start()
        inspector.start("anything damaged")
        slowest, ticks = 0.0, 0
        for i in range(0, len(views), len(names)):
            batch = views[i:i + len(names)]
            for name, view in zip(names, batch):
                hover = fleet.drones[name]
                hover.x, hover.y, hover.z, hover.heading = view
            arrived = time.monotonic()
            while time.monotonic() - arrived < dwell_s:
                fleet.t += 0.05
                poses = {name: _pose_of(fleet.drones[name]) for name, _ in zip(names, batch)}
                if time.monotonic() - arrived >= 0.5:  # the mission only feeds drones that have settled
                    start = time.perf_counter()
                    inspector.tick(fleet.t, poses)
                    slowest = max(slowest, time.perf_counter() - start)
                    ticks += 1
                time.sleep(0.05)
        inspector.stop()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            status = inspector.status()
            if not status["queued"] and not status["busy"]:
                break
            time.sleep(0.25)
        status = inspector.status()
        findings = store.list()
        inspector.close()
        store.close()
    frames.stop()

    def visible(where) -> list[str]:
        x, y, z, yaw = where
        return [d.name for d in world.damage if in_view((x, y, z), yaw, np.asarray([d.where]), tilt=DEFAULT_TILT)[0]]

    with_damage = hits = clean = alarms = 0
    found: set[str] = set()
    rows = []
    for jpeg, verdict in answers:
        names_seen = visible(spy.where[jpeg])
        if names_seen:
            with_damage += 1
            hits += verdict["damage"]
            if verdict["damage"]:
                found.update(names_seen)
        else:
            clean += 1
            alarms += verdict["damage"]
        rows.append((names_seen, verdict))
    return {"checked": len(answers), "with_damage": with_damage, "reported": hits, "clean": clean, "false_alarms": alarms,
            "staged": [d.name for d in world.damage], "found": sorted(found), "findings": findings, "status": status, "rows": rows,
            "slowest_tick_ms": slowest * 1000.0, "ticks": ticks, "events": len(events)}


def _pose_of(hover):
    return type("Pose", (), {"p": (hover.x, hover.y, hover.z), "yaw": hover.heading, "sigma": 0.05, "fix_age": 0.0})()


def report_demo(result: dict) -> None:
    for names, verdict in result["rows"]:
        truth = ", ".join(names) or "nothing staged"
        print(f"said={verdict['damage']!s:5s} conf={verdict['confidence']:.2f} in view: {truth:60.60s} | {verdict['summary'][:80]}")
    print(f"\nSimulated demo room: {result['checked']} photos checked. Staged damage in view in {result['with_damage']}, "
          f"reported in {result['reported']}. No staged damage in {result['clean']}, false alarms {result['false_alarms']}.")
    print(f"Staged damage reported at least once: {len(result['found'])}/{len(result['staged'])}. Not reported: "
          f"{', '.join(sorted(set(result['staged']) - set(result['found']))) or 'none'}.")
    print(f"Findings stored: {len(result['findings'])}, all {sorted({f['status'] for f in result['findings']}) or ['-']}.")
    print(f"Inspector: {result['status']}")
    print(f"Slowest of {result['ticks']} ticks: {result['slowest_tick_ms']:.2f} ms.")


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
