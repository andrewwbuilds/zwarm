import asyncio
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import web_server
from control import trajectory
from control.strategies import IncrementalStrategy, LinearStrategy
from models.s2x_rc import S2xDroneModel
from protocols.s2x_rc_protocol_adapter import S2xRCProtocolAdapter
from services.flight_controller import FlightController
from sim.config import FlightTuning
from sim.server import Server
from tests.test_web_server_disconnect import FakeWebSocket

TUNE = FlightTuning()


def ends(p: trajectory.Plan, prediction: dict) -> list[tuple]:
    """Where the simulator put the drone at the end of each movement, beside where the plan aimed."""
    groups: dict[int, list] = {}
    for seg in p.segments:
        groups.setdefault(seg.line, []).append(seg)
    out = []
    for segs in groups.values():
        if segs[0].action:
            continue
        done = segs[-1].t + segs[-1].dur
        point = trajectory.pose(prediction["track"], done)
        out.append((segs[0].end, point))
    return out


class ParseTests(unittest.TestCase):
    def test_errors_name_the_line(self):
        for text, fragment in (("takeoff\nfly 1 2", "line 2: unknown command"),
                               ("takeoff\ngoto 1", "line 2: goto takes 2 to 3"),
                               ("takeoff\nhold x", "line 2: arguments must be numbers"),
                               ("land", "line 1: not flying"),
                               ("takeoff\ntakeoff", "line 2: already flying"),
                               ("speed 99", "line 1: speed must be"),
                               ("# nothing", "empty")):
            with self.assertRaises(trajectory.PlanError) as caught:
                trajectory.plan(text)
            self.assertIn(fragment, str(caught.exception))

    def test_comments_and_blank_lines_are_ignored(self):
        p = trajectory.plan("# a\n\ntakeoff  # go\nland\n")
        self.assertEqual([s.action for s in p.segments], ["takeoff", "land"])


class PlanTests(unittest.TestCase):
    def test_forward_leg_sets_pitch_from_the_speed(self):
        p = trajectory.plan("takeoff\nspeed 1\ngoto 2 0\nland")
        leg = p.segments[1]
        self.assertAlmostEqual(leg.dur, 2.0)
        self.assertAlmostEqual(leg.sticks[2], 1 / TUNE.max_speed_mps, places=3)
        self.assertEqual(leg.end[:2], (2.0, 0.0))

    def test_left_is_negative_roll_and_left_turns_are_negative_yaw(self):
        p = trajectory.plan("takeoff\ngoto 0 1\nturn 90\nland")
        self.assertLess(p.segments[1].sticks[3], 0)
        turn = next(s for s in p.segments if s.sticks[1])
        self.assertLess(turn.sticks[1], 0)

    def test_moves_follow_the_nose_after_a_turn(self):
        p = trajectory.plan("takeoff\nturn 90\nmove 1 0\nland")
        move = next(s for s in p.segments if s.text == "move 1 0" and any(s.sticks))
        self.assertGreater(move.sticks[2], 0)
        self.assertAlmostEqual(move.end[0], 0.0, places=6)
        self.assertAlmostEqual(move.end[1], 1.0, places=6)

    def test_small_components_get_their_own_leg_above_the_deadband(self):
        p = trajectory.plan("takeoff\nspeed 2\ngoto 4 0.1\nland")
        legs = [s for s in p.segments if s.text == "goto 4 0.1" and any(s.sticks)]
        self.assertEqual(len(legs), 2)
        self.assertEqual(legs[0].sticks[3], 0.0)
        self.assertAlmostEqual(abs(legs[1].sticks[3]), trajectory.DEADBAND)

    def test_warns_above_the_flow_sensors_range(self):
        p = trajectory.plan("takeoff\nclimb 1\ngoto 0 0 5\nland")
        self.assertTrue(any("flow sensor" in w for w in p.warnings))


class RefineTests(unittest.TestCase):
    def test_refined_legs_reach_their_targets_in_the_simulator(self):
        text = "takeoff\nspeed 1\ngoto 2 0\nturn 90\ngoto 2 2 1.6\nyaw 0\ngoto 0 0 1.2\nland"
        raw = trajectory.plan(text)
        before = ends(raw, trajectory.predict(raw))
        p, prediction = trajectory.refine(trajectory.plan(text))
        after = ends(p, prediction)
        error = lambda pairs: max(math.hypot(g[0] - a[1], g[1] - a[2]) for g, a in pairs)
        self.assertLess(error(after), 0.15)
        self.assertLess(error(after), error(before))
        self.assertTrue(all(abs(trajectory.wrap(g[3] - a[4])) < 3 for g, a in after))
        self.assertEqual(prediction["state"], "READY")


class TurnTests(unittest.TestCase):
    def test_turns_past_half_a_circle_are_refined_as_whole_turns(self):
        p, prediction = trajectory.refine(trajectory.plan("takeoff\nrate 90\nturn 400\nland"))
        track = trajectory.unwrap(prediction["track"])
        turned = track[-1][4] - track[0][4]
        self.assertAlmostEqual(turned, 400, delta=8)


class FakeFlight:
    def __init__(self):
        self.model = S2xDroneModel()
        self.model.set_strategy(IncrementalStrategy())
        self.model.set_speed_index(0)
        self.axes = []

    def set_axes_from(self, source, *axes):
        self.axes.append((source, axes))


class RunnerTests(unittest.TestCase):
    def test_runner_sends_the_plan_and_restores_the_pilots_settings(self):
        flight = FakeFlight()
        p = trajectory.plan("hold 0.1\nturn 10\nhold 0.1")
        with tempfile.TemporaryDirectory() as folder:
            runner = trajectory.Runner(flight, p, Path(folder))
            runner.start()
            self.assertIsInstance(flight.model.strategy, LinearStrategy)
            self.assertEqual(flight.model.speed_index, 2)
            deadline = time.monotonic() + 5
            while runner.running and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(runner.reason, "done")
            self.assertIsInstance(flight.model.strategy, IncrementalStrategy)
            self.assertEqual(flight.model.speed_index, 0)
            self.assertTrue(any(axes[1] != 0 for _, axes in flight.axes))
            self.assertEqual(flight.axes[-1], ("trajectory", (0.0, 0.0, 0.0, 0.0)))
            lines = (Path(folder) / "commands.jsonl").read_text().splitlines()
            self.assertGreater(len(lines), 5)
            self.assertEqual(json.loads((Path(folder) / "result.json").read_text())["reason"], "done")

    def test_pilot_sticks_abort_a_running_plan(self):
        flight = FakeFlight()
        runner = trajectory.Runner(flight, trajectory.plan("hold 5"))
        runner.start()
        sticks = {"type": "axes", "mode": "inc", "throttle": 0, "yaw": 0, "pitch": 1, "roll": 0}
        with mock.patch.object(web_server, "flight_controller", flight), \
                mock.patch.object(web_server, "runner", runner), \
                mock.patch.object(web_server, "plugin_manager", None):
            asyncio.run(web_server.ws_endpoint(FakeWebSocket([sticks])))
        self.assertEqual(runner.reason, "pilot moved the sticks")

    def test_centred_sticks_from_the_browser_do_not_abort(self):
        flight = FakeFlight()
        runner = trajectory.Runner(flight, trajectory.plan("hold 5"))
        runner.start()
        self.addCleanup(runner.stop, "test over")
        idle = {"type": "axes", "mode": "inc", "throttle": 0, "yaw": 0, "pitch": 0, "roll": 0}
        with mock.patch.object(web_server, "flight_controller", flight), \
                mock.patch.object(web_server, "runner", runner), \
                mock.patch.object(web_server, "plugin_manager", None):
            socket = FakeWebSocket([idle, idle])
            asyncio.run(web_server.ws_endpoint(socket))
        self.assertEqual(runner.reason, "web client disconnected")


class LiveRunTests(unittest.TestCase):
    """A refined plan flown in real time through the backend's flight controller and the network simulator."""

    def test_plan_flies_the_simulated_drone_to_its_target(self):
        server = Server(control_port=0, tcp_port=0, video_source_port=0, api_port=0, log=lambda _: None)
        server.start()
        self.addCleanup(server.stop)
        rc = S2xRCProtocolAdapter("127.0.0.1", server.control_port)
        self.addCleanup(rc.sock.close)
        flight = FlightController(S2xDroneModel(), rc, 80.0)
        flight.start()
        self.addCleanup(flight.stop)
        time.sleep(2.6)
        with server.lock:
            scene, start = server.sim.scene, (server.sim.drone.x, server.sim.drone.y, server.sim.drone.yaw())
        p, prediction = trajectory.refine(trajectory.plan("takeoff\nspeed 1\nmove 1.5 0\nland"),
                                          scene=scene, start=start)
        with tempfile.TemporaryDirectory() as folder:
            url = f"http://127.0.0.1:{server.api_port}"
            runner = trajectory.Runner(flight, p, Path(folder), url)
            runner.start()
            deadline = time.monotonic() + p.duration + 5
            while runner.running and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertEqual(runner.reason, "done")
            truth = [json.loads(line) for line in (Path(folder) / "truth.jsonl").read_text().splitlines()]
        with server.lock:
            d, state = server.sim.drone, server.sim.fc.state
        self.assertEqual(state, "READY")
        self.assertTrue(any(t["state"] == "FLYING" for t in truth))
        planned_end = prediction["track"][-1]
        self.assertAlmostEqual(d.x - start[0], 1.5, delta=0.35)
        self.assertAlmostEqual(d.x - start[0], planned_end[1], delta=0.3)


if __name__ == "__main__":
    unittest.main()
