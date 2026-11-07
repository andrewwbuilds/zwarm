import json
import unittest
from pathlib import Path

from sim.config import DEFAULT_FIRMWARE, FIRMWARES
from sim.pilot import FLY, Pilot
from sim.rc_frames import hy_packet
from sim.scene import Scene
from sim.simulator import Simulator

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "s20_capture.json").read_text())
CLIENT = ("172.16.10.2", 50000)


def simulator(firmware: str = DEFAULT_FIRMWARE) -> Simulator:
    return Simulator(firmware=firmware, scene=Scene(name="open"), seed=1)


class CapturedRepliesTests(unittest.TestCase):
    """Every byte here was captured from the user's real drone."""

    def test_each_query_gets_the_exact_captured_reply(self):
        for query, reply in CAPTURE["udp_8080_replies"].items():
            replies = simulator().receive_udp(CLIENT, bytes.fromhex(query))
            self.assertEqual(replies[0], bytes.fromhex(reply), query)

    def test_every_datagram_is_answered_with_the_board_status(self):
        sim = simulator()
        for packet in (b"\x0f", b"\x28", b"\x2c", b"\x27", b"\x08\x7f\x00\x00\x01", b"\x09\x7f\x00\x00\x01",
                       hy_packet(), b"nonsense", bytes(100)):
            replies = sim.receive_udp(CLIENT, packet)
            self.assertTrue(replies, packet)
            status = replies[-1]
            self.assertEqual(status[:4], bytes.fromhex(CAPTURE["status_reply_prefix"]), packet)
            self.assertEqual(len(status), 5)

    def test_query_2c_gets_only_the_status_reply(self):
        self.assertEqual(len(simulator().receive_udp(CLIENT, b"\x2c")), 1)

    def test_status_value_stays_in_the_range_seen_at_measured_signal_strengths(self):
        sim = simulator()
        seen = CAPTURE["status_values_seen"]
        for distance in (0.5, 1.0, 2.0):
            sim.link.pilot_distance_m = distance
            for _ in range(50):
                sim.step(0.05)
                value = sim.receive_udp(CLIENT, b"\x2c")[-1][4]
                self.assertGreaterEqual(value, min(seen) - 8)
                self.assertLessEqual(value, max(seen) + 8)

    def test_tcp_heartbeat_gets_the_captured_reply(self):
        reply = simulator().board.on_tcp(bytes.fromhex(CAPTURE["tcp_8888_heartbeat"]))
        self.assertEqual(reply, bytes.fromhex(CAPTURE["tcp_8888_reply"]))

    def test_no_flight_controller_frames_by_default(self):
        # Measured: 28 seconds of asking, including with control packets flowing, produced none.
        sim = simulator()
        sim.receive_udp(CLIENT, b"\x09\x7f\x00\x00\x01")
        for _ in range(int(5 / 0.01)):
            sim.step(0.01)
        self.assertEqual(sim.board.take_outbox(), [])


class UartBridgeTests(unittest.TestCase):
    def test_control_packets_reach_the_flight_controller_after_the_researched_delay(self):
        sim = simulator()
        sim.receive_udp(CLIENT, hy_packet(flags=0))
        sim.step(0.005)
        self.assertIsNone(sim.fc.last_frame_time)
        for _ in range(20):
            sim.step(0.001)
        self.assertIsNotNone(sim.fc.last_frame_time)

    def test_flow_reports_use_the_researched_frame_at_about_22_hz(self):
        sim = simulator()
        frames = []
        original = sim.fc.on_uart
        sim.fc.on_uart = lambda t, data: (frames.append(data), original(t, data))
        for _ in range(100):
            sim.step(0.01)
        flow = [f for f in frames if f[:2] == b"\x77\x0e"]
        self.assertAlmostEqual(len(flow), 22, delta=2)
        for frame in flow:
            self.assertEqual(len(frame), 14)
            self.assertEqual(frame[13], 0x99)
            checksum = 0
            for value in frame[2:12]:
                checksum ^= value
            self.assertEqual(frame[12], checksum)

    def test_stationary_flow_frame_matches_the_research_notes(self):
        sim = simulator()
        frames = []
        original = sim.fc.on_uart
        sim.fc.on_uart = lambda t, data: (frames.append(data), original(t, data))
        for _ in range(20):
            sim.step(0.01)
        self.assertEqual(frames[0].hex(), "770e0000000000000000000000" + "99")


class VideoControlTests(unittest.TestCase):
    def test_heartbeat_payload_sets_where_video_goes(self):
        sim = simulator()
        self.assertIsNone(sim.board.video_target(sim.t))
        sim.receive_udp(("10.0.0.9", 1234), b"\x08\xac\x10\x0a\x02")
        self.assertEqual(sim.board.video_target(sim.t), ("172.16.10.2", 8888))

    def test_video_outlives_upstreams_two_second_heartbeat_interval(self):
        sim = simulator()
        sim.receive_udp(CLIENT, b"\x08\xac\x10\x0a\x02")
        for _ in range(int(2.5 / 0.01)):
            sim.step(0.01)
        self.assertIsNotNone(sim.board.video_target(sim.t))

    def test_video_stops_when_heartbeats_stop_for_good(self):
        sim = simulator()
        sim.receive_udp(CLIENT, b"\x08\xac\x10\x0a\x02")
        for _ in range(int(8 / 0.01)):
            sim.step(0.01)
        self.assertIsNone(sim.board.video_target(sim.t))


class TelemetryTests(unittest.TestCase):
    def test_telemetry_firmware_sends_battery_and_airborne_flag_as_captured(self):
        pilot = Pilot(firmware="telemetry")
        pilot.fly(2.5)
        pilot.sim.receive_udp(CLIENT, b"\x09\xac\x10\x0a\x02")
        pilot.fly(2.0)
        frames = pilot.sim.board.take_outbox()
        self.assertTrue(frames)
        (address, frame) = frames[-1]
        self.assertEqual(address, ("172.16.10.2", 8081))
        self.assertEqual(len(frame), 10)
        self.assertEqual(frame[0], 0x66)
        self.assertGreater(frame[1], 90)
        self.assertEqual(frame[2] & 0x04, 0)
        checksum = 0
        for value in frame[1:9]:
            checksum ^= value
        self.assertEqual(frame[9], checksum)
        pilot.sim.receive_udp(CLIENT, b"\x09\xac\x10\x0a\x02")
        pilot.hold(FLY, 1.0)
        pilot.sim.receive_udp(CLIENT, b"\x09\xac\x10\x0a\x02")
        pilot.fly(3.0)
        (_, frame) = pilot.sim.board.take_outbox()[-1]
        self.assertEqual(frame[2] & 0x04, 0x04)


class FirmwareListTests(unittest.TestCase):
    def test_default_firmware_matches_what_was_measured(self):
        firmware = FIRMWARES[DEFAULT_FIRMWARE]
        self.assertFalse(firmware.status_frames)
        one_packet = CAPTURE["takeoff"]["probe_single_packet_at_80hz_started_motors"]
        self.assertEqual(firmware.fly_min_hold_s < 1 / 80, one_packet)

    def test_hold_hypothesis_still_needs_more_than_one_packet(self):
        self.assertGreater(FIRMWARES["hold"].fly_min_hold_s, 0.0125)


if __name__ == "__main__":
    unittest.main()
