import errno
import socket
import threading
import unittest

from models.s2x_video_model import S2xVideoModel
from protocols.s2x_video_protocol import S2xVideoProtocolAdapter


class S2xVideoProtocolTests(unittest.TestCase):
    def _adapter(self):
        adapter = S2xVideoProtocolAdapter.__new__(S2xVideoProtocolAdapter)
        adapter.model = S2xVideoModel()
        return adapter

    def _packet(self, frame_id: int, total_chunks: int, chunk_id: int, body: bytes) -> bytes:
        payload = (
            b"\x40\x40"
            + frame_id.to_bytes(2, "little")
            + bytes([total_chunks, chunk_id])
        )
        packet_len = 8 + len(body) + 2
        return payload + packet_len.to_bytes(2, "little") + body + b"##"

    def test_keepalive_survives_send_errors(self):
        # macOS raises EHOSTUNREACH while Wi-Fi is switching networks. If that
        # kills the heartbeat thread, video never comes back.
        adapter = self._adapter()
        stop = threading.Event()
        calls = []

        def flaky_send():
            calls.append(1)
            if len(calls) == 1:
                raise OSError(errno.EHOSTUNREACH, "No route to host")
            stop.set()

        adapter.send_start_command = flaky_send
        thread = threading.Thread(target=adapter._keepalive_loop, args=(0.001, stop), daemon=True)
        thread.start()
        thread.join(timeout=2.0)

        self.assertEqual(len(calls), 2)

    def test_start_command_tracks_current_local_ip(self):
        # A Mac's single radio often joins the drone's Wi-Fi after the backend
        # is already running, so the address must not be cached from startup.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as drone:
            drone.bind(("127.0.0.1", 0))
            drone.settimeout(2.0)
            adapter = S2xVideoProtocolAdapter(
                drone_ip="127.0.0.1", control_port=drone.getsockname()[1], video_port=0
            )
            try:
                adapter.local_ip = "192.168.1.139"
                adapter._discover_local_ip = lambda: "172.16.10.2"

                adapter.send_start_command()

                self.assertEqual(drone.recv(16), bytes([0x08, 172, 16, 10, 2]))
            finally:
                adapter.stop()

    def test_packet_buffer_is_bounded(self):
        # Nothing drains this buffer unless packet dumping is on.
        adapter = S2xVideoProtocolAdapter(drone_ip="127.0.0.1", video_port=0)
        try:
            self.assertIsNotNone(adapter._pkt_buffer.maxlen)
        finally:
            adapter.stop()

    def test_emits_frame_when_all_declared_chunks_arrive(self):
        adapter = self._adapter()
        first = self._packet(0x1234, 2, 0, b"\xff\xd8hello")
        second = self._packet(0x1234, 2, 1, b" world\xff\xd9")

        self.assertIsNone(adapter.handle_payload(first))
        frame = adapter.handle_payload(second)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_id, 0x1234)
        self.assertEqual(frame.data, b"\xff\xd8hello world\xff\xd9")
        self.assertEqual(frame.format, "jpeg")

    def test_accepts_out_of_order_chunks(self):
        adapter = self._adapter()
        second = self._packet(7, 2, 1, b" tail\xff\xd9")
        first = self._packet(7, 2, 0, b"\xff\xd8head")

        self.assertIsNone(adapter.handle_payload(second))
        frame = adapter.handle_payload(first)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b"\xff\xd8head tail\xff\xd9")

    def test_rejects_mismatched_declared_length(self):
        adapter = self._adapter()
        packet = bytearray(self._packet(1, 1, 0, b"\xff\xd8x\xff\xd9"))
        packet[6:8] = (len(packet) + 1).to_bytes(2, "little")

        self.assertIsNone(adapter.handle_payload(bytes(packet)))


if __name__ == "__main__":
    unittest.main()
