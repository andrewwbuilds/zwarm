import io
import threading
import time
import unittest

from swarm.frames import StreamFrames, jpeg_chunks

ONE = b"\xff\xd8first\xff\xd9"
TWO = b"\xff\xd8second\xff\xd9"


class Blocking(io.BytesIO):
    """A stream that delivers its frames, then stays open until told to stop, like a live camera."""

    def __init__(self, data: bytes, release: threading.Event):
        super().__init__(data)
        self.release = release

    def read(self, n=-1):
        data = super().read(n)
        if not data:
            self.release.wait(2.0)
        return data


class JpegChunksTest(unittest.TestCase):
    def test_finds_frames_split_across_reads(self):
        stream = io.BytesIO(b"--frame\r\n" + ONE + b"\r\n--frame\r\n" + TWO)
        self.assertEqual(list(jpeg_chunks(stream, chunk=3)), [ONE, TWO])


class StreamFramesTest(unittest.TestCase):
    def test_keeps_only_the_newest_frame(self):
        release = threading.Event()
        frames = StreamFrames({"a": "http://pi"}, opener=lambda url: Blocking(ONE + TWO, release))
        frames.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and (frames.latest("a") is None or frames.latest("a").number < 2):
            time.sleep(0.01)
        frame = frames.latest("a")
        release.set()
        frames.stop()
        self.assertEqual(frame.jpeg, TWO)
        self.assertEqual(frame.number, 2)

    def test_unknown_drone_and_dead_stream_give_none_without_raising(self):
        def refuse(url):
            raise OSError("no route to host")

        frames = StreamFrames({"a": "http://pi"}, opener=refuse)
        frames.start()
        time.sleep(0.05)
        self.assertIsNone(frames.latest("a"))
        self.assertIsNone(frames.latest("nobody"))
        frames.stop()


if __name__ == "__main__":
    unittest.main()
