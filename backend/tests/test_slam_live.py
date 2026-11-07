import io
import unittest

import cv2
import numpy as np

from slam.live import draw_map, jpeg_frames


def jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), value, np.uint8))
    return buf.tobytes()


class MjpegTests(unittest.TestCase):
    def test_splits_a_multipart_stream_into_frames(self):
        frames = [jpeg(10), jpeg(120), jpeg(250)]
        body = b"".join(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n" for f in frames)
        self.assertEqual(list(jpeg_frames(io.BytesIO(body), chunk=7)), frames)

    def test_a_truncated_last_frame_is_dropped(self):
        body = jpeg(10) + jpeg(20)[:-5]
        self.assertEqual(len(list(jpeg_frames(io.BytesIO(body), chunk=16))), 1)


class MapTests(unittest.TestCase):
    def test_draws_a_path_and_waits_when_there_is_none(self):
        self.assertEqual(draw_map(np.zeros((1, 3)), None, []).shape, (480, 480, 3))
        path = np.stack([np.linspace(0, 3, 20), np.zeros(20), np.linspace(0, 2, 20)], axis=1)
        image = draw_map(path, 0.5, [], unit="m")
        self.assertLess(int(image.min()), 200)  # something was drawn


if __name__ == "__main__":
    unittest.main()
