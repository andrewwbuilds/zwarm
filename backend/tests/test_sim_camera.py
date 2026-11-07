import json
import math
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from models.s2x_video_model import S2xVideoModel
from protocols.s2x_video_protocol import S2xVideoProtocolAdapter
from sim.camera import Camera, Pose, VideoEncoder, packetize
from sim.scene import Box, Scene
from sim.scenes import SCENES

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "s20_capture.json").read_text())
LEVEL = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def rotation(roll=0.0, pitch=0.0, yaw=0.0):
    """Body-to-world rotation, built the same way as the physics model (Z, then Y, then X)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr)


OPEN = Scene(name="open", floor_material="grass")


def sky_mask(camera: Camera, pose: Pose):
    _, objects, _ = camera.trace(pose)
    return objects == Camera.SKY


class WireFormatTests(unittest.TestCase):
    """The captured headers came from the user's real drone."""

    def _envelope(self, jpeg=b"\xff\xd8" + bytes(34197) + b"\xff\xd9"):
        return VideoEncoder(seed=1).envelope(jpeg, sequence=0xFD, camera_counter=0x83, time_s=0.0)

    def test_envelope_fields_sit_where_the_capture_has_them(self):
        real = bytes.fromhex(CAPTURE["video"]["first_chunk_heads"][0].replace(" ", ""))[8:]
        ours = self._envelope(b"\xff\xd8" + bytes(0x8599 - 4) + b"\xff\xd9")
        self.assertEqual(ours[:4], real[:4])  # start code and frame kind
        self.assertEqual(ours[6], real[6])  # camera frame counter
        self.assertEqual(ours[10], real[10])  # sequence number
        self.assertEqual(ours[11:14], real[11:14])
        self.assertEqual(ours[16:20], real[16:20])  # JPEG length
        self.assertEqual(ours[24:28], real[24:28])  # advertises 640x360, whatever the JPEG is

    def test_envelope_clock_counts_microseconds_like_the_capture(self):
        # The two captured frames were 102 sequence numbers and 6.03 s apart: 16.9 fps of microseconds.
        heads = [bytes.fromhex(h.replace(" ", ""))[8:] for h in CAPTURE["video"]["first_chunk_heads"]]
        real_delta = int.from_bytes(heads[1][32:40], "little") - int.from_bytes(heads[0][32:40], "little")
        self.assertAlmostEqual(real_delta / 1e6, 6.03, delta=0.01)
        encoder = VideoEncoder(seed=1)
        first = encoder.envelope(b"\xff\xd8\xff\xd9", sequence=0, camera_counter=0, time_s=1.0)
        second = encoder.envelope(b"\xff\xd8\xff\xd9", sequence=1, camera_counter=1, time_s=1.5)
        self.assertEqual(int.from_bytes(second[32:40], "little") - int.from_bytes(first[32:40], "little"), 500000)

    def test_chunks_match_the_captured_layout(self):
        real = bytes.fromhex(CAPTURE["video"]["first_chunk_heads"][0].replace(" ", ""))
        chunks = packetize(0x04A5, self._envelope(b"\xff\xd8" + bytes(0x8599 - 4) + b"\xff\xd9"))
        self.assertEqual(len(chunks), CAPTURE["video"]["chunks_per_frame"])
        self.assertEqual(chunks[0][:8], real[:8])
        for index, chunk in enumerate(chunks):
            self.assertEqual(int.from_bytes(chunk[6:8], "little"), len(chunk))
            self.assertEqual(chunk[5], index)
            self.assertEqual(chunk[-2:], b"##")
            if index < len(chunks) - 1:
                self.assertEqual(len(chunk), CAPTURE["video"]["chunk_bytes"])

    def test_turbodrones_real_parser_recovers_the_exact_jpeg(self):
        camera = Camera(OPEN, render_scale=0.25)
        encoder = VideoEncoder(seed=1)
        image = camera.render(Pose(0, 0, 1, LEVEL))
        envelope, jpeg = encoder.encode(image, time_s=0.0)
        chunks = packetize(7, envelope)
        adapter = S2xVideoProtocolAdapter.__new__(S2xVideoProtocolAdapter)
        adapter.model = S2xVideoModel()
        frame = None
        for chunk in reversed(chunks):  # out of order, as UDP may deliver them
            frame = adapter.handle_payload(chunk) or frame
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, jpeg)


class EncoderTests(unittest.TestCase):
    def test_frames_are_decodable_at_the_captured_resolution(self):
        camera = Camera(SCENES["room"](), render_scale=0.5)
        envelope, jpeg = VideoEncoder(seed=1).encode(camera.render(Pose(0, 0, 1, LEVEL)), time_s=0.0)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((decoded.shape[1], decoded.shape[0]), tuple(CAPTURE["video"]["jpeg_size"]))
        self.assertEqual(int.from_bytes(envelope[16:20], "little"), len(jpeg))

    def test_frame_sizes_settle_inside_the_captured_range(self):
        scene = SCENES["room"]()
        camera = Camera(scene, render_scale=0.5)
        encoder = VideoEncoder(seed=1)
        limits = CAPTURE["video"]["jpeg_bytes"]
        sizes = []
        for i in range(40):
            _, jpeg = encoder.encode(camera.render(Pose(-1 + i * 0.02, 0, 1.2, LEVEL)), time_s=i / 17)
            sizes.append(len(jpeg))
        for size in sizes[-20:]:
            self.assertGreaterEqual(size, limits["min"])
            self.assertLessEqual(size, limits["max"])

    def test_sequence_and_frame_counters_advance(self):
        encoder = VideoEncoder(seed=1)
        image = np.zeros((656, 1184, 3), np.uint8)
        first, _ = encoder.encode(image, time_s=0.0)
        second, _ = encoder.encode(image, time_s=0.06)
        self.assertEqual((second[10] - first[10]) & 0xFF, 1)
        self.assertGreaterEqual((second[6] - first[6]) & 0xFF, 1)


class GeometryTests(unittest.TestCase):
    def test_distance_to_a_wall_ahead_is_exact(self):
        scene = Scene(name="s", boxes=(Box((3, -1, 0), (4, 1, 3), "wall"),))
        camera = Camera(scene, render_scale=0.25)
        distances, objects, _ = camera.trace(Pose(0, 0, 1, LEVEL))
        height, width = distances.shape
        self.assertAlmostEqual(float(distances[height // 2, width // 2]), 3.0, delta=0.02)
        self.assertEqual(objects[height // 2, width // 2], Camera.FIRST_BOX)

    def test_camera_can_see_its_height_above_the_floor(self):
        camera = Camera(OPEN, render_scale=0.25)
        distances, objects, _ = camera.trace(Pose(0, 0, 1.2, rotation(pitch=math.pi / 2)))
        height, width = distances.shape
        self.assertAlmostEqual(float(distances[height // 2, width // 2]), 1.2, delta=0.02)
        self.assertEqual(objects[height // 2, width // 2], Camera.FLOOR)

    def test_something_on_the_left_appears_on_the_left(self):
        scene = Scene(name="s", boxes=(Box((3, 0.5, 0), (4, 1.5, 3), "wall"),))
        camera = Camera(scene, render_scale=0.25)
        _, objects, _ = camera.trace(Pose(0, 0, 1, LEVEL))
        columns = np.where((objects == Camera.FIRST_BOX).any(axis=0))[0]
        self.assertLess(columns.max(), objects.shape[1] / 2)

    def test_something_high_appears_at_the_top(self):
        scene = Scene(name="s", boxes=(Box((3, -1, 1.5), (4, 1, 3), "wall"),))
        camera = Camera(scene, render_scale=0.25)
        _, objects, _ = camera.trace(Pose(0, 0, 1, LEVEL))
        rows = np.where((objects == Camera.FIRST_BOX).any(axis=1))[0]
        self.assertLess(rows.max(), objects.shape[0] / 2)

    def test_turning_left_moves_the_view_to_the_right(self):
        scene = Scene(name="s", boxes=(Box((3, -0.3, 0), (4, 0.3, 3), "wall"),))
        camera = Camera(scene, render_scale=0.25)
        _, straight, _ = camera.trace(Pose(0, 0, 1, LEVEL))
        _, turned, _ = camera.trace(Pose(0, 0, 1, rotation(yaw=math.radians(20))))
        before = np.where((straight == Camera.FIRST_BOX).any(axis=0))[0].mean()
        after = np.where((turned == Camera.FIRST_BOX).any(axis=0))[0].mean()
        self.assertGreater(after, before + 20)

    def test_nose_down_shows_less_sky(self):
        camera = Camera(OPEN, render_scale=0.25)
        level = sky_mask(camera, Pose(0, 0, 1.2, LEVEL)).sum()
        nose_down = sky_mask(camera, Pose(0, 0, 1.2, rotation(pitch=math.radians(20)))).sum()
        self.assertLess(nose_down, level * 0.75)

    def test_rolling_right_lowers_the_right_side_of_the_view(self):
        camera = Camera(OPEN, render_scale=0.25)
        sky = sky_mask(camera, Pose(0, 0, 1.2, rotation(roll=math.radians(20))))
        width = sky.shape[1]
        self.assertGreater(sky[:, : width // 4].sum(), sky[:, 3 * width // 4 :].sum())

    def test_room_walls_surround_the_camera(self):
        scene = SCENES["room"]()
        camera = Camera(scene, render_scale=0.25)
        _, objects, _ = camera.trace(Pose(0, 0, 1.2, LEVEL))
        self.assertFalse((objects == Camera.SKY).any())


class RenderingTests(unittest.TestCase):
    def test_the_view_changes_when_the_drone_moves(self):
        camera = Camera(SCENES["room"](), render_scale=0.5)
        first = camera.render(Pose(-1, 0, 1.2, LEVEL)).astype(np.int16)
        second = camera.render(Pose(1, 0, 1.2, LEVEL)).astype(np.int16)
        self.assertGreater(np.abs(first - second).mean(), 5.0)

    def test_output_is_clearly_marked_as_simulated(self):
        camera = Camera(OPEN, render_scale=0.25)
        marked = camera.render(Pose(0, 0, 1, LEVEL))
        self.assertGreater(int(marked[-40:, -220:].max()), 200)

    def test_rendering_and_encoding_keep_up_with_the_measured_frame_rate(self):
        camera = Camera(SCENES["room"]())
        encoder = VideoEncoder(seed=1)
        started = time.perf_counter()
        for i in range(20):
            encoder.encode(camera.render(Pose(-1 + 0.05 * i, 0, 1.2, rotation(yaw=0.05 * i))), time_s=i / 17)
        per_frame = (time.perf_counter() - started) / 20
        self.assertLess(per_frame, 1 / CAPTURE["video"]["fps"]["max"])


class SceneListTests(unittest.TestCase):
    def test_every_scene_builds_and_renders(self):
        for name, build in SCENES.items():
            scene = build()
            camera = Camera(scene, render_scale=0.25)
            image = camera.render(Pose(scene.start[0], scene.start[1], 1.2, rotation(yaw=scene.start[2])))
            self.assertEqual(image.shape, (656, 1184, 3), name)
            self.assertGreater(float(image.std()), 5.0, name)


if __name__ == "__main__":
    unittest.main()
