import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mapping.pointcloud import read_points, write_ply

POINTS_TXT = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 4, mean track length: 3.5
1 1.0 2.0 3.0 255 0 0 0.5 1 0 2 1 3 2
2 -1.0 0.5 0.0 0 255 0 2.9 1 5 2 6 3 7 4 8
3 0.0 0.0 1.0 0 0 255 0.4 1 9 2 10
4 5.0 5.0 5.0 9 9 9 0.3 1 11
"""


class ReadPointsTests(unittest.TestCase):
    def test_reads_positions_and_colours(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "points3D.txt"
            path.write_text(POINTS_TXT)
            xyz, rgb = read_points(path, max_error=1.5, min_track=2)
        self.assertEqual(xyz.tolist(), [[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        self.assertEqual(rgb.tolist(), [[255, 0, 0], [0, 0, 255]])

    def test_drops_points_with_high_error_or_a_single_photo(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "points3D.txt"
            path.write_text(POINTS_TXT)
            xyz, _ = read_points(path, max_error=1.5, min_track=2)
        self.assertEqual(len(xyz), 2)  # point 2 has error 2.9, point 4 is seen once


class WritePlyTests(unittest.TestCase):
    def test_writes_a_binary_ply_a_viewer_can_read(self):
        import numpy as np
        xyz = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
        rgb = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            write_ply(path, xyz, rgb)
            data = path.read_bytes()
        header, body = data.split(b"end_header\n")
        self.assertIn(b"element vertex 2", header)
        x, y, z, r, g, b = struct.unpack("<fffBBB", body[:15])
        self.assertEqual((x, y, z, r, g, b), (1.0, 2.0, 3.0, 10, 20, 30))
        self.assertEqual(len(body), 30)



class ViewerTests(unittest.TestCase):
    def test_the_viewer_page_carries_every_point(self):
        import numpy as np
        from mapping.cloud_view import read_ply, write_viewer
        xyz = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], np.float32)
        rgb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.uint8)
        with TemporaryDirectory() as tmp:
            ply, html = Path(tmp) / "c.ply", Path(tmp) / "c.html"
            write_ply(ply, xyz, rgb)
            back_xyz, back_rgb = read_ply(ply)
            self.assertEqual(write_viewer(ply, html), 3)
            page = html.read_text()
        self.assertEqual(back_xyz.tolist(), xyz.tolist())
        self.assertEqual(back_rgb.tolist(), rgb.tolist())
        self.assertIn("3 points", page)
        self.assertNotIn("__DATA__", page)


if __name__ == "__main__":
    unittest.main()
