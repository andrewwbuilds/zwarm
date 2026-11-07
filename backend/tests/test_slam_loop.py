import unittest

import cv2
import numpy as np

from slam.geometry import apply_sim3, exp_so3
from slam.loop import DescriptorDatabase, find_loop, ransac_sim3

RNG = np.random.default_rng(11)


def descriptors(n):
    return RNG.integers(0, 256, (n, 32), dtype=np.uint8)


def flip_bits(d, count):
    out = d.copy()
    for row in out:
        for bit in RNG.choice(256, count, replace=False):
            row[bit // 8] ^= 1 << (bit % 8)
    return out


class Sim3RansacTests(unittest.TestCase):
    def test_finds_the_transform_despite_wrong_pairs(self):
        points = RNG.normal(size=(60, 3)) * 3
        transform = (1.3, exp_so3([0.2, -0.3, 0.9]), np.array([1.0, 2.0, -3.0]))
        moved = apply_sim3(transform, points)
        moved[:15] = RNG.normal(size=(15, 3)) * 5
        found, mask = ransac_sim3(points, moved, 0.05, np.random.default_rng(1))
        self.assertAlmostEqual(found[0], 1.3, places=3)
        self.assertGreaterEqual(mask.sum(), 44)
        self.assertFalse(mask[:15].any())

    def test_gives_up_on_unrelated_points(self):
        found = ransac_sim3(RNG.normal(size=(40, 3)), RNG.normal(size=(40, 3)), 0.01, np.random.default_rng(2))
        self.assertTrue(found is None or found[1].sum() < 10)


class LoopTests(unittest.TestCase):
    def build(self):
        database = DescriptorDatabase()
        world = RNG.normal(size=(300, 3)) * 2
        points_xyz = {i: world[i] for i in range(300)}
        base = descriptors(300)
        database.add(0, base[:150], np.arange(150))
        database.add(1, base[100:250], np.arange(100, 250))
        for k in range(2, 6):  # unrelated keyframes later in time
            database.add(k, descriptors(150), np.arange(1000 * k, 1000 * k + 150))
        return database, base, points_xyz

    def test_recognises_a_revisited_place_and_gives_the_drift_transform(self):
        database, base, xyz = self.build()
        drift = (1.15, exp_so3([0.0, 0.0, 0.05]), np.array([0.4, -0.2, 0.1]))
        current_ids = np.arange(5000, 5100)
        seen = np.arange(50, 150)  # these old points are seen again, with an inverse-drifted estimate
        for cur, old in zip(current_ids, seen):
            inverse_s = 1 / drift[0]
            xyz[int(cur)] = inverse_s * drift[1].T @ (xyz[int(old)] - drift[2])
        query = flip_bits(base[seen], 12)
        found = find_loop(database, query, current_ids, xyz, newest_allowed=3, rng=np.random.default_rng(3))
        self.assertIsNotNone(found)
        self.assertEqual(found.keyframe, 0)
        self.assertAlmostEqual(found.transform[0], 1.15, places=2)
        self.assertGreaterEqual(len(found.pairs), 40)

    def test_a_new_place_is_not_a_loop(self):
        database, _, xyz = self.build()
        ids = np.arange(7000, 7100)
        for i in ids:
            xyz[int(i)] = RNG.normal(size=3)
        self.assertIsNone(find_loop(database, descriptors(100), ids, xyz, newest_allowed=5))

    def test_recent_keyframes_are_off_limits(self):
        database, base, xyz = self.build()
        ids = np.arange(8000, 8100)
        for i, old in zip(ids, range(50, 150)):
            xyz[int(i)] = xyz[old]
        self.assertIsNone(find_loop(database, flip_bits(base[50:150], 10), ids, xyz, newest_allowed=-1))

    def test_dropping_a_point_removes_its_descriptors(self):
        database, *_ = self.build()
        before = len(database)
        database.drop_point(10)
        self.assertEqual(len(database), before - 1)


if __name__ == "__main__":
    unittest.main()
