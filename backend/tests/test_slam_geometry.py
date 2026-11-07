import unittest

import numpy as np

from slam.camera import Intrinsics
from slam.evaluate import evaluate
from slam.geometry import (BODY_FROM_CAMERA, apply_sim3, camera_pose_from_body, exp_so3, invert, log_so3,
                           make_pose, triangulate, umeyama)

RNG = np.random.default_rng(3)


class GeometryTests(unittest.TestCase):
    def test_invert_undoes_a_pose(self):
        pose = make_pose(exp_so3([0.3, -0.2, 0.5]), [1, 2, 3])
        np.testing.assert_allclose(pose @ invert(pose), np.eye(4), atol=1e-12)

    def test_log_and_exp_round_trip(self):
        vector = np.array([0.2, -0.4, 0.1])
        np.testing.assert_allclose(log_so3(exp_so3(vector)), vector, atol=1e-12)

    def test_umeyama_recovers_a_known_similarity(self):
        points = RNG.normal(size=(40, 3))
        rotation, translation, scale = exp_so3([0.5, 0.2, -1.0]), np.array([3.0, -1.0, 2.0]), 2.7
        moved = scale * points @ rotation.T + translation
        found = umeyama(points, moved)
        self.assertAlmostEqual(found[0], scale, places=9)
        np.testing.assert_allclose(apply_sim3(found, points), moved, atol=1e-9)

    def test_umeyama_without_scale_keeps_scale_one(self):
        points = RNG.normal(size=(20, 3))
        self.assertEqual(umeyama(points, points * 3.0, with_scale=False)[0], 1.0)

    def test_a_drone_facing_forward_has_a_camera_looking_forward(self):
        pose = camera_pose_from_body([1, 2, 3], np.eye(3))
        np.testing.assert_allclose(pose[:3, 2], [1, 0, 0])  # camera z (forward) is body x
        np.testing.assert_allclose(pose[:3, 0], [0, -1, 0])  # camera x (right) is body -y
        np.testing.assert_allclose(pose[:3, 1], [0, 0, -1])  # camera y (down) is body -z
        self.assertAlmostEqual(abs(np.linalg.det(BODY_FROM_CAMERA)), 1.0)

    def test_triangulation_recovers_points(self):
        intrinsics = Intrinsics.from_fov(640, 360, 90)
        pose_a = make_pose(np.eye(3), [0, 0, 0])
        pose_b = make_pose(exp_so3([0, 0.1, 0]), [0.5, 0, 0])
        world = RNG.uniform([-2, -1, 3], [2, 1, 8], size=(30, 3))
        pixels = []
        for pose in (pose_a, pose_b):
            inv = invert(pose)
            pixels.append(intrinsics.project(world @ inv[:3, :3].T + inv[:3, 3]))
        found, depth_a, depth_b = triangulate(intrinsics.k, pose_a, pose_b, *pixels)
        np.testing.assert_allclose(found, world, atol=1e-6)
        self.assertTrue((depth_a > 0).all() and (depth_b > 0).all())

    def test_intrinsics_project_and_bearing_agree(self):
        intrinsics = Intrinsics.from_fov(1184, 656, 90)
        self.assertAlmostEqual(intrinsics.fx, 592.0, places=6)
        pixels = intrinsics.project(np.array([[1.0, 0.5, 2.0]]))
        ray = intrinsics.bearing(pixels)[0]
        np.testing.assert_allclose(ray, np.array([1.0, 0.5, 2.0]) / np.linalg.norm([1.0, 0.5, 2.0]), atol=1e-9)


class EvaluateTests(unittest.TestCase):
    def path(self, n=200):
        t = np.linspace(0, 4 * np.pi, n)
        return np.stack([np.cos(t) * 2, np.sin(t) * 2, t * 0.1], axis=1)

    def test_a_perfect_trajectory_in_another_frame_and_scale_has_no_error(self):
        truth = self.path()
        moved = (truth * 0.37) @ exp_so3([0.2, 0.1, 0.9]).T + [4, 5, 6]
        report = evaluate(moved, truth)
        self.assertLess(report.ate_rmse, 1e-9)
        self.assertAlmostEqual(report.scale, 1 / 0.37, places=6)

    def test_added_drift_shows_up_as_error(self):
        truth = self.path()
        drifting = truth + np.linspace(0, 0.5, len(truth))[:, None] * [1, 0, 0]
        report = evaluate(drifting, truth)
        self.assertGreater(report.ate_rmse, 0.05)
        self.assertGreater(report.end_error, report.ate_rmse * 0.5)
        self.assertGreater(report.drift_percent, 0.0)

    def test_scale_error_is_visible_in_one_second_motion(self):
        truth = self.path()
        report = evaluate(np.concatenate([truth[:100], truth[100:] * 1.0 + 0.0]), truth)
        self.assertLess(report.rpe_percent, 1e-6)


if __name__ == "__main__":
    unittest.main()
