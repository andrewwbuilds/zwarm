import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from slam.camera import Intrinsics
from slam.geometry import invert, make_pose
from slam.optimizer import bundle_adjust, optimize_pose_graph

RNG = np.random.default_rng(5)
K = Intrinsics.from_fov(640, 360, 90).k


def world(n=300):
    return RNG.uniform([-3, -1.5, 3], [3, 1.5, 9], size=(n, 3))


def camera_poses(n=6):
    return {i: make_pose(Rotation.from_euler("y", 0.04 * i).as_matrix(), [0.35 * i, 0.02 * i, 0.0]) for i in range(n)}


def observe(poses, points, noise=0.3):
    observations = []
    for i, pose in poses.items():
        inv = invert(pose)
        camera = points @ inv[:3, :3].T + inv[:3, 3]
        pixels = (camera @ K.T)[:, :2] / camera[:, 2:3] + RNG.normal(0, noise, (len(points), 2))
        inside = (camera[:, 2] > 0.1) & (pixels[:, 0] > 0) & (pixels[:, 0] < 640) & (pixels[:, 1] > 0) & (pixels[:, 1] < 360)
        observations += [(i, p, *pixels[p]) for p in np.where(inside)[0]]
    return observations


class BundleAdjustmentTests(unittest.TestCase):
    def test_recovers_perturbed_poses_and_points(self):
        truth = camera_poses()
        points = world()
        observations = observe(truth, points)
        noisy = {i: p.copy() for i, p in truth.items()}
        for i in range(2, 6):  # cameras 0 and 1 are held fixed, so they stay exact
            noisy[i][:3, 3] += RNG.normal(0, 0.05, 3)
            noisy[i][:3, :3] = noisy[i][:3, :3] @ Rotation.from_rotvec(RNG.normal(0, 0.01, 3)).as_matrix()
        noisy_points = {p: points[p] + RNG.normal(0, 0.1, 3) for p in range(len(points))}
        result = bundle_adjust(K, noisy, {0, 1}, noisy_points, observations)
        self.assertLess(result.error_after, 0.6)
        self.assertLess(result.error_after, result.error_before / 3)
        for i in range(2, 6):
            self.assertLess(np.linalg.norm(result.poses[i][:3, 3] - truth[i][:3, 3]), 0.05)

    def test_fixed_cameras_do_not_move(self):
        truth = camera_poses(3)
        points = world(80)
        observations = observe(truth, points)
        result = bundle_adjust(K, truth, {0, 1}, {p: points[p] + 0.05 for p in range(80)}, observations)
        np.testing.assert_array_equal(result.poses[0], truth[0])
        np.testing.assert_array_equal(result.poses[1], truth[1])

    def test_an_outlier_observation_is_reported_but_does_not_wreck_the_solution(self):
        truth = camera_poses(4)
        points = world(120)
        observations = observe(truth, points)
        bad = list(observations[5])
        bad[2] += 120
        observations[5] = tuple(bad)
        result = bundle_adjust(K, truth, {0, 1}, {p: points[p] for p in range(120)}, observations)
        self.assertGreater(result.residuals[5], 50)
        self.assertLess(np.median(result.residuals), 1.0)


class PoseGraphTests(unittest.TestCase):
    def chain(self, n=12):
        """A ring of poses with exact relative measurements."""
        poses, truth = {}, []
        for i in range(n):
            angle = 2 * np.pi * i / n
            rot = Rotation.from_euler("z", angle).as_matrix()
            truth.append(make_pose(rot, [3 * np.cos(angle), 3 * np.sin(angle), 0]))
        return truth

    def edge(self, truth, i, j, scale=1.0, weight=1.0):
        rel = invert(truth[i]) @ truth[j]
        return (i, j, rel[:3, :3], rel[:3, 3], scale, weight)

    def test_a_loop_edge_pulls_a_drifted_chain_back(self):
        truth = self.chain()
        drifted = {}
        drift = np.eye(4)
        for i in range(len(truth)):
            if i:
                rel = invert(truth[i - 1]) @ truth[i]
                rel[:3, 3] *= 1.04
                rel[:3, :3] = rel[:3, :3] @ Rotation.from_euler("z", 0.01).as_matrix()
                drift = drift @ rel
                drifted[i] = truth[0] @ drift
            else:
                drifted[0] = truth[0].copy()
        before = np.linalg.norm(drifted[11][:3, 3] - truth[11][:3, 3])
        edges = [self.edge(truth, i, i + 1) for i in range(11)] + [self.edge(truth, 11, 0, 1.0, 1.0)]
        # The sequential edges are measured from the drifted estimates; the loop edge is the truth.
        edges = [(i, i + 1, (invert(drifted[i]) @ drifted[i + 1])[:3, :3], (invert(drifted[i]) @ drifted[i + 1])[:3, 3], 1.0, 1.0)
                 for i in range(11)] + [self.edge(truth, 11, 0)]
        fixed = optimize_pose_graph(drifted, edges, {0})
        after = np.linalg.norm(fixed[11][1] - truth[11][:3, 3])
        self.assertLess(after, before * 0.5)

    def test_fixed_node_stays_put(self):
        truth = self.chain(5)
        poses = {i: truth[i] for i in range(5)}
        edges = [self.edge(truth, i, i + 1) for i in range(4)]
        out = optimize_pose_graph(poses, edges, {0})
        np.testing.assert_allclose(out[0][1], truth[0][:3, 3], atol=1e-9)
        self.assertAlmostEqual(out[0][2], 1.0)

    def test_scale_constraint_changes_the_scale_of_later_nodes(self):
        truth = self.chain(6)
        poses = {i: truth[i] for i in range(6)}
        edges = [self.edge(truth, i, i + 1) for i in range(5)] + [self.edge(truth, 5, 0, scale=1.0)]
        edges[2] = self.edge(truth, 2, 3, scale=1.2)
        out = optimize_pose_graph(poses, edges, {0})
        self.assertNotAlmostEqual(out[4][2], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
