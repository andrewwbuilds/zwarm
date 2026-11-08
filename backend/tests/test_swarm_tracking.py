"""The floor-tag tracker against rendered frames, where the true poses are known exactly.

These run the same detector real flights use, on simulated frames. They bound the maths. They say
nothing about what the real camera will do.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from swarm.tracking import (
    CameraModel,
    PoseStore,
    Sighting,
    camera_position,
    flat_marker_corners,
    in_view,
    lens_from_checkerboards,
    locate_camera,
    marker_image,
    read_markers,
)

WIDTH, HEIGHT = 1920, 1080
ANCHORS = {40: (-1.5, -1.0, 0.0), 41: (1.5, -1.0, 0.0), 42: (1.5, 1.0, 0.0), 43: (-1.5, 1.0, 0.0)}
ANCHOR_SIZE = 0.15
MARKER_SIZE = 0.07


def overhead_camera(position=(0.1, -0.1, 2.6), fov=90.0, look_tilt=0.0):
    """A camera at a position looking straight down, top of the image toward world +x."""
    model = CameraModel.from_fov(WIDTH, HEIGHT, fov)
    # Camera axes in world terms: x right = world -y, y down = world -x, z forward = world -z.
    rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
    if look_tilt:
        tilt, _ = cv2.Rodrigues(np.array([look_tilt, 0.0, 0.0]))
        rotation = tilt @ rotation
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ np.array(position, float)
    return CameraModel(model.matrix, model.distortion, rvec.ravel(), tvec)


def paint_marker(image, camera, marker_id, corners_world):
    """Draw a marker with its white border where this camera would see it."""
    source = marker_image(marker_id, 400)
    side = 400 + 2 * 60
    tile = cv2.copyMakeBorder(source, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    centre = corners_world.mean(0)
    outer = centre + (corners_world - centre) * (side / 400)
    pixels, _ = cv2.projectPoints(outer, camera.rvec, camera.tvec, camera.matrix, camera.distortion)
    tile_corners = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    warp = cv2.getPerspectiveTransform(tile_corners, pixels.reshape(4, 2).astype(np.float32))
    painted = cv2.warpPerspective(tile, warp, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(tile, 255), warp, (image.shape[1], image.shape[0]))
    image[mask > 0] = painted[mask > 0]


def scene_image(camera, drones=(), anchors=True, noise=2.0, seed=1):
    image = np.full((HEIGHT, WIDTH), 120, np.uint8)
    if anchors:
        for marker, centre in ANCHORS.items():
            paint_marker(image, camera, marker, flat_marker_corners(centre, ANCHOR_SIZE))
    for marker, centre, yaw in drones:
        paint_marker(image, camera, marker, flat_marker_corners(centre, MARKER_SIZE, yaw))
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class CameraLocationTests(unittest.TestCase):
    def test_anchors_on_the_floor_give_the_cameras_position(self):
        truth = overhead_camera(position=(0.2, 0.1, 2.6))
        lens = CameraModel(truth.matrix, truth.distortion)
        found = locate_camera(scene_image(truth), lens, ANCHORS, ANCHOR_SIZE)
        np.testing.assert_allclose(camera_position(found), [0.2, 0.1, 2.6], atol=0.05)
        self.assertLess(found.reprojection_px, 1.0)

    def test_a_located_camera_puts_drones_where_they_really_are(self):
        # The number that matters: anchors on the floor, drones well above them.
        truth = overhead_camera(position=(0.2, 0.1, 2.6))
        found = locate_camera(scene_image(truth), CameraModel(truth.matrix, truth.distortion), ANCHORS, ANCHOR_SIZE)
        for x, y, z in ((0.0, 0.0, 1.0), (0.5, 0.5, 1.2), (-0.4, -0.5, 0.8)):
            image = scene_image(truth, [(1, (x, y, z), 0.3)], anchors=False)
            sightings = read_markers(image, found, MARKER_SIZE)
            self.assertEqual(len(sightings), 1)
            np.testing.assert_allclose(sightings[0].position, (x, y, z), atol=0.07)

    def test_one_anchor_is_not_enough(self):
        truth = overhead_camera()
        image = np.full((HEIGHT, WIDTH), 120, np.uint8)
        paint_marker(image, truth, 40, flat_marker_corners(ANCHORS[40], ANCHOR_SIZE))
        with self.assertRaises(ValueError):
            locate_camera(image, CameraModel(truth.matrix, truth.distortion), ANCHORS, ANCHOR_SIZE)


class MarkerReadingTests(unittest.TestCase):
    """A fixed set of poses across the flying volume. All of them must read back accurately."""

    POSES = [
        (x, y, z, yaw)
        for z in (0.0, 0.6, 1.0, 1.4)
        for x, y in ((0.0, 0.0), (-1.0, 0.5), (1.0, -0.5), (0.6, 0.6), (-0.6, -0.6))
        for yaw in (0.0, 1.0, -2.2)
    ]

    def test_a_camera_knows_what_it_can_see(self):
        camera = overhead_camera(position=(0, 0, 2.6))
        self.assertTrue(in_view(camera, (0.0, 0.0, 1.4), WIDTH, HEIGHT))
        self.assertFalse(in_view(camera, (1.0, 0.0, 1.4), WIDTH, HEIGHT))  # top of the image
        self.assertTrue(in_view(camera, (0.0, 1.0, 1.4), WIDTH, HEIGHT))  # the wide way
        self.assertFalse(in_view(camera, (0.0, 0.0, 3.0), WIDTH, HEIGHT))  # above the camera

    def test_position_and_heading_across_the_flying_volume(self):
        camera = overhead_camera()
        errors_xy, errors_z, errors_yaw = [], [], []
        poses = [pose for pose in self.POSES if in_view(camera, pose[:3], WIDTH, HEIGHT)]
        self.assertGreater(len(poses), 30)
        for seed, (x, y, z, yaw) in enumerate(poses):
            image = scene_image(camera, [(1, (x, y, z), yaw)], anchors=False, seed=seed)
            sightings = read_markers(image, camera, MARKER_SIZE)
            self.assertEqual([s.marker for s in sightings], [1], f"pose {(x, y, z, yaw)}")
            s = sightings[0]
            errors_xy.append(math.hypot(s.position[0] - x, s.position[1] - y))
            errors_z.append(abs(s.position[2] - z))
            errors_yaw.append(abs((s.yaw - yaw + math.pi) % (2 * math.pi) - math.pi))
        # Rendered images with light noise: this bounds the maths, not what a real camera will do.
        self.assertLess(max(errors_xy), 0.04, "horizontal error")
        self.assertLess(max(errors_z), 0.08, "height error")  # worst on the floor, furthest from the camera
        self.assertLess(max(errors_yaw), math.radians(3), "heading error")

    def test_several_drones_keep_their_identities(self):
        camera = overhead_camera()
        drones = [(1, (-0.5, 0.0, 1.0), 0.0), (2, (0.5, 0.3, 1.1), 0.5), (3, (0.0, -0.6, 0.9), -1.0)]
        sightings = {s.marker: s for s in read_markers(scene_image(camera, drones), camera, MARKER_SIZE)}
        self.assertEqual(set(sightings), {1, 2, 3})  # anchors are left out
        for marker, centre, _ in drones:
            np.testing.assert_allclose(sightings[marker].position, centre, atol=0.06)

    def test_an_empty_scene_reads_nothing(self):
        camera = overhead_camera()
        self.assertEqual(read_markers(scene_image(camera, anchors=False), camera, MARKER_SIZE), [])

    def test_a_tilted_camera_still_reads_true_positions(self):
        camera = overhead_camera(position=(0.0, -0.8, 2.5), look_tilt=-0.3)
        image = scene_image(camera, [(2, (0.3, 0.0, 1.0), 0.7)], anchors=False)
        sightings = read_markers(image, camera, MARKER_SIZE)
        self.assertEqual(len(sightings), 1)
        np.testing.assert_allclose(sightings[0].position, (0.3, 0.0, 1.0), atol=0.06)


class PoseStoreTests(unittest.TestCase):
    def test_height_is_measured_from_where_the_marker_sits_when_landed(self):
        store = PoseStore({1: "a"}, mount_height=0.05)
        store.add(1.0, [Sighting(1, (0.0, 0.0, 1.05), 0.0, 50.0)])
        self.assertAlmostEqual(store.poses()["a"].p[2], 1.0)

    def test_each_pose_is_delivered_once(self):
        store = PoseStore({1: "a"})
        store.add(1.0, [Sighting(1, (0.0, 0.0, 1.0), 0.0, 50.0)])
        self.assertIn("a", store.poses())
        self.assertEqual(store.poses(), {})

    def test_a_far_cameras_view_does_not_replace_a_close_ones(self):
        store = PoseStore({1: "a"})
        store.add(1.00, [Sighting(1, (1.0, 0.0, 1.0), 0.0, 60.0)])
        store.add(1.02, [Sighting(1, (1.2, 0.0, 1.0), 0.0, 20.0)])
        self.assertEqual(store.poses()["a"].p[0], 1.0)

    def test_a_far_camera_takes_over_when_the_close_one_loses_sight(self):
        store = PoseStore({1: "a"})
        store.add(1.0, [Sighting(1, (1.0, 0.0, 1.0), 0.0, 60.0)])
        store.poses()
        store.add(1.5, [Sighting(1, (1.2, 0.0, 1.0), 0.0, 20.0)])
        self.assertEqual(store.poses()["a"].p[0], 1.2)

    def test_marked_obstacles_appear_as_moving_things_until_they_go_stale(self):
        store = PoseStore({1: "a"})
        store.add(1.0, [Sighting(30, (1.0, 1.0, 1.5), 0.0, 40.0)])
        self.assertEqual([m.id for m in store.moving_things(1.2)], ["marker-30"])
        self.assertEqual(store.moving_things(3.0), [])


class LensTests(unittest.TestCase):
    def test_a_checkerboard_recovers_the_focal_length(self):
        columns, rows, square = 9, 6, 0.03
        truth = overhead_camera(fov=70.0)
        images = []
        rng = np.random.default_rng(2)
        board = np.full(((rows + 3) * 80, (columns + 3) * 80), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(columns + 1):
                if (r + c) % 2 == 0:
                    board[80 + r * 80:160 + r * 80, 80 + c * 80:160 + c * 80] = 0
        height, width = board.shape
        scale = square / 80
        for _ in range(12):
            rvec = rng.uniform(-0.45, 0.45, 3)
            tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.55)])
            flat = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], float) * scale
            flat -= np.array([width, height, 0]) * scale / 2
            pixels, _ = cv2.projectPoints(flat, rvec, tvec, truth.matrix, truth.distortion)
            warp = cv2.getPerspectiveTransform(
                np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32), pixels.reshape(4, 2).astype(np.float32))
            images.append(cv2.warpPerspective(board, warp, (WIDTH, HEIGHT), borderValue=128))
        model = lens_from_checkerboards(images, (columns, rows), square)
        self.assertAlmostEqual(model.matrix[0, 0] / truth.matrix[0, 0], 1.0, delta=0.02)
        self.assertLess(model.reprojection_px, 1.0)

    def test_too_few_views_are_refused(self):
        with self.assertRaises(ValueError):
            lens_from_checkerboards([np.zeros((100, 100), np.uint8)], (9, 6), 0.03)


if __name__ == "__main__":
    unittest.main()
