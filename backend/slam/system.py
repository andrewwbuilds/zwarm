"""Monocular visual SLAM for the S20's camera.

Per frame: track corners with optical flow, find the camera pose from the 3D points they belong
to, and every so often make a keyframe: triangulate new points, refine the recent keyframes and
points together (local bundle adjustment), and look for a place seen before. A recognised place
closes a loop: the keyframe poses are re-solved as a graph, which also removes accumulated scale
drift. Positions are in map units until a known distance sets the scale (see set_scale).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from slam.camera import Intrinsics
from slam.geometry import invert, make_pose, triangulate
from scipy.spatial import cKDTree

from slam.loop import DescriptorDatabase
from slam.optimizer import bundle_adjust, optimize_pose_graph, relative_sim3

log = logging.getLogger(__name__)

INIT, OK, LOST = "init", "ok", "lost"


@dataclass
class Config:
    scale: float = 0.5  # processing resolution relative to the camera's
    max_features: int = 450
    min_distance: int = 12
    lk_window: int = 21
    lk_levels: int = 4
    init_px: float = 18.0  # median pixel motion needed before the first two-view solve
    init_min_points: int = 60
    triangulation_deg: float = 1.0
    reprojection_px: float = 2.5
    keyframe_ratio: float = 0.65  # new keyframe when this share of the last one's points remain
    keyframe_min_frames: int = 3
    keyframe_max_frames: int = 20
    ba_window: int = 8
    ba_nfev: int = 12
    loop_gap: int = 12  # keyframes that must separate a loop's ends
    loop_confirmations: int = 2
    loop_min_inliers: int = 20
    prior_adapt: float = 0.05  # how fast the guessed scene depth follows the map; 0 keeps the first view's
    calibrate_seconds: float = 1.5  # stick-seconds of well-tracked flight that fix map units per stick-second
    stick_window: int = 6  # keyframes compared against the sticks' distance
    stick_tolerance: float = 1.25
    seed: int = 0


@dataclass
class Estimate:
    index: int
    state: str
    pose: np.ndarray | None  # camera-to-world, map units
    tracked: int = 0
    keyframe: bool = False
    loop: bool = False
    seconds: float = 0.0


@dataclass
class MapPoint:
    id: int
    xyz: np.ndarray
    obs: dict = field(default_factory=dict)  # keyframe id -> (pixel, feature index)
    provisional: bool = False  # depth is a guess from the scene's typical depth, not yet triangulated


@dataclass
class KeyFrame:
    id: int
    index: int  # frame number
    t: float
    pose: np.ndarray
    px: np.ndarray  # (N, 2) pixels
    tids: np.ndarray
    pids: np.ndarray  # map point id per feature, or -1
    tid_index: dict = field(default_factory=dict)
    stick_path: float = 0.0


class Slam:
    def __init__(self, intrinsics: Intrinsics, config: Config | None = None):
        self.cfg = config or Config()
        self.camera = intrinsics.scaled(self.cfg.scale)
        self.k = self.camera.k
        self.rng = np.random.default_rng(self.cfg.seed)
        self.orb = cv2.ORB_create(nfeatures=800, edgeThreshold=15, patchSize=31)
        self.keyframes: dict[int, KeyFrame] = {}
        self.points: dict[int, MapPoint] = {}
        self.database = DescriptorDatabase()
        self.loop_edges: list = []
        self.loops: list = []  # (keyframe, older keyframe) pairs that were closed
        self.state = INIT
        self.frames: list = []  # (index, t, ref keyframe, R_rel, t_rel) ; None ref if untracked
        self.metric_scale = 1.0
        self._next_point = self._next_track = self._next_kf = 0
        self._reset_tracking()
        self._pending: list = []
        self._prev_gray = None
        self._frame_index = -1
        self._init_gray = self._init_pts = None
        self._last_loop_kf = -10**9
        self.pose = np.eye(4)
        self._velocity = np.eye(4)
        self._frames_since_kf = 0
        self._stick_path = 0.0  # stick deflection integrated over time: a ruler, once calibrated
        self._units_per_stick_second: float | None = None
        self._speed_frozen = False
        self._calib_moved = self._calib_stick = 0.0
        self._last_t = None

    # ----------------------------------------------------------------- tracks
    def _reset_tracking(self) -> None:
        self.t_px = np.zeros((0, 2), np.float32)
        self.t_pid = np.zeros(0, np.int64)
        self.t_tid = np.zeros(0, np.int64)
        self.t_origin = np.zeros(0, np.int64)
        self.t_origin_px = np.zeros((0, 2), np.float32)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.cfg.scale != 1.0:
            gray = cv2.resize(gray, (self.camera.width, self.camera.height), interpolation=cv2.INTER_AREA)
        return gray

    def _detect(self, gray: np.ndarray, count: int, avoid: np.ndarray) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 2), np.float32)
        mask = np.full(gray.shape, 255, np.uint8)
        for x, y in avoid:
            cv2.circle(mask, (int(x), int(y)), self.cfg.min_distance, 0, -1)
        found = cv2.goodFeaturesToTrack(gray, count, 0.008, self.cfg.min_distance, mask=mask, blockSize=5)
        if found is None:
            return np.zeros((0, 2), np.float32)
        return found.reshape(-1, 2).astype(np.float32)

    def _lk(self, a: np.ndarray, b: np.ndarray, pts: np.ndarray, guess: np.ndarray | None = None):
        """Track pts from image a to image b. Returns (new pts, ok mask) after a forward-backward check."""
        if len(pts) == 0:
            return pts, np.zeros(0, bool)
        window = (self.cfg.lk_window, self.cfg.lk_window)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0
        init = (guess if guess is not None else pts).reshape(-1, 1, 2).astype(np.float32)
        new, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts.reshape(-1, 1, 2), init, winSize=window,
                                                  maxLevel=self.cfg.lk_levels, criteria=criteria, flags=flags)
        back, status_b, _ = cv2.calcOpticalFlowPyrLK(b, a, new, None, winSize=window, maxLevel=self.cfg.lk_levels,
                                                     criteria=criteria)
        new, back = new.reshape(-1, 2), back.reshape(-1, 2)
        h, w = b.shape
        ok = (status.ravel() == 1) & (status_b.ravel() == 1) & (np.linalg.norm(back - pts, axis=1) < 1.0)
        ok &= (new[:, 0] > 4) & (new[:, 0] < w - 4) & (new[:, 1] > 4) & (new[:, 1] < h - 4)
        return new, ok

    # --------------------------------------------------------------- main entry
    def process(self, image: np.ndarray, t: float, command: tuple | None = None) -> Estimate:
        """Track one frame. command is the sticks flown, (roll, pitch, throttle, yaw) in -1..1, if known."""
        started = time.perf_counter()
        self._note_command(t, command)
        self._frame_index += 1
        gray = self._prepare(image)
        index = self._frame_index
        if self.state == INIT:
            estimate = self._initialize(gray, t, index)
        elif self.state == OK:
            estimate = self._track(gray, t, index)
        else:
            estimate = self._relocalize(gray, t, index)
        self._prev_gray = gray
        estimate.seconds = time.perf_counter() - started
        return estimate

    def _note_command(self, t: float, command: tuple | None) -> None:
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.2)
        self._last_t = t
        self._stick = 0.0
        if command is not None and self.state != INIT:
            self._stick = float(np.hypot(command[0], command[1]))  # sideways and forward sticks; height is separate
            self._stick_path += self._stick * dt
        self._dt = dt

    def _learn_speed(self, previous: np.ndarray, current: np.ndarray) -> None:
        """Calibrate map units per stick-second from the first stretch of flight tracked on real points.

        It then stays fixed. A value that kept following the map would shrink along with it, which is the
        very error the sticks are there to catch.
        """
        if self._speed_frozen or self._stick < 0.3 or self._dt <= 0:
            return
        self._calib_moved += float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
        self._calib_stick += self._stick * self._dt
        if self._calib_stick >= self.cfg.calibrate_seconds:
            self._units_per_stick_second = self._calib_moved / self._calib_stick
            self._speed_frozen = True
            log.info("Speed calibrated: %.3f map units per stick-second", self._units_per_stick_second)

    def _check_scale_against_sticks(self, kf: KeyFrame) -> None:
        """If the sticks say the recent keyframes flew much farther or shorter than the map says, rescale them."""
        cfg = self.cfg
        if not self._speed_frozen or len(self.keyframes) < cfg.stick_window + 2:
            return
        ids = sorted(self.keyframes)[-cfg.stick_window:]
        first = self.keyframes[ids[0]]
        expected = (kf.stick_path - first.stick_path) * self._units_per_stick_second
        actual = sum(float(np.linalg.norm(self.keyframes[b].pose[:3, 3] - self.keyframes[a].pose[:3, 3])) for a, b in zip(ids, ids[1:]))
        if expected < 0.15 * getattr(self, "_depth_prior", 1.0) or actual < 1e-6:
            return
        ratio = expected / actual
        if 1 / cfg.stick_tolerance < ratio < cfg.stick_tolerance:
            return
        ratio = float(np.clip(ratio, 0.25, 4.0))
        centre = first.pose[:3, 3].copy()
        later = set(ids[1:])
        for kid in ids[1:]:
            self.keyframes[kid].pose[:3, 3] = centre + ratio * (self.keyframes[kid].pose[:3, 3] - centre)
        for point in self.points.values():
            if point.obs and min(point.obs) in later:
                point.xyz = centre + ratio * (point.xyz - centre)
        for i, pid in enumerate(self.t_pid):
            if pid >= 0 and self.points[int(pid)].provisional:
                self.points[int(pid)].xyz = centre + ratio * (self.points[int(pid)].xyz - centre)
        self.pose = self.keyframes[kf.id].pose.copy()
        self._velocity = np.eye(4)
        log.debug("Rescaled the last %d keyframes by %.2f to agree with the sticks", len(ids), ratio)

    # ---------------------------------------------------------- initialization
    def _initialize(self, gray: np.ndarray, t: float, index: int) -> Estimate:
        cfg = self.cfg
        if self._init_gray is None or len(self._init_pts) < cfg.init_min_points:
            self._init_gray, self._init_pts = gray, self._detect(gray, cfg.max_features, np.zeros((0, 2)))
            self._init_origin = self._init_pts.copy()
            self._init_index, self._init_t = index, t
            return Estimate(index, INIT, None)
        new, ok = self._lk(self._prev_gray, gray, self._init_pts)
        self._init_pts, self._init_origin = new[ok], self._init_origin[ok]
        self._prev_gray = gray
        if len(self._init_pts) < cfg.init_min_points:
            self._init_gray = None
            return Estimate(index, INIT, None)
        motion = np.median(np.linalg.norm(self._init_pts - self._init_origin, axis=1))
        if motion < cfg.init_px:
            return Estimate(index, INIT, None, tracked=len(self._init_pts))
        if self._two_view_init(gray, t, index):
            return Estimate(index, OK, self.pose.copy(), tracked=int((self.t_pid >= 0).sum()), keyframe=True)
        self._init_gray = None  # try again from a fresh reference
        return Estimate(index, INIT, None)

    def _two_view_init(self, gray: np.ndarray, t: float, index: int) -> bool:
        cfg = self.cfg
        p0, p1 = self._init_origin.astype(np.float64), self._init_pts.astype(np.float64)
        e, mask = cv2.findEssentialMat(p0, p1, self.k, cv2.RANSAC, 0.999, 1.0)
        if e is None or e.shape != (3, 3):
            return False
        count, rotation, translation, mask = cv2.recoverPose(e, p0, p1, self.k, mask=mask)
        if count < cfg.init_min_points:
            return False
        pose0 = np.eye(4)
        pose1 = invert(make_pose(rotation, translation.ravel()))
        keep = mask.ravel() > 0
        world, depth0, depth1 = triangulate(self.k, pose0, pose1, p0[keep], p1[keep])
        pixels = [p0[keep], p1[keep]]
        good = (depth0 > 0) & (depth1 > 0) & self._parallax_ok(pose0, pose1, world)
        for pose, px in zip((pose0, pose1), pixels):
            good &= self._reprojection(pose, world, px) < cfg.reprojection_px
        if good.sum() < cfg.init_min_points:
            return False
        scale = 1.0 / float(np.median(depth0[good]))  # the first view's median depth is one map unit
        pose1[:3, 3] *= scale
        world = world[good] * scale
        kf0 = self._new_keyframe(index - 1 if False else self._init_index, self._init_t, pose0)
        # Both views see exactly the same points: register them as map points and tracks.
        n = len(world)
        px0, px1 = pixels[0][good], pixels[1][good]
        pids = np.arange(self._next_point, self._next_point + n)
        self._next_point += n
        tids = np.arange(self._next_track, self._next_track + n)
        self._next_track += n
        for pid, xyz in zip(pids, world):
            self.points[int(pid)] = MapPoint(int(pid), xyz)
        self._register(kf0, px0, tids, pids)
        self.t_px, self.t_pid, self.t_tid = px1.astype(np.float32), pids.copy(), tids.copy()
        self.t_origin = np.full(n, kf0.id, np.int64)
        self.t_origin_px = px0.astype(np.float32)
        self.pose = pose1
        self._add_keyframe(gray, t, index, pose1, initial=True)
        self.frames.append((self._init_index, self._init_t, kf0.id, np.eye(3), np.zeros(3)))
        self.state = OK
        self._velocity = np.eye(4)
        return True

    def _parallax_ok(self, pose_a: np.ndarray, pose_b: np.ndarray, world: np.ndarray) -> np.ndarray:
        ray_a = world - pose_a[:3, 3]
        ray_b = world - pose_b[:3, 3]
        cosine = (ray_a * ray_b).sum(1) / (np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cosine, -1, 1))) > self.cfg.triangulation_deg

    @staticmethod
    def _median_parallax_deg(pose_a: np.ndarray, pose_b: np.ndarray, world: np.ndarray) -> float:
        ray_a, ray_b = world - pose_a[:3, 3], world - pose_b[:3, 3]
        cosine = (ray_a * ray_b).sum(1) / (np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-12)
        return float(np.median(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))

    def _reprojection(self, pose: np.ndarray, world: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        inverse = invert(pose)
        camera = world @ inverse[:3, :3].T + inverse[:3, 3]
        z = np.where(camera[:, 2] == 0, 1e-9, camera[:, 2])
        projected = np.stack([self.k[0, 0] * camera[:, 0] / z + self.k[0, 2], self.k[1, 1] * camera[:, 1] / z + self.k[1, 2]], 1)
        return np.linalg.norm(projected - pixels, axis=1)

    # ----------------------------------------------------------------- tracking
    def _track(self, gray: np.ndarray, t: float, index: int) -> Estimate:
        cfg = self.cfg
        predicted = self.pose @ self._velocity
        guess = self._guess(predicted)
        old_px = self.t_px.copy()
        new, ok = self._lk(self._prev_gray, gray, self.t_px, guess)
        self._keep(ok, new)
        old_px = old_px[ok]
        known = self.t_pid >= 0
        found = None
        real = np.zeros(len(self.t_pid), bool)
        if known.sum() >= 12:
            real = known.copy()
            real[known] = [not self.points[int(p)].provisional for p in self.t_pid[known]]
            use = real if real.sum() >= 20 else known  # guessed depths only help when real points run short
            xyz = np.array([self.points[int(p)].xyz for p in self.t_pid[use]])
            found = self._pnp(xyz, self.t_px[use], predicted)
            if found is not None:
                found = (found[0], self._expand(found[1], use, known))
        real_count = int(real.sum()) if known.sum() >= 12 else 0
        booted = None
        if real_count < 20:
            # The map points in view are mostly guesses, which cannot show how far the drone moved. Take the
            # motion from the 2D tracks instead and let that fix real depths for them.
            booted = self._bootstrap()
        if booted is not None:
            new_pose = booted
        elif found is None or found[1].sum() < 12:
            # Too few 3D points to fix a pose, e.g. while turning on the spot. Fall back to the turn
            # the 2D tracks alone show, keeping the position, until new points can be triangulated.
            return self._rotate_only(old_px, index, t, gray)
        else:
            pose_cw, inliers = found
            if real_count >= 60:
                self._learn_speed(self.pose, invert(pose_cw))
            outliers = np.where(known)[0][~inliers]
            drop = np.ones(len(self.t_px), bool)
            drop[outliers] = False
            self._keep(drop, self.t_px)
            new_pose = invert(pose_cw)
        previous = self.pose
        self.pose = new_pose
        self._velocity = invert(previous) @ self.pose
        self._frames_since_kf += 1
        is_kf = self._wants_keyframe()
        if is_kf:
            self._add_keyframe(gray, t, index, self.pose)
        self._record(index, t)
        return Estimate(index, OK, self.pose.copy(), tracked=int((self.t_pid >= 0).sum()), keyframe=is_kf,
                        loop=is_kf and bool(self.loops) and self.loops[-1][0] == self._next_kf - 1)

    def _bootstrap(self) -> np.ndarray | None:
        """A pose from the 2D tracks of guessed-depth points, when real depths are missing.

        The relative pose to the keyframe where those tracks began comes from the epipolar geometry. Its
        length is not observable, so the median depth is set to the scene's typical depth, once. The points
        are triangulated with that and become real points, so later frames can use them directly.
        """
        cfg = self.cfg
        guessed = np.where([pid >= 0 and self.points[int(pid)].provisional for pid in self.t_pid])[0]
        if len(guessed) < 40:
            return None
        origins = self.t_origin[guessed]
        origin = int(np.bincount(origins).argmax())
        idx = guessed[origins == origin]
        if len(idx) < 40 or origin not in self.keyframes:
            return None
        base = self.keyframes[origin]
        a, b = self.t_origin_px[idx].astype(np.float64), self.t_px[idx].astype(np.float64)
        if np.median(np.linalg.norm(b - a, axis=1)) < 6.0:
            return None
        e, mask = cv2.findEssentialMat(a, b, self.k, cv2.RANSAC, 0.999, 1.0)
        if e is None or e.shape != (3, 3):
            return None
        count, rotation, translation, mask = cv2.recoverPose(e, a, b, self.k, mask=mask)
        if count < 40:
            return None
        unit = base.pose @ invert(make_pose(rotation, translation.ravel()))
        keep = mask.ravel() > 0
        world, depth_a, depth_b = triangulate(self.k, base.pose, unit, a[keep], b[keep])
        good = (depth_a > 0) & (depth_b > 0) & self._parallax_ok(base.pose, unit, world)
        if good.sum() < 30:
            return None
        if self._median_parallax_deg(base.pose, unit, world[good]) < 2.5:
            return None  # a turn on the spot also moves the picture; only a real change of viewpoint counts
        prior = getattr(self, "_depth_prior", 1.0)
        lam = prior / float(np.median(depth_a[good]))
        travelled = (self._stick_path - base.stick_path) * (self._units_per_stick_second or 0.0)
        if self._units_per_stick_second is not None:
            if travelled < 0.1 * prior:
                return None  # the sticks did not ask for any sideways or forward motion
            lam = travelled  # the sticks say how far it flew, which beats a guess at how deep the scene is
        centre = base.pose[:3, 3]
        pose = unit.copy()
        pose[:3, 3] = centre + lam * (unit[:3, 3] - centre)
        world = centre + lam * (world - centre)
        chosen = idx[keep][good]
        for i, xyz in zip(chosen, world[good]):
            point = self.points[int(self.t_pid[i])]
            point.xyz, point.provisional = xyz, False
        log.debug("Bootstrapped %d tracks from keyframe %d: step %.3f (prior-based %.3f, sticks %.3f, k=%s)", len(chosen), origin, lam, prior / float(np.median(depth_a[good])), travelled, self._units_per_stick_second)
        return pose

    @staticmethod
    def _expand(inliers: np.ndarray, used: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Inlier flags for the known tracks, given flags for the subset that was used. Unused ones count as inliers."""
        full = np.ones(int(known.sum()), bool)
        full[used[known]] = inliers
        return full

    def _rotate_only(self, old_px: np.ndarray, index: int, t: float, gray: np.ndarray) -> Estimate:
        if len(self.t_px) < 15:
            return self._lose(index, f"only {len(self.t_px)} tracks left")
        h, mask = cv2.findHomography(old_px.astype(np.float64), self.t_px.astype(np.float64), cv2.RANSAC, 2.0)
        if h is None or mask.sum() < 12:
            return self._lose(index, "no consistent turn in the 2D tracks")
        u, _, vt = np.linalg.svd(np.linalg.inv(self.k) @ h @ self.k)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        keep = mask.ravel() > 0
        self._keep(keep, self.t_px)
        previous = self.pose
        self.pose = previous.copy()
        self.pose[:3, :3] = previous[:3, :3] @ rotation.T
        self._velocity = np.eye(4)
        self._velocity[:3, :3] = rotation.T
        self._frames_since_kf += 1
        is_kf = self._wants_keyframe()
        if is_kf:
            self._add_keyframe(gray, t, index, self.pose)
        self._record(index, t)
        return Estimate(index, OK, self.pose.copy(), tracked=int((self.t_pid >= 0).sum()), keyframe=is_kf)

    def _guess(self, predicted: np.ndarray) -> np.ndarray:
        guess = self.t_px.copy()
        known = self.t_pid >= 0
        if known.any():
            inverse = invert(predicted)
            xyz = np.array([self.points[int(p)].xyz for p in self.t_pid[known]])
            camera = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            front = camera[:, 2] > 0.05
            projected = self.camera.project(camera)
            sub = guess[known]
            sub[front] = projected[front]
            guess[known] = sub
            shift = np.median(guess[known] - self.t_px[known], axis=0)
            guess[~known] = self.t_px[~known] + shift
        return guess

    def _keep(self, ok: np.ndarray, new_px: np.ndarray) -> None:
        self.t_px = new_px[ok].astype(np.float32)
        self.t_pid, self.t_tid, self.t_origin = self.t_pid[ok], self.t_tid[ok], self.t_origin[ok]
        self.t_origin_px = self.t_origin_px[ok]

    def _pnp(self, xyz: np.ndarray, pixels: np.ndarray, predicted: np.ndarray):
        cw = invert(predicted)
        rvec, _ = cv2.Rodrigues(cw[:3, :3])
        tvec = cw[:3, 3].reshape(3, 1)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz.astype(np.float64), pixels.astype(np.float64), self.k, None, rvec, tvec,
                                                     True, 80, 3.0, 0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 8:
            return None
        mask = np.zeros(len(xyz), bool)
        mask[inliers.ravel()] = True
        rvec, tvec = cv2.solvePnPRefineLM(xyz[mask].astype(np.float64), pixels[mask].astype(np.float64), self.k, None, rvec, tvec)
        return make_pose(cv2.Rodrigues(rvec)[0], tvec.ravel()), mask

    def _lose(self, index: int, reason: str = "") -> Estimate:
        log.info("Tracking lost at frame %d: %s", index, reason)
        self.state = LOST
        self.frames.append((index, None, None, None, None))
        return Estimate(index, LOST, None)

    def _wants_keyframe(self) -> bool:
        cfg = self.cfg
        known = int((self.t_pid >= 0).sum())
        last = self.keyframes[self._next_kf - 1]
        if self._frames_since_kf < cfg.keyframe_min_frames:
            return False
        if known < 40 or self._frames_since_kf >= cfg.keyframe_max_frames:
            return True
        shared = np.isin(self.t_pid[self.t_pid >= 0], last.pids[last.pids >= 0]).sum()
        return shared < cfg.keyframe_ratio * (last.pids >= 0).sum()

    # --------------------------------------------------------------- keyframes
    def _new_keyframe(self, index: int, t: float, pose: np.ndarray) -> KeyFrame:
        kf = KeyFrame(self._next_kf, index, t, pose.copy(), np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64))
        kf.stick_path = self._stick_path
        self.keyframes[kf.id] = kf
        self._next_kf += 1
        return kf

    def _register(self, kf: KeyFrame, px: np.ndarray, tids: np.ndarray, pids: np.ndarray) -> None:
        kf.px, kf.tids, kf.pids = px.astype(np.float32), tids.astype(np.int64), pids.astype(np.int64)
        kf.tid_index = {int(tid): i for i, tid in enumerate(kf.tids)}
        for i, pid in enumerate(kf.pids):
            if pid >= 0 and int(pid) in self.points:
                self.points[int(pid)].obs[kf.id] = (kf.px[i].copy(), i)

    def _add_keyframe(self, gray: np.ndarray, t: float, index: int, pose: np.ndarray, initial: bool = False) -> None:
        cfg = self.cfg
        if not initial:
            self._triangulate_new(pose)
        n_before = len(self.t_px)
        corners = self._detect(gray, cfg.max_features - n_before, self.t_px)
        kf = self._new_keyframe(index, t, pose)
        if len(corners):
            n = len(corners)
            self.t_px = np.concatenate([self.t_px, corners])
            self.t_pid = np.concatenate([self.t_pid, np.full(n, -1, np.int64)])
            tids = np.arange(self._next_track, self._next_track + n)
            self._next_track += n
            self.t_tid = np.concatenate([self.t_tid, tids])
            self.t_origin = np.concatenate([self.t_origin, np.full(n, kf.id, np.int64)])
            self.t_origin_px = np.concatenate([self.t_origin_px, corners])
        self._seed_provisional(pose)
        self._register(kf, self.t_px, self.t_tid, self.t_pid)
        # Point descriptors for recognising this place again.
        self._describe(kf, gray)
        self._frames_since_kf = 0
        self._local_bundle_adjust(kf)
        self._check_scale_against_sticks(kf)
        if not initial:
            self._look_for_loop(kf, gray)

    def _seed_provisional(self, pose: np.ndarray) -> None:
        """Give tracks with no depth yet a guessed one, so a pose can still be found when the real points run out."""
        real = np.where(self.t_pid >= 0)[0]
        real = [i for i in real if not self.points[int(self.t_pid[i])].provisional]
        if len(real) >= 8:
            inverse = invert(pose)
            depths = [(inverse[:3, :3] @ self.points[int(self.t_pid[i])].xyz + inverse[:3, 3])[2] for i in real]
            depths = [d for d in depths if d > 0]
            if depths:
                # Slow to change: a prior that follows the map would shrink along with it.
                self._depth_prior = (1.0 - self.cfg.prior_adapt) * getattr(self, "_depth_prior", 1.0) + self.cfg.prior_adapt * float(np.median(depths))
        prior = getattr(self, "_depth_prior", 1.0)
        centre = pose[:3, 3]
        for i in np.where(self.t_pid < 0)[0]:
            ray = np.array([(self.t_px[i, 0] - self.k[0, 2]) / self.k[0, 0], (self.t_px[i, 1] - self.k[1, 2]) / self.k[1, 1], 1.0])
            xyz = pose[:3, :3] @ (ray * prior) + centre
            pid = self._next_point
            self._next_point += 1
            self.points[pid] = MapPoint(pid, xyz, provisional=True)
            self.t_pid[i] = pid

    def _triangulate_new(self, pose: np.ndarray) -> None:
        cfg = self.cfg
        guessed = np.array([pid >= 0 and self.points[int(pid)].provisional for pid in self.t_pid], bool)
        fresh = np.where((self.t_pid < 0) | guessed)[0]
        if not len(fresh):
            return
        by_origin: dict[int, list] = {}
        for i in fresh:
            by_origin.setdefault(int(self.t_origin[i]), []).append(i)
        for origin, idx in by_origin.items():
            idx = np.array(idx)
            okf = self.keyframes[origin]
            if np.linalg.norm(okf.pose[:3, 3] - pose[:3, 3]) < 1e-6:
                continue
            world, d_a, d_b = triangulate(self.k, okf.pose, pose, self.t_origin_px[idx].astype(float), self.t_px[idx].astype(float))
            good = (d_a > 0) & (d_b > 0) & self._parallax_ok(okf.pose, pose, world)
            good &= self._reprojection(okf.pose, world, self.t_origin_px[idx]) < cfg.reprojection_px
            good &= self._reprojection(pose, world, self.t_px[idx]) < cfg.reprojection_px
            if good.any():
                depth = np.median(d_a[good])
                good &= d_a < 12 * depth
            for j in np.where(good)[0]:
                i = idx[j]
                if self.t_pid[i] >= 0:
                    point = self.points[int(self.t_pid[i])]
                    point.xyz, point.provisional = world[j], False
                    pid = point.id
                else:
                    pid = self._next_point
                    self._next_point += 1
                    point = MapPoint(pid, world[j])
                    self.points[pid] = point
                    self.t_pid[i] = pid
                feature = okf.tid_index.get(int(self.t_tid[i]))
                if feature is not None:
                    okf.pids[feature] = pid
                    point.obs[okf.id] = (okf.px[feature].copy(), feature)

    def _describe(self, kf: KeyFrame, gray: np.ndarray) -> None:
        """Remember what this keyframe's mapped points look like, for recognising the place later.

        ORB keypoints carry orientation and scale, so they match across a change of view. Each is tied to
        the nearest tracked point that has a real 3D position.
        """
        known = np.array([i for i in np.where(kf.pids >= 0)[0]
                          if int(kf.pids[i]) in self.points and not self.points[int(kf.pids[i])].provisional], int)
        if len(known) < 8:
            return
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or not len(keypoints):
            return
        orb_px = np.array([kp.pt for kp in keypoints], np.float32)
        distance, nearest = cKDTree(kf.px[known]).query(orb_px, distance_upper_bound=3.0)
        hit = np.isfinite(distance)
        # One descriptor per point: the closest ORB keypoint.
        best: dict[int, tuple] = {}
        for row in np.where(hit)[0]:
            feature = int(nearest[row])
            if feature not in best or distance[row] < best[feature][0]:
                best[feature] = (distance[row], row)
        rows = [r for _, r in best.values()]
        pids = np.array([kf.pids[known[f]] for f in best], np.int64)
        self.database.add(kf.id, descriptors[rows], pids)

    def _orb_localize(self, gray: np.ndarray, newest_allowed: int, min_inliers: int = 25):
        """Find this frame in the map by appearance. Returns (camera-from-world pose, matched keyframe, inliers) or None."""
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(self.database) < 50:
            return None
        query, rows = self.database.match(descriptors, newest_allowed)
        if len(query) < min_inliers:
            return None
        pids = self.database.point[rows]
        valid = np.array([int(q) in self.points and not self.points[int(q)].provisional for q in pids], bool)
        if valid.sum() < min_inliers:
            return None
        pixels = np.array([keypoints[i].pt for i in query[valid]], np.float64)
        xyz = np.array([self.points[int(q)].xyz for q in pids[valid]], np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, pixels, self.k, None, iterationsCount=250, reprojectionError=3.5,
                                                     confidence=0.995)
        if not ok or inliers is None or len(inliers) < min_inliers:
            return None
        owners = self.database.owner[rows][valid][inliers.ravel()]
        keyframe = int(np.bincount(owners).argmax())
        mask = np.zeros(len(xyz), bool)
        mask[inliers.ravel()] = True
        return make_pose(cv2.Rodrigues(rvec)[0], tvec.ravel()), keyframe, int(len(inliers)), pixels[mask].astype(np.float32), pids[valid][mask]

    # ---------------------------------------------------------------- mapping
    def _local_bundle_adjust(self, current: KeyFrame) -> None:
        window = sorted(self.keyframes)[-self.cfg.ba_window:]
        if len(window) < 3:
            return
        in_window = set(window)
        pids = set()
        for kid in window:
            pids.update(int(p) for p in self.keyframes[kid].pids if p >= 0)
        pids = {p for p in pids if p in self.points and not self.points[p].provisional and len(self.points[p].obs) >= 2}
        observations = []
        for pid in pids:
            for kid, (px, _) in self.points[pid].obs.items():
                if kid in in_window:
                    observations.append((kid, pid, float(px[0]), float(px[1])))
        if len(observations) < 40:
            return
        poses = {kid: self.keyframes[kid].pose for kid in window}
        fixed = set(window[:2])  # two fixed keyframes pin down position and scale
        result = bundle_adjust(self.k, poses, fixed, {p: self.points[p].xyz for p in pids}, observations, max_nfev=self.cfg.ba_nfev)
        for kid in window:
            self.keyframes[kid].pose = result.poses[kid]
        for pid, xyz in result.points.items():
            self.points[pid].xyz = xyz
        self._prune(observations, result.residuals)
        self.pose = self.keyframes[current.id].pose.copy()
        self._velocity = np.eye(4)

    def _prune(self, observations: list, residuals: np.ndarray) -> None:
        """Forget observations that disagree badly with the refined map, and points that lose their support."""
        for (kid, pid, _, _), error in zip(observations, residuals):
            if error > 4.0 and pid in self.points:
                self.points[pid].obs.pop(kid, None)
        for pid in {o[1] for o in observations}:
            point = self.points.get(pid)
            if point is not None and len(point.obs) < 2:
                self._remove_point(pid)

    def _remove_point(self, pid: int) -> None:
        point = self.points.pop(pid, None)
        if point is None:
            return
        for kid, (_, feature) in point.obs.items():
            kf = self.keyframes.get(kid)
            if kf is not None and feature < len(kf.pids) and kf.pids[feature] == pid:
                kf.pids[feature] = -1
        self.t_pid[self.t_pid == pid] = -1
        self.database.drop_point(pid)

    # ---------------------------------------------------------- loop closure
    def _look_for_loop(self, kf: KeyFrame, gray: np.ndarray) -> None:
        cfg = self.cfg
        if kf.id < cfg.loop_gap + 2:
            return
        found = self._orb_localize(gray, kf.id - cfg.loop_gap, min_inliers=cfg.loop_min_inliers)
        if found is None:
            self._pending = []
            return
        if self._pending and abs(self._pending[-1][1][1] - found[1]) <= 6 and self._pending[-1][0] >= kf.id - 3:
            self._pending.append((kf.id, found))
        else:
            self._pending = [(kf.id, found)]
        if len(self._pending) >= cfg.loop_confirmations:
            self._close_loop(kf, found)
            self._pending = []

    def _close_loop(self, kf: KeyFrame, found) -> None:
        pose_cw, old_id, inliers, seen_px, seen_pid = found
        old = self.keyframes[old_id]
        corrected = invert(pose_cw)  # where the old map says the current keyframe really is
        scale = self._loop_scale(kf, corrected, seen_px, seen_pid)
        if not 0.6 < scale < 1.7:
            log.info("Loop to keyframe %d rejected: it would change the scale by %.2fx", old_id, scale)
            return
        rot_i, t_i, s_i = Rotation.from_matrix(old.pose[:3, :3]), old.pose[:3, 3], np.array(1.0)
        rot_j, t_j, s_j = Rotation.from_matrix(corrected[:3, :3]), corrected[:3, 3], np.array(scale)
        rot_ij, t_ij, s_ij = relative_sim3(rot_i, t_i, s_i, rot_j, t_j, s_j)
        self.loop_edges.append((old.id, kf.id, rot_ij.as_matrix(), np.asarray(t_ij), float(s_ij), 3.0))
        self._optimize_graph()
        self.loops.append((kf.id, old.id))
        log.info("Loop closed: keyframe %d recognised keyframe %d (%d points, scale %.3f)", kf.id, old.id, inliers, scale)

    def _loop_scale(self, kf: KeyFrame, corrected: np.ndarray, seen_px: np.ndarray, seen_pid: np.ndarray) -> float:
        """How much the current local map must stretch to match the old one, from the distances to shared points."""
        known = np.array([i for i in np.where(kf.pids >= 0)[0]
                          if int(kf.pids[i]) in self.points and not self.points[int(kf.pids[i])].provisional], int)
        if len(known) < 8 or len(seen_px) < 8:
            return 1.0
        distance, nearest = cKDTree(kf.px[known]).query(seen_px, distance_upper_bound=3.0)
        ratios = []
        for row in np.where(np.isfinite(distance))[0]:
            mine = self.points[int(kf.pids[known[nearest[row]]])].xyz
            theirs = self.points.get(int(seen_pid[row]))
            if theirs is None:
                continue
            near = np.linalg.norm(mine - kf.pose[:3, 3])
            if near > 1e-9:
                ratios.append(np.linalg.norm(theirs.xyz - corrected[:3, 3]) / near)
        if len(ratios) < 8:
            return 1.0
        return float(np.clip(np.median(ratios), 0.4, 2.5))

    def _optimize_graph(self) -> None:
        ids = sorted(self.keyframes)
        poses = {k: self.keyframes[k].pose for k in ids}
        edges = []
        for a, b in zip(ids, ids[1:]):
            rel = invert(poses[a]) @ poses[b]
            edges.append((a, b, rel[:3, :3], rel[:3, 3], 1.0, 1.0))
        edges += self.loop_edges
        result = optimize_pose_graph(poses, edges, {ids[0]}, scale_weight=1.0)
        # Re-anchor every point on its first keyframe, so it moves and stretches with that keyframe.
        anchor = {}
        for point in self.points.values():
            if point.obs:
                anchor[point.id] = min(point.obs)
        old_poses = {k: self.keyframes[k].pose.copy() for k in ids}
        for pid, point in self.points.items():
            a = anchor.get(pid)
            if a is None:
                continue
            rot, t, s = result[a]
            local = old_poses[a][:3, :3].T @ (point.xyz - old_poses[a][:3, 3])
            point.xyz = s * rot @ local + t
        scales = {}
        for k in ids:
            rot, t, s = result[k]
            self.keyframes[k].pose = make_pose(rot, t)
            scales[k] = s
        self.frames = [(i, tt, ref, rr, rt * scales[ref] if ref in scales and rt is not None else rt) for i, tt, ref, rr, rt in self.frames]
        last = self.keyframes[max(ids)]
        self.pose = last.pose.copy()
        self._velocity = np.eye(4)
        self.loop_edges = [e for e in self.loop_edges]

    # ------------------------------------------------------------ relocalizing
    def _relocalize(self, gray: np.ndarray, t: float, index: int) -> Estimate:
        found = self._orb_localize(gray, max(self.keyframes), min_inliers=25)
        if found is None:
            self.frames.append((index, None, None, None, None))
            return Estimate(index, LOST, None)
        pose_cw, _, _, seed_px, seed_pid = found
        self.pose = invert(pose_cw)
        self._velocity = np.eye(4)
        self._reset_tracking()
        corners = self._detect(gray, self.cfg.max_features, seed_px)
        # Seed tracking with the matched 3D points (so it continues in the old map) plus fresh corners.
        self.t_px = np.concatenate([seed_px, corners]) if len(corners) else seed_px
        self.t_pid = np.concatenate([seed_pid, np.full(len(corners), -1, np.int64)])
        n = len(self.t_px)
        self.t_tid = np.arange(self._next_track, self._next_track + n)
        self._next_track += n
        self.t_origin = np.full(n, max(self.keyframes), np.int64)
        self.t_origin_px = self.t_px.copy()
        self.state = OK
        self._frames_since_kf = self.cfg.keyframe_min_frames
        self._add_keyframe(gray, t, index, self.pose)
        self._record(index, t)
        log.info("Relocalized at frame %d", index)
        return Estimate(index, OK, self.pose.copy(), tracked=int((self.t_pid >= 0).sum()), keyframe=True)

    # --------------------------------------------------------------- outputs
    def _record(self, index: int, t: float) -> None:
        ref = max(self.keyframes)
        kf = self.keyframes[ref]
        rel = invert(kf.pose) @ self.pose
        self.frames.append((index, t, ref, rel[:3, :3].copy(), rel[:3, 3].copy()))

    def trajectory(self) -> list[tuple[int, float, np.ndarray]]:
        """(frame index, time, camera-to-world pose) for every tracked frame, after all corrections."""
        out = []
        for index, t, ref, rot, trans in self.frames:
            if ref is None or ref not in self.keyframes:
                continue
            kf = self.keyframes[ref]
            out.append((index, t, make_pose(kf.pose[:3, :3] @ rot, kf.pose[:3, 3] + kf.pose[:3, :3] @ trans)))
        return out

    def set_scale(self, metres_per_unit: float) -> None:
        self.metric_scale = float(metres_per_unit)

    def positions(self, metric: bool = True) -> np.ndarray:
        scale = self.metric_scale if metric else 1.0
        return np.array([p[:3, 3] * scale for _, _, p in self.trajectory()])

    def map_points(self) -> np.ndarray:
        return np.array([p.xyz for p in self.points.values()]).reshape(-1, 3)

    def global_bundle_adjust(self, max_nfev: int = 15) -> float:
        """Refine every keyframe and point together. Slow on long flights; meant for when the flight is over."""
        ids = sorted(self.keyframes)
        pids = [p for p, pt in self.points.items() if not pt.provisional and len(pt.obs) >= 2]
        observations = [(kid, pid, float(px[0]), float(px[1])) for pid in pids for kid, (px, _) in self.points[pid].obs.items()]
        result = bundle_adjust(self.k, {k: self.keyframes[k].pose for k in ids}, {ids[0]}, {p: self.points[p].xyz for p in pids},
                               observations, max_nfev=max_nfev)
        old = {k: self.keyframes[k].pose.copy() for k in ids}
        for k in ids:
            self.keyframes[k].pose = result.poses[k]
        for pid, xyz in result.points.items():
            self.points[pid].xyz = xyz
        # Frames between keyframes follow their keyframe; their relative motion is unchanged.
        return result.error_after
