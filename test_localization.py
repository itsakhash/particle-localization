"""
Tests for localization.py (and a smoke test for run_localization.py).
Needs numpy; the plotting smoke test is skipped if matplotlib is missing.

Run from the project folder:
    python -m unittest -v test_localization.py
"""
import math
import os
import tempfile
import unittest

import numpy as np

import localization as loc


def empty_room(width_m=5.0, height_m=4.0, res=0.1):
    """A rectangular room with 0.2 m walls all round."""
    rows, cols = int(height_m / res), int(width_m / res)
    occ = np.zeros((rows, cols), dtype=bool)
    occ[:2, :] = occ[-2:, :] = True
    occ[:, :2] = occ[:, -2:] = True
    return loc.GridMap(occ, res)


def slow_raycast(grid, poses, angles, max_range=loc.MAX_RANGE, step=loc.RAY_STEP):
    """Obvious, slow reference implementation to check the fast one against."""
    out = np.zeros((len(poses), len(angles)))
    for i, (x, y, th) in enumerate(poses):
        for j, a in enumerate(angles):
            d = max_range
            for k in range(1, int(round(max_range / step)) + 1):
                t = k * step
                if grid.is_occupied(x + t * math.cos(th + a), y + t * math.sin(th + a)):
                    d = t
                    break
            out[i, j] = d
    return out


class TestAngles(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(float(loc.wrap_angle(3 * math.pi / 2)), -math.pi / 2)
        self.assertAlmostEqual(float(loc.wrap_angle(-3 * math.pi / 2)), math.pi / 2)
        self.assertAlmostEqual(float(loc.wrap_angle(0.5)), 0.5)
        np.testing.assert_allclose(loc.wrap_angle(np.array([2 * math.pi, -2 * math.pi])), [0, 0], atol=1e-12)


class TestGridMap(unittest.TestCase):
    def setUp(self):
        self.room = empty_room()

    def test_size_in_meters(self):
        self.assertAlmostEqual(self.room.width, 5.0)
        self.assertAlmostEqual(self.room.height, 4.0)

    def test_walls_free_space_and_outside(self):
        self.assertTrue(self.room.is_occupied(0.1, 2.0))      # inside the left wall
        self.assertFalse(self.room.is_occupied(2.5, 2.0))     # open floor
        self.assertTrue(self.room.is_occupied(-1.0, 2.0))     # outside the map counts as wall
        self.assertTrue(self.room.is_occupied(2.5, 99.0))

    def test_accepts_arrays(self):
        r = self.room.is_occupied(np.array([0.1, 2.5, -3.0]), np.array([2.0, 2.0, 2.0]))
        self.assertEqual(r.tolist(), [True, False, True])

    def test_blocked_uses_the_robot_radius(self):
        self.assertFalse(self.room.is_occupied(0.25, 2.0))
        self.assertTrue(self.room.blocked(0.25, 2.0, radius=0.12))   # body touches the wall
        self.assertFalse(self.room.blocked(2.5, 2.0))

    def test_sample_free_poses_are_all_valid(self):
        poses = self.room.sample_free(300, np.random.default_rng(0))
        self.assertEqual(poses.shape, (300, 3))
        self.assertFalse(self.room.blocked(poses[:, 0], poses[:, 1]).any())
        self.assertTrue(np.all(np.abs(poses[:, 2]) <= math.pi))

    def test_sample_free_is_reproducible(self):
        a = self.room.sample_free(10, np.random.default_rng(3))
        b = self.room.sample_free(10, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_map_must_be_2d(self):
        with self.assertRaises(ValueError):
            loc.GridMap(np.zeros(5))


class TestDemoMap(unittest.TestCase):
    def test_shape_borders_and_free_space(self):
        g = loc.make_demo_map()
        self.assertAlmostEqual(g.width, 12.0)
        self.assertAlmostEqual(g.height, 9.0)
        self.assertTrue(g.occ[0, :].all() and g.occ[-1, :].all() and g.occ[:, 0].all() and g.occ[:, -1].all())
        free = 1 - g.occ.mean()
        self.assertGreater(free, 0.7)
        self.assertLess(free, 0.98)

    def test_is_not_mirror_symmetric(self):
        g = loc.make_demo_map()
        self.assertFalse(np.array_equal(g.occ, g.occ[:, ::-1]))
        self.assertFalse(np.array_equal(g.occ, g.occ[::-1, :]))


class TestRaycast(unittest.TestCase):
    def setUp(self):
        self.room = empty_room()

    def test_distance_to_the_wall_straight_ahead(self):
        r = loc.raycast(self.room, [[2.5, 2.0, 0.0]], [0.0])
        # right wall's inner face is at x = 4.8, so 2.3 m ahead (within one ray step)
        self.assertAlmostEqual(float(r[0, 0]), 2.3, delta=loc.RAY_STEP)

    def test_beam_angles_are_relative_to_the_heading(self):
        facing_east = loc.raycast(self.room, [[2.5, 2.0, 0.0]], [0.0])[0, 0]
        facing_north = loc.raycast(self.room, [[2.5, 2.0, math.pi / 2]], [0.0])[0, 0]
        left_of_east = loc.raycast(self.room, [[2.5, 2.0, 0.0]], [math.pi / 2])[0, 0]
        self.assertAlmostEqual(float(left_of_east), float(facing_north), delta=1e-6)
        self.assertGreater(float(facing_east), float(facing_north))   # 2.3 m vs 1.8 m

    def test_nothing_in_range_returns_max_range(self):
        r = loc.raycast(self.room, [[2.5, 2.0, 0.0]], [0.0], max_range=1.0)
        self.assertEqual(float(r[0, 0]), 1.0)

    def test_pose_outside_the_map_reads_as_touching_a_wall(self):
        r = loc.raycast(self.room, [[-5.0, 2.0, 0.0]], [0.0])
        self.assertAlmostEqual(float(r[0, 0]), loc.RAY_STEP, delta=1e-6)

    def test_output_shape(self):
        poses = self.room.sample_free(7, np.random.default_rng(0))
        self.assertEqual(loc.raycast(self.room, poses, loc.scan_angles(11)).shape, (7, 11))

    def test_matches_a_slow_reference_on_the_demo_map(self):
        g = loc.make_demo_map()
        poses = g.sample_free(25, np.random.default_rng(1))
        angles = loc.scan_angles(12)
        fast = loc.raycast(g, poses, angles)
        slow = slow_raycast(g, poses, angles)
        self.assertLess(np.mean(np.abs(fast - slow) > 1e-4), 0.01)   # float32 may differ on a razor edge

    def test_thin_wall_is_not_skipped(self):
        occ = np.zeros((40, 60), dtype=bool)
        occ[:, 30] = True               # a wall one cell (0.1 m) thick
        g = loc.GridMap(occ)
        r = loc.raycast(g, [[1.0, 2.0, 0.0]], [0.0])
        self.assertAlmostEqual(float(r[0, 0]), 2.0, delta=0.11)


class TestSensor(unittest.TestCase):
    def setUp(self):
        self.room = empty_room()
        self.angles = loc.scan_angles(24)
        self.pose = np.array([2.5, 2.0, 0.3])

    def test_noise_is_centered_with_the_requested_spread(self):
        rng = np.random.default_rng(0)
        true = loc.raycast(self.room, self.pose, self.angles)[0]
        scans = np.array([loc.simulate_scan(self.room, self.pose, self.angles, rng, outlier=0.0, sigma=0.1)
                          for _ in range(400)])
        # clipping at max_range biases a few beams, so check the clearly interior ones
        inner = true < loc.MAX_RANGE - 0.5
        self.assertLess(abs(float(np.mean(scans[:, inner] - true[inner]))), 0.02)
        self.assertAlmostEqual(float(np.std(scans[:, inner] - true[inner])), 0.1, delta=0.02)

    def test_readings_stay_inside_the_sensor_range(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            s = loc.simulate_scan(self.room, self.pose, self.angles, rng, sigma=1.0, outlier=0.3)
            self.assertTrue(np.all(s >= 0) and np.all(s <= loc.MAX_RANGE))

    def test_outliers_happen_at_roughly_the_requested_rate(self):
        rng = np.random.default_rng(2)
        true = loc.raycast(self.room, self.pose, self.angles)[0]
        bad = 0
        total = 0
        for _ in range(300):
            s = loc.simulate_scan(self.room, self.pose, self.angles, rng, sigma=0.0, outlier=0.2)
            bad += int(np.sum(np.abs(s - true) > 0.05))
            total += len(s)
        self.assertAlmostEqual(bad / total, 0.2, delta=0.04)


class TestMotion(unittest.TestCase):
    def test_straight_line(self):
        p = loc.move([[1.0, 1.0, 0.0]], 0.5, 0.0, 2.0)
        np.testing.assert_allclose(p[0], [2.0, 1.0, 0.0], atol=1e-9)

    def test_heading_matters(self):
        p = loc.move([[1.0, 1.0, math.pi / 2]], 1.0, 0.0, 1.0)
        np.testing.assert_allclose(p[0], [1.0, 2.0, math.pi / 2], atol=1e-9)

    def test_turn_in_place(self):
        p = loc.move([[1.0, 1.0, 0.0]], 0.0, 1.0, 0.5)
        np.testing.assert_allclose(p[0], [1.0, 1.0, 0.5], atol=1e-9)

    def test_heading_wraps(self):
        p = loc.move([[0.0, 0.0, 3.0]], 0.0, 1.0, 1.0)
        self.assertLess(p[0, 2], 0.0)
        self.assertAlmostEqual(float(p[0, 2]), 4.0 - 2 * math.pi, places=9)

    def test_input_is_not_modified(self):
        start = np.array([[1.0, 1.0, 0.0]])
        loc.move(start, 1.0, 0.0, 1.0)
        np.testing.assert_array_equal(start, [[1.0, 1.0, 0.0]])

    def test_noise_gives_every_pose_its_own_motion(self):
        rng = np.random.default_rng(0)
        p = loc.move(np.zeros((500, 3)), 1.0, 0.0, 1.0, rng, 0.1, 0.1)
        self.assertAlmostEqual(float(np.std(p[:, 0])), 0.1, delta=0.02)
        self.assertAlmostEqual(float(np.mean(p[:, 0])), 1.0, delta=0.02)
        self.assertGreater(float(np.std(p[:, 2])), 0.05)

    def test_zero_noise_is_deterministic(self):
        rng = np.random.default_rng(0)
        a = loc.move(np.zeros((5, 3)), 1.0, 0.2, 1.0, rng, 0.0, 0.0)
        np.testing.assert_allclose(a, loc.move(np.zeros((5, 3)), 1.0, 0.2, 1.0), atol=1e-12)


class TestSimRobot(unittest.TestCase):
    def setUp(self):
        self.room = empty_room()

    def test_never_drives_through_a_wall(self):
        robot = loc.SimRobot(self.room, [4.0, 2.0, 0.0], np.random.default_rng(0))
        for _ in range(100):
            robot.step(0.5, 0.0, 0.2)        # keeps pushing east into the wall
            self.assertFalse(self.room.blocked(robot.pose[0], robot.pose[1], loc.ROBOT_RADIUS * 0.9))

    def test_can_still_turn_while_blocked(self):
        robot = loc.SimRobot(self.room, [4.55, 2.0, 0.0], np.random.default_rng(0))
        for _ in range(30):
            robot.step(0.5, 1.0, 0.2)
        self.assertNotAlmostEqual(float(robot.pose[2]), 0.0, places=1)

    def test_scan_comes_from_the_true_pose(self):
        robot = loc.SimRobot(self.room, [2.5, 2.0, 0.0], np.random.default_rng(0))
        scan = robot.scan(loc.scan_angles(24))
        self.assertEqual(scan.shape, (24,))


class TestWanderPolicy(unittest.TestCase):
    ANGLES = loc.scan_angles(24)

    def test_drives_forward_when_clear(self):
        v, w = loc.wander_command(np.full(24, 4.0), self.ANGLES)
        self.assertGreater(v, 0.0)

    def test_stops_and_turns_when_the_front_is_blocked(self):
        scan = np.full(24, 4.0)
        scan[np.abs(self.ANGLES) < math.radians(35)] = 0.4
        v, w = loc.wander_command(scan, self.ANGLES)
        self.assertEqual(v, 0.0)
        self.assertNotEqual(w, 0.0)

    def test_turns_toward_the_more_open_side(self):
        scan = np.full(24, 1.0)
        scan[np.abs(self.ANGLES) < math.radians(35)] = 0.4
        scan[self.ANGLES > 0.5] = 4.0            # lots of room on the left (positive angles)
        self.assertGreater(loc.wander_command(scan, self.ANGLES)[1], 0)
        scan[self.ANGLES > 0.5] = 1.0
        scan[self.ANGLES < -0.5] = 4.0           # now the right
        self.assertLess(loc.wander_command(scan, self.ANGLES)[1], 0)


class TestParticleFilterPieces(unittest.TestCase):
    def setUp(self):
        self.g = loc.make_demo_map()
        self.angles = loc.scan_angles()

    def pf(self, n=300, seed=0, **kw):
        return loc.ParticleFilter(self.g, n, self.angles, rng=np.random.default_rng(seed), **kw)

    def test_global_init_covers_free_space_with_equal_weights(self):
        pf = self.pf(400)
        self.assertEqual(pf.particles.shape, (400, 3))
        self.assertFalse(self.g.blocked(pf.particles[:, 0], pf.particles[:, 1]).any())
        np.testing.assert_allclose(pf.weights, 1 / 400)
        self.assertGreater(pf.spread(), 2.5)     # really spread out

    def test_gaussian_init_is_centered_on_the_pose(self):
        pf = self.pf(2000)
        pf.init_gaussian([6.0, 4.0, 1.0], std_xy=0.2, std_theta=0.1)
        est = pf.estimate()
        np.testing.assert_allclose(est, [6.0, 4.0, 1.0], atol=0.05)
        self.assertAlmostEqual(pf.spread(), 0.2 * math.sqrt(2 * math.log(2)), delta=0.03)   # median of a 2D Gaussian cloud

    def test_spread_ignores_a_few_far_away_particles(self):
        pf = self.pf(100)
        pf.particles[:] = [6.0, 4.0, 0.0]
        pf.particles[:3] = [1.0, 1.0, 0.0]            # 3% strays (like injected random particles)
        pf.weights = np.full(100, 0.01)
        self.assertLess(pf.spread(), 0.25)   # a plain mean distance would be about 0.34

    def test_spread_grows_when_most_of_the_cloud_is_spread_out(self):
        pf = self.pf(100)
        pf.particles[:50] = [6.0, 4.0, 0.0]
        pf.particles[50:] = [1.0, 1.0, 0.0]
        pf.weights = np.full(100, 0.01)
        self.assertGreater(pf.spread(), 2.0)

    def test_neff_range(self):
        pf = self.pf(100)
        self.assertAlmostEqual(pf.neff(), 100.0)
        pf.weights = np.zeros(100)
        pf.weights[7] = 1.0
        self.assertAlmostEqual(pf.neff(), 1.0)

    def test_weights_stay_normalized(self):
        pf = self.pf()
        truth = pf.particles[0].copy()
        pf.update(loc.raycast(self.g, truth, self.angles)[0])
        self.assertAlmostEqual(float(pf.weights.sum()), 1.0)
        pf.predict(0.5, 0.1, 0.2)
        self.assertAlmostEqual(float(pf.weights.sum()), 1.0)

    def test_update_prefers_the_true_pose(self):
        pf = self.pf(3)
        truth = np.array([3.0, 3.0, 0.5])
        pf.particles = np.array([truth, truth + [1.0, 0.0, 0.0], truth + [0.0, 0.0, math.pi]])
        pf.weights = np.full(3, 1 / 3)
        pf.update(loc.raycast(self.g, truth, self.angles)[0])
        self.assertEqual(int(np.argmax(pf.weights)), 0)
        self.assertGreater(pf.weights[0], 0.9)

    def test_predict_kills_particles_that_drive_into_walls(self):
        pf = self.pf(2)
        pf.particles = np.array([[3.0, 3.0, 0.0], [0.35, 3.0, math.pi]])   # second faces the left wall
        pf.weights = np.array([0.5, 0.5])
        pf.predict(1.0, 0.0, 0.5)
        self.assertAlmostEqual(float(pf.weights[1]), 0.0, places=6)
        self.assertGreater(pf.weights[0], 0.99)

    def test_all_particles_dead_restarts_globally(self):
        pf = self.pf(50)
        pf.particles[:] = [-10.0, -10.0, 0.0]          # everything off the map
        pf.predict(0.1, 0.0, 0.1)
        self.assertFalse(self.g.blocked(pf.particles[:, 0], pf.particles[:, 1]).any())
        self.assertAlmostEqual(float(pf.weights.sum()), 1.0)

    def test_resample_concentrates_on_the_heavy_particle(self):
        pf = self.pf(200)
        keep = pf.particles[5].copy()
        pf.weights = np.zeros(200)
        pf.weights[5] = 1.0
        pf.inject_fraction = 0.0
        pf.resample()
        self.assertTrue(np.all(np.abs(pf.particles[:, :2] - keep[:2]) < 0.2))
        np.testing.assert_allclose(pf.weights, 1 / 200)

    def test_resample_adds_a_little_jitter_so_copies_are_not_identical(self):
        pf = self.pf(300, inject_fraction=0.0)
        pf.particles[:] = [6.0, 4.0, 0.5]
        pf.weights = np.full(300, 1 / 300)
        pf.resample()
        self.assertAlmostEqual(float(np.std(pf.particles[:, 0])), loc.ROUGHEN_XY, delta=0.01)
        self.assertAlmostEqual(float(np.std(pf.particles[:, 2])), loc.ROUGHEN_THETA, delta=0.01)

    def test_resample_copies_in_proportion_to_weight(self):
        pf = self.pf(1000, inject_fraction=0.0)
        a, b = np.array([2.0, 2.0, 0.0]), np.array([8.0, 6.0, 0.0])
        pf.particles = np.vstack([np.tile(a, (500, 1)), np.tile(b, (500, 1))])
        pf.weights = np.concatenate([np.full(500, 0.75 / 500), np.full(500, 0.25 / 500)])
        pf.resample()
        near_a = np.sum(np.hypot(pf.particles[:, 0] - 2.0, pf.particles[:, 1] - 2.0) < 0.5)
        self.assertAlmostEqual(near_a / 1000, 0.75, delta=0.02)

    def test_injection_keeps_some_random_particles(self):
        pf = self.pf(500, inject_fraction=0.1)
        pf.particles[:] = [6.0, 4.5, 0.0]
        pf.weights = np.full(500, 1 / 500)
        pf.resample()
        far = np.sum(np.hypot(pf.particles[:, 0] - 6.0, pf.particles[:, 1] - 4.5) > 1.0)
        self.assertGreater(far, 20)
        self.assertLessEqual(far, 60)

    def test_heading_estimate_is_a_circular_mean(self):
        pf = self.pf(2)
        pf.particles = np.array([[3.0, 3.0, 3.1], [3.0, 3.0, -3.1]])
        pf.weights = np.array([0.5, 0.5])
        self.assertAlmostEqual(abs(float(pf.estimate()[2])), math.pi, delta=0.05)

    def test_estimate_is_the_weighted_mean(self):
        pf = self.pf(2)
        pf.particles = np.array([[2.0, 2.0, 0.0], [4.0, 6.0, 0.0]])
        pf.weights = np.array([0.75, 0.25])
        np.testing.assert_allclose(pf.estimate()[:2], [2.5, 3.0])

    def test_step_resamples_only_when_weights_degenerate(self):
        pf = self.pf(200)
        before = pf.particles.copy()
        pf.update = lambda scan: None        # weights stay equal -> neff = n -> no resample
        pf.predict = lambda v, w, dt: None
        pf.step(0, 0, 0.1, None)
        np.testing.assert_array_equal(pf.particles, before)


class TestMetrics(unittest.TestCase):
    def test_pose_error_wraps_the_heading(self):
        d, a = loc.pose_error([1.0, 1.0, 3.1], [4.0, 5.0, -3.1])
        self.assertAlmostEqual(d, 5.0)
        self.assertAlmostEqual(a, 2 * math.pi - 6.2, places=6)

    def test_converged_step_needs_a_streak(self):
        errors = [5, 4, 0.1, 0.1, 3, 0.1, 0.1, 0.1, 0.1]
        self.assertEqual(loc.converged_step(errors, tol=0.3, hold=3), 5)
        self.assertIsNone(loc.converged_step(errors, tol=0.3, hold=5))

    def test_converged_step_can_also_require_a_tight_cloud(self):
        errors = [0.1] * 6
        self.assertIsNone(loc.converged_step(errors, hold=3, spreads=[2.0] * 6))
        self.assertEqual(loc.converged_step(errors, hold=3, spreads=[2.0, 2.0, 0.1, 0.1, 0.1, 0.1]), 2)

    def test_never_converging_is_none(self):
        self.assertIsNone(loc.converged_step([9.0] * 20))


class TestEndToEnd(unittest.TestCase):
    """Full simulations (a few seconds each)."""

    def test_filter_finds_a_lost_robot_across_seeds(self):
        ev = loc.evaluate(trials=4, first_seed=100, steps=150, n_particles=1500)
        self.assertGreaterEqual(ev["success_rate"], 0.75)
        self.assertLess(ev["median_final_error_m"], 0.15)

    def test_the_cloud_starts_wide_and_ends_tight(self):
        r = loc.run_trial(5, steps=120, n_particles=1500)
        self.assertGreater(r["spreads"][0], 1.0)
        self.assertLess(r["spreads"][-1], 0.3)
        self.assertGreater(r["errors"][0], r["errors"][-1])

    def test_same_seed_gives_identical_results(self):
        a = loc.run_trial(3, steps=15, n_particles=300)
        b = loc.run_trial(3, steps=15, n_particles=300)
        self.assertEqual(a["errors"], b["errors"])

    def test_recovers_after_being_kidnapped(self):
        r = loc.run_trial(7, steps=260, n_particles=1500, kidnap_at=120)
        self.assertIsNotNone(r["converged_step"])
        self.assertGreaterEqual(r["converged_step"], 120)
        self.assertLess(float(np.mean(r["errors"][-20:])), 0.2)

    def test_recording_keeps_one_frame_per_step(self):
        r = loc.run_trial(1, steps=8, n_particles=200, record=True)
        self.assertEqual(len(r["frames"]), 8)
        particles, truth, est = r["frames"][0]
        self.assertEqual(particles.shape, (200, 3))
        self.assertEqual(truth.shape, (3,))
        self.assertEqual(est.shape, (3,))

    def test_evaluate_reports_the_headline_numbers(self):
        ev = loc.evaluate(trials=2, first_seed=100, steps=60, n_particles=300)
        for key in ("trials", "success_rate", "median_steps_to_converge",
                    "median_seconds_to_converge", "median_final_error_m"):
            self.assertIn(key, ev)
        self.assertEqual(ev["trials"], 2)


class TestPlotting(unittest.TestCase):
    def test_snapshot_renders_to_a_png(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("matplotlib not installed")
        import run_localization as rl
        grid = loc.make_demo_map()
        result = loc.run_trial(1, steps=6, n_particles=150, record=True)
        fig, show = rl.make_figure(grid, result)
        show(5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            fig.savefig(path)
            self.assertGreater(os.path.getsize(path), 1000)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
