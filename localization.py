"""
Monte Carlo localization (a particle filter) for a robot on a known 2D map.

The robot is "kidnapped": it starts with no idea where it is. A cloud of particles
(guesses of x, y, heading) is spread over the whole map. Each step:

  1. predict   move every particle with the robot's motion command plus noise
  2. update    weight each particle by how well the range scan it WOULD see from
               that pose matches the scan the robot actually saw
  3. resample  copy likely particles, drop unlikely ones

After a few dozen steps the cloud collapses onto the true pose.

Everything is NumPy only. matplotlib is used just by run_localization.py.

Conventions: x, y in meters (y up), heading theta in radians (0 = +x axis),
map cells are indexed grid[row=y, col=x].
"""
import math

import numpy as np

# ---------------------------------------------------------------- defaults
RESOLUTION = 0.1          # meters per map cell
MAX_RANGE = 5.0           # meters, the range sensor's limit
NUM_BEAMS = 24            # beams spread over 360 degrees
RAY_STEP = 0.1            # meters between samples along a ray
ROBOT_RADIUS = 0.12       # meters, used for collision checks
SENSOR_SIGMA = 0.10       # meters, true sensor noise
SENSOR_OUTLIER = 0.05     # probability a beam returns a random value
DEFAULT_PARTICLES = 2000
HIT_SIGMA = 0.20          # meters, the filter's assumed sensor noise (a bit pessimistic on purpose)
Z_RAND = 0.10             # floor of the beam likelihood (protects against outliers)
LIKELIHOOD_SCALE = 0.15   # < 1 softens the sum of beam log-likelihoods (beams are not independent)
MOTION_SIGMA_V = 0.04     # m/s noise assumed on the commanded speed
MOTION_SIGMA_W = 0.08     # rad/s noise assumed on the commanded turn rate
TRUE_SIGMA_V = 0.03       # noise the simulated robot really has
TRUE_SIGMA_W = 0.05
MAX_ADAPTIVE_INJECT = 0.5  # most particles that can be replaced at once when the filter detects it is lost
SLOW_RATE = 0.05          # how fast the long-term average match quality updates
FAST_RATE = 0.3           # ...and the short-term one (a sudden drop of fast below slow means 'I am lost')
LOST_RATIO = 0.6          # recent/long-term match quality below this means the robot was probably moved
INJECT_FRACTION = 0.02    # share of particles replaced by random ones at each resample (recovers from being lost)
ROUGHEN_XY = 0.02         # meters of jitter after resampling (keeps the cloud diverse)
ROUGHEN_THETA = 0.02      # radians


def wrap_angle(a):
    """Wrap an angle (or array of angles) into [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- the map
class GridMap:
    """An occupancy grid. occupied[row, col] is True for walls and obstacles."""

    def __init__(self, occupied, resolution=RESOLUTION):
        self.occ = np.asarray(occupied, dtype=bool)
        if self.occ.ndim != 2:
            raise ValueError("map must be 2D")
        self.res = float(resolution)
        self.rows, self.cols = self.occ.shape
        self.padded = np.pad(self.occ, 1, constant_values=True)   # wall border for fast ray lookups

    @property
    def width(self):
        return self.cols * self.res

    @property
    def height(self):
        return self.rows * self.res

    def is_occupied(self, x, y):
        """True where (x, y) is inside a wall or outside the map. Accepts arrays."""
        ix = np.floor(np.asarray(x) / self.res).astype(int)
        iy = np.floor(np.asarray(y) / self.res).astype(int)
        inside = (ix >= 0) & (ix < self.cols) & (iy >= 0) & (iy < self.rows)
        out = np.ones(np.shape(ix), dtype=bool)
        out[inside] = self.occ[iy[inside], ix[inside]]
        return out

    def blocked(self, x, y, radius=ROBOT_RADIUS):
        """True where a robot of this radius centered at (x, y) would touch a wall."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        hit = self.is_occupied(x, y)
        for k in range(8):
            a = k * math.pi / 4
            hit = hit | self.is_occupied(x + radius * math.cos(a), y + radius * math.sin(a))
        return hit

    def sample_free(self, n, rng, margin=ROBOT_RADIUS):
        """n random poses (x, y, theta) where the robot is not touching anything."""
        poses = np.empty((0, 3))
        while len(poses) < n:
            m = max(2 * (n - len(poses)), 16)
            x = rng.uniform(0, self.width, m)
            y = rng.uniform(0, self.height, m)
            ok = ~self.blocked(x, y, margin)
            th = rng.uniform(-math.pi, math.pi, m)
            poses = np.vstack([poses, np.column_stack([x, y, th])[ok]])
        return poses[:n]


def make_demo_map(resolution=RESOLUTION):
    """A 12 m x 9 m floor plan with rooms, a doorway, and some asymmetric furniture.
    (Symmetric maps make localization ambiguous, so this one is deliberately lopsided.)"""
    rows, cols = int(9 / resolution), int(12 / resolution)
    occ = np.zeros((rows, cols), dtype=bool)

    def box(x0, y0, x1, y1):
        occ[int(y0 / resolution):int(y1 / resolution), int(x0 / resolution):int(x1 / resolution)] = True

    t = 0.2
    box(0, 0, 12, t)            # outer walls
    box(0, 9 - t, 12, 9)
    box(0, 0, t, 9)
    box(12 - t, 0, 12, 9)
    box(4.5, t, 4.5 + t, 5.5)   # vertical wall with a doorway above y = 5.5
    box(4.5, 7.0, 4.5 + t, 9 - t)
    box(4.5 + t, 5.0, 8.5, 5.0 + t)       # horizontal wall to the right
    box(9.5, 5.0, 12 - t, 5.0 + t)        # ...with a doorway between x = 8.5 and 9.5
    box(1.5, 1.5, 2.5, 2.3)     # furniture
    box(2.0, 6.0, 2.4, 8.0)
    box(6.5, 2.0, 7.7, 2.6)
    box(10.0, 1.0, 10.4, 3.5)
    box(6.0, 6.5, 6.8, 7.3)
    return GridMap(occ, resolution)


# ---------------------------------------------------------------- range sensor
def scan_angles(num_beams=NUM_BEAMS):
    """Beam directions relative to the robot's heading, covering 360 degrees."""
    return np.linspace(-math.pi, math.pi, num_beams, endpoint=False)


def raycast(grid, poses, rel_angles, max_range=MAX_RANGE, step=RAY_STEP):
    """Expected range readings. poses: (N, 3) -> returns (N, B) distances in meters.

    Each ray is walked in small steps until it enters an occupied cell; rays that
    hit nothing return max_range. Anything outside the map counts as a wall.
    (The map is padded with a wall border so the inner loop is plain integer indexing.)
    """
    poses = np.atleast_2d(np.asarray(poses, dtype=np.float32))
    ang = poses[:, 2:3] + np.asarray(rel_angles, dtype=np.float32)[None, :]       # (N, B)
    ts = np.arange(step, max_range + step / 2, step, dtype=np.float32)            # (S,)
    inv = np.float32(1.0 / grid.res)
    gx = poses[:, 0, None, None] * inv + ts[None, None, :] * inv * np.cos(ang)[:, :, None]
    gy = poses[:, 1, None, None] * inv + ts[None, None, :] * inv * np.sin(ang)[:, :, None]
    ix = np.clip(np.floor(gx).astype(np.int32) + 1, 0, grid.cols + 1)
    iy = np.clip(np.floor(gy).astype(np.int32) + 1, 0, grid.rows + 1)
    hit = grid.padded.ravel()[iy * (grid.cols + 2) + ix]
    first = hit.argmax(axis=2)
    return np.where(hit.any(axis=2), ts[first], np.float32(max_range)).astype(float)


def simulate_scan(grid, pose, rel_angles, rng, max_range=MAX_RANGE,
                  sigma=SENSOR_SIGMA, outlier=SENSOR_OUTLIER):
    """What the real sensor reports: true ranges plus Gaussian noise and rare outliers."""
    true = raycast(grid, pose, rel_angles, max_range)[0]
    noisy = true + rng.normal(0.0, sigma, true.shape)
    bad = rng.random(true.shape) < outlier
    noisy[bad] = rng.uniform(0.0, max_range, int(bad.sum()))
    return np.clip(noisy, 0.0, max_range)


# ---------------------------------------------------------------- motion
def move(poses, v, w, dt, rng=None, sigma_v=0.0, sigma_w=0.0):
    """Drive poses (N, 3) with speed v (m/s) and turn rate w (rad/s) for dt seconds.
    With noise, every pose gets its own random speed and turn rate."""
    poses = np.atleast_2d(np.asarray(poses, dtype=float)).copy()
    n = len(poses)
    vs = np.full(n, v, dtype=float)
    ws = np.full(n, w, dtype=float)
    if rng is not None:
        vs += rng.normal(0.0, sigma_v, n)
        ws += rng.normal(0.0, sigma_w, n)
    mid = poses[:, 2] + 0.5 * ws * dt
    poses[:, 0] += vs * dt * np.cos(mid)
    poses[:, 1] += vs * dt * np.sin(mid)
    poses[:, 2] = wrap_angle(poses[:, 2] + ws * dt)
    return poses


class SimRobot:
    """The 'real' robot in the simulation: noisy motion, never drives through walls."""

    def __init__(self, grid, pose, rng):
        self.grid = grid
        self.pose = np.asarray(pose, dtype=float)
        self.rng = rng

    def step(self, v, w, dt):
        new = move(self.pose, v, w, dt, self.rng, TRUE_SIGMA_V, TRUE_SIGMA_W)[0]
        if self.grid.blocked(new[0], new[1]):
            new = self.pose.copy()
            new[2] = wrap_angle(self.pose[2] + w * dt)   # can still turn in place
        self.pose = new
        return self.pose

    def scan(self, rel_angles):
        return simulate_scan(self.grid, self.pose, rel_angles, self.rng)


def wander_command(scan, rel_angles, t=0.0):
    """A simple exploring policy: go forward, turn toward open space when blocked."""
    rel = np.asarray(rel_angles)
    front = np.abs(rel) < math.radians(35)
    if scan[front].min() < 0.8:
        left = scan[(rel > 0) & (rel < math.pi * 0.75)].mean()
        right = scan[(rel < 0) & (rel > -math.pi * 0.75)].mean()
        return 0.0, (1.2 if left >= right else -1.2)
    return 0.5, 0.25 * math.sin(0.7 * t)


# ---------------------------------------------------------------- the filter
class ParticleFilter:
    def __init__(self, grid, n=DEFAULT_PARTICLES, rel_angles=None, max_range=MAX_RANGE,
                 hit_sigma=HIT_SIGMA, z_rand=Z_RAND, likelihood_scale=LIKELIHOOD_SCALE,
                 sigma_v=MOTION_SIGMA_V, sigma_w=MOTION_SIGMA_W,
                 inject_fraction=INJECT_FRACTION, rng=None):
        self.grid = grid
        self.n = int(n)
        self.rel_angles = scan_angles() if rel_angles is None else np.asarray(rel_angles)
        self.max_range = max_range
        self.hit_sigma = hit_sigma
        self.z_rand = z_rand
        self.likelihood_scale = likelihood_scale
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.inject_fraction = inject_fraction
        self.rng = rng if rng is not None else np.random.default_rng()
        self.particles = np.zeros((self.n, 3))
        self.weights = np.full(self.n, 1.0 / self.n)
        self.w_slow = self.w_fast = None
        self.init_global()

    # --- initialization
    def init_global(self):
        """No idea where the robot is: spread particles over all free space."""
        self.particles = self.grid.sample_free(self.n, self.rng)
        self.weights = np.full(self.n, 1.0 / self.n)
        self.w_slow = self.w_fast = None

    def init_gaussian(self, pose, std_xy=0.3, std_theta=0.2):
        """Roughly know where the robot is."""
        pose = np.asarray(pose, dtype=float)
        noise = self.rng.normal(0.0, [std_xy, std_xy, std_theta], (self.n, 3))
        self.particles = pose + noise
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.weights = np.full(self.n, 1.0 / self.n)
        self.w_slow = self.w_fast = None

    # --- the three steps
    def predict(self, v, w, dt):
        self.particles = move(self.particles, v, w, dt, self.rng, self.sigma_v, self.sigma_w)
        crashed = self.grid.blocked(self.particles[:, 0], self.particles[:, 1])
        self.weights = np.where(crashed, 0.0, self.weights)
        self._normalize()

    def update(self, scan):
        """Re-weight every particle by how well its expected scan matches the real one."""
        expected = raycast(self.grid, self.particles, self.rel_angles, self.max_range)
        diff = np.asarray(scan, dtype=float)[None, :] - expected
        p_hit = np.exp(-0.5 * (diff / self.hit_sigma) ** 2) / (self.hit_sigma * math.sqrt(2 * math.pi))
        p = (1.0 - self.z_rand) * p_hit + self.z_rand / self.max_range
        log_w = self.likelihood_scale * np.log(p).sum(axis=1)
        self._track_match_quality(float(np.mean(np.exp(log_w))))
        self.weights = self.weights * np.exp(log_w - log_w.max())
        self._normalize()

    def neff(self):
        """Effective number of particles (n = all equal, 1 = one particle has everything)."""
        return 1.0 / float(np.sum(self.weights ** 2))

    def resample(self):
        """Systematic resampling, then a little jitter, then optional random injection."""
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.minimum(idx, self.n - 1)
        self.particles = self.particles[idx].copy()
        self.particles[:, :2] += self.rng.normal(0.0, ROUGHEN_XY, (self.n, 2))
        self.particles[:, 2] = wrap_angle(self.particles[:, 2] + self.rng.normal(0.0, ROUGHEN_THETA, self.n))
        k = int(round(max(self.inject_fraction, self.lost_fraction()) * self.n))
        if k:
            self.particles[self.rng.choice(self.n, k, replace=False)] = self.grid.sample_free(k, self.rng)
        self.weights = np.full(self.n, 1.0 / self.n)

    def _track_match_quality(self, quality):
        """Keep a slow and a fast running average of how well the scans match the particles."""
        if self.w_slow is None:
            self.w_slow = self.w_fast = quality
        else:
            self.w_slow += SLOW_RATE * (quality - self.w_slow)
            self.w_fast += FAST_RATE * (quality - self.w_fast)

    def lost_fraction(self):
        """0 while scans match well. When the recent match is much worse than the long-term
        average (e.g. the robot was picked up and moved), the share of particles to replace
        with random ones so the filter can find the robot again."""
        if self.w_slow is None or self.w_slow <= 0:
            return 0.0
        ratio = self.w_fast / self.w_slow
        if ratio >= LOST_RATIO:                     # ordinary ups and downs: leave the cloud alone
            return 0.0
        return float(np.clip((LOST_RATIO - ratio) / LOST_RATIO, 0.0, MAX_ADAPTIVE_INJECT))

    def step(self, v, w, dt, scan):
        """One full cycle: predict, update, resample when the weights have degenerated."""
        self.predict(v, w, dt)
        self.update(scan)
        if self.neff() < self.n / 2:
            self.resample()

    # --- results
    def estimate(self):
        """Weighted mean pose (heading uses a circular mean so +179 and -179 average to 180)."""
        w = self.weights
        x = float(np.sum(w * self.particles[:, 0]))
        y = float(np.sum(w * self.particles[:, 1]))
        th = float(math.atan2(np.sum(w * np.sin(self.particles[:, 2])),
                              np.sum(w * np.cos(self.particles[:, 2]))))
        return np.array([x, y, th])

    def spread(self):
        """How spread out the cloud is: the weighted median distance (meters) of the particles
        from the estimate. A median, so the few random particles kept for recovery don't
        make a tight cloud look loose."""
        est = self.estimate()
        d = np.hypot(self.particles[:, 0] - est[0], self.particles[:, 1] - est[1])
        order = np.argsort(d)
        cum = np.cumsum(self.weights[order])
        return float(d[order][min(int(np.searchsorted(cum, 0.5)), self.n - 1)])

    def _normalize(self):
        total = self.weights.sum()
        if not np.isfinite(total) or total <= 0:
            self.init_global()                              # every particle died: the filter is lost, start over
        else:
            self.weights = self.weights / total


# ---------------------------------------------------------------- experiments
def pose_error(estimate, truth):
    """(position error in meters, heading error in radians)."""
    d = math.hypot(estimate[0] - truth[0], estimate[1] - truth[1])
    a = abs(float(wrap_angle(estimate[2] - truth[2])))
    return d, a


def converged_step(errors, tol=0.3, hold=10, spreads=None, spread_tol=0.5):
    """First step from which the estimate stays within tol meters of the truth for `hold`
    steps in a row, else None. If spreads are given, the particle cloud must also be
    tighter than spread_tol, so a lucky guess from a wide-open cloud doesn't count."""
    run = 0
    for i, e in enumerate(errors):
        good = e < tol and (spreads is None or spreads[i] < spread_tol)
        run = run + 1 if good else 0
        if run >= hold:
            return i - hold + 1
    return None


def run_trial(seed, steps=250, dt=0.2, n_particles=DEFAULT_PARTICLES, grid=None,
              kidnap_at=None, inject_fraction=INJECT_FRACTION, tol=0.3, hold=10, record=False):
    """Simulate one run: robot placed at random, filter starts lost.

    Returns a dict with per-step position errors, when it converged (or None),
    and, if record=True, the history needed to animate it.
    """
    rng = np.random.default_rng(seed)
    grid = grid if grid is not None else make_demo_map()
    angles = scan_angles()
    robot = SimRobot(grid, grid.sample_free(1, rng)[0], rng)
    pf = ParticleFilter(grid, n_particles, angles, inject_fraction=inject_fraction, rng=rng)
    scan = robot.scan(angles)
    errors, spreads, frames = [], [], []
    for t in range(steps):
        if kidnap_at is not None and t == kidnap_at:
            robot.pose = grid.sample_free(1, rng)[0]
        v, w = wander_command(scan, angles, t * dt)
        robot.step(v, w, dt)
        scan = robot.scan(angles)
        pf.step(v, w, dt, scan)
        err, _ = pose_error(pf.estimate(), robot.pose)
        errors.append(err)
        spreads.append(pf.spread())
        if record:
            frames.append((pf.particles.copy(), robot.pose.copy(), pf.estimate()))
    start = kidnap_at if kidnap_at is not None else 0
    conv = converged_step(errors[start:], tol, hold, spreads[start:])
    return {"seed": seed, "errors": errors, "spreads": spreads, "dt": dt,
            "converged_step": None if conv is None else conv + start,
            "frames": frames, "kidnap_at": kidnap_at}


def evaluate(trials=20, first_seed=0, **kwargs):
    """Run many seeded trials and summarize. Returns a dict of the headline numbers."""
    results = [run_trial(first_seed + i, **kwargs) for i in range(trials)]
    ok = [r for r in results if r["converged_step"] is not None]
    start = lambda r: r["kidnap_at"] or 0
    steps_to_converge = [r["converged_step"] - start(r) for r in ok]
    final = [float(np.mean(r["errors"][-20:])) for r in results]
    return {
        "trials": trials,
        "success_rate": len(ok) / trials,
        "median_steps_to_converge": float(np.median(steps_to_converge)) if ok else None,
        "median_seconds_to_converge": (float(np.median(steps_to_converge)) * results[0]["dt"]) if ok else None,
        "median_final_error_m": float(np.median(final)),
        "results": results,
    }
