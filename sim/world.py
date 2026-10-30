"""2D world made of line segments, unicycle robots, and a raycast LiDAR."""
import numpy as np
from pgo.core import compose, wrap


def box(x0, y0, x1, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def default_world():
    segs = box(0, 0, 24, 24)
    segs += box(6, 6, 9, 10) + box(14, 4, 17, 7) + box(10, 15, 13, 20) + box(18, 14, 21, 18)
    segs += [(0, 12, 5, 12), (24, 9, 19, 9)]
    return np.array(segs, float)


def raycast(segs, pose, angles, max_range):
    """Distance to nearest segment along each ray (max_range if none)."""
    th = pose[2] + angles
    d = np.stack([np.cos(th), np.sin(th)], 1)  # (N,2)
    p = pose[:2]
    a = segs[:, :2]
    e = segs[:, 2:] - a  # (M,2)
    # solve p + t d = a + u e
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (N,M)
    ap = a[None] - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ap[..., 0] * e[None, :, 1] - ap[..., 1] * e[None, :, 0]) / den
        u = (ap[..., 0] * d[:, None, 1] - ap[..., 1] * d[:, None, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    return np.minimum(t.min(1), max_range)


class Lidar:
    def __init__(self, n_beams=360, max_range=8.0, sigma=0.02, rng=None):
        self.angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        self.max_range = max_range
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def scan(self, segs, pose):
        r = raycast(segs, pose, self.angles, self.max_range)
        return r + self.rng.normal(0, self.sigma, r.shape)

    def points(self, r):
        """Hits as 2D points in the sensor frame."""
        hit = r < self.max_range - 1e-3
        return np.stack([r[hit] * np.cos(self.angles[hit]), r[hit] * np.sin(self.angles[hit])], 1)


class Unicycle:
    """Pose integrates (v*dt, w*dt); odometry measures it with noise."""

    def __init__(self, pose, rng, sigma_t=0.02, sigma_r=0.01):
        self.pose = np.array(pose, float)
        self.rng, self.sigma_t, self.sigma_r = rng, sigma_t, sigma_r

    def step(self, v, w):
        u = np.array([v, 0.0, w])
        self.pose = compose(self.pose, u)
        z = u + self.rng.normal(0, [self.sigma_t, self.sigma_t, self.sigma_r])
        z[2] = wrap(z[2])
        return z


def control(lidar, ranges, v=0.3, rng=None):
    """Reactive wanderer: turn toward the freer side when something is ahead."""
    ang = lidar.angles
    if ranges[np.abs(ang) < 0.5].min() < 1.2:
        left = ranges[(ang > 0.3) & (ang < 1.5)].mean()
        right = ranges[(ang < -0.3) & (ang > -1.5)].mean()
        return 0.1, 0.5 if left > right else -0.5
    return v, rng.normal(0, 0.08) if rng is not None else 0.0
