"""Run the simulation and build a Problem whose inter-robot edges come from LiDAR ICP."""
import numpy as np
from pgo.core import Problem, Edge, compose, between
from .world import default_world, Lidar, Unicycle, control
from .icp import icp


def simulate(n_robots=3, steps=150, seed=0, loop_range=5.0, stride=4,
             guess_sigma=(0.4, 0.4, 0.15), max_rms=0.06, min_frac=0.5, segs=None, verbose=True):
    """Returns (Problem, segs, scans). Each robot starts somewhere different, odometry is noisy,
    and inter-robot loop closures are computed by ICP between scans of different robots.

    ICP needs an initial guess for the relative pose. We emulate a rendezvous/place-recognition
    front end by perturbing the true relative pose with `guess_sigma`; ICP then refines it
    using only scan data, and the result is validated by residual and overlap.
    """
    rng = np.random.default_rng(seed)
    segs = default_world() if segs is None else segs
    lidar = Lidar(rng=rng)
    starts = [(3, 3, 0.5), (20, 20, 3.5), (3, 20, -0.7), (20, 3, 2.5), (12, 11, 0.0)][:n_robots]
    robots = [Unicycle(p, rng) for p in starts]
    gt, init, edges, scans = {}, {}, [], {}
    for r, rb in enumerate(robots):
        gt[(r, 0)] = rb.pose.copy()
        init[(r, 0)] = rb.pose.copy()
        scans[(r, 0)] = lidar.scan(segs, rb.pose)
    for k in range(1, steps):
        for r, rb in enumerate(robots):
            v, w = control(lidar, scans[(r, k - 1)], rng=rng)
            z = rb.step(v, w)
            gt[(r, k)] = rb.pose.copy()
            scans[(r, k)] = lidar.scan(segs, rb.pose)
            edges.append(Edge((r, k - 1), (r, k), z))
            init[(r, k)] = compose(init[(r, k - 1)], z)

    pts = {key: lidar.points(s) for key, s in scans.items()}
    tried = ok = bad = 0
    for ka in range(0, steps, stride):
        for kb in range(0, steps, stride):
            for a in range(n_robots):
                for b in range(a + 1, n_robots):
                    ia, ib = (a, ka), (b, kb)
                    true = between(gt[ia], gt[ib])
                    if np.linalg.norm(true[:2]) > loop_range:
                        continue
                    tried += 1
                    T, rms, frac = icp(pts[ia], pts[ib], true + rng.normal(0, guess_sigma))
                    if rms < max_rms and frac > min_frac:
                        ok += 1
                        bad += np.linalg.norm(T[:2] - true[:2]) > 0.3
                        edges.append(Edge(ia, ib, T))
    if verbose:
        print(f"ICP candidates {tried}, accepted {ok}, of which wrong (>0.3 m): {bad}")
    return Problem(n_robots, gt, init, edges), segs, scans
