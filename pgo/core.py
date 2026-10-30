"""SE(2) pose graph: synthetic data, centralized and distributed solvers."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def between(a, b):
    """Relative pose a^-1 * b."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = b[:2] - a[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap(b[2] - a[2])])


@dataclass
class Edge:
    i: tuple  # (robot, index)
    j: tuple
    z: np.ndarray  # measured relative pose (x, y, theta)


@dataclass
class Problem:
    n_robots: int
    gt: dict  # (robot, k) -> pose
    init: dict  # odometry-only initial guess
    edges: list

    def keys(self, r=None):
        return sorted(k for k in self.gt if r is None or k[0] == r)


def make_problem(n_robots=3, steps=60, sigma_t=0.02, sigma_r=0.01, loop_range=1.5, seed=0):
    """Unicycle robots on smooth random trajectories, noisy odometry,
    inter-robot loop closures when robots are within `loop_range`."""
    rng = np.random.default_rng(seed)
    gt, init, edges = {}, {}, []
    for r in range(n_robots):
        p = np.array([2.0 * r, 0.0, rng.uniform(0, np.pi)])
        gt[(r, 0)] = p.copy()
        init[(r, 0)] = p.copy()  # initial poses known (shared frame)
        w = 0.0
        for k in range(1, steps):
            w = 0.9 * w + rng.normal(0, 0.08)
            u = np.array([0.3, 0.0, w])  # unicycle: v*dt forward, w*dt turn
            p = compose(p, u)
            gt[(r, k)] = p.copy()
            z = u + rng.normal(0, [sigma_t, sigma_t, sigma_r])
            z[2] = wrap(z[2])
            edges.append(Edge((r, k - 1), (r, k), z))
            init[(r, k)] = compose(init[(r, k - 1)], z)
    for k in range(steps):
        for a in range(n_robots):
            for b in range(a + 1, n_robots):
                if np.linalg.norm(gt[(a, k)][:2] - gt[(b, k)][:2]) < loop_range:
                    z = between(gt[(a, k)], gt[(b, k)]) + rng.normal(0, [sigma_t, sigma_t, sigma_r])
                    z[2] = wrap(z[2])
                    edges.append(Edge((a, k), (b, k), z))
    return Problem(n_robots, gt, init, edges)


def _residual(xi, xj, z, w):
    return between(z, between(xi, xj)) * w  # z^-1 * (xi^-1 xj)


def solve(prob, free, fixed, x0, sigma_t=0.05, sigma_r=0.02, edges=None):
    """Least squares over poses in `free`; poses in `fixed` are constants.
    x0: dict with current estimates of every key (free and fixed)."""
    edges = prob.edges if edges is None else edges
    free = list(free)
    idx = {k: n for n, k in enumerate(free)}
    w = np.array([1 / sigma_t, 1 / sigma_t, 1 / sigma_r])
    rel = [e for e in edges if e.i in idx or e.j in idx]

    def fun(v):
        X = lambda k: v[3 * idx[k]:3 * idx[k] + 3] if k in idx else fixed[k]
        return np.concatenate([_residual(X(e.i), X(e.j), e.z, w) for e in rel])

    v0 = np.concatenate([x0[k] for k in free])
    m = 3 * len(rel)
    S = np.zeros((m, len(v0)), dtype=int)
    for n, e in enumerate(rel):
        for k in (e.i, e.j):
            if k in idx:
                S[3 * n:3 * n + 3, 3 * idx[k]:3 * idx[k] + 3] = 1
    from scipy.sparse import csr_matrix
    res = least_squares(fun, v0, jac_sparsity=csr_matrix(S), x_scale="jac", method="trf")
    return {k: res.x[3 * n:3 * n + 3] for k, n in idx.items()}


def centralized(prob):
    """Ground-truth-free baseline; anchor = first pose of robot 0 and all other
    first poses (known initial poses)."""
    anchors = {(r, 0): prob.init[(r, 0)] for r in range(prob.n_robots)}
    free = [k for k in prob.keys() if k not in anchors]
    out = solve(prob, free, anchors, prob.init)
    out.update(anchors)
    return out


def distributed(prob, iters=30, drop_prob=0.0, seed=0, callback=None):
    """Block-coordinate (Gauss-Seidel style) distributed optimization.
    Each robot optimizes only its own poses using the last *received* poses of
    its neighbors on inter-robot edges; then it broadcasts its poses.
    Messages are dropped with probability `drop_prob`."""
    rng = np.random.default_rng(seed)
    R = prob.n_robots
    est = [dict(prob.init) for _ in range(R)]  # est[r]: robot r's view of all poses
    for it in range(iters):
        for r in range(R):  # sequential = Gauss-Seidel
            mine = [k for k in prob.keys(r) if k[1] > 0]
            edges = [e for e in prob.edges if e.i[0] == r or e.j[0] == r]
            new = solve(prob, mine, est[r], est[r], edges=edges)
            est[r].update(new)
            for s in range(R):  # communication channel
                if s != r and rng.random() >= drop_prob:
                    est[s].update(new)
        if callback:
            callback(it, merge(est, prob))
    return merge(est, prob)


def merge(est, prob):
    return {k: est[k[0]][k] for k in prob.gt}


def ate(prob, x):
    """RMS translation error w.r.t. ground truth (frames already aligned)."""
    return float(np.sqrt(np.mean([np.sum((x[k][:2] - prob.gt[k][:2]) ** 2) for k in prob.gt])))
