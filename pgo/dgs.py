"""Distributed Gauss-Seidel (Choudhary et al., ICRA 2017) for SE(2) pose graphs
with unknown relative frames, plus a GTSAM centralized baseline.

Stage 1: rotations via chordal relaxation (linear LS), block Gauss-Seidel, project to SO(2).
Stage 2: translations given rotations (linear LS), block Gauss-Seidel.
Only the poses on inter-robot edges (separators) need to be exchanged.
"""
import numpy as np
import gtsam
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .core import Problem, compose, between, wrap

ANCHOR = (0, 0)  # global frame = first pose of robot 0


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def to_global_frame(prob):
    """Express ground truth in the anchor frame (what the estimators can recover)."""
    a = prob.gt[ANCHOR]
    return {k: between(a, p) for k, p in prob.gt.items()}


def _project(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, np.linalg.det(U @ Vt)]) @ Vt
    return np.arctan2(R[1, 0], R[0, 0])


def _odometry_init(prob):
    """Each robot chains its own odometry from identity (own, unknown frame)."""
    x = {}
    for r in range(prob.n_robots):
        x[(r, 0)] = np.zeros(3)
        for e in (e for e in prob.edges if e.i[0] == r and e.j[0] == r):
            x[e.j] = compose(x[e.i], e.z)
    return x


def _gid(prob):
    return {k: n for n, k in enumerate(prob.keys())}


def _rot_system(prob, gid):
    """Rows of R_j - R_i R_ij = 0 over all poses (4 unknowns per pose, row-major)."""
    rows, cols, vals = [], [], []
    for n, e in enumerate(prob.edges):
        Rij = rot(e.z[2])
        for a in range(2):
            for b in range(2):
                r = 4 * n + 2 * a + b
                rows.append(r); cols.append(4 * gid[e.j] + 2 * a + b); vals.append(1.0)
                for c in range(2):
                    rows.append(r); cols.append(4 * gid[e.i] + 2 * a + c); vals.append(-Rij[c, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(4 * len(prob.edges), 4 * len(gid))), \
        np.zeros(4 * len(prob.edges))


def _trans_system(prob, gid, theta):
    """t_j - t_i = R_i t_ij with rotations fixed."""
    rows, cols, vals, c = [], [], [], np.zeros(2 * len(prob.edges))
    for n, e in enumerate(prob.edges):
        c[2 * n:2 * n + 2] = rot(theta[e.i]) @ e.z[:2]
        for o in range(2):
            rows += [2 * n + o, 2 * n + o]
            cols += [2 * gid[e.j] + o, 2 * gid[e.i] + o]
            vals += [1.0, -1.0]
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * len(prob.edges), 2 * len(gid))), c


class _Stage:
    """Block Gauss-Seidel on A x = c. Each block's normal matrix is factorized once."""

    def __init__(self, prob, gid, A, c, nvar, blocks, fixed_keys):
        self.nvar, self.c = nvar, c
        self.blocks = []
        edge_rows = np.repeat(np.arange(len(prob.edges)), 1)
        for blk in blocks:
            free = [k for k in prob.keys() if k[0] in blk and k not in fixed_keys]
            fc = np.array([nvar * gid[k] + o for k in free for o in range(nvar)], int)
            rows = np.unique(A[:, fc].nonzero()[0])
            Ar = A[rows]
            oc = np.setdiff1d(np.unique(Ar.nonzero()[1]), fc)
            Af = Ar[:, fc].tocsc()
            self.blocks.append((blk, free, fc, oc, rows, Af, Ar[:, oc], spla.splu((Af.T @ Af).tocsc())))

    def sweep(self, x, A, rng, drop_prob, omega=1.0):
        """x: list of per-robot state vectors. One Gauss-Seidel pass over all blocks."""
        R = len(x)
        for blk, free, fc, oc, rows, Af, Ao, lu in self.blocks:
            me = blk[0]
            v = lu.solve(Af.T @ (self.c[rows] - Ao @ x[me][oc]))
            v = x[me][fc] + omega * (v - x[me][fc])  # successive over-relaxation
            for r in blk:
                x[r][fc] = v
            for s in range(R):
                if s not in blk and rng.random() >= drop_prob:
                    x[s][fc] = v


def dgs(prob, blocks=None, iters=(30, 30), drop_prob=0.0, seed=0, callback=None, omega=(1.0, 1.0)):
    """blocks: list of lists of robot ids solved jointly (default: one per robot).
    Each robot keeps its own copy of the poses it has *received*; messages are lost
    with probability drop_prob."""
    rng = np.random.default_rng(seed)
    R = prob.n_robots
    blocks = blocks or [[r] for r in range(R)]
    gid = _gid(prob)
    N = len(gid)
    x0 = _odometry_init(prob)

    # ---- Stage 1: rotations
    A1, c1 = _rot_system(prob, gid)
    xr = np.zeros(4 * N)
    for k, p in x0.items():
        xr[4 * gid[k]:4 * gid[k] + 4] = rot(p[2]).ravel()
    xr[4 * gid[ANCHOR]:4 * gid[ANCHOR] + 4] = np.eye(2).ravel()
    st1 = _Stage(prob, gid, A1, c1, 4, blocks, {ANCHOR})
    views = [xr.copy() for _ in range(R)]
    for _ in range(iters[0]):
        st1.sweep(views, A1, rng, drop_prob, omega[0])
    theta = {}
    for k in prob.keys():
        v = views[k[0]][4 * gid[k]:4 * gid[k] + 4]
        theta[k] = 0.0 if k == ANCHOR else _project(v.reshape(2, 2))

    # ---- Stage 2: translations (every robot uses the final rotations)
    A2, c2 = _trans_system(prob, gid, theta)
    xt = np.zeros(2 * N)
    for k, p in x0.items():
        xt[2 * gid[k]:2 * gid[k] + 2] = p[:2]
    xt[2 * gid[ANCHOR]:2 * gid[ANCHOR] + 2] = 0
    st2 = _Stage(prob, gid, A2, c2, 2, blocks, {ANCHOR})
    views = [xt.copy() for _ in range(R)]
    for _ in range(iters[1]):
        st2.sweep(views, A2, rng, drop_prob, omega[1])
        if callback:
            callback(_assemble(prob, theta, gid, views))
    return _assemble(prob, theta, gid, views)


def _assemble(prob, theta, gid, views):
    return {k: np.array([*views[k[0]][2 * gid[k]:2 * gid[k] + 2], theta[k]]) for k in prob.keys()}


def gtsam_solve(prob, init):
    """Centralized baseline: LM on the full graph, prior on the anchor."""
    g = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.02]))
    g.add(gtsam.PriorFactorPose2(_s(ANCHOR), gtsam.Pose2(0, 0, 0),
                                 gtsam.noiseModel.Diagonal.Sigmas(np.full(3, 1e-6))))
    for e in prob.edges:
        g.add(gtsam.BetweenFactorPose2(_s(e.i), _s(e.j), gtsam.Pose2(*e.z), noise))
    v = gtsam.Values()
    for k, p in init.items():
        v.insert(_s(k), gtsam.Pose2(*p))
    res = gtsam.LevenbergMarquardtOptimizer(g, v).optimize()
    return {k: np.array([res.atPose2(_s(k)).x(), res.atPose2(_s(k)).y(),
                         res.atPose2(_s(k)).theta()]) for k in init}


def _s(k):
    return gtsam.symbol(chr(ord("a") + k[0]), k[1])


def ate(prob, x):
    gt = to_global_frame(prob)
    return float(np.sqrt(np.mean([np.sum((x[k][:2] - gt[k][:2]) ** 2) for k in gt])))


def rot_err(prob, x):
    gt = to_global_frame(prob)
    return float(np.sqrt(np.mean([wrap(x[k][2] - gt[k][2]) ** 2 for k in gt])))
