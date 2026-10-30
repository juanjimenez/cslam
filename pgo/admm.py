"""Consensus ADMM for SE(2) pose graphs with unknown robot frames.

Each robot r optimizes its own poses plus *ghost* copies of the neighbor poses that appear in the
inter-robot edges it owns (edge owner = lower robot id). Ghost copies and the owner's copy of the
same pose are tied by a consensus constraint z; scaled duals u penalize disagreement.
Per iteration (scaled form):
  1. every robot: argmin  f_r(x) + rho/2 * sum ||x_k - z_k + u_k||^2     (a few Gauss-Newton/TRF steps)
  2. ghost holder sends x_k + u_k to the pose owner; owner updates z_k = mean of received copies
  3. owner sends z_k back; duals u_k += x_k - z_k
Messages may be lost; the receiver then keeps using the last value it got.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .core import compose, between, wrap
from .dgs import _odometry_init, ANCHOR


def init_frames(prob, edges=None):
    """Place every robot's odometry chain in the anchor frame using one inter-robot edge
    per robot (breadth-first from robot 0). Needs only a rendezvous measurement."""
    edges = prob.edges if edges is None else edges
    odo = _odometry_init(prob)
    T = {0: np.zeros(3)}
    zero = np.zeros(3)
    changed = True
    while changed and len(T) < prob.n_robots:
        changed = False
        for e in edges:
            a, b = e.i[0], e.j[0]
            if a == b:
                continue
            if a in T and b not in T:
                xj = compose(compose(T[a], odo[e.i]), e.z)
                T[b] = compose(xj, between(odo[e.j], zero))
                changed = True
            elif b in T and a not in T:
                xi = compose(compose(T[b], odo[e.j]), between(e.z, zero))
                T[a] = compose(xi, between(odo[e.i], zero))
                changed = True
    return {k: compose(T.get(k[0], zero), p) for k, p in odo.items()}


def _circ_mean_around(ref, vals):
    return ref + np.mean([wrap(v - ref) for v in vals])


class _Agent:
    def __init__(self, prob, r, edges, x0, sig):
        self.r = r
        self.sig = np.array([1 / sig[0], 1 / sig[0], 1 / sig[1]])
        self.edges = [e for e in edges if e.i[0] == r and e.j[0] == r or
                      (e.i[0] != e.j[0] and min(e.i[0], e.j[0]) == r)]
        own = [k for k in prob.keys(r) if k != ANCHOR]
        self.fixed = {ANCHOR: x0[ANCHOR]} if r == ANCHOR[0] else {}
        self.ghosts = sorted({k for e in self.edges for k in (e.i, e.j) if k[0] != r})
        self.keys = own + self.ghosts
        self.idx = {k: n for n, k in enumerate(self.keys)}
        self.x = np.concatenate([x0[k] for k in self.keys])
        self.cons = []  # keys with a consensus term (own shared poses + ghosts)
        self.z = {}
        self.u = {}

    def set_consensus(self, keys, z0):
        self.cons = keys
        self.z = {k: z0[k].copy() for k in keys}
        self.u = {k: np.zeros(3) for k in keys}
        m = 3 * len(self.edges) + 3 * len(keys)
        S = lil_matrix((m, len(self.x)), dtype=int)
        for n, e in enumerate(self.edges):
            for k in (e.i, e.j):
                if k in self.idx:
                    S[3 * n:3 * n + 3, 3 * self.idx[k]:3 * self.idx[k] + 3] = 1
        for n, k in enumerate(keys):
            r0 = 3 * len(self.edges) + 3 * n
            S[r0:r0 + 3, 3 * self.idx[k]:3 * self.idx[k] + 3] = 1
        self.S = S.tocsr()

    def pose(self, v, k):
        return v[3 * self.idx[k]:3 * self.idx[k] + 3] if k in self.idx else self.fixed[k]

    def solve(self, rho, max_nfev):
        sr = np.sqrt(rho)

        def fun(v):
            out = [between(e.z, between(self.pose(v, e.i), self.pose(v, e.j))) * self.sig
                   for e in self.edges]
            for k in self.cons:
                d = self.pose(v, k) - self.z[k] + self.u[k]
                d[2] = wrap(d[2])
                out.append(sr * d)
            return np.concatenate(out)

        self.x = least_squares(fun, self.x, jac_sparsity=self.S, method="trf",
                               x_scale="jac", max_nfev=max_nfev).x

    def get(self, k):
        return self.x[3 * self.idx[k]:3 * self.idx[k] + 3]


def admm(prob, edges=None, iters=100, rho=1.0, drop_prob=0.0, seed=0, max_nfev=3,
         sigma=(0.05, 0.02), callback=None, x0=None):
    rng = np.random.default_rng(seed)
    edges = prob.edges if edges is None else edges
    x0 = init_frames(prob, edges) if x0 is None else x0
    R = prob.n_robots
    ag = [_Agent(prob, r, edges, x0, sigma) for r in range(R)]
    # shared poses: key -> (owner, [holders])
    holders = {}
    for a in ag:
        for k in a.ghosts:
            holders.setdefault(k, []).append(a.r)
    for a in ag:
        mine = [k for k in holders if k[0] == a.r and k in a.idx]
        a.set_consensus(mine + a.ghosts, x0)
    # what the owner last received from each holder (value x+u), and what holders last got (z)
    recv = {(k, h): x0[k].copy() for k, hs in holders.items() for h in hs}

    def merge():
        return {k: ag[k[0]].get(k) if k != ANCHOR else x0[ANCHOR] for k in prob.keys()}

    for it in range(iters):
        for a in ag:
            a.solve(rho, max_nfev)
        # holders -> owners
        for k, hs in holders.items():
            for h in hs:
                if rng.random() >= drop_prob:
                    recv[(k, h)] = ag[h].get(k) + ag[h].u[k]
        # owners update consensus and send z back
        for k, hs in holders.items():
            o = ag[k[0]]
            if k not in o.idx:  # anchor lives fixed at robot 0
                continue
            vals = [o.get(k) + o.u[k]] + [recv[(k, h)] for h in hs]
            ref = o.z[k]
            zk = np.array([np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]),
                           _circ_mean_around(ref[2], [v[2] for v in vals])])
            o.z[k] = zk
            for h in hs:
                if rng.random() >= drop_prob:
                    ag[h].z[k] = zk.copy()
        # dual updates
        for a in ag:
            for k in a.cons:
                d = a.get(k) - a.z[k]
                d[2] = wrap(d[2])
                a.u[k] = a.u[k] + d
        if callback:
            callback(it, merge())
    out = merge()
    return {k: np.array([p[0], p[1], wrap(p[2])]) for k, p in out.items()}
