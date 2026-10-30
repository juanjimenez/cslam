"""Outlier rejection for inter-robot loop closures (Pairwise Consistency Maximization style).

Two loop closures between the same robot pair, (ia->ib, z1) and (ja->jb, z2), are consistent
if going ia -> ja (robot a's odometry), ja -> jb (z2), jb -> ib (robot b's odometry) lands on z1.
Only each robot's own odometry plus the exchanged measurements are needed.
The largest set of mutually consistent closures per robot pair is kept.
"""
import numpy as np
from pgo.core import compose, between, wrap


def _consistency_error(z1, z2, oa_i, oa_j, ob_i, ob_j):
    chain = compose(compose(between(oa_i, oa_j), z2), between(ob_j, ob_i))
    d = between(z1, chain)
    return np.linalg.norm(d[:2]), abs(wrap(d[2]))


def _max_clique(adj):
    best = []

    def bk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = R
            return
        if len(R) + len(P) <= len(best):
            return
        u = max(P | X, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[u]):
            bk(R + [v], P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    bk([], set(range(len(adj))), set())
    return best


def pcm_filter(prob, odo, tol_t=0.25, tol_r=0.08, drift_t=0.01, drift_r=0.004):
    """Returns a new edge list without the inter-robot edges rejected by PCM.
    Tolerance grows with the odometry distance (in steps) spanned by the pair."""
    inter = [e for e in prob.edges if e.i[0] != e.j[0]]
    intra = [e for e in prob.edges if e.i[0] == e.j[0]]
    kept = []
    pairs = sorted({(e.i[0], e.j[0]) for e in inter})
    for a, b in pairs:
        es = [e for e in inter if (e.i[0], e.j[0]) == (a, b)]
        n = len(es)
        adj = [set() for _ in range(n)]
        for p in range(n):
            for q in range(p + 1, n):
                e1, e2 = es[p], es[q]
                et, er = _consistency_error(e1.z, e2.z, odo[e1.i], odo[e2.i], odo[e1.j], odo[e2.j])
                span = abs(e1.i[1] - e2.i[1]) + abs(e1.j[1] - e2.j[1])
                if et < tol_t + drift_t * span and er < tol_r + drift_r * span:
                    adj[p].add(q)
                    adj[q].add(p)
        kept += [es[i] for i in _max_clique(adj)] if n else []
    return intra + kept
