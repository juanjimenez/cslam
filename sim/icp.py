"""Point-to-point 2D ICP with trimming."""
import numpy as np
from scipy.spatial import cKDTree
from pgo.core import wrap


def _tf(pts, T):
    c, s = np.cos(T[2]), np.sin(T[2])
    return pts @ np.array([[c, s], [-s, c]]) + T[:2]


def icp(A, B, guess, iters=40, trim=0.8, max_dist=1.0):
    """Estimate T=(x,y,th) such that T*B aligns with A (T = pose of B's sensor in A's frame).
    Returns T, rms residual of the inliers, inlier fraction of B."""
    tree = cKDTree(A)
    T = np.array(guess, float)
    rms, frac = np.inf, 0.0
    for _ in range(iters):
        d, j = tree.query(_tf(B, T))
        keep = d < max_dist
        if keep.sum() < 10:
            return T, np.inf, 0.0
        order = np.argsort(np.where(keep, d, np.inf))[: int(trim * keep.sum())]
        P, Q = B[order], A[j[order]]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        R = Vt.T @ np.diag([1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        t = qc - R @ pc
        Tn = np.array([t[0], t[1], wrap(np.arctan2(R[1, 0], R[0, 0]))])
        done = np.linalg.norm(Tn - T) < 1e-5
        T = Tn
        rms = float(np.sqrt(np.mean(d[order] ** 2)))
        frac = len(order) / len(B)
        if done:
            break
    return T, rms, frac
