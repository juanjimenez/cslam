import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import replace
from pgo.core import Edge, wrap
from pgo.dgs import dgs, gtsam_solve, ate, rot_err, _odometry_init
from pgo.robust import pcm_filter
from sim.run import simulate

prob, segs, scans = simulate(n_robots=3, steps=150, seed=3, verbose=False)
odo = _odometry_init(prob)
rng = np.random.default_rng(0)
inter_idx = [n for n, e in enumerate(prob.edges) if e.i[0] != e.j[0]]
print("clean inter-robot edges:", len(inter_idx))

for frac in [0.0, 0.1, 0.3, 0.5]:
    edges = list(prob.edges)
    bad = set(rng.choice(inter_idx, int(frac * len(inter_idx)), replace=False)) if frac else set()
    for n in bad:
        e = edges[n]
        edges[n] = Edge(e.i, e.j, np.array([*rng.uniform(-4, 4, 2), rng.uniform(-np.pi, np.pi)]))
    p = replace(prob, edges=edges)
    q = replace(prob, edges=pcm_filter(p, odo))
    kept_inter = {id(e) for e in q.edges if e.i[0] != e.j[0]}
    rejected_good = sum(1 for n in inter_idx if n not in bad and id(edges[n]) not in kept_inter)
    accepted_bad = sum(1 for n in bad if id(edges[n]) in kept_inter)
    print(f"\n{int(frac*100)}% outliers ({len(bad)}): PCM kept {len(kept_inter)}, "
          f"accepted outliers {accepted_bad}, rejected inliers {rejected_good}")
    for name, pr in [("no filter", p), ("PCM", q)]:
        g = gtsam_solve(pr, dgs(pr, blocks=[[0, 1, 2]], iters=(1, 1)))
        d = dgs(pr, iters=(1000, 1000), omega=(1.9, 1.9))
        print(f"  {name:10s} GTSAM ATE={ate(prob, g):8.3f}   DGS ATE={ate(prob, d):8.3f}")
