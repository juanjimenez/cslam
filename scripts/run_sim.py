import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim.run import simulate
from pgo.dgs import dgs, gtsam_solve, ate, rot_err, to_global_frame, _odometry_init

prob, segs, scans = simulate(n_robots=3, steps=150, seed=3)
print("inter-robot edges:", sum(e.i[0] != e.j[0] for e in prob.edges))
odo = _odometry_init(prob)
cen = gtsam_solve(prob, dgs(prob, blocks=[[0, 1, 2]], iters=(1, 1)))
d = dgs(prob, iters=(1000, 1000), omega=(1.9, 1.9))
d_loss = dgs(prob, iters=(1000, 1000), omega=(1.9, 1.9), drop_prob=0.3)
two = dgs(prob, blocks=[[0, 1, 2]], iters=(1, 1))
for n, x in [("odometry (own frames)", odo), ("GTSAM", cen), ("two-stage centralized", two), ("DGS", d), ("DGS 30% msg loss", d_loss)]:
    print(f"{n:24s} ATE={ate(prob, x):.4f} rot={rot_err(prob, x):.4f}")

gt = to_global_frame(prob)
a0 = prob.gt[(0, 0)]
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
for a, (name, x) in zip(ax, [("world + ground truth", None), ("DGS estimate (anchor frame)", d)]):
    if x is None:
        for s in segs:
            a.plot([s[0], s[2]], [s[1], s[3]], "k-", lw=1)
        for r in range(prob.n_robots):
            a.plot(*np.array([prob.gt[k][:2] for k in prob.keys(r)]).T, lw=1.5)
    else:
        for r in range(prob.n_robots):
            a.plot(*np.array([gt[k][:2] for k in prob.keys(r)]).T, "k--", lw=.8)
            a.plot(*np.array([x[k][:2] for k in prob.keys(r)]).T, lw=1.5)
    a.set_title(name); a.set_aspect("equal")
plt.savefig("/home/juan/casl/demo_sim.png", dpi=100)
