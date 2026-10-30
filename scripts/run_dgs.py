import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pgo.core import make_problem
from pgo.dgs import dgs, gtsam_solve, ate, rot_err, to_global_frame, _odometry_init

# Robots start at different, unknown poses; frames are only related via inter-robot edges.
prob = make_problem(n_robots=3, steps=80, loop_range=5.0, sigma_t=0.05, sigma_r=0.02, seed=1)
print("inter-robot edges:", sum(e.i[0] != e.j[0] for e in prob.edges))

odo = _odometry_init(prob)
cen2 = dgs(prob, blocks=[[0, 1, 2]], iters=(1, 1))      # all robots in one block = centralized two-stage
gt_sam = gtsam_solve(prob, cen2)
d = dgs(prob, iters=(120, 120))
d_loss = dgs(prob, iters=(120, 120), drop_prob=0.3)
d_ref = gtsam_solve(prob, d)  # DGS output as initial guess for a final refinement

def row(n, x): print(f"{n:28s} ATE={ate(prob, x):.4f}  rot={rot_err(prob, x):.4f}")
row("odometry (own frames)", odo)
row("centralized two-stage", cen2)
row("GTSAM (LM)", gt_sam)
row("DGS distributed", d)
row("DGS, 30% msg loss", d_loss)
row("DGS + GTSAM refinement", d_ref)
print("DGS vs GTSAM (rms xy):",
      round(float(np.sqrt(np.mean([np.sum((d[k][:2]-gt_sam[k][:2])**2) for k in d]))), 4))

hist = []
dgs(prob, iters=(120, 120), callback=lambda x: hist.append(ate(prob, x)))
gt = to_global_frame(prob)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (name, x) in zip(ax[:2], [("GTSAM", gt_sam), ("DGS", d)]):
    for r in range(prob.n_robots):
        ks = prob.keys(r)
        a.plot(*np.array([gt[k][:2] for k in ks]).T, "k--", lw=.8)
        a.plot(*np.array([x[k][:2] for k in ks]).T, lw=1.5)
    a.set_title(name); a.set_aspect("equal")
ax[2].semilogy(hist); ax[2].set_title("DGS: ATE vs translation iteration")
plt.savefig("/home/juan/casl/demo_dgs.png", dpi=110)
