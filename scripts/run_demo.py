import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pgo.core import make_problem, centralized, distributed, ate

prob = make_problem(n_robots=3, steps=80, loop_range=5.0, sigma_t=0.05, sigma_r=0.02, seed=1)
print("edges:", len(prob.edges), "| inter-robot:", sum(e.i[0] != e.j[0] for e in prob.edges))
print("ATE odometry     :", round(ate(prob, prob.init), 4))
cen = centralized(prob)
print("ATE centralized  :", round(ate(prob, cen), 4))
hist = []
dist = distributed(prob, iters=15, callback=lambda i, x: hist.append(ate(prob, x)))
print("ATE distributed  :", round(ate(prob, dist), 4))
print("dist vs central  :", round(float(np.sqrt(np.mean([np.sum((dist[k][:2]-cen[k][:2])**2) for k in cen]))), 5))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (name, x) in zip(ax[:2], [("odometry", prob.init), ("distributed", dist)]):
    for r in range(prob.n_robots):
        ks = prob.keys(r)
        a.plot(*np.array([prob.gt[k][:2] for k in ks]).T, "k--", lw=.8)
        a.plot(*np.array([x[k][:2] for k in ks]).T, lw=1.5)
    a.set_title(name); a.set_aspect("equal")
ax[2].semilogy(hist); ax[2].set_title("ATE per iteration"); ax[2].set_xlabel("iteration")
plt.savefig("/home/juan/casl/demo.png", dpi=110)
