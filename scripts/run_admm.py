import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import replace
from pgo.dgs import dgs, gtsam_solve, ate, rot_err
from pgo.admm import admm, init_frames
from pgo.robust import pcm_filter
from pgo.dgs import _odometry_init
from sim.run import simulate

prob, _, _ = simulate(n_robots=3, steps=150, seed=3, verbose=False)
q = replace(prob, edges=pcm_filter(prob, _odometry_init(prob)))
x0 = init_frames(q, q.edges)
print(f"frame-init only        ATE={ate(prob, x0):.3f}")
g = gtsam_solve(q, x0)
print(f"GTSAM (centralized)    ATE={ate(prob, g):.3f}")
for dp in [0.0, 0.1, 0.3, 0.5]:
    t = time.time()
    hist = []
    a = admm(q, iters=60, rho=1.0, drop_prob=dp, callback=lambda i, x: hist.append(ate(prob, x)))
    d = dgs(q, iters=(1000, 1000), omega=(1.9, 1.9), drop_prob=dp)
    print(f"drop={dp:.1f}  ADMM ATE={ate(prob, a):.3f} (it10={hist[9]:.3f}, it30={hist[29]:.3f})   "
          f"DGS(w=1.9) ATE={ate(prob, d):.3g}   [{time.time()-t:.0f}s]", flush=True)
