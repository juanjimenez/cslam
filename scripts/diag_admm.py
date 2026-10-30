import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool
from dataclasses import replace
from pgo.dgs import gtsam_solve, ate, _odometry_init
from pgo.admm import admm, init_frames, init_frames_ls
from pgo.robust import pcm_filter
from sim.run import simulate

def job(a):
    n, seed, rho, init = a
    prob,_,_ = simulate(n_robots=n, steps=150, seed=seed, verbose=False)
    q = replace(prob, edges=pcm_filter(prob, _odometry_init(prob)))
    x0 = (init_frames_ls if init == "ls" else init_frames)(q, q.edges)
    h=[]
    admm(q, iters=100, rho=rho, x0=x0, callback=lambda i,x:h.append(ate(prob,x)))
    return n, seed, init, rho, ate(prob,x0), h[19], h[59], h[-1], ate(prob, gtsam_solve(q,x0))
if __name__=="__main__":
    jobs=[(n,s,i,r) for n,s in [(3,2),(3,0),(5,0),(5,6)] for i in ("one","ls") for r in (0.03,0.1,0.3)]
    with Pool(4) as p:
        for r in p.imap(job,jobs):
            print("N=%d s=%d init=%s rho=%g | init=%.2f it20=%.2f it60=%.2f it100=%.2f | gtsam=%.2f"%r, flush=True)
