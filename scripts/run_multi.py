import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool
from dataclasses import replace
from pgo.dgs import dgs, gtsam_solve, ate, _odometry_init
from pgo.admm import admm, init_frames
from pgo.robust import pcm_filter
from sim.run import simulate

ITERS = 40


def connected(prob):
    seen, todo = {0}, [0]
    while todo:
        a = todo.pop()
        for e in prob.edges:
            for u, v in ((e.i[0], e.j[0]), (e.j[0], e.i[0])):
                if u == a and v not in seen:
                    seen.add(v); todo.append(v)
    return len(seen) == prob.n_robots


def run(args):
    n, seed = args
    prob, _, _ = simulate(n_robots=n, steps=150, seed=seed, verbose=False)
    q = replace(prob, edges=pcm_filter(prob, _odometry_init(prob)))
    n_inter = sum(e.i[0] != e.j[0] for e in q.edges)
    if not connected(q):
        return dict(n=n, seed=seed, ok=False, inter=n_inter)
    x0 = init_frames(q, q.edges)
    res = dict(n=n, seed=seed, ok=True, inter=n_inter, init=ate(prob, x0),
               gtsam=ate(prob, gtsam_solve(q, x0)))
    for dp in (0.0, 0.3):
        res[f"admm{dp}"] = ate(prob, admm(q, iters=ITERS, drop_prob=dp))
    res["dgs"] = ate(prob, dgs(q, iters=(1000, 1000), omega=(1.9, 1.9)))
    res["dgs_loss"] = ate(prob, dgs(q, iters=(1000, 1000), omega=(1.9, 1.9), drop_prob=0.3))
    return res


if __name__ == "__main__":
    jobs = [(n, s) for n in (3, 5) for s in range(8)]
    t = time.time()
    with Pool() as p:
        for r in p.imap(run, jobs):
            if r["ok"]:
                print(f"N={r['n']} seed={r['seed']} inter={r['inter']:3d} init={r['init']:.2f} "
                      f"gtsam={r['gtsam']:.2f} admm={r['admm0.0']:.2f} admm30%={r['admm0.3']:.2f} "
                      f"dgs={r['dgs']:.3g} dgs30%={r['dgs_loss']:.3g}", flush=True)
            else:
                print(f"N={r['n']} seed={r['seed']} inter={r['inter']:3d} DISCONNECTED", flush=True)
    print(f"[{time.time()-t:.0f}s]")
