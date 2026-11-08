"""One repetition of the 16.7M cell, protocol identical to scale.py.

scale.py fits all three sizes in one process, which means the third size is
measured once per invocation and the whole sweep is one data point per size.
That is enough to see whether kappa moves with |D|; it is not enough to say
what kappa *is* at 16.7M, because the gather at that size sits closest to the
memory hierarchy's edge and one run is one sample.

This script is the same fit for that one size, so it can be run in a fresh
process as often as needed; scale_16m_reps.py runs it 25 times and reports the
span.  Same seed, same grids, same order (all path-R points, then all path-B
points), min of five rounds per point -- changing any of that would measure a
different thing.

    python exp/n0_impl/scale_16m.py out.json
"""
import json
import sys
import time

import numpy as np

SEED = 20261105
THETA = 500000
OMEGA = 1
NS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
ROUNDS = 5
BUDGET = 4.0e5
N = 1 << 24
N_CORPUS, K_CORPUS = 394.0, 182.0


def time_batch(fn, rounds=ROUNDS, budget=BUDGET):
    t0 = time.perf_counter_ns()
    fn()
    one = max(time.perf_counter_ns() - t0, 1)
    inner = int(max(1, min(2_000_000, budget / one)))
    best, acc = float("inf"), 0
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            acc += fn()
        best = min(best, (time.perf_counter_ns() - t0) / inner)
    return float(best)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scale_16m_run.json"
    t_start = time.time()

    rng = np.random.default_rng(SEED)
    v = rng.integers(0, 10 ** 6, N).astype(np.int64)
    v.sort()
    mod = (rng.random(N) < 0.6).astype(np.int8)
    cand = np.sort(np.random.default_rng(7).choice(N, 8192, replace=False))

    def path_R(n, v=v, mod=mod):
        return int(np.count_nonzero((v[:n] >= THETA) & ~((mod[:n] == 1) & (OMEGA == 1))))

    def path_B(k, v=v, mod=mod, cand=cand):
        c = cand[:k]
        vv = v.take(c)
        mm = mod.take(c)
        return int(np.count_nonzero((vv >= THETA) & ~((mm == 1) & (OMEGA == 1))))

    tR = {n: time_batch(lambda n=n: path_R(n)) for n in NS}
    tB = {k: time_batch(lambda k=k: path_B(k)) for k in KS}
    cR, _ = np.polyfit(np.array(NS, float), np.array([tR[n] for n in NS]), 1)
    cB, _ = np.polyfit(np.array(KS, float), np.array([tB[k] for k in KS]), 1)
    FR = time_batch(lambda: path_R(0))
    FB = time_batch(lambda: path_B(0))

    d = {
        "N": N,
        "c_R": float(cR), "c_B": float(cB),
        "F_R": float(FR), "F_B": float(FB),
        "kappa": float(cR / cB),
        "n0": float((FB - FR) / cR),
        "allow": float((cR / cB) * N_CORPUS - K_CORPUS),
        # slope over the small and the large half of the k grid: if these two
        # diverge the fit is not a line and the single slope is a summary
        "cB_small_half": float(np.polyfit(np.array(KS[:5], float),
                                          np.array([tB[k] for k in KS[:5]]), 1)[0]),
        "cB_large_half": float(np.polyfit(np.array(KS[5:], float),
                                          np.array([tB[k] for k in KS[5:]]), 1)[0]),
        "tR_ns": {str(n): tR[n] for n in NS},
        "tB_ns": {str(k): tB[k] for k in KS},
        "wall_seconds": round(time.time() - t_start, 3),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(d, fh)


if __name__ == "__main__":
    main()
