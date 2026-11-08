#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the boundary move with the table?  -- corrected instrument.

The paper's corpus point is one instance: |D| = 4.19 M rows, scope 65 Ki, band
182.  A reviewer is entitled to ask whether kappa and n0 are properties of the
layout or of that one instance.  This holds the semantics fixed and grows the
table 16x, refitting both paths at each size.

    N in {2^20, 2^22, 2^24}, same predicate, same candidate count, same scope
    path R: the scope's n rows, sequential
    path B: k candidates scattered over the whole table

If kappa is the ratio of two per-element times it should not care how many rows
there are; if the band's per-candidate cost is a cache property it will fall as
the table outgrows the last-level cache.  Both outcomes are informative and this
script reports whichever it gets.

Two defects of the previous version of this file, both fixed here; the numbers
it produced (kappa 0.3218 / 0.3357 / 0.3155) are withdrawn:

  predicate  `~((m == 1) & (OMEGA == 1))` is semantically identical to
             `~(m == 1)` at OMEGA = 1 and costs 2.78 ns/row more.  On a scan
             that costs 0.26 ns/row that is an 11x inflation of c_R -- and
             because the excess is a constant, it also flattened c_R across
             |D|, which is the one thing this experiment was supposed to test.
  band       `np.sort(choice(N, 8192))[:k]` returns the k smallest identifiers,
             so the gather touched a cache-resident prefix of the table instead
             of the whole of it, and -- the table being value-sorted -- returned
             zero rows for every k <= 4096: 0% pass rate, i.e. it timed rows a
             band could not return.  Candidates are now drawn from the rows that
             satisfy the threshold and kept unsorted, as in exp/n0_holdout.

With both fixed, kappa = 0.0337 / 0.0305 / 0.0275 (50 processes per cell) and
the direction is the opposite of what the defect reported: the scan is flat at
0.255 ns/row across the 16x, the gather rises 7.59 -> 9.32 ns/candidate, and
kappa inherits that rise.  See README, "Cross-machine and scale".

    python exp/n0_impl/scale.py                  # three sizes, one run each
    python exp/n0_impl/scale.py 16m out.json     # one cell, for scale_reps.py

Output: exp/n0_impl/scale.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scale.json")

SEED = 20261105
PER_GROUP = 65536
THETA = 500000
OMEGA = 1
NS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
POOL = 32768
ROUNDS = 5
BUDGET = 4.0e5
SIZES = {"1m": 1 << 20, "4m": 1 << 22, "16m": 1 << 24}


def time_batch(fn, rounds=ROUNDS, budget=BUDGET):
    t0 = time.perf_counter_ns()
    fn()
    one = max(time.perf_counter_ns() - t0, 1)
    inner = int(max(1, min(2_000_000, budget / one)))
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter_ns() - t0) / inner)
    return float(best)


def build(N):
    """Rows in insertion order; the band is the set that satisfies THETA."""
    rng = np.random.default_rng(SEED)
    v = rng.integers(0, 10 ** 6, N).astype(np.int64)
    mod = (rng.random(N) < 0.6).astype(np.int8)
    band = np.flatnonzero(v >= THETA)
    if band.size < POOL:
        raise RuntimeError("band too small: %d" % band.size)
    cand = rng.choice(band, POOL, replace=False)      # unsorted: no id bias
    return v, mod, cand


def fit(v, mod, cand):
    def path_R(n):
        return int(np.count_nonzero((v[:n] >= THETA) & ~(mod[:n] == 1)))

    def path_B(k):
        c = cand[:k]
        return int(np.count_nonzero((v.take(c) >= THETA) & ~(mod.take(c) == 1)))

    tR = {n: time_batch(lambda n=n: path_R(n)) for n in NS}
    tB = {k: time_batch(lambda k=k: path_B(k)) for k in KS}
    cR = float(np.polyfit(np.array(NS, float), np.array([tR[n] for n in NS]), 1)[0])
    cB = float(np.polyfit(np.array(KS, float), np.array([tB[k] for k in KS]), 1)[0])
    FR, FB = time_batch(lambda: path_R(0)), time_batch(lambda: path_B(0))
    # is the fit even a line?  slope over the two halves of the k grid
    s_lo = float(np.polyfit(np.array(KS[:5], float), np.array([tB[k] for k in KS[:5]]), 1)[0])
    s_hi = float(np.polyfit(np.array(KS[5:], float), np.array([tB[k] for k in KS[5:]]), 1)[0])
    return {
        "c_R": cR, "c_B": cB, "F_R": FR, "F_B": FB,
        "kappa": cR / cB, "n0": (FB - FR) / cR,
        "cB_small_half": s_lo, "cB_large_half": s_hi,
        # the pass rates a band would see: 0 means the arm timed rows no band
        # can return, which is what the defect did
        "pass_R": path_R(PER_GROUP), "pass_B_4096": path_B(4096),
    }


def main():
    # worker mode: one size, one process, for scale_reps.py
    if len(sys.argv) > 1:
        tag = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        N = SIZES[tag]
        r = fit(*build(N))
        r.update({"N": N, "tag": tag})
        if out:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
        print(json.dumps(r))
        return

    rows = []
    for tag, N in SIZES.items():
        r = fit(*build(N))
        rows.append({"N": N, "bytes_per_col": N * 9,
                     "c_R": round(r["c_R"], 3), "c_B": round(r["c_B"], 3),
                     "F_R": round(r["F_R"], 1), "F_B": round(r["F_B"], 1),
                     "kappa": round(r["kappa"], 4), "n0": round(r["n0"], 1),
                     "pass_B_4096": r["pass_B_4096"]})
        print(json.dumps(rows[-1]), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": "numpy backend, scope 65536, same predicate, 16x table, "
                    "corrected instrument (predicate and band construction "
                    "fixed -- see module docstring).  One run per size here; the "
                    "medians the paper quotes are in scale_reps.json, 50 "
                    "processes per cell.",
            "rows": rows}, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
