#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The two primitives, their crossover, and why a build rule exists.

Why this file exists
--------------------
`masked_pipeline.py` reports a rule rather than a per-cell choice: compact with
`np.compress(..., out=)` above 16,384 elements, with a boolean index below, and
gather with fancy indexing at every size.  A rule needs a measurement behind it,
and this is that measurement -- the only file here that times the primitives
rather than the pipeline.

The two facts it establishes
----------------------------
  gather      `src[idx]` beats `np.take(src, idx, out=)` at every k measured,
              because numpy routes `take` with an output array through the
              buffered generic loop while fancy indexing has a specialised one.
              The gap widens with k: 1.4x at k = 80, 2.6x at k = 75,556.
  compact     boolean indexing `src[m]` loses above n ~ 1.6e4, because it
              materialises an `intp` array of length n to hold selected
              positions (3.3 MB at n = 412,875) while `np.compress(out=)` does
              not.  Below that size the call overhead of `compress` dominates
              and boolean indexing wins.

Consequence, and why it is not a footnote: path R compacts n, path B compacts k.
One implementation choice therefore moves the two arms in opposite directions,
and a ratio quoted without saying which build produced it is measuring the build.

  python prim_cost.py
"""
import json
import os
import time

import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# The scope lengths and band densities the paper reports, so the sizes here are
# the sizes the pipeline meets rather than round numbers.
NS = [197, 6969, 44606, 412875]
DS = [0.042, 0.183, 0.401]
# A dense ladder for the crossover fit, geometric from 1e3 to 5e5.
LADDER = [int(round(10 ** e)) for e in np.linspace(3, 5.7, 14)]
KS = [80, 1275, 8163, 75556, 165563]
REPS = 25
SEED = 7


def bench(f, reps=REPS):
    for _ in range(5):
        f()
    v = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        v.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(v))


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="where to write the record; reproduce.py passes a "
                         "scratch path so a gate run cannot replace the curated "
                         "prim_cost.json")
    return ap.parse_args()


def main():
    ARGS = _args()
    rng = np.random.default_rng(SEED)
    # The array the pipeline actually gathers over.  This matters: a scattered
    # gather's cost depends on how much of the source array the accesses touch,
    # so measuring on a 5e5 array and quoting the result for a 1.4e6 store would
    # be the store-size confound in miniature.  masked_pipeline's twelve-code
    # store is 1,393,941 lines; match it.
    n_big = 1_400_000
    k_big = max(KS)
    # the values a cap rule is judged against: small integers, int16
    src16 = rng.integers(1, 21, n_big).astype(np.int16)
    src32 = rng.integers(0, 1 << 20, n_big).astype(np.int32)
    # Output buffers are sized per measurement, not shared: a shared buffer
    # carries its cache state from the previous size, and that alone moved the
    # largest gather between 320 and 543 us across runs.

    # ---- gathers: scattered identifiers over the whole array
    gathers = []
    for k in KS:
        idx = rng.choice(n_big, k, replace=False).astype(np.int32)
        o16 = np.empty(k, dtype=np.int16)
        t_fancy = bench(lambda: src16[idx])
        t_take = bench(lambda: np.take(src16, idx, out=o16))
        gathers.append(dict(k=k, fancy_us=t_fancy, take_us=t_take,
                            ratio=t_take / t_fancy))
        print("gather k %7d | fancy %9.1f us | take(out=) %9.1f us | take/fancy "
              "%.2f" % (k, t_fancy, t_take, t_take / t_fancy), flush=True)

    # ---- compactions: a boolean mask over n, keeping ~k of the positions
    compacts = []
    for n in LADDER:
        for d in (0.183,):
            m = rng.random(n) < d
            cnt = int(m.sum())
            o32 = np.empty(cnt, dtype=np.int32)
            t_bool = bench(lambda: src32[:n][m])
            t_comp = bench(lambda: np.compress(m, src32[:n], out=o32))
            compacts.append(dict(n=n, kept=cnt, bool_us=t_bool, compress_us=t_comp,
                                 ratio=t_comp / t_bool))
            print("compact n %7d (kept %6d) | bool %9.1f us | compress %9.1f us "
                  "| compress/bool %.2f"
                  % (n, cnt, t_bool, t_comp, t_comp / t_bool), flush=True)

    # ---- the crossover: the largest n where a boolean index still wins
    RULE_RANGE = 320_000
    band = [c for c in compacts if c["n"] <= RULE_RANGE]
    wins = [c["n"] for c in band if c["ratio"] > 1.0]
    loses = [c["n"] for c in band if c["ratio"] <= 1.0]
    cross = (max(wins), min(loses)) if wins and loses else (None, None)
    above = [c for c in compacts if c["n"] > RULE_RANGE]
    print("\ncrossover: boolean index wins up to n = %s, compress wins from "
          "n = %s" % cross)
    # No cell in the paper falls inside the bracket, so the rule's constant is
    # free within it; 2**14 is chosen conservatively and stated as such.
    inside = [n for n in NS if cross[0] < n < cross[1]]
    print("  scopes inside the bracket: %s" % (inside or "none"))
    print("  above %d both are memory-bound and compress still wins:" % RULE_RANGE)
    for c in above:
        print("    n %7d  compress/boolean %.2f" % (c["n"], c["ratio"]))

    # ---- the same cell the paper quotes, both ways, so the reader can check
    n0, k0 = 44606, int(round(44606 * 0.183))
    m = rng.random(n0) < (k0 / n0)
    kept = int(m.sum())
    cand = np.flatnonzero(m).astype(np.int32)
    outA = np.empty(kept, dtype=np.int32)
    both_alloc = bench(lambda: src32[:n0][m])
    both_comp = bench(lambda: np.compress(m, src32[:n0], out=outA))
    o16 = np.empty(len(cand), dtype=np.int16)
    g_fancy = bench(lambda: src16[cand])
    g_take = bench(lambda: np.take(src16, cand, out=o16))
    print("\nprimitive-level check at the paper's (n, k) = (%d, %d), on a "
          "synthetic array:" % (n0, k0))
    print("  compaction over n   boolean %8.1f us   compress %8.1f us"
          % (both_alloc, both_comp))
    print("  gather of k         fancy   %8.1f us   take     %8.1f us"
          % (g_fancy, g_take))
    print("  so B/R = %.3f built throughout, %.3f built per path"
          % (g_fancy / both_alloc, g_fancy / both_comp))
    print("  NOTE: these are primitives on a synthetic array with one code, not")
    print("  the pipeline cell of the paper (0.36 / 0.63).  The gap is the store")
    print("  itself -- twelve interleaved codes make both arms' accesses worse --")
    print("  which is why masked_pipeline.py is the file that reports the cell.")

    res = dict(reps=REPS, seed=SEED, gathers=gathers, compacts=compacts,
               crossover=dict(boolean_wins_up_to=cross[0],
                              compress_wins_from=cross[1],
                              range_used=RULE_RANGE,
                              above_range=[(c["n"], c["ratio"]) for c in above]),
               rule="compress above 2**14 elements, boolean index below, fancy "
                    "indexing for every gather",
               rule_note="the measured crossover is between %s and %s; no scope "
                         "in the paper falls inside that bracket, so the constant "
                         "is free within it" % cross,
               scopes_inside_bracket=inside,
               label="primitive-level, synthetic one-code array; NOT the "
                     "pipeline cell of Sect. V-A",
               primitive_check=dict(n=n0, k=k0,
                               R_bool_us=both_alloc, R_compress_us=both_comp,
                               B_fancy_us=g_fancy, B_take_us=g_take,
                               ratio_built_throughout=g_fancy / both_alloc,
                               ratio_built_per_path=g_fancy / both_comp))
    os.makedirs(OUT, exist_ok=True)
    p = ARGS.out if ARGS.out else os.path.join(OUT, "prim_cost.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
