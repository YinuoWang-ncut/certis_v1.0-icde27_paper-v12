"""The build triple: one cell, two instruments, three implementations per stage.

Why this file exists
--------------------
Sect. V-A says the same cell reads 0.36x built one way, 0.63x built per path and
0.90x through the pipeline, and that a table reporting one of these without
naming the build has measured the build.  Those three numbers come from *two
instruments*, and until this file existed no script printed any of them:
`masked_pipeline.py` reports the pipeline, `prim_cost.py` reports the two
primitives on a synthetic one-code array, and the bare-kernel pair on the real
store -- which is what the 0.36 and the 0.63 are -- had no producer at all.

So this file is the missing instrument.  It builds the same store as
`masked_pipeline.py`, takes path R and path B *without* the mask, the probe and
the overlay (a bare kernel, so it is comparable to `prim_cost.py` and not to the
pipeline), and measures both under both compaction styles and both gather
styles:

    path R   scan the scope, two-state test, compact        n elements
    path B   gather k candidates, two-state test, compact    k elements

The point it makes is structural: path R compacts n and path B compacts k, so
the two stages do not share the winner.  Boolean indexing allocates an intp
array as long as its input -- 3.3 MB at the largest scope -- and `compress` into
a caller buffer does not, so the compaction that is right for the scope is wrong
for the band.  A table that picks one style for both has measured its own
implementation, by a factor that exceeds the effect it reports.

What is *not* here
------------------
The mask, the probe and the overlay.  Those live in `masked_pipeline.py`, and
its ratio for this cell is reported beside this one by `reproduce.py` so the two
never get quoted as if they were the same measurement.

Usage
-----
    python exp/build_triple.py              # both scopes, 25 reps
    python exp/build_triple.py --reps 41
    python exp/build_triple.py --env        # also print the drift guard

Output: exp/out/build_triple.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import masked_pipeline as MP  # noqa: E402

OUT = os.path.join(HERE, "out")

# The same store `masked_pipeline.py` builds, cell for cell.  This matters more
# than it looks: a gather's cost depends on how large the array it scatters into
# is, so measuring on a two-code store instead of this one changes which gather
# wins -- `take` beats fancy indexing on a 46k-element source and loses on a
# 1.4M-element one.  A build triple taken on a different store than the
# pipeline's is a triple about a different machine's memory, not about this one.
PLANS = [(tag, n, d) for tag, n in MP.NS for d in MP.DS]
CELLS = [("median", 44606, 0.183), ("q3", 412875, 0.183)]


def bench(fn, reps):
    for _ in range(5):
        fn()
    v = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        v.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--out", default=None,
                    help="where to write the record; reproduce.py passes a "
                         "scratch path so a gate run cannot replace the curated "
                         "build_triple.json the paper quotes")
    a = ap.parse_args()

    store, sigma, n_groups = MP.build(PLANS)
    n_max = max(int(store.scope_len(c)) for c in range(len(PLANS)))
    k_max = max(len(store.band_u(c, MP.CAP_NEW, MP.CAP_OLD)) for c in range(len(PLANS)))
    buf = MP.Buf(n_max, k_max)

    ref0 = MP.environment()
    rows = []
    print("bare kernels on the pipeline's own store (%d codes, %d lines), %d reps"
          % (len(PLANS), store.n_l, a.reps))
    print("-- no mask, no probe, no overlay")
    print("%-7s %8s %8s | %-26s %-24s | %s"
          % ("scope", "n", "k", "path R alloc->compress",
             "path B fancy->take", "B/R"))
    for tag, n, d in CELLS:
        code = PLANS.index((tag, n, d))
        cand = store.band_u(code, MP.CAP_NEW, MP.CAP_OLD)
        k = int(len(cand))
        r_alloc, r_alloc_lo, r_alloc_hi = bench(
            lambda c=code: MP.scan_into(store, c, buf, "alloc"), a.reps)
        r_fast, r_fast_lo, r_fast_hi = bench(
            lambda c=code: MP.scan_into(store, c, buf, "fast"), a.reps)
        b_fancy, b_fancy_lo, b_fancy_hi = bench(
            lambda cd=cand: MP.gather_into(store, cd, buf, "fast", "fast"), a.reps)
        b_take, b_take_lo, b_take_hi = bench(
            lambda cd=cand: MP.gather_into(store, cd, buf, "fast", "alloc"), a.reps)

        # "throughout" = one style for both stages; "per path" = each stage's
        # own best, which is what a builder does for its own path and is not a
        # like-for-like comparison unless the table says which it is.  All four
        # combinations are stored, because the paper quotes their span and a
        # span with three of four corners missing is not checkable.
        ratio_alloc = b_fancy / r_alloc
        row = dict(tag=tag, n=n, k=k, dens=d,
                   R_alloc_us=r_alloc, R_alloc_p10=r_alloc_lo, R_alloc_p90=r_alloc_hi,
                   R_compress_us=r_fast, R_compress_p10=r_fast_lo,
                   R_compress_p90=r_fast_hi,
                   B_fancy_us=b_fancy, B_fancy_p10=b_fancy_lo, B_fancy_p90=b_fancy_hi,
                   B_take_us=b_take, B_take_p10=b_take_lo, B_take_p90=b_take_hi,
                   ratio_all_alloc=ratio_alloc,
                   ratio_all_compress=b_take / r_fast,
                   ratio_per_path=b_fancy / r_fast,
                   # the four corners, named, so the span the paper quotes is
                   # recomputable from this file alone
                   ratio_Ralloc_Bfancy=b_fancy / r_alloc,
                   ratio_Ralloc_Btake=b_take / r_alloc,
                   ratio_Rcompress_Bfancy=b_fancy / r_fast,
                   ratio_Rcompress_Btake=b_take / r_fast,
                   ratio_min=min(b_fancy, b_take) / max(r_alloc, r_fast),
                   ratio_max=max(b_fancy, b_take) / min(r_alloc, r_fast),
                   compact_rule_R=MP.build_rule(n),
                   compact_rule_B=MP.build_rule(k),
                   R_speedup_from_compress=r_alloc / r_fast,
                   B_speedup_from_take=b_fancy / b_take)
        rows.append(row)
        print("%-7s %8d %8d | %8.1f -> %8.1f us   %8.1f -> %8.1f us   "
              "thru %.3f  perpath %.3f  R x%.2f"
              % (tag, n, k, r_alloc, r_fast, b_fancy, b_take,
                 ratio_alloc, row["ratio_per_path"],
                 row["R_speedup_from_compress"]))
    ref1 = MP.environment()

    print()
    med = rows[0]
    print("the sentence Sect. V-A prints, from row 'median':")
    print("  path R  %.1f us boolean-index, %.1f us compress"
          % (med["R_alloc_us"], med["R_compress_us"]))
    print("  path B  %.1f us fancy, %.1f us take"
          % (med["B_fancy_us"], med["B_take_us"]))
    print("  so the cell is %.2fx one way, %.2fx per path -- both bare kernels."
          % (med["ratio_all_alloc"], med["ratio_per_path"]))
    print("  the pipeline value for the same (n,k) is printed by")
    print("  masked_pipeline.py and is NOT comparable to these: it charges the")
    print("  mask on both paths, which cancels in the criterion but not in a")
    print("  ratio of two costs.")

    if a.env:
        print("\nenvironment: reference %.0f us, drift %.2f"
              % (MP.environment() * 1e6, ref1 / ref0))

    os.makedirs(OUT, exist_ok=True)
    res = dict(reps=a.reps, cap=(MP.CAP_OLD, MP.CAP_NEW), plans=PLANS,
               instrument="bare kernel on the pipeline's store; no mask, no probe",
               store_lines=int(store.n_l), rows=rows,
               ref_us=float(ref0 * 1e6), drift=float(ref1 / ref0))
    dest = a.out if a.out else os.path.join(OUT, "build_triple.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nwrote %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
