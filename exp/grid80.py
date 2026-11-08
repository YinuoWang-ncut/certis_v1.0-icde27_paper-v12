#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The calibration grid: 8 published scope sizes x 10 band densities = 80 cells.

Why this file exists
--------------------
`masked_pipeline.py` measures twelve cells -- the four published per-code sizes
crossed with the three ceilings of Sec. V-D(c) -- and the paper reports "19 of
24" from `bigscope.py` and "4 of 4" from that file.  Twelve and twenty-four
points are a case list, not a calibration: a reader who asks "does the sign hold
at some other (n, k)?" cannot be answered except by rerunning, and the paper has
no statement about where its criterion stops being right.

This file replaces the case list with a grid.  It keeps the pipeline of
Sec. IV-F (probe, decide, execute, mask, overlay), the layout of `bigscope.py`
(code drawn at random over the whole store, so path B gathers through a
scattered id space while path R scans a code-clustered copy), and the two-rule
support structure that makes the mask fire, and varies only what the criterion
is a function of: the scope n and the band density d = k/n.

  n in {197, 1k, 6969, 20k, 44606, 150k, 412875, 1.6M}      3.9 decades
  d in {0.02, 0.05, 0.09, 0.14, 0.183, 0.25, 0.30, 0.401, 0.50, 0.60}

The four measured sizes are the published per-code beneficiary-day counts of
`realband.py` (5th percentile, quartile, median, upper quartile of six years of
CMS data); the four added ones interpolate and extrapolate that range by half a
decade each so the grid is regular in log n without moving the measured points.
Two densities are the measured ceilings (0.183, 0.401); the rest straddle them.

What is reported
----------------
Not "how many cells agree" alone, which is a score, but the two things a
calibrated instrument owes a reader:

  1. the sign surface -- for every cell, what the criterion says and what the
     stopwatch says, as a grid over (log n, d), so a reader sees the shape;
  2. the error envelope -- for every disagreement, |T_B/T_R - 1|, so a reader
     sees whether the criterion is wrong anywhere that matters.

One scope at a time, because a 1.6M-line code x 10 densities is 16M lines and
there is no reason to hold three of those at once.

  python grid80.py --reps 11
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import certis as C                      # noqa: E402
import masked_pipeline as M             # noqa: E402  (build, stages, block, med)

# The four published per-code sizes, plus four that make the grid regular in
# log n.  The added ones are marked; nothing in the paper's tables changes.
SCOPES = [("197", 197, False),
          ("1k", 1000, True),
          ("6969", 6969, False),
          ("20k", 20000, True),
          ("44606", 44606, False),
          ("150k", 150000, True),
          ("412875", 412875, False),
          ("1m6", 1600000, True)]

# 0.02 to 0.60 in ten steps, keeping both measured ceilings (0.183, 0.401).
DENS = [0.02, 0.05, 0.09, 0.14, 0.183, 0.25, 0.30, 0.401, 0.50, 0.60]

TIE = 0.20          # |T_B/T_R - 1| within which the two paths are a tie


def measure_scope(tag, n, reps, verbose=True):
    """One store per scope, one code per density, then the same block as
    masked_pipeline: eight arms, round-robin, arms warm before any is timed."""
    plans = [("%s@%.3f" % (tag, d), n, d) for d in DENS]
    store, sigma, n_g = M.build(plans)
    ovh = np.zeros(store.n_l, dtype=bool)
    # every stage writes into caller-owned scratch, so one buffer serves the
    # whole store; sizing it by the largest single scope and band, not by |D|,
    # is what keeps the block from measuring the allocator
    buf = M.Buf(max(int(store.scope_len(c)) for c in range(len(plans))),
                max(len(store.band_u(c, M.CAP_NEW, M.CAP_OLD))
                    for c in range(len(plans))))

    for code in range(len(plans)):          # warm every cell before timing any
        M.run_R(store, sigma, code, ovh, buf)
        M.run_B(store, sigma, code, ovh, buf)
        store.band_u(code, M.CAP_NEW, M.CAP_OLD)

    rows = []
    for code, d in enumerate(DENS):
        cand = store.band_u(code, M.CAP_NEW, M.CAP_OLD)
        k, n_scope = int(len(cand)), int(store.scope_len(code))

        def kern_R(c=code):
            M.scan_into(store, c, buf)

        def kern_B(c=cand):
            M.gather_into(store, c, buf)

        def routed(c=code, kk=k, nn=n_scope):
            band = store.band_u(c, M.CAP_NEW, M.CAP_OLD)
            if kk + M.DP < M.KAPPA * nn:
                return M.run_B(store, sigma, c, ovh, buf, band)
            return M.run_R(store, sigma, c, ovh, buf)

        t = M.block({
            "R": lambda c=code: M.run_R(store, sigma, c, ovh, buf),
            "B": lambda c=code, cd=cand: M.run_B(store, sigma, c, ovh, buf, cd),
            # Bf: the whole band path, probe included, run whatever the
            # criterion says.  This is the arm eq. (12) prices -- T_B carries
            # the rank probe -- so the stopwatch reading must carry it too.
            # Measuring verification alone (B) and comparing it with a
            # criterion that charges the probe is the mistake this arm exists
            # to make impossible: at n = 197 it flips the sign of ten cells.
            "Bf": lambda c=code: M.run_B(store, sigma, c, ovh, buf, None),
            "Q": lambda c=code: store.band_u(c, M.CAP_NEW, M.CAP_OLD),
            "kR": kern_R, "kB": kern_B,
            "rt": routed,
        }, reps)

        mR, mB, mQ = M.med(t["R"]), M.med(t["B"]), M.med(t["Q"])
        mBf = M.med(t["Bf"])
        pred = bool(k + M.DP < M.KAPPA * n_scope)
        meas = bool(mBf < mR)
        # Delta-completeness on this cell, outside the timing
        flips = M.flips_of(store, sigma, code, buf, cand)
        same = bool(np.array_equal(np.sort(M.ground_truth(store, code)),
                                   np.sort(flips)))
        r = dict(scope=tag, n=n_scope, d_target=d, dens=k / max(n_scope, 1), k=k,
                 tR=mR, tB=mB, tQ=mQ,
                 tR_lo=float(np.percentile(t["R"], 10)),
                 tR_hi=float(np.percentile(t["R"], 90)),
                 tB_lo=float(np.percentile(t["B"], 10)),
                 tB_hi=float(np.percentile(t["B"], 90)),
                 # Paired: block() runs every arm once per repetition, so
                 # t["Bf"][i] and t["R"][i] are neighbours in time and the
                 # per-repetition ratio is the statistic.  At the two largest
                 # scopes the p10-p90 spread of a single arm reaches 2.6x, so a
                 # ratio of two medians is not the same number as a median of
                 # ratios; bigscope.py reports the latter and so does this file.
                 ratio=float(np.median(t["Bf"] / t["R"])),
                 ratio_of_medians=mBf / mR,
                 ratio_verify=float(np.median(t["B"] / t["R"])),
                 tBf=mBf,
                 ratio_routed=(mQ + (mB if pred else mR)) / mR,
                 pred_band=pred, meas_band=meas, agree=bool(pred == meas),
                 kappa_eff=(mR / max(n_scope, 1)) / (mB / max(k, 1)),
                 mask_us=0.0,
                 same=same)
        rows.append(r)
        if verbose:
            print("  %-13s n %7d  k %6d  d %.3f | R %9.1f  Bf%9.1f  Q %5.1f"
                  " | B/R %6.3f | pred %-4s meas %-4s %s"
                  % (r["scope"], r["n"], r["k"], r["dens"], r["tR"], r["tBf"],
                     r["tQ"], r["ratio"],
                     "band" if pred else "scan",
                     "band" if meas else "scan",
                     "" if r["agree"] else "<-- WRONG"), flush=True)
    del store, sigma
    return rows


def surface(rows, key, band_mark="b", scan_mark="."):
    """A text grid: rows are scopes (small to large), columns are densities."""
    out = []
    scopes = []
    for r in rows:
        if r["scope"] not in scopes:
            scopes.append(r["scope"])
    for s in scopes:
        cells = sorted([r for r in rows if r["scope"] == s],
                       key=lambda r: r["dens"])
        out.append("%8d " % cells[0]["n"]
                   + " ".join(band_mark if c[key] else scan_mark for c in cells))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(OUT, "grid80.json"))
    a = ap.parse_args()

    t0 = time.time()
    rows = []
    print("densities: " + " ".join("%.3f" % d for d in DENS))
    for tag, n, synthetic in SCOPES:
        print("\n== n = %d%s (%.1fx the 1.6M reference)"
              % (n, ", interpolated" if synthetic else ", published",
                 n / 197.0), flush=True)
        rows += measure_scope(tag, n, a.reps)

    # ---------------- aggregation -------------------------------------------
    agree = sum(1 for r in rows if r["agree"])
    wrong = [r for r in rows if not r["agree"]]
    ties = [r for r in rows if abs(r["ratio"] - 1) <= TIE]
    off = [r for r in rows if abs(r["ratio"] - 1) > TIE]

    print()
    print("=" * 78)
    print("SIGN SURFACE  (b = criterion says band, . = says scan)")
    print("=" * 78)
    print("           " + " ".join("%5.2f" % d for d in DENS))
    for line in surface(rows, "pred_band"):
        print(line)
    print()
    print("MEASURED SURFACE  (b = the stopwatch says band)")
    for line in surface(rows, "meas_band"):
        print(line)
    print()
    print("DISAGREEMENTS  (%d of %d cells)" % (len(wrong), len(rows)))
    for r in sorted(wrong, key=lambda r: abs(r["ratio"] - 1), reverse=True):
        print("   n %7d  d %.3f  k %6d | B/R %6.3f  |B/R-1| %.3f  (%s)"
              % (r["n"], r["dens"], r["k"], r["ratio"], abs(r["ratio"] - 1),
                 "inside the tie band" if abs(r["ratio"] - 1) <= TIE
                 else "OUTSIDE the tie band"))
    print()
    print("agreement: %d/%d" % (agree, len(rows)))
    print("    within the tie band (|B/R-1| <= %.2f): %d/%d"
          % (TIE, sum(1 for r in ties if r["agree"]), len(ties)))
    print("    beyond it:                            %d/%d"
          % (sum(1 for r in off if r["agree"]), len(off)))
    false_admit = [r for r in off if r["pred_band"] and not r["meas_band"]]
    false_refuse = [r for r in off if r["meas_band"] and not r["pred_band"]]
    print("    false admissions beyond the tie band: %d" % len(false_admit))
    print("    false refusals beyond it:             %d" % len(false_refuse))
    if wrong:
        worst = max(abs(r["ratio"] - 1) for r in wrong)
        print("    largest |B/R-1| among disagreements:  %.3f" % worst)
    print("delta-completeness on all %d cells: %s"
          % (len(rows), all(r["same"] for r in rows)))
    print("\nwall %.0fs" % (time.time() - t0))

    res = dict(reps=a.reps, scopes=[(t, n, s) for t, n, s in SCOPES],
               densities=DENS, tie=TIE, cells=len(rows), rows=rows,
               agree=agree, tie_cells=len(ties), off_cells=len(off),
               agree_tie=sum(1 for r in ties if r["agree"]),
               agree_off=sum(1 for r in off if r["agree"]),
               false_admissions=len(false_admit),
               false_refusals=len(false_refuse),
               max_abs_err=float(max((abs(r["ratio"] - 1) for r in wrong),
                                    default=0.0)),
               delta_complete=bool(all(r["same"] for r in rows)),
               tuple="eq.(12) with kappa 0.2142, dp 198.8")
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
