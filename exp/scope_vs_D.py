#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sect. V-A's four scopes, measured at several store sizes.

Why this file exists
--------------------
Sect. V-A prints one number per published scope: at d = 0.183 the band costs
1.01, 0.91, 0.92 and 0.99 times the scan at 197, 6,969, 44,606 and 412,875 rows.
Sect. V-E prints 1.2, 1.2, 1.1 and 1.7 for the same four scopes at the same
density, and explains the gap by the store: V-A's cells share 1,393,941 rows
(four scopes x three densities), V-E's share 6,356,470.  That explanation has
never been tested -- the two numbers come from different scripts and different
stores at once -- and the two sections differ in a second respect that has
nothing to do with the store.

`masked_pipeline.py` reports `ratio`, which is path B *without* the rank probe
against path R, and `ratio_full`, which adds the probe.  Fig. 2 plots `ratio`;
Sect. V-E's grid times the probe inside the band arm.  So comparing the two
sections mixes

  (i)  the store, 1.39M rows against 6.36M, and
  (ii) the accounting, band without the probe against band with it,

and this file separates them.  It rebuilds V-A's twelve-cell store exactly --
archived seed, plan list, build rule, arm definitions -- and confirms it
reproduces the published four both ways; then it rebuilds the same four scopes
inside stores of 0.46, 1.39, 2, 4 and 6.36 million rows and measures again.

  python scope_vs_D.py --reps 11
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import masked_pipeline as M                     # noqa: E402

CN, CO = M.CAP_NEW, M.CAP_OLD

# V-A's four published per-code scopes and the three densities it crosses them
# with, imported rather than retyped: a control that re-specifies its own input
# is not a control.
SCOPES = [(t, n) for t, n in M.NS]
DENS = list(M.DS)
D_MEASURED = 0.183

TARGETS = [("mp12", None),               # exactly masked_pipeline's store
           ("cells4", 464_647),          # the four codes and nothing else
           ("pad1393k", 1_393_941),      # same |D| as the control, built differently
           ("pad2M", 2_000_000),
           ("pad4M", 4_000_000),
           ("pad6356k", 6_356_470)]      # same |D| as the fixed-|D| grid

PAD_DENS = 0.05      # filler code's band density; never timed, only occupies rows


def make_runners(mod, n_max, k_max):
    """Bind run_R/run_B whatever their signature is: the working copy takes a
    scratch buffer, the committed one allocates."""
    need = len(inspect.signature(mod.run_R).parameters) >= 5
    buf = mod.Buf(n_max, k_max) if need else None

    def R(store, sigma, code, ov, how):
        return mod.run_R(store, sigma, code, ov, buf, how) if need else \
            mod.run_R(store, sigma, code, ov)

    def B(store, sigma, code, ov, cand, gap, how, mgap):
        if need:
            return mod.run_B(store, sigma, code, ov, buf, cand, gap, how, mgap)
        return mod.run_B(store, sigma, code, ov, cand)

    return R, B, need


def build_mp12():
    return [(t, n, d) for t, n in SCOPES for d in DENS], 0


def build_padded(target):
    """One code per scope at the measured density, plus a filler holding the
    rest of the rows, so |D| is set independently of the cells under test."""
    plans = [(t, n, D_MEASURED) for t, n in SCOPES]
    pad = max(0, target - sum(n for _, n in SCOPES))
    if pad:
        plans = plans + [("~pad", pad, PAD_DENS)]
    return plans, pad


def measure(store, sigma, cells, reps, R, B):
    """The arm set of masked_pipeline, so the control is a control: cand is
    probed before timing, path B runs without it, and the probe is its own arm.
    `ratio` is band-without-probe over scan; `ratio_full` adds the probe the way
    eq. (12) prices it."""
    ov = np.zeros(store.n_l, dtype=bool).view(np.uint8)
    # Warm as hard as masked_pipeline does before it times a cell: that script
    # runs eight build variants at five repetitions each first, so its cells are
    # measured on a store the sweep has already pulled through cache.  Two
    # passes leave the largest scope 18% slower here and the control then does
    # not reproduce the archived record -- which would make this file's numbers
    # incomparable to the ones it is checking.
    WARM = 40
    for code, _ in cells:
        for _ in range(WARM):
            R(store, sigma, code, ov, "fast")
            B(store, sigma, code, ov, None, "fast", "fast", "fast")
        store.band_u(code, CN, CO)

    rows = []
    for code, tag in cells:
        cand = store.band_u(code, CN, CO)
        k, n = int(len(cand)), int(store.scope_len(code))
        cR, cB = M.build_rule(n), M.build_rule(k)     # V-A's build rule
        t = M.block({
            "R": lambda: R(store, sigma, code, ov, cR),
            "B": lambda: B(store, sigma, code, ov, cand, "fast", cB, "fast"),
            "Q": lambda: store.band_u(code, CN, CO),
        }, reps)
        mR, mB, mQ = M.med(t["R"]), M.med(t["B"]), M.med(t["Q"])
        rows.append(dict(tag=tag, n=n, k=k, dens=k / n, tR=mR, tB=mB, tQ=mQ,
                         ratio=float(np.median(t["B"] / t["R"])),
                         ratio_full=float(np.median((t["B"] + t["Q"]) / t["R"])),
                         ratio_of_medians=mB / mR,
                         c_R_ns=mR * 1e3 / n,
                         c_B_ns=mB * 1e3 / max(k, 1),
                         build_R=cR, build_B=cB,
                         R_lo=float(np.percentile(t["R"], 10)),
                         R_hi=float(np.percentile(t["R"], 90))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(OUT, "scope_vs_D.json"))
    a = ap.parse_args()

    runs = []
    t0 = time.time()
    for label, target in TARGETS:
        plans, pad = build_mp12() if target is None else build_padded(target)
        cells = []
        for i, (name, n, d) in enumerate(plans):
            if name == "~pad":
                continue
            if target is None and abs(d - D_MEASURED) > 1e-9:
                continue
            cells.append((i, name))

        store, sigma, n_g = M.build(plans)
        R, B, need = make_runners(M, max(n for _, n in SCOPES),
                                  int(max(DENS) * max(n for _, n in SCOPES)) + 8)
        rows = measure(store, sigma, cells, a.reps, R, B)
        D = int(store.n_l)
        runs.append(dict(label=label, D=D, pad=pad, codes=len(plans),
                         reps=a.reps, buffer_path=bool(need), rows=rows))
        print("%-9s |D| %9d (%4.2f MB of units, %2d codes, pad %d)"
              % (label, D, D * 2 / 1e6, len(plans), pad), flush=True)
        for r in rows:
            print("   n %7d  k %6d | R %8.1f  B %8.1f  Q %5.2f us"
                  " | B/R %6.3f  +probe %6.3f"
                  % (r["n"], r["k"], r["tR"], r["tB"], r["tQ"],
                     r["ratio"], r["ratio_full"]), flush=True)
        del store, sigma

    # ---------------- report -------------------------------------------------
    def table(field, title):
        print("\n" + title)
        hdr = "%-10s %9s" % ("store", "|D|")
        for t, n in SCOPES:
            hdr += " %7d" % n
        print(hdr)
        for r in runs:
            line = "%-10s %9d" % (r["label"], r["D"])
            for _, n in SCOPES:
                cell = next(x for x in r["rows"] if x["n"] == n)
                line += " %7.3f" % cell[field]
            print(line)

    print("\n" + "=" * 78)
    table("ratio", "T_B/T_R, band WITHOUT the probe (what Fig. 2 plots)")
    table("ratio_full", "T_B/T_R, band WITH the probe (what Sect. V-E plots)")

    print("\ncontrol: masked_pipeline.json prints 1.01, 0.91, 0.92, 0.99 (ratio)")
    ctrl = next(r for r in runs if r["label"] == "mp12")
    print("  this run: " + ", ".join("%.3f" % x["ratio"]
                                      for x in sorted(ctrl["rows"], key=lambda x: x["n"])))
    print("  its ratio_full at the same cells: " +
          ", ".join("%.3f" % x["ratio_full"]
                    for x in sorted(ctrl["rows"], key=lambda x: x["n"])))
    print("Sect. V-E (70 cells, |D| = 6.36M, probe inside the band arm):"
          " 1.2, 1.2, 1.1, 1.7")

    lo = next(r for r in runs if r["label"] == "cells4")
    hi = next(r for r in runs if r["label"] == "pad6356k")
    print("\nmovement from the four-cell store to the 6.36M one:")
    for _, n in SCOPES:
        a0 = next(x for x in lo["rows"] if x["n"] == n)
        b0 = next(x for x in hi["rows"] if x["n"] == n)
        print("  n %7d  without probe %.3f -> %.3f   with probe %.3f -> %.3f"
              % (n, a0["ratio"], b0["ratio"], a0["ratio_full"], b0["ratio_full"]))

    # A flat projection of the two tables above, so a reader (and
    # expected.json) can address one number without walking the run list.
    flat = {}
    for r in runs:
        for x in r["rows"]:
            # key without a dot or an @: reproduce.py resolves recorded values
            # as dotted paths, so a label carrying either would not parse
            flat["%s_n%d" % (r["label"], x["n"])] = dict(
                D=r["D"], n=x["n"], k=x["k"],
                ratio=round(x["ratio"], 4),
                ratio_full=round(x["ratio_full"], 4),
                c_R_ns=round(x["c_R_ns"], 3), c_B_ns=round(x["c_B_ns"], 2))

    res = dict(reps=a.reps, dens=D_MEASURED, scopes=[[t, n] for t, n in SCOPES],
               targets=[[l, t] for l, t in TARGETS], pad_dens=PAD_DENS,
               ratios=flat, runs=runs, seconds=time.time() - t0,
               how=dict(
                   mp12="masked_pipeline's own twelve-cell store; the control",
                   cells4="the four codes alone, no filler",
                   pad="a filler code holds the remaining rows so |D| moves "
                       "without touching the cells under test",
                   ratio="band without the probe over scan (what Fig. 2 plots)",
                   ratio_full="band with the probe over scan"))
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s  (%.0fs)" % (a.out, res["seconds"]))


if __name__ == "__main__":
    main()
