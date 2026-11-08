#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One store, one |D|, one layout: the grid measured side by side.

Why this file exists
--------------------
`grid80.py` varies the scope n and lets the store grow with it -- one store per
scope, ten codes each -- so |D| is a hidden variable, and it is not small:
`out/store_size.json` moves a single fixed cell from 0.368 to 1.102 as the store
grows from one code to sixteen.

`grid_fixedD.py` removes that by holding |D| and filling the remainder with
filler codes.  Its sign surface collapses to rows: within a row, ten densities
from 0.02 to 0.60 return the same verdict, so the criterion's density term is
not what the machine measures.  But that file rebuilds the store for every one
of its seventy cells, and a rebuild also re-rolls the *layout*: where each code
lands in the address space, which `out/layout.json` shows is worth up to 2x at
the largest scope.  Its own `c_R` column says so -- at n = 197, cost per scanned
row ranges 33.5 to 93.4 ns with nothing varying but the density, which cannot be
a property of a scan over a fixed number of rows.

This file keeps both fixed.  Every cell lives in *one* store of fixed size, one
code per cell, produced by one permutation, measured in one block.  Between two
cells only n and k differ.  What survives that is a property of the comparison;
what does not was the harness.

The second thing it reports is the cost identity behind the surface.  Path R
does not only read n units: after the scan it gathers the maintained state at
the positions that flipped, and that count is k/2 by construction.  So

    T_R = F_R + a·n + b·k        T_B = F_B + c·k

and the band wins iff (c − b)·k + (F_B − F_R) < a·n.  When b ≈ c -- the two
gathers fetch the same array at the same scatter -- the k terms cancel and the
criterion is a threshold in n alone.  That is what the fixed-|D| surface shows,
and this file fits b and c rather than asserting them.

  python same_store.py --reps 15                 # 70 cells, |D| = sum of scopes
  python same_store.py --published --reps 15     # 4 published scopes, |D| = 2.0M
  python same_store.py --seeds 3                 # three rebuilds: is the surface stable?
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

SCOPES = [("197", 197, False), ("1k", 1000, True), ("6969", 6969, False),
          ("20k", 20000, True), ("44606", 44606, False),
          ("150k", 150000, True), ("412875", 412875, False)]

DENS = [0.02, 0.05, 0.09, 0.14, 0.183, 0.25, 0.30, 0.401, 0.50, 0.60]

TIE = 0.20


# ------------------------------------------------------------------ plumbing
def make_runners(mod, n_max, k_max):
    """Bind run_R/run_B whatever their signature is.

    `exp/masked_pipeline.py` gained a scratch-buffer argument between the
    committed version and the working copy, and this file has to run on both:
    the record in out/ was made with one and a reader may have the other.
    """
    sig = inspect.signature(mod.run_R)
    need_buf = len(sig.parameters) >= 5
    buf = mod.Buf(n_max, k_max) if need_buf else None

    def R(store, sigma, code, ov):
        if need_buf:
            return mod.run_R(store, sigma, code, ov, buf)
        return mod.run_R(store, sigma, code, ov)

    def B(store, sigma, code, ov, cand):
        if need_buf:
            return mod.run_B(store, sigma, code, ov, buf, cand)
        return mod.run_B(store, sigma, code, ov, cand)

    return R, B, need_buf


def build_one(plans, pad):
    """One store for every cell.  `pad` appends filler codes so that |D| can be
    set independently of the sum of the scopes; the pad codes are never timed."""
    p = list(plans)
    if pad > 0:
        p.append(("~pad", int(pad), 0.05))
    store, sigma, n_g = M.build(p)
    return store, sigma, len(plans)


def measure(store, sigma, cells, reps, R, B):
    """cells: [(code, tag, n_target, d_target)].  Warm every code, then time
    every cell in one interleaved block, so no cell is measured on a colder or
    a busier machine than its neighbours."""
    ov = np.zeros(store.n_l, dtype=bool)

    for code, _, _, _ in cells:
        R(store, sigma, code, ov)
        B(store, sigma, code, ov, None)
        store.band_u(code, CN, CO)

    rows = []
    for code, tag, n_t, d_t in cells:
        cand = store.band_u(code, CN, CO)
        k, n = int(len(cand)), int(store.scope_len(code))
        t = M.block({
            "R": lambda c=code: R(store, sigma, c, ov),
            "B": lambda c=code: B(store, sigma, c, ov, cand),
            "Bf": lambda c=code: B(store, sigma, c, ov, None),
            "Q": lambda c=code: store.band_u(c, CN, CO),
        }, reps)
        mR, mB, mBf, mQ = (float(np.median(t[x])) for x in ("R", "B", "Bf", "Q"))
        pred = bool(k + M.DP < M.KAPPA * n)
        rows.append(dict(
            scope=tag, n=n, n_target=n_t, d_target=d_t, dens=k / n, k=k,
            tR=mR, tB=mB, tBf=mBf, tQ=mQ,
            ratio=float(np.median(t["Bf"] / t["R"])),
            ratio_of_medians=mBf / mR,
            c_R_ns=mR * 1e3 / n,          # per scanned row
            c_B_ns=mB * 1e3 / max(k, 1),  # per fetched candidate, verification only
            c_Bf_ns=mBf * 1e3 / max(k, 1),  # probe included
            pred_band=pred, meas_band=bool(mBf < mR),
            agree=bool(pred == (mBf < mR)),
            R_lo=float(np.percentile(t["R"], 10)),
            R_hi=float(np.percentile(t["R"], 90))))
        r = rows[-1]
        print("  %-7s n %7d d %.3f k %6d | R %9.1f  Bf %9.1f | B/R %6.3f"
              " | c_R %5.2f  c_B %5.2f | pred %-4s meas %-4s %s"
              % (tag, n, r["dens"], k, mR, mBf, r["ratio"], r["c_R_ns"],
                 r["c_B_ns"], "band" if pred else "scan",
                 "band" if r["meas_band"] else "scan",
                 "" if r["agree"] else "<-- WRONG"), flush=True)
    return rows


# ------------------------------------------------------------------ analysis
def surface(rows, key, mark):
    scopes = []
    for r in rows:
        if r["scope"] not in scopes:
            scopes.append(r["scope"])
    out = []
    for s in scopes:
        c = sorted([r for r in rows if r["scope"] == s], key=lambda r: r["dens"])
        out.append(("%8d " % c[0]["n"]) + " ".join(mark(r[key]) for r in c))
    return out


def fit_costs(rows):
    """Fit T_R = F_R + a n + b k and T_B = F_B + c k, then ask what the fitted
    pair predicts cell by cell.  Fitting the costs rather than the sign is the
    point: a criterion fitted to signs on the same cells cannot be wrong.

    Units: times are microseconds, so a is us/row and b, c are us/candidate;
    both are scaled to nanoseconds on the way out, which is how the paper's
    kappa is quoted."""
    n = np.array([r["n"] for r in rows], float)
    k = np.array([r["k"] for r in rows], float)
    tR = np.array([r["tR"] for r in rows], float)
    tB = np.array([r["tBf"] for r in rows], float)

    XR = np.column_stack([np.ones_like(n), n, k])
    XB = np.column_stack([np.ones_like(k), k])
    bR, *_ = np.linalg.lstsq(XR, tR, rcond=None)
    bB, *_ = np.linalg.lstsq(XB, tB, rcond=None)

    def r2(y, pred):
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    net = bB[1] - bR[2]
    return dict(
        F_R=bR[0], a_ns_per_row=bR[1] * 1e3, b_ns_per_cand=bR[2] * 1e3,
        F_B=bB[0], c_ns_per_cand=bB[1] * 1e3,
        share_of_c=bR[2] / bB[1] if bB[1] else float("nan"),
        R2_R=r2(tR, XR @ bR), R2_B=r2(tB, XB @ bB),
        net_k_ns=net * 1e3,
        kappa=bR[1] / net if net else float("inf"),
        delta_p=(bB[0] - bR[0]) / net if net else float("inf"),
        pred=list(bool(x) for x in ((XB @ bB) < (XR @ bR))))


def score(rows, pred, tag):
    meas = np.array([r["meas_band"] for r in rows], bool)
    p = np.asarray(pred, bool)
    ok = int((p == meas).sum())
    fa = int((p & ~meas).sum())
    fr = int((~p & meas).sum())
    print("  %-38s %2d/%d   false-admit %2d  false-refuse %2d"
          % (tag, ok, len(rows), fa, fr))
    return dict(tag=tag, agree=ok, cells=len(rows), false_admit=fa,
                false_refuse=fr)


def report(rows, reps, D, label, path):
    print("\n" + "=" * 78)
    print("|D| = %d rows, %d cells, %d repetitions%s" % (D, len(rows), reps, label))
    print("=" * 78)
    print("\nRATIO  T_B/T_R   (>1: the band is dearer)")
    print("%-9s" % "n" + " ".join("%6.2f" % r["dens"] for r in
                                  sorted([x for x in rows if x["scope"] == rows[0]["scope"]],
                                         key=lambda x: x["dens"])))
    for line in surface(rows, "ratio", lambda v: "%6.2f" % v):
        print(line)
    print("\nMEASURED  (b = the stopwatch prefers the band)")
    for line in surface(rows, "meas_band", lambda v: "%6s" % ("b" if v else ".")):
        print(line)
    print("\nCRITERION eq.(12), kappa %.4f, dp %.1f" % (M.KAPPA, M.DP))
    for line in surface(rows, "pred_band", lambda v: "%6s" % ("b" if v else ".")):
        print(line)

    meas = [r for r in rows if r["meas_band"]]
    print("\nbaselines: all-band %d/%d, all-scan %d/%d"
          % (len(meas), len(rows), len(rows) - len(meas), len(rows)))
    base = score(rows, [r["pred_band"] for r in rows], "eq.(12) as printed")

    ties = [r for r in rows if abs(r["ratio"] - 1) <= TIE]
    off = [r for r in rows if abs(r["ratio"] - 1) > TIE]
    print("  within the tie band %d/%d, beyond it %d/%d"
          % (sum(1 for r in ties if r["agree"]), len(ties),
             sum(1 for r in off if r["agree"]), len(off)))

    print("\nis the density term visible?  per row, the ratio against d:")
    slopes = []
    for s in dict.fromkeys(r["scope"] for r in rows):
        c = sorted([r for r in rows if r["scope"] == s], key=lambda r: r["dens"])
        x = np.array([r["dens"] for r in c])
        y = np.array([r["ratio"] for r in c])
        sl = np.polyfit(x, y, 1)[0] / 10.0
        pred = np.polyval(np.polyfit(x, y, 1), x)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        slopes.append(dict(n=c[0]["n"], slope_per_01d=float(sl), r2=float(r2),
                           lo=float(y.min()), hi=float(y.max())))
        print("   n %7d  %+6.2f per 0.1 of d   R2 %.2f   ratio spans %.2f-%.2f"
              % (c[0]["n"], sl, r2, y.min(), y.max()))

    f = fit_costs(rows)
    print("\ncost fit (|D| = %d)  T_R = %.1f us + %.2f ns/row * n + %.1f ns/cand * k"
          "   (R2 %.3f)" % (D, f["F_R"], f["a_ns_per_row"], f["b_ns_per_cand"],
                            f["R2_R"]))
    print("                     T_B = %.1f us + %.2f ns/cand * k"
          "                       (R2 %.3f)"
          % (f["F_B"], f["c_ns_per_cand"], f["R2_B"]))
    print("  path R already pays %.0f%% of path B's per-candidate cost (mask and"
          " emit gather the flips)" % (100 * f["share_of_c"]))
    print("  net k coefficient c-b = %+.2f ns/cand;  kappa = a/(c-b) = %.3f"
          % (f["net_k_ns"], f["kappa"]))
    print("  dp = (F_B-F_R)/(c-b) = %.0f rows" % f["delta_p"])

    fit_score = score(rows, f["pred"], "fitted costs, in-sample")

    n = np.array([r["n"] for r in rows], float)
    k = np.array([r["k"] for r in rows], float)

    # leave one scope out: the fitted criterion must predict a scope it never saw
    loo = np.zeros(len(rows), bool)
    for s in dict.fromkeys(r["scope"] for r in rows):
        te = np.array([r["scope"] == s for r in rows])
        tr = ~te
        if te.all() or not tr.any():
            continue
        XR = np.column_stack([np.ones(tr.sum()), n[tr], k[tr]])
        XB = np.column_stack([np.ones(tr.sum()), k[tr]])
        cR, *_ = np.linalg.lstsq(XR, np.array([r["tR"] for r in rows])[tr], rcond=None)
        cB, *_ = np.linalg.lstsq(XB, np.array([r["tBf"] for r in rows])[tr], rcond=None)
        loo[te] = (np.column_stack([np.ones(te.sum()), k[te]]) @ cB
                   < np.column_stack([np.ones(te.sum()), n[te], k[te]]) @ cR)
    loo_score = score(rows, loo, "fitted costs, leave one scope out")

    res = dict(reps=reps, D=int(D), cells=len(rows), rows=rows, tie=TIE,
               baselines=dict(all_band=len(meas), all_scan=len(rows) - len(meas)),
               printed=base, fitted=fit_score, loo=loo_score,
               cost_fit=f, density_slopes=slopes,
               tuple="eq.(12) with kappa %.4f, dp %.1f" % (M.KAPPA, M.DP))
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s" % path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--pad", type=int, default=0,
                    help="filler rows appended so |D| is set independently")
    ap.add_argument("--published", action="store_true",
                    help="the four measured scopes only")
    ap.add_argument("--scopes", default=None,
                    help="comma-separated n values, a subset of the seven")
    ap.add_argument("--dens", default=None,
                    help="comma-separated densities, a subset of the ten")
    ap.add_argument("--seeds", type=int, default=1,
                    help="rebuild the store with a new permutation this many times")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    scopes = [s for s in SCOPES if (not a.published or not s[2])]
    if a.scopes:
        keep = {int(x) for x in a.scopes.split(",")}
        scopes = [s for s in scopes if s[1] in keep]
    dens = list(DENS)
    if a.dens:
        dens = [float(x) for x in a.dens.split(",")]
    plans = [("%s@%.3f" % (tag, d), n, d) for tag, n, _ in scopes for d in dens]
    n_max = max(n for _, n, _ in scopes)
    k_max = int(max(dens) * n_max) + 8

    runs = []
    for seed_i in range(a.seeds):
        t0 = time.time()
        if a.seeds > 1:
            M.SEED = M.SEED + seed_i * 7919
            print("\n### rebuild %d/%d, seed %d" % (seed_i + 1, a.seeds, M.SEED))
        store, sigma, n_cell = build_one(plans, a.pad)
        R, B, need_buf = make_runners(M, max(n_max, a.pad), k_max)
        cells = [(c, scopes[c // len(dens)][0], scopes[c // len(dens)][1],
                  dens[c % len(dens)]) for c in range(len(plans))]
        print("store %d rows (%.1f MB of units), %d cells, buffer path %s"
              % (store.n_l, store.n_l * 2 / 1e6, n_cell,
                 "on" if need_buf else "off"))
        rows = measure(store, sigma, cells, a.reps, R, B)
        runs.append(dict(seed=int(M.SEED), D=int(store.n_l), rows=rows,
                         seconds=time.time() - t0))
        del store, sigma

    label = " (published scopes)" if a.published else ""
    if a.pad:
        label += " (padded)"
    path = a.out or os.path.join(
        OUT, "same_store_published.json" if a.published else "same_store.json")
    res = report(runs[-1]["rows"], a.reps, runs[-1]["D"], label, path)
    res["runs"] = runs

    if a.seeds > 1:
        print("\n" + "=" * 78)
        print("STABILITY ACROSS REBUILDS  (same |D|, new permutation)")
        print("=" * 78)
        for r in runs:
            meas = sum(1 for x in r["rows"] if x["meas_band"])
            agree = sum(1 for x in r["rows"] if x["agree"])
            print("  seed %-7d  all-band baseline %2d/%d   eq.(12) %2d/%d"
                  % (r["seed"], meas, len(r["rows"]), agree, len(r["rows"])))
        signs = [[x["meas_band"] for x in r["rows"]] for r in runs]
        flips = sum(1 for i in range(len(signs[0]))
                    if len({s[i] for s in signs}) > 1)
        print("  cells whose measured sign changes across rebuilds: %d/%d"
              % (flips, len(signs[0])))
        res["rebuild_flips"] = flips
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
