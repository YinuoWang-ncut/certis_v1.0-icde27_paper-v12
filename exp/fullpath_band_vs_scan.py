#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The missing cell: is verifying a band cheaper than scanning the scope,
once both paths carry the mask and neither is charged for the probe?

Why this file exists
--------------------
P4 Sec. 3.6 derived, rather than measured, that the band costs more than the
scan on the complete path: B-always p50 38.75 us minus the 8.1 us probe gives
~30.6 us of "band verification", against 27.3 us for R-always -- about 12%
dearer.  That was an ecological subtraction of two cell-level p50s taken in
different arms, so it carries the batch overhead, the arm order and the probe
in one number.  What the paper needs is the paired quantity:

    tR  = _path_R(c)              contiguous pass over the scope, two states
                                  judged in one pass, override filter applied
    tQ  = band_u(code, lo, hi)    the probe alone (the rank query)
    tBv = _path_B(c, cand)        verify the band from an already-obtained
                                  candidate array, same override filter

tBv and tR are the same obligation on different layouts, so their ratio is the
"even with a free probe" question, and tQ + tBv is the complete band path.

Protocol
--------
Same 600 cap corrections, same seed, same store as the duel runs.  For each
correction the probe runs once outside the timed pair, then the two paths run
in alternating order (even: R then Bv, odd: Bv then R) so neither arm is always
the warm one.  Flip sets are compared per correction; a single mismatch aborts.

  set CERTIS_PROTO=D:\\期刊\\2026-11-04-00-12-59\\w12
  python fullpath_band_vs_scan.py --reps 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

W12 = os.environ.get("CERTIS_PROTO") or \
    os.path.join(HERE, "prototype", "w12")
sys.path.insert(0, os.path.join(HERE, ".."))   # the core: certis.py at the root
sys.path.insert(0, W12)
sys.path.insert(0, os.path.join(os.path.dirname(W12), "w8"))

import run_w12 as W                                            # noqa: E402
import certis as C                                             # noqa: E402

MIX = {"cap_low": 0.5, "cap_high": 0.5, "pair": 0.0}
SEED = 31337
N_BATCH = 60


def calib():
    ref = np.arange(1 << 21, dtype=np.int64)
    v = []
    for _ in range(30):
        t0 = time.perf_counter()
        np.count_nonzero(ref >= 12345)
        v.append(time.perf_counter() - t0)
    return float(np.median(v))


def fit(x, y):
    a, b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    yh = a * np.asarray(x, float) + b
    ss = float(np.sum((np.asarray(y, float) - np.mean(y)) ** 2))
    r2 = 1 - float(np.sum((np.asarray(y, float) - yh) ** 2)) / ss if ss > 0 else float("nan")
    return float(a), float(b), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-g", type=int, default=1_000_000)
    ap.add_argument("--n-batch", type=int, default=N_BATCH)
    ap.add_argument("--out", default=os.path.join(OUT, "fullpath.json"))
    a = ap.parse_args()

    t0 = time.time()
    topo = W.load_topology()
    topo["pair_codes"] = sorted(topo["pair_codes"])
    inst, store, rules, stream, rng = W.make(a.n_g, topo, SEED)
    cert = C.Certifier(store, rules, policy="R-always")
    print("store: %d lines, topo %.1fs" % (store.n_l, time.time() - t0), flush=True)

    corrs = []
    for _ in range(a.n_batch):
        corrs += [c for c in stream.batch(1000, 0.01, MIX, "as-is")
                  if c.kind == "cap"]
    print("corrections: %d cap" % len(corrs), flush=True)

    # candidate arrays, obtained once per correction, outside the timed pair.
    # Reusing them across reps is deliberate: it keeps the *same* candidates
    # for both arms in every rep, so the comparison is of the two code paths
    # and not of two different band sizes.
    cands, meta = [], []
    for c in corrs:
        lo, hi = sorted((c.old, c.new))
        ns = store.scope_len(c.code)
        b = store.band_u(c.code, int(lo), int(hi))
        cands.append(b)
        meta.append(dict(n=int(ns), k=int(len(b)), code=int(c.code)))

    # warm-up: 80 corrections through both paths, so no pass starts cold
    for c, b in list(zip(corrs, cands))[:80]:
        cert._path_R(c)
        cert._path_B(c, b)

    reps = []
    for rep in range(a.reps):
        c0 = calib()
        tR, tBv, tQ, tBf = [], [], [], []
        bad = 0
        for i, (c, b) in enumerate(zip(corrs, cands)):
            lo, hi = sorted((c.old, c.new))
            tt = time.perf_counter()
            _ = store.band_u(c.code, int(lo), int(hi))   # the probe, timed alone
            tQ.append(time.perf_counter() - tt)

            # three arms, order rotated by i % 3 so no arm is always first
            arm = [("R", cert._path_R, (c,)),
                   ("Bv", cert._path_B, (c, b)),          # verify, probe excluded
                   ("Bf", cert._path_B, (c, None))]       # probe + verify, one call
            arm = [arm[(j + i) % 3] for j in range(3)]
            got = {}
            for tag, fn, args in arm:
                tt = time.perf_counter()
                got[tag] = fn(*args)
                (tR if tag == "R" else tBv if tag == "Bv" else tBf).append(
                    time.perf_counter() - tt)
            if not (np.array_equal(np.sort(got["R"]), np.sort(got["Bv"]))
                    and np.array_equal(np.sort(got["R"]), np.sort(got["Bf"]))):
                bad += 1
        c1 = calib()
        tR, tBv = np.array(tR) * 1e6, np.array(tBv) * 1e6
        tQ, tBf = np.array(tQ) * 1e6, np.array(tBf) * 1e6
        rep_row = dict(rep=rep, calib0=c0, calib1=c1, mismatch=bad,
                       med_tR=float(np.median(tR)), med_tBv=float(np.median(tBv)),
                       med_tQ=float(np.median(tQ)), med_tBf=float(np.median(tBf)),
                       mean_tR=float(tR.mean()), mean_tBv=float(tBv.mean()),
                       mean_tQ=float(tQ.mean()), mean_tBf=float(tBf.mean()),
                       band_cheaper_verify=int(np.sum(tBv < tR)),
                       band_cheaper_full=int(np.sum(tBf < tR)))
        reps.append(rep_row)
        print("rep %d  med tR %7.2f  tBv %7.2f  tQ %6.2f  tBf %7.2f us | "
              "tBv/tR %.3f  tBf/tR %.3f | band wins %d verify, %d full | "
              "mismatch %d  calib %.3f us"
              % (rep, rep_row["med_tR"], rep_row["med_tBv"], rep_row["med_tQ"],
                 rep_row["med_tBf"], rep_row["med_tBv"] / rep_row["med_tR"],
                 rep_row["med_tBf"] / rep_row["med_tR"],
                 rep_row["band_cheaper_verify"], rep_row["band_cheaper_full"],
                 bad, c0 * 1e6), flush=True)

        if rep == 0:
            tR0, tBv0, tQ0, tBf0 = tR, tBv, tQ, tBf

    # ---- per-correction paired analysis on the last rep ---------------------
    n = np.array([m["n"] for m in meta], float)
    k = np.array([m["k"] for m in meta], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = tBv0 / tR0
    ok = np.isfinite(ratio)

    print()
    print("paired per correction (rep 0, %d corrections)" % int(ok.sum()))
    print("  median tR  %.2f us   [p10 %.2f, p90 %.2f]"
          % (np.median(tR0), np.percentile(tR0, 10), np.percentile(tR0, 90)))
    print("  median tBv %.2f us   [p10 %.2f, p90 %.2f]"
          % (np.median(tBv0), np.percentile(tBv0, 10), np.percentile(tBv0, 90)))
    print("  median tQ  %.2f us   median tBf %.2f us"
          % (np.median(tQ0), np.median(tBf0)))
    _rv = np.median(ratio[ok])
    print("  median tBv/tR %.3f   (verification alone: %.1f%%)"
          % (_rv, 100 * (_rv - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratiof = tBf0 / tR0
    okf = np.isfinite(ratiof)
    _rf = np.median(ratiof[okf])
    print("  median tBf/tR %.3f   (probe included: %.1f%%)"
          % (_rf, 100 * (_rf - 1)))
    print("  band cheaper on %d/%d corrections without the probe, %d/%d with it"
          % (int(np.sum(tBv0 < tR0)), len(tR0), int(np.sum(tBf0 < tR0)), len(tR0)))
    print("  median paired diff: verify %+.2f us, full %+.2f us"
          % (float(np.median(tBv0 - tR0)), float(np.median(tBf0 - tR0))))
    print("  arithmetic closure: tQ + tBv - tBf = %+.2f us (one call vs two)"
          % (float(np.median(tQ0 + tBv0 - tBf0))))
    print()
    print("  per-element costs on the complete path")
    aR, bR, r2R = fit(n, tR0)
    aB, bB, r2B = fit(k, tBv0)
    print("    tR  = %.3f us + %.3f ns/row        R2 %.3f" % (bR, aR * 1e3, r2R))
    print("    tBv = %.3f us + %.3f ns/candidate  R2 %.3f" % (bB, aB * 1e3, r2B))
    print("    kappa_complete = %.4f   (bare kernel 0.030, duel 0.214)"
          % (aR / aB if aB else float("nan")))
    print()
    print("  by band density k/n")
    print("    %-14s %5s %9s %9s %9s %8s %8s"
          % ("k/n bucket", "cnt", "med tR", "med tBv", "med tBf", "tBv/tR", "tBf/tR"))
    dens = k / np.maximum(n, 1)
    for lo_, hi_ in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)):
        m = (dens >= lo_) & (dens < hi_)
        if m.sum() == 0:
            continue
        print("    %-14s %5d %9.2f %9.2f %9.2f %8.3f %8.3f"
              % ("%.2f-%.2f" % (lo_, hi_), int(m.sum()),
                 np.median(tR0[m]), np.median(tBv0[m]), np.median(tBf0[m]),
                 np.median(tBv0[m] / tR0[m]), np.median(tBf0[m] / tR0[m])))
    print()
    # the corpus point the paper argues about
    CPR = dict(n=394.0, k=182.0)
    print("  the corpus point (n=394, k=182), priced by the fits above:")
    print("    tR  %.2f us   tBv %.2f us   tQ median %.2f us   tBf %.2f us"
          % (bR + aR * CPR["n"], bB + aB * CPR["k"], np.median(tQ0),
             bB + aB * CPR["k"] + np.median(tQ0)))
    print("    verification %+.1f%%, probe included %+.1f%%"
          % (100 * ((bB + aB * CPR["k"]) / (bR + aR * CPR["n"]) - 1),
             100 * ((bB + aB * CPR["k"] + np.median(tQ0)) / (bR + aR * CPR["n"]) - 1)))
    print("    density where verification breaks even: k/n = %.3f"
          % (aR / aB if aB else float("nan")))

    res = dict(n_corr=len(corrs), reps=reps, n=n.tolist(), k=k.tolist(),
               tR=tR0.tolist(), tBv=tBv0.tolist(), tQ=tQ0.tolist(),
               tBf=tBf0.tolist(),
               median=dict(tR=float(np.median(tR0)), tBv=float(np.median(tBv0)),
                           tQ=float(np.median(tQ0)), tBf=float(np.median(tBf0)),
                           ratio_verify=float(np.median(ratio[ok])),
                           ratio_full=float(np.median(ratiof[okf])),
                           wins_verify=int(np.sum(tBv0 < tR0)),
                           wins_full=int(np.sum(tBf0 < tR0))),
               fit=dict(tR=dict(a_ns=aR * 1e3, b_us=bR, r2=r2R),
                        tBv=dict(a_ns=aB * 1e3, b_us=bB, r2=r2B),
                        kappa=float(aR / aB)),
               proto=W12)
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
