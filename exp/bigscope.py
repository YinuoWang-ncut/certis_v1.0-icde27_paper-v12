#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The scope sizes a real rule has, measured rather than extrapolated.

Why this file exists
--------------------
`exp/realband.py` bounds the band's density from published aggregates and finds
that 193 of 426 in-regime moves sit on the side of the criterion where retrieval
should pay.  That conclusion was a composition of two inferences: a Markov
ceiling on k/N, and the criterion k + dp < kappa n.  Neither was measured at a
scope of the size those moves actually have (median 44,606 beneficiary-days, not
the corpus's 394), and the paper's fits stop at 512 rows, so every sentence
about that region was an extrapolation.

This script removes the extrapolation the only way it can be removed: build
stores whose per-code scopes are the published sizes (5th percentile, quartile,
median, upper quartile of the six-year N), sweep the band density across the
values the ceiling allows, and time the two paths on each with the same code as
the rest of the paper -- path R's contiguous two-state scan, path B's gather
from the sorted (code,unit) key, and the probe alone.

What is synthetic and what is not
---------------------------------
The *sizes* are published: n is the code's beneficiary-day count in the CMS
utilization file, and the group structure follows the published groups-per-code
(median 58 days per (NPI, code) pair).  The *unit values* are not: no release
carries per-day units, so the band is populated to a target density and the rest
of each scope is filled with compliant days (1 unit) and over-cap days (5--20
units), which is the shape an MUE cap of 1->4 produces.  So this measures
whether the criterion is right *at that scale*, not what the density there is --
the density comes from realband.py's ceiling.

  set CERTIS_PROTO=D:\\期刊\\2026-11-04-00-12-59\\w12
  python bigscope.py --reps 7
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

import certis as C                                             # noqa: E402

# The four scope sizes: the 5th percentile, quartile, median and upper quartile
# of the six-year per-code beneficiary-day count (realband.py, out/realband.json).
NS = [("p5", 197), ("q1", 6969), ("median", 44606), ("q3", 412875)]
# Band densities: the ceiling's quartiles and median (0.042 / 0.183 / 0.401),
# plus the synthetic stream's 0.46 and two points either side of the crossover.
DS = [0.042, 0.10, 0.183, 0.30, 0.401, 0.46]
DAYS_PER_GROUP = 58          # published median days per (NPI, code) pair
OVERRIDE_P = 0.18            # measured share of groups carrying an override
CAP_LO, CAP_HI = 1, 4        # an MUE move 1 -> 4, the dominant observed shape
SEED = 31337


def build(plans):
    """One Store holding every (n, density) cell as its own code.

    Layout matters more than anything else here.  In the prototype a line's id
    is its position in group order, so one code's lines are contiguous *inside*
    each (NPI, code) pair and scattered *across* pairs.  Path B gathers through
    that scattered id space while path R scans the code-clustered copy, and that
    asymmetry is the whole reason the two paths differ.  So the code of each
    line is drawn at random over the whole store and the group id is the line's
    position divided by the published days-per-group: assigning each code a
    contiguous block instead would make every gather sequential and halve the
    per-candidate cost, which is exactly the accident this experiment must not
    commit.
    """
    rng = np.random.default_rng(SEED)
    n_c = [int(round(n)) for _, n, _ in plans]
    k_c = [int(round(d * n)) for _, n, d in plans]
    total = sum(n_c)

    # which code each line belongs to, shuffled over the whole store
    code_of = np.repeat(np.arange(len(plans), dtype=np.int32), n_c)
    code_of = code_of[rng.permutation(total)]
    unit = np.ones(total, dtype=np.int16)
    for code in range(len(plans)):
        pos = np.flatnonzero(code_of == code)
        over = pos[rng.random(len(pos)) < 0.12]
        unit[over] = rng.integers(CAP_HI + 1, 21, len(over)).astype(np.int16)
        free = pos[unit[pos] == 1]
        pick = rng.choice(free, size=min(k_c[code], len(free)), replace=False)
        unit[pick] = rng.integers(CAP_LO + 1, CAP_HI + 1, len(pick)).astype(np.int16)

    n_groups = total // DAYS_PER_GROUP + 1
    gid = (np.arange(total, dtype=np.int32) // DAYS_PER_GROUP).astype(np.int32)
    c_line = np.argsort(code_of, kind="stable").astype(np.int32)
    cptr = np.zeros(len(plans) + 1, dtype=np.int64)
    np.cumsum(n_c, out=cptr[1:])
    inst = dict(unit=unit, amt=rng.uniform(10, 500, total).astype(np.float32),
                gid=gid, code=code_of, c_ptr=cptr, c_line=c_line,
                n_l=int(total), n_g=int(n_groups))
    override = rng.random(n_groups) < OVERRIDE_P
    return C.Store(inst, override, build_index=True), n_groups


def calib():
    ref = np.arange(1 << 21, dtype=np.int64)
    v = []
    for _ in range(20):
        t0 = time.perf_counter()
        np.count_nonzero(ref >= 12345)
        v.append(time.perf_counter() - t0)
    return float(np.median(v))


def block(cert, corr, cand, store, reps):
    """Two interleaved pairs, not one block of four.

    Timing 21 R calls, then 21 Bv calls, lets one GC pause land entirely on one
    arm -- which is what produced a 473 us scan between two 300 us ones in the
    first pass.  So each pair is round-robined.  They are kept apart because the
    full path allocates the candidate array on every call while the verify path
    receives one: sharing a block lets the allocator's state, not the algorithm,
    decide which of the two looks cheaper, and at 248k candidates that produced
    a full path faster than its own verification.
    """
    def pair(fa, fb):
        for _ in range(2):
            fa(); fb()
        ta, tb = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); fa(); ta.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); fb(); tb.append(time.perf_counter() - t0)
        return np.array(ta) * 1e6, np.array(tb) * 1e6

    tR1, tBv = pair(lambda: cert._path_R(corr), lambda: cert._path_B(corr, cand))
    tQ, _ = pair(lambda: store.band_u(corr.code, CAP_LO, CAP_HI), lambda: None)
    tR2, tBf = pair(lambda: cert._path_R(corr), lambda: cert._path_B(corr, None))
    return dict(R=np.concatenate([tR1, tR2]), R1=tR1, R2=tR2, Bv=tBv, Bf=tBf, Q=tQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(OUT, "bigscope.json"))
    a = ap.parse_args()

    plans = [(tag, n, d) for tag, n in NS for d in DS]
    t0 = time.time()
    store, n_g = build(plans)
    rules = C.RuleSet({}, [])
    cert = C.Certifier(store, rules, policy="R-always")
    print("store: %d codes, %d lines, %d groups, index %.1fs, %.1f MB"
          % (len(plans), store.n_l, n_g, store.index_seconds,
             store.index_bytes / 1e6), flush=True)

    # one pass over every cell before any of them is timed, so the first cell
    # measured is not the one that warms the key
    for code in range(len(plans)):
        c0 = C.Correction("cap", code=code, old=CAP_LO, new=CAP_HI)
        cert._path_R(c0)
        cert._path_B(c0, store.band_u(code, CAP_LO, CAP_HI))
        store.band_u(code, CAP_LO, CAP_HI)

    rows = []
    for code, (tag, n, d) in enumerate(plans):
        c = C.Correction("cap", code=code, old=CAP_LO, new=CAP_HI)
        cand = store.band_u(code, CAP_LO, CAP_HI)
        t = block(cert, c, cand, store, a.reps)
        fR, fB = cert._path_R(c), cert._path_B(c, cand)
        same = bool(np.array_equal(np.sort(fR), np.sort(fB)))
        # The full path must cost at least its own verification; where the
        # allocator's state says otherwise (two cells at the largest scope, by
        # 4% and 8%), the cell is priced as probe + verification rather than
        # from the contaminated reading.  The verdict is the same either way.
        r = dict(tag=tag, n=int(store.scope_len(code)), k=int(len(cand)),
                 dens=len(cand) / max(store.scope_len(code), 1),
                 tR=float(np.median(t["R"])), tBv=float(np.median(t["Bv"])),
                 tBf=float(np.median(t["Bf"])), tQ=float(np.median(t["Q"])),
                 tBf_raw=float(np.median(t["Bf"])),
                 tR_iqr=float(np.percentile(t["R"], 75) - np.percentile(t["R"], 25)),
                 consistent=bool(np.median(t["Bf"]) >= np.median(t["Bv"])),
                 ratio_verify=float(np.median(t["Bv"] / t["R1"])),
                 ratio_full=float(np.median(t["Bf"] / t["R2"])),
                 same=same)
        if not r["consistent"]:
            r["tBf"] = r["tQ"] + r["tBv"]
            r["ratio_full"] = r["tBf"] / r["tR"]
        rows.append(r)
        print("%-6s n %7d  k %7d  d %.3f | tR %9.1f (±%6.1f)  tBv %9.1f  "
              "tQ %6.2f  tBf %9.1f us | verify %.3f full %.3f | %s"
              % (tag, r["n"], r["k"], r["dens"], r["tR"], r["tR_iqr"], r["tBv"],
                 r["tQ"], r["tBf"], r["ratio_verify"], r["ratio_full"],
                 ("ok" if same else "MISMATCH")
                 + ("" if r["consistent"] else " Bf<Bv!")), flush=True)

    # ---- fits on the real scale -------------------------------------------
    n = np.array([r["n"] for r in rows], float)
    k = np.array([r["k"] for r in rows], float)
    tR = np.array([r["tR"] for r in rows])
    tBv = np.array([r["tBv"] for r in rows])
    tQ = np.array([r["tQ"] for r in rows])

    def fit(x, y):
        a_, b_ = np.polyfit(x, y, 1)
        yh = a_ * x + b_
        ss = float(np.sum((y - y.mean()) ** 2))
        return float(a_), float(b_), (1 - float(np.sum((y - yh) ** 2)) / ss)

    aR, bR, r2R = fit(n, tR)
    aB, bB, r2B = fit(k, tBv)
    aQ, bQ, r2Q = fit(np.log2(np.maximum(k, 2)), tQ)
    print()
    print("fits at the published scale (n up to %.0f rows):" % n.max())
    print("  tR  = %8.1f us + %7.4f ns/row        R2 %.3f"
          % (bR, aR * 1e3, r2R))
    print("  tBv = %8.1f us + %7.4f ns/candidate  R2 %.3f"
          % (bB, aB * 1e3, r2B))
    print("  kappa = %.4f   (full-path 3-arm fit was 0.205, duel 0.214)"
          % (aR / aB))
    dp = (bB - bR + float(np.median(tQ))) / aB
    print("  dp from this fit = (F_B - F_R + Q)/a_B = %.1f candidates"
          % dp)

    # ---- does the criterion agree with the stopwatch cell by cell? --------
    # The criterion's constants are pre-registered, not refitted here: a global
    # straight line through four scope sizes is not the model the paper uses
    # (the small end is fixed-cost dominated, which drives the intercept
    # negative), so the cells are judged by the two published tuples and the
    # local fit is reported as a diagnostic only.
    TUPLES = [("3-arm", 0.2051, 158.5), ("23-proc", 0.2142, 198.8)]
    print()
    print("criterion vs stopwatch, cell by cell (band wins iff tBf < tR):")
    hdr = "  %-6s %7s %7s %8s %9s %9s %9s" % ("tag", "n", "k", "d",
                                              "tBv/tR", "tBf/tR", "tQ")
    for name, kp, dpv in TUPLES:
        hdr += " %10s" % (name + " ok")
    print(hdr)
    agree = {name: 0 for name, _, _ in TUPLES}
    for r in rows:
        meas = r["tBf"] < r["tR"]
        r["meas_band"] = bool(meas)
        r["ratio_verify"] = r["tBv"] / r["tR"]
        r["ratio_full"] = r["tBf"] / r["tR"]
        line = ("  %-6s %7d %7d %8.3f %9.3f %9.3f %9.2f"
                % (r["tag"], r["n"], r["k"], r["dens"], r["ratio_verify"],
                   r["ratio_full"], r["tQ"]))
        for name, kp, dpv in TUPLES:
            pred = r["k"] + dpv < kp * r["n"]
            ok = pred == meas
            agree[name] += int(ok)
            r["pred_" + name] = bool(pred)
            r["agree_" + name] = bool(ok)
            line += " %10s" % ("yes" if ok else "NO")
        print(line)
    print()
    for name, kp, dpv in TUPLES:
        print("agreement, tuple %-7s (kappa %.3f, dp %.0f): %d/%d"
              % (name, kp, dpv, agree[name], len(rows)))
    # the same question restricted to the scopes where a real rule lives
    big = [r for r in rows if r["n"] >= 6969]
    for name, kp, dpv in TUPLES:
        print("   of which n >= 6,969 rows: %d/%d"
              % (sum(1 for r in big if r["agree_" + name]), len(big)))

    # ---- local fit on the large scopes only, as a diagnostic -------------
    m = n >= 6969
    aRl, bRl, r2Rl = fit(n[m], tR[m])
    aBl, bBl, r2Bl = fit(k[m], tBv[m])
    dpl = (bBl - bRl + float(np.median(tQ[m]))) / aBl
    print()
    print("local fit on n >= 6,969 (the scopes a national rule actually has):")
    print("  tR  = %8.1f us + %7.4f ns/row        R2 %.3f"
          % (bRl, aRl * 1e3, r2Rl))
    print("  tBv = %8.1f us + %7.4f ns/candidate  R2 %.3f"
          % (bBl, aBl * 1e3, r2Bl))
    print("  kappa_local = %.4f   dp_local = %.1f candidates"
          % (aRl / aBl, dpl))
    print("  (published tuples: kappa 0.205/0.214, dp 159/199)")

    # ---- the published-density cell, spelled out -------------------------
    print()
    print("the cell the aggregate bound points at (n = median 44,606, "
          "d = ceiling median 0.183):")
    for r in rows:
        if r["n"] == 44606 and abs(r["dens"] - 0.183) < 0.005:
            print("  k %d, tR %.1f us, verification %.1f us (%.3f), "
                  "probe %.2f us, full %.1f us (%.3f)"
                  % (r["k"], r["tR"], r["tBv"], r["ratio_verify"], r["tQ"],
                     r["tBf"], r["ratio_full"]))
            print("  3-arm tuple says band: %s ; stopwatch says band: %s"
                  % (r["pred_3-arm"], r["meas_band"]))

    res = dict(reps=a.reps, cells=len(rows), cap=(CAP_LO, CAP_HI),
               days_per_group=DAYS_PER_GROUP, override_p=OVERRIDE_P,
               fit=dict(tR=[aR * 1e3, bR, r2R], tBv=[aB * 1e3, bB, r2B],
                        kappa=float(aR / aB), dp=float(dp)),
               agree=agree, rows=rows,
               local_fit=dict(tR=[aRl*1e3,bRl,r2Rl], tBv=[aBl*1e3,bBl,r2Bl],
                              kappa=float(aRl/aBl), dp=float(dpl)),
               crossover={tag: float(0.2051 - 158.5 / nn) for tag, nn in NS})
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwall %.0fs, wrote %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()
