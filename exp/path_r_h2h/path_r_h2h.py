#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path R and band retrieval, timed on the instance Table V uses.

Table V divides a differential-dataflow measurement by path R's *predicted*
19.7 us at 436 rows, because path R, and the band, had never been timed on
that instance.  A prediction in a denominator is a reference, not a result,
and the caption had to say so.  This closes it: same instance, same code, same
box, both arms.

Instance.  exp/ivm_h2h/mkdata.py -- rows 2e6, groups 5410, seed 20261105,
amt = round(lognormal(6,1),2), gid = rng.integers(0,5410).  The largest group
(4007) is the scope at 436 rows; theta = 1.5*mean(amt) = 997.97, and the five
placements reproduce ivm_h2h's k = 0/2/20/52/69.

Two cache states, because the two differ by more than the arms do and mixing
them is how a denominator gets chosen for you:
  warm  the same scope, 450 corrections -- what a stream that keeps hitting one
        rule looks like, and the state the differential-dataflow arm is in
  cold  a different scope per correction, 90 of them -- what P7 measures as the
        cold cell, and the state a stream that spreads across rules is in

The `unit` axis is a monotone quantisation of `amt` into int16, matching the
corpus store's unit_c dtype; the flip set is unchanged and the scope length,
which is what sets the cost, is untouched.

  CERTIS_PROTO=... python path_r_h2h.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U, G, LAM, SEED = 2_000_000, 5410, 1.5, 20261105
REPS, RUNS = 30, 3


def p50(a):
    return float(np.percentile(a, 50))


def main():
    rng = np.random.default_rng(SEED)                    # mkdata's own order
    amt = np.round(rng.lognormal(6.0, 1.0, U), 2).astype(np.float64)
    gid = rng.integers(0, G, U).astype(np.int32)

    # CERTIS_PROTO overrides; without it the tree ships the core at its
    # own root, so the script runs from a clean checkout with no setup.
    proto = os.environ.get("CERTIS_PROTO") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, proto)
    import certis as C

    counts = np.bincount(gid, minlength=G).astype(np.int64)
    order = np.argsort(gid, kind="stable")
    cptr = np.zeros(G + 1, dtype=np.int64)
    cptr[1:] = np.cumsum(counts)

    amt_max = float(amt.max())
    q = lambda x: int(np.clip(np.rint(x / amt_max * 30000.0), 0, 32000))
    unit = np.clip(np.rint(amt / amt_max * 30000.0), 0, 32000).astype(np.int16)
    override = np.random.default_rng(SEED + 1).random(G) < 0.18

    inst = dict(unit=unit, amt=amt, gid=gid, code=gid, c_ptr=cptr,
                c_line=order.astype(np.int32), n_l=U, n_g=G)
    store = C.Store(inst, override, build_index=True)

    gmax = int(np.argmax(counts))
    n_scope = int(counts[gmax])
    theta0 = float(LAM * float(amt.mean()))
    s = amt[store.cline[cptr[gmax]:cptr[gmax + 1]]]

    places = [("down1.01", theta0 * 0.99, theta0),
              ("up1.02", theta0, theta0 * 1.02),
              ("up1.10", theta0, theta0 * 1.10),
              ("up1.50", theta0, theta0 * 1.50),
              ("up2.00", theta0, theta0 * 2.00)]
    plan = []
    for name, lo, hi in places:
        k = int(((s >= lo) & (s < hi)).sum())
        plan.append((name, q(lo), q(hi), k))

    cert = C.Certifier(store, None, policy="R-always")

    def cell(code, ca, cb, band=False):
        c = C.Correction("cap", code=code, old=ca, new=cb)
        t0 = time.perf_counter()
        f = cert._path_B(c) if band else cert._path_R(c)
        return (time.perf_counter() - t0) * 1e6, len(f)

    # ---- warm: one scope, ABBA inside each rep
    for _ in range(300):
        cell(gmax, plan[0][1], plan[0][2])
    warm_r, warm_b, per_k = [], [], {p[0]: [] for p in plan}
    flips_r, flips_b = {}, {}
    for run in range(RUNS):
        for rep in range(REPS):
            seq = plan if rep % 2 == 0 else list(reversed(plan))
            for name, ca, cb, k in seq:
                dt, nf = cell(gmax, ca, cb, band=False)
                warm_r.append(dt); per_k[name].append(dt); flips_r[name] = nf
                dt, nf = cell(gmax, ca, cb, band=True)
                warm_b.append(dt); flips_b[name] = nf

    # ---- cold: a fresh scope per correction, walked across the group space
    import random
    codes = list(range(G))
    random.Random(12345).shuffle(codes)
    cands = [c for c in codes if 340 <= counts[c] <= 470][:120]
    cold_r, cold_b, cold_n = [], [], []
    for i, code in enumerate(cands):
        name, ca, cb, k = plan[i % len(plan)]
        v = store.unit_c[cptr[code]:cptr[code + 1]]
        ca, cb = q(theta0), q(theta0 * (1.0 + 0.02 * (i % 3)))
        dt, _ = cell(code, ca, cb, band=False); cold_r.append(dt); cold_n.append(int(counts[code]))
        dt, _ = cell(code, ca, cb, band=True); cold_b.append(dt)

    def summ(v):
        return dict(p50_us=round(p50(v), 3), mean_us=round(float(np.mean(v)), 3),
                    sd_us=round(float(np.std(v, ddof=1)), 3),
                    min_us=round(float(np.min(v)), 3),
                    max_us=round(float(np.max(v)), 3), n=len(v))

    res = dict(
        instance=dict(rows=U, groups=G, scope_rows=n_scope, scope_code=gmax,
                      theta=round(theta0, 3), mean_amt=round(float(amt.mean()), 4),
                      scope_range=[int(counts.min()), int(counts.max())],
                      unit_dtype="int16"),
        design=dict(reps_per_point=REPS, runs=RUNS, warm_corrections=len(warm_r),
                    cold_corrections=len(cold_r), order="ABBA per rep (warm)"),
        path_r=dict(warm=summ(warm_r), cold=summ(cold_r)),
        band=dict(warm=summ(warm_b), cold=summ(cold_b)),
        per_placement={name: dict(k=int([p for p in plan if p[0] == name][0][3]),
                                  flips=int(flips_r[name]),
                                  band_flips=int(flips_b[name]),
                                  p50_us=round(p50(v), 3), n=len(v))
                       for name, v in per_k.items()},
        cold_scopes=dict(rows_min=int(min(cold_n)), rows_max=int(max(cold_n)),
                         rows_p50=int(np.percentile(cold_n, 50))),
    )
    res["prediction"] = dict(was_us=19.7,
                             warm_over_pred=round(res["path_r"]["warm"]["p50_us"] / 19.7, 3))
    print(json.dumps(res, indent=1))
    out = os.path.join(HERE, "path_r_h2h.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
