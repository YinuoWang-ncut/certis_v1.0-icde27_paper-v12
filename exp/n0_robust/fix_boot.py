#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-do the bootstrap in n0_robust.json with the correct estimator.

The original bootstrap drew ONE repetition per point (rng.choice(r) returns a
scalar, so .min() on it is a no-op) and fitted that.  The point estimate fits
the *minimum* of the repetitions, so the resample has to resample the
repetitions and take the minimum of the resample; otherwise the resampled curve
sits above the fitted curve and the slope ratio moves 20-40% (sqlite 0.157 ->
0.090), which is larger than any interval it was meant to justify.

Nothing is re-measured: every repetition is in the JSON.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "n0_robust.json"), encoding="utf-8"))
N, K = 394.0, 182.0
RNG = np.random.default_rng(11)
NB = 4000


def slopes(pts, axis):
    x = np.asarray([p[axis] for p in pts], float)
    return np.polyfit(x, np.asarray([p["ns"] for p in pts], float), 1)


def boot(row):
    R, B = row["points_R"], row["points_B"]
    xR = np.asarray([p["n"] for p in R], float)
    xB = np.asarray([p["k"] for p in B], float)
    repsR = [np.asarray(p["reps"], float) for p in R]
    repsB = [np.asarray(p["reps"], float) for p in B]
    fR = np.asarray(row["fixed_R_rounds"], float)
    fB = np.asarray(row["fixed_B_rounds"], float)
    rk = np.asarray(row.get("rank_rounds") or [row["rank_ns"]], float)

    out = []
    nR, nB = len(R), len(B)
    for _ in range(NB):
        iR = RNG.integers(0, nR, nR)          # resample sweep points too, or the
        iB = RNG.integers(0, nB, nB)          # interval is only repetition noise
        yR = [RNG.choice(repsR[i], size=repsR[i].size, replace=True).min() for i in iR]
        yB = [RNG.choice(repsB[i], size=repsB[i].size, replace=True).min() for i in iB]
        if min(yR) <= 0 or min(yB) <= 0:
            continue
        cR = np.polyfit(xR[iR], yR, 1)[0]
        cB = np.polyfit(xB[iB], yB, 1)[0]
        FR = RNG.choice(fR, size=fR.size, replace=True).min()
        FB = RNG.choice(fB, size=fB.size, replace=True).min()
        RK = RNG.choice(rk, size=rk.size, replace=True).min()
        if cR <= 0 or cB <= 0:
            continue
        kp = cR / cB
        out.append((kp, (FB - FR) / cR, (FB + RK - FR) / cR,
                    kp * (N - (FB - FR) / cR), kp * (N - (FB + RK - FR) / cR)))
    a = np.asarray(out)
    q = lambda j: [round(float(np.percentile(a[:, j], 5)), 4),
                   round(float(np.percentile(a[:, j], 95)), 4)]
    return {"kappa_ci": q(0), "n0_ci": q(1), "n0eff_ci": q(2),
            "kstar_ci": q(3), "kstare_ci": q(4),
            "kappa_med": round(float(np.median(a[:, 0])), 4),
            "n0_med": round(float(np.median(a[:, 1])), 1),
            "n0eff_med": round(float(np.median(a[:, 2])), 1)}


print("%-13s %7s %7s %16s %8s %8s %8s %8s %6s" % (
    "backend", "kappa", "nsd", "n0 [90%]", "k*", "k*(pr)", "flip_n0", "d/n", "verdict"))
res = []
for r in D["results"]:
    if not r.get("included", True):
        continue
    b = boot(r)
    cR = slopes(r["points_R"], "n")[0]
    cB = slopes(r["points_B"], "k")[0]
    kap = cR / cB
    # the intercepts are the *measured* empty-call costs, not the regression's
    # x=0 extrapolation, which is unstable over n = 512..65536
    FR, FB = r["F_R_ns"], r["F_B_ns"]
    n0 = (FB - FR) / cR
    n0e = (FB + r["rank_ns"] - FR) / cR
    kstar, kstare = kap * (N - n0), kap * (N - n0e)
    flip = N - K / kap
    win = kstare > K
    res.append(dict(backend=r["backend"], kappa=round(float(kap), 4), n0=round(float(n0), 1),
                    n0_eff=round(float(n0e), 1), kstar=round(float(kstar), 1),
                    kstar_eff=round(float(kstare), 1), flip_n0=round(float(flip), 1),
                    drop=round(float((flip - n0e) / N), 3), win=bool(win), **b))
    print("%-13s %7.4f %7.4f %8.1f [%6.1f,%6.1f] %8.1f %8.1f %8.1f %8.2f %6s" % (
        r["backend"], kap, b["kappa_med"], n0, b["n0_ci"][0], b["n0_ci"][1],
        kstar, kstare, flip, (flip - n0e) / N, "WIN" if win else "lose"))

w = sum(1 for x in res if x["win"])
print("\n%s of %d lose at (n=%d, k=%d); kappa above k/n=%.3f wins, below loses"
      % (len(res) - w, len(res), N, K, K / N))
for x in sorted(res, key=lambda x: -x["kappa"]):
    print("  %-13s kappa=%.3f  %s  (k/n = %.3f)" % (
        x["backend"], x["kappa"], "WIN " if x["win"] else "lose", K / N))

json.dump({"corpus": {"n": N, "k": K}, "configs": res},
          open(os.path.join(HERE, "n0_robust_fix.json"), "w", encoding="utf-8"),
          indent=1)
print("\nwrote n0_robust_fix.json")
