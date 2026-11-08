#!/usr/bin/env python
"""Interval for kappa, from the paired per-correction records of W12.

SUPERSEDED as the paper's error bar (kept for the record).  Everything below is
correct arithmetic on the wrong question: it resamples the 300 corrections of one
run, so it sees how well those scopes pin two slopes down and never sees the
error that dominates -- the large-|B| gather moving between 53 and 88 us across
processes.  Seven re-runs of the same stream (exp/run_duel_big.py) put kappa at
0.26-0.46, i.e. 2.5x wider than the interval computed here and not centred on it.
The paper prints the span; exp/kappa_runs.json holds it.

`exp/boundary.json` reports kappa = 0.4414 as a bare ratio of two fitted slopes.
A ratio of two slopes has a sampling distribution, and the paper's decisive
comparison is kappa against the corpus density k/|sigma| = 182/394 = 0.462,
where the margin is only 4.5%. So the interval is not decoration: if it straddles
0.462 the sign of the comparison at Delta_p = 0 is not settled by these data.

Input: ../../w12/w12_results.json, key `duel_cell.duel`. Every record is one
correction executed on *both* paths:

    {n_scope, n_band, tR, tB, ...}

300 corrections. The fit is the one in run_w12.py:fit_two_param:

    t_R = c_R * |sigma| + F_R        t_B = c_B * |B| + F_B
    kappa = c_R / c_B                n_0 = (F_B - F_R) / c_R

Three resamplings, because the error that matters here is not one thing:

  paired    resample the 300 (x_R, y_R, x_B, y_B) tuples as units.  The two
            slopes are estimated on the same corrections and are therefore
            correlated; resampling the tuple keeps that correlation, which
            narrows the ratio.  This is the primary interval.
  block     resample the 30 batches of 10 instead of the 10 corrections.
            Corrections inside a batch share the process, the page cache and
            the clock state, so residuals are not independent within a batch.
            Wider than `paired` by construction; reported as the honest ceiling.
  residual  keep x fixed, resample residuals within each path and refit.  This
            asks what the timing noise alone does to the ratio, holding the
            corpus of scopes fixed.

Output: exp/bootstrap_kappa.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("CERTIS_W12")
if not SRC:
    raise SystemExit(
        "this script resamples the 300 per-correction records of one W12\n"
        "duel, which the artifact does not ship: set CERTIS_W12 to that\n"
        "w12_results.json.  The interval it produces is superseded -- it\n"
        "cannot see the process-to-process drift that dominates -- and is\n"
        "kept only as the record; the paper's error bar is the seven-run\n"
        "span in exp/duel_big/, reduced by exp/fit_kappa_runs.py.")
OUT = os.path.join(HERE, "bootstrap_kappa.json")

N_BOOT = 20000
BATCH = 10
RHO = 182.0 / 394.0          # the corpus density k/|sigma| the band must beat
SEED = 20261105


def fit(xr, yr, xb, yb):
    ar, br = np.polyfit(xr, yr, 1)
    ab, bb = np.polyfit(xb, yb, 1)
    return ar, br, ab, bb


def stats(aR, bR, aB, bB):
    kappa = aR / aB
    n0 = (bB - bR) / aR
    dp = (bB - bR) / aB                    # Delta_p, in candidates
    allow = kappa * 394.0 - 182.0          # kappa*n - k, in rows
    return kappa, n0, dp, allow


def summarize(v, lo=5.0, hi=95.0):
    v = np.asarray(v, float)
    return {
        "median": float(np.median(v)),
        "p%d" % lo: float(np.percentile(v, lo)),
        "p%d" % hi: float(np.percentile(v, hi)),
        "p2.5": float(np.percentile(v, 2.5)),
        "p97.5": float(np.percentile(v, 97.5)),
        "sd": float(v.std(ddof=1)),
    }


def main():
    D = json.load(open(SRC, encoding="utf-8"))
    du = D["duel_cell"]["duel"]
    xr = np.array([d["n_scope"] for d in du], float)
    xb = np.array([d["n_band"] for d in du], float)
    yr = np.array([d["tR"] for d in du], float)
    yb = np.array([d["tB"] for d in du], float)

    aR, bR, aB, bB = fit(xr, yr, xb, yb)
    kap, n0, dp, allow = stats(aR, bR, aB, bB)
    print("point  c_R=%.2f ns  F_R=%.2f us  c_B=%.2f ns  F_B=%.2f us" %
          (aR * 1e9, bR * 1e6, aB * 1e9, bB * 1e6))
    print("point  kappa=%.4f  n0=%.1f rows  Delta_p=%.1f  allowance=%.1f rows" %
          (kap, n0, dp, allow))
    print("point  margin over rho=%.4f: %.2f%%" % (RHO, 100 * (kap / RHO - 1)))

    # analytic OLS standard errors, for a sanity check on the resampling
    def slope_se(x, y):
        n = len(x)
        p, *_ = np.polyfit(x, y, 1)
        resid = y - np.polyval([p, np.polyfit(x, y, 1)[1]], x)
        s2 = resid.dot(resid) / (n - 2)
        return float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))

    seR, seB = slope_se(xr, yr), slope_se(xb, yb)
    r2R = np.corrcoef(xr, yr)[0, 1] ** 2
    r2B = np.corrcoef(xb, yb)[0, 1] ** 2
    print("slopes c_R=%.3f+-%.3f (R2=%.4f)  c_B=%.3f+-%.3f (R2=%.4f)" %
          (aR * 1e9, seR * 1e9, r2R, aB * 1e9, seB * 1e9, r2B))

    rng = np.random.default_rng(SEED)
    n = len(du)
    out = {
        "source": os.path.relpath(SRC, os.path.dirname(HERE)).replace("\\", "/"),
        "n_corrections": int(n),
        "n_batches": int(n // BATCH),
        "rho": RHO,
        "point": {"c_R_ns": aR * 1e9, "F_R_us": bR * 1e6,
                  "c_B_ns": aB * 1e9, "F_B_us": bB * 1e6,
                  "kappa": kap, "n_0_rows": n0, "delta_p": dp,
                  "allowance_rows": allow,
                  "se_c_R_ns": seR * 1e9, "se_c_B_ns": seB * 1e9,
                  "r2_R": r2R, "r2_B": r2B},
        "resamplings": {},
    }

    # 1. paired i.i.d. over corrections
    k1, n1, d1, a1 = [], [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        try:
            s = stats(*fit(xr[i], yr[i], xb[i], yb[i]))
        except Exception:
            continue
        k1.append(s[0]); n1.append(s[1]); d1.append(s[2]); a1.append(s[3])
    out["resamplings"]["paired"] = {
        "kappa": summarize(k1), "n_0": summarize(n1),
        "delta_p": summarize(d1), "allowance_rows": summarize(a1),
        "p_kappa_above_rho": float(np.mean(np.asarray(k1) > RHO)),
    }

    # 2. block bootstrap over batches
    nb = n // BATCH
    blocks = [np.arange(b * BATCH, (b + 1) * BATCH) for b in range(nb)]
    k2, n2, d2, a2 = [], [], [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, nb, nb)
        i = np.concatenate([blocks[b] for b in pick])
        try:
            s = stats(*fit(xr[i], yr[i], xb[i], yb[i]))
        except Exception:
            continue
        k2.append(s[0]); n2.append(s[1]); d2.append(s[2]); a2.append(s[3])
    out["resamplings"]["block_batch"] = {
        "kappa": summarize(k2), "n_0": summarize(n2),
        "delta_p": summarize(d2), "allowance_rows": summarize(a2),
        "p_kappa_above_rho": float(np.mean(np.asarray(k2) > RHO)),
    }

    # 3. residual bootstrap, x held fixed
    fr = np.polyval([aR, bR], xr)
    fb = np.polyval([aB, bB], xb)
    rr, rb = yr - fr, yb - fb
    k3, n3, d3, a3 = [], [], [], []
    for _ in range(N_BOOT):
        yr2 = fr + rng.choice(rr, n, replace=True)
        yb2 = fb + rng.choice(rb, n, replace=True)
        try:
            s = stats(*fit(xr, yr2, xb, yb2))
        except Exception:
            continue
        k3.append(s[0]); n3.append(s[1]); d3.append(s[2]); a3.append(s[3])
    out["resamplings"]["residual"] = {
        "kappa": summarize(k3), "n_0": summarize(n3),
        "delta_p": summarize(d3), "allowance_rows": summarize(a3),
        "p_kappa_above_rho": float(np.mean(np.asarray(k3) > RHO)),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print()
    for name in ("paired", "block_batch", "residual"):
        r = out["resamplings"][name]
        kk = r["kappa"]
        print("%-11s kappa %.4f  [90%%: %.4f, %.4f]  [95%%: %.4f, %.4f]  sd=%.4f  P(kappa>rho)=%.4f" %
              (name, kk["median"], kk["p5"], kk["p95"], kk["p2.5"], kk["p97.5"],
               kk["sd"], r["p_kappa_above_rho"]))
    for name in ("paired", "block_batch"):
        r = out["resamplings"][name]
        aa = r["allowance_rows"]
        print("%-11s allowance %.1f rows  [90%%: %.1f, %.1f]" %
              (name, aa["median"], aa["p5"], aa["p95"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
