"""Turn the seven duel runs into the error bar the paper prints.

The paper does not print a confidence interval for kappa.  It prints the span
over seven independent re-runs of the same correction stream in fresh
processes, because the bootstrap interval is 2.5x too narrow: resampling the
corrections of one run cannot see the error that dominates, which is the
large-band gather moving between 53 and 83 us depending on what is resident.
See exp/bootstrap_kappa.py (kept, marked superseded) for the narrower one.

Inputs are the seven raw files exp/duel_big/duel_big_run*.json.  Nothing is
re-measured here, so this script is deterministic: same files in, same
kappa_runs.json out, on any machine.

  python exp/fit_kappa_runs.py

Per run, the two paths are fitted as
    t_R = c_R * |sigma| + F_R        t_B = c_B * |B| + F_B
and the three quantities the boundary is written in are
    kappa  = c_R / c_B               (dimensionless; clock cancels)
    dp     = (F_B - F_R) / c_B       (the surcharge, in candidates)
    n_0    = dp / kappa              (rows; a derived quantity, not a constant)

Run 3 is the 10000-correction run; the other six are 3000.  Every other run
truncated to 3000 everywhere would be wrong: run 3's extra corrections are the
point of running it.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUEL = os.path.join(HERE, "duel_big")
N, K = 394.0, 182.0
NUSE = {1: 3000, 2: 3000, 3: 10000, 4: 3000, 5: 3000, 6: 3000, 7: 3000}
SMALL, LARGE = 7, 1284


def load(i):
    p = os.path.join(DUEL, "duel_big_run%d.json" % i)
    d = json.load(open(p, encoding="utf-8"))
    n = NUSE[i]
    ns = np.asarray(d["n_scope"][:n], float)
    nb = np.asarray(d["n_band"][:n], float)
    tr = np.asarray(d["tR"][:n], float)
    tb = np.asarray(d["tB"][:n], float)
    return ns, nb, tr, tb


def one(i):
    ns, nb, tr, tb = load(i)
    aR, bR = np.polyfit(ns, tr, 1)
    aB, bB = np.polyfit(nb, tb, 1)
    kappa = aR / aB
    dp = (bB - bR) / aB
    return {
        "run": i,
        "n": int(ns.size),
        "kappa": float(kappa),
        "n0": float(dp / kappa),
        "dp": float(dp),
        # the rows the band would need against the corpus point
        "allow": float(kappa * N - K),
        # the rows it is short by, at the measured surcharge
        "deficit": float(K + dp - kappa * N),
        "c_R": float(aR * 1e9),
        "c_B": float(aB * 1e9),
        "F_R": float(bR * 1e6),
        "F_B": float(bB * 1e6),
        # the two band sizes whose timing the paper compares run to run
        "tB_small": float(tb[nb <= SMALL].mean() * 1e6),
        "tB_large": float(tb[nb >= LARGE].mean() * 1e6),
    }


def main():
    runs = [one(i) for i in sorted(NUSE)]
    v = lambda k: np.array([r[k] for r in runs])
    kap, allow, tb_s, tb_l = v("kappa"), v("allow"), v("tB_small"), v("tB_large")

    out = {"note": "seven independent processes, same 300-instance stream, "
                   "single-shot per correction", "runs": runs}
    p = os.path.join(HERE, "kappa_runs.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("run      n   kappa      n0    dp    allow  deficit   c_B    tB(<=7)  tB(>=1284)")
    for r in runs:
        print("%3d %6d  %.4f %6.0f %5.0f %7.1f %7.0f  %5.2f   %6.1f   %6.1f"
              % (r["run"], r["n"], r["kappa"], r["n0"], r["dp"], r["allow"],
                 r["deficit"], r["c_B"], r["tB_small"], r["tB_large"]))
    print()
    print("kappa   %.3f - %.3f   median %.3f" % (kap.min(), kap.max(), np.median(kap)))
    print("allow   %.0f - %.0f rows at Delta_p = 0" % (allow.min(), allow.max()))
    print("deficit %.0f - %.0f rows at the measured surcharge"
          % (v("deficit").min(), v("deficit").max()))
    print()
    # the sentence in section V-E: which size moves, and by how much
    print("bands <= %d move %.0f%% run to run; bands >= %d move %.0f%%"
          % (SMALL, 100 * (tb_s.max() / tb_s.min() - 1),
             LARGE, 100 * (tb_l.max() / tb_l.min() - 1)))
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
