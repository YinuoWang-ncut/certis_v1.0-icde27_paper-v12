#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The one line in the decision term: where (6) would have to flip.

Sect. V-E says the table straddles a value of a_d.  This file computes that
value for each accounting, from the same four constants the table uses, so the
sentence in the paper can be checked instead of believed.

    T_R = F_R + (a_s + a_d) n,   T_B = F_B + (a_p + a_d) k + Q
    band wins  <=>  k + D/(a_p + a_d) < n (a_s + a_d)/(a_p + a_d)
               <=>  a_d > (k a_p + D - n a_s) / (n - k),   D = F_B - F_R + Q

Q is the rank probe, measured at 3.0 us for one corpus band; it is charged to
the band path only, so it enters D.  The n0 shift it causes is Q/c_R, not
Q/(a_p + a_d): n0 is the scope at which the two paths tie, and the scan's own
per-row cost is what converts a fixed cost into rows.

Two accountings, both from measured constants and neither mixing the two:

  array   the numpy backend of exp/n0_impl: a_s = 2.87, a_p = 8.72 ns, no
          decision term at all (a_d = 0), and D = n0 c_R = 269 * 2.87 = 772 ns.
  CERTIS  the w12 end-to-end fit of Sect. V-A: a_s + a_d = 11.3 ns,
          a_p + a_d = 25.6 ns, D = n0 c_R = 451 * 11.3 = 5.10 us, with a_s =
          1.45 (the int64 CSR gather) so a_d = 9.85 and a_p = 15.75.

D is taken from the table's own (kappa, n0) rather than from a fresh timing, so
the check is arithmetic on what the paper prints and nothing else.

Output: exp/n0_impl/dline.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dline.json")

N = 394        # corpus median scope
K = 182        # corpus median band


def line(a_s, a_p, D):
    return (K * a_p + D - N * a_s) / (N - K)


def main():
    # the probe is per layout: 3.05 us for the numpy backend, 3.00 us for CERTIS
    arr = {"a_s": 2.87, "a_p": 8.72, "a_d": 0.0, "c_R": 2.87, "n0": 269.0,
           "Q_ns": 3051.0}
    cer = {"a_s": 1.45, "a_p": 25.6 - 9.85, "a_d": 9.85, "c_R": 11.3, "n0": 451.0,
           "Q_ns": 3000.0}
    for c in (arr, cer):
        c["D_ns"] = round(c["n0"] * c["c_R"] + c["Q_ns"], 1)
        c["n0_eff"] = round(c["D_ns"] / c["c_R"], 1)

    rows = []
    for tag, c in (("array", arr), ("CERTIS", cer)):
        need = line(c["a_s"], c["a_p"], c["D_ns"])
        kappa = (c["a_s"] + c["a_d"]) / (c["a_p"] + c["a_d"])
        dp = c["D_ns"] / (c["a_p"] + c["a_d"])
        rows.append({
            "accounting": tag, **c,
            "kappa": round(kappa, 4), "Delta_p": round(dp, 1),
            "budget_kappa_n": round(kappa * N, 1),
            "needed_k_plus_Dp": round(K + dp, 1),
            "band_wins": bool(K + dp < kappa * N),
            "a_d_needed_ns": round(need, 2),
            "a_d_over_a_p": round(need / c["a_p"], 2),
        })
        print("%-7s kappa %.3f Dp %6.1f  k+Dp %6.1f vs kappa n %6.1f  wins=%s"
              "  a_d>%.1f ns (%.1fx a_p), actual %.2f"
              % (tag, kappa, dp, K + dp, kappa * N, K + dp < kappa * N,
                 need, need / c["a_p"], c["a_d"]))

    # the constants must reproduce the paper's two anchors before anything else
    assert abs((1.45 + 9.85) / 25.6 - 0.441) < 0.002, "CERTIS kappa"
    assert abs(8100 / 25.6 - 316) < 2, "CERTIS Delta_p with probe"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n": N, "k": K, "rows": rows}, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
