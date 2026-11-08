#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does a cell's ratio depend on how big the store is, at fixed n and k?

Why this file exists
--------------------
`masked_pipeline.py` measures the scope 412,875 at density 0.183 in a store of
1.39M lines (12 codes x 3 densities) and gets band/scan = 0.52.
`bigscope.py` measures the same (n, k) in a store of 2.79M lines (24 codes x 6
densities) and gets 1.01.  Same scope, same band size, same two paths, factor 2
apart.  `grid80.py` measures it in a store of 4.65M lines and gets 0.87.

Three scripts, three answers, and the difference is not the cell: it is the
store the index lives in.  A scatter gather costs what it costs because of where
the candidates sit in the address space and in the last-level cache, and both
depend on how many other codes share the store.

If that is what is happening, then a cell is only comparable to another cell
measured at the same store size, and the paper's two tables -- which never state
a store size -- are comparing cells measured in stores of different sizes.  That
is a real methodological defect and it is cheap to test: hold n and k fixed,
vary the number of codes the store holds, and see whether the ratio moves.

  python store_size.py            # n = 412,875, d = 0.183, 1/2/4/8/16 codes
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import masked_pipeline as M             # noqa: E402

import argparse

_ap = argparse.ArgumentParser(description="store-size confound at fixed n and k")
_ap.add_argument("--out", default=None,
                 help="where to write the record; default exp/out/store_size.json. "
                      "reproduce.py passes a scratch path so a smoke run does not "
                      "replace the curated record the paper cites.")
_ap.add_argument("--reps", type=int, default=None)
ARGS = _ap.parse_args()

TARGET = 412875        # the largest published per-code scope
DENS = 0.183           # its measured ceiling median
CODE_COUNTS = [1, 2, 4, 8, 16]


def main():
    reps = 15
    rows = []
    for nc in CODE_COUNTS:
        # code 0 is the measured cell; the rest are fillers at the same density,
        # so the store holds nc x TARGET lines and the only variable is its size
        plans = [("cell", TARGET, DENS)] + \
                [("f%d" % i, TARGET, DENS) for i in range(1, nc)]
        store, sigma, n_g = M.build(plans)
        ovh = np.zeros(store.n_l, dtype=bool)
        code = 0
        cand = store.band_u(code, M.CAP_NEW, M.CAP_OLD)
        buf = M.Buf(int(store.scope_len(code)), len(cand))
        for _ in range(2):
            M.run_R(store, sigma, code, ovh, buf)
            M.run_B(store, sigma, code, ovh, buf, cand)
            M.run_B(store, sigma, code, ovh, buf, None)
        t = M.block({
            "R": lambda: M.run_R(store, sigma, code, ovh, buf),
            "Bf": lambda: M.run_B(store, sigma, code, ovh, buf, None),
        }, reps)
        ratio = float(np.median(t["Bf"] / t["R"]))
        r = dict(codes=nc, store_lines=int(store.n_l),
                 store_mb=store.n_l * 2 / 1e6,
                 n=int(store.scope_len(code)), k=int(len(cand)),
                 tR=M.med(t["R"]), tBf=M.med(t["Bf"]),
                 ratio=ratio,
                 ratio_of_medians=M.med(t["Bf"]) / M.med(t["R"]),
                 # microseconds in, nanoseconds per element out
                 c_R_ns=M.med(t["R"]) * 1e3 / store.scope_len(code),
                 c_B_ns=M.med(t["Bf"]) * 1e3 / max(len(cand), 1),
                 # candidates as a share of the whole store: the quantity that
                 # sets how much of the gather lands in cache
                 global_density=float(sum(len(store.band_u(c, M.CAP_NEW, M.CAP_OLD))
                                          for c in range(len(plans))) / store.n_l),
                 R_spread=float(np.percentile(t["R"], 90) / np.percentile(t["R"], 10)),
                 Bf_spread=float(np.percentile(t["Bf"], 90) / np.percentile(t["Bf"], 10)))
        rows.append(r)
        print("codes %2d  store %9d lines (%6.1f MB unit) | R %8.1f us  Bf %8.1f us"
              " | B/R %6.3f | c_R %5.2f ns/row  c_B %5.2f ns/cand"
              % (nc, r["store_lines"], r["store_mb"], r["tR"], r["tBf"],
                 r["ratio"], r["c_R_ns"], r["c_B_ns"]), flush=True)
        del store, sigma

    lo, hi = rows[0]["ratio"], rows[-1]["ratio"]
    print()
    print("ratio moves %.3f -> %.3f as the store grows %dx (%.3fx)"
          % (lo, hi, CODE_COUNTS[-1] / CODE_COUNTS[0], hi / lo))
    for r in rows:
        print("   %2d code(s): c_B %5.2f ns/candidate, c_R %5.2f ns/row, kappa %.3f"
              % (r["codes"], r["c_B_ns"], r["c_R_ns"], r["c_R_ns"] / r["c_B_ns"]))

    res = dict(target=TARGET, dens=DENS, reps=reps, rows=rows,
               ratio_low=lo, ratio_high=hi, factor=hi / lo,
               note="code 0 is the measured cell; fillers at the same density "
                    "so the only variable is the store size")
    dest = ARGS.out if ARGS.out else os.path.join(OUT, "store_size.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s" % os.path.join(OUT, "store_size.json"))


if __name__ == "__main__":
    main()
