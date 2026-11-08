#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The third platform we can actually run: two widths, four machines' worth.

Why there is no third box
-------------------------
The paper has one microarchitecture for the matrix (Zen3) and one replication on
a second (mobile Zen4, `cross_machine.py`).  A third box is not available: it
would have to be borrowed, and a number nobody can re-run is worth less than the
limitation stated.  So the third platform is replaced by the two parameters a
box swap changes, measured on the box we have:

  value width   the column being tested (int8 vs int16 vs int32).  A narrower
                column is what a wide machine buys: both paths touch the same
                number of elements, but path R touches n of them and path B k,
                so the width should move the ratio in proportion to k/n -- which
                is exactly what a constant kappa denies.
  id width      the identifier array (int32 vs int64).  This is the parameter a
                32-bit address space would have fixed at 4 bytes and a 64-bit one
                at 8.  It costs path B twice (the gather index and the output)
                and path R once (the compaction output).

If the sign at a scope survives all six combinations, then the ratio is not an
artefact of how wide this box happens to be; if it does not survive, that is a
finding about the ratio's portability and the paper should say so.

What is timed is the two kernels, not the pipeline: a width change does not touch
the mask or the probe, and including them would only dilute the effect being
measured.  The kernels are the ones `masked_pipeline.py` uses, so the numbers are
comparable to that file's per-stage rows.

  python layout_sensitivity.py --reps 25
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))

import masked_pipeline as MP                           # noqa: E402

PLANS = [("p5", 197, 0.183), ("q1", 6969, 0.183), ("median", 44606, 0.183),
         ("q3", 412875, 0.183)]
VWIDTH = [np.int8, np.int16, np.int32]
IWIDTH = [np.int32, np.int64]


def bench(fns, reps):
    for f in fns.values():
        f()
    out = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            out[k].append((time.perf_counter() - t0) * 1e6)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(OUT, "layout.json"))
    a = ap.parse_args()

    store, sigma, _ = MP.build([(t, n, d) for t, n, d in PLANS])
    rows = []
    for code, (tag, n, d) in enumerate(PLANS):
        cand32 = store.band_u(code, MP.CAP_NEW, MP.CAP_OLD)
        k = int(len(cand32))
        n_scope = int(store.scope_len(code))
        lo, hi = int(store.cptr[code]), int(store.cptr[code + 1])
        cells = []
        for vw in VWIDTH:
            for iw in IWIDTH:
                v_c = store.unit_c[lo:hi].astype(vw)      # path R's column
                v = store.unit.astype(vw)                 # path B gathers these
                cand = cand32.astype(iw)
                m_r = np.zeros(hi - lo, dtype=bool)    # path R, both states
                m_r2 = np.zeros(hi - lo, dtype=bool)
                m_b = np.zeros(k, dtype=bool)          # path B, both states
                m_b2 = np.zeros(k, dtype=bool)
                out_r = np.zeros(hi - lo, dtype=iw)
                out_b = np.zeros(k, dtype=iw)

                def R():
                    np.greater(v_c, 4, out=m_r)
                    np.greater(v_c, 2, out=m_r2)
                    np.not_equal(m_r, m_r2, out=m_r)
                    c = int(np.count_nonzero(m_r))
                    if c:
                        np.compress(m_r, np.arange(lo, hi, dtype=iw),
                                    out=out_r[:c])

                def B():
                    g = v[cand]
                    np.greater(g, 4, out=m_b)
                    np.greater(g, 2, out=m_b2)
                    np.not_equal(m_b, m_b2, out=m_b)
                    c = int(np.count_nonzero(m_b))
                    if c:
                        np.compress(m_b, cand, out=out_b[:c])

                t = bench({"R": R, "B": B}, a.reps)
                cells.append(dict(vwidth=np.dtype(vw).name,
                                  iwidth=np.dtype(iw).name,
                                  tR=t["R"], tB=t["B"], ratio=t["B"] / t["R"]))
        ratios = [c["ratio"] for c in cells]
        r = dict(tag=tag, n=n_scope, k=k, dens=k / n_scope, cells=cells,
                 ratio_min=min(ratios), ratio_max=max(ratios),
                 sides=sorted({c["ratio"] < 1 for c in cells}))
        rows.append(r)
        print("%-6s n %7d k %7d d %.3f | B/R across six width pairs "
              "%.3f - %.3f | %s"
              % (tag, n_scope, k, r["dens"], r["ratio_min"], r["ratio_max"],
                 "one side" if len(r["sides"]) == 1 else "SIDES DISAGREE"))
        for c in cells:
            print("        %-6s / %-6s  R %8.1f  B %8.1f  B/R %.3f"
                  % (c["vwidth"], c["iwidth"], c["tR"], c["tB"], c["ratio"]))

    res = dict(reps=a.reps, vwidths=[np.dtype(v).name for v in VWIDTH],
               iwidths=[np.dtype(v).name for v in IWIDTH], rows=rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
