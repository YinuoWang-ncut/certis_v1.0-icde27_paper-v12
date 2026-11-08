#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The same duel, with the arm order swapped inside each correction.

Why.  `run_duel_big.py` executes, for every correction, R first and B second.
B therefore always runs on a heap that R has just touched -- the scope's rows,
the group ids, the override flags are all in cache -- while R pays the cold
miss.  On a 400-row scope that difference is 2-5 us, which is the same size as
the quantity the criterion is trying to decide, and it flatters exactly one arm.

The fix is not more repeats (repeats measure the same order) and not a median
over processes (the order is fixed in every process).  It is to measure each
correction twice in opposite orders and average the two:

    order A:  R then B      tR_A (cold),  tB_A (warm)
    order B:  B then R      tB_B (cold),  tR_B (warm)
    tR = mean(tR_A, tR_B),  tB = mean(tB_A, tB_B)

Each arm now pays one cold and one warm observation, so the pair is symmetric
and the difference tR - tB is a property of the paths rather than of the order
the harness happened to use.

Cost: four executions per correction instead of two.  The 600 corrections take
seconds, which is why this is worth doing rather than arguing about.

  CERTIS_PROTO=... python veto_duel_rot.py --batches 60 --rep 0
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

import run_w12 as W                                          # noqa: E402
import certis as C                                           # noqa: E402

MIX = {"cap_low": 0.5, "cap_high": 0.5, "pair": 0.0}
SEED = 31337


def calib():
    """A speed readout for the process.  The same 2M-element pass, on an array
    small enough to stay resident, so it measures the machine's state and not
    the workload's.  Without it, a 1.6x swing between processes (boost state,
    SMT neighbour, a background scan) is indistinguishable from a fitted slope
    moving, and the between-process spread of c_B is read as estimation error
    when it is the box."""
    ref = np.arange(1 << 21, dtype=np.int64)
    v = []
    for _ in range(30):
        t0 = time.perf_counter()
        np.count_nonzero(ref >= 12345)
        v.append(time.perf_counter() - t0)
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--order", choices=["rot", "AB", "BA"], default="rot")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    topo = W.load_topology()
    inst, store, rules, stream, rng = W.make(1_000_000, topo, SEED)
    cert = C.Certifier(store, rules, policy="R-always")

    corrs = []
    for _ in range(a.batches):
        corrs += [c for c in stream.batch(1000, 0.01, MIX, "as-is")
                  if c.kind == "cap"]

    calib0 = calib()
    rows = []
    for i, c in enumerate(corrs):
        lo, hi = sorted((c.old, c.new))
        n_s = store.scope_len(c.code)
        k_n = len(store.band_u(c.code, int(lo), int(hi)))
        flip = (a.order == "rot" and i % 2 == 1) or (a.order == "BA")
        if not flip:
            _, _, tR1 = cert.apply(c, path="R")
            _, _, tB2 = cert.apply(c, path="B")
            _, _, tB1 = cert.apply(c, path="B")
            _, _, tR2 = cert.apply(c, path="R")
        else:
            _, _, tB1 = cert.apply(c, path="B")
            _, _, tR2 = cert.apply(c, path="R")
            _, _, tR1 = cert.apply(c, path="R")
            _, _, tB2 = cert.apply(c, path="B")
        rows.append(dict(n=n_s, k=k_n, batch=i,
                         tR_first=tR1, tB_second=tB2,
                         tB_first=tB1, tR_second=tR2))
        # Field names are positional in the *pair*, not in the call sequence:
        #   order A (R,B,B,R): tR_first = 1st call, tB_second = 2nd,
        #                      tB_first = 3rd,       tR_second = 4th
        #   order B (B,R,R,B): tB_first = 1st,       tR_second = 2nd,
        #                      tR_first = 3rd,       tB_second = 4th
        # so "first"/"second" always mean "earlier/later of that arm's two
        # observations", and tR = mean(tR_first, tR_second) puts each arm one
        # cold and one warm observation.  Decomposing by call position needs
        # the parity of i, which is why it is kept in the file.

    ns = [r["n"] for r in rows]
    tR = [(r["tR_first"] + r["tR_second"]) / 2 for r in rows]
    tB = [(r["tB_first"] + r["tB_second"]) / 2 for r in rows]
    res = dict(rep=a.rep, order=a.order, n_corr=len(rows), n_l=store.n_l,
               seed=SEED, mix=MIX, calib0=calib0, calib1=calib(),
               n_scope=ns, n_band=[r["k"] for r in rows], batch=[r["batch"] for r in rows],
               tR=tR, tB=tB,
               tR_first=[r["tR_first"] for r in rows],
               tR_second=[r["tR_second"] for r in rows],
               tB_first=[r["tB_first"] for r in rows],
               tB_second=[r["tB_second"] for r in rows],
               wall_seconds=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    p = a.out or os.path.join(OUT, "veto_duel_%s_%d.json" % (a.order, a.rep))
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    win = sum(1 for a_, b_ in zip(tR, tB) if b_ < a_)
    print("order %-3s rep %d  n=%d  band faster %d/%d  med tR %.2f  med tB %.2f  "
          "wall %.1fs -> %s"
          % (a.order, a.rep, len(rows), win, len(rows),
             float(np.median(tR)) * 1e6, float(np.median(tB)) * 1e6,
             time.time() - t0, os.path.basename(p)), flush=True)


if __name__ == "__main__":
    main()
