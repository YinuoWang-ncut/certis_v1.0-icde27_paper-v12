#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the end-to-end pipeline re-runnable on this box, today?

P1's report said no: "工件不随附 certis.py，端到端流水线无法重跑".  The artifact
zip indeed does not ship the prototype's loader, but the loader was never lost --
`run_w12.py` and `w8/w8_lib.py` sit in 2026-11-04-00-12-59, and CERTIS_PROTO is
the only thing that has to be set.  So the claim to test is narrow: with
CERTIS_PROTO pointing at that directory, does W.run_cell reproduce a cell?

Reproduction means three things, in increasing strength:

  1. it runs at all (imports resolve, the corpus loads, a policy returns);
  2. Delta-completeness holds (missed = extra = 0) for every policy;
  3. the ordering of the arms matches Table II, and the cell level is within
     the run-to-run spread the paper reports.

What it cannot establish is bit-identical latency: every process re-times, and
mix_order.py already documented a 1.6x between-process swing on this box.  So
the comparison below is *inside one process*, arms interleaved by repetition,
and the output prints the calibration readout beside each arm.

  set CERTIS_PROTO=D:\\期刊\\2026-11-04-00-12-59\\w12
  python e2e_repro.py --batches 3 --reps 2
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

MIX = {"cap_low": 0.55, "pair": 0.40, "rel": 0.05}
SEED = 31337


def calib():
    ref = np.arange(1 << 21, dtype=np.int64)
    v = []
    for _ in range(30):
        t0 = time.perf_counter()
        np.count_nonzero(ref >= 12345)
        v.append(time.perf_counter() - t0)
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-rules", type=int, default=1000)
    ap.add_argument("--n-g", type=int, default=1_000_000)
    ap.add_argument("--policies", default="R-always,B-always,CERTIS,CERTIS-crit,CERTIS-veto")
    ap.add_argument("--out", default=os.path.join(OUT, "e2e_repro.json"))
    a = ap.parse_args()

    t0 = time.time()
    topo = W.load_topology()
    # the same determinism fix mix_order.py needed: pair_codes comes out of a set
    topo["pair_codes"] = sorted(topo["pair_codes"])
    print("topology: %d pair codes (canonicalised), load %.1fs"
          % (len(topo["pair_codes"]), time.time() - t0), flush=True)

    rows = []
    for rep in range(a.reps):
        for pol in a.policies.split(","):
            c0 = calib()
            r = W.run_cell(a.n_g, topo, SEED, a.n_rules, 0.01, a.batches, pol,
                           MIX, cap_mode="as-is", measure_paths=False)
            c1 = calib()
            row = dict(rep=rep, policy=pol, n_corr=r["n_corr"],
                       n_batches=r["n_batches"], n_l=r["n_l"],
                       lat_p50=r["lat_p50"], lat_mean=r["lat_mean"],
                       lat_p99=r["lat_p99"], paths=r["paths"],
                       band_share=r["band_share"], missed=r["missed"],
                       extra=r["extra"], flips_total=r["flips_total"],
                       calib0=c0, calib1=c1)
            rows.append(row)
            print("rep %d  %-13s p50 %7.2f us  paths %s  missed %d extra %d  "
                  "flips %d  calib %.3f us"
                  % (rep, pol, r["lat_p50"] * 1e6, r["paths"], r["missed"],
                     r["extra"], r["flips_total"], c0 * 1e6), flush=True)

    # ---- summary: medians across reps, and the ordering ---------------------
    print()
    print("%-14s %9s %9s %9s %7s %7s" %
          ("policy", "p50 us", "mean us", "p99 us", "missed", "extra"))
    med = {}
    for pol in a.policies.split(","):
        v = [r for r in rows if r["policy"] == pol]
        p50 = float(np.median([r["lat_p50"] for r in v])) * 1e6
        mean = float(np.median([r["lat_mean"] for r in v])) * 1e6
        p99 = float(np.median([r["lat_p99"] for r in v])) * 1e6
        miss = sum(r["missed"] for r in v)
        extra = sum(r["extra"] for r in v)
        med[pol] = dict(p50=p50, mean=mean, p99=p99, missed=miss, extra=extra,
                        paths=v[0]["paths"], n_corr=v[0]["n_corr"])
        print("%-14s %9.2f %9.2f %9.2f %7d %7d"
              % (pol, p50, mean, p99, miss, extra))

    base = med.get("R-always")
    if base:
        print()
        print("relative to R-always (p50):")
        for pol, m in med.items():
            if pol == "R-always":
                continue
            print("  %-13s %+6.1f%%   (%+.2f us)"
                  % (pol, 100 * (m["p50"] / base["p50"] - 1),
                     m["p50"] - base["p50"]))
        print()
        print("paper (Table II, 10^3 rules, 1% rate): R 19.2--36.1, B 21.9--39.7, "
              "CERTIS 21.9--40.6 us; CERTIS vs scan +12.5 to +14.4%.")

    comp = all(med[p]["missed"] == 0 and med[p]["extra"] == 0 for p in med)
    print()
    print("Delta-completeness on every arm: %s" % ("holds (0 missed, 0 extra)"
                                                   if comp else "VIOLATED"))
    print("wall %.1fs, batches %d, corrections %s"
          % (time.time() - t0, a.batches,
             sorted({m["n_corr"] for m in med.values()})))

    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(batches=a.batches, reps=a.reps, n_rules=a.n_rules,
                       n_g=a.n_g, seed=SEED, mix=MIX, proto=W12,
                       complete=bool(comp), rows=rows, medians=med),
                  fh, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
