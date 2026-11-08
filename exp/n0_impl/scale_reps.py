#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh-process repetitions of the three-size sweep, reduced to one row per cell.

The paper quotes three cells -- kappa 0.0337, 0.0305 and 0.0275 at |D| = 1M, 4M
and 16M -- and a cell is only defensible with the raw draws beside it, so this
writes every process's fit to exp/n0_impl/scale_runs/<size>_runNN.json and
reduces *every* run in that directory into exp/n0_impl/scale_reps.json, so a
second batch widens the sample instead of replacing it.

    python exp/n0_impl/scale_reps.py [reps]            # default 50 per cell

Why process-fresh at all: the fitted gather slope is the sensitive quantity, and
a fresh process is the unit a reader would repeat.  On the corrected instrument
the spread is small (c_B sd 2% at 16M) -- the three-fold tail the previous
version of this file documented was the defect, not the host, because a sorted
candidate prefix mixes cache-resident and cache-miss gathers inside one fit.

Re-running on another machine is the cross-machine check: nothing here needs the
corpus, the prototype, or a download -- only numpy.  Compare kappa.median and
c_B.median against the shipped file.  Within one machine the median moves by
about 0.001 between batches; a difference of 0.003 in kappa, or 0.3 ns in
c_B, is the smallest that reads as a machine effect.
"""
import glob
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "scale_runs")
OUT = os.path.join(HERE, "scale_reps.json")
WORKER = os.path.join(HERE, "scale.py")
SIZES = ["1m", "4m", "16m"]


def reduce(v):
    q = st.quantiles(v, n=4)
    return {"min": min(v), "p25": q[0], "median": st.median(v), "p75": q[2],
            "max": max(v), "mean": st.mean(v), "sd": st.pstdev(v)}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs(RUNS, exist_ok=True)
    py = sys.executable or "python"
    t0 = time.time()

    for tag in SIZES:
        have = sorted(glob.glob(os.path.join(RUNS, "%s_run*.json" % tag)))
        start = len(have) + 1
        for i in range(start, start + reps):
            fp = os.path.join(RUNS, "%s_run%02d.json" % (tag, i))
            r = subprocess.run([py, WORKER, tag, fp], capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit("%s run %d failed:\n%s" % (tag, i, r.stderr[-800:]))
            d = json.load(open(fp, encoding="utf-8"))
            print("%-4s run %2d  kappa %.4f  c_R %.3f  c_B %6.3f  F_R %6.0f  F_B %6.0f"
                  % (tag, i, d["kappa"], d["c_R"], d["c_B"], d["F_R"], d["F_B"]),
                  flush=True)

    out = {"note": "%d process-fresh runs per cell, corrected instrument "
                   "(predicate and band construction fixed, see scale.py).  "
                   "Protocol: seed 20261105, grids n in [512, 65536] and "
                   "k in [8, 4096], single shot per point, min of five rounds, "
                   "all path-R points before all path-B points.  Raw draws in "
                   "scale_runs/." % reps}
    for tag in SIZES:
        rows = [json.load(open(fp, encoding="utf-8"))
                for fp in sorted(glob.glob(os.path.join(RUNS, "%s_run*.json" % tag)))]
        out[tag] = {"N": rows[0]["N"], "reps": len(rows)}
        for key in ("kappa", "c_R", "c_B"):
            out[tag][key] = reduce([r[key] for r in rows])
        out[tag]["pass_B_4096"] = st.median([r["pass_B_4096"] for r in rows])

    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print()
    for tag in SIZES:
        d = out[tag]
        print("%-4s %d runs: kappa median %.4f [%.4f, %.4f] sd %.4f   "
              "c_R %.3f   c_B %.3f" % (tag, d["reps"], d["kappa"]["median"],
                                       d["kappa"]["min"], d["kappa"]["max"],
                                       d["kappa"]["sd"], d["c_R"]["median"],
                                       d["c_B"]["median"]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
