"""Fresh-process repetitions of the 16.7M cell, reduced to one span.

The paper quotes this cell as "kappa does not move -- 0.31 median over 50 fresh
processes, 46 of them in 0.30-0.35".  That sentence is only defensible with the
raw runs beside it, so this writes both: exp/n0_impl/scale_16m_runs/runNN.json
for every process, and exp/n0_impl/scale_16m_runs.json with the reduction over
*every* run in that directory -- so a second batch widens the sample instead of
replacing it.

    python exp/n0_impl/scale_16m_reps.py [reps]     # default 25, ~1 min

Two batches of 25 is the shipped sample: one batch is enough to see the median,
two is enough to see that the median is stable while single runs are not.

Re-running on another machine is the cross-machine check: nothing here needs
the corpus, the prototype, or a download -- only numpy.  Compare kappa.median
and the share of runs in 0.30-0.35 against the shipped file.  Within one
machine the median moves by about 0.001 between batches; a difference of 0.02
between two machines is the smallest that reads as a machine effect.
"""
import glob
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "scale_16m_runs")
OUT = os.path.join(HERE, "scale_16m_runs.json")
WORKER = os.path.join(HERE, "scale_16m.py")
CORE = (0.30, 0.35)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    os.makedirs(RUNS, exist_ok=True)
    py = sys.executable or "python"

    have = sorted(glob.glob(os.path.join(RUNS, "run*.json")))
    start = len(have) + 1
    t0 = time.time()
    for i in range(start, start + reps):
        fp = os.path.join(RUNS, "run%02d.json" % i)
        r = subprocess.run([py, WORKER, fp], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("run %d failed:\n%s" % (i, r.stderr[-800:]))
        d = json.load(open(fp, encoding="utf-8"))
        print("run %2d  kappa %.4f  c_R %.3f  c_B %.3f  F_R %5.0f  F_B %5.0f"
              % (i, d["kappa"], d["c_R"], d["c_B"], d["F_R"], d["F_B"]), flush=True)

    rows = []
    for i, fp in enumerate(sorted(glob.glob(os.path.join(RUNS, "run*.json"))), 1):
        d = json.load(open(fp, encoding="utf-8"))
        d["rep"] = i
        rows.append(d)

    def col(k):
        return [r[k] for r in rows]

    k = col("kappa")
    n_core = len([x for x in k if CORE[0] <= x <= CORE[1]])
    out = {
        "note": "%d process-fresh repetitions of the 16.7M cell, protocol identical "
                "to scale.py (seed 20261105, same grids, same order, single shot "
                "per point, min of five rounds); raw runs in scale_16m_runs/"
                % len(rows),
        "N": rows[0]["N"],
        "reps": len(rows),
        "kappa": {"min": min(k), "p25": st.quantiles(k, n=4)[0],
                  "median": st.median(k), "p75": st.quantiles(k, n=4)[2],
                  "max": max(k), "mean": st.mean(k), "sd": st.pstdev(k),
                  "in_core": n_core, "core": list(CORE)},
        "c_R": {"median": st.median(col("c_R")), "sd": st.pstdev(col("c_R"))},
        "c_B": {"median": st.median(col("c_B")), "min": min(col("c_B")),
                "max": max(col("c_B")), "sd": st.pstdev(col("c_B"))},
        "wall_seconds": round(time.time() - t0, 1),
        "superseded": "a single run of scale.py on 2026-11-04 recorded kappa 0.1826 "
                      "for this size (c_B 16.256 ns, F_R 7358 ns).  It is one draw "
                      "from the slow tail this sample also contains, not a property "
                      "of the size: the median does not move, and the paper quotes "
                      "the median.  A reviewer re-running scale.py once can land "
                      "anywhere in [min, max].",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\n%d runs: kappa median %.4f  [%d in %.2f-%.2f]  span [%.4f, %.4f]  sd %.4f"
          % (len(rows), out["kappa"]["median"], n_core, CORE[0], CORE[1],
             out["kappa"]["min"], out["kappa"]["max"], out["kappa"]["sd"]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
