"""Duel, re-measured with repeated sweeps -- the data behind kappa.

The 300 records in w12_results.json measure each correction once per path, so
each is one draw from a distribution whose large-scope tail has sd ~30 us.  The
bootstrap over those records therefore cannot separate "kappa is 0.44" from
"kappa is 0.46", and the 4.0% margin at the corpus is inside the interval.

This script re-runs exactly the same 300 corrections, in the same stream order,
but sweeps the whole list R times.  Per correction the reported time is the
median over sweeps.  Two properties matter:

  * the protocol per correction is unchanged -- path R then path B, on the same
    rows, cold-then-warm exactly as before -- so the fitted constants stay
    comparable to the published ones rather than becoming warm-cache numbers;
  * a sweep visits all 300 corrections, so the cache state a correction sees is
    the one the previous correction in the stream left, not the one its own
    previous repeat left.  Repeating inside a correction would have quietly
    turned a cold 4000-row scan into a hot one.

Outputs exp/duel_repeat/duel_rep.json.

Result, and why it is not the number the paper uses: 31 sweeps give
c_R 11.34 -> 9.09 ns, F_R 14.76 -> 10.46 us, kappa 0.444 -> 0.349.  The repeats
did not remove noise, they changed the regime -- from sweep 2 on the same 4000-row
scope is resident, so the scan is no longer paying for memory.  What they do
establish is that the timer is not the problem: refitting each sweep separately
gives kappa +- 0.002.  The interval is therefore about *which corrections* and
*what memory state*, and the fix is more corrections (run_duel_big.py), not more
repeats of the same ones.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W12 = os.environ.get("CERTIS_PROTO")
if not W12:
    raise SystemExit(
        "this harness re-times the prototype, which the artifact does\n"
        "not ship: set CERTIS_PROTO to the directory holding certis.py\n"
        "and run_w12.py.  The discarded 31-sweep run it produced (kappa\n"
        "0.444 -> 0.349, a change of regime, not of noise) is recorded in\n"
        "exp/duel_repeat/duel_rep.json.")
else:
    sys.path.insert(0, W12)
sys.path.insert(0, os.path.normpath(os.path.join(W12, "..", "w8")))

import run_w12 as W                                    # noqa: E402
import certis as C                                     # noqa: E402

MIX = {"cap_low": 0.5, "cap_high": 0.5, "pair": 0.0}
N_BATCH = 30
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
OUTDIR = os.path.join(HERE, "duel_repeat")


def main():
    t0 = time.time()
    topo = W.load_topology()
    inst, store, rules, stream, rng = W.make(1_000_000, topo, 31337)
    cert = C.Certifier(store, rules, policy="R-always")

    corrs = []
    for _ in range(N_BATCH):
        corrs.extend(c for c in stream.batch(1000, 0.01, MIX, "as-is")
                     if c.kind == "cap")
    n = len(corrs)
    n_scope = np.array([store.scope_len(c.code) for c in corrs], dtype=np.int64)
    n_band = np.empty(n, dtype=np.int64)
    for i, c in enumerate(corrs):
        lo, hi = sorted((c.old, c.new))
        n_band[i] = len(store.band_u(c.code, int(lo), int(hi)))
    print(f"corrections {n}  |scope| min/med/max {n_scope.min()}/"
          f"{int(np.median(n_scope))}/{n_scope.max()}  build {time.time()-t0:.1f}s",
          flush=True)

    tR = np.empty((REPS, n))
    tB = np.empty((REPS, n))
    same = np.ones(n, dtype=bool)
    for r in range(REPS):
        for i, c in enumerate(corrs):
            fR, _, dt = cert.apply(c, path="R")
            tR[r, i] = dt
            fB, _, dt = cert.apply(c, path="B")
            tB[r, i] = dt
            if r == 0:
                same[i] = bool(np.array_equal(np.sort(fR), np.sort(fB)))
        print(f"  sweep {r+1}/{REPS}  wall {time.time()-t0:.1f}s  "
              f"median tR {np.median(tR[r])*1e6:.2f}us "
              f"tB {np.median(tB[r])*1e6:.2f}us", flush=True)

    mR = np.median(tR, axis=0)
    mB = np.median(tB, axis=0)
    sdR = tR.std(axis=0, ddof=1)
    sdB = tB.std(axis=0, ddof=1)

    out = dict(
        reps=REPS, n_batch=N_BATCH, mix=MIX,
        n_corr=int(n),
        n_scope=n_scope.tolist(), n_band=n_band.tolist(), frac=(n_band / np.maximum(n_scope, 1)).tolist(),
        tR=mR.tolist(), tB=mB.tolist(),
        sdR=sdR.tolist(), sdB=sdB.tolist(),
        same=[bool(x) for x in same],
        raw_tR=tR.tolist(), raw_tB=tB.tolist(),
        wall_seconds=time.time() - t0,
    )
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "duel_rep.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nidentical flip sets {int(same.sum())}/{n}")
    print(f"within-correction sd: tR med {np.median(sdR)*1e6:.2f}us "
          f"p90 {np.percentile(sdR,90)*1e6:.2f}us | "
          f"tB med {np.median(sdB)*1e6:.2f}us p90 {np.percentile(sdB,90)*1e6:.2f}us")
    print(f"between-correction spread of medians: tR sd {mR.std(ddof=1)*1e6:.2f}us "
          f"tB sd {mB.std(ddof=1)*1e6:.2f}us")
    print(f"wrote {p}  wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
