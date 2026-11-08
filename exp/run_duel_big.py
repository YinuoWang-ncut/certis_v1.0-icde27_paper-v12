"""Duel at larger N -- the fix the pilot says is the right one.

Why not repeats.  Re-measuring the same 300 corrections 31 times and taking
medians does *not* leave the protocol alone.  Each sweep re-touches the same
contiguous scope, so from sweep 2 on the 4000-row scan runs out of L2/L3 instead
of main memory, and the fitted constants move: c_R 11.34 -> 9.09 ns, F_R
14.76 -> 10.46 us, kappa 0.444 -> 0.349.  That is a change of *regime*, not a
reduction of noise, and it would silently replace the numbers in the paper with
warm-cache ones.

What the repeats did establish is how little noise there is: refitting each of
the 31 sweeps separately gives kappa = 0.349 +- 0.002 (sd 0.011, se 0.002).  The
wide interval in bootstrap_kappa.json is therefore not the timer; it is the fact
that 300 corrections are few enough that which scopes you drew moves the slope.

So the interval shrinks the only honest way: more corrections, each still
measured once, cold, in stream order -- the protocol the published constants
came from.  Same seed, so the first 300 corrections are exactly the ones in
w12_results.json; the rest extend the sample.

  python run_duel_big.py <n_batches> <output.json>

Per batch: 1000 active rules, 1% correction rate, 50/50 cap_low/cap_high, so
~10 cap corrections per batch, each run down both paths and timed once.
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
        "this harness times the prototype, which the artifact does not\n"
        "ship: set CERTIS_PROTO to the directory holding certis.py and\n"
        "run_w12.py.  The seven recorded runs are in exp/duel_big/, and\n"
        "exp/fit_kappa_runs.py reduces them into exp/kappa_runs.json.")
sys.path.insert(0, W12)
sys.path.insert(0, os.path.normpath(os.path.join(W12, "..", "w8")))

import run_w12 as W                                    # noqa: E402
import certis as C                                     # noqa: E402

MIX = {"cap_low": 0.5, "cap_high": 0.5, "pair": 0.0}
SEED = 31337


def main():
    n_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "duel_big",
                                                            "duel_big.json")
    t0 = time.time()
    topo = W.load_topology()
    inst, store, rules, stream, rng = W.make(1_000_000, topo, SEED)
    cert = C.Certifier(store, rules, policy="R-always")

    ns, nb, tr, tb, batch_id, same = [], [], [], [], [], []
    for b in range(n_batch):
        corrs = [c for c in stream.batch(1000, 0.01, MIX, "as-is") if c.kind == "cap"]
        for c in corrs:
            lo, hi = sorted((c.old, c.new))
            n_s = store.scope_len(c.code)
            fR, _, dtR = cert.apply(c, path="R")
            fB, _, dtB = cert.apply(c, path="B")
            ns.append(n_s)
            nb.append(len(store.band_u(c.code, int(lo), int(hi))))
            tr.append(dtR)
            tb.append(dtB)
            batch_id.append(b)
            same.append(bool(np.array_equal(np.sort(fR), np.sort(fB))))
        if (b + 1) % max(1, n_batch // 10) == 0:
            print(f"  batch {b+1}/{n_batch}  n={len(ns)}  "
                  f"wall {time.time()-t0:.1f}s", flush=True)

    d = dict(n_batches=n_batch, seed=SEED, mix=MIX,
             n_corr=len(ns), n_scope=ns, n_band=nb,
             tR=tr, tB=tb, batch=batch_id,
             frac=[x / max(y, 1) for x, y in zip(nb, ns)],
             same=same, wall_seconds=time.time() - t0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    print(f"\n{len(ns)} corrections, identical flip sets {int(np.sum(same))}, "
          f"wall {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
