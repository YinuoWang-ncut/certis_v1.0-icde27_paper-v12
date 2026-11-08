#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Second-machine companion: the two readings reproduce.py does not produce.

`reproduce.py --no-data` prints a `record` block of host-dependent numbers --
kappa, scan_ns_per_row, fetch_over_scan, c_B.  That block is NOT a second
column.  It re-reads the shipped `exp/*.json` (see `recorded` in reproduce.py),
so on a second box it prints the first box's numbers back, digit for digit;
only the run header changes.  The cross-machine comparison the README
designates is `exp/n0_impl/scale_reps.py 50` -- bare-layout kappa, 50 fresh
processes per cell, about a minute.  This file adds the two readings that
script cannot give:

  P0  Table V's denominator, rebuilt from scratch on the Table V instance:
      |D| = 2e6 rows over 5,410 groups, seed 20261105, scope 436 rows, k = 69,
      path R and path B, warm (one scope, thousands of corrections) and cold
      (a fresh scope per correction).  The paper prints 9.5 / 13.7 us and a
      ratio of 1.44.  If P0 does not land near that on the *first* box, the
      port is wrong and the second box is worth nothing.
  P1  the band's own slope on that same instance: k over 8 -> 2,048 at fixed
      scope.  T_B = F_B + c_B k; this is the one line in the cost model that
      is identifiable at the scope lengths we can actually reach, because the
      fixed cost dominates below n ~ 4,000 and path R's marginal cost hides
      under it.  c_B is the half of kappa that can be measured here; the scan
      half comes from `reproduce.py`'s scan_ns_per_row.
  P2  two machine constants the paper claims to have measured rather than
      quoted: streaming bandwidth and pointer-chase latency.

Why there is no kappa here.  Fitting c_R on scopes of 200-4,000 rows, the
signal (c_R * n) is smaller than the run-to-run spread of the fixed cost, so
the fit returns anything between -0.1 and +0.9 ns/row depending on the order
of the runs.  A number produced that way would not survive an artifact
evaluation, and kappa across machines is worth claiming only if each machine's
kappa is.  Use the release's own fit (exp/n0_holdout, exp/cost_model.py) for
that, and use this file for the two readings it cannot give.

Usage
-----
  Windows:  set CERTIS_PROTO=D:\\path\\to\\dir\\holding\\certis.py
  POSIX  :  export CERTIS_PROTO=/path/to/dir/holding/certis.py

  python cross_machine.py --quick --tag box2      # P0 only, ~1 min
  python cross_machine.py --tag box2              # P0-P2, ~5 min
  python cross_machine.py --merge box2            # aggregate the r*.json of that tag

Each run writes `exp/cross_machine_<tag>-rK.json`, K auto-incrementing, so
three repetitions are three files and neither overwrites the other.  `--merge`
is the step that produces the two lines Table V prints for the second box:
the per-run ratios, their median, and the level ratios against the first
box's 9.5 / 13.7 us.  The paper's caption quotes the median of the paired
ratios, not the ratio of the medians; the aggregate prints both so the two
cannot be confused.

Port discipline -- a comparison between undisciplined runs is worth less than
none
-----
  1. nothing else running; close the browser and every VM
  2. mains power, high-performance plan, no thermal throttling
  3. one process; --affinity N pins to one cpu if you want to be strict
  4. same python and numpy on both boxes: the json header records both
  5. if the box is a VM, put it in --tag: a noisy neighbour voids P1
"""
import argparse
import glob
import json
import os
import platform
import random
import re
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U, G, SEED = 2_000_000, 5410, 20261105
LAM = 1.5
K_SWEEP = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BOXP = dict(R=9.5, B=13.7)      # Table V, box 1, warm p50 us
COLD1 = 1.462                   # box 1, cold band / path R (15.35 / 10.5)


def cold_ratio(r, b):
    return float(b) / float(r) if r else float("nan")


def pct(a, q):
    return float(np.percentile(a, q))


def summarise(v):
    v = np.asarray(v, dtype=float)
    return dict(p50=round(pct(v, 50), 3), p95=round(pct(v, 95), 3),
                mean=round(float(v.mean()), 3),
                sd=round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
                min=round(float(v.min()), 3), max=round(float(v.max()), 3),
                n=int(len(v)))


def build_store(C):
    """The Table V instance: exp/path_r_h2h.py's generator, unchanged."""
    rng = np.random.default_rng(SEED)
    amt = np.round(rng.lognormal(6.0, 1.0, U), 2).astype(np.float64)
    gid = rng.integers(0, G, U).astype(np.int32)

    counts = np.bincount(gid, minlength=G).astype(np.int64)
    order = np.argsort(gid, kind="stable")
    cptr = np.zeros(G + 1, dtype=np.int64)
    cptr[1:] = np.cumsum(counts)

    amt_max = float(amt.max())
    unit = np.clip(np.rint(amt / amt_max * 30000.0), 0, 32000).astype(np.int16)
    override = np.random.default_rng(SEED + 1).random(G) < 0.18

    inst = dict(unit=unit, amt=amt, gid=gid, code=gid, c_ptr=cptr,
                c_line=order.astype(np.int32), n_l=U, n_g=G)
    return C.Store(inst, override, build_index=True), amt, counts, cptr, amt_max


def timed(fn, corr, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(corr)
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def abba(cert, corr, reps):
    """Alternate which path goes first, so neither keeps the warm slot."""
    R, B = [], []
    for i in range(reps):
        if i % 2 == 0:
            R += timed(cert._path_R, corr, 2)
            B += timed(cert._path_B, corr, 2)
        else:
            B += timed(cert._path_B, corr, 2)
            R += timed(cert._path_R, corr, 2)
    return R, B


def fits(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(set(x.tolist())) < 3:
        return float("nan"), float("nan")
    A = np.vstack([np.ones_like(x), x]).T
    (F, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(F), float(c)


def bandwidth(nbytes=512 << 20):
    a = np.ones(nbytes // 8, dtype=np.float64)
    b = np.empty_like(a)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        np.copyto(b, a)
        best = min(best, time.perf_counter() - t0)
    return round(nbytes / best / 1e9, 2)


def bare_store(C, n):
    """One group of n rows: the layout the cost model is fitted on.

    The Table V instance caps every scope at 436 rows, and below ~4,000 rows
    the fixed cost is larger than c_B * k, so a k sweep there measures noise.
    The paper fits over k in [8, 4096] on bare layouts; this reproduces that
    range on the second box.
    """
    rng = np.random.default_rng(SEED)
    amt = np.round(rng.lognormal(6.0, 1.0, n), 2).astype(np.float64)
    gid = np.zeros(n, dtype=np.int32)
    cptr = np.array([0, n], dtype=np.int64)
    order = np.arange(n, dtype=np.int32)
    unit = np.clip(np.rint(amt / float(amt.max()) * 30000.0), 0,
                   32000).astype(np.int16)
    inst = dict(unit=unit, amt=amt, gid=gid, code=gid, c_ptr=cptr,
                c_line=order, n_l=n, n_g=1)
    return C.Store(inst, np.array([False]), build_index=True), amt


def chase(n=1 << 22, hops=15_000_000):
    idx = np.random.default_rng(7).permutation(n).astype(np.int64)
    nxt = np.empty(n, dtype=np.int64)
    nxt[idx] = np.roll(idx, -1)
    p = 0
    for _ in range(200_000):
        p = nxt[p]
    t0 = time.perf_counter()
    for _ in range(hops):
        p = nxt[p]
    return round((time.perf_counter() - t0) / hops * 1e9, 2)


def run_files(tag):
    """Every `cross_machine_<tag>-rK.json`, in K order."""
    pat = os.path.join(HERE, "cross_machine_%s-r*.json" % tag)
    found = []
    for f in glob.glob(pat):
        m = re.search(r"-r(\d+)\.json$", f)
        if m:
            found.append((int(m.group(1)), f))
    return [f for _, f in sorted(found)]


def merge_report(tag, box1=(BOXP["R"], BOXP["B"])):
    """The aggregate Table V quotes, computed rather than read off a log."""
    files = run_files(tag)
    if not files:
        sys.exit("no cross_machine_%s-r*.json beside this file; run the "
                 "script on that box first" % tag)
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)["P0"]
        r, b = p["path_R"]["p50"], p["path_B"]["p50"]
        cr, cb = p["cold"]["path_R"]["p50"], p["cold"]["path_B"]["p50"]
        rows.append(dict(file=os.path.basename(f), R=r, B=b, ratio=b / r,
                         cold=cold_ratio(cr, cb)))
    med = lambda a: float(np.median(a))
    ratio_of_medians = med([r["B"] for r in rows]) / med([r["R"] for r in rows])
    agg = dict(
        tag=tag, runs=len(rows),
        R_p50=round(med([r["R"] for r in rows]), 3),
        B_p50=round(med([r["B"] for r in rows]), 3),
        ratio_of_medians=round(ratio_of_medians, 3),
        ratio_median=round(med([r["ratio"] for r in rows]), 3),
        cold_median=round(med([r["cold"] for r in rows]), 3),
        level_over_box1=dict(R=round(med([r["R"] for r in rows]) / box1[0], 2),
                             B=round(med([r["B"] for r in rows]) / box1[1], 2)),
        per_run=[{k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in r.items()} for r in rows],
        box1=dict(R_p50=box1[0], B_p50=box1[1],
                  ratio=round(box1[1] / box1[0], 3)),
    )
    print("TABLE V, box '%s', %d run(s)" % (tag, len(rows)))
    print("  %-28s %7s %7s %7s %7s" % ("file", "R p50", "B p50", "B/R", "cold B/R"))
    for r in rows:
        print("  %-28s %7.2f %7.2f %7.3f %7.3f"
              % (r["file"], r["R"], r["B"], r["ratio"], r["cold"]))
    print("  %-28s %7.2f %7.2f %7.3f %7.3f"
          % ("median of the runs", agg["R_p50"], agg["B_p50"],
             agg["ratio_median"], agg["cold_median"]))
    print("\n  box 1 row       R %5.2f  B %5.2f  B/R %.2f"
          % (box1[0], box1[1], agg["box1"]["ratio"]))
    print("  box 2 cells     R %5.2f  B %5.2f  B/R %.3f   <- what the table prints"
          % (agg["R_p50"], agg["B_p50"], agg["ratio_of_medians"]))
    print("  box 2 caption   B/R %.3f            <- median of %d paired ratios"
          % (agg["ratio_median"], len(rows)))
    print("  level moved %.2fx / %.2fx;  warm ratios %.0f%% apart, cold %.0f%%"
          % (agg["level_over_box1"]["R"], agg["level_over_box1"]["B"],
             100 * abs(agg["ratio_of_medians"] / agg["box1"]["ratio"] - 1),
             100 * abs(agg["cold_median"] / COLD1 - 1)))
    print("  cold %.3f (box 2) vs %.3f (box 1)" % (agg["cold_median"], COLD1))

    out = os.path.join(HERE, "table_v_%s.json" % tag)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    print("\nwrote", out)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="P0 only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--affinity", type=int, default=-1)
    ap.add_argument("--merge", metavar="TAG", default=None,
                    help="aggregate cross_machine_<TAG>-r*.json; no timing")
    args = ap.parse_args()

    if args.merge:
        merge_report(args.merge)
        return

    proto = (os.environ.get("CERTIS_PROTO")
             or os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    if not os.path.exists(os.path.join(proto, "certis.py")):
        sys.exit("certis.py not found in %s\n"
                 "  It ships at the tree root; set CERTIS_PROTO only to"
                 " re-use a different copy." % proto)
    sys.path.insert(0, proto)
    import certis as C

    if args.affinity >= 0:
        try:
            os.sched_setaffinity(0, {args.affinity})
        except Exception:
            pass

    store, amt, counts, cptr, amt_max = build_store(C)
    q = lambda x: int(np.clip(np.rint(x / amt_max * 30000.0), 0, 32000))
    gmax = int(np.argmax(counts))
    n_scope = int(counts[gmax])
    cert = C.Certifier(store, None, policy="R-always")
    theta0 = float(LAM * float(amt.mean()))

    sv = np.sort(amt[store.cline[cptr[gmax]:cptr[gmax + 1]]])
    i0 = len(sv) // 2
    i1 = min(len(sv) - 1, i0 + 69)
    p0 = C.Correction("cap", code=gmax, old=q(float(sv[i0])), new=q(float(sv[i1])))
    for _ in range(300):
        cert._path_R(C.Correction("cap", code=gmax, old=q(theta0),
                                  new=q(theta0 * 1.02)))

    R, B = abba(cert, p0, args.reps)
    k_p0 = len(cert._path_B(p0))

    codes = list(range(G))
    random.Random(12345).shuffle(codes)
    cold = [c for c in codes if 340 <= counts[c] <= 470][:40]
    cR, cB = [], []
    for code in cold:
        c = C.Correction("cap", code=code, old=q(theta0), new=q(theta0 * 1.02))
        cR += timed(cert._path_R, c, 1)
        cB += timed(cert._path_B, c, 1)

    res = dict(
        host=dict(name=socket.gethostname(), tag=args.tag,
                  platform=platform.platform(), machine=platform.machine(),
                  cpu=platform.processor(),
                  python=platform.python_version(), numpy=np.__version__,
                  affinity=args.affinity, reps=args.reps),
        instance=dict(rows=U, groups=G, seed=SEED, scope_rows=n_scope,
                      scope_code=gmax, unit_dtype="int16"),
        P0=dict(n=n_scope, k=int(k_p0), path_R=summarise(R),
                path_B=summarise(B),
                ratio_over_R=round(pct(B, 50) / pct(R, 50), 3),
                cold=dict(path_R=summarise(cR), path_B=summarise(cB),
                          scopes=len(cold))),
    )
    p = res["P0"]
    print("P0  n=%d k=%d   R p50 %.2f us   B p50 %.2f us   B/R %.3f"
          % (p["n"], p["k"], p["path_R"]["p50"], p["path_B"]["p50"],
             p["ratio_over_R"]))
    print("    cold   R %.2f  B %.2f  (B/R %.3f)"
          % (p["cold"]["path_R"]["p50"], p["cold"]["path_B"]["p50"],
             p["cold"]["path_B"]["p50"] / p["cold"]["path_R"]["p50"]))
    print("    paper  R 9.50  B 13.70  (B/R 1.44)   cold R 10.50")

    if not args.quick:
        N_BARE = 1 << 16
        bst, bamt = bare_store(C, N_BARE)
        bq = lambda x: int(np.clip(np.rint(x / float(bamt.max()) * 30000.0),
                                   0, 32000))
        bsv = np.sort(bamt)
        bce = C.Certifier(bst, None, policy="R-always")
        for _ in range(200):
            bce._path_B(C.Correction("cap", code=0, old=bq(float(bsv[0])),
                                     new=bq(float(bsv[1]))))
        pairs = []
        for k in K_SWEEP:
            if len(bsv) < k + 16:
                continue
            i0 = len(bsv) // 2 - k // 2
            c = C.Correction("cap", code=0, old=bq(float(bsv[i0])),
                             new=bq(float(bsv[i0 + k])))
            d = timed(bce._path_B, c, args.reps)
            fl = int(len(bce._path_B(c)))
            pairs.append((k, round(pct(d, 50), 3), fl))
            print("P1  k=%-5d  B %6.2f us   flips %d%s"
                  % (pairs[-1] + ("   <- empty band, dropped" if fl == 0 else "",)))
        pairs = [p for p in pairs if p[2] > 0]
        F_B, c_B = fits([p[2] for p in pairs], [p[1] for p in pairs])
        res["P1"] = dict(pairs=pairs, F_B_us=round(F_B, 3),
                         c_B_ns_per_cand=round(c_B, 3),
                         layout="bare single group, n=%d" % N_BARE)
        res["P2"] = dict(stream_GBs=bandwidth(), chase_ns=chase())
        print("\nP1  T_B = %.2f us + %.3f ns/candidate   (paper: 25.6 ns)"
              % (F_B, c_B))
        print("P2  streaming %.2f GB/s   pointer chase %.1f ns"
              % (res["P2"]["stream_GBs"], res["P2"]["chase_ns"]))
        print("\nkernel ratio c_R/c_B is NOT fitted here on purpose; run"
              "\n`python reproduce.py --no-data` on this box and read its"
              "\nrecord block for scan_ns_per_row, fetch_over_scan and kappa.")

    tag = res["host"]["tag"] or socket.gethostname().split(".")[0]
    prev = [int(re.search(r"-r(\d+)\.json$", f).group(1))
            for f in glob.glob(os.path.join(HERE, "cross_machine_%s-r*.json" % tag))]
    idx = (max(prev) + 1) if prev else 1
    out = os.path.join(HERE, "cross_machine_%s-r%d.json" % (tag, idx))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote", out)
    if idx > 1:
        print("this tag now has %d runs; `--merge %s` aggregates them" % (idx, tag))


if __name__ == "__main__":
    main()
