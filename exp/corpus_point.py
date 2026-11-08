#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The corpus point, computed instead of asserted.

Both numbers the paper reads every fitting against -- n = 394 and k = 182 --
were literals in the artifact.  ``exp/n0_holdout/analyze.py`` printed 182 into
``n0_holdout_summary.json``; ``exp/accounting_table.py`` held its own copy;
``exp/n0_impl/dline.py`` a third.  A reviewer who asks "where does 182 come
from" gets no answer from the code, only a comment saying "corpus median band".

This script answers it.  The corpus the paper's constants were fitted on is the
300-correction stream recorded in ``exp/duel_repeat/duel_rep.json`` -- the same
stream ``run_duel_big.py`` re-runs from the same seed.  Both n and k are
recorded per correction, so both can be read off at any quantile.

The point of the exercise is that the two numbers are *not* read at the same
quantile.  n = 394 is the median of the 300 scopes, exactly and reproducibly.
k = 182 is the 76.7th percentile of the 300 band sizes, and no correction in
that stream has k = 182.  Taking a p50 of one and a p77 of the other inflates
the line by 84% -- and, as ``verdicts`` below shows, that inflation is what
decides whether the python-loop layout admits the band in Table VIII.

  python corpus_point.py            # audit to stdout, write the json
  python corpus_point.py --check    # exit 1 if the paper's k does not reproduce
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STREAM = os.path.join(HERE, "duel_repeat", "duel_rep.json")
BIG = os.path.join(HERE, "duel_big", "duel_big_run1.json")
SUMMARY = os.path.join(HERE, "n0_holdout", "n0_holdout_summary.json")
OUT = os.path.join(HERE, "n0_holdout", "corpus_point.json")

PAPER = {"n": 394, "k": 182}
Q = [0.10, 0.25, 0.50, 0.75, 0.90]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def boot_median(x, reps=4000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def split_valley(n):
    """The mix is 50/50 cap_low/cap_high by construction, so n is bimodal.
    Take the widest multiplicative gap between 300 and 3000 as the valley."""
    u = np.unique(n[(n >= 300) & (n <= 3000)])
    g = np.diff(np.log(u))
    i = int(np.argmax(g))
    return float(np.sqrt(u[i] * u[i + 1]))


def main():
    check = "--check" in sys.argv
    d = load(STREAM)
    n = np.asarray(d["n_scope"], float)
    k = np.asarray(d["n_band"], float)
    assert len(n) == len(k) and len(n) == d["n_corr"], (len(n), d["n_corr"])

    n50, k50 = float(np.median(n)), float(np.median(k))
    pct_k_of_paper = float((k < PAPER["k"]).mean())
    hit_k = int((k == PAPER["k"]).sum())

    # is 394 the p50 to the digit, or a round-off that happens to look like one?
    srt = np.sort(n)
    n_mid = (srt[len(srt) // 2 - 1], srt[len(srt) // 2])

    qtab = []
    for q in Q:
        qtab.append(dict(q=q, n=float(np.quantile(n, q)), k=float(np.quantile(k, q)),
                         density=float(np.quantile(k, q) / np.quantile(n, q))))

    thr = split_valley(n)
    lo, hi = n <= thr, n > thr
    regimes = []
    for tag, sel in (("narrow (low-cap)", lo), ("wide (high-cap)", hi)):
        regimes.append(dict(
            tag=tag, count=int(sel.sum()),
            med_n=float(np.median(n[sel])), med_k=float(np.median(k[sel])),
            med_frac=float(np.median(k[sel] / n[sel])),
            med_speedup=None,
            k_range=[int(k[sel].min()), int(k[sel].max())]))

    # the six fittings, re-judged at the reproduced k and at the paper's k
    S = load(SUMMARY)["backends"]
    verdicts = []
    for name, b in S.items():
        kn = b["kappa"] * (PAPER["n"] - b["n0"])
        verdicts.append(dict(
            layout=name, kappa=b["kappa"], n0=b["n0"], dp=b["dp"],
            kstar=round(kn, 1),
            at_paper=("band" if kn > PAPER["k"] else "scan"),
            at_reproduced=("band" if kn > k50 else "scan")))
    verdicts.sort(key=lambda r: -r["kstar"])
    flips = [v["layout"] for v in verdicts if v["at_paper"] != v["at_reproduced"]]

    # crossovers: the two thresholds the paper quotes, at both k's
    canon = dict(kappa=0.4414, dp=316.0)
    refit = dict(kappa=0.48067237250629136, dp=368.26010181747165)
    cross = []
    for tag, c in (("paper", canon), ("refit600", refit)):
        row = dict(source=tag, kappa=c["kappa"], dp=c["dp"],
                   free_veto=round(c["dp"] / c["kappa"], 1))
        for kk, lab in ((PAPER["k"], "at_k182"), (k50, "at_k%d" % round(k50))):
            row[lab] = round((c["dp"] + kk) / c["kappa"], 1)
        cross.append(row)

    sk = np.sort(k)
    nearest = float(sk[int(np.abs(sk - PAPER["k"]).argmin())])
    repro = dict(
        n=dict(paper=PAPER["n"], got=n50, ok=bool(n50 == PAPER["n"]),
               mid_two=[float(x) for x in n_mid]),
        k=dict(paper=PAPER["k"], got=k50, ok=bool(k50 == PAPER["k"]),
               paper_percentile=pct_k_of_paper,
               observed_at_paper_value=hit_k,
               nearest_observed=nearest))

    out = dict(
        source=dict(stream=os.path.relpath(STREAM, HERE), sha256=sha(STREAM),
                    corrections=int(len(n)),
                    seed=d.get("seed"), n_batch=d.get("n_batch"),
                    mix=d.get("mix"),
                    big_stream=dict(path=os.path.relpath(BIG, HERE),
                                    sha256=sha(BIG),
                                    median_n=float(np.median(load(BIG)["n_scope"])),
                                    median_k=float(np.median(load(BIG)["n_band"])))),
        corpus_point=dict(n=n50, k=k50,
                          n_ci95=list(boot_median(n)), k_ci95=list(boot_median(k))),
        paper_point=PAPER,
        reproduction=repro,
        quantiles=qtab,
        regimes=dict(split_at=thr, rows=regimes),
        verdicts=verdicts, flips=flips, crossovers=cross)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")

    # ---- audit ----------------------------------------------------------
    print("stream  %s  (%d corrections, seed %s)"
          % (os.path.basename(STREAM), len(n), d.get("seed")))
    print("sha256  %s" % out["source"]["sha256"][:32])
    print("\n%-6s %10s %10s %8s" % ("q", "n", "k", "k/n"))
    for r in qtab:
        print("%-6.2f %10.0f %10.0f %8.3f" % (r["q"], r["n"], r["k"], r["density"]))
    print("\ncorpus point at p50: n=%.0f [%.0f,%.0f]  k=%.0f [%.0f,%.0f] (95%% boot)"
          % (n50, *out["corpus_point"]["n_ci95"], k50, *out["corpus_point"]["k_ci95"]))
    print("paper reads          n=%d  k=%d" % (PAPER["n"], PAPER["k"]))
    print("  n: %s (the two middle scopes are %s)"
          % ("reproduces" if n50 == PAPER["n"] else "DOES NOT reproduce", n_mid))
    print("  k: %s -- 182 is the %.1f-th percentile of k; %d of %d corrections equal it"
          % ("reproduces" if k50 == PAPER["k"] else "DOES NOT reproduce",
             100 * pct_k_of_paper, hit_k, len(k)))
    print("  ratio paper/reproduced = %.2fx  (density %.3f vs %.3f)"
          % (PAPER["k"] / k50, PAPER["k"] / PAPER["n"], k50 / n50))

    print("\nthe corpus point is not one point (valley at n=%.0f)" % thr)
    for r in regimes:
        print("  %-18s %3d corrections  med n %5.0f  med k %6.0f  med k/n %.3f"
              % (r["tag"], r["count"], r["med_n"], r["med_k"], r["med_frac"]))

    print("\nTable VIII re-judged (k* = kappa(394 - n0))")
    print("  %-10s %8s %8s  %8s  %10s" % ("layout", "kappa", "k*", "@k=182", "@k=%d" % k50))
    for v in verdicts:
        print("  %-10s %8.3f %8.1f  %8s  %10s%s"
              % (v["layout"], v["kappa"], v["kstar"], v["at_paper"], v["at_reproduced"],
                 "   <- flips" if v["at_paper"] != v["at_reproduced"] else ""))
    print("  flips: %s" % (", ".join(flips) if flips else "none"))

    print("\ncrossovers (rows at which band breaks even)")
    print("  %-10s %8s %8s %12s %12s" % ("fit", "kappa", "dp", "n*(k=182)", "n*(k=%d)" % k50))
    for c in cross:
        print("  %-10s %8.4f %8.1f %12.1f %12.1f"
              % (c["source"], c["kappa"], c["dp"], c["at_k182"], c["at_k%d" % round(k50)]))
    print("\nwrote %s" % OUT)

    if check and (n50 != PAPER["n"] or k50 != PAPER["k"]):
        print("\n--check: the paper's corpus point does not reproduce", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
