#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn n0_holdout.json into the table the paper would print.

Three questions, in order of how much they matter:

1.  Is kappa a number?  Both cost curves are only piecewise linear (the scan
    leaves L2 above ~16k rows; the retrieval curve steepens above ~4k
    candidates).  A single slope fitted on [512, 65536] therefore gives one
    kappa and a slope fitted on [512, 16384] gives another.  Report both.
2.  Does the model reproduce the decision at the corpus point?  Compare the
    predicted threshold k* = kappa*n - Delta_p with the threshold the measured
    curves actually cross at n = 394.
3.  Does the corrected instrument change Table VII?  Print old against new.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "n0_holdout.json"), encoding="utf-8"))
BASE = os.path.abspath(os.path.join(HERE, "..", "n0_robust", "n0_robust.json"))
OLD = {r["backend"]: r for r in json.load(open(BASE, encoding="utf-8"))["results"]}

ORDER = ["numpy", "pyloop", "sqlite", "java c2", "java c1", "java int"]

# The corpus point is computed, not asserted: exp/corpus_point.py reads the 300
# scopes and band sizes out of exp/duel_repeat/duel_rep.json and writes this
# file.  n = 394 reproduces exactly (it is that stream's median scope).  k does
# not: the stream's median band is 99, and the 182 this script used to print is
# its 76.7th percentile, with no correction at that value.  Both are carried
# below so the tables can be read either way, but only K_CORPUS is measured.
CP = json.load(open(os.path.join(HERE, "corpus_point.json"), encoding="utf-8"))
N_CORPUS = int(CP["corpus_point"]["n"])
K_CORPUS = int(CP["corpus_point"]["k"])
K_PAPER = int(CP["paper_point"]["k"])
CP_SRC = CP["source"]["stream"]


def slope(xs, ys):
    a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(a), float(b)


def kappa_at(b, nmax, kmax):
    tR, tB = b["tR"], b["tB"]
    ns = [int(k) for k in tR if 512 <= int(k) <= nmax and tR[k] >= 100]
    ks = [int(k) for k in tB if 8 <= int(k) <= kmax and tB[k] >= 100]
    if len(ns) < 2 or len(ks) < 2:
        return None
    aS, _ = slope(ns, [tR[str(n)] for n in ns])
    aP, _ = slope(ks, [tB[str(k)] for k in ks])
    FR, FB = b["coef"]["FR"], b["coef"]["FB"]
    return dict(kappa=aS / aP, dp=(FB - FR) / aP, aS=aS, aP=aP,
                n=[min(ns), max(ns)], k=[min(ks), max(ks)])


def measured_kstar(b):
    """Largest k at which the measured band curve is still under tR(394)."""
    tR, tB = b["tR"], b["tB"]
    if str(N_CORPUS) not in tR:
        return None
    r = tR[str(N_CORPUS)]
    ks = sorted(int(k) for k in tB)
    prev = 0.0
    for k in ks:
        if tB[str(k)] >= r:
            if k == ks[0]:
                return 0.0 if tB[str(k)] >= r else None
            k0, t0 = prev, tB[str(prev_k)]
            # linear interpolation between the bracketing points
            frac = (r - t0) / (tB[str(k)] - t0) if tB[str(k)] > t0 else 0.0
            return round(prev_k + frac * (k - prev_k), 0)
        prev_k, prev = k, tB[str(k)]
    return float("inf")


def main():
    print("corpus point computed from %s: n=%d, k=%d; the paper reads k=%d "
          "(%s)" % (CP_SRC, N_CORPUS, K_CORPUS, K_PAPER,
                    "same" if K_CORPUS == K_PAPER else "does not reproduce"))
    print("\nTable A -- kappa depends on the range it is fitted on (instance F)\n")
    inst = J["instances"][0]
    print("%-9s %10s %10s %10s %10s %10s" %
          ("backend", "kappa[512,64k]", "kappa[512,16k]", "k*[512,64k]",
           "k*[512,16k]", "k* measured"))
    rows = []
    for name in ORDER:
        b = inst["backends"][name]
        a = kappa_at(b, 65536, 4096)
        c = kappa_at(b, 16384, 4096)
        mk = measured_kstar(b)
        ka = a["kappa"] * N_CORPUS - a["dp"] if a else float("nan")
        kc = c["kappa"] * N_CORPUS - c["dp"] if c else float("nan")
        rows.append((name, a, c, ka, kc, mk))
        print("%-9s %10.3f %10.3f %10.0f %10.0f %10s" %
              (name, a["kappa"] if a else float("nan"),
               c["kappa"] if c else float("nan"), ka, kc,
               "%.0f" % mk if mk is not None else "-"))

    print("\nTable B -- published versus corrected (instance F, fit [512,64k]/[8,4k]);"
          " verdicts read against k=%d, the computed corpus band" % K_CORPUS)
    print("%-9s %8s %8s %8s %8s %10s %10s" %
          ("backend", "k old", "k new", "k* old", "k* new", "verdict old", "verdict new"))
    for name, a, c, ka, kc, mk in rows:
        o = OLD[name]
        ko = o["kappa"]
        n0 = o["n0_rows"]
        dpo = o["delta_p_cands"]
        kso = ko * N_CORPUS - dpo
        print("%-9s %8.3f %8.3f %8.0f %8.0f %10s %10s" %
              (name, ko, a["kappa"] if a else float("nan"), kso, ka,
               "band" if kso > K_CORPUS else "scan",
               "band" if ka > K_CORPUS else "scan"))

    print("\nTable C -- transfer: kappa refitted on each instance, same ranges\n")
    print("%-9s" % "backend" + "".join("%16s" % i["tag"] for i in J["instances"]))
    for name in ORDER:
        cells = []
        for inst in J["instances"]:
            a = kappa_at(inst["backends"][name], 65536, 4096)
            cells.append("%16.3f" % a["kappa"] if a else "%16s" % "-")
        print("%-9s" % name + "".join(cells))

    print("\nTable D -- corpus point (n=394, k=%d) measured directly" % K_PAPER)
    print("  k=%d is a point on the holdout grid, not the computed band (%d is"
          " off-grid, between 64 and 128); read this panel as 'what the measured"
          " curves do at the value the paper quotes'." % (K_PAPER, K_CORPUS))
    print("%-9s" % "backend" + "".join("%18s" % i["tag"] for i in J["instances"]))
    for name in ORDER:
        cells = []
        for inst in J["instances"]:
            b = inst["backends"][name]
            tr = b["tR"].get(str(N_CORPUS))
            tb = b["tB"].get(str(K_PAPER))
            if tr is None or tb is None:
                cells.append("%18s" % "-")
                continue
            w = "band" if tb < tr else "scan"
            cells.append("%18s" % ("%s %.0f/%.0f" % (w, tr, tb)))
        print("%-9s" % name + "".join(cells))

    print("\nTable E -- holdout error and misrouting cost\n")
    print("%-9s %-14s %8s %8s %8s %8s %8s" %
          ("backend", "instance", "MAPE_R", "MAPE_B", "agree", "regret", "pairs"))
    for inst in J["instances"]:
        for name in ORDER:
            h = inst["backends"][name]["holdout"]
            agree = h["agreements"]
            tot = sum(v["n"] for v in agree.values())
            ok = sum(v["ok"] for v in agree.values())
            print("%-9s %-14s %7.1f%% %7.1f%% %7.1f%% %8.3f %8d" %
                  (name, inst["tag"], h["mape_R"], h["mape_B"],
                   100 * ok / tot, h["regret"], tot))



def write_summary():
    """The numbers Table VII prints, in the form reproduce.py asserts."""
    inst = J["instances"][0]
    out = {"fitted_instance": inst["tag"], "n_corpus": N_CORPUS,
           # k_corpus is the measured value; k_corpus_paper is what Table VII
           # reads.  They differ by 84% -- see exp/corpus_point.py.
           "k_corpus": K_CORPUS, "k_corpus_paper": K_PAPER,
           "corpus_point_source": CP_SRC, "backends": {}}
    for name in ORDER:
        ks = [i["backends"][name]["coef"]["kappa"]
              * (N_CORPUS - i["backends"][name]["coef"]["n0"])
              for i in J["instances"]]
        c = inst["backends"][name]["coef"]
        out["backends"][name] = dict(
            kappa=round(c["kappa"], 3), n0=round(c["n0"], 1),
            dp=round(c["dp"], 2),
            kstar=round(c["kappa"] * (N_CORPUS - c["n0"]), 1),
            kstar_min=round(min(ks)), kstar_max=round(max(ks)))
    fp = os.path.join(HERE, "n0_holdout_summary.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("\nwrote", os.path.basename(fp))


if __name__ == "__main__":
    main()
    write_summary()
