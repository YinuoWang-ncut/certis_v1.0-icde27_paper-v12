#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 2: the band's cost against the scope recompute, as a function of scope.

Why this figure
---------------
The paper's claim is one sentence -- the band's relative cost falls as the scope
grows and crosses one at a scope the criterion names in advance -- and until this
file existed a reader had to assemble that sentence out of four numbers in
Sect. V-A, a table and a paragraph of prose.  One panel makes it one look.

What it plots is T_band / T_scope for the *fixed* band path (no probe), because
that is the quantity eq. (9) prices; the routed policy adds a probe that cancels
in the comparison and is reported separately in the text.

Where the numbers come from: `exp/out/mp_run*.json`, three independent processes
of `masked_pipeline.py` at 21 repetitions each, all under the build rule of
Sect. V-A.  The point is the median over processes; the bar is the union of the
per-process 95% paired-bootstrap intervals, which is conservative -- it is wider
than any single run's, because this box's run-to-run spread exceeds its
within-run bootstrap width.

The curve is not monotone and the figure says so: the ratio falls to a minimum at
the median scope and returns to parity at the upper quartile, because the mask
adds a term proportional to k to both paths and k keeps growing after the scan's
per-row cost has flattened.  Only the first crossing is a prediction of eq. (9)
at this density, and it sits inside the interval the criterion marks.

  python fig_ratio.py          # writes paper/fig_ratio.{pdf,png}
"""
import glob
import argparse
import json
import shutil
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import matplotlib.ticker as mticker                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
# The artifact writes its own figure.  A path into a manuscript directory is a
# build hook, not a dependency: if CERTIS_PAPER is set the figure is also copied
# there, and if it is not, nothing outside this tree is touched.  A release that
# assumes a sibling manuscript tree is a release that cannot be unpacked.
FIGS = os.path.join(OUT, "figs")
PAPER = os.environ.get("CERTIS_PAPER")

DPI = 400
W_IN, H_IN = 3.36, 1.70
KAPPA, DP = 0.2142, 198.8
DENS = 0.183


def load():
    runs = [json.load(open(f)) for f in sorted(
        glob.glob(os.path.join(OUT, "mp_run*.json")))]
    if not runs:
        sys.exit("no mp_run*.json: run masked_pipeline.py three times first")
    return runs


def points(runs):
    out = []
    idxs = [i for i, r in enumerate(runs[0]["rows"])
            if abs(r["dens"] - DENS) < 0.01]
    for i in idxs:
        ratio, lo, hi, ks = [], [], [], []
        for r in runs:
            row = r["rows"][i]
            if abs(row["dens"] - DENS) >= 0.01:
                continue
            ratio.append(row["ratio"])
            lo.append(row["ratio_ci"][0])
            hi.append(row["ratio_ci"][1])
            ks.append(row["k"])
        out.append(dict(n=runs[0]["rows"][i]["n"], k=int(np.median(ks)),
                        ratio=float(np.median(ratio)),
                        lo=float(min(lo)), hi=float(max(hi)),
                        runs=[round(v, 4) for v in ratio]))
    return sorted(out, key=lambda c: c["n"])


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="where to write the points record; reproduce.py passes "
                         "a scratch path so a gate run cannot replace the curated "
                         "fig_ratio.json")
    ap.add_argument("--figdir", default=None,
                    help="where to write the figure; default exp/out/figs")
    ap.add_argument("--no-copy", action="store_true",
                    help="do not overwrite the manuscript's copy of the figure")
    return ap.parse_args()


def main():
    ARGS = _args()
    FIGS = ARGS.figdir or FIGS
    pts = points(load())
    for p in pts:
        print("n %7d k %7d  T_B/T_R %.3f [%.3f, %.3f]  runs %s"
              % (p["n"], p["k"], p["ratio"], p["lo"], p["hi"], p["runs"]))

    ns = np.array([p["n"] for p in pts], float)
    rr = np.array([p["ratio"] for p in pts], float)
    lo = np.array([p["lo"] for p in pts], float)
    hi = np.array([p["hi"] for p in pts], float)

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    fig.subplots_adjust(left=0.150, right=0.978, top=0.885, bottom=0.235)

    ax.axhline(1.0, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=1)
    # No connecting line.  These are four measured scopes, not samples of a
    # continuous function, and a line between them asserts a shape the intervals
    # do not support -- at 413k the interval spans both sides of one.
    ax.errorbar(ns, rr, yerr=[rr - lo, hi - rr], fmt="o", ms=3.8, mfc="#1d1d1d",
                mec="#1d1d1d", mew=0.4, ecolor="#1d1d1d", elinewidth=0.7,
                capsize=1.9, capthick=0.7, zorder=4)

    n0 = DP / KAPPA                       # the floor: no k helps below it
    nstar = DP / (KAPPA - DENS)           # where the two paths tie at this density
    for x, lab in ((n0, r"$n_0$"), (nstar, r"$n^{*}$")):
        ax.axvline(x, color="#8a3b12", lw=0.65, ls=(0, (1.6, 1.7)), zorder=2)
        ax.annotate(lab, xy=(x, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(2.0, 3.0), textcoords="offset points", fontsize=5.6,
                    color="#8a3b12", ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(140, 6.2e5)
    ax.set_ylim(0.775, 1.125)
    ax.set_xticks([197, 1000, 6969, 44606, 412875])
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter(["197", "1k", "7k", "45k", "413k"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([0.85, 0.90, 0.95, 1.00, 1.05])
    ax.yaxis.set_major_formatter(
        mticker.FixedFormatter(["0.85", "0.90", "0.95", "1.00", "1.05"]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("scope length $n$ (beneficiary-days, log scale)", fontsize=6.3,
                  labelpad=2)
    ax.set_ylabel("$T_{\\mathrm{band}}/T_{\\mathrm{scope}}$", fontsize=6.3,
                  labelpad=3)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", labelsize=5.7, length=2.3,
                   width=0.65, pad=1.5)

    # Three labels, placed where nothing else is: the number line at 1.0 is the
    # only horizontal the eye needs.
    ax.annotate("band dearer", xy=(2.05e2, 1.072), fontsize=5.5, color="#a3282b",
                ha="left", va="center")
    ax.annotate("band cheaper", xy=(5.0e3, 0.828), fontsize=5.5, color="#1f5e3a",
                ha="left", va="center")
    ax.annotate("both crossings lie\nbetween $n_0$ and $n^{*}$",
                xy=(2.05e2, 0.948), fontsize=4.9, color="#555555",
                ha="left", va="top", linespacing=1.3)

    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = os.path.join(FIGS, "fig_ratio." + ext)
        fig.savefig(dest, dpi=DPI)
        print("wrote exp/out/figs/fig_ratio." + ext)
        if PAPER and os.path.isdir(PAPER) and not ARGS.no_copy:
            shutil.copyfile(dest, os.path.join(PAPER, "fig_ratio." + ext))
            print("  copied to %s" % PAPER)

    json.dump(dict(points=pts, n0=n0, n_star=nstar, density=DENS,
                   quantity="T_band/T_scope at fixed k (no probe)",
                   sources="exp/out/mp_run*.json"),
              open(ARGS.out or os.path.join(OUT, "fig_ratio.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
