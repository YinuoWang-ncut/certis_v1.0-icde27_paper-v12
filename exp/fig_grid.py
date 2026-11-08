#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two panels for the calibration grid.

Left  -- the sign surface, measured the way a comparison has to be measured:
         seventy cells inside *one* store of fixed size and one layout, so that
         |D| cannot move between two cells.  Band/scan against scope, one thin
         line per band density, log x.  The lines do not collapse: at 412,875
         rows the same scope spans 0.52 to 1.83 as the density goes from 0.02 to
         0.60, and the density at which a line crosses parity moves right as the
         scope grows.  At 197 and 6,969 rows no density crosses at all.

         The earlier version of this panel drew the same axes from
         `grid80.py`, one store per scope, and there the ten densities did
         nearly collapse into one curve.  That collapse was the store growing
         with the scope, not a property of the comparison, which is what the
         right panel isolates.

Right -- the confound: at one fixed cell (n = 412,875, k = 75,556) the ratio
         moves from 0.37 to 1.02 as the store grows from one code to sixteen.
         c_R is flat; c_B is not.  A cell is therefore only comparable to another
         cell drawn from a store of the same size.

Output: paper/fig_grid.pdf (and a PNG for eyeballing).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Two layouts: inside the manuscript tree the records live in the artifact
# checkout beside it; inside the released archive they are exp/out/ right here.
# Resolving this way is what lets a reviewer regenerate the figure from the
# archive alone, which is the only place the numbers are supposed to come from.
if os.path.exists(os.path.join(HERE, "out", "same_store.json")):
    ART = HERE
else:
    JOURNAL = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    ART = os.path.join(JOURNAL, "2026-11-06-21-54-20", "release", "certis", "exp")
OUT = os.path.join(HERE, "..", "paper")

INK = "#1b2430"
MUTED = "#8a97a6"
RED = "#8a3b12"
GRID = "#e6eaee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "xtick.color": "#4a5764", "ytick.color": "#4a5764",
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    g = json.load(open(os.path.join(ART, "out", "same_store.json"),
                       encoding="utf-8"))
    s = json.load(open(os.path.join(ART, "out", "store_size.json"),
                       encoding="utf-8"))
    rows = g["rows"]
    dens = sorted({round(r["d_target"], 4) for r in rows})
    ns = sorted({r["n"] for r in rows})
    idx = {(r["n"], round(r["d_target"], 4)): r for r in rows}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.42, 1.30),
                                  gridspec_kw=dict(width_ratios=[1.42, 1]))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.94, bottom=0.275,
                        wspace=0.46)

    # ---------------- left: the sign surface, one store ----------------------
    ax.axhspan(0.80, 1.20, color=GRID, zorder=0)           # the tie band
    ax.axhline(1.0, color=INK, lw=0.7, zorder=3)
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(dens):
        ys = [idx[(n, d)]["ratio"] for n in ns]
        ax.plot(ns, ys, color=cmap(i / (len(dens) - 1.0)), lw=0.75,
                alpha=0.95, zorder=4, solid_capstyle="round")
    # name the two densities whose lines bound the fan
    for i, lab in ((0, "0.02"), (len(dens) - 1, "0.60")):
        ys = [idx[(n, dens[i])]["ratio"] for n in ns]
        ax.annotate(lab, xy=(ns[-1], ys[-1]), xytext=(3, 0),
                    textcoords="offset points", fontsize=5.6,
                    color=cmap(i / (len(dens) - 1.0)), va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(150, 1.1e6)
    ax.set_ylim(0.40, 2.25)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("scope $n$, rows")
    ax.set_ylabel("$T_B/T_R$")

    ax.text(175, 2.22, "band dearer", fontsize=5.8, color=MUTED, va="top")
    ax.text(175, 0.425, "band cheaper", fontsize=5.8, color=MUTED, va="bottom")
    ax.annotate("nothing wins\nbelow $2\\cdot10^{4}$", xy=(3200, 1.19),
                xytext=(2600, 1.84), fontsize=5.6, color=RED, ha="left",
                va="bottom", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=RED,
                                shrinkA=1.5, shrinkB=1.5))

    # ---------------- right: the confound -----------------------------------
    lines = [r["store_lines"] for r in s["rows"]]
    rat = [r["ratio"] for r in s["rows"]]
    cB = [r["c_B_ns"] for r in s["rows"]]
    ax2.axhline(1.0, color=INK, lw=0.7)
    ax2.plot(lines, rat, "o-", color=RED, ms=2.5, lw=0.8,
             markeredgecolor="white", markeredgewidth=0.35)
    ax2.set_xscale("log")
    ax2.set_xlim(3.0e5, 1.1e7)
    ax2.set_ylim(0.25, 1.20)
    ax2.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax2.set_xticks([4e5, 1e6, 3e6, 6e6])
    ax2.set_xticklabels(["0.4M", "1M", "3M", "6M"])
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_xlabel("store, lines")
    ax2.set_ylabel("$T_B/T_R$", labelpad=1.5)
    ax2.text(0.045, 0.975, "one cell: $n=412{,}875$,\n$k=75{,}556$",
             transform=ax2.transAxes, fontsize=5.7, color=MUTED, va="top",
             linespacing=1.2)
    ax2.text(0.97, 0.055, "$c_B$ %.1f$\\to$%.1f ns\n$c_R$ flat %.1f"
             % (cB[0], cB[-1], s["rows"][0]["c_R_ns"]),
             transform=ax2.transAxes, fontsize=5.6, color=RED, va="bottom",
             ha="right", linespacing=1.25)

    for a in (ax, ax2):
        a.tick_params(length=2, width=0.6, pad=1.6)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "fig_grid.%s" % ext), dpi=400)
    print("wrote paper/fig_grid.pdf")
    print("left: %d scopes x %d densities, one store of %d rows"
          % (len(ns), len(dens), g["D"]))
    for n in ns:
        v = [idx[(n, d)]["ratio"] for d in dens]
        print("   n %7d  %.2f - %.2f" % (n, min(v), max(v)))
    print("right: ratio %.3f -> %.3f over %.1fx store"
          % (rat[0], rat[-1], lines[-1] / lines[0]))


if __name__ == "__main__":
    main()
