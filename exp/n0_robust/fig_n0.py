#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two panels for the n0 robustness section.

(a) the floor per configuration, with its 90% interval, against the corpus
    median scope;
(b) the same six configurations in the only units the decision uses: the band
    k* = kappa(n - n0) each tolerates at n = 394, against the k = 182 the corpus
    actually needs.  Both axes are ratios of times, so hardware drops out.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "n0_robust_fix.json"), encoding="utf-8"))
N, K = F["corpus"]["n"], F["corpus"]["k"]        # 394, 182
ROWS = [{"backend": c["backend"], "n0_rows": c["n0"],
         "n0_ci": c["n0_ci"], "kappa": c["kappa"], "kstar": c["kstar"],
         "kstar_ci": c["kstar_ci"],
         "win": c["win"]} for c in F["configs"]]

BLUE, RED = "#2166ac", "#b2182b"


def kstar(r):
    """tolerated band, with an envelope over the two fitted intervals."""
    return r["kstar"], r["kstar_ci"]


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#333",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.dpi": 300,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6),
                               gridspec_kw={"width_ratios": [1.06, 1.0]})

# ---------------------------------------------------------------- panel (a)
rows = sorted(ROWS, key=lambda r: r["n0_rows"])
ys = range(len(rows))
for i, r in enumerate(rows):
    lo, hi = r["n0_ci"]
    c = RED if r["win"] else BLUE
    ax1.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="butt", alpha=0.5)
    ax1.plot([r["n0_rows"]], [i], "o", ms=4.2, color=c, mec="white", mew=0.7, zorder=3)

ax1.axvline(N, color="#444", lw=0.8, ls=(0, (4, 2)), zorder=1)
ax1.text(N - 10, -1.02, "corpus scope %d rows" % N,
         ha="right", va="bottom", fontsize=6.6, color="#555")
ax1.set_yticks(list(ys))
ax1.set_yticklabels([r["backend"] for r in rows], fontsize=7)
ax1.set_xlabel(r"floor $n_0$ (rows), 90% interval", fontsize=7.5)
ax1.set_xlim(-110, max(470, N + 46))
ax1.set_ylim(-1.15, len(rows) - 0.35)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.set_title("(a) the floor, six configurations of one semantics",
              fontsize=7.6, pad=4, loc="left")

# ---------------------------------------------------------------- panel (b)
rows2 = sorted(ROWS, key=lambda r: kstar(r)[0])
ys2 = range(len(rows2))
ax2.axvspan(-1e9, K, color="#eef3f8", zorder=0)
for i, r in enumerate(rows2):
    k, (klo, khi) = kstar(r)
    c = RED if r["win"] else BLUE
    ax2.plot([max(klo, 0), khi], [i, i], color=c, lw=1.6, solid_capstyle="butt", alpha=0.5)
    ax2.plot([k], [i], "o", ms=4.2, color=c, mec="white", mew=0.7, zorder=3)
    ax2.annotate("%d" % round(k), (khi, i), textcoords="offset points",
                 xytext=(4, -2.4), fontsize=6.2, color="#555")

ax2.axvline(K, color="#444", lw=0.9, ls=(0, (4, 2)), zorder=1)
ax2.text(K - 8, len(rows2) - 0.45, "corpus needs\nk = %d" % round(K),
         ha="right", va="top", fontsize=6.6, color="#333")
ax2.set_yticks(list(ys2))
ax2.set_yticklabels([r["backend"] for r in rows2], fontsize=7)
ax2.set_xlabel(r"band tolerated $k^{*}=\kappa(394-n_0)$, candidates", fontsize=7.5)
ax2.set_xlim(0, 760)
ax2.set_ylim(-0.8, len(rows2) - 0.35)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.text(0.02, 0.60, "band loses", fontsize=6.8, color=BLUE, ha="left", va="center",
         transform=ax2.transAxes)
ax2.text(0.97, 0.04, "band wins", fontsize=6.8, color=RED, ha="right", va="bottom",
         transform=ax2.transAxes)
ax2.set_title(r"(b) the band each tolerates, against what the corpus needs",
              fontsize=7.6, pad=4, loc="left")

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_n0." + ext), bbox_inches="tight")
print("wrote fig_n0.pdf/.png")
for r in rows2:
    k, ci = kstar(r)
    print("  %-16s k*=%6.1f  [%5.1f,%5.1f]  %s" % (
        r["backend"], k, ci[0], ci[1], "WIN" if r["win"] else "lose"))

