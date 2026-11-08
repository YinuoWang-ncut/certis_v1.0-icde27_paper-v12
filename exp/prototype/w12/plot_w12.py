"""W12 figure 1: what actually decides the path, and it is not the ratio.

Left: the two paths on the same 300 corrections, against |B|/|scope| -- the
variable the paper's criterion uses.  The band loses everywhere, including where
the criterion says it should win (below C_HAT).

Right: the same 300 corrections against |scope| -- the variable the two paths'
fixed costs turn on.  The ordering reverses above ~10^3 rows.

Drawn from w12_results.json; no number here is typed by hand.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "w12_results.json"), encoding="utf-8") as fh:
    R = json.load(fh)

du = R["duel_cell"]["duel"]
ns = np.array([d["n_scope"] for d in du], dtype=float)
nb = np.array([d["n_band"] for d in du], dtype=float)
tr = np.array([d["tR"] for d in du], dtype=float) * 1e6
tb = np.array([d["tB"] for d in du], dtype=float) * 1e6
fr = np.array([d["frac"] for d in du], dtype=float)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#888",
})
C_HAT = 0.05
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)

# ---------------- left: binned by the ratio the criterion uses
bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.51)]
ctr, vr, vb, cnt = [], [], [], []
for lo, hi in bins:
    m = (fr >= lo) & (fr < hi)
    if m.sum() == 0:
        continue
    ctr.append((lo + hi) / 2 if hi <= 0.25 else 0.375)
    vr.append(np.median(tr[m]))
    vb.append(np.median(tb[m]))
    cnt.append(int(m.sum()))
ax[0].axvspan(0, C_HAT, color="#f0e6c8", alpha=0.55, zorder=0)
ax[0].plot(ctr, vr, "o-", color="#b3261e", lw=1.4, ms=4, label="路径 R（扫描作用域）")
ax[0].plot(ctr, vb, "s-", color="#1b4f8a", lw=1.4, ms=4, label="路径 B（带探测）")
ax[0].set_xscale("log")
ax[0].set_xlim(3e-3, 0.6)
ax[0].set_ylim(8, 30)
ax[0].set_xlabel(r"$|B|/|\sigma|$  —— 论文判据使用的变量")
ax[0].set_ylabel("单条修正延迟 (µs)")
ax[0].set_title("按判据变量分桶：B 从不取胜", fontsize=9.5)
ax[0].text(C_HAT * 1.15, 27.6, r"$\hat C=5\%$", fontsize=8, color="#8a5a00")
ax[0].grid(alpha=0.22, lw=0.6)
ax[0].legend(fontsize=7.4, loc="lower center", ncol=2, frameon=False,
             handlelength=1.6, columnspacing=1.2)

# ---------------- right: binned by |scope|
sb = [(0, 250), (250, 600), (600, 1200), (1200, 3000), (3000, 9000), (9000, 1e9)]
cs, vs_r, vs_b = [], [], []
for lo, hi in sb:
    m = (ns >= lo) & (ns < hi)
    if m.sum() < 3:
        continue
    cs.append(np.sqrt(max(lo, 10) * min(hi, 2e4)))
    vs_r.append(np.median(tr[m]))
    vs_b.append(np.median(tb[m]))
s_star = R.get("switch", {}).get("sigma_star", float("nan"))
if s_star == s_star:
    ax[1].axvline(s_star, color="#8a5a00", ls="--", lw=1.0, zorder=0)
    ax[1].text(s_star * 1.10, 11.5, f"拟合切换点 {s_star:.0f} 行", fontsize=7.0,
               color="#8a5a00", va="bottom")
ax[1].plot(cs, vs_r, "o-", color="#b3261e", lw=1.4, ms=4)
ax[1].plot(cs, vs_b, "s-", color="#1b4f8a", lw=1.4, ms=4)
ax[1].set_xscale("log")
ax[1].set_xlim(250, 1.2e4)
ax[1].set_ylim(8, 95)
ax[1].set_xlabel(r"$|\sigma|$  —— 固定成本摊销在它上面")
ax[1].set_ylabel("单条修正延迟 (µs)")
ax[1].set_title("按作用域绝对大小分桶：排序反转", fontsize=9.5)
ax[1].grid(alpha=0.22, lw=0.6, which="both")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout(pad=0.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"w12_fig1.{ext}"),
                bbox_inches="tight", facecolor="white")
print("wrote w12_fig1.png / w12_fig1.pdf")
print("left bins:", list(zip([f"{c:.3f}" for c in ctr], cnt,
                             [f"{v:.1f}" for v in vr], [f"{v:.1f}" for v in vb])))
print("right bins:", [(f"{c:.0f}", f"{a:.1f}", f"{b:.1f}") for c, a, b in zip(cs, vs_r, vs_b)])
