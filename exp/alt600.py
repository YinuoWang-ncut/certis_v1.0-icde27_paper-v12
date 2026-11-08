#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The 600-correction replay, re-read under the alternating-order protocol.

What this is
------------
`run_duel_big.py` runs R and then B for every correction, so B always runs on a
heap R has just touched: B is never cold, R is always cold.  `veto_duel_rot.py`
runs each correction four times, alternating which arm goes first, so each arm
gets one cold and one warm observation and the mean of the two is symmetric.
That is the order-controlled replay.  It was run 23 times on the
same 600 corrections as the seven recorded runs (duel_big run 1--7); this
script asserts the streams are bit-identical and then re-derives every decision
number the paper prints from both protocols side by side.

What it decides
---------------
1. Arm-level bias.  Same 600 corrections, two protocols: what does the order
   do to median tR and tB, and hence to kappa?
2. Per-correction vote.  Seven recorded runs and nine alternating runs give 16
   observations per correction.  How many corrections does the band win in all
   of one family and none of the other, and how many is the answer protocol-
   dependent?
3. Criterion agreement.  Each run's own tuple decides each correction; the
   count it takes and the count it gets wrong, per protocol.
4. The sign at the corpus point.  Every run, both protocols, one question: is
   (n, k) = (394, 182) inside the band's acceptance region?
5. The routing surcharge measured rather than attributed: the paired
   difference tB - tR over the whole stream, under both protocols.

  python alt600.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REC = os.path.join(HERE, "duel_big")
SEED = 31337
N = 600
N_CORPUS, K_CORPUS = 394.0, 182.0
Q_PAPER = 3.0          # us, the paper's probe reading
Q_NUMPY = 8.1          # us, numpy probe interleaved with the scans it decides


def load(path, n=None):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    m = n or len(d["n_scope"])
    n_ = np.asarray(d["n_scope"][:m], dtype=float)
    k_ = np.asarray(d["n_band"][:m], dtype=float)
    tR = np.asarray(d["tR"][:m], dtype=float) * 1e6
    tB = np.asarray(d["tB"][:m], dtype=float) * 1e6
    return dict(file=os.path.basename(path), n=n_, k=k_, tR=tR, tB=tB,
                calib0=d.get("calib0"))


def two_lines(tR, tB, n, k):
    """The paper's two straight lines: tR = F_R + c_R n, tB = F_B + c_B k."""
    aR, bR = np.polyfit(n, tR, 1)          # us per row, us
    aB, bB = np.polyfit(k, tB, 1)          # us per candidate, us
    kappa = aR / aB
    p = (bB - bR) / aB                     # candidates, probe already paid
    out = dict(aR_ns=float(aR) * 1e3, bR_us=float(bR),
               aB_ns=float(aB) * 1e3, bB_us=float(bB),
               kappa=float(kappa), p=float(p), n=n, k=k)
    for tag, Q in (("q3.0", Q_PAPER), ("q8.1", Q_NUMPY)):
        dp = (bB - bR + Q) / aB
        out["dp_" + tag] = float(dp)
        out["n0_" + tag] = float(dp / kappa)
    return out


def r2(x, y):
    a, b = np.polyfit(x, y, 1)
    yh = a * x + b
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 0 else float("nan")


def vote(files):
    """Per-correction count of runs in which the band is faster than the scan."""
    v = None
    tR_all, tB_all = [], []
    for f in files:
        vv = (f["tB"] < f["tR"]).astype(int)
        v = vv if v is None else v + vv
        tR_all.append(f["tR"])
        tB_all.append(f["tB"])
    return v, np.vstack(tR_all), np.vstack(tB_all)


def hist(v, m):
    return [int(np.sum(v == i)) for i in range(m + 1)]


def main():
    # The alternating family is 23 processes: nine `veto_duel_rot_*` without a
    # calibration readout and fourteen `calib_series_*` with one.  Both run the
    # same 600 corrections in rotated order, so they are one protocol measured
    # twice over; the 14 are the ones P4 normalises to calib = 1332 us, which is
    # where the paper's 0.214 / 959 / 198.8 comes from.  Reporting the 9 alone
    # would give 0.196 / 1111 / 218 and a second alternating tuple nobody wants.
    rot = [load(p, N) for p in
           sorted(glob.glob(os.path.join(OUT, "veto_duel_rot_*.json")))]
    cal = [load(p, N) for p in
           sorted(glob.glob(os.path.join(OUT, "calib_series_*.json")))]
    alt = rot + cal
    rec = [load(p, N) for p in
           [os.path.join(REC, "duel_big_run%d.json" % i) for i in range(1, 8)]]
    if not alt or not rec:
        raise SystemExit("need out/veto_duel_rot_*.json, out/calib_series_*.json "
                         "and duel_big_run1..7")

    # ---- 0. the two protocols are the same 600 corrections ------------------
    same = all(np.array_equal(rec[0]["n"], f["n"]) and
               np.array_equal(rec[0]["k"], f["k"]) for f in alt + rec)
    print("stream identity across %d runs (%d recorded, %d alternating = "
          "%d rot + %d calib): %s"
          % (len(alt) + len(rec), len(rec), len(alt), len(rot), len(cal),
             "identical" if same else "DIFFERENT"))
    n, k = rec[0]["n"], rec[0]["k"]
    print("n: median %.0f [%.0f, %.0f]   k: median %.0f [%.0f, %.0f]   "
          "k/n median %.3f" % (np.median(n), n.min(), n.max(),
                               np.median(k), k.min(), k.max(),
                               np.median(k / np.maximum(n, 1))))
    print("corrections with n > 959 rows (order-controlled floor): %d; "
          "n > 1111: %d" % (int(np.sum(n > 959)), int(np.sum(n > 1111))))
    print()

    # ---- 1. fits, per protocol, per process --------------------------------
    print("1. per-process two-line fits")
    print("   %-8s %-6s %8s %8s %7s %7s %7s %7s %7s"
          % ("prot", "file", "c_R ns", "c_B ns", "kappa", "p", "dp(q3.0)",
             "n0(q3.0)", "R2 tR"))
    rows = {}
    for tag, fam in (("rec", rec), ("alt", alt)):
        rows[tag] = []
        for f in fam:
            ft = two_lines(f["tR"], f["tB"], f["n"], f["k"])
            ft["R2_tR"] = r2(f["n"], f["tR"])
            ft["R2_tB"] = r2(f["k"], f["tB"])
            ft["file"] = f["file"]
            ft["calib0"] = f["calib0"]
            rows[tag].append(ft)
            print("   %-8s %-6s %8.2f %8.2f %7.4f %7.1f %7.1f %7.0f %7.3f"
                  % (tag, f["file"].replace("veto_duel_rot_", "rot")
                     .replace("duel_big_run", "r").replace(".json", ""),
                     ft["aR_ns"], ft["aB_ns"], ft["kappa"], ft["p"],
                     ft["dp_q3.0"], ft["n0_q3.0"], ft["R2_tR"]))
    print()
    for tag in ("rec", "alt"):
        S = rows[tag]
        med = {kk: float(np.median([r[kk] for r in S]))
               for kk in ("aR_ns", "aB_ns", "kappa", "p", "dp_q3.0",
                          "n0_q3.0", "dp_q8.1", "n0_q8.1", "R2_tR", "R2_tB")}
        lo = {kk: float(np.min([r[kk] for r in S])) for kk in med}
        hi = {kk: float(np.max([r[kk] for r in S])) for kk in med}
        print("   %-4s median %d procs: c_R %.2f [%.2f, %.2f] ns  "
              "c_B %.2f [%.2f, %.2f] ns  kappa %.4f [%.4f, %.4f]"
              % (tag, len(S), med["aR_ns"], lo["aR_ns"], hi["aR_ns"],
                 med["aB_ns"], lo["aB_ns"], hi["aB_ns"],
                 med["kappa"], lo["kappa"], hi["kappa"]))
        print("        p %.1f  dp(Q=3.0) %.1f -> n0 %.0f [%.0f, %.0f]   "
              "dp(Q=8.1) %.1f -> n0 %.0f [%.0f, %.0f]"
              % (med["p"], med["dp_q3.0"], med["n0_q3.0"], lo["n0_q3.0"],
                 hi["n0_q3.0"], med["dp_q8.1"], med["n0_q8.1"],
                 lo["n0_q8.1"], hi["n0_q8.1"]))
        print("        R2: tR %.3f  tB %.3f"
              % (med["R2_tR"], med["R2_tB"]))
        rows[tag + "_med"] = med
    print()

    # ---- 2. the order effect on the arms -----------------------------------
    print("2. arm-level order effect (same 600 corrections, both protocols)")
    mR_rec = float(np.median(np.median(np.vstack([f["tR"] for f in rec]), 0)))
    mB_rec = float(np.median(np.median(np.vstack([f["tB"] for f in rec]), 0)))
    mR_alt = float(np.median(np.median(np.vstack([f["tR"] for f in alt]), 0)))
    mB_alt = float(np.median(np.median(np.vstack([f["tB"] for f in alt]), 0)))
    print("   median tR: recorded %.2f us, alternating %.2f us  (%+.1f%%)"
          % (mR_rec, mR_alt, 100 * (mR_alt / mR_rec - 1)))
    print("   median tB: recorded %.2f us, alternating %.2f us  (%+.1f%%)"
          % (mB_rec, mB_alt, 100 * (mB_alt / mB_rec - 1)))
    print("   band is cheaper in %d/600 recorded (median run), %d/600 "
          "alternating (median run)"
          % (int(np.sum(np.median(np.vstack([f["tB"] < f["tR"] for f in rec]),
                                  0) > 0.5)),
             int(np.sum(np.median(np.vstack([f["tB"] < f["tR"] for f in alt]),
                                  0) > 0.5))))
    print()

    # ---- 3. per-correction vote -------------------------------------------
    v_rec, TRr, TBr = vote(rec)
    v_alt, TRa, TBa = vote(alt)
    nr, na = len(rec), len(alt)
    print("3. per-correction vote (%d recorded runs, %d alternating runs)" % (nr, na))
    print("   recorded   histogram 0..%d: %s" % (nr, hist(v_rec, nr)))
    print("   alternating histogram 0..%d: %s" % (na, hist(v_alt, na)))
    print("   recorded    band in all %d runs: %d    band in none: %d"
          % (nr, int(np.sum(v_rec == nr)), int(np.sum(v_rec == 0))))
    print("   alternating band in all %d runs: %d    band in none: %d"
          % (na, int(np.sum(v_alt == na)), int(np.sum(v_alt == 0))))
    comb = v_rec + v_alt
    print("   all %d runs  band in every run: %d    band in no run: %d"
          % (nr + na, int(np.sum(comb == nr + na)), int(np.sum(comb == 0))))
    flip = int(np.sum((v_rec == nr) & (v_alt == 0)))
    flip2 = int(np.sum((v_rec == 0) & (v_alt == na)))
    print("   protocol decides it: %d corrections win 7/7 recorded but 0/%d "
          "alternating; %d the other way" % (flip, na, flip2))
    print()

    # groups, each family on its own runs (the two have different arm counts)
    print("   %-26s %5s %7s %7s %7s %10s %10s"
          % ("class", "count", "med n", "med k", "k/n", "speedup rec", "speedup alt"))
    for label, mask in (("band all alt", v_alt == na),
                        ("split alt (0<v<na)", (v_alt > 0) & (v_alt < na)),
                        ("band never alt", v_alt == 0),
                        ("band all 7 rec", v_rec == nr),
                        ("band never rec", v_rec == 0)):
        if mask.sum() == 0:
            continue
        sp_r = np.median(np.median(TBr[:, mask] / TRr[:, mask], 0))
        sp_a = np.median(np.median(TBa[:, mask] / TRa[:, mask], 0))
        print("   %-26s %5d %7.0f %7.0f %7.3f %10.2f %10.2f"
              % (label, int(mask.sum()), np.median(n[mask]), np.median(k[mask]),
                 np.median(k[mask] / np.maximum(n[mask], 1)), sp_r, sp_a))
    print()

    # ---- 4. criterion agreement, per run's own tuple ----------------------
    print("4. criterion vs measurement, each run judged by its own fit")
    print("   %-6s %-6s %6s %6s %6s %6s %6s %6s"
          % ("prot", "file", "takes", "right", "wrong", "miss", "agree%", "R2 tR"))
    agree = {}
    for tag, fam, vv in (("rec", rec, v_rec), ("alt", alt, v_alt)):
        agree[tag] = []
        for f, ft in zip(fam, rows[tag]):
            takes = ft["k"] + ft["p"] < ft["kappa"] * ft["n"]
            meas = f["tB"] < f["tR"]
            right = int(np.sum(takes == meas))
            wrong = int(np.sum(takes & ~meas))
            miss = int(np.sum(~takes & meas))
            agree[tag].append(dict(file=f["file"], takes=int(takes.sum()),
                                   right=right, wrong=wrong, miss=miss,
                                   agree=right / len(takes),
                                   kappa=ft["kappa"], n0=ft["n0_q3.0"]))
            print("   %-6s %-6s %6d %6d %6d %6d %6.1f %6.3f"
                  % (tag, f["file"].replace("veto_duel_rot_", "rot")
                     .replace("duel_big_run", "r").replace(".json", ""),
                     int(takes.sum()), right, wrong, miss,
                     100 * right / len(takes), ft["R2_tR"]))
    print()
    for tag in ("rec", "alt"):
        t = [a["takes"] for a in agree[tag]]
        ag = [a["agree"] for a in agree[tag]]
        print("   %-4s takes %d--%d (median %.0f)  agreement %.1f%%--%.1f%% "
              "(median %.1f%%)" % (tag, min(t), max(t), float(np.median(t)),
                                   100 * min(ag), 100 * max(ag),
                                   100 * float(np.median(ag))))
    print()

    # ---- 5. the corpus point, run by run ----------------------------------
    print("5. the corpus point (n=394, k=182): does the band win?")
    print("   %-6s %-6s %8s %8s %9s %9s %s"
          % ("prot", "file", "kappa", "n0", "k+p", "kappa*n", "verdict"))
    verdicts = {}
    for tag, fam in (("rec", rec), ("alt", alt)):
        verdicts[tag] = []
        for f, ft in zip(fam, rows[tag]):
            lhs = K_CORPUS + ft["p"]
            rhs = ft["kappa"] * N_CORPUS
            verdicts[tag].append(bool(lhs < rhs))
            print("   %-6s %-6s %8.4f %8.0f %9.0f %9.0f %s"
                  % (tag, f["file"].replace("veto_duel_rot_", "rot")
                     .replace("duel_big_run", "r").replace(".json", ""),
                     ft["kappa"], ft["n0_q3.0"], lhs, rhs,
                     "band" if lhs < rhs else "recompute"))
    print()
    for tag in ("rec", "alt"):
        print("   %-4s corpus point: band admitted in %d/%d runs"
              % (tag, sum(verdicts[tag]), len(verdicts[tag])))
    print()

    # ---- 6. the surcharge, measured ---------------------------------------
    print("6. routing surcharge over the whole stream (paired, per correction)")
    for tag, TR, TB in (("rec", TRr, TBr), ("alt", TRa, TBa)):
        d = (TB - TR)
        m = d.mean(0)
        print("   %-4s mean(tB-tR) %+.2f us = %+.1f%% of mean tR %.2f us   "
              "median %+.2f us" % (tag, float(m.mean()),
                                   100 * float(m.mean() / TR.mean()),
                                   float(TR.mean()), float(np.median(m))))
    # and restricted to the corrections the alternating criterion would take
    take_alt = np.zeros(N, dtype=bool)
    for f, ft in zip(alt, rows["alt"]):
        take_alt |= (ft["k"] + ft["p"] < ft["kappa"] * ft["n"])
    if take_alt.sum():
        d = (TBa - TRa)[:, take_alt].mean(0)
        print("   alt  on the %d corrections the alternating criterion admits: "
              "mean(tB-tR) %+.2f us (%+.1f%%)"
              % (int(take_alt.sum()), float(d.mean()),
                 100 * float(d.mean() / TRa[:, take_alt].mean())))
        d2 = (TBa - TRa)[:, ~take_alt].mean(0)
        print("   alt  on the other %d: %+.2f us (%+.1f%%)"
              % (int((~take_alt).sum()), float(d2.mean()),
                 100 * float(d2.mean() / TRa[:, ~take_alt].mean())))
    print()

    # ---- 7. what the paper prints, restated --------------------------------
    print("7. restatement of the printed constants")
    print("   %-34s %10s %10s" % ("quantity", "recorded", "alternating"))
    m_r, m_a = rows["rec_med"], rows["alt_med"]
    for key, lab in (("kappa", "kappa (c_R/c_B)"),
                     ("p", "p, candidates (ex post)"),
                     ("dp_q3.0", "dp, candidates (Q=3.0)"),
                     ("n0_q3.0", "n0, rows (Q=3.0)"),
                     ("dp_q8.1", "dp, candidates (Q=8.1)"),
                     ("n0_q8.1", "n0, rows (Q=8.1)")):
        print("   %-34s %10.2f %10.2f" % (lab, m_r[key], m_a[key]))

    res = dict(stream_identical=bool(same), n_corr=N,
               n_median=float(np.median(n)), k_median=float(np.median(k)),
               recorded_runs=len(rec), alternating_runs=len(alt),
               alternating_rot=len(rot), alternating_calib=len(cal),
               fits={t: [{kk: vv for kk, vv in f.items() if kk not in ("n", "k")}
                         for f in rows[t]] for t in ("rec", "alt")},
               fits_median={t: {kk: float(vv) for kk, vv in rows[t + "_med"].items()}
                            for t in ("rec", "alt")},
               agree=agree, verdicts=verdicts,
               votes=dict(recorded=hist(v_rec, nr), alternating=hist(v_alt, na),
                          both=hist(comb, nr + na)),
               med_tR=dict(recorded=mR_rec, alternating=mR_alt),
               med_tB=dict(recorded=mB_rec, alternating=mB_alt),
               take_alt=int(take_alt.sum()))
    p = os.path.join(OUT, "alt600.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
