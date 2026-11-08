#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The pipeline cell at the published scope sizes -- implementation-resolved.

Why this file exists
--------------------
The paper has two results that a reader is entitled to compare and cannot.
`fullpath_band_vs_scan.py` runs the *five-stage pipeline* -- probe, decide,
execute, mask, overlay -- but on the replay instance, whose median scope is 401
rows.  `bigscope.py` runs at the *published* per-code sizes, up to 412,875
beneficiary-days, but on bare paths: it passes an empty rule set, so the mask
folds into the override filter alone and no support multiplicity is ever
consulted.  So "retrieval loses at 401 rows" and "retrieval wins at 44,606
rows" are statements about two different objects, and neither is the claim the
paper makes, which is that the *pipeline* changes sign with scale.

This file runs the pipeline of Sec. IV-F at the published scope sizes, on stores
whose layout is the one `bigscope.py` establishes: the code of each line is drawn
at random over the whole store, so path B gathers through a scattered id space
while path R scans a code-clustered copy -- the asymmetry that is the entire
reason the two paths differ.

Why the primitives are enumerated instead of chosen
--------------------------------------------------
Each stage can be built two ways, and on this machine the two ways differ by a
factor of two *in opposite directions per stage*:

  gather of k values    `src[idx]`                212 us   (k = 75,556)
                        `np.take(src, idx, out=)` 543 us
  compaction over n     `src[m]`                  990 us   (n = 412,875, k = 75,805)
                        `np.compress(m, src, out=)` 487 us

Boolean indexing materialises an `intp` array of length n to hold the selected
positions -- 3.3 MB at the largest scope -- and `np.compress` does not; but
numpy's `take` with `out=` runs the buffered generic loop where fancy indexing
has a specialised gather.  So neither "preallocate everything" nor "allocate
everything" is the fast implementation, and a paper that reports one number
without saying which build it used is reporting an artefact of one line of
Python.

Hence: the pipeline is built from (compact, gather, mask-gather), each in
{fast, alloc}, and every cell is timed under all eight builds.  What is reported
is (a) the per-stage best, which is realisable because the stages are
independent, and (b) the span across the eight, because that span is the
implementation sensitivity a reader should see next to the ratio.

What the pipeline adds over the bare kernel
-------------------------------------------
  mask       Thm. 3: a candidate flips only if its support multiplicity before
             the correction equals 1[theta_new > theta_old].  Two rules sit on
             every code -- the moved cap and a standing stricter one, as the MUE
             table carries several MAI variants per HCPCS code -- so sigma is not
             identically 1 and the mask actually fires.  sigma is *maintained*,
             as the paper assumes: one byte per line, refreshed by the pass that
             emits the previous flip set, read here as one gather.
  dedup      flip identifiers deduplicated before emission.
  overlay    the boolean overlay absorbing this correction's flips.
  selector   Eq. (12) with the calibrated tuple, probe paid on either branch.

The interval.  The two arms are measured in the same round-robin block, so what
has a distribution is the *ratio* of paired repetitions, not either arm.  The
reported interval is a paired bootstrap of that ratio.

  python masked_pipeline.py --reps 15
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))          # certis.py at the root

import certis as C                                     # noqa: E402

# The published per-code beneficiary-day counts (realband.py): 5th percentile,
# quartile, median, upper quartile of six years of CMS data.
NS = [("p5", 197), ("q1", 6969), ("median", 44606), ("q3", 412875)]
# The band densities of Sec. V-D(c): the ceiling's median and its quartiles.
DS = [0.042, 0.183, 0.401]

CAP_OLD, CAP_NEW = 4, 2        # the dominant observed MUE shape, tightened
AUX_CAP = 3                    # a stricter standing cap on the same code
DAYS_PER_GROUP = 58            # published median days per (NPI, code) pair
OVERRIDE_P = 0.18              # measured share of groups carrying an override
SEED = 31337

# The selector's tuple: order-controlled over 23 runs (Table V, last row).
KAPPA = 0.2142
DP = 198.8

TUPLE_NOTE = "eq.(12) with kappa 0.2142, dp 198.8 as printed in Table V"
S_STAR = 1 if CAP_NEW > CAP_OLD else 0

BUILDS = [(c, g, m) for c in ("fast", "alloc")
          for g in ("fast", "alloc") for m in ("fast", "alloc")]

# The build the paper reports.  It is a *rule*, not a per-cell choice: choosing
# the fastest build separately for each cell from a nine-repetition sweep makes
# the reported ratio depend on which way the sweep's noise fell, and a ninth
# combination (compress for path R, boolean index for path B) that was never
# timed as a whole then gets reported as if it had been.  The rule comes from
# `prim_cost.py`, which measures the two compactions against each other:
# boolean indexing wins below about 1.6e4 elements because it avoids a function
# call, and loses above because it materialises an intp array of the input's
# length.  Gathers are fancy indexing at every size measured (212 vs 543 us at
# k = 75,556; 25 vs 36 at k = 8,163).
COMPACT_CROSSOVER = 16384


def build_rule(n_elem):
    return "alloc" if n_elem < COMPACT_CROSSOVER else "fast"


def build(plans):
    """One store, one code per (n, density) cell."""
    rng = np.random.default_rng(SEED)
    n_c = [int(round(n)) for _, n, _ in plans]
    d_c = [d for _, _, d in plans]
    total = sum(n_c)

    code_of = np.repeat(np.arange(len(plans), dtype=np.int32), n_c)
    code_of = code_of[rng.permutation(total)]
    unit = np.ones(total, dtype=np.int16)

    for code in range(len(plans)):
        pos = np.flatnonzero(code_of == code)
        d = d_c[code]
        # candidates are unit in (CAP_NEW, CAP_OLD]; half at each atom, so the
        # mask (sigma = s*) keeps the lower atom and drops the upper one
        k = int(round(d * len(pos)))
        k_lo, k_hi = k // 2, k - k // 2
        pick = rng.choice(pos, size=min(k, len(pos)), replace=False)
        unit[pick[:k_lo]] = CAP_NEW + 1                    # = 3: flips
        unit[pick[k_lo:k_lo + k_hi]] = CAP_OLD             # = 4: masked by AUX
        rest = np.setdiff1d(pos, pick, assume_unique=False)
        if len(rest):
            below = rest[rng.random(len(rest)) < 0.20]
            unit[below] = rng.integers(1, CAP_NEW + 1, len(below)).astype(np.int16)
            over_cap = rest[rng.random(len(rest)) < 0.12]
            unit[over_cap] = rng.integers(CAP_OLD + 1, 21,
                                          len(over_cap)).astype(np.int16)

    n_groups = total // DAYS_PER_GROUP + 1
    gid = (np.arange(total, dtype=np.int32) // DAYS_PER_GROUP).astype(np.int32)
    c_line = np.argsort(code_of, kind="stable").astype(np.int32)
    cptr = np.zeros(len(plans) + 1, dtype=np.int64)
    np.cumsum(n_c, out=cptr[1:])
    inst = dict(unit=unit, amt=rng.uniform(10, 500, total).astype(np.float32),
                gid=gid, code=code_of, c_ptr=cptr, c_line=c_line,
                n_l=int(total), n_g=int(n_groups))
    override = rng.random(n_groups) < OVERRIDE_P
    store = C.Store(inst, override, build_index=True)
    # maintained support multiplicity of the current verdict, one byte a line
    ov = override[store.gid]
    sigma = ((store.unit > CAP_OLD) & ~ov).astype(np.int8)
    sigma += (store.unit > AUX_CAP).astype(np.int8)
    # the override conjunct of Thm. 3 folded into the same byte as -1, so the
    # mask is one gather rather than a gather plus a level of indirection
    mask_state = np.where(ov, np.int8(-1), sigma).astype(np.int8)
    return store, mask_state, n_groups


# ------------------------------------------------------------------- buffers
class Buf:
    """All scratch a cell needs, allocated once.  Sized by the largest single
    scope and the largest band in the store, not by the store: nothing here
    scales with |D|, which is the point of the fixed-|D| design in grid_fixedD."""

    def __init__(self, n_max, k_max):
        self.b1 = np.zeros(n_max, dtype=bool)
        self.b2 = np.zeros(n_max, dtype=bool)
        self.i8 = np.zeros(n_max, dtype=np.int8)
        self.ov = np.zeros(n_max, dtype=np.uint8)
        self.i16 = np.zeros(n_max, dtype=np.int16)
        self.o1 = np.zeros(k_max, dtype=np.int32)
        self.o2 = np.zeros(k_max, dtype=np.int32)
        self.n_max = n_max
        self.k_max = k_max


# -------------------------------------------------------------------- stages
def _compact(m, src, cnt, out, how):
    """Compact the positions of m out of src into out[:cnt]."""
    if cnt == 0:
        return 0
    if how == "fast":
        np.compress(m, src, out=out[:cnt])      # no n-sized intp temporary
    else:
        out[:cnt] = src[m]                      # boolean index: allocates one
    return cnt


def _gather(src, idx, out, how):
    if how == "fast":
        return src[idx]                         # specialised gather loop
    np.take(src, idx, out=out[:len(idx)])       # buffered loop, no allocation
    return out[:len(idx)]


def scan_into(store, code, buf, how="fast"):
    """Path R: one contiguous pass, both states judged, compacted into buf.o1.
    Two passes over the scope (one per threshold) would inflate R by 2x -- the
    accident that makes an index look good."""
    lo, hi = store.cptr[code], store.cptr[code + 1]
    v = store.unit_c[lo:hi]
    m = buf.b1[:hi - lo]
    np.greater(v, CAP_OLD, out=m)
    np.greater(v, CAP_NEW, out=buf.b2[:hi - lo])
    np.not_equal(m, buf.b2[:hi - lo], out=m)
    cnt = int(np.count_nonzero(m))
    return _compact(m, store.cline[lo:hi], cnt, buf.o1, how)


def gather_into(store, cand, buf, gap="fast", how="fast"):
    """Path B's scattered fetch, same two-state test, into buf.o1."""
    k = len(cand)
    if not k:
        return 0
    vals = _gather(store.unit, cand, buf.i16, gap)
    m = buf.b1[:k]
    np.greater(vals, CAP_OLD, out=m)
    np.greater(vals, CAP_NEW, out=buf.b2[:k])
    np.not_equal(m, buf.b2[:k], out=m)
    cnt = int(np.count_nonzero(m))
    return _compact(m, cand, cnt, buf.o1, how)


def mask_into(sigma, src, cnt, buf, out, gap="fast", how="fast"):
    """Thm. 3: flip iff sigma == s*.  One gather against maintained state."""
    if not cnt:
        return 0
    ms = _gather(sigma, src[:cnt], buf.i8[:cnt], gap)
    mk = buf.b1[:cnt]
    np.equal(ms, S_STAR, out=mk)
    keep = int(np.count_nonzero(mk))
    return _compact(mk, src[:cnt], keep, out, how)


def emit_into(overlay8, src, cnt, buf):
    """Dedup then absorb.  `overlay8` is the bool overlay viewed as uint8, so the
    read is one gather and the write one scatter."""
    if not cnt:
        return 0
    g = _gather(overlay8, src[:cnt], buf.ov[:cnt], "fast")
    mk = buf.b1[:cnt]
    np.equal(g, 0, out=mk)
    fresh = int(np.count_nonzero(mk))
    overlay8[src[:cnt]] = 1
    return fresh


def run_R(store, sigma, code, overlay8, buf, how="fast"):
    cnt = scan_into(store, code, buf, how)
    cnt = mask_into(sigma, buf.o1, cnt, buf, buf.o2, "fast", "fast")
    return emit_into(overlay8, buf.o2, cnt, buf)


def run_B(store, sigma, code, overlay8, buf, cand=None, gap="fast", how="fast",
          mgap="fast"):
    if cand is None:
        cand = store.band_u(code, CAP_NEW, CAP_OLD)
    cnt = gather_into(store, cand, buf, gap, how)
    cnt = mask_into(sigma, buf.o1, cnt, buf, buf.o2, mgap, "fast")
    return emit_into(overlay8, buf.o2, cnt, buf)


def ground_truth(store, code):
    """Full recomputation of the disjunction on both sides -- outside every
    timing, and allowed to allocate."""
    lo, hi = store.cptr[code], store.cptr[code + 1]
    lid = store.cline[lo:hi]
    v = store.unit[lid]
    ov = store.override[store.gid[lid]]
    old = ((v > CAP_OLD) & ~ov) | (v > AUX_CAP)
    new = ((v > CAP_NEW) & ~ov) | (v > AUX_CAP)
    return np.sort(lid[old != new]).astype(np.int32)


def flips_of(store, sigma, code, buf, cand, gap="fast", how="fast", mgap="fast"):
    cnt = gather_into(store, cand, buf, gap, how)
    cnt = mask_into(sigma, buf.o1, cnt, buf, buf.o2, mgap, "fast")
    return np.sort(buf.o2[:cnt].copy())


# ------------------------------------------------------------- the environment
_REF = np.arange(1 << 22)


def environment(reps=9):
    """One number for "how fast is this box right now".

    A benchmark that reports a ratio without reporting the machine's state
    cannot distinguish a slow path from a busy box: this file's own history has
    a cell at 0.78 and the same cell at 0.96 because a downloader was writing to
    the same disk.  The reference is a fixed numpy loop with almost no memory
    traffic, timed before and after each cell's block."""
    v = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.count_nonzero(_REF >= 2048)
        v.append(time.perf_counter() - t0)
    return float(np.median(v))


def null_band(pool, n_before=9, n_after=9, reps=2000, seed=11):
    """The drift a *quiet* box produces, from the box's own distribution.

    This machine's reference loop has a coefficient of variation of 16% within a
    second while its per-second medians move by 3%, so a fixed threshold on the
    drift flags jitter rather than interference.  Resampling two groups of the
    size the cell actually uses from the startup pool gives the null distribution
    of the before/after ratio; the cell is flagged only if it falls outside that
    distribution's 1st and 99th percentiles."""
    pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    r = np.median(pool[rng.integers(0, len(pool), (reps, n_after))], axis=1) /         np.median(pool[rng.integers(0, len(pool), (reps, n_before))], axis=1)
    return float(np.percentile(r, 1)), float(np.percentile(r, 99))


# ---------------------------------------------------------------- statistics
def block(fns, reps):
    """Round-robin, so one GC pause cannot land entirely on one arm."""
    out = {k: [] for k in fns}
    for _ in range(2):
        for f in fns.values():
            f()
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            out[k].append((time.perf_counter() - t0) * 1e6)
    return {k: np.asarray(v) for k, v in out.items()}


def med(a):
    return float(np.median(a))


def paired_ci(num, den, reps=4000, seed=7):
    """Bootstrap CI for the *ratio*, resampling repetitions in pairs.

    The arms are measured in the same round-robin block, so repetition r of one
    shares whatever the machine was doing with repetition r of the other.
    Ratioing them removes that shared term; bootstrapping the arms separately
    would put it back.  Percentile interval, median statistic."""
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(reps, n))
    r = np.median(num[idx] / den[idx], axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def paired_point(num, den):
    """The point estimate the interval is built around: the median of the paired
    ratios.  Reporting ratio-of-medians next to a median-of-ratios interval puts
    a point outside its own interval at the smallest scopes, where the fixed
    cost's jitter dominates -- two arms that differ by 2% in the median can have
    every paired ratio above 1."""
    return float(np.median(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--sweep-reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(OUT, "masked_pipeline.json"))
    a = ap.parse_args()

    plans = [(tag, n, d) for tag, n in NS for d in DS]
    t0 = time.time()
    store, sigma, n_g = build(plans)
    n_max = max(int(store.scope_len(c)) for c in range(len(plans)))
    k_max = max(int(len(store.band_u(c, CAP_NEW, CAP_OLD)))
                for c in range(len(plans)))
    buf = Buf(n_max, k_max)
    print("store: %d codes, %d lines, %d groups, index %.1fs, %.1f MB; "
          "scratch %.1f MB (n_max %d, k_max %d)"
          % (len(plans), store.n_l, n_g, store.index_seconds,
             store.index_bytes / 1e6,
             sum(x.nbytes for x in (buf.b1, buf.b2, buf.i8, buf.ov, buf.o1,
                                    buf.o2, buf.i16)) / 1e6, n_max, k_max),
          flush=True)
    ov8 = np.zeros(store.n_l, dtype=bool).view(np.uint8)

    # the box's own distribution, sampled once, so the drift guard below is a
    # test against this machine rather than against a number chosen by hand
    ref_pool = []
    for _ in range(60):
        tr = time.perf_counter()
        np.count_nonzero(_REF >= 2048)
        ref_pool.append(time.perf_counter() - tr)
    band = null_band(ref_pool)
    print("environment: reference %.0f us (cv %.1f%%), quiet-box drift band "
          "[%.2f, %.2f]" % (np.median(ref_pool) * 1e6,
                            100 * np.std(ref_pool) / np.mean(ref_pool),
                            band[0], band[1]), flush=True)

    for code in range(len(plans)):          # warm every cell before timing any
        run_R(store, sigma, code, ov8, buf)
        run_B(store, sigma, code, ov8, buf)
        store.band_u(code, CAP_NEW, CAP_OLD)

    rows = []
    for code, (tag, n, d) in enumerate(plans):
        ov8[:] = 0
        gt = ground_truth(store, code)
        cand = store.band_u(code, CAP_NEW, CAP_OLD)
        k = int(len(cand))
        n_scope = int(store.scope_len(code))
        pred = bool(k + DP < KAPPA * n_scope)

        # ---- the eight builds in one block, so they share the machine's weather
        arms = {}
        for c, g, m in BUILDS:
            key = c[0] + g[0] + m[0]
            arms["R_" + key] = (
                lambda c=c, g=g, m=m: run_R(store, sigma, code, ov8, buf, c))
            arms["B_" + key] = (
                lambda c=c, g=g, m=m: run_B(store, sigma, code, ov8, buf, cand,
                                            g, c, m))
        ref0 = environment()
        t = block(arms, a.sweep_reps)
        ref_sweep = environment()

        builds = []
        for c, g, m in BUILDS:
            key = c[0] + g[0] + m[0]
            tr, tb = med(t["R_" + key]), med(t["B_" + key])
            builds.append(dict(compact=c, gather=g, mask=m, tR=tr, tB=tb,
                               ratio=tb / tr))
        # the reported build follows the rule; the sweep above only supplies the
        # span, which is the honest error bar on any single build here
        cR, cB = build_rule(n_scope), build_rule(k)
        bestR = dict(compact=cR)
        bestB = dict(compact=cB, gather="fast", mask="fast")
        ratios = sorted(x["ratio"] for x in builds)

        # ---- the reported cell: per-stage best, timed on its own with --reps
        t2 = block({
            "R": lambda: run_R(store, sigma, code, ov8, buf, bestR["compact"]),
            "B": lambda: run_B(store, sigma, code, ov8, buf, cand,
                               bestB["gather"], bestB["compact"], bestB["mask"]),
            "Q": lambda: store.band_u(code, CAP_NEW, CAP_OLD),
        }, a.reps)
        ref1 = environment()
        same = bool(np.array_equal(
            gt, flips_of(store, sigma, code, buf, cand, bestB["gather"],
                         bestB["compact"], bestB["mask"])))

        mR, mB, mQ = med(t2["R"]), med(t2["B"]), med(t2["Q"])
        routed = mQ + (mB if pred else mR)
        lo, hi = paired_ci(t2["B"], t2["R"])
        lof, hif = paired_ci(t2["B"] + mQ, t2["R"])
        r = dict(tag=tag, n=n_scope, k=k, dens=k / max(n_scope, 1),
                 tR=mR, tB=mB, tQ=mQ, tRouted=routed,
                 ratio=paired_point(t2["B"], t2["R"]), ratio_ci=(lo, hi),
                 ratio_of_medians=mB / mR,
                 ratio_full=paired_point(t2["B"] + mQ, t2["R"]),
                 ratio_full_ci=(lof, hif),
                 ratio_routed=paired_point(t2["B"] + t2["Q"], t2["R"]),
                 ratio_routed_of_medians=routed / mR,
                 ratio_span=(ratios[0], ratios[-1]),
                 build_R=dict(compact=bestR["compact"]),
                 build_B=dict(compact=bestB["compact"], gather=bestB["gather"],
                              mask=bestB["mask"]),
                 builds=builds,
                 kappa_eff=((mR / max(n_scope, 1)) / (mB / max(k, 1))),
                 pred_band=pred, meas_band=bool(paired_point(t2["B"], t2["R"]) < 1),
                 agree=bool(pred == (paired_point(t2["B"], t2["R"]) < 1)),
                 drift=ref1 / ref0, ref_us=ref0 * 1e6,
                 contaminated=bool(not (band[0] <= ref1 / ref0 <= band[1])),
                 flip_count=len(gt), same=same)
        rows.append(r)
        print("%-6s n %7d k %7d d %.3f | R %8.1f  B %8.1f  Q %5.2f | "
              "B/R %.3f [%.3f-%.3f]  builds %.3f-%.3f | drift %.2f | %s %s%s"
              % (tag, n_scope, k, r["dens"], mR, mB, mQ, r["ratio"], lo, hi,
                 ratios[0], ratios[-1], r["drift"],
                 "pred-band" if pred else "pred-scan",
                 "ok" if same else "MISMATCH",
                 "  CONTAMINATED" if r["contaminated"] else ""), flush=True)

    pick = lambda tag, dd: min((r for r in rows if r["tag"] == tag),
                               key=lambda r: abs(r["dens"] - dd))
    cell = pick("median", 0.183)
    print()
    print("THE CELL: n = %d (published median scope), d = 0.183, k = %d"
          % (cell["n"], cell["k"]))
    print("  R %8.1f   B %8.1f   Q %5.2f   routed %8.1f -> %.3f x the scan"
          "  CI [%.3f, %.3f]"
          % (cell["tR"], cell["tB"], cell["tQ"], cell["tRouted"],
             cell["ratio_routed"], cell["ratio_full_ci"][0],
             cell["ratio_full_ci"][1]))
    print("  B/R %.3f  95%% CI [%.3f, %.3f]   across the eight builds [%.3f, %.3f]"
          % (cell["ratio"], cell["ratio_ci"][0], cell["ratio_ci"][1],
             cell["ratio_span"][0], cell["ratio_span"][1]))
    print("  build (rule): compact=%s for path R (n=%d), compact=%s for path B "
          "(k=%d), gather=fancy" % (cell["build_R"]["compact"], cell["n"],
                                    cell["build_B"]["compact"], cell["k"]))
    print()
    print("the four cells at d = 0.183, paired bootstrap on the routed policy:")
    for r in rows:
        if abs(r["dens"] - 0.183) < 0.01:
            ci = r["ratio_full_ci"]
            resolved = (ci[0] - 1) * (ci[1] - 1) > 0
            print("  n %7d  routed %8.1f  -> %.3f  CI [%.3f, %.3f]  %s"
                  % (r["n"], r["tRouted"], r["ratio_routed"], ci[0], ci[1],
                     "sign resolved" if resolved else "UNRESOLVED"))
    print()
    clean = [r for r in rows if not r["contaminated"]]
    print("criterion vs stopwatch: %d of %d cells agree  (%d of %d on the cells "
          "the box was quiet for)"
          % (sum(1 for r in rows if r["agree"]), len(rows),
             sum(1 for r in clean if r["agree"]), len(clean)))
    print("environment: reference loop %.0f us, drift %.2f-%.2f over the %d cells"
          % (rows[0]["ref_us"], min(r["drift"] for r in rows),
             max(r["drift"] for r in rows), len(rows)))
    worst = max(rows, key=lambda r: r["ratio_span"][1] - r["ratio_span"][0])
    print("widest implementation span: n %d d %.3f  [%.3f, %.3f]  (%.2fx)"
          % (worst["n"], worst["dens"], worst["ratio_span"][0],
             worst["ratio_span"][1],
             worst["ratio_span"][1] / worst["ratio_span"][0]))

    res = dict(reps=a.reps, sweep_reps=a.sweep_reps, cap=(CAP_OLD, CAP_NEW),
               ref_us=float(np.median(ref_pool) * 1e6),
               ref_cv=float(np.std(ref_pool) / np.mean(ref_pool)),
               quiet_band=band,
               aux_cap=AUX_CAP, days_per_group=DAYS_PER_GROUP,
               override_p=OVERRIDE_P, tuple=TUPLE_NOTE, cells=len(rows),
               rows=rows, buffers=dict(n_max=n_max, k_max=k_max),
               agree=int(sum(1 for r in rows if r["agree"])),
               delta_complete=bool(all(r["same"] for r in rows)))
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwall %.0fs, wrote %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()
