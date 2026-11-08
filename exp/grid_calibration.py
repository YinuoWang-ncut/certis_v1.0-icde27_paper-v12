#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1: the calibration surface, 8 scope sizes x 10 densities = 80 cells.

Why this file exists
--------------------
`bigscope.py` scores the criterion on 24 cells (4 published sizes x 6 densities)
and calls it right on 18. A reviewer is entitled to read that as a list of points:
"would it hold at 20,000 rows, or at density 0.10?" The answer has to be a
surface, not a list, or the criterion is an observation rather than an instrument.

So this is the same five-stage pipeline as `masked_pipeline.py` -- probe, decide,
execute, mask, dedup, overlay, selector -- run on a grid that spans 3.9 orders of
magnitude in scope (197 to 1,600,000 rows) and 1.5 in band density (0.02 to 0.60),
and scored two ways:

  sign          does the criterion pick the path the stopwatch picks
  magnitude     how far is the cell from break-even when it does not

The second matters more than the first. A selector that is wrong on cells within
10% of break-even is a different object from one that is wrong on cells where the
other path is twice as cheap, and a single agreement count cannot tell them apart.

Layout, and why the cells are not built one at a time
-----------------------------------------------------
Path B gathers through a scattered id space while path R scans a code-clustered
copy, and that asymmetry is the whole reason the two paths differ (see
`bigscope.py`). A store holding a single code would put every line of that code in
consecutive groups, which turns the gather sequential and halves the per-candidate
cost -- the exact accident this experiment must not commit. So each store holds
several cells' codes, drawn at random over the whole store, and only stores are
split -- by a row cap -- never codes.

Memory is the reason for the cap: the store materialises two clustered copies and
three sorted keys, about 74 bytes a row, so a 4M-row store is ~300 MB and a
single 16M-row store would be over a gigabyte.

  python grid_calibration.py --reps 9
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
sys.path.insert(0, os.path.join(HERE, '..'))

import certis as C                                             # noqa: E402

# Scope sizes: the four published per-code sizes of Sec. V-D(c) plus four
# bracketing points, 197 to 1.6M (3.9 orders of magnitude).
SCOPES = [('n197', 197), ('n1k', 1_000), ('n7k', 6_969), ('n20k', 20_000),
          ('n45k', 44_606), ('n150k', 150_000), ('n413k', 412_875),
          ('n1m6', 1_600_000)]
# Band densities: the ceiling's median and quartiles, the replay's 0.303, and
# enough either side of the crossover to locate it.
DENS = [0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.28, 0.36, 0.46, 0.60]

CAP_OLD, CAP_NEW = 4, 2        # an MUE move 4 -> 2, tightened
AUX_CAP = 3                    # a stricter standing cap on the same code
DAYS_PER_GROUP = 58
OVERRIDE_P = 0.18
SEED = 31337
S_STAR = 0                     # theta_new < theta_old, so a flip needs sigma = 0

KAPPA = 0.2142                 # order-controlled tuple, Table VI
DP = 198.8
ROW_CAP = 4_000_000            # per store, see the note on memory above


def build(plans):
    """One store over several (tag, n, density) cells, one code each."""
    rng = np.random.default_rng(SEED)
    n_c = [int(round(n)) for _, n, _ in plans]
    d_c = [d for _, _, d in plans]
    total = sum(n_c)

    code_of = np.repeat(np.arange(len(plans), dtype=np.int32), n_c)
    code_of = code_of[rng.permutation(total)]
    unit = np.ones(total, dtype=np.int16)

    for code in range(len(plans)):
        pos = np.flatnonzero(code_of == code)
        k = int(round(d_c[code] * len(pos)))
        k_lo, k_hi = k // 2, k - k // 2
        pick = rng.choice(pos, size=min(k, len(pos)), replace=False)
        unit[pick[:k_lo]] = CAP_NEW + 1          # unit 3: sigma 0, flips
        unit[pick[k_lo:k_lo + k_hi]] = CAP_OLD   # unit 4: sigma 1, masked
        rest = np.setdiff1d(pos, pick)
        if len(rest):
            below = rest[rng.random(len(rest)) < 0.20]
            unit[below] = rng.integers(1, CAP_NEW + 1, len(below)).astype(np.int16)
            over = rest[rng.random(len(rest)) < 0.12]
            unit[over] = rng.integers(CAP_OLD + 1, 21, len(over)).astype(np.int16)

    n_groups = total // DAYS_PER_GROUP + 1
    gid = (np.arange(total, dtype=np.int32) // DAYS_PER_GROUP).astype(np.int32)
    c_line = np.argsort(code_of, kind='stable').astype(np.int32)
    cptr = np.zeros(len(plans) + 1, dtype=np.int64)
    np.cumsum(n_c, out=cptr[1:])
    inst = dict(unit=unit, amt=rng.uniform(10, 500, total).astype(np.float32),
                gid=gid, code=code_of, c_ptr=cptr, c_line=c_line,
                n_l=int(total), n_g=int(n_groups))
    override = rng.random(n_groups) < OVERRIDE_P
    store = C.Store(inst, override, build_index=True)
    ov = override[store.gid]
    sigma = ((store.unit > CAP_OLD) & ~ov).astype(np.int8)
    sigma += (store.unit > AUX_CAP).astype(np.int8)
    mask_state = np.where(ov, np.int8(-1), sigma).astype(np.int8)
    return store, mask_state


# ------------------------------------------------------------------ pipeline
def stage_scan(store, code):
    lo, hi = store.cptr[code], store.cptr[code + 1]
    v = store.unit_c[lo:hi]
    m = (v > CAP_OLD) != (v > CAP_NEW)
    if not m.any():
        return np.empty(0, np.int32)
    return store.cline[lo:hi][m].astype(np.int32)


def stage_mask(sigma, lids):
    if len(lids) == 0:
        return np.empty(0, np.int32)
    keep = sigma[lids] == S_STAR
    return lids[keep].astype(np.int32)


def run_R(store, sigma, code, overlay):
    flips = stage_mask(sigma, stage_scan(store, code))
    if len(flips):
        overlay[flips] = True
    return len(flips)


def run_B(store, sigma, code, overlay, cand):
    if len(cand):
        v = store.unit[cand]
        m = (v > CAP_OLD) != (v > CAP_NEW)
        flips = stage_mask(sigma, cand[m].astype(np.int32))
    else:
        flips = np.empty(0, np.int32)
    if len(flips):
        overlay[flips] = True
    return len(flips)


def block(fns, reps):
    """Round-robin: one GC pause must not land entirely on one arm."""
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


def plan_stores():
    """Group cells into stores under the row cap, never splitting a code."""
    stores, cur, rows = [], [], 0
    for tag, n in SCOPES:
        for d in DENS:
            if cur and rows + n > ROW_CAP:
                stores.append(cur)
                cur, rows = [], 0
            cur.append((tag, n, d))
            rows += n
    if cur:
        stores.append(cur)
    return stores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=9)
    ap.add_argument('--out', default=os.path.join(OUT, 'grid_calibration.json'))
    a = ap.parse_args()

    stores = plan_stores()
    print('%d cells in %d stores (cap %.1fM rows)'
          % (sum(len(s) for s in stores), len(stores), ROW_CAP / 1e6), flush=True)

    rows = []
    t_all = time.time()
    for si, plans in enumerate(stores):
        t0 = time.time()
        store, sigma = build(plans)
        ovh = np.zeros(store.n_l, dtype=bool)
        print('\nstore %d/%d: %d cells, %d rows, index %.1fs, %.0f MB'
              % (si + 1, len(stores), len(plans), store.n_l,
                 store.index_seconds, store.index_bytes / 1e6), flush=True)

        for code, (tag, n, d) in enumerate(plans):
            n_scope = int(store.scope_len(code))
            cand = store.band_u(code, CAP_NEW, CAP_OLD)
            k = int(len(cand))
            pred = bool(k + DP < KAPPA * n_scope)

            t = block({
                'R': lambda c=code: run_R(store, sigma, c, ovh),
                'B': lambda c=code, cd=cand: run_B(store, sigma, c, ovh, cd),
                'Q': lambda c=code: store.band_u(c, CAP_NEW, CAP_OLD),
            }, a.reps)

            mR = float(np.median(t['R']))
            mB = float(np.median(t['B']))
            mQ = float(np.median(t['Q']))
            # Eq. (11) prices a FIXED band path -- one that probes for k -- so the
            # object the criterion is scored against must carry the probe too.
            band_full = mQ + mB
            # The routed policy is a different comparison: it pays Q on either
            # branch, so Q cancels and the step in this column is the selector
            # switching, not the measurement moving.
            routed = mQ + (mB if pred else mR)
            # per-repetition sign, to separate "the criterion is right" from
            # "the measurement is stable"
            sign_admits = t['B'] < t['R']
            r = dict(tag=tag, n=n_scope, d=d, k=k, k_over_n=k / n_scope,
                     tR=mR, tB=mB, tQ=mQ, band_full=band_full, routed=routed,
                     ratio_routed=routed / mR, ratio_B=mB / mR,
                     ratio_full=band_full / mR,
                     # effective slope: the ratio of per-row to per-candidate cost
                     # this cell actually exhibits.  A constant kappa is the
                     # claim that this column is flat in d; it is not.
                     kappa_eff=((mR / n_scope) / (mB / k)) if k else float('nan'),
                     pred=pred, meas=bool(band_full < mR),
                     agree=bool(pred == (band_full < mR)),
                     signs_admit=int(sign_admits.sum()), signs_total=len(sign_admits),
                     tR_lo=float(np.percentile(t['R'], 10)),
                     tR_hi=float(np.percentile(t['R'], 90)))
            rows.append(r)
            print('  %-6s n %8d d %.2f k %8d | R %9.1f B %9.1f Q %6.2f '
                  '| full %6.3fx  kappa_eff %.3f | %s %s'
                  % (tag, n_scope, d, k, mR, mB, mQ, r['ratio_full'],
                     r['kappa_eff'], 'pred-B' if pred else 'pred-R',
                     'ok' if r['agree'] else 'MISS'), flush=True)
        print('  (%.0fs)' % (time.time() - t0), flush=True)

    # ---------------------------------------------------------------- scoring
    def bucket(r):
        return 'tie' if abs(r['ratio_full'] - 1) <= 0.20 else 'beyond'

    for r in rows:
        r['region'] = bucket(r)

    agree = sum(1 for r in rows if r['agree'])
    conf = {}
    for r in rows:
        key = ('B' if r['pred'] else 'R') + '|' + ('B' if r['meas'] else 'R')
        conf[key] = conf.get(key, 0) + 1

    def summarise(sel, label):
        n = len(sel)
        ok = sum(1 for r in sel if r['agree'])
        false_adm = sum(1 for r in sel if r['pred'] and not r['meas']
                        and r['region'] == 'beyond')
        false_ref = sum(1 for r in sel if not r['pred'] and r['meas']
                        and r['region'] == 'beyond')
        print('  %-22s %2d cells  agree %2d/%2d  false adm %d  false ref %d'
              % (label, n, ok, n, false_adm, false_ref))
        return dict(n=n, agree=ok, false_admit=false_adm, false_refuse=false_ref)

    print('\n=== sign correctness ===')
    all_s = summarise(rows, 'all')
    print('  confusion:', conf)
    print()
    by_reg = {g: summarise([r for r in rows if r['region'] == g], g)
              for g in ('tie', 'beyond')}
    print()
    by_scope = {}
    for tag, n in SCOPES:
        by_scope[tag] = summarise([r for r in rows if r['tag'] == tag], tag)

    errs = [r for r in rows if not r['agree']]
    print('\n=== the %d disagreements ===' % len(errs))
    for r in sorted(errs, key=lambda x: abs(x['ratio_full'] - 1), reverse=True):
        print('  n %8d d %.2f  pred %s meas %s  full/R %.3f  kappa_eff %.3f  (%s)'
              % (r['n'], r['d'], 'B' if r['pred'] else 'R',
                 'B' if r['meas'] else 'R', r['ratio_full'], r['kappa_eff'],
                 r['region']))

    # stability: does the sign of B vs R hold in every repetition
    unstable = [r for r in rows if 0 < r['signs_admit'] < r['signs_total']]
    print('\n=== stability ===')
    print('  cells whose B-vs-R sign flips across repetitions: %d / %d'
          % (len(unstable), len(rows)))
    for r in unstable:
        print('    n %8d d %.2f  %d/%d repetitions admit' %
              (r['n'], r['d'], r['signs_admit'], r['signs_total']))

    # kappa_eff by density, which is what A2 needs
    print('\n=== effective slope by density (kappa_eff = (tR/n)/(tB/k)) ===')
    kap_by_d = []
    for d in DENS:
        vals = [r['kappa_eff'] for r in rows if r['d'] == d and r['n'] >= 6969]
        if vals:
            print('  d %.2f  median %.3f  min %.3f  max %.3f  (n >= 6,969)'
                  % (d, float(np.median(vals)), float(np.min(vals)),
                     float(np.max(vals))))
            kap_by_d.append(dict(d=d, median=float(np.median(vals)),
                                 lo=float(np.min(vals)), hi=float(np.max(vals))))

    # the crossover, per scope: where does the fixed band path cross the scan
    print('\n=== crossover per scope (full band path / scan) ===')
    cross = {}
    for tag, n in SCOPES:
        cells = sorted((r for r in rows if r['tag'] == tag), key=lambda r: r['d'])
        lo = hi = None
        for x, y in zip(cells, cells[1:]):
            if (x['ratio_full'] - 1) * (y['ratio_full'] - 1) < 0:
                lo, hi = x['d'], y['d']
                break
        cross[tag] = [lo, hi]
        seq = ' '.join('%.2f' % r['ratio_full'] for r in cells)
        print('  %-6s n %8d  crossing %s  | %s'
              % (tag, n, ('between d=%.2f and %.2f' % (lo, hi)) if lo else 'none',
                 seq))

    res = dict(reps=a.reps, scopes=[list(s) for s in SCOPES], densities=DENS,
               kappa=KAPPA, dp=DP, cap=(CAP_OLD, CAP_NEW), aux_cap=AUX_CAP,
               cells=len(rows), all=all_s, by_region=by_reg,
               by_scope=by_scope, confusion=conf,
               unstable=len(unstable), crossover=cross, kappa_by_density=kap_by_d,
               disagreements=[dict(n=r['n'], d=r['d'], pred=r['pred'],
                                   meas=r['meas'], ratio_full=r['ratio_full'],
                                   kappa_eff=r['kappa_eff'],
                                   region=r['region']) for r in errs],
               rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=float)
    print('\nwall %.0fs, wrote %s' % (time.time() - t_all, a.out))


if __name__ == '__main__':
    main()
