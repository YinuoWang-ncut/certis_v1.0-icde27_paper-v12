#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1, done the way the deployment is shaped: fixed table, scope inside it.

Why this file exists
--------------------
The grid of `grid_calibration.py` and `grid80.json` varies the scope n while the
store grows with it, because each store holds several cells' codes. That makes the
store size a hidden variable, and it is not a small one: at one fixed cell
(n = 412,875, k = 75,556, d = 0.183) the band-to-scan ratio moves from 0.368 to
1.102 -- a factor of three -- as the store grows from one code to sixteen
(`out/store_size.json`). The same cell therefore reads "band wins by 2.7x" or
"band loses by 10%" depending on how many lines happen to share the array.

That is not a measurement artefact to be normalised away. It is the physics of the
comparison. Path B fetches candidates from the base table, so its per-candidate
cost is a property of |D| -- the size of the heap it fetches through -- while path
R reads a code-clustered copy, so its per-row cost is a property of n alone. A
ratio of the two is therefore a function of (n, k, |D|), and the paper's plane is a
slice at whatever |D| the harness happened to build.

A deployment has a |D|. So this experiment holds it fixed: one store of exactly
D rows, its first code carrying the scope n and every other row belonging to
filler codes at the same density, so the only thing that varies across cells is
(n, k). D is a parameter, and the last check sweeps it, which is what turns the
confound into a reported sensitivity instead of an unexplained scatter.

  python grid_fixedD.py --D 2000000 --reps 5
  python grid_fixedD.py --sweep            # the |D| sensitivity of three cells
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

SCOPES = [('n197', 197), ('n1k', 1_000), ('n7k', 6_969), ('n20k', 20_000),
          ('n45k', 44_606), ('n150k', 150_000), ('n413k', 412_875)]
DENS = [0.02, 0.04, 0.06, 0.10, 0.15, 0.183, 0.28, 0.36, 0.46, 0.60]

CAP_OLD, CAP_NEW = 4, 2
AUX_CAP = 3
DAYS_PER_GROUP = 58
OVERRIDE_P = 0.18
S_STAR = 0
KAPPA, DP = 0.2142, 198.8
SEED = 31337
N_FILLER = 15                     # filler codes, so no single one dominates


def build(D, n, d, rng):
    """A store of exactly D rows: code 0 carries n rows at density d; the rest is
    filler at the same density, so the fillers' band sizes are irrelevant to the
    query but their presence sets the heap the gather fetches through."""
    code_of = np.zeros(D, dtype=np.int32)
    perm = rng.permutation(D)
    code_of[:n] = 0
    rest = D - n
    if rest:
        code_of[n:] = 1 + rng.integers(0, N_FILLER, rest, dtype=np.int32)
    code_of = code_of[perm]

    unit = np.ones(D, dtype=np.int16)
    # every code, the measured one included, gets the same density treatment
    for c in range(1 + (1 if rest else 0)):
        idx = np.flatnonzero(code_of == c)
        if not len(idx):
            continue
        kk = int(round(d * len(idx)))
        pick = rng.choice(idx, size=min(kk, len(idx)), replace=False)
        half = len(pick) // 2
        unit[pick[:half]] = CAP_NEW + 1
        unit[pick[half:]] = CAP_OLD
        free = np.setdiff1d(idx, pick)
        if len(free):
            b = free[rng.random(len(free)) < 0.20]
            unit[b] = rng.integers(1, CAP_NEW + 1, len(b)).astype(np.int16)
            o = free[rng.random(len(free)) < 0.12]
            unit[o] = rng.integers(CAP_OLD + 1, 21, len(o)).astype(np.int16)

    codes = code_of.max() + 1
    c_line = np.argsort(code_of, kind='stable').astype(np.int32)
    counts = np.bincount(code_of, minlength=codes)
    cptr = np.zeros(codes + 1, dtype=np.int64)
    np.cumsum(counts, out=cptr[1:])
    n_groups = D // DAYS_PER_GROUP + 1
    gid = (np.arange(D, dtype=np.int32) // DAYS_PER_GROUP).astype(np.int32)
    inst = dict(unit=unit, amt=rng.uniform(10, 500, D).astype(np.float32),
                gid=gid, code=code_of, c_ptr=cptr, c_line=c_line,
                n_l=int(D), n_g=int(n_groups))
    override = rng.random(n_groups) < OVERRIDE_P
    store = C.Store(inst, override, build_index=True)
    ov = override[store.gid]
    sigma = ((store.unit > CAP_OLD) & ~ov).astype(np.int8)
    sigma += (store.unit > AUX_CAP).astype(np.int8)
    mask_state = np.where(ov, np.int8(-1), sigma).astype(np.int8)
    return store, mask_state


def scan_flips(store, sigma, code):
    lo, hi = store.cptr[code], store.cptr[code + 1]
    v = store.unit_c[lo:hi]
    m = (v > CAP_OLD) != (v > CAP_NEW)
    if not m.any():
        return
    sigma[store.cline[lo:hi][m]]


def gather_flips(store, sigma, cand):
    if not len(cand):
        return
    v = store.unit[cand]
    m = (v > CAP_OLD) != (v > CAP_NEW)
    sigma[cand[m].astype(np.int32)]


def block(fns, reps):
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


def measure(D, n, d, reps, rng):
    store, sigma = build(D, n, d, rng)
    code = 0
    cand = store.band_u(code, CAP_NEW, CAP_OLD)
    k = int(len(cand))
    t = block({
        'R': lambda: scan_flips(store, sigma, code),
        'B': lambda: gather_flips(store, sigma, cand),
        'Q': lambda: store.band_u(code, CAP_NEW, CAP_OLD),
    }, reps)
    mR, mB, mQ = (float(np.median(t[x])) for x in 'RBQ')
    full = mQ + mB
    return dict(n=n, D=D, d=d, k=k, k_over_n=k / n,
                tR=mR, tB=mB, tQ=mQ, ratio_full=full / mR, ratio_B=mB / mR,
                kappa_eff=((mR / n) / (mB / k)) if k else float('nan'),
                pred=bool(k + DP < KAPPA * n), meas=bool(full < mR),
                c_R_ns=mR / n * 1e3 if n else float('nan'),
                c_B_ns=mB / k * 1e3 if k else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=2_000_000)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()

    if a.sweep:
        cells = [('n7k', 6_969, 0.183), ('n45k', 44_606, 0.183),
                 ('n413k', 412_875, 0.183)]
        for tag, n, d in cells:
            for D in (n, 400_000, 1_000_000, 2_000_000, 8_000_000):
                if D < n:
                    continue
                r = measure(D, n, d, a.reps, rng)
                r['tag'] = tag
                rows.append(r)
                print('  %-6s n %7d  D %8d  ratio %6.3f  c_R %.2f ns  c_B %.1f ns'
                      % (tag, n, D, r['ratio_full'], r['c_R_ns'], r['c_B_ns']),
                      flush=True)
        res = dict(reps=a.reps, mode='sweep', rows=rows)
    else:
        print('fixed |D| = %d, %d scopes x %d densities, reps %d'
              % (a.D, len(SCOPES), len(DENS), a.reps), flush=True)
        for tag, n in SCOPES:
            for d in DENS:
                r = measure(a.D, n, d, a.reps, rng)
                r['tag'] = tag
                r['agree'] = bool(r['pred'] == r['meas'])
                rows.append(r)
                print('  %-6s n %7d d %.3f k %7d | R %8.1f B %8.1f Q %5.2f '
                      '| full %6.3fx  kappa_eff %.3f | %s %s'
                      % (tag, n, d, r['k'], r['tR'], r['tB'], r['tQ'],
                         r['ratio_full'], r['kappa_eff'],
                         'pred-B' if r['pred'] else 'pred-R',
                         'ok' if r['agree'] else 'MISS'), flush=True)
        agree = sum(1 for r in rows if r['agree'])
        tie = [r for r in rows if abs(r['ratio_full'] - 1) <= 0.20]
        beyond = [r for r in rows if abs(r['ratio_full'] - 1) > 0.20]
        conf = {}
        for r in rows:
            key = ('B' if r['pred'] else 'R') + '|' + ('B' if r['meas'] else 'R')
            conf[key] = conf.get(key, 0) + 1
        fa = sum(1 for r in beyond if r['pred'] and not r['meas'])
        fr = sum(1 for r in beyond if not r['pred'] and r['meas'])
        print('\n=== at |D| = %d ===' % a.D)
        print('  agree %d/%d   tie %d/%d   beyond %d/%d'
              % (agree, len(rows),
                 sum(1 for r in tie if r['agree']), len(tie),
                 sum(1 for r in beyond if r['agree']), len(beyond)))
        print('  confusion %s   false admit beyond tie %d   false refuse %d'
              % (conf, fa, fr))
        print('\n  kappa_eff by density:')
        for d in DENS:
            v = [r['kappa_eff'] for r in rows if r['d'] == d]
            print('    d %.3f  median %.3f  [%.3f, %.3f]'
                  % (d, float(np.median(v)), float(np.min(v)), float(np.max(v))))
        res = dict(reps=a.reps, mode='grid', D=a.D, cells=len(rows),
                   agree=agree, tie=len(tie), beyond=len(beyond),
                   confusion=conf, false_admit=fa, false_refuse=fr, rows=rows)

    out = a.out or os.path.join(OUT, 'fixedD_sweep.json' if a.sweep
                                else 'fixedD_grid.json')
    os.makedirs(OUT, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=float)
    print('\nwall %.0fs, wrote %s' % (time.time() - t0, out))


if __name__ == '__main__':
    main()
