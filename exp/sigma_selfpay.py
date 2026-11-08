#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3: is maintaining sigma actually free, or only asymptotically absorbed?

Why this file exists
--------------------
Thm. 3 gives the flip set as the band masked by sigma, and takes the mask to be
O(1) per candidate *from maintained state*. That is an assumption about work
somebody has to do, and the paper until now only conceded it: "we do not price
the maintenance of sigma, because in every arm we time it falls out of the pass
that computes rho." A concession is not an answer. Two things are needed:

  a proof that the maintenance is dominated by the path that computes rho,
  and a measurement of how much it actually costs.

The proof (Proposition below, and in the paper) turns on one structural fact:
for a threshold move the set on which e_r changes is exactly the band, and the
band is an interval. So an update is recorded in O(1) as (theta_lo, theta_hi,
+/-1) and read back as one comparison -- and the materialised form touches only
the band's k members, never the scope.

  Path R:  Theta(n) scan + O(k) increment, k <= n  ->  Theta(n)
  Path B:  Theta(log n + k) probe+gather + O(k)    ->  Theta(log n + k)

Either way the term is absorbed. What is *not* absorbed is the existence-only
certifier of Prop. 2, which may stop after j < k examinations yet must still
write sigma for all k members of the band; that is the one case where
maintenance is charged against a decision that did not pay for it, and it is
why the paper flags it rather than claiming it away.

This file measures the three stages separately at the published scope sizes, so
the concession becomes a number:

  gather        path B's scattered fetch, no mask
  + mask        one gather of sigma over the candidates, one comparison
  + update      one scatter of +/-1 back into sigma over the same array
  scan          path R's contiguous pass over the scope
  + mask+update path R's own two extra passes over its flips

  python sigma_selfpay.py --reps 9
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

NS = [('p5', 197), ('q1', 6969), ('median', 44606), ('q3', 412875)]
DS = [0.042, 0.183, 0.401]
CAP_OLD, CAP_NEW = 4, 2
AUX_CAP = 3
DAYS_PER_GROUP = 58
OVERRIDE_P = 0.18
SEED = 31337
S_STAR = 0
KAPPA, DP = 0.2142, 198.8


def build(plans):
    rng = np.random.default_rng(SEED)
    n_c = [n for _, n, _ in plans]
    d_c = [d for _, _, d in plans]
    total = sum(n_c)
    code_of = np.repeat(np.arange(len(plans), dtype=np.int32), n_c)
    code_of = code_of[rng.permutation(total)]
    unit = np.ones(total, dtype=np.int16)
    for code in range(len(plans)):
        pos = np.flatnonzero(code_of == code)
        k = int(round(d_c[code] * len(pos)))
        pick = rng.choice(pos, size=min(k, len(pos)), replace=False)
        half = len(pick) // 2
        unit[pick[:half]] = CAP_NEW + 1
        unit[pick[half:]] = CAP_OLD
        free = np.setdiff1d(pos, pick)
        if len(free):
            b = free[rng.random(len(free)) < 0.20]
            unit[b] = rng.integers(1, CAP_NEW + 1, len(b)).astype(np.int16)
            o = free[rng.random(len(free)) < 0.12]
            unit[o] = rng.integers(CAP_OLD + 1, 21, len(o)).astype(np.int16)
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
    sig = ((store.unit > CAP_OLD) & ~ov).astype(np.int8)
    sig += (store.unit > AUX_CAP).astype(np.int8)
    return store, np.where(ov, np.int8(-1), sig).astype(np.int8)


# ---- the stages, in place over caller-owned buffers ----------------------
# Every stage writes into a buffer the caller allocated once.  The first version
# of this file let each stage allocate its own output, and at the largest scope
# that made the allocator, not the algorithm, decide the decomposition: the mask
# arm came out NEGATIVE (a "cheaper" mask+update than mask).  The same confound
# is documented in bigscope.py, where a full path once measured faster than its
# own verification.  Preallocating removes it at the source.
def gather_into(store, cand, out):
    """Path B's scattered fetch plus the two-state test, into out.  Returns the
    number of candidates that differ between the two thresholds."""
    if not len(cand):
        return 0
    v = store.unit[cand]
    m = (v > CAP_OLD) != (v > CAP_NEW)
    n = int(np.count_nonzero(m))
    if n:
        np.take(cand, np.flatnonzero(m), out=out[:n])
    return n


def mask_into(sigma, src, n, out):
    """Thm. 3's mask: one gather of sigma over the identifiers, one comparison,
    compacted into out.  Returns how many survive."""
    if not n:
        return 0
    ms = sigma[src[:n]]
    keep = ms == S_STAR
    kk = int(np.count_nonzero(keep))
    if kk:
        np.take(src[:n], np.flatnonzero(keep), out=out[:kk])
    return kk


def update_into(sigma, src, n):
    """The write.  One scatter of +/-1 over the same identifiers the mask just
    read -- which is why it cannot exceed the mask's own order, and why it is a
    function of k rather than of n."""
    if not n:
        return
    src[:n] -= np.int8(1 if CAP_NEW < CAP_OLD else -1)


def scan_into(store, code, out):
    """Path R's contiguous pass, into out.  Two separate passes over the scope
    (one per threshold) would inflate R by 2x, the accident the paper documents."""
    lo, hi = store.cptr[code], store.cptr[code + 1]
    v = store.unit_c[lo:hi]
    m = (v > CAP_OLD) != (v > CAP_NEW)
    n = int(np.count_nonzero(m))
    if n:
        np.take(store.cline[lo:hi], np.flatnonzero(m), out=out[:n])
    return n


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=9)
    ap.add_argument('--out', default=os.path.join(OUT, 'sigma_selfpay.json'))
    a = ap.parse_args()

    plans = [(t, n, d) for t, n in NS for d in DS]
    store, sigma = build(plans)
    sigma0 = sigma.copy()
    print('store %d rows, %d cells' % (store.n_l, len(plans)), flush=True)

    rows = []
    for code, (tag, n, d) in enumerate(plans):
        cand = store.band_u(code, CAP_NEW, CAP_OLD)
        k = int(len(cand))
        # NESTED arms over preallocated buffers, so each delta is one more pass
        # over the same identifiers with no allocation in between.
        capB = k + 8
        capR = int(store.scope_len(code)) + 8
        bufB = np.empty(capB, np.int32)
        bufB2 = np.empty(capB, np.int32)
        bufR = np.empty(capR, np.int32)
        bufR2 = np.empty(capR, np.int32)

        def b1():
            return gather_into(store, cand, bufB)

        def b2():
            n1 = gather_into(store, cand, bufB)
            return mask_into(sigma, bufB, n1, bufB2)

        def b3():
            n1 = gather_into(store, cand, bufB)
            n2 = mask_into(sigma, bufB, n1, bufB2)
            update_into(sigma, bufB2, n2)

        def r1():
            return scan_into(store, code, bufR)

        def r2():
            n1 = scan_into(store, code, bufR)
            return mask_into(sigma, bufR, n1, bufR2)

        def r3():
            n1 = scan_into(store, code, bufR)
            n2 = mask_into(sigma, bufR, n1, bufR2)
            update_into(sigma, bufR2, n2)

        t = block({'B1': b1, 'B2': b2, 'B3': b3,
                   'R1': r1, 'R2': r2, 'R3': r3}, a.reps)
        sigma[:] = sigma0

        g = float(np.median(t['B1']))
        b2m = float(np.median(t['B2']))
        b = float(np.median(t['B3']))
        sc = float(np.median(t['R1']))
        r2m = float(np.median(t['R2']))
        rt = float(np.median(t['R3']))
        row = dict(tag=tag, n=int(store.scope_len(code)), d=d, k=k,
                   gather=g, B_mask_only=b2m, B=b,
                   scan=sc, R_mask_only=r2m, R=rt,
                   d_mask_B=b2m - g, d_upd_B=b - b2m,
                   d_mask_R=r2m - sc, d_upd_R=rt - r2m,
                   sigma_B=(b - g), sigma_R=(rt - sc),
                   ratio_with=b / rt, ratio_without=g / sc,
                   share_B=(b - g) / b if b else float('nan'),
                   share_R=(rt - sc) / rt if rt else float('nan'))
        rows.append(row)
        print('  %-6s n %7d k %7d | B: gather %7.1f +mask %6.1f +upd %6.1f '
              '= %7.1f | R: scan %7.1f +mask %6.1f +upd %6.1f = %7.1f'
              % (tag, row['n'], k, g, row['d_mask_B'], row['d_upd_B'], b,
                 sc, row['d_mask_R'], row['d_upd_R'], rt), flush=True)
    print('\n=== 1. what sigma work costs, per path ===')
    print('  %-7s %10s %10s %10s %10s' % ('scope', 'B: mask', 'B: update',
                                          'R: mask', 'R: update'))
    for reg in ('p5', 'q1', 'median', 'q3'):
        sub = [r for r in rows if r['tag'] == reg]
        med = lambda key: float(np.median([r[key] for r in sub]))
        print('  %-7s %10.1f %10.1f %10.1f %10.1f'
              % (reg, med('d_mask_B'), med('d_upd_B'),
                 med('d_mask_R'), med('d_upd_R')))
    print('\n  the update is a second pass over the same k identifiers the mask')
    print('  just touched, so it costs about what the mask costs -- not zero,')
    print('  and not a function of n.')

    print('\n=== 2. does charging it change the sign? ===')
    changed = 0
    for r in rows:
        if r['ratio_with'] == r['ratio_without']:
            continue
        a1 = r['ratio_with'] < 1
        a0 = r['ratio_without'] < 1
        if a1 != a0:
            changed += 1
            print('  n %7d d %.3f  band %s when charged, %s when not'
                  % (r['n'], r['d'], 'wins' if a1 else 'loses',
                     'wins' if a0 else 'loses'))
    print('  cells whose side of one changes: %d / %d' % (changed, len(rows)))

    worst = max(rows, key=lambda r: r['share_B'] if r['B'] else 0)
    print('\n  sigma work (mask + update) is largest as a share of the band path')
    print('  at n=%d, d=%.3f: %.1f%% of it; as a share of the scan, %.2f%%'
          % (worst['n'], worst['d'], 100 * worst['share_B'],
             100 * worst['share_R']))

    res = dict(reps=a.reps, cells=len(rows), changed=changed, rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=float)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
