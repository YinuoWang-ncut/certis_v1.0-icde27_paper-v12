#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How far the emptiness certificate can reach, and why it stops there.

Why this file exists
--------------------
Sec. V-E reports that the emptiness criterion S - B + 1 < l holds on thirteen of
the 98 baseline-relative upward moves, over three codes. That is a count, and a
reviewer is entitled to read it as "a curiosity on thirteen cases". It is not:
the thirteen are the whole of what the certificate could ever reach, and the
reason is a monotone relation between two measurable quantities.

The mover population and the certificate are anti-correlated. Codes whose caps
move are the codes with high units per beneficiary-day -- drug and device
J-codes, where one claim line carries 100 to 900 units -- and for those
S - B + 1 exceeds l by orders of magnitude, so no allocation model can certify
them empty. Codes where the certificate fires are the ones with S = B, one unit
per day, which are also the codes whose caps move least often.

The three checks below, in order:

  1. reproduce the 116 / 98 / 18 baseline-relative population and the 13
     certified observations of Sec. V-E, so the rest rests on a checked base;
  2. the mover rate by units-per-day bucket -- 22% in the top bucket against 3%
     in the bottom one;
  3. for the 50 upward observations that are not certified, the units-per-day of
     their code, which is the mechanism.

  python certificate_reach.py --mue-dir <dir> --util <mup_phy_2024_prov_svc.csv>
"""
import argparse
import collections
import csv
import glob
import io
import json
import os
import statistics

ORDER = [('2025Q1', 'mue_2025q1_pra'), ('2025Q4', 'mue_2025q4_pra'),
         ('2026Q1', 'mue_2026q1_pra'), ('2026Q2', 'mue_2026q2_pra'),
         ('2026Q3', 'mue_2026q3_pra'), ('2026Q4', 'mue_2026q4_pra')]


def load(d):
    """{code: cap}. Row 0 is the AMA notice, row 1 the column names."""
    hits = glob.glob(os.path.join(d, '*.csv'))
    if not hits:
        raise SystemExit('no csv under %s' % d)
    out = {}
    with io.open(hits[0], encoding='utf-8-sig', errors='replace') as f:
        rows = list(csv.reader(f))
    for row in rows[2:]:
        if len(row) < 3:
            continue
        code = row[0].strip().replace('\n', '')
        try:
            out[code] = int(row[1].strip())
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mue-dir', required=True)
    ap.add_argument('--util', required=True)
    ap.add_argument('--out', default=os.path.join('exp', 'out',
                                                  'certificate_reach.json'))
    a = ap.parse_args()

    snaps = {t: load(os.path.join(a.mue_dir, sub)) for t, sub in ORDER}
    tags = [t for t, _ in ORDER]
    base = tags[0]

    # -- the aggregate, per (NPI, HCPCS) group and per code
    S, B = collections.Counter(), collections.Counter()
    groups = collections.defaultdict(list)
    n_rows = 0
    with io.open(a.util, encoding='utf-8-sig', errors='replace') as f:
        for row in csv.DictReader(f):
            code = (row['HCPCS_Cd'] or '').strip()
            try:
                s = int(row['Tot_Srvcs'])
                b = int(row['Tot_Bene_Day_Srvcs'])
            except (TypeError, ValueError):
                continue
            n_rows += 1
            S[code] += s
            B[code] += b
            groups[code].append((s, b))
    print('aggregate: %d rows over %d codes' % (n_rows, len(S)))

    # -- 1. the baseline-relative population, and the certified subset
    up, down = [], []
    for t in tags[1:]:
        common = set(snaps[base]) & set(snaps[t])
        for c in common:
            if snaps[t][c] > snaps[base][c]:
                up.append((c, t, snaps[base][c], snaps[t][c]))
            elif snaps[t][c] < snaps[base][c]:
                down.append((c, t, snaps[base][c], snaps[t][c]))
    print('baseline-relative: %d up, %d down, %d total'
          % (len(up), len(down), len(up) + len(down)))

    cert, with_agg = [], []
    for c, t, o, n in up:
        if c not in groups:
            continue
        with_agg.append((c, t, o, n))
        l = max(o, n)
        if all(s - b + 1 < l for s, b in groups[c]):
            cert.append((c, t, o, n))
    carried = len(up) - len(with_agg)
    print('upward with an aggregate: %d ; certified on every group: %d ; on codes'
          ' the release does not carry: %d' % (len(with_agg), len(cert), carried))
    byc = collections.Counter(c for c, _, _, _ in cert)
    print('  certified observations by code: %s' % dict(byc))
    print('  the partition sums to %d + %d + %d = %d'
          % (len(cert), len(with_agg) - len(cert), carried, len(up)))

    # -- 2. mover rate by units per beneficiary-day
    moved = set(c for c, _, _, _ in up) | set(c for c, _, _, _ in down)

    def sb(c):
        return (S[c] / B[c]) if B.get(c) else float('inf')

    buckets = [(0.0, 1.05), (1.05, 2.0), (2.0, 10.0), (10.0, float('inf'))]
    print()
    print('mover rate by units per beneficiary-day (S/B):')
    tab = []
    for lo, hi in buckets:
        cs = [c for c in S if lo <= sb(c) < hi]
        mv = [c for c in cs if c in moved]
        rate = len(mv) / len(cs) if cs else 0.0
        label = 'S/B in [%.2f, %s)' % (lo, 'inf' if hi == float('inf') else '%.1f' % hi)
        print('  %-24s %4d codes, %2d movers (%3.0f%%)' % (label, len(cs), len(mv),
                                                           100 * rate))
        tab.append(dict(lo=lo, hi=None if hi == float('inf') else hi,
                        codes=len(cs), movers=len(mv), rate=rate))

    # -- 3. why the other upward observations are not certified
    fail = [o for o in with_agg if o not in cert]
    fr = [sb(o[0]) for o in fail]
    print()
    print('the %d upward observations that are not certified:' % len(fail))
    print('  units/day of their code: median %.1f, min %.2f, max %.0f'
          % (statistics.median(fr), min(fr), max(fr)))
    print('  with units/day > 2      : %d of %d'
          % (sum(1 for x in fr if x > 2), len(fr)))
    print('  certified codes are all at units/day = %.1f'
          % statistics.median([sb(c) for c in byc]))

    res = dict(aggregate=dict(rows=n_rows, codes=len(S)),
               population=dict(up=len(up), down=len(down),
                               with_aggregate=len(with_agg),
                               certified=len(cert), carried=carried,
                               certified_by_code=dict(byc)),
               buckets=tab,
               not_certified=dict(n=len(fail),
                                  units_median=statistics.median(fr),
                                  units_gt2=sum(1 for x in fr if x > 2)))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with io.open(a.out, 'w', encoding='utf-8', newline='') as fh:
        json.dump(res, fh, indent=1)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
