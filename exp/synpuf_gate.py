#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The feasibility gate for the DE-SynPUF branch (D1-D3 of the 30-day plan).

Why this file exists
--------------------
The paper's per-code scope sizes and the emptiness certificate both rest on two
columns of a CMS utilization aggregate: units billed S and beneficiary-days B.
Neither column can be checked against claims, because no public release carries
per-line units -- the DE-SynPUF carrier and outpatient files ship HCPCS codes
but no REV_CNTR_UNIT_CNT, and the OMOP conversion of them leaves QUANTITY empty.
So the question this gate settles is not "can we replace the aggregate" (we
cannot) but three narrower ones:

  1. COVERAGE. Do the codes the paper's MUE corpus actually moved appear in the
     claims at all? If they do not, nothing below can be computed and the branch
     is dead -- which is worth knowing on day one rather than day twenty.

  2. THE PREMISE of Eq. (13). The certificate S - B + 1 < l is derived from a
     line count C with C >= B: "every beneficiary-day must appear on at least one
     line, and further lines only raise C". That is an assertion about claims, and
     claims can test it directly. This is the check with the highest value per
     hour in the whole 30-day plan, because it validates an assumption the paper
     currently only argues.

  3. PAIR CO-OCCURRENCE. Family B's ripple radius currently has only a bound,
     |rho| <= min(n_c1, n_c2). A visit-level co-occurrence count turns the bound
     into a measurement -- or shows that the bound is all the data supports.

Inputs are local and public; nothing here downloads anything.

  python synpuf_gate.py --proc <procedure_occurrence.csv[.gz|.bz2]> \\
      --mue-dir <dir with mue_YYYYqN_pra/*.csv> --util <mup_phy_2024_prov_svc.csv> \\
      --ptp <ccioph f1 zip> --out exp/out/synpuf_gate.json
"""
import argparse
import bz2
import collections
import csv
import glob
import gzip
import io
import json
import os
import statistics
import sys
import zipfile

# The six MUE releases the paper's Table IV is computed from, in release order.
ORDER = [('2025Q1', 'mue_2025q1_pra'), ('2025Q4', 'mue_2025q4_pra'),
         ('2026Q1', 'mue_2026q1_pra'), ('2026Q2', 'mue_2026q2_pra'),
         ('2026Q3', 'mue_2026q3_pra'), ('2026Q4', 'mue_2026q4_pra')]


def opener(path):
    low = path.lower()
    if low.endswith('.bz2'):
        return bz2.open(path, 'rt', encoding='utf-8', errors='replace')
    if low.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return io.open(path, encoding='utf-8', errors='replace')


# ------------------------------------------------------------------ the corpus
def mue_snapshots(mue_dir):
    """{tag: {code: (cap, mai)}}. The header is two rows down: row 0 is the AMA
    notice, row 1 the four column names."""
    out = {}
    for tag, sub in ORDER:
        hits = glob.glob(os.path.join(mue_dir, sub, '*.csv'))
        if not hits:
            raise SystemExit('missing MUE release %s under %s' % (sub, mue_dir))
        table = {}
        with io.open(hits[0], encoding='utf-8-sig', errors='replace') as f:
            rows = list(csv.reader(f))
        for row in rows[2:]:
            if len(row) < 3:
                continue
            code = row[0].strip().replace('\n', '')
            try:
                cap = int(row[1].strip())
            except ValueError:
                continue
            table[code] = (cap, ' '.join(row[2].split()))
        out[tag] = table
    return out


def movers(snaps):
    """Move events across successive observed releases, and the codes carrying
    them. Both counts matter and they are not the same number: 57 events touch
    57 codes, of which only 29 are present in both endpoints -- which is where
    the paper's '29 of 15,212' comes from."""
    tags = [t for t, _ in ORDER]
    events = []
    hit = set()
    for a, b in zip(tags, tags[1:]):
        A, B = snaps[a], snaps[b]
        for code in set(A) & set(B):
            if A[code][0] != B[code][0]:
                events.append((code, A[code][0], B[code][0], a, b))
                hit.add(code)
    in_both = set(snaps[tags[0]]) & set(snaps[tags[-1]])
    return events, hit, hit & in_both, in_both


def tp_pairs(zip_path, limit=None):
    """Column1/Column2 pairs whose deletion date is in the future or absent --
    the edits a correction can add or remove."""
    pairs = []
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.lower().endswith('.txt')][0]
        with z.open(name) as raw:
            txt = io.TextIOWrapper(raw, encoding='utf-8', errors='replace')
            for line in txt:
                f = line.rstrip('\n').split('\t')
                if len(f) < 5 or not f[0] or not f[1]:
                    continue
                if f[0] == 'Column 1':
                    continue
                pairs.append((f[0].strip(), f[1].strip(), f[4].strip()))
                if limit and len(pairs) >= limit:
                    break
    return pairs


# ------------------------------------------------------------------ the claims
def scan_claims(path, want_codes):
    """One pass over the OMOP procedure_occurrence.

    Returns per-(provider, code): line count C, distinct (person, date) B, and
    per-code distinct (person, date); plus visit -> set of codes for the pair
    check, restricted to the codes the caller cares about.
    """
    C = collections.Counter()
    days = collections.defaultdict(set)
    code_days = collections.defaultdict(set)
    code_lines = collections.Counter()
    visit_codes = collections.defaultdict(set)
    rows = 0
    with opener(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if 'PROCEDURE_SOURCE_VALUE' not in fields:
            raise SystemExit('no PROCEDURE_SOURCE_VALUE column; got %s' % fields[:6])
        has_visit = 'VISIT_OCCURRENCE_ID' in fields
        has_prov = 'PROVIDER_ID' in fields
        for row in reader:
            code = (row.get('PROCEDURE_SOURCE_VALUE') or '').strip()
            if not code:
                continue
            rows += 1
            person = (row.get('PERSON_ID') or '').strip()
            date = (row.get('PROCEDURE_DATE') or '').strip()
            prov = (row.get('PROVIDER_ID') or '').strip() if has_prov else ''
            k = (prov, code)
            C[k] += 1
            days[k].add((person, date))
            code_days[code].add((person, date))
            code_lines[code] += 1
            if code in want_codes and has_visit:
                v = (row.get('VISIT_OCCURRENCE_ID') or '').strip()
                if v:
                    visit_codes[v].add(code)
    return dict(C=C, days=days, code_days=code_days, code_lines=code_lines,
                visit_codes=visit_codes, rows=rows, has_visit=has_visit)


def util_by_code(path):
    """{code: (S, B)} summed over providers, from the four-column extract."""
    S = collections.Counter()
    B = collections.Counter()
    with io.open(path, encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        need = {'HCPCS_Cd', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs'}
        if not need.issubset(set(cols)):
            raise SystemExit('unexpected utilization columns: %s' % cols)
        for row in reader:
            code = (row['HCPCS_Cd'] or '').strip()
            try:
                s = int(row['Tot_Srvcs'])
                b = int(row['Tot_Bene_Day_Srvcs'])
            except (TypeError, ValueError):
                continue
            S[code] += s
            B[code] += b
    return S, B


# -------------------------------------------------------------------- the gate
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc', required=True)
    ap.add_argument('--mue-dir', required=True)
    ap.add_argument('--util', required=True)
    ap.add_argument('--ptp', required=True)
    ap.add_argument('--out', default=os.path.join('exp', 'out', 'synpuf_gate.json'))
    a = ap.parse_args()

    snaps = mue_snapshots(a.mue_dir)
    events, hit, in_both, endpoints = movers(snaps)
    print('MUE: %d move events over %d codes; %d of them in both endpoints (of %d)'
          % (len(events), len(hit), len(in_both), len(endpoints)))

    pairs = tp_pairs(a.ptp)
    pair_codes = {c for c1, c2, _ in pairs for c in (c1, c2)}
    print('PTP: %d Column1/Column2 edits over %d distinct codes'
          % (len(pairs), len(pair_codes)))

    want = hit | pair_codes
    store = scan_claims(a.proc, want)
    print('claims: %d procedure rows' % store['rows'])

    # ---- 1. coverage -----------------------------------------------------
    present = set(store['code_days'])
    cov_movers = sorted(hit & present)
    cov_both = sorted(in_both & present)
    cov_pair = sorted(pair_codes & present)
    print()
    print('CHECK 1  coverage in the claims')
    print('  mover codes present      : %d / %d' % (len(cov_movers), len(hit)))
    print('  of the 29 in both ends   : %d / %d' % (len(cov_both), len(in_both)))
    print('  PTP pair codes present   : %d / %d' % (len(cov_pair), len(pair_codes)))

    # ---- 2. the premise of Eq. (13) --------------------------------------
    viol = 0
    ratios = []
    for k, c in store['C'].items():
        b = len(store['days'][k])
        if b == 0:
            continue
        if c < b:
            viol += 1
        ratios.append(c / b)
    ratios.sort()
    exact = sum(1 for r in ratios if r == 1.0)
    print()
    print('CHECK 2  C >= B, the premise of the emptiness certificate')
    print('  groups (provider, code)  : %d' % len(ratios))
    print('  violations of C >= B     : %d (%.3f%%)'
          % (viol, 100.0 * viol / max(len(ratios), 1)))
    print('  C/B median %.2f, p10 %.2f, p90 %.2f, max %.1f'
          % (statistics.median(ratios), ratios[int(len(ratios) * .10)],
             ratios[int(len(ratios) * .90)], ratios[-1]))
    print('  C == B exactly           : %d (%.1f%%)'
          % (exact, 100.0 * exact / max(len(ratios), 1)))

    # does B behave like "days" in the aggregate too? Compare B/S, which is
    # scale-free, so the 2008-2010 vs CY2024 gap is a caveat rather than a
    # blocker.
    S, Bu = util_by_code(a.util)
    common = sorted(set(store['code_days']) & set(Bu))
    pairs_bs = []
    for code in common:
        days = len(store['code_days'][code])
        lines = store['code_lines'][code]
        if lines == 0 or S.get(code, 0) == 0:
            continue
        pairs_bs.append((code, days / lines, Bu[code] / S[code]))
    print('  codes in claims and in the CY2024 aggregate: %d' % len(pairs_bs))
    if pairs_bs:
        claim_bs = sorted(p[1] for p in pairs_bs)
        util_bs = sorted(p[2] for p in pairs_bs)
        print('  B/S  claims  median %.3f  p90 %.3f' % (
            statistics.median(claim_bs), claim_bs[int(len(claim_bs) * .9)]))
        print('  B/S  CY2024  median %.3f  p90 %.3f' % (
            statistics.median(util_bs), util_bs[int(len(util_bs) * .9)]))
        same_side = sum(1 for _, a_, b_ in pairs_bs if (a_ > 1) == (b_ > 1))
        print('  agree on B/S > 1         : %d / %d' % (same_side, len(pairs_bs)))

    # ---- 3. pair co-occurrence -------------------------------------------
    print()
    print('CHECK 3  pair co-occurrence, the ripple radius of Family B')
    if not store['has_visit']:
        print('  no VISIT_OCCURRENCE_ID: cannot group lines into a claim')
        co = {}
    else:
        both = 0
        bound_ok = 0
        co = {}
        for c1, c2, _ in pairs:
            if c1 not in present or c2 not in present:
                continue
            n = sum(1 for codes in store['visit_codes'].values()
                    if c1 in codes and c2 in codes)
            co[(c1, c2)] = n
            both += 1
            if n <= min(len(store['code_days'][c1]), len(store['code_days'][c2])):
                bound_ok += 1
        print('  pairs with both codes in claims : %d' % both)
        print('  pairs with at least one hit     : %d' % sum(1 for v in co.values() if v))
        print('  bound |rho| <= min(n_c1,n_c2) holds: %d / %d' % (bound_ok, both))
        if co:
            vals = sorted(co.values())
            print('  co-occurring visits: median %d, p90 %d, max %d'
                  % (statistics.median(vals), vals[int(len(vals) * .9)], vals[-1]))

    res = dict(
        mue=dict(events=len(events), mover_codes=len(hit),
                 in_both=sorted(in_both), endpoints=len(endpoints)),
        ptp=dict(pairs=len(pairs), codes=len(pair_codes)),
        claims=dict(rows=store['rows'], codes=len(present),
                    has_visit=bool(store['has_visit'])),
        coverage=dict(movers=len(cov_movers), movers_of=len(hit),
                      in_both=len(cov_both), in_both_of=len(in_both),
                      pair_codes=len(cov_pair), pair_codes_of=len(pair_codes),
                      mover_codes_present=cov_movers),
        premise=dict(groups=len(ratios), violations=viol,
                     violation_rate=viol / max(len(ratios), 1),
                     ratio_median=statistics.median(ratios) if ratios else None,
                     exact_equal=exact,
                     codes_with_aggregate=len(pairs_bs),
                     bs_claim_median=(statistics.median([p[1] for p in pairs_bs])
                                      if pairs_bs else None),
                     bs_util_median=(statistics.median([p[2] for p in pairs_bs])
                                     if pairs_bs else None)),
        pairs=dict(scored=len(co),
                   with_hit=sum(1 for v in co.values() if v),
                   samples={('%s+%s' % k): v for k, v in list(co.items())[:40]}),
    )
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with io.open(a.out, 'w', encoding='utf-8', newline='') as fh:
        json.dump(res, fh, indent=1)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
