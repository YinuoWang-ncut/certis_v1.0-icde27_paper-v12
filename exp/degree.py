#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Degree of the real edit set: how many rules touch one claim?

Why this matters.  The band is a *superset* of the flip set: it contains every
claim whose judged value lies between the two thresholds, and the exact flip set
is that superset intersected with a masking test -- "was the moved rule the only
support of the verdict?".  The masking test costs one evaluation per *other*
applicable rule, so the band's cost per candidate is a_p + d*a_e, not a_p.
d is therefore a first-class quantity, and it is a property of the published
rule table, not of the workload.  We measure it here.

Output: exp/degree.json
"""
import glob
import io
import json
import os
import zipfile
from collections import Counter

_DATA = os.environ.get("CERTIS_DATA") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW = os.path.join(_DATA, "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "degree.json")


def read_pairs(paths):
    """(c1, c2, eff, del, modifier) for every row of every shard."""
    rows = []
    for p in paths:
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                with z.open(name) as f:
                    txt = io.TextIOWrapper(f, encoding="cp1252", errors="replace")
                    for line in txt:
                        f0 = line.rstrip("\r\n").split("\t")
                        if len(f0) < 6:
                            continue
                        c1, c2 = f0[0].strip(), f0[1].strip()
                        # a data row starts with two codes; the banner rows do not
                        if not c1 or not c2 or " " in c1 or "/" in c1:
                            continue
                        rows.append((c1, c2, f0[3].strip(), f0[4].strip(),
                                     f0[5].strip()))
    return rows


def stats(cnt):
    v = sorted(cnt.values())
    n = len(v)

    def q(p):
        return v[min(n - 1, int(p * n))]

    return {
        "codes": n,
        "mean": round(sum(v) / n, 2),
        "p50": q(0.50), "p75": q(0.75), "p90": q(0.90),
        "p99": q(0.99), "max": v[-1],
    }


def main():
    sets = {
        "2026Q3 practitioner": sorted(glob.glob(os.path.join(RAW, "q2026q3_ccipra_f*.zip"))),
        "2026Q3 hospital": sorted(glob.glob(os.path.join(
            RAW, "medicare-ncci-2026q3-hospital-ptp-edits-*-f*.zip"))),
        "2026Q4 practitioner": sorted(glob.glob(os.path.join(RAW, "q2026q4_ccipra_f*.zip"))),
        "2026Q4 hospital": sorted(glob.glob(os.path.join(RAW, "q2026q4_ccioph_f*.zip"))),
    }
    out = {}
    for name, paths in sets.items():
        if not paths:
            continue
        rows = read_pairs(paths)
        if not rows:
            continue
        pairs = set((c1, c2) for c1, c2, _, _, _ in rows)
        # degree of a code: in how many pairs does it appear, in either column
        deg = Counter()
        for c1, c2 in pairs:
            deg[c1] += 1
            deg[c2] += 1
        d1 = Counter()
        for c1, _ in pairs:
            d1[c1] += 1
        # for a *random pair*, the number of rules that touch either of its codes
        per_pair = sorted(deg[c1] + deg[c2] - 1 for c1, c2 in pairs)
        npair = len(per_pair)
        # validity: how many rows already carry a deletion date, and is it past?
        dated = [r for r in rows if r[3] and r[3].isdigit()]
        past = [r for r in dated if int(r[3]) <= 20260930]
        # modifier indicator distribution
        mods = Counter(r[4] for r in rows)
        out[name] = {
            "rows": len(rows),
            "pairs": len(pairs),
            "column_one_codes": len(d1),
            "all_codes": len(deg),
            "degree_all": stats(deg),
            "degree_column_one": stats(d1),
            "rules_mentioning_either_code": {
                "p50": per_pair[npair // 2],
                "mean": round(sum(per_pair) / npair, 1),
                "p90": per_pair[int(0.90 * npair)],
                "p99": per_pair[int(0.99 * npair)],
                "max": per_pair[-1],
            },
            "rows_with_deletion_date": len(dated),
            "rows_deletion_date_past": len(past),
            "modifier_indicator": dict(mods),
        }
        print(name, "rows", len(rows), "pairs", len(pairs),
              "d_median", out[name]["rules_mentioning_either_code"]["p50"],
              "d_p90", out[name]["rules_mentioning_either_code"]["p90"],
              "del_dated", len(dated), "past", len(past))

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
