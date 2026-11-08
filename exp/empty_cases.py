"""One line per empty-band case, in the form the paper's Sect. V-G states it.

The paper's certificate is arithmetic on two published columns, not a property
of our prototype:

    a group with S units billed and B beneficiary-days served has at least B
    lines, each carrying at least one unit, so the largest single line it can
    hold is S - B + 1.  A band [l, h) is unreachable in that group iff
    S - B + 1 < l, i.e. iff the slack  l - 1 - S + B  is positive.  A move is
    necessarily empty iff every observed group of its code is.

This script prints that slack, per group, for every MUE move, and writes
exp/empty_cases.csv with one line per case carrying the minimum over groups.
Anyone with the two CMS tables can therefore check the "thirteen of 98"
without reading our code: sort the slack column, count the positive minima.

    python exp/empty_cases.py            # the certified-empty moves only
    python exp/empty_cases.py --all      # every observed move, slacks included

Columns: code,q,from,to,l,groups,min_slack,certified_empty
  l              lower endpoint of the band the move opens (from + 1)
  groups         observed (NPI, HCPCS) groups for that code, CY2024
  min_slack      min over groups of l - 1 - S_g + B_g;  > 0  <->  empty
  certified_empty  1 iff min_slack > 0

Inputs are the same two releases exp/utilization.py reads (set CERTIS_DATA):
  data/cms/mup_phy_2024_prov_svc.csv
  data/mue/mue_<q>_pra.zip   for q in 2025q1 2025q4 2026q1..q4
Output: exp/empty_cases.csv
"""
import collections
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from delta import mue_table                                     # noqa: E402

DATA = os.environ.get("CERTIS_DATA") or os.path.join(os.path.dirname(HERE), "data")
PROV = os.path.join(DATA, "cms", "mup_phy_2024_prov_svc.csv")
MUE = os.path.join(DATA, "mue")
OUT = os.path.join(HERE, "empty_cases.csv")

Q = ["2025q1", "2025q4", "2026q1", "2026q2", "2026q3", "2026q4"]
WANT = ("Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs", "Tot_Bene_Day_Srvcs")


def num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def read_groups(codes):
    """(npi, hcpcs) -> [units, bene_days], restricted to the codes we need."""
    g = collections.defaultdict(lambda: [0, 0])
    with io.open(PROV, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {k: head.index(k) for k in WANT}
        i_npi, i_c = ix["Rndrng_NPI"], ix["HCPCS_Cd"]
        i_s, i_d = ix["Tot_Srvcs"], ix["Tot_Bene_Day_Srvcs"]
        for row in rd:
            if row[i_c] not in codes:
                continue
            a = g[(row[i_npi], row[i_c])]
            a[0] += num(row[i_s])
            a[1] += num(row[i_d])
    return g


def main(argv):
    show_all = "--all" in argv
    if not os.path.exists(PROV):
        raise SystemExit(
            "need the CMS provider-service release:\n  %s\n"
            "  data.cms.gov -> 'Medicare Physician & Other Practitioners' -> "
            "'by Provider and Service', practitioner, 2024" % PROV)

    caps = {}
    for q in Q:
        t = mue_table(os.path.join(MUE, "mue_%s_pra.zip" % q))
        if t:
            caps[q] = t
    base = caps[Q[0]]

    moves = []
    for q in Q[1:]:
        for c, m0 in base.items():
            if c in caps[q] and caps[q][c] != m0:
                moves.append({"code": c, "q": q, "from": m0, "to": caps[q][c]})

    by_code = collections.defaultdict(list)
    for (_, c), (s, d) in read_groups({m["code"] for m in moves}).items():
        by_code[c].append((s, d))

    rows, empty = [], 0
    for m in moves:
        lo = min(m["from"], m["to"]) + 1
        groups = by_code.get(m["code"])
        if not groups:
            continue
        slack = min(lo - 1 - s + max(1, d) for s, d in groups)
        ok = slack > 0
        empty += ok
        rows.append([m["code"], m["q"], m["from"], m["to"], lo,
                     len(groups), slack, 1 if ok else 0])

    rows.sort(key=lambda r: (not r[7], r[0], r[1]))
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["code", "q", "from", "to", "l", "groups",
                    "min_slack", "certified_empty"])
        for r in rows:
            if show_all or r[7]:
                w.writerow(r)

    up = sum(1 for m in moves if m["to"] > m["from"])
    print("moves observed: %d (%d up, %d down)" % (len(moves), up, len(moves) - up))
    print("certified empty: %d of %d upward moves" % (empty, up))
    print("wrote", OUT, "(%d rows%s)" % (empty if not show_all else len(rows),
                                         "" if show_all else ", certified only"))


if __name__ == "__main__":
    main(sys.argv[1:])
