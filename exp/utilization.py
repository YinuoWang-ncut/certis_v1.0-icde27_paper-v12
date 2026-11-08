"""Section V-G, observed population.

Replaces the population that exp/cheap.py *inverted* from the published caps with
one *taken from* CMS's own utilisation release.

Inputs (set CERTIS_DATA; defaults to ./data):
  data/cms/mup_phy_2024_prov_svc.csv   Medicare Physician & Other Practitioners,
                                       by Provider and Service, CY2024
  data/mue/mue_<q>_pra.zip             the six MUE releases exp/delta.py already reads

Why the old population was a model and this one is a measurement
---------------------------------------------------------------
exp/cheap.py anchored a synthetic population on the earliest cap table: a cap is
the maximum over six months of submitted claims, so it argued the only population
consistent with a published table has per-code maximum *equal* to that cap.  Every
upward move then had an empty band by construction, and the 98/98 it reported is a
property of that construction, not of the data.

The CMS release publishes, for every (NPI, HCPCS) pair, Tot_Srvcs (units billed)
and Tot_Bene_Day_Srvcs (beneficiary-days served).  A beneficiary-day carries at
least one line, so Tot_Bene_Day_Srvcs is a lower bound on the number of lines, and
mu = Tot_Srvcs / Tot_Bene_Day_Srvcs is the mean units per line in that group --
observable, and exactly the quantity an MUE caps.  That is what calibrates n_c (the
code's real service volume) and the atom mass (how many lines sit at each value).

What remains unobservable is the *within-group* spread of units per line: the
release aggregates a group's lines.  We therefore do not invent a spread.  We
report, per move, a decision that follows from the aggregate alone:

  necessarily empty   for every observed group, S_g < lo + C_g - 1, i.e. no
                      allocation of the group's units to its lines can put one
                      line inside [lo, hi).  (Each line carries >= 1 unit.)
  forced overtake     some group has mu_g >= lo, so that group has a line above
                      the old cap, whether or not it lands inside the band.
  inside on the mean  some group has lo <= mu_g < hi: the group's mean already
                      sits in the band the move opens.
  undecided           neither: the aggregate admits both readings.

Output: exp/utilization.json
"""
import collections
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from delta import mue_table                                   # noqa: E402

DATA = os.environ.get("CERTIS_DATA") or os.path.join(os.path.dirname(HERE), "data")
PROV = os.path.join(DATA, "cms", "mup_phy_2024_prov_svc.csv")
GEO = os.path.join(DATA, "cms", "mup_phy_2024_geo.csv")
MUE = os.path.join(DATA, "mue")
OUT = os.path.join(HERE, "utilization.json")

Q = ["2025q1", "2025q4", "2026q1", "2026q2", "2026q3", "2026q4"]
WANT = ("Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs", "Tot_Bene_Day_Srvcs")

# The two releases this section needs are not in data/manifest.csv: at 9.8M
# rows the provider-service table is two orders of magnitude larger than the
# NCCI extracts, and both are CMS publications rather than CPT-derived ones.
# Say so here, where someone will actually read it, instead of dying on an
# OSError three functions later.
if not (os.path.exists(PROV) and os.path.exists(GEO)):
    missing = [p for p in (PROV, GEO) if not os.path.exists(p)]
    raise SystemExit(
        "section V-G needs two CMS releases that the manifest does not list:\n"
        + "".join("  missing: %s\n" % p for p in missing)
        + "\n  Medicare Physician & Other Practitioners, by Provider and Service\n"
        "  (CY2024): data.cms.gov -> 'Medicare Physician & Other Practitioners'\n"
        "  -> 'by Provider and Service', practitioner, 2024; save the two CSVs as\n"
        "    data/cms/mup_phy_2024_prov_svc.csv\n"
        "    data/cms/mup_phy_2024_geo.csv\n"
        "  The shipped exp/utilization.json is this run's output; reproduce.py\n"
        "  prints its tallies without re-deriving them.")


def read_groups(codes):
    """(npi, hcpcs) -> [units, bene_days], for the codes we need only."""
    g = collections.defaultdict(lambda: [0, 0])
    n = 0
    with io.open(PROV, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {k: head.index(k) for k in WANT}
        i_npi, i_c, i_s, i_d = ix["Rndrng_NPI"], ix["HCPCS_Cd"], ix["Tot_Srvcs"], ix["Tot_Bene_Day_Srvcs"]
        for row in rd:
            n += 1
            c = row[i_c]
            if c not in codes:
                continue
            try:
                s = _num(row[i_s])
                d = _num(row[i_d])
            except ValueError:
                continue
            a = g[(row[i_npi], c)]
            a[0] += s
            a[1] += d
    return g, n


def _num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def read_national():
    """HCPCS -> (total units, total bene-days, providers), from the geography file."""
    nat = collections.defaultdict(lambda: [0, 0, 0])
    with io.open(GEO, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["Rndrng_Prvdr_Geo_Lvl"] != "National":
                continue
            a = nat[r["HCPCS_Cd"]]
            a[0] += _num(r["Tot_Srvcs"])
            a[1] += _num(r["Tot_Bene_Day_Srvcs"])
            a[2] = max(a[2], _num(r["Tot_Rndrng_Prvdrs"]))
    return nat


def quant(v, p):
    if not v:
        return None
    v = sorted(v)
    k = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[k]


def main():
    if not (os.path.exists(PROV) and os.path.exists(GEO)):
        res = {"skipped": True,
               "reason": "CY2024 Medicare Physician & Other Practitioners release not "
                         "present; run data/fetch.py (3.3 GB, optional for the rest)"}
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print("SKIP: %s" % PROV)
        return res

    caps = {}
    for q in Q:
        t = mue_table(os.path.join(MUE, "mue_%s_pra.zip" % q))
        if t:
            caps[q] = t
    base = caps[Q[0]]

    moves = []
    for q in Q[1:]:
        t = caps[q]
        for c, m0 in base.items():
            if c in t and t[c] != m0:
                moves.append({"q": q, "code": c, "from": m0, "to": t[c]})
    codes = {m["code"] for m in moves}

    groups, nlines = read_groups(codes)
    nat = read_national()

    per_code = collections.defaultdict(lambda: {"groups": 0, "units": 0, "days": 0, "mu": []})
    by_code = collections.defaultdict(list)
    for (npi, c), (s, d) in groups.items():
        a = per_code[c]
        a["groups"] += 1
        a["units"] += s
        a["days"] += d
        if d > 0:
            a["mu"].append(s / float(d))
        by_code[c].append((s, d))

    res = {"source": {
        "provider_service": os.path.basename(PROV),
        "geo": os.path.basename(GEO),
        "rows_read": nlines,
        "mue_releases": sorted(caps),
    }}

    # ---- coverage -------------------------------------------------------
    have = sorted(c for c in codes if c in per_code)
    res["coverage"] = {
        "move_codes": len(codes),
        "codes_with_observation": len(have),
        "missing_codes": sorted(codes - set(have)),
        "observed_groups": sum(per_code[c]["groups"] for c in have),
        "observed_lines_lb": sum(per_code[c]["days"] for c in have),
        "observed_units": sum(per_code[c]["units"] for c in have),
        "mue_codes_all": len([c for c in base if base[c] >= 1]),
        "mue_codes_matched_geo": len([c for c in base if base[c] >= 1 and c in nat]),
        "geo_codes": len(nat),
    }

    # per-code observed volume, the real n_c
    vols = [per_code[c]["units"] for c in have]
    res["volume"] = {
        "n_c_p25": quant(vols, .25), "n_c_p50": quant(vols, .5),
        "n_c_p75": quant(vols, .75), "n_c_max": max(vols) if vols else None,
        "mu_p10": round(quant([m for c in have for m in per_code[c]["mu"]], .10), 4),
        "mu_p50": round(quant([m for c in have for m in per_code[c]["mu"]], .50), 4),
        "mu_p90": round(quant([m for c in have for m in per_code[c]["mu"]], .90), 4),
        "mu_ge_2": None,
    }
    mus = [m for c in have for m in per_code[c]["mu"]]
    res["volume"]["mu_ge_2"] = round(sum(1 for m in mus if m >= 2) / float(len(mus)), 4) if mus else None
    # the share of observed lines sitting at a value >= 2, in the codes that matter
    res["volume"]["mu_eq_1"] = round(sum(1 for m in mus if abs(m - 1) < 5e-4) / float(len(mus)), 4) if mus else None

    # ---- per move --------------------------------------------------------
    verdicts = []
    tally = collections.Counter()
    for m in moves:
        lo, hi = m["from"] + 1, m["to"] + 1
        if lo > hi:
            lo, hi = hi, lo
        c = m["code"]
        pc = per_code.get(c)
        if not pc or not pc["mu"]:
            tally["unobserved"] += 1
            verdicts.append({"code": c, "q": m["q"], "from": m["from"], "to": m["to"],
                             "verdict": "unobserved"})
            continue
        # a group's own line count is at least its bene-days; with C lines of >= 1
        # unit, a line inside [lo,hi) exists only if S >= lo + C - 1
        can = False
        over = 0
        inside = 0
        for s, d in by_code[c]:
            C = max(1, d)
            if s >= lo + C - 1:
                can = True
            mu = s / float(C)
            if mu >= lo:
                over += 1
            if lo <= mu < hi:
                inside += 1
        if not can:
            v = "necessarily_empty"
        elif inside:
            v = "inside_on_the_mean"
        elif over:
            v = "forced_overtake"
        else:
            v = "undecided"
        tally[v] += 1
        verdicts.append({"code": c, "q": m["q"], "from": m["from"], "to": m["to"],
                         "groups": pc["groups"], "verdict": v,
                         "groups_inside": inside, "groups_over": over})

    res["moves"] = {"total": len(moves),
                    "up": sum(1 for m in moves if m["to"] > m["from"]),
                    "down": sum(1 for m in moves if m["to"] < m["from"]),
                    "tally": dict(tally)}
    res["up_tally"] = dict(collections.Counter(
        v["verdict"] for v in verdicts if v["to"] > v["from"]))
    res["verdicts"] = verdicts

    # ---- the summary's size, on the observed population ------------------
    # cells are (code, value) pairs.  The old reconstruction took the row count as
    # the binding constraint; with the observed volume it is the value domain that
    # binds, because a code's real volume exceeds its cap.
    cells_bound = 0
    cells_lb = 0
    n_obs = 0
    for c, cap in base.items():
        if cap < 1:
            continue
        u = nat.get(c, (None,))[0]
        if u is None:
            cells_bound += cap                    # unobserved: the table's own bound
            continue
        n_obs += 1
        cells_bound += min(cap, u)
        # a code whose every observed group spends exactly one unit per line has a
        # single atom; any group above that forces at least a second one
        spread = any(s > d for s, d in by_code.get(c, ()))
        cells_lb += 2 if spread else 1
    total_units = sum(v[0] for v in nat.values())
    res["summary"] = {
        "cells_bound": cells_bound,
        "cells_lower_bound": cells_lb,
        "codes_with_volume": n_obs,
        "sum_cap_all": sum(v for v in base.values() if v >= 1),
        "column_bytes_4B": 4 * total_units,
        "exact_array_bytes": 2 * cells_bound,
        "counting_bloom_bytes": int(4 * (-cells_bound * __import__("math").log(0.01) / (__import__("math").log(2) ** 2)) / 8),
        "array_over_column": round(2 * cells_bound / float(4 * total_units), 6),
    }

    print(json.dumps({k: v for k, v in res.items() if k != "verdicts"}, indent=1))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
