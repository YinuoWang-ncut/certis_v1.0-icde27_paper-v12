#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A ceiling on band density computed from published aggregates, not from
synthetic claims.

The objection this answers
--------------------------
Table II is measured on generated claims, because per-claim units appear in no
public release.  If the whole comparison rests on that stream, a reader may say
the sign is an artefact of the generator.  It need not be: CMS publishes, for
every (NPI, HCPCS) pair, the units billed S and the beneficiary-days B, and an
MUE is a *per beneficiary-day* limit.  That makes the day the row, B the row
count, and the band a set of days whose units fall between the two caps.  What
is unpublished is only how the units are spread over the days -- and a bound
needs no spread.

The bound
---------
For one group, let m = S/B be its mean units per day, and U = max(theta0,
theta1) the band's upper edge.  Units are positive integers, so Markov gives
P(units >= U) <= m/U; every day in the band is below U, hence

    k_g <= B_g * min(1, m_g / U),        k/N <= sum_g that / sum_g B_g.

That is an upper bound from two published columns and no assumption about the
spread.  It can only overstate the band, so a density ceiling below the value
at which the criterion would admit retrieval is a statement about real data
that the synthetic stream cannot overturn.

What this is not
----------------
It is a bound, not a measurement: where the mean sits far above U the bound is
1 and says nothing.  The output therefore reports how many moves it binds and
how many it leaves open, and the paper must quote both.
"""
import collections
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.normpath(os.path.join(HERE, "..", "p3"))
CMS = os.path.normpath(os.path.join(HERE, "..", "data", "cms"))

OUT = os.path.join(HERE, "out")


def moves():
    """Every in-regime threshold move with the year its data must come from."""
    d = json.load(open(os.path.join(P3, "certificate_6y.json"), encoding="utf-8"))
    out = []
    for y in d["years"]:
        for m in y.get("detail", []):
            if m.get("verdict") == "unobserved":
                continue
            out.append(dict(code=m["code"], lo=min(m["from"], m["to"]),
                            hi=max(m["from"], m["to"]), year=y["year"],
                            quarter=m.get("q"), verdict=m["verdict"]))
    return out


def groups(year, keep):
    """(code, NPI) -> [S, B] for one year, restricted to the codes we need."""
    g = collections.defaultdict(lambda: [0, 0])
    path = os.path.join(CMS, "mup_phy_%d_prov_svc.csv" % year)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            c = row["HCPCS_Cd"]
            if c not in keep:
                continue
            try:
                s = int(row["Tot_Srvcs"])
                b = int(row["Tot_Bene_Day_Srvcs"])
            except (TypeError, ValueError):
                continue
            a = g[(c, row["Rndrng_NPI"])]
            a[0] += s
            a[1] += b
    return g


def main():
    mv = moves()
    print("in-regime moves with a year: %d over %d codes"
          % (len(mv), len({m["code"] for m in mv})))
    by_year = collections.defaultdict(list)
    for m in mv:
        by_year[m["year"]].append(m)

    rows = []
    for year, ms in sorted(by_year.items()):
        keep = {m["code"] for m in ms}
        g = groups(year, keep)
        print("  %d: %d groups over %d codes" % (year, len(g), len(keep)))
        for m in ms:
            gs = [(s, b) for (c, _), (s, b) in g.items() if c == m["code"] and b > 0]
            if not gs:
                rows.append(dict(m, status="no data"))
                continue
            U = m["hi"]
            N = sum(b for _, b in gs)
            ks = [b * min(1.0, s / (b * U)) if U else float(b) for s, b in gs]
            K = sum(ks)
            rows.append(dict(m, status="bounded", N=int(N), K_ub=float(K),
                             dens_ub=float(K / N), ngroups=len(gs),
                             S_total=int(sum(s for s, _ in gs))))

    b = [r for r in rows if r.get("status") == "bounded"]
    n = [r for r in rows if r.get("status") == "no data"]
    print("\nbounded %d, no data %d" % (len(b), len(n)))
    d = np.array([r["dens_ub"] for r in b])
    N = np.array([r["N"] for r in b])
    print("density ceiling k/N over real groups:")
    for q in (10, 25, 50, 75, 90, 100):
        print("   p%-4d %.4f" % (q, np.percentile(d, q)))
    print("moves whose ceiling is below the paper's corpus density 0.46: %d/%d"
          % (int(np.sum(d < 0.46)), len(d)))
    print("moves whose ceiling binds (below 1): %d/%d"
          % (int(np.sum(d < 0.999)), len(d)))
    print("\nreal row counts N (beneficiary-days per code), the scope of a "
          "per-day rule:")
    for q in (5, 25, 50, 75, 95):
        print("   p%-4d %10.0f" % (q, np.percentile(N, q)))
    print("codes with N < 512: %d/%d ; N < 394: %d/%d"
          % (int(np.sum(N < 512)), len(N), int(np.sum(N < 394)), len(N)))

    # the aggregate ceiling over the whole in-regime set, weighted by rows
    K = sum(r["K_ub"] for r in b)
    Ntot = sum(r["N"] for r in b)
    print("\nweighted overall ceiling: %.4f  (sum K_ub %.3g / sum N %.3g)"
          % (K / Ntot, K, Ntot))

    res = dict(n_moves=len(rows), n_bounded=len(b), n_nodata=len(n),
               density_ceiling=dict(
                   p10=float(np.percentile(d, 10)),
                   p25=float(np.percentile(d, 25)),
                   p50=float(np.percentile(d, 50)),
                   p75=float(np.percentile(d, 75)),
                   p90=float(np.percentile(d, 90)),
                   max=float(d.max()),
                   n_below_corpus=int(np.sum(d < 0.46)),
                   n_binding=int(np.sum(d < 0.999))),
               rows_N=dict(p5=float(np.percentile(N, 5)),
                           p50=float(np.percentile(N, 50)),
                           p95=float(np.percentile(N, 95)),
                           n_lt_512=int(np.sum(N < 512)),
                           n_lt_394=int(np.sum(N < 394))),
               weighted_ceiling=float(K / Ntot), rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "realband.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote out/realband.json")


if __name__ == "__main__":
    main()
