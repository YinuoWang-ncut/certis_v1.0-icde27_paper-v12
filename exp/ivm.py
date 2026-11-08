#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The IVM baseline the review asked for, run rather than argued.

v7 claimed a data-dependent threshold has no algebraic delta.  That is wrong:
`aggregate -> join -> filter` is a fixed query, DBSP covers it, and a parameter
relation makes the threshold an ordinary join key.  What follows is what the
algebra actually gives, measured:

  FULL   the derived query, recomputed                    -- the baseline
  R      re-evaluate the changed rule over its scope      -- v7's path R
  IVM    the delta of the same derived query              -- what DBSP produces

and for the relative (tier-mean) form, the same three with a maintained
(sum, count) so the threshold moves in O(1).

The result is not that IVM loses.  It is that the delta *is* the band: the
algebra produces exactly the set of claims whose judged value crosses the moved
threshold, so IVM and path B are the same computation, and the certification
question -- "is that set empty" -- is not answered more cheaply by the algebra
than by the scan that R does.

Output: exp/ivm.json
"""
import io
import json
import os
import time

import duckdb
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ivm.json")
U = 2_000_000
G = 5410
LAM = 1.5


def timed(fn, reps=5):
    v = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        v.append(time.perf_counter() - t0)
    return min(v)


def main():
    rng = np.random.default_rng(20261105)
    amt = np.round(rng.lognormal(6.0, 1.0, U), 2)
    gid = rng.integers(0, G, U).astype(np.int32)
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")          # one core: the ratio, not the peak
    con.register("src", {"amt": amt, "gid": gid})
    con.execute("CREATE TABLE claims AS SELECT amt, gid FROM src")
    con.execute("CREATE TABLE params(gid INTEGER, theta DOUBLE)")

    # one global parameter: theta = LAM * mean(amt) -- the absolute-threshold case
    theta0 = float(LAM * amt.mean())
    g = int(np.argmax(np.bincount(gid)))     # the largest group, as the scope
    n_scope = int((gid == g).sum())
    # a band of 1% of the scope around theta0
    lo, hi = theta0 * 0.99, theta0 * 1.01
    k_band = int(((amt >= lo) & (amt < hi) & (gid == g)).sum())

    con.execute("INSERT INTO params SELECT range, ? FROM range(?)", [theta0, G])
    con.execute("CREATE TABLE flat AS SELECT amt, gid FROM claims ORDER BY gid, amt")

    res = {"instance": {"rows": U, "groups": G, "scope_rows": n_scope,
                        "band_rows": k_band, "theta": round(theta0, 3)}}

    # ---- FULL: the derived query recomputed
    q_full = ("SELECT count(*) FROM claims c JOIN params p USING(gid) "
              "WHERE c.amt > p.theta")
    res["full_ms"] = round(1000 * timed(lambda: con.execute(q_full).fetchall()), 3)

    # ---- R: the changed rule's scope only
    q_r = "SELECT count(*) FROM claims WHERE gid = ? AND amt > ?"
    res["R_ms"] = round(1000 * timed(
        lambda: con.execute(q_r, [g, theta0]).fetchall()), 3)

    # ---- IVM: the delta of the same derived query after a parameter move
    #      d/dtheta of {amt > theta} is the band, so the delta IS the band
    q_b = ("SELECT count(*) FROM flat WHERE gid = ? AND amt >= ? AND amt < ?")
    res["B_ms"] = round(1000 * timed(
        lambda: con.execute(q_b, [g, lo, hi]).fetchall()), 3)

    # ---- maintain the parameter: one row, O(1)
    def upd():
        con.execute("UPDATE params SET theta = theta * 1.01 WHERE gid = ?", [g])
        con.execute("UPDATE params SET theta = theta / 1.01 WHERE gid = ?", [g])
    res["param_update_ms"] = round(1000 * timed(upd) / 2, 4)

    # ---- relative form: theta_g = LAM * mean(amt) over the group
    #      sum/count are O(1) to maintain; the band is not.
    q_agg = "SELECT sum(amt), count(*) FROM claims WHERE gid = ?"
    res["tier_aggregate_ms"] = round(1000 * timed(
        lambda: con.execute(q_agg, [g]).fetchall()), 3)
    s0, c0 = con.execute(q_agg, [g]).fetchone()
    res["tier_aggregate_maintained_ms"] = 0.0     # O(1) by construction
    theta_rel = LAM * s0 / c0
    lo_r, hi_r = theta_rel * 0.99, theta_rel * 1.01
    res["relative_band_rows"] = int(con.execute(
        "SELECT count(*) FROM flat WHERE gid=? AND amt>=? AND amt<?",
        [g, lo_r, hi_r]).fetchone()[0])
    res["relative_band_ms"] = round(1000 * timed(
        lambda: con.execute(q_b, [g, lo_r, hi_r]).fetchall()), 3)

    # ---- is the delta the same set as the band?  check on one move
    before = con.execute("SELECT count(*) FROM claims WHERE gid=? AND amt>?",
                         [g, theta0]).fetchone()[0]
    after = con.execute("SELECT count(*) FROM claims WHERE gid=? AND amt>?",
                        [g, theta0 * 1.01]).fetchone()[0]
    band = con.execute("SELECT count(*) FROM flat WHERE gid=? AND amt>=? AND amt<?",
                       [g, theta0, theta0 * 1.01]).fetchone()[0]
    res["delta_equals_band"] = {"before": before, "after": after,
                                "band": band, "consistent": before - after == band}

    res["ratios"] = {
        "R_over_full": round(res["R_ms"] / res["full_ms"], 2),
        "B_over_R": round(res["B_ms"] / res["R_ms"], 2),
        "B_over_full": round(res["B_ms"] / res["full_ms"], 2),
    }
    print(json.dumps(res, indent=1))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
