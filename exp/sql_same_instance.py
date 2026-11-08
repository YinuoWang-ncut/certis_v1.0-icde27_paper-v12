#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two SQL engines, one instance: the matched-instance baseline.

What was wrong before
---------------------
Table VII compares CERTIS against three embedded engines, and every one of those
ratios crosses a denominator: the engine runs on its own instance at its own
scale and CERTIS runs on the replay, so the ratio prices the denominator as much
as the access path.  The paper says so and calls the comparison a reference.  A
reviewer is entitled to the obvious question -- on *one* instance, the same rows,
the same code, the same scope length, the same band, does an independent engine
still see what the paper sees? -- and until this file existed there was no
answer.

The logical query is the paper's own candidate list:

    code = c AND min(t0,t1) < unit <= max(t0,t1)

What differs between cells is the physical design and the engine, never the rows.
Two designs each:

  clustered   stored in (code, unit) order -- what a claims warehouse looks like
              after it has been laid out for this workload, and the premise of
              path R (the scope is one contiguous read)
  insertion   insertion order preserved -- what a claims table looks like before
              anyone laid it out, and where an index on (code, unit) has to pay
              for random row fetches

Two engines, because they answer different halves of the question:

  duckdb    ships an ART the planner may consider for a range predicate.  It does
            not consider it: the plan is a sequential scan whose row-group zone
            maps skip the other codes' groups ("Row Groups Scanned: 1/2").  So
            DuckDB's best plan for this query *is* path R, and the engine never
            builds path B at all.
  sqlite    will use a B-tree index for a range predicate, so it is the one engine
            here that can actually execute path B.  Its two plans are the paper's
            two paths, on the paper's instance.

Both are timed twice: the access path (an aggregation, so nothing has to be
materialised) and the candidate list (the ids, as CERTIS produces them).  The
first isolates the path; the second adds the constant both paths pay in Python.
The paper's claim is about the path, so the first is what is quoted.

  python sql_same_instance.py --reps 25
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, ".."))

import masked_pipeline as MP                           # noqa: E402

PLANS = [("median", 44606, 0.183), ("q3", 412875, 0.183)]
Q_IDS = "SELECT lid FROM {t} WHERE code = ? AND unit > ? AND unit <= ?"
Q_CNT = "SELECT count(*) FROM {t} WHERE code = ? AND unit > ? AND unit <= ?"
# SQLite's three cells.  A B-tree on (code, unit) is chosen for every predicate
# that mentions both columns, so "the scope" has to be said the way a clustered
# store says it: a contiguous rowid range, which is what path R reads.
Q_SQ_R = ("SELECT lid FROM lit WHERE rowid BETWEEN ? AND ? "
          "AND unit > ? AND unit <= ?")
Q_SQ_RC = ("SELECT count(*) FROM lit WHERE rowid BETWEEN ? AND ? "
           "AND unit > ? AND unit <= ?")
Q_SQ_SCAN = ("SELECT lid FROM lit_in NOT INDEXED "
             "WHERE code = ? AND unit > ? AND unit <= ?")
Q_SQ_SCANC = ("SELECT count(*) FROM lit_in NOT INDEXED "
              "WHERE code = ? AND unit > ? AND unit <= ?")
# The instrument that matters: the engine materialises the candidate list
# itself.  It pays the row fetch (which a covering-index count skips) and the
# write of k identifiers (which CERTIS also pays), but not the conversion of k
# rows into Python tuples (which CERTIS does not pay), so the two engines are
# comparable to each other and to CERTIS's own arms.
Q_MAT_SQ = "CREATE TEMP TABLE r_{i} AS {q}"
Q_MAT_DK = "CREATE OR REPLACE TEMP TABLE r_{i} AS {q}"


def roundrobin(fns, reps):
    """One block, so a plan cannot inherit a warm store the others miss."""
    for f in fns.values():
        f()
    out = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            out[k].append((time.perf_counter() - t0) * 1e6)
    return {k: float(np.median(v)) for k, v in out.items()}


def duckdb_plan(con, q, params):
    txt = con.execute("EXPLAIN " + q, params).fetchall()[0][-1]
    up = txt.upper()
    kind = "INDEX" if ("INDEX_SCAN" in up or "INDEX SCAN" in up) else \
        ("SEQ" if "SEQ" in up else "?")
    ana = con.execute("EXPLAIN ANALYZE " + q, params).fetchall()[0][-1]
    rg = re.search(r"Row Groups Scanned:\s*([0-9]+)\s*/\s*([0-9]+)", ana)
    tot = re.search(r"Total Time:\s*([0-9.]+)\s*s", ana)
    return dict(kind=kind,
                row_groups=(int(rg.group(1)), int(rg.group(2))) if rg else None,
                total_us=float(tot.group(1)) * 1e6 if tot else None)


def _certis_ratio(tag):
    """The pipeline's own number for the same cell, so the two sit side by side."""
    p = os.path.join(OUT, "masked_pipeline.json")
    if not os.path.exists(p):
        return float("nan")
    d = json.load(open(p))
    for r in d["rows"]:
        if r["tag"] == tag and abs(r["dens"] - 0.183) < 0.01:
            return r["ratio"]
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(OUT, "sql_same_instance.json"))
    a = ap.parse_args()

    import duckdb
    import pyarrow as pa
    t_load = time.time()
    store, sigma, _ = MP.build([(t, n, d) for t, n, d in PLANS])
    n_row = store.n_l
    print("instance: %d rows, %d codes, two published scopes"
          % (n_row, len(PLANS)), flush=True)

    # ---------------------------------------------------------------- duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")          # CERTIS's paths are single-threaded
    tbl = pa.table({"lid": np.arange(n_row, dtype=np.int64),
                    "code": store.code.astype(np.int32),
                    "unit": store.unit.astype(np.int32),
                    "gid": store.gid.astype(np.int32)})
    con.register("raw", tbl)
    con.execute("CREATE TABLE duck_cl AS SELECT * FROM raw ORDER BY code, unit")
    con.execute("CREATE TABLE duck_in AS SELECT * FROM raw")
    con.execute("CREATE INDEX di ON duck_cl(code, unit)")
    con.execute("CREATE INDEX di2 ON duck_in(code, unit)")

    # ---------------------------------------------------------------- sqlite
    db = os.path.join(OUT, "same_instance.sqlite")
    if os.path.exists(db):
        os.remove(db)
    sq = sqlite3.connect(db)
    sq.execute("PRAGMA journal_mode=OFF")
    sq.execute("PRAGMA synchronous=OFF")
    # 64 MB page cache.  The default is 2 MB, which for a 457k-row table makes
    # the index-then-fetch cell an exercise in page thrashing rather than in
    # access-path choice; a deployment on this data would not leave it there.
    sq.execute("PRAGMA cache_size=-65536")
    # No PRIMARY KEY on lid: `lid INTEGER PRIMARY KEY` would alias rowid, the
    # insertion order would stop being the physical order, and the scope would
    # not be a contiguous rowid range.  This is the whole point of the cell.
    sq.execute("CREATE TABLE lit(lid INT, code INT, unit INT)")
    sq.execute("CREATE TABLE lit_in(lid INT, code INT, unit INT)")
    order = np.argsort(store.code.astype(np.int64) * 32 + store.unit, kind="stable")
    sq.executemany("INSERT INTO lit VALUES (?,?,?)",
                   ((int(i), int(store.code[i]), int(store.unit[i])) for i in order))
    sq.executemany("INSERT INTO lit_in VALUES (?,?,?)",
                   ((int(i), int(store.code[i]), int(store.unit[i]))
                    for i in range(n_row)))
    sq.execute("CREATE INDEX si ON lit(code, unit)")
    sq.execute("CREATE INDEX si2 ON lit_in(code, unit)")
    sq.commit()
    print("loaded both engines in %.0fs" % (time.time() - t_load), flush=True)

    rows = []
    for code, (tag, n, d) in enumerate(PLANS):
        cand = store.band_u(code, MP.CAP_NEW, MP.CAP_OLD)
        k = int(len(cand))
        n_scope = int(store.scope_len(code))
        p = [code, MP.CAP_NEW, MP.CAP_OLD]
        truth = sorted(int(x) for x in cand)

        # ---- duckdb: the plan is read off EXPLAIN, the time is wall clock in
        # the same round-robin as sqlite, so neither engine gets a different clock
        d_plan = {}
        for name in ("duck_cl", "duck_in"):
            assert sorted(r[0] for r in con.execute(
                Q_IDS.format(t=name), p).fetchall()) == truth
            d_plan[name] = duckdb_plan(con, Q_IDS.format(t=name), p)

        # ---- sqlite: path R (rowid range), path B (B-tree band), and the
        # wrong baseline (full scan), all on the same rows
        lo, hi = int(store.cptr[code]) + 1, int(store.cptr[code + 1])
        pR = [lo, hi, MP.CAP_NEW, MP.CAP_OLD]
        sq_cells = {
            "R": (Q_SQ_R, Q_SQ_RC, pR),
            "B": (Q_IDS.format(t="lit_in"), Q_CNT.format(t="lit_in"), p),
            "scan": (Q_SQ_SCAN, Q_SQ_SCANC, p),
        }
        s_plan = {}
        for name, (qi, qc, par) in sq_cells.items():
            s_plan[name] = " | ".join(
                r[3] for r in sq.execute("EXPLAIN QUERY PLAN " + qi,
                                         par).fetchall())
            assert sorted(r[0] for r in sq.execute(qi, par).fetchall()) == truth, name
        fns = {}
        for t in ("duck_cl", "duck_in"):
            fns[t + "_cnt"] = (lambda t=t: con.execute(Q_CNT.format(t=t), p)
                               .fetchall())
            fns[t + "_mat"] = (lambda t=t: con.execute(
                "CREATE OR REPLACE TEMP TABLE rd AS " + Q_IDS.format(t=t),
                p).fetchall())
        for name, (qi, qc, par) in sq_cells.items():
            fns[name + "_cnt"] = (lambda qc=qc, par=par: sq.execute(qc, par)
                                  .fetchall())
            fns[name + "_mat"] = (lambda name=name, qi=qi, par=par:
                                  (sq.execute("DROP TABLE IF EXISTS rs"),
                                   sq.execute("CREATE TEMP TABLE rs AS " + qi,
                                              par), sq.execute("DROP TABLE rs")))
        s_time = roundrobin(fns, a.reps)
        d_eng = {k2: v for k2, v in s_time.items() if k2.startswith("duck")}

        # SQLite's materialised cells must carry the same rows, checked off-clock
        for name, (qi, qc, par) in sq_cells.items():
            sq.execute("CREATE TEMP TABLE chk AS " + qi, par)
            got = sorted(r[0] for r in sq.execute("SELECT lid FROM chk")
                         .fetchall())
            sq.execute("DROP TABLE chk")
            assert got == truth, ("mat", name, len(got), k)

        r = dict(tag=tag, code=code, n=n_scope, k=k, dens=k / max(n_scope, 1),
                 duck_plan=d_plan, duck_engine_us=d_eng,
                 sqlite_plan=s_plan, sqlite_us=s_time,
                 sq_ratio_B_over_R=s_time["B_cnt"] / s_time["R_cnt"],
                 sq_ratio_B_over_R_mat=s_time["B_mat"] / s_time["R_mat"],
                 sq_ratio_B_over_scan=s_time["B_cnt"] / s_time["scan_cnt"],
                 certis_ratio=_certis_ratio(tag))
        rows.append(r)

        print("\n%s: n %d, k %d, d %.3f" % (tag, n_scope, k, r["dens"]))
        for name, label in (("duck_cl", "clustered"), ("duck_in", "insertion")):
            print("  duckdb %-9s %-4s groups %-6s  count %7.0f us   "
                  "materialise %8.0f us"
                  % (label, d_plan[name]["kind"], d_plan[name]["row_groups"],
                     d_eng[name + "_cnt"], d_eng[name + "_mat"]))
        for name, label in (("R", "scope (rowid)"), ("B", "band (B-tree)"),
                            ("scan", "full scan")):
            print("  sqlite %-14s  count %7.0f us   materialise %8.0f us"
                  % (label, s_time[name + "_cnt"], s_time[name + "_mat"]))
        for name in ("R", "B", "scan"):
            print("          %-5s plan: %s" % (name, s_plan[name][:84]))
        print("  sqlite  band/scope  %.3f (count)  %.3f (materialised)"
              % (s_time["B_cnt"] / s_time["R_cnt"],
                 s_time["B_mat"] / s_time["R_mat"]))
        print("  CERTIS  band/scope  %.3f  (same instance, same cell)"
              % r["certis_ratio"])

    res = dict(reps=a.reps, instance_rows=n_row, duckdb=duckdb.__version__,
               sqlite=sqlite3.sqlite_version, rows=rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\nwrote %s" % a.out)

    # The SQLite file is 21 MB and is rebuilt from `rows` on every run.  Leaving
    # it behind makes the release tree 21 MB heavier than the archive it is
    # documented as, and it was doing exactly that.
    sq.close()
    if os.path.exists(db):
        os.remove(db)
        print("removed the %s scratch database" % os.path.basename(db))


if __name__ == "__main__":
    main()
