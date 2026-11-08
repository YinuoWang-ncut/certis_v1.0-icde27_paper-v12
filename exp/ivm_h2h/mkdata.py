#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the exact instance of exp/ivm.py and dump it for the Rust harness.

The point of dumping rather than re-sampling in Rust: the differential-dataflow
arm and the DuckDB arm must see the same 2e6 rows, byte for byte, or the
head-to-head is between two instances and not between two engines.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U, G, LAM = 2_000_000, 5410, 1.5

rng = np.random.default_rng(20261105)
amt = np.round(rng.lognormal(6.0, 1.0, U), 2).astype(np.float64)
gid = rng.integers(0, G, U).astype(np.uint32)

rec = np.empty(U, dtype=np.dtype([("gid", "<u4"), ("amt", "<f8")]))
rec["gid"], rec["amt"] = gid, amt
rec.tofile(os.path.join(HERE, "instance.bin"))

theta0 = float(LAM * amt.mean())
gmax = int(np.argmax(np.bincount(gid)))
scope = int((gid == gmax).sum())

# same three placements exp/ivm.py prices, plus corpus-style multipliers
bands = {}
for m in (1.01, 1.02, 1.10, 1.50, 2.00):
    lo, hi = theta0 * (2 - m), theta0           # a move DOWN by (m-1)*theta
    bands[f"{m:.2f}"] = int(((amt >= lo) & (amt < hi) & (gid == gmax)).sum())
    lo2, hi2 = theta0, theta0 * m               # a move UP
    bands[f"up{m:.2f}"] = int(((amt >= lo2) & (amt < hi2) & (gid == gmax)).sum())

meta = dict(rows=U, groups=G, lam=LAM, theta=theta0, gmax=gmax, scope_rows=scope,
            mean=float(amt.mean()), bands=bands)
with open(os.path.join(HERE, "instance.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta, indent=1))
