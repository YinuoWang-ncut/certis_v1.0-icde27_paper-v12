#!/usr/bin/env python
r"""Section V-H: the two cheaper certification structures, built and priced.

v8 said they "remain unbuilt: the largest remaining gap".  Both are built here
and priced in the currency the boundary already uses -- probes and row touches.

  OCCUPANCY SUMMARY   a no-false-negative set over (scope, value) cells, queried
                      with H hashes per cell spanned by the interval.  It can
                      prove a band empty, never locate a member.
  PARTITION SUMMARY   per-sub-partition min/max/count (a zone map).  It prunes
                      sub-partitions by range and then scans what survives.

The claim population is reconstructed, not invented.  An MUE cap is by
definition the maximum over six months of submitted claims (NCCI Policy Manual
Ch.1 Sec.V), so the only population consistent with a published table is one
whose per-code maximum *is* the published cap.  We anchor on the earliest table
and hold that population fixed while applying every later published cap move --
which is the certifier's own question: the rule moves, the data does not.  The
sub-maximum shape is Zipf over price points, the shape Table 6 already measures.

The reject rate that follows is a property of the anchoring -- an increase
cannot flip a claim above the population maximum -- and is reported as such.
The cost of a rejection is not: it is set by the atom count, which the cap table
publishes.

Output: exp/cheap.json
"""
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from delta import mue_table                       # noqa: E402

DATA = os.environ.get("CERTIS_DATA") or os.path.join(
    os.path.dirname(HERE), "data")          # artifact layout; set CERTIS_DATA otherwise
if not os.path.isdir(DATA):
    raise SystemExit("no data tree at %s -- set CERTIS_DATA or run data/fetch.py" % DATA)
MUE = os.path.join(DATA, "mue")
OUT = os.path.join(HERE, "cheap.json")

Q = ["2025q1", "2025q4", "2026q1", "2026q2", "2026q3", "2026q4"]
ZIPF_A = 1.2          # sub-maximum mass, price points
H = 4                 # summary hashes per cell
EPS = 0.01            # target false-positive rate
PARTS = (8, 16, 32, 64, 128)   # zone-map granularity sweep


# ---------------------------------------------------------------- population
def load():
    caps = {}
    for q in Q:
        t = mue_table(os.path.join(MUE, "mue_%s_pra.zip" % q))
        if t:
            caps[q] = t
    return caps


def population(caps, rng):
    codes = sorted(c for c, m in caps.items() if m >= 1)
    cap = np.array([caps[c] for c in codes], dtype=np.int64)
    n = np.clip(np.round(rng.lognormal(6.0, 0.9, len(codes))), 2, 20000).astype(np.int64)
    units, owner = [], []
    for m, k in zip(cap, n):
        if m <= 1:
            u = np.ones(k, dtype=np.int32)
        else:
            u = np.empty(k, dtype=np.int32)
            fill = 0
            for _ in range(60):
                if fill >= k:
                    break
                draw = rng.zipf(ZIPF_A, size=(k - fill) * 2)
                draw = draw[draw <= m]
                take = min(k - fill, draw.size)
                if take:
                    u[fill:fill + take] = draw[:take]
                    fill += take
            u[fill:] = 1
            u[-1] = m                      # the cap is attained, by definition
        units.append(u)
        owner.append(np.full(k, 0, dtype=np.int32))
    units = np.concatenate(units)
    owner = np.repeat(np.arange(len(codes), dtype=np.int32), n)
    return codes, cap, units, owner, n


# --------------------------------------------------------------------- bloom
M64 = (1 << 64) - 1


def mix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


class Bloom:
    def __init__(self, cells, eps):
        self.m = int(max(64, -cells * math.log(eps) / (math.log(2) ** 2)))
        self.h = H
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)

    def _pos(self, key):
        a = mix(key)
        b = mix(key ^ 0xD1B54A32D192ED03) | 1
        return [(a + i * b) % self.m for i in range(self.h)]

    def add(self, key):
        for p in self._pos(key):
            self.bits[p >> 3] |= np.uint8(1 << (p & 7))

    def has(self, key):
        for p in self._pos(key):
            if not (self.bits[p >> 3] >> (p & 7)) & 1:
                return False
        return True

    def key(self, i, u):
        return (int(i) << 40) | int(u)


def main():
    rng = np.random.default_rng(20261104)
    caps = load()
    bases = caps[Q[0]]
    codes, cap, units, owner, n = population(bases, rng)
    nl = int(units.size)

    # ---- the summary's size, bounded from the published table alone -----
    res = {"bound_from_table": {
        "codes": len(codes),
        "sum_cap": int(cap.sum()),
        "sum_min_cap_rows": int(sum(min(int(c), int(k)) for c, k in zip(cap, n))),
        "sum_cap_bytes_at_10bit": int(cap.sum() * 10 / 8),
    }}

    cells = len(set(zip(owner.tolist(), units.tolist())))
    bf = Bloom(cells, EPS)
    for i, u in zip(owner.tolist(), units.tolist()):
        bf.add(bf.key(i, u))
    res["population"] = {
        "anchor_table": Q[0], "codes": len(codes), "rows": nl,
        "distinct_cells": cells, "cells_per_row": round(cells / nl, 4),
        "zipf_shape": ZIPF_A,
        "summary_bytes": int(bf.bits.size), "summary_bits_per_row": round(8 * bf.bits.size / nl, 3),
        "summary_bits_per_cell": round(8 * bf.bits.size / cells, 2),
        "hashes": H, "eps": EPS,
        # an exact counter array, which a plain Bloom cannot give: deletions need
        # counts, and a counting Bloom costs 4x the bits, i.e. more than the array.
        "exact_bytes_2B": 2 * cells,
        "counting_bloom_bytes": 4 * int(bf.bits.size),
        "exact_over_bloom": round(2 * cells / bf.bits.size, 2),
    }

    # ---- per-move measurement -------------------------------------------
    order = np.argsort(owner, kind="stable")
    owner_s, units_s = owner[order], units[order]
    start = np.searchsorted(owner_s, np.arange(len(codes)), side="left")
    stop = np.searchsorted(owner_s, np.arange(len(codes)), side="right")
    idx = {c: i for i, c in enumerate(codes)}

    mv = []
    for q in Q[1:]:
        t = caps[q]
        for c, m0 in bases.items():
            if c not in t or t[c] == m0 or c not in idx:
                continue
            mv.append((q, c, m0, t[c], idx[c]))

    rows = []
    for q, c, m0, m1, i in mv:
        lo, hi = (m0 + 1, m1 + 1) if m1 > m0 else (m1 + 1, m0 + 1)
        u = units_s[start[i]:stop[i]]
        band = int(np.count_nonzero((u >= lo) & (u < hi)))
        sig = int(u.size)
        # occupancy summary: H probes per cell spanned, no false negatives
        absent = not any(bf.has(bf.key(i, x)) for x in range(lo, hi))
        rec = {"quarter": q, "code": c, "from": m0, "to": m1,
               "w": hi - lo, "band": band, "scope": sig,
               "summary_absent": bool(absent),
               "summary_probes": H * (hi - lo),
               "zonemap": {}}
        # zone map at each granularity
        for P in PARTS:
            part = (np.arange(sig, dtype=np.int64) % P)
            pmax = np.full(P, -1, dtype=np.int64)
            pmin = np.full(P, 1 << 30, dtype=np.int64)
            np.maximum.at(pmax, part, u)
            np.minimum.at(pmin, part, u)
            surv = (pmax >= lo) & (pmin < hi)
            rec["zonemap"][P] = {
                "survivors": int(np.count_nonzero(surv)),
                "rows": int(np.count_nonzero(surv[part])),
            }
        rows.append(rec)
        if absent and band:            # soundness: never a false negative
            raise SystemExit("false negative on %s %s" % (c, q))

    band = np.array([r["band"] for r in rows])
    scope = np.array([r["scope"] for r in rows])
    w = np.array([r["w"] for r in rows])
    probes = np.array([r["summary_probes"] for r in rows])
    absent = np.array([r["summary_absent"] for r in rows])

    res["workload"] = {
        "moves": len(rows),
        "moves_up": sum(1 for r in rows if r["to"] > r["from"]),
        "moves_down": sum(1 for r in rows if r["to"] < r["from"]),
        "band_zero": int(np.count_nonzero(band == 0)),
        "band_zero_frac": round(float(np.mean(band == 0)), 4),
        "summary_absent": int(np.count_nonzero(absent)),
        "summary_false_positive": int(np.count_nonzero(absent & (band > 0))),
        "summary_fp_on_nonempty": round(float(np.count_nonzero(~absent & (band == 0)) /
                                              max(1, np.count_nonzero(band == 0))), 4),
        "exact_certified": int(np.count_nonzero(band == 0)),
        "summary_expensive_moves": int(np.count_nonzero(probes * 25.6e-9 >
                                                        scope * 11.3e-9)),
        "missed_by_width": sorted(int(x) for x in w[~absent & (band == 0)]),
        "median_band": int(np.median(band)), "median_scope": int(np.median(scope)),
        "median_w": int(np.median(w)), "max_w": int(w.max()),
        "median_probes": int(np.median(probes)), "max_probes": int(probes.max()),
    }

    # ---- zone map: pruning against granularity ---------------------------
    zm = {}
    for P in PARTS:
        over = np.array([r["zonemap"][P]["rows"] / max(r["scope"], 1) for r in rows])
        surv = np.array([r["zonemap"][P]["survivors"] for r in rows])
        nz = band > 0
        zm[P] = {
            "rows_per_partition": round(nl / (P * len(codes)), 2),
            "metadata_bytes": 8 * P * len(codes),
            "metadata_over_column": round(8 * P * len(codes) / (4 * nl), 4),
            "rows_over_scope_median": round(float(np.median(over)), 4),
            "rows_over_scope_nonempty": round(float(np.median(over[nz])), 4) if nz.any() else 0.0,
            "survivors_of_P_median": round(float(np.median(surv)), 1),
            "probes": P,
        }
    res["zonemap"] = zm

    # ---- the decision, in the boundary's currency -----------------------
    A_SIG, A_P = 11.3e-9, 25.6e-9
    res["decision"] = {
        "a_sigma": A_SIG, "a_p": A_P,
        "summary_us_median": round(float(np.median(probes)) * A_P * 1e6, 3),
        "summary_us_max": round(float(probes.max()) * A_P * 1e6, 2),
        "recompute_us_median": round(float(np.median(scope)) * A_SIG * 1e6, 3),
        "recompute_us_p90": round(float(np.percentile(scope, 90)) * A_SIG * 1e6, 2),
        "band_us_median": round(float(np.median(scope)) * (A_SIG + A_P) * 1e6, 3),
        "zonemap_us_at_64": round(64 * A_P * 1e6, 3),
        "breakeven_absent_frac": round(float(np.median(probes) * A_P /
                                            (np.median(scope) * A_SIG)), 4),
        # the paper's own median single-rule scope, for a like-for-like ratio
        "at_paper_scope_394_us": round(394 * A_SIG * 1e6, 2),
        "ratio_at_paper_scope": round(394 * A_SIG / (12 * A_P), 1),
    }

    print(json.dumps(res, indent=1))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
