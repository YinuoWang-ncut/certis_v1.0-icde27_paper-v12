#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What does it cost to learn k = |B|?

The band's endpoints are closed form.  Its cardinality is not (T4 in
tests/test_theorems.py).  So a selector that compares k/n against a constant
must first *obtain* k, and the cheapest way to obtain it exactly is a rank probe:
two binary searches in each affected group's sorted array.

This script prices that probe against the per-row cost both paths already pay,
so the probe can be expressed as a number of rows -- which is the only form in
which it can be added to the floor of the decision.

It deliberately reports a ratio, not an absolute latency: the absolute number is
a property of this interpreter and this machine, the ratio is a property of the
memory hierarchy (a binary search touches log2(n_group) cache lines; a scan
touches n_group/16).

CAUTION on the per-group figure.  `rank_probe_ns_per_group` times two numpy calls
per group, and at 1,024 values the call overhead dominates the search: it reports
3,851 ns where the same two searches in compiled code take 142 ns.  The paper
therefore quotes the compiled number and attributes it to exp/java/CountOracle.java
(structure A), not to this script.  What this script is still the authority for is
`fetch_over_scan`, which is a ratio between two numpy paths and so cancels the
overhead, and `scan_ns_per_row`, which is amortised over a whole pass.

Output: exp/rank_probe.json
"""
import io
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rank_probe.json")


def _best(fn, reps=15):
    v = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        v.append(time.perf_counter_ns() - t0)
    return min(v)


def main():
    rng = np.random.default_rng(11)
    per_group = 1024
    n_groups = 5410
    N = per_group * n_groups
    val = rng.integers(0, 10 ** 6, N).astype(np.int64)
    val.sort()
    offs = np.arange(0, N + 1, per_group, dtype=np.int64)

    lo, hi = 400000, 500000

    # one group's rank probe, hoisted out of the loop so the loop overhead is
    # measured separately and subtracted
    def probe_empty():
        s = 0
        for g in range(n_groups):
            s += 1
        return s

    def probe_rank():
        s = 0
        for g in range(n_groups):
            seg = val[offs[g]:offs[g + 1]]
            s += np.searchsorted(seg, hi) - np.searchsorted(seg, lo)
        return s

    t_loop = _best(probe_empty)
    t_rank = _best(probe_rank)
    per_group_ns = (t_rank - t_loop) / n_groups

    # a scan row: contiguous read + compare, amortised over a whole pass
    nrow = 1 << 20
    t_scan = _best(lambda: np.count_nonzero(val[:nrow] >= lo))
    per_row_ns = t_scan / nrow

    # a scattered fetch of one candidate + the same compare
    idx = rng.integers(0, N, 4096)
    t_fetch = _best(lambda: np.count_nonzero(val.take(idx) >= lo))
    per_cand_ns = (t_fetch - 0) / idx.size

    res = {
        "groups": n_groups,
        "rows_per_group": per_group,
        "rank_probe_ns_per_group": round(float(per_group_ns), 1),
        "scan_ns_per_row": round(float(per_row_ns), 4),
        "scattered_fetch_ns_per_candidate": round(float(per_cand_ns), 3),
        "probe_in_rows_per_group": round(float(per_group_ns / per_row_ns), 1),
        "probe_in_rows_total": round(float(per_group_ns * n_groups / per_row_ns), 0),
        "fetch_over_scan": round(float(per_cand_ns / per_row_ns), 2),
    }
    print(json.dumps(res, indent=1))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
