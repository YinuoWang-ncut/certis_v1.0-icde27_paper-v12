#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One accounting for both execution paths.

The v7 draft mixed three different per-element costs: a bare gather (1.45 ns),
a gather charged with the two-state decision and the override filter (11.3 ns),
and a probe (28-30 ns), then divided one by the sum of two others to publish a
"5%" switch.  That division is what the review called out, and it is not a
rounding issue: the numerator and the denominator were charging different work.

This script measures all the components in one process, on one instance, with
one definition of "work done", and fits the two lines

    T_R(n) = F_R + c_R * n
    T_B(k) = F_B + c_B * k            (+ a rank probe per group)

so the comparison is between quantities that both include the verdict decision.
From the fits it reports

    kappa = c_R / c_B                    -- the ratio at large n
    n0    = (F_B - F_R) / c_R            -- the floor, in rows
    band wins iff k < kappa*(n - n0)

Output: exp/cost_model.json
"""
import io
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_model.json")
RNG = np.random.default_rng(20261105)

N = 1 << 22                 # 4.19 M rows
REPS = 9


def best(fn, reps=REPS):
    """minimum over reps: the estimator of a cost floor, not of the mean."""
    t = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t.append(time.perf_counter_ns() - t0)
    return min(t)


def main():
    res = {}

    # ---------------------------------------------------------------- layout
    # sorted within group: groups of equal size, values int64
    per_group = 1024
    n_groups = N // per_group
    val = RNG.integers(0, 10 ** 6, N).astype(np.int64)
    val.sort()                       # one global sorted copy, groups are slices
    gid = (np.arange(N) // per_group).astype(np.int32)
    offs = np.arange(0, N + 1, per_group, dtype=np.int64)
    mod = (RNG.random(N) < 0.6).astype(np.int8)
    omega = np.int8(1)

    theta = np.int64(500000)

    # ------------------------------------------------ 1. bare gather, a_g
    # contiguous read + integer add, nothing else: the floor of any pass
    def bare(n):
        s = 0
        for j in range(n // 65536):
            s += int(np.sum(val[j * 65536:(j + 1) * 65536]))
        return s

    n_probe = 65536 * 32
    t = best(lambda: bare(n_probe))
    a_g = t / n_probe

    # ------------------------------------------------ 2. full path R, c_R
    # gather + compare + override filter + popcount: the work path R does.
    # The filter is `~(m == 1)`: at omega = 1 the two-state conjunction
    # `~((m == 1) & (omega == 1))` is the same predicate at 2.8 ns/row more,
    # which is correction 3 in README and was still in this file until
    # 2026-11-06.  With it, c_R read 3.27-3.41 ns/row and derived.kappa 0.72-0.76;
    # with the predicate fixed, c_R 0.33 and derived.kappa 0.069.  Neither is a
    # paper number -- the six bare layouts come from exp/n0_holdout.
    def scan(n):
        for j in range(n // 65536):
            v = val[j * 65536:(j + 1) * 65536]
            m = mod[j * 65536:(j + 1) * 65536]
            hit = (v >= theta) & ~(m == 1)
            np.count_nonzero(hit)

    t = best(lambda: scan(n_probe))
    c_R = t / n_probe

    # ------------------------------------------------ 3. random fetch, a_p
    # the fetch a band retrieval pays for a candidate located by rank
    idx = RNG.integers(0, N, 8192)
    t = best(lambda: val.take(idx))
    a_p = t / idx.size

    # ------------------------------------------------ 4. pointer chase
    # what a probe costs when the accesses are dependent: a permutation cycle
    M = 1 << 22
    perm = RNG.permutation(M).astype(np.int64)
    nxt = np.empty(M, dtype=np.int64)
    nxt[perm[:-1]] = perm[1:]
    nxt[perm[-1]] = perm[0]
    # numpy cannot chase; do it in a tight python loop over a smaller cycle
    cyc = perm[:200000]

    def chase():
        p = 0
        for _ in range(2000):
            p = int(nxt[p])
        return p

    t = best(chase, reps=5)
    a_chase = t / 2000.0

    # ------------------------------------------------ 5. rank probe per group
    # two binary searches inside one group's sorted slice: the price of |B|
    gs = 8
    gidx = RNG.integers(0, n_groups, gs)

    def rank():
        k = 0
        for g in gidx:
            lo = np.searchsorted(val[offs[g]:offs[g + 1]], 400000)
            hi = np.searchsorted(val[offs[g]:offs[g + 1]], 500000)
            k += hi - lo
        return k

    t = best(rank)
    a_rank = t / gs        # per group, both endpoints

    # ------------------------------------------------ 6. lines, both paths
    # fit by least squares on the measured points, not assumed
    ns = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20]
    tr = []
    for n in ns:
        t = best(lambda n=n: scan(n))
        tr.append((n, t))
    slope_R, int_R = np.polyfit([x for x, _ in tr], [y for _, y in tr], 1)

    tr_meas = [{"n": n, "ns": t} for n, t in tr]

    ks = [1, 4, 16, 64, 256, 1024, 4096, 8192]
    tb = []
    for k in ks:
        ii = RNG.integers(0, N, k)
        t = best(lambda ii=ii: (val.take(ii) >= theta).sum())
        tb.append((k, t))
    slope_B, int_B = np.polyfit([x for x, _ in tb], [y for _, y in tb], 1)
    tb_meas = [{"k": k, "ns": t} for k, t in tb]

    kappa = slope_R / slope_B
    n0 = (int_B - int_R) / slope_R

    res.update({
        "instance": {"rows": int(N), "groups": int(n_groups),
                     "rows_per_group": per_group},
        "components_ns": {
            "a_gather": round(float(a_g), 3),
            "c_R_full_pass": round(float(c_R), 3),
            "a_random_fetch": round(float(a_p), 3),
            "a_pointer_chase": round(float(a_chase), 3),
            "a_rank_per_group": round(float(a_rank), 1),
        },
        "fit": {
            "c_R_ns_per_row": round(float(slope_R), 4),
            "F_R_ns": round(float(int_R), 1),
            "c_B_ns_per_candidate": round(float(slope_B), 4),
            "F_B_ns": round(float(int_B), 1),
        },
        "derived": {
            "kappa": round(float(kappa), 4),
            "n0_rows": round(float(n0), 1),
        },
        "points_R": tr_meas,
        "points_B": tb_meas,
        # the v7 arithmetic, for comparison: a_gather / (a_probe + a_gather)
        "v7_ratio": round(float(a_g / (a_p + a_g)), 4),
    })

    print(json.dumps(res, indent=1))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
