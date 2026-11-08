"""W12 -- CERTIS end-to-end: driver.

Pre-registration (fixed before the first run; echoed unedited into the output)

Question.  The paper claims the R/B split is decided per correction by
|B|/|scope| against C_HAT ~ 5%.  W8 priced that on hand-picked scenarios; W13
priced engines on one query.  Neither turns a correction stream into a flip set.
Does the pipeline, run end to end on a real rule topology, produce (a) exactly
the right flip set and (b) the latency/throughput the claim implies?

P1  FLATNESS.  At fixed band fraction, certification cost grows as
    O(|Gamma| log n + k), so the exponent of cost against |D| is below 0.20.
    Falsified if the fitted exponent exceeds 0.20.

P2  TAIL.  p99 is set by the batches that trigger an aggregate refresh
    (regime b/c), not by the common path.  Concretely: p99 of a mixed stream is
    at least 3x its p50, and the p99 batch contains a regime (b)/(c) correction.
    Falsified if p99/p50 < 3 on the mixed stream.

P3  SCALING.  Against FULL recomputation the advantage grows with |D| at fixed
    ABSOLUTE correction size (exponent > 0.5) and is constant at fixed RELATIVE
    size (exponent < 0.2).  Against SCOPE recompute neither holds -- no growth
    is expected there, and if growth appears the R/B claim is mis-stated.

P4  SPLIT.  On the corpus's correction mix, the CERTIS policy routes fewer than
    10% of corrections to the band.  Falsified if the band share exceeds 10%,
    which would mean the paper's "most corrections take path R" is wrong.

P5  COMPLETENESS.  Delta-completeness holds in every cell: missed flips == 0 and
    extra flips == 0.  This is not falsifiable in the statistical sense -- a
    single missed flip voids the run, and the driver aborts rather than reports.

Verdict rule, fixed in advance.  P1/P2/P3 failing means the claim's quantitative
form is wrong and the text must say so; P4 failing means the split is wrong in
direction and S5.7's "most corrections take path R" must be withdrawn.  P5
failing is not a finding, it is a bug in the system, and the run is void.

Scale, per CP2: 10^6 lines, 10^3 live rules, 1% correction rate.  The |D| axis
is swept at three points because P1 and P3 are about exponents and one point
cannot have a slope.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "w8")))

import certis as C                                    # noqa: E402
import w8_lib as L                                    # noqa: E402

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def hr(c="-", n=96):
    say(c * n)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def fit_log(xs, ys):
    """Exponent of y ~ x^a, by least squares on logs.  Two points give a slope;
    three give a check, which is why the axis has three."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = (xs > 0) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    a = np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0]
    return float(a)


# --------------------------------------------------------------------- setup

def load_topology():
    pairs, raw_rows = L.parse_ptp()
    mue = L.parse_mue()
    cid, rule_keys, topo = L.topology(pairs, mue)
    n_codes = len(cid)
    cap_of_code = np.full(n_codes, -1, dtype=np.int32)
    k = 0
    for c in mue:
        cap_of_code[cid[c]] = k
        k += 1
    deg = np.zeros(n_codes)
    for a, b in pairs:
        deg[cid[a]] += 1
        deg[cid[b]] += 1
    p_uni = np.ones(n_codes) / n_codes
    wdeg = deg + 1.0
    p_deg = wdeg / wdeg.sum()
    pair_codes = [(cid[a], cid[b]) for a, b in pairs]
    return dict(cid=cid, cap_of_code=cap_of_code, rule_keys=rule_keys,
                topo=topo, raw_rows=raw_rows, p_uni=p_uni, p_deg=p_deg,
                pair_codes=pair_codes, n_codes=n_codes)


_BASE_CACHE = {}


def make(n_g, topo, seed, code_model="degree"):
    """One instance per (|D|, code model), rebuilt only once.  The three policies
    must see the same rows and the same correction stream, or the comparison is
    between instances rather than between policies -- so the base is cached and
    the stream is re-seeded per cell from the same seed."""
    k = (int(n_g), code_model)
    if k not in _BASE_CACHE:
        rng0 = np.random.default_rng(20261104)
        inst = L.build_instance(n_g, np.array([0.10, 0.30, 0.35, 0.15, 0.10]),
                                topo["p_deg"] if code_model == "degree" else topo["p_uni"],
                                topo["cid"], rng0)
        override = rng0.random(inst["n_g"]) < C.OVERRIDE_P
        store = C.Store(inst, override)
        freq = np.diff(store.cptr)
        cap_codes = np.argsort(-freq)[:4000].astype(np.int32)
        rules = C.RuleSet(topo["cap_of_code"], topo["pair_codes"], rel_k=1.0, glob_k=1.0)
        _BASE_CACHE[k] = (inst, store, rules, cap_codes)
    inst, store, rules, cap_codes = _BASE_CACHE[k]
    rng = np.random.default_rng(seed)
    stream = C.Stream(store, rules, rng, cap_codes, topo["pair_codes"])
    return inst, store, rules, stream, rng


# ------------------------------------------------------------------- one cell

def run_cell(n_g, topo, seed, n_rules, pi_c, n_batches, policy, mix,
             cap_mode="as-is", measure_paths=True, cost=None):
    """One configuration: build, run n_batches, verify every batch."""
    inst, store, rules, stream, rng = make(n_g, topo, seed)
    cert = C.Certifier(store, rules, policy=policy, cost=cost)

    lat_all, lat_by_kind = [], {}
    missed_tot = extra_tot = flips_tot = 0
    per_batch = []
    gt_rows = 0
    duel = []               # per-correction R vs B, for the criterion check

    for b in range(n_batches):
        corrs = stream.batch(n_rules, pi_c, mix, cap_mode)
        if not corrs:
            continue
        t_batch0 = time.perf_counter()
        batch_flips = []
        kinds = []
        for corr in corrs:
            if policy == "both" and corr.kind == "cap":
                # execute the same correction down both paths and keep both
                # times: this is what checks the criterion rather than assuming
                # it, and it is the only measurement in the paper that can put
                # C_HAT where it claims to be
                fR, _, tR = cert.apply(corr, path="R")
                fB, _, tB = cert.apply(corr, path="B")
                frac = cert.learned.get(corr.key(), float("nan"))
                if frac != frac:
                    ns = store.scope_len(corr.code)
                    lo, hi = sorted((corr.old, corr.new))
                    nb = len(store.band_u(corr.code, int(lo), int(hi)))
                    frac = nb / max(ns, 1)
                    cert.learned[corr.key()] = frac
                duel.append(dict(frac=frac, tR=tR, tB=tB, n_scope=store.scope_len(corr.code),
                                 n_band=int(frac * store.scope_len(corr.code)),
                                 same=bool(np.array_equal(np.sort(fR), np.sort(fB)))))
                flips, path, dt = (fR, "R", tR) if tR <= tB else (fB, "B", tB)
            else:
                flips, path, dt = cert.apply(corr)
            lat_all.append(dt)
            lat_by_kind.setdefault(corr.kind, []).append(dt)
            batch_flips.append(flips)
            kinds.append(path)
        t_batch = time.perf_counter() - t_batch0

        # system flip set for the batch: union of the members' (codes distinct
        # within a batch, so no pair of corrections can undo each other)
        sys_set = np.unique(np.concatenate(batch_flips)) if batch_flips else np.empty(0, np.int32)

        # ground truth, off the system path
        t_gt0 = time.perf_counter()
        gt_set = np.unique(np.concatenate([C.ground_truth(store, c) for c in corrs])) \
            if corrs else np.empty(0, np.int32)
        gt_rows += time.perf_counter() - t_gt0
        missed, extra = C.delta_complete(sys_set, gt_set)
        missed_tot += missed
        extra_tot += extra
        flips_tot += len(gt_set)

        per_batch.append(dict(batch=b, n=len(corrs), t=t_batch,
                              p_path=bool(any(k == "P" for k in kinds)),
                              p_share=float(np.mean([k == "P" for k in kinds])),
                              flips=int(len(gt_set)), missed=missed, extra=extra))
        if missed or extra:
            say(f"  !! BATCH {b}: missed={missed} extra={extra} "
                f"kinds={[c.kind for c in corrs]}")
            break

    res = dict(
        n_g=int(n_g), n_l=int(store.n_l), n_rules=int(n_rules), pi_c=float(pi_c),
        policy=policy, cap_mode=cap_mode, n_batches=len(per_batch),
        index_seconds=store.index_seconds, index_mb=store.index_bytes / 1e6,
        lat_p50=med(lat_all) if lat_all else float("nan"),
        lat_p90=pct(lat_all, 0.90), lat_p99=pct(lat_all, 0.99),
        lat_mean=float(np.mean(lat_all)) if lat_all else float("nan"),
        lat_max=max(lat_all) if lat_all else float("nan"),
        n_corr=len(lat_all),
        throughput_hz=float(len(lat_all) / sum(p["t"] for p in per_batch))
        if per_batch and sum(p["t"] for p in per_batch) > 0 else float("nan"),
        paths=dict(cert.stats),
        band_share=float(cert.stats["B"] / max(sum(cert.stats.values()), 1)),
        p_share=float(cert.stats["P"] / max(sum(cert.stats.values()), 1)),
        missed=int(missed_tot), extra=int(extra_tot), flips_total=int(flips_tot),
        gt_seconds=float(gt_rows),
        batch_detail=per_batch,
        duel=duel,
    )
    # per-kind latency, the (ii) evidence: which kind owns the tail
    res["lat_by_kind"] = {k: dict(p50=med(v), p99=pct(v, 0.99), n=len(v),
                                  mean=float(np.mean(v)))
                          for k, v in lat_by_kind.items()}
    # the ratio the decision turns on, as the instance actually produced it
    fr = [v for v in cert.learned.values() if v == v]
    res["fracs"] = dict(n=len(fr), p10=pct(fr, 0.10), median=med(fr),
                        p90=pct(fr, 0.90), max=max(fr) if fr else float("nan"),
                        below_C=float(np.mean([v < C.C_HAT for v in fr])) if fr else float("nan"))
    if measure_paths:
        res["full_recompute_seconds"] = full_cached(store, rules, topo, store.n_l)
    # keep the counters (probe / scan / rows / veto) before they are cleared:
    # "how many probes the policy declined" is the whole evidence for the veto
    res["counters"] = dict(cert.counters)
    cert.reset_counters()
    return res


_FULL_CACHE = {}


def microbench_fixed_k(store, k, rng, n_codes=40):
    """Band retrieval with a FIXED member count, on instances of different |D|.

    This is the only way to test prediction (i): in a real stream |B| grows with
    |D| because the scope grows, so an end-to-end p50 against |D| measures the
    scope, not the retrieval.  Holding k fixed isolates the O(|Gamma| log n)
    term, which is the term the prediction is about.
    """
    freq = np.diff(store.cptr)
    cand = np.flatnonzero(freq >= k + 8)
    if len(cand) == 0:
        return float("nan"), float("nan")
    want = np.linspace(0, len(cand) - 1, min(n_codes, len(cand))).astype(int)
    picked = cand[want]
    tt, nb = [], []
    for c in picked:
        lo_c, hi_c = store.cptr[c], store.cptr[c + 1]
        v = np.sort(store.amt_c[lo_c:hi_c])
        i0 = len(v) // 2
        i1 = min(len(v) - 1, i0 + k)
        if i1 - i0 < k // 2:
            continue
        t_lo, t_hi = float(v[i0]), float(v[i1])
        f = lambda c=c, t_lo=t_lo, t_hi=t_hi: store.band_a(c, t_lo, t_hi)
        tt.append(L.timeit(f, target=0.008, repeat=5))
        nb.append(len(f()))
    if not tt:
        return float("nan"), float("nan")
    return med(tt), med(nb)


def fit_two_param(duel):
    """Per-element slopes and fixed costs for the two paths, fitted on the same
    corrections.  A two-parameter model is the minimum needed to see why the
    ratio criterion transfers badly: both paths have a floor, and which floor is
    larger decides every small correction regardless of the ratio."""
    ns = np.array([d["n_scope"] for d in duel], dtype=float)
    nb = np.array([d["n_band"] for d in duel], dtype=float)
    tr = np.array([d["tR"] for d in duel], dtype=float)
    tb = np.array([d["tB"] for d in duel], dtype=float)
    aR, bR = np.polyfit(ns, tr, 1)
    aB, bB = np.polyfit(nb, tb, 1)
    fbar = float(np.mean(nb / np.maximum(ns, 1.0)))
    return float(aR), float(bR), float(aB), float(bB), fbar


def full_cached(store, rules, topo, key):
    """FULL recomputation depends only on the instance, and re-running it once
    per policy would triple the wall time for the same number."""
    if key not in _FULL_CACHE:
        _FULL_CACHE[key] = measure_full(store, rules, topo)
    return _FULL_CACHE[key]


def measure_full(store, rules, topo):
    """FULL: evaluate every (rule, group) pair in the instance, the baseline
    every incremental claim is against.  Same construction as W8 so the two
    sections cannot disagree about what 'full' costs."""
    inst = store.inst
    pr, pg, pl = L.match_pairs(inst, topo["rule_keys"], L.pair_offsets(inst))
    mr, mg, ml = L.match_mue(inst, topo["cap_of_code"])
    if len(pr) == 0 and len(mr) == 0:
        return float("nan")
    n_ev = len(pr) + len(mr)
    gmod = store.override.astype(np.float32)
    val = np.empty(n_ev, dtype=np.float32)
    thr = np.empty(n_ev, dtype=np.float32)
    val[:len(pr)] = 1.0 - gmod[pg]
    val[len(pr):] = store.unit[ml]
    thr[:len(pr)] = 0.5
    thr[len(pr):] = 1.5

    def once():
        v = np.empty(n_ev, dtype=np.float32)
        v[:len(pr)] = 1.0 - gmod[pg]
        v[len(pr):] = store.unit[np.concatenate([pl, ml])[len(pr):]]
        return v <= thr

    return float(L.timeit(once, target=0.02, repeat=5))


# ---------------------------------------------------------------------- main

def main():
    say("W12 -- CERTIS end to end")
    say("real topology: CMS NCCI PTP 2026Q4 practitioner + MUE 2026Q2")
    hr("=")

    topo = load_topology()
    say(f"  codes={topo['n_codes']:,d} ptp_pairs={len(topo['pair_codes']):,d} "
        f"ptp_rows_raw={topo['raw_rows']:,d}")

    smoke = "--smoke" in sys.argv
    n_rules = 1000
    pi_c = 0.01
    n_batches = 4 if smoke else 60
    grid = [20000] if smoke else [100_000, 350_000, 1_000_000]

    mix_main = {"cap_low": 0.55, "pair": 0.40, "rel": 0.05}
    mix_tail = {"cap_low": 0.50, "pair": 0.35, "rel": 0.10, "global": 0.05}

    hr()
    say("A. correctness gate (P5): Delta-completeness on every path")
    hr()
    r = run_cell(grid[0], topo, 20261104, n_rules, pi_c, n_batches, "CERTIS", mix_main)
    say(f"  n_g={r['n_g']:,d} lines={r['n_l']:,d} corr={r['n_corr']} "
        f"paths={r['paths']} band_share={r['band_share']:.3f}")
    say(f"  missed={r['missed']} extra={r['extra']} flips={r['flips_total']} "
        f"-> {'PASS' if r['missed'] == 0 and r['extra'] == 0 else 'FAIL'}")
    if r["missed"] or r["extra"]:
        say("  aborting: a missed flip voids the run")
        return 2
    for k, v in sorted(r["lat_by_kind"].items()):
        say(f"    {k:<8s} n={v['n']:>4d} p50={v['p50']*1e6:>9.2f}us "
            f"p99={v['p99']*1e6:>9.2f}us")

    if smoke:
        say()
        say("  smoke only; no axis sweep")
        write(r, topo)
        return 0

    hr()
    say("B. P1 flatness and P3 scaling: the |D| axis")
    hr()
    cells = []
    for n_g in grid:
        for pol in ("CERTIS", "R-always", "B-always"):
            c = run_cell(n_g, topo, 20261104 + n_g % 97, n_rules, pi_c,
                         n_batches, pol, mix_main)
            cells.append(c)
            say(f"  n_g={c['n_g']:>9,d} lines={c['n_l']:>10,d} {pol:<9s} "
                f"p50={c['lat_p50']*1e6:>9.2f}us p99={c['lat_p99']*1e6:>9.2f}us "
                f"thr={c['throughput_hz']:>9,.0f}/s FULL={c['full_recompute_seconds']*1e3:>8.2f}ms "
                f"missed={c['missed']}")
    say()
    for pol in ("CERTIS", "R-always", "B-always"):
        sub = [c for c in cells if c["policy"] == pol]
        ns = [c["n_l"] for c in sub]
        e = fit_log(ns, [c["lat_p50"] for c in sub])
        say(f"  {pol:<9s} exponent of p50 vs |D| = {e:+.3f}  "
            f"(end-to-end: |scope| grows with |D|, so this is the wrong test "
            f"for P1 -- see B2)")

    hr()
    say("B2. P1 done properly: band cost at FIXED |B|, varying only |D|")
    hr()
    p1 = {}
    for k in (64, 512):
        ts, ns, kk = [], [], []
        for n_g in grid:
            _i, store, _r, _st, rng = make(n_g, topo, 20261104)
            t, nb = microbench_fixed_k(store, k, rng)
            if t != t:
                continue
            ts.append(t)
            ns.append(store.n_l)
            kk.append(nb)
            say(f"  k={k:>4d} |D|={store.n_l:>10,d} |B|={nb:>6.0f} t={t*1e6:>8.2f}us")
        e = fit_log(ns, ts)
        p1[k] = dict(exponent=e, ts=ts, ns=ns, k=kk)
        say(f"  k={k}: exponent of band cost vs |D| = {e:+.3f} "
            f"{'<= P1 OK' if e < 0.20 else '> P1 FALSIFIED'}")

    hr()
    say("C. P2 tail: does the aggregate refresh own the p99?")
    hr()
    tail = run_cell(grid[-1], topo, 777, n_rules, pi_c, n_batches, "CERTIS", mix_tail)
    say(f"  mixed stream: p50={tail['lat_p50']*1e6:.2f}us p99={tail['lat_p99']*1e6:.2f}us "
        f"ratio={tail['lat_p99']/max(tail['lat_p50'],1e-12):.1f}x "
        f"P-share={tail['p_share']:.3f} missed={tail['missed']}")
    for k, v in sorted(tail["lat_by_kind"].items()):
        say(f"    {k:<8s} n={v['n']:>4d} p50={v['p50']*1e6:>9.2f}us p99={v['p99']*1e6:>9.2f}us")
    lat_all = [p["t"] for p in tail["batch_detail"]]
    say(f"  batch latency p50={med(lat_all)*1e3:.2f}ms max={max(lat_all)*1e3:.2f}ms")

    hr()
    say("D. P4 split, and whether the criterion can be checked rather than assumed")
    hr()
    for mname, c in (("corpus mix", next(x for x in cells if x["policy"] == "CERTIS")),
                     ("with tail", tail)):
        f = c["fracs"]
        say(f"  {mname:<12s} fraction |B|/|scope|: p10={f['p10']:.3f} "
            f"median={f['median']:.3f} p90={f['p90']:.3f} "
            f"share below C_HAT={f['below_C']*100:.1f}%")
        say(f"  {'':<12s} R={c['paths']['R']:>5d} B={c['paths']['B']:>4d} "
            f"P={c['paths']['P']:>4d} band_share={c['band_share']*100:>5.1f}% "
            f"{'<= P4 OK' if c['band_share'] < 0.10 else '> P4 FALSIFIED'}")

    # the criterion itself: both paths down the same correction, binned by ratio
    say()
    say("  criterion check: both paths on the same correction, binned by |B|/|scope|")
    du = run_cell(grid[-1], topo, 31337, n_rules, pi_c, max(12, n_batches // 2),
                  "both", {"cap_low": 0.5, "cap_high": 0.5, "pair": 0.0},
                  measure_paths=False)
    say(f"  corrections={du['n_corr']}  identical flip sets: "
        f"{sum(d['same'] for d in du['duel'])}/{len(du['duel'])}")
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25),
            (0.25, 0.50), (0.50, 1.01)]
    say(f"  {'|B|/|s|':<14s} {'n':>5s} {'tR(us)':>9s} {'tB(us)':>9s} {'tB/tR':>7s}  winner")
    for lo, hi in bins:
        sub = [d for d in du["duel"] if lo <= d["frac"] < hi]
        if not sub:
            continue
        tr = med([d["tR"] for d in sub])
        tb = med([d["tB"] for d in sub])
        win = "B" if tb < tr else "R"
        say(f"  [{lo:.2f},{hi:.2f})   {len(sub):>5d} {tr*1e6:>9.2f} {tb*1e6:>9.2f} "
            f"{tb/max(tr,1e-12):>7.2f}  {win}"
            + ("   <- C_HAT" if lo < C.C_HAT <= hi else ""))
    cross = None
    for lo, hi in bins:
        sub = [d for d in du["duel"] if lo <= d["frac"] < hi]
        if sub and med([d["tB"] for d in sub]) < med([d["tR"] for d in sub]):
            cross = lo
    say(f"  measured crossover is at or below |B|/|scope| = {cross}; "
        f"C_HAT = {C.C_HAT:.2f}")
    say()
    say("  same corrections binned by |scope|: what path R actually pays for")
    say(f"  {'|scope|':<18s} {'n':>5s} {'tR(us)':>9s} {'tB(us)':>9s} {'|B|/|s|':>8s} winner")
    for lo, hi in ((0, 1000), (1000, 4000), (4000, 10000), (10000, 10 ** 9)):
        sub = [d for d in du["duel"] if lo <= d["n_scope"] < hi]
        if not sub:
            continue
        tr = med([d["tR"] for d in sub])
        tb = med([d["tB"] for d in sub])
        fm = med([d["frac"] for d in sub])
        tag = f"[{lo},{hi})" if hi < 10 ** 9 else f"[{lo},inf)"
        say(f"  {tag:<18s} {len(sub):>5d} {tr*1e6:>9.2f} {tb*1e6:>9.2f} "
            f"{fm:>8.3f}  {'B' if tb < tr else 'R'}")
    # the fixed cost each path cannot avoid: the floor that decides small cases
    tiny = [d for d in du["duel"] if d["n_scope"] < 1000]
    if tiny:
        say(f"  floor: on scopes under 1,000 rows path R already costs "
            f"{med([d['tR'] for d in tiny])*1e6:.1f}us and path B "
            f"{med([d['tB'] for d in tiny])*1e6:.1f}us -- neither is bounded by")
        say(f"         its per-element term at this scale, which is why the "
            f"per-element criterion does not transfer to end-to-end times")

    hr()
    say("F. What the criterion should be: route on |scope|, not on |B|/|scope|")
    hr()
    aR, bR, aB, bB, fbar = fit_two_param(du["duel"])
    say(f"  fitted from the duel:  t_R = {aR*1e9:>6.2f}ns*|s| + {bR*1e6:>6.2f}us")
    say(f"                         t_B = {aB*1e9:>6.2f}ns*|B| + {bB*1e6:>6.2f}us")
    say(f"  mean |B|/|s| = {fbar:.3f}")
    denom = aR - aB * fbar
    if denom > 0:
        s_star = (bB - bR) / denom
        say(f"  predicted switch at |scope| = {s_star:,.0f} rows; below it the "
            f"band's fixed probe is not repaid")
    else:
        s_star = float("nan")
        say("  no switch predicted: the band's per-element term never falls "
            "below the scope's")
    cost = (aR, bR, aB, bB, fbar)
    c2 = []
    for n_g in grid:
        c = run_cell(n_g, topo, 20261104 + n_g % 97, n_rules, pi_c, n_batches,
                     "CERTIS2", mix_main, cost=cost)
        c2.append(c)
        base = next(x for x in cells if x["n_g"] == c["n_g"] and x["policy"] == "R-always")
        say(f"  n_g={c['n_g']:>9,d} lines={c['n_l']:>10,d} CERTIS2 "
            f"p50={c['lat_p50']*1e6:>8.2f}us band_share={c['band_share']*100:>5.1f}% "
            f"vs R-always {base['lat_p50']*1e6:>8.2f}us "
            f"({base['lat_p50']/max(c['lat_p50'],1e-12):>5.2f}x) missed={c['missed']}")

    hr()
    say("E. P3: absolute vs relative correction size, against FULL")
    hr()
    say("  capped at the 90th percentile, because at the median the corpus's cap")
    say("  is 1 and +1 and x2 are the same correction -- the arms would not differ")
    abs_cells = []
    p3 = {}
    for mode in ("abs", "rel"):
        rows = []
        for n_g in grid:
            c = run_cell(n_g, topo, 4242, n_rules, pi_c, max(12, n_batches // 3),
                         "CERTIS", {"cap_high": 1.0}, cap_mode=mode)
            rows.append(c)
            say(f"  {mode:<3s} n_g={c['n_g']:>9,d} lines={c['n_l']:>10,d} "
                f"p50={c['lat_p50']*1e6:>9.2f}us FULL={c['full_recompute_seconds']*1e3:>8.2f}ms "
                f"FULL/cert={c['full_recompute_seconds']/max(c['lat_p50'],1e-12):>9,.0f}x "
                f"|B|/|s| med={c['fracs']['median']:.3f}")
        e_full = fit_log([c["n_l"] for c in rows],
                         [c["full_recompute_seconds"] / max(c["lat_p50"], 1e-12)
                          for c in rows])
        e_r = fit_log([c["n_l"] for c in rows], [c["lat_p50"] for c in rows])
        e_frac = fit_log([c["n_l"] for c in rows],
                         [max(c["fracs"]["median"], 1e-6) for c in rows])
        p3[mode] = dict(rows=rows, e_full=e_full, e_r=e_r, e_frac=e_frac)
        say(f"  {mode:<3s} exponent: FULL/cert={e_full:+.3f}  cert vs |D|={e_r:+.3f}  "
            f"|B|/|s| vs |D|={e_frac:+.3f}")
        if mode == "abs":
            say(f"      -> P3 first half {'OK' if e_full > 0.5 else 'FALSIFIED'}")
        abs_cells.extend(rows)

    expo = {p: fit_log([c["n_l"] for c in cells if c["policy"] == p],
                       [c["lat_p50"] for c in cells if c["policy"] == p])
            for p in ("CERTIS", "R-always", "B-always")}
    pay = dict(cells=cells, tail=tail, p1=p1, p3=p3, abs_cells=abs_cells,
               duel_cell=du, certis2=c2,
               switch=dict(aR=aR, bR=bR, aB=aB, bB=bB, fbar=fbar,
                           sigma_star=s_star),
               exponents=dict(p50_vs_D=expo,
                              full_over_R=p3["abs"]["e_full"],
                              R_vs_D=p3["abs"]["e_r"]),
               grid=grid, n_rules=n_rules, pi_c=pi_c,
               topology=topo["topo"], ptp_rows_raw=topo["raw_rows"])
    write(pay, topo)
    return 0


def write(pay, topo):
    pay = dict(pay)
    pay["prereg"] = __doc__
    with open(os.path.join(HERE, "w12_results.json"), "w", encoding="utf-8") as fh:
        json.dump(pay, fh, indent=1, ensure_ascii=False, default=float)
    with open(os.path.join(HERE, "w12_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    say()
    say("wrote w12_results.json / w12_out.txt")


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    say(f"[wall {time.time()-t0:.1f}s]")
    sys.exit(rc or 0)
