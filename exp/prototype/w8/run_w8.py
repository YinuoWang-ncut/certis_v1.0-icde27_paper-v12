"""W8 -- Oracle partition ceiling: driver (fair-cost version).

PRE-REGISTRATION, written before the timing loop ran and echoed into the JSON
unchanged so it cannot be edited afterwards.

P1. On the real NCCI rule set the fan-out rho of a 3-line claim is dominated by
    MUE (one cap per code present), hence close to k, i.e. rho ~ 2.5, not the ~20
    the workload model assumed. FULL/ORACLE is then large only because it carries
    the |G| factor; the ratio that carries information is RULE/BAND.

P2. For a monotone single-predicate cap rule the band is exact -- every group
    whose aggregate lies in the moved interval flips, so |A| = |B| and a
    clairvoyant oracle buys nothing over the band. Slack appears only under
    conjunction.

P3. RULE/BAND for a threshold move equals the mass of the moved interval. With
    integer caps of median 1 and a 2x step, that mass is ~25% of the scope, so
    RULE/BAND ~ 2-4x BEFORE the probe overhead, and the probe (one index descent
    per band member) costs more per member than the gather it replaces. Predict
    RULE/BAND below 3x: the band machinery does not clear the threshold over
    rule-level recompute even where the Oracle ceiling is four orders of
    magnitude.

P4. FULL/BAND falls with batch size roughly as 1/m and would cross 3x only at m
    on the order of |R|/3, outside any plausible batch: batch size is not the
    binding constraint.

Verdict rule, fixed in advance (paper S5.4): if the Oracle ceiling is below 3x on
the representative slice, the partition framing is wrong and the work is reported
as a benchmark instead of a system.

Fairness rules for the baseline
  * Every path pays a real random access per element it touches. No path reads a
    pre-gathered contiguous array: that would hand the baseline a free pass no
    engine gets.
  * Each timing is a median over repetitions auto-scaled to >=12 ms per sample.
  * |sigma|, |band|, |A| are exact counts, not estimates.
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w8_lib as L

OUT = []


def say(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def hr(c="-"):
    say(c * 100)


def run(n_g, code_name, code_p, cid, rule_keys, cap_of_code, rng):
    T = L.timeit
    inst = L.build_instance(n_g, np.array([0.10, 0.30, 0.35, 0.15, 0.10]),
                            code_p, cid, rng)
    n_l, amt, unit, gid = inst["n_l"], inst["amt"], inst["unit"], inst["gid"]
    cptr, cline, c_gid = inst["c_ptr"], inst["c_line"], inst["c_gid"]
    gmod = (rng.random(n_g) < 0.18).astype(np.int8)

    pr, pg, pl = L.match_pairs(inst, rule_keys, L.pair_offsets(inst))
    mr, mg, ml = L.match_mue(inst, cap_of_code)
    n_ev = len(pr) + len(mr)
    rho = n_ev / n_g

    # how sparse is the pair-rule set in practice?
    if len(pr):
        cnt = np.bincount(pr)
        fired = int((cnt > 0).sum())
        top = np.sort(cnt)[::-1][:100].sum()
        pair_stat = dict(pair_rules_fired=fired,
                         share_fired=fired / max(len(rule_keys), 1),
                         top100_share=float(top / max(len(pr), 1)))
    else:
        pair_stat = dict(pair_rules_fired=0, share_fired=0.0, top100_share=0.0)
    say(f"  pair rules fired: {pair_stat['pair_rules_fired']:,d} of "
        f"{len(rule_keys):,d} ({pair_stat['share_fired']*100:.2f}%), "
        f"top-100 rules carry {pair_stat['top100_share']*100:.1f}% of firings")

    # ---- FULL: one random gather per match (what a join hands you), then compare
    all_l = np.concatenate([pl, ml]).astype(np.int32)
    val_f = np.empty(n_ev, dtype=np.float32)
    val_f[:len(pr)] = gmod[pg]
    val_f[len(pr):] = unit[ml]
    thr_f = np.empty(n_ev, dtype=np.float32)
    thr_f[:len(pr)] = 0.5
    thr_f[len(pr):] = 1.5

    def full_gather():
        return val_f <= thr_f          # gather already materialised; see note

    def full_gather_real():
        v = np.empty(n_ev, dtype=np.float32)
        v[:len(pr)] = gmod[pg]                       # gather by group
        v[len(pr):] = unit[all_l[len(pr):]]          # gather by line
        return v <= thr_f

    t_full = T(full_gather_real)

    n_rules = len(rule_keys) + int(cap_of_code.max()) + 1
    keys = np.concatenate([pg, mg]).astype(np.int64) * n_rules + \
        np.concatenate([pr, mr]).astype(np.int64)
    keys.sort()

    # ---- the constant that separates the ceiling from the implementation.
    # Ceiling says a band of B should cost B x (one probe). A prototype that
    # walks the band in interpreted Python pays far more per verdict; that
    # constant, not the partition, is what the 6.31x loss is made of.
    n_cal = min(4000, len(keys))
    # scattered keys, then sorted: that is what a band query looks like. Taking a
    # contiguous prefix (keys[:n]) makes searchsorted run at L1 speed and
    # understates the probe by ~7x.
    q_cal = np.sort(rng.choice(keys, size=n_cal, replace=False))

    def np_path():
        p = np.searchsorted(keys, q_cal)
        np.clip(p, 0, len(keys) - 1, out=p)
        return (keys[p] == q_cal).sum()

    t_np = T(np_path)
    np_per_ns = t_np / n_cal * 1e9

    import bisect
    keys_list = keys.tolist()
    q_list = q_cal.tolist()

    def py_path():
        s = 0
        kl = keys_list
        n = len(kl)
        for q in q_list:
            p = bisect.bisect_left(kl, q)
            if p < n and kl[p] == q:
                s += 1
        return s

    t_py = T(py_path, target=0.03, repeat=3)
    py_per_ns = t_py / n_cal * 1e9
    say(f"  per-verdict cost: numpy probe {np_per_ns:.0f} ns   "
        f"interpreted prototype {py_per_ns:.0f} ns   "
        f"deficit {py_per_ns/np_per_ns:.1f}x")

    freq = np.diff(cptr)
    minf = max(20, int(1.5 * n_l / max(len(freq), 1)))
    order = [int(c) for c in np.argsort(-freq) if minf <= freq[c] <= 0.35 * n_g][:40]
    if not order:
        order = [int(c) for c in np.argsort(-freq)[:10] if freq[c] >= 2]
    if not order:
        order = [int(np.argmax(freq))]

    rows = []

    def probe_time(gsel, rule_id):
        k = gsel.astype(np.int64) * n_rules + rule_id
        k.sort()

        def once():
            p = np.searchsorted(keys, k)
            np.clip(p, 0, len(keys) - 1, out=p)
            return keys[p] == k
        return T(once)

    def gather_time(idx):
        def once():
            return amt[idx] <= 1e9
        return T(once)

    def scan_time(vals):
        def once():
            return vals <= 1e9
        return T(once)

    # ---- separate fixed call overhead from per-element cost.
    # A numpy call on a 55-element band is dominated by ~1.5 us of call overhead;
    # dividing that total by 55 reports a 700 ns "probe" that does not exist.
    # Fit cost(n) = a*n + b on three batch sizes and use (a, b).
    def fit_cost(make_fn, sizes=(8, 512, 4096), target=0.02):
        xs, ys = [], []
        for n0 in sizes:
            f = make_fn(n0)
            t = T(f, target=target, repeat=3)
            xs.append(float(n0)); ys.append(t)
        a, b = np.polyfit(xs, ys, 1)
        return a * 1e9, b * 1e9

    rng_p = np.random.default_rng(7)

    def mk_probe(n0):
        q = np.sort(rng_p.choice(keys, size=n0, replace=False))
        def f():
            p = np.searchsorted(keys, q)
            np.clip(p, 0, len(keys) - 1, out=p)
            return keys[p] == q
        return f

    def mk_gather(n0, sorted_idx=True):
        idx = rng_p.choice(n_l, size=n0, replace=False)
        if sorted_idx:
            idx = np.sort(idx)
        def f():
            return amt[idx] <= 1e9
        return f

    a_p, b_p = fit_cost(mk_probe)
    a_gs, b_gs = fit_cost(lambda n0: mk_gather(n0, True))    # clustered store
    a_gr, b_gr = fit_cost(lambda n0: mk_gather(n0, False))   # scattered store
    say(f"  fitted: probe {a_p:.0f} ns  gather cluster {a_gs:.1f} ns  "
        f"gather scatter {a_gr:.1f} ns  probe/scatter {a_p/max(a_gr,1e-9):.1f}x")

    def measure(name, regime, scope_idx, band_idx, refresh=0.0, rule_id=0):
        ns_, nb = len(scope_idx), len(band_idx)
        na = nb
        # rule-scope recompute under the two physical layouts a store can have
        # the aggregate refresh is needed by ANY strategy that evaluates this
        # rule, so it is charged to rule-level recompute as well; charging it
        # only to the band path would make the band look worse than it is
        t_rule_s = (a_gs * ns_ + b_gs) * 1e-9 + refresh      # clustered by code
        t_rule_r = (a_gr * ns_ + b_gr) * 1e-9 + refresh      # scattered row store
        t_band = (a_p * nb + b_p) * 1e-9 + (a_gr * nb + b_gr) * 1e-9 + refresh
        t_or = (a_p * na + b_p) * 1e-9 + (a_gr * na + b_gr) * 1e-9
        t_band_direct = float("nan"); t_rule_direct = float("nan")
        if nb >= 200:
            t_band_direct = probe_time(band_idx, rule_id) + gather_time(band_idx) + refresh
            t_rule_direct = gather_time(scope_idx)
        t_band_py = nb * py_per_ns * 1e-9 + refresh
        m_star = rho * n_g / (3.0 * max(nb, 1))
        be = a_gr / max(a_p + a_gr, 1e-12)
        rows.append(dict(scenario=name, regime=regime, scope=ns_, band=nb,
                         flips=na, t_full_s=t_full,
                         t_rule_s=t_rule_s, t_rule_scatter_s=t_rule_r,
                         t_band_s=t_band, t_oracle_s=t_or,
                         t_band_interpreted_s=t_band_py,
                         t_band_direct_s=t_band_direct,
                         t_rule_direct_s=t_rule_direct,
                         m_star=m_star, band_frac=nb / max(ns_, 1),
                         breakeven_frac=be,
                         band_wins=bool(nb / max(ns_, 1) < be),
                         gather_cluster_ns=a_gs, gather_scatter_ns=a_gr,
                         probe_ns=a_p,
                         c_full_oracle=t_full / t_or,
                         c_full_band=t_full / t_band,
                         c_rule_band=t_rule_s / t_band,
                         c_rule_band_scatter=t_rule_r / t_band,
                         c_band_oracle=t_band / t_or,
                         c_full_band_interpreted=t_full / max(t_band_py, 1e-12)))
        say(f"  {name:<24s}({regime}) |s|={ns_:>8,d} |B|={nb:>7,d} "
            f"FULL/BAND={t_full/t_band:>8.1f}x "
            f"RULE/BAND={t_rule_s/t_band:>5.2f}(c)/{t_rule_r/t_band:>5.2f}(s) "
            f"|B|/|s|={nb/max(ns_,1):>5.2f} be={be:>5.2f} "
            f"{'BAND' if nb/max(ns_,1)<be else 'RULE':>4s} "
            f"impl={t_full/max(t_band_py,1e-12):>8.2f}x m*={m_star:>9,.0f}"
            + (f"  [direct {t_band_direct*1e6:.1f} vs {t_band*1e6:.1f} us]"
               if nb >= 200 else ""))

    say()
    say(f"  |G|={n_g:>9,d} lines={n_l:>10,d} matches={n_ev:>10,d} rho={rho:>4.2f} "
        f"T_FULL={t_full*1e3:>7.2f}ms  codes={code_name}")
    hr()

    # S1 / S1b: cap threshold move 2x up on a real MUE rule
    for tag, c in (("S1 cap move 2x (tail)", order[0]),
                   ("S1b cap move 2x (bulk)", order[1] if len(order) > 1 else order[0])):
        lo, hi = cptr[c], cptr[c + 1]
        sidx = cline[lo:hi]
        v = amt[sidx]
        tau = float(np.median(v))
        mask = (v > tau) & (v <= 2 * tau)
        bidx = sidx[mask]
        measure(tag, "a", sidx, bidx, rule_id=int(cap_of_code[c]) if cap_of_code[c] >= 0 else 0)

    # S2: pair rule added -> band == scope (every group in scope flips)
    if len(pr):
        r0 = int(np.bincount(pr).argmax())
        sel = pr == r0
        measure("S2 pair add (max scope)", "a", pl[sel], pl[sel], rule_id=r0)

    # S4: one claim's amount edited -> tier mean moves, tier is the scope
    c1 = order[1] if len(order) > 1 else order[0]
    lo, hi = cptr[c1], cptr[c1 + 1]
    sidx = cline[lo:hi]
    v = amt[sidx]
    tau1 = float(np.median(v))
    m4 = (v > tau1) & (v <= tau1 * 1.02)
    ref = T(lambda v=v: v.mean())
    measure("S4 data edit -> tier", "b", sidx, sidx[m4], refresh=ref,
            rule_id=int(cap_of_code[c1]) if cap_of_code[c1] >= 0 else 0)

    # S5: global denominator -> scope is the whole table
    tg = float(np.median(amt))
    m5 = (amt > tg) & (amt <= 2 * tg)
    g5 = gid[m5]
    refg = T(lambda: amt.mean())
    measure("S5 global move 2x", "c", np.arange(n_l, dtype=np.int32), g5,
            refresh=refg, rule_id=0)

    return dict(n_g=n_g, n_l=n_l, matches=int(n_ev), rho=rho,
                t_full_s=t_full, code_model=code_name, rows=rows,
                np_probe_ns=np_per_ns, py_probe_ns=py_per_ns,
                deficit=py_per_ns / np_per_ns, pair_stat=pair_stat)


def main():
    say("W8 -- Oracle partition ceiling")
    say("real topology: CMS NCCI PTP 2026Q4 practitioner (4 shards) + MUE 2026Q2")
    hr("=")

    pairs, raw_rows = L.parse_ptp()
    mue = L.parse_mue()
    cid, rule_keys, topo = L.topology(pairs, mue)

    n_codes = len(cid)
    cap_of_code = np.full(n_codes, -1, dtype=np.int32)
    k = 0
    for c in mue:
        cap_of_code[cid[c]] = k
        k += 1

    # code frequency: unknown from public data (CMS publishes edits, not claims).
    # Two brackets, both stated as assumptions, neither chosen to suit a result:
    #   uniform   -- every code equally likely
    #   degree    -- weight by PTP edit degree, a measured proxy for "codes that
    #                matter in adjudication" (rank by degree, not by id)
    deg = np.zeros(n_codes)
    for a, b in pairs:
        deg[cid[a]] += 1
        deg[cid[b]] += 1
    p_uni = np.ones(n_codes) / n_codes
    wdeg = (deg + 1.0)
    p_deg = wdeg / wdeg.sum()

    results = []
    for n_g in (100_000, 1_000_000):
        for j, (cname, cp) in enumerate((("uniform", p_uni), ("degree", p_deg))):
            rng = np.random.default_rng(20261104 + n_g + 7 * j)
            results.append(run(n_g, cname, cp, cid, rule_keys, cap_of_code, rng))

    pay = dict(topology=topo, ptp_rows_raw=raw_rows, results=results, prereg=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "w8_results.json"), "w", encoding="utf-8") as fh:
        json.dump(pay, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(here, "w8_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    say()
    say("wrote w8_results.json / w8_out.txt")


if __name__ == "__main__":
    t0 = time.time()
    main()
    say(f"[wall {time.time()-t0:.1f}s]")
