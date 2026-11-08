#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the propositions of the ICDE submission.

Every test is a *falsification attempt*, in the order the review raised them.
Each test has two halves:
  (a) the refutation -- the v7 statement is false, and here is the witness;
  (b) the repair      -- the v8 statement is true, checked by brute force.

A test prints PASS only if BOTH halves hold.  Run:
    python test_theorems.py            # summary
    python test_theorems.py -v         # per-case detail
"""
import itertools
import random
import sys

VERBOSE = "-v" in sys.argv
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name)
    if VERBOSE or not ok:
        for line in detail:
            print("      " + line)


# ----------------------------------------------------------------------------
# A minimal, honest implementation of the model.
#
# verdict:  F(t) = OR_r  e_r(t)
# firing:   f_r(t) = [ v_r(t) >= theta_r ]
# effective:e_r(t) = f_r(t) AND NOT( m(t) AND omega_r )
#           omega_r = 1 -> an override modifier can defeat the rule
#           omega_r = 0 -> absolute: the modifier cannot defeat it
# v_r      = amount for a local rule, sum over the entity window for a window rule
# ----------------------------------------------------------------------------

class Rule:
    def __init__(self, rid, theta, omega=0, family="local", code=None, W=0):
        self.rid, self.theta, self.omega = rid, theta, omega
        self.family, self.code, self.W = family, code, W

    def value(self, D, t):
        if self.family == "local":
            return t["amt"]
        # window: sum of amounts on the same entity in (date-W, date]
        s = 0
        for u in D:
            if u["ent"] == t["ent"] and t["date"] - self.W < u["date"] <= t["date"]:
                s += u["amt"]
        return s

    def fires(self, D, t):
        return self.value(D, t) >= self.theta

    def eff(self, D, t):
        if not self.fires(D, t):
            return False
        return not (t["mod"] and self.omega)


def verdict(D, t, rules):
    """as-of verdict: each rule is judged under its own theta (already versioned)."""
    return any(r.eff(D, t) for r in rules)


def support(D, t, rules):
    return sum(1 for r in rules if r.eff(D, t))


def flip_set(D0, D1, rules0, rules1):
    """brute force: the claims whose stored verdict changes."""
    idx = {t["id"]: t for t in D1}
    out = []
    for t0 in D0:
        t1 = idx[t0["id"]]
        if verdict(D0, t0, rules0) != verdict(D1, t1, rules1):
            out.append(t0["id"])
    return sorted(out)


# ----------------------------------------------------------------------------
# T1.  A single disjunct changing does not change the disjunction.
#     Witness from the review: r1 moves 900 -> 1000, judged value 950,
#     r2 always true.  r1 flips, F does not.
# ----------------------------------------------------------------------------

def t1_single_rule_is_not_disjunction():
    r1 = Rule("r1", 900)
    r2 = Rule("r2", 0)                     # theta 0 -> always fires
    rules0 = [r1, r2]
    D0 = [{"id": 1, "ent": 1, "date": 100, "amt": 950, "mod": 0}]

    # v7 claim: v_r in [min,max) => F flips.  Band is [900,1000), 950 is inside.
    v7_says_flip = 900 <= 950 < 1000
    # truth: r1 goes true->false, but r2 keeps F true.
    f0 = verdict(D0, D0[0], rules0)
    r1b = Rule("r1", 1000)
    f1 = verdict(D0, D0[0], [r1b, r2])
    refuted = v7_says_flip and (f0 == f1)

    # v8 claim: flip iff in band AND support level equals s* (=1 for a tightening).
    s = support(D0, D0[0], rules0)
    v8_says_flip = (900 <= 950 < 1000) and (s == 1)
    repaired = (not v8_says_flip) and (f0 == f1)

    report("T1  single-rule flip is masked by another disjunct", refuted and repaired, [
        "v7: v in [900,1000) -> flip ?  %s" % v7_says_flip,
        "truth: F before=%s after=%s (unchanged)" % (f0, f1),
        "support before = %d, s* = 1 -> v8 predicts flip ? %s" % (s, v8_says_flip),
    ])


# ----------------------------------------------------------------------------
# T2.  The repaired radius formula, on random instances.
#
#     Tightening (theta1 > theta0):  |rho| = |{ t : y in [t0,t1), support==1 }|
#     Loosening (theta1 < theta0):   |rho| = |{ t : y in [t1,t0), support==0 }|
#     Both with the override conjunction applied (mod and omega).
# ----------------------------------------------------------------------------

def t2_radius_with_support_multiplicity():
    rng = random.Random(20261105)
    bad = 0
    trials = 300
    for _ in range(trials):
        n = rng.randint(1, 40)
        D = [{"id": i, "ent": 0, "date": 100, "amt": rng.randint(0, 50),
              "mod": rng.randint(0, 1)} for i in range(n)]
        theta0 = rng.randint(1, 40)
        theta1 = theta0 + rng.choice([-7, -2, -1, 1, 2, 7])
        omega1 = rng.randint(0, 1)
        r0 = Rule("r1", theta0, omega=omega1)
        r1 = Rule("r1", theta1, omega=omega1)
        other = [Rule("r%d" % k, rng.randint(1, 45), omega=rng.randint(0, 1))
                 for k in range(rng.randint(0, 3))]
        rules0 = [r0] + other
        rules1 = [r1] + other

        truth = len(flip_set(D, D, rules0, rules1))

        lo, hi = min(theta0, theta1), max(theta0, theta1)
        s_star = 1 if theta1 > theta0 else 0
        pred = 0
        for t in D:
            if not (lo <= t["amt"] < hi):
                continue
            # the override status of r must permit the transition
            if t["mod"] and omega1:
                continue
            if support(D, t, rules0) == s_star:
                pred += 1
        if pred != truth:
            bad += 1
    report("T2  |rho| = |band AND support==s*| (300 random instances)",
           bad == 0, ["mismatches: %d / %d" % (bad, trials)])


# ----------------------------------------------------------------------------
# T3.  A date correction moves the window's coverage to the NEW date as well.
#     Witness: W = 10, the record moves 100 -> 0.  A claim dated 5 gains the
#     record in its window, but 5 is not in [100,110).
# ----------------------------------------------------------------------------

def t3_date_window_union():
    W = 10
    rec = {"id": 0, "ent": 1, "date": 100, "amt": 50, "mod": 0}
    nb = {"id": 1, "ent": 1, "date": 5, "amt": 10, "mod": 0}
    D0 = [rec, nb]
    # rule: window sum <=? threshold 20; nb was alone (10), now 10+50=60
    r0 = Rule("w", 20, family="window", W=W)
    moved = dict(rec); moved["date"] = 0
    D1 = [moved, nb]

    f_before = verdict(D0, nb, [r0])
    f_after = verdict(D1, nb, [r0])
    affected = f_before != f_after

    old_bound = (100 <= nb["date"] < 110)          # v7: [d, d+W)
    new_bound = (100 <= nb["date"] < 110) or (0 <= nb["date"] < 10)

    report("T3  date correction: the window neighbourhood is the union of both dates",
           affected and (not old_bound) and new_bound, [
        "neighbour flips: %s (10 -> 60 against theta 20)" % affected,
        "v7 old-only bound [100,110) contains 5 ? %s" % old_bound,
        "v8 union bound contains 5 ? %s" % new_bound,
    ])


# ----------------------------------------------------------------------------
# T4.  Closed-form endpoints are not a closed-form cardinality.
#     Two datasets, identical interval [10,11), n = 2, identical mean 10.5,
#     cardinalities 0 and 2.
# ----------------------------------------------------------------------------

def t4_endpoints_are_not_count():
    a = [9.0, 12.0]
    b = [10.2, 10.8]
    lo, hi = 10.0, 11.0
    ka = sum(1 for x in a if lo <= x < hi)
    kb = sum(1 for x in b if lo <= x < hi)
    same = (len(a) == len(b)) and (sum(a) == sum(b))
    report("T4  identical endpoints, identical n and mean, different cardinality",
           same and ka != kb and (ka, kb) == (0, 2), [
        "A = %s -> k = %d ; B = %s -> k = %d" % (a, ka, b, kb),
        "both have n = 2 and mean 10.5",
    ])


# ----------------------------------------------------------------------------
# T5.  Monotonicity in |delta| needs a fixed direction.
#     Same |delta|, opposite direction: the bands are not nested, so the
#     counts are not comparable.
# ----------------------------------------------------------------------------

def t5_direction_breaks_nesting():
    ys = [10, 12, 14, 16, 18, 20]
    down = [y for y in ys if 8 <= y < 12]    # theta 12 -> 8,  loosening
    up = [y for y in ys if 12 <= y < 16]     # theta 12 -> 16, tightening
    nested = set(down) <= set(up) or set(up) <= set(down)
    report("T5  |rho| monotone in |delta| only at fixed direction",
           (not nested) and down == [10] and up == [12, 14], [
        "delta = -4 gives %s ; delta = +4 gives %s" % (down, up),
        "neither set contains the other, and the counts differ (1 vs 2)",
    ])


# ----------------------------------------------------------------------------
# T6.  A wide band can straddle several atoms, so the sup over placement is not
#     the largest single atom.
# ----------------------------------------------------------------------------

def t6_wide_band_cover_several_atoms():
    atoms = [(1000, 5), (1001, 3), (1002, 4), (1010, 1)]
    y = sum(m for _, m in atoms)
    # a band of width 3 (theta1 - theta0 = 2.5) can cover 1000..1002
    width = 2.5
    best = 0
    for a, _ in atoms:
        inside = sum(m for v, m in atoms if a <= v < a + width)
        best = max(best, inside)
    max_single = max(m for _, m in atoms)
    report("T6  sup band mass != max single atom mass when the band is wide",
           best == 12 and max_single == 5, [
        "band of width 2.5 best mass = %d ; largest atom = %d" % (best, max_single),
    ])


# ----------------------------------------------------------------------------
# T7.  The selector.  v7 says "5%" and "862 rows" as one switch; the fit says
#     the band wins iff k < kappa*(n - n0) with kappa = 11.3/25.6 and
#     n0 = 5100/11.3, so the floor n0, not the ratio, is binding at small n.
# ----------------------------------------------------------------------------

def t7_selector_floor_not_ratio():
    FR, FB = 14.8, 19.9            # microseconds, fitted fixed cost
    cR, cB = 11.3, 25.6            # nanoseconds per unit
    kappa = cR / cB
    n0 = (FB - FR) * 1000.0 / cR   # rows

    def tR(n): return FR + cR * n / 1000.0
    def tB(k): return FB + cB * k / 1000.0

    n, k = 100, 1                  # k/n = 1% < 5%, the v7 switch says "band"
    v7_says_band = (k / n) < 0.05
    truth_band = tB(k) < tR(n)
    repaired = (k < kappa * (n - n0))

    # the printed 862 is what the same criterion gives at a fixed k/n = 0.21
    rho = 0.21
    n_star = kappa * n0 / (kappa - rho)

    report("T7  the switch is a floor, not a 5% ratio", (
        v7_says_band and (not truth_band) and (not repaired)
        and abs(n_star - 861) < 2
        and abs(kappa - 0.4414) < 1e-3 and abs(n0 - 451) < 2
    ), [
        "kappa = %.4f ; n0 = %.1f rows" % (kappa, n0),
        "n=100,k=1: k/n=1%% < 5%% but tB=%.2fus > tR=%.2fus -> band loses"
        % (tB(k), tR(n)),
        "v8 criterion k < kappa*(n-n0) -> %.1f < %.1f ? %s" % (k, kappa * (n - n0), repaired),
        "at k/n = 0.21 the crossover is n = %.1f rows (v7 printed 862)" % n_star,
    ])


# ----------------------------------------------------------------------------
# T8.  void is not always a rebuild: SUM and COUNT decrement; the median cannot.
# ----------------------------------------------------------------------------

def t8_void_is_decomposable_for_sum_count():
    vals = [3, 7, 9, 11, 40]
    total, cnt = sum(vals), len(vals)
    after = [v for v in vals if v != 40]
    sum_ok = (total - 40) == sum(after)
    cnt_ok = (cnt - 1) == len(after)
    # median(5) = 9 ; median(4) = 8 -> a decrement is not available
    import statistics
    med_ok = statistics.median(vals) != statistics.median(after)
    report("T8  void: SUM/COUNT are decremented, a median is not",
           sum_ok and cnt_ok and med_ok, [
        "sum %d -> %d ; count %d -> %d" % (total, sum(after), cnt, len(after)),
        "median %s -> %s (no delta)" % (statistics.median(vals), statistics.median(after)),
    ])


# ----------------------------------------------------------------------------
# T9.  A value correction plus a threshold move defeats an index keyed on the
#     old value: the row is in the new band but stored under the old key.
# ----------------------------------------------------------------------------

def t9_stale_index_misses_value_move():
    rec = {"id": 0, "ent": 1, "date": 100, "amt": 0, "mod": 0}
    # correction 1: value 0 -> 100.  Under theta = 50 the verdict flips to violation.
    corrected = dict(rec, amt=100)
    f_after_value = verdict([corrected], corrected, [Rule("r", 50)])
    # correction 2: theta 50 -> 150.  The band is [50,150); 100 is inside, so the
    # verdict flips back.  An index still keyed at 0 cannot be found by searching
    # [50,150), and the system reports "no change".
    f_after_theta = verdict([corrected], corrected, [Rule("r", 150)])
    truth_flips = f_after_value != f_after_theta
    key_in_index = 0                       # the stale key the sort left behind
    index_finds_it = 50 <= key_in_index < 150
    report("T9  index keyed on the old value misses a value correction in the band",
           f_after_value and (not f_after_theta) and truth_flips
           and (not index_finds_it), [
        "after the value correction, F = %s (theta 50, value 100)" % f_after_value,
        "after the threshold move,  F = %s (theta 150, value 100)" % f_after_theta,
        "so the threshold move does flip this row: %s" % truth_flips,
        "an index keyed on the pre-correction value 0 finds it in [50,150): %s"
        % index_finds_it,
        "repair: maintain a delta index on the post-correction key",
    ])


# ----------------------------------------------------------------------------
# T10.  An indicator (override) flip changes the verdict with no threshold move.
# ----------------------------------------------------------------------------

def t10_override_flip_without_threshold_move():
    t = {"id": 0, "ent": 1, "date": 100, "amt": 1000, "mod": 1}
    D = [t]
    r_ovr = Rule("r", 500, omega=1)     # overridable: the modifier defeats it
    r_abs = Rule("r", 500, omega=0)     # absolute: the modifier does not
    f_ovr = verdict(D, t, [r_ovr])
    f_abs = verdict(D, t, [r_abs])
    report("T10 an MAI/indicator flip moves the verdict with theta fixed",
           (not f_ovr) and f_abs, [
        "omega=1 with a modifier present: F=%s" % f_ovr,
        "omega=0 with the same theta and the same amount: F=%s" % f_abs,
        "theta never moved, so a (predicate, theta) model cannot express this",
    ])


# ----------------------------------------------------------------------------
# T11.  Window bound with the union, checked by brute force on random data.
# ----------------------------------------------------------------------------

def t11_window_bound_union_is_sound():
    rng = random.Random(7)
    W = 8
    worst = 0
    for _ in range(120):
        n = rng.randint(2, 30)
        D = [{"id": i, "ent": rng.randint(0, 2), "date": rng.randint(0, 60),
              "amt": rng.randint(0, 20), "mod": 0} for i in range(n)]
        t0 = rng.randrange(n)
        D1 = [dict(x) for x in D]
        d0 = D1[t0]["date"]
        d1 = max(0, d0 + rng.choice([-20, -5, 5, 20]))
        D1[t0]["date"] = d1
        rules = [Rule("w%d" % j, rng.randint(20, 120), family="window", W=W)
                 for j in range(rng.randint(1, 3))]
        flipped = flip_set(D, D1, rules, rules)
        ent = D[t0]["ent"]
        lo, hi = min(d0, d1), max(d0, d1)
        # candidates: same entity, date in [d0,d0+W) union [d1,d1+W), plus the record
        cand = [x["id"] for x in D if x["ent"] == ent and (
            (d0 <= x["date"] < d0 + W) or (d1 <= x["date"] < d1 + W))]
        allowed = set(cand) | {D[t0]["id"]}
        if not set(flipped) <= allowed:
            worst += 1
    report("T11 the union bound covers every window flip (120 random cases)",
           worst == 0, ["violations: %d" % worst])


if __name__ == "__main__":
    t1_single_rule_is_not_disjunction()
    t2_radius_with_support_multiplicity()
    t3_date_window_union()
    t4_endpoints_are_not_count()
    t5_direction_breaks_nesting()
    t6_wide_band_cover_several_atoms()
    t7_selector_floor_not_ratio()
    t8_void_is_decomposable_for_sum_count()
    t9_stale_index_misses_value_move()
    t10_override_flip_without_threshold_move()
    t11_window_bound_union_is_sound()

    npass = sum(1 for _, ok in RESULTS if ok)
    print("\n%d / %d passed" % (npass, len(RESULTS)))
    sys.exit(0 if npass == len(RESULTS) else 1)
