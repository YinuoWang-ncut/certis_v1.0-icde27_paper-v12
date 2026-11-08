"""W12 -- CERTIS end-to-end: the pipeline of Fig. 1, actually executed.

What this file is, and what it is not
-------------------------------------
W8 priced *strategies* on scenarios selected by hand.  W13 priced *engines* on
one query.  Neither runs a certificate: neither turns a correction stream into a
flip set and checks that flip set against full recomputation.  This module does.

The pipeline, in the order the code runs it:

    classify(corr)   rule identity -> (scope pointer, regime)      O(1)
    decide(corr)     |B|/|scope| against C_HAT                     one lookup
    execute(corr)    R: scan the scope slice, evaluate             streaming
                     B: bisect the (code,value) key, gather        random
                     P: refresh an aggregate, then a band          Theta(|D|)
    verify(batch)    flips == full recompute, exactly              offline

`verify` is *not* on the system path.  It is the experiment's ground truth, paid
once per batch and reported in its own column.  Folding it into system latency
would be the cheapest way to fake a bad result, so the driver keeps it out and
the report says which column is which.

Four constraints, all load-bearing, all of them things that would make the
numbers look better if violated:

1. A band is a *superset* of the flip set, never the answer.  That is only true
   if the predicate can fail inside the band, so the instance carries the
   corpus's override flag (63.8% of added pairs carry one, S5.1): a group with
   the flag set is not in violation however the threshold moves.  A band
   without such a filter would equal the flip set, and then "verify members of
   the band" would be a no-op -- every timing test would pass and the
   Delta-completeness assertion would be vacuous.

2. The judged quantity is quantised (Thm. 3).  A cap counts units, so moving it
   by one moves the band by a whole item's mass.  We keep a high-cap arm because
   that is the only place a narrow band exists on this axis; the report labels
   it a diagnostic, not the corpus's shape.

3. Aggregates are real and they are refreshed.  Regime (b) refreshes
   mean(value) over the tier, regime (c) over the table; both run on the data
   and both land in the latency distribution.  That is what prediction (ii) is
   about, and computing them as a constant would make (ii) untestable.

4. The decision is not free.  To route on |B|/|scope| the system must have that
   ratio; the first sighting of a rule pays one scope scan to learn it, and that
   scan is charged to the system.  A policy that learned for nothing would win
   every experiment it took part in.
"""
import time

import numpy as np

C_HAT = 0.05          # the paper's measured boundary; not tuned here
BIG = 1 << 20         # (code,unit) key stride; > max code id
AMT_STEP = 1 << 40    # (code,amt) key stride
OVERRIDE_P = 0.18     # measured: 18% of groups carry an overridable edit

# --------------------------------------------------------------------- veto
# A scope small enough that the band cannot win for any k >= 0 does not need a
# probe: |scope| is one array read (cptr[c+1]-cptr[c]) while k costs two
# bisections on the sorted key.  Two thresholds live below and answer two
# different questions -- which is why Sec. V-E prints both and why the guard
# must take the second:
#
#   N0_POST = p/kappa   where the band stops winning GIVEN that the probe is
#                       already paid (k + p < kappa n).  Paper: 451.
#   N0_PRE  = dp/kappa  where PROBING stops paying for itself, even in the best
#                       case that the probe would reveal k = 0 (k + dp <
#                       kappa n).  Paper: 717, same row as "dp in rows".
#
# The two differ because the band branch reuses the probe (see _band_cap_pair's
# `cand`), so the only thing a probe can buy is (tR - tB) - Q, whose maximum
# over k is F_B - F_R - Q + c_R n.  That is negative for n <= N0_PRE and
# positive above it: the guard is not a heuristic, it is the optimum of the
# model it is derived from.
KAPPA = 0.4414        # canonical, w12's 300 paired corrections
DP = 316.0            # candidates, = (F_B - F_R + Q) / c_B
P_POST = 199.0        # candidates, ex-post floor, p = dp - Q/c_B
N0_POST = 451         # p / kappa, rounded as the paper prints it
N0_PRE = 716          # dp / kappa = 8.1us / 11.3ns per row
# keep exp/veto_rule.py in step with these four numbers; veto600.py asserts it


# --------------------------------------------------------------------- store

class Store:
    """Two physical layouts over one instance, because the two paths need
    different ones: path R wants the scope contiguous (CSR by code), path B
    wants the (code, value) key sorted.  A store holding one and not the other
    would compare layouts rather than paths, so we build both and charge both
    to index build time.

    `unit` is the integer judged quantity for cap rules; `amt` the monetary one
    for relative and global rules.  Different axes, two indexes, as a real
    store needs.
    """

    def __init__(self, inst, override, build_index=True):
        self.inst = inst
        self.unit = inst["unit"]
        self.amt = inst["amt"]
        self.gid = inst["gid"]
        self.code = inst["code"]
        self.cptr = inst["c_ptr"]
        self.cline = inst["c_line"]
        self.n_l = int(inst["n_l"])
        self.n_g = int(inst["n_g"])
        self.override = override            # bool per group
        self.index_seconds = 0.0
        self.index_bytes = 0
        # Materialised, code-clustered copies of the judged quantities.  Path R
        # scans these contiguously; without them the scan is an indirect gather
        # through c_line, which costs an order of magnitude more and would make
        # the band look better than it is.  W8 draws the same distinction
        # (t_rule_s vs t_rule_r) and this is the clustered side of it.
        self.unit_c = self.unit[self.cline]
        self.amt_c = self.amt[self.cline]
        self.gid_c = self.gid[self.cline]
        if build_index:
            t0 = time.perf_counter()
            key_u = self.code.astype(np.int64) * BIG + self.unit.astype(np.int64)
            o = np.argsort(key_u, kind="stable")
            self.key_u = key_u[o].astype(np.int64)
            self.klid_u = o.astype(np.int32)
            self.key_u_raw = key_u
            self.index_bytes += self.key_u.nbytes + self.klid_u.nbytes

            a = np.round(self.amt.astype(np.float64) * 100.0).astype(np.int64)
            key_a = self.code.astype(np.int64) * AMT_STEP + (a + (1 << 39))
            oa = np.argsort(key_a, kind="stable")
            self.key_a = key_a
            self.key_a_sorted = key_a[oa]
            self.klid_a = oa.astype(np.int32)
            self.index_bytes += self.key_a_sorted.nbytes + self.klid_a.nbytes

            og = np.argsort(self.amt.astype(np.float64), kind="stable")
            self.amt_sorted = self.amt[og].astype(np.float32)
            self.amt_sorted_lid = og.astype(np.int32)
            self.index_bytes += self.amt_sorted.nbytes + self.amt_sorted_lid.nbytes
            self.index_seconds = time.perf_counter() - t0

    # --- path R's structure
    def scope(self, code):
        return self.cline[self.cptr[code]:self.cptr[code + 1]]

    def scope_len(self, code):
        return int(self.cptr[code + 1] - self.cptr[code])

    # --- path B's structure
    def band_u(self, code, lo_v, hi_v):
        """Line ids on this code with lo_v < unit <= hi_v.  Two bisections on a
        sorted key: the dependent random-access chain the paper says does not
        parallelise."""
        base = int(code) * BIG
        a = np.searchsorted(self.key_u, base + int(lo_v), side="right")
        b = np.searchsorted(self.key_u, base + int(hi_v), side="right")
        return self.klid_u[a:b]

    def band_a(self, code, lo_v, hi_v):
        """Band on the monetary axis.  The key is quantised to cents, so the
        bisections are widened by one unit at each end: a band that stops a
        half-cent short of its own threshold would drop a real flip, and a band
        is permitted to over-include (members are verified) but never to
        under-include.  Getting this wrong is how a Delta-completeness check
        fails one flip in a batch and looks like noise."""
        base = int(code) * AMT_STEP + (1 << 39)
        a = np.searchsorted(self.key_a_sorted,
                            base + int(round(lo_v * 100.0)) - 1, side="right")
        b = np.searchsorted(self.key_a_sorted,
                            base + int(round(hi_v * 100.0)) + 1, side="right")
        return self.klid_a[a:b]

    def band_a_global(self, lo_v, hi_v):
        """No code constraint: this is regime (c), whose band is a slice of the
        whole table -- S5.7's "70% of one pass".  Same one-unit widening."""
        a = np.searchsorted(self.amt_sorted, np.float32(lo_v), side="left")
        b = np.searchsorted(self.amt_sorted, np.float32(hi_v), side="right")
        return self.amt_sorted_lid[a:b]

    def mean_amt(self, code=None):
        if code is None:
            return float(self.amt.mean())
        lo, hi = self.cptr[code], self.cptr[code + 1]
        return float(self.amt_c[lo:hi].mean()) if hi > lo else 0.0


# --------------------------------------------------------------------- rules

class RuleSet:
    """The versioned rule store.  A correction carries a rule identity, which is
    what makes the scope free -- it never carries an old threshold, which is
    what makes delta-reconstruction necessary (S5.5).

    Regimes follow Table 1:
      a  point threshold     scope = one code's lines
      b  tier-relative       threshold = mean(tier) * k, scope = the tier
      c  global denominator  threshold = mean(table) * k, scope = the table
    """

    def __init__(self, cap_of_code, pair_codes, rel_k=1.0, glob_k=1.0):
        self.cap_of_code = cap_of_code
        self.pair_codes = pair_codes          # (c1, c2) code ids
        self.rel_k = rel_k
        self.glob_k = glob_k

    @staticmethod
    def regime(kind):
        return {"cap": "a", "pair": "a", "mai": "a",
                "rel": "b", "global": "c", "claim": "b"}[kind]


# ---------------------------------------------------------------- corrections

class Correction:
    """One row of a published delta file as the loader sees it: an identity, a
    kind, the new state.  The old state rides along because the *evaluator*
    needs a second state to diff; the *classifier* never touches it."""

    __slots__ = ("kind", "code", "c1", "c2", "old", "new", "delta", "gid")

    def __init__(self, kind, code=-1, c1=-1, c2=-1, old=None, new=None,
                 delta=0.0, gid=-1):
        self.kind = kind
        self.code = code
        self.c1 = c1
        self.c2 = c2
        self.old = old
        self.new = new
        self.delta = delta
        self.gid = gid

    def key(self):
        if self.kind == "pair":
            return ("pair", self.c1, self.c2)
        if self.kind in ("cap", "mai", "rel", "claim"):
            return (self.kind, self.code)
        return ("global",)

    def __repr__(self):
        return f"C({self.kind},code={self.code},old={self.old},new={self.new})"


# ------------------------------------------------------------------ pipeline

class Certifier:
    """The system: one instance, one rule set, a stream of corrections.

    Pre-registered policies (see run_w12.py):

      R-always   never build or consult the band index; scan the scope
      B-always   always probe the band
      CERTIS     |B|/|scope| < C_HAT -> B, else R, using the ratio this rule
                 showed the last time it was seen; first sighting pays one
                 scope scan, charged to the system
      CERTIS-crit  probe k, then k + DP < KAPPA*|scope| -> B.  The paper's
                 Eq. (criterion) executed as written; the probe is paid on
                 every cap correction
      CERTIS-veto  the same criterion behind a metadata guard: |scope| <=
                 N0_PRE -> R with no probe.  The only policy here that never
                 pays a probe it cannot recover
      ORACLE     clairvoyant routing: the cheaper path for this correction,
                 measured.  An upper bound on any policy, not a competitor.
    """

    def __init__(self, store, rules, policy="CERTIS", cost=None):
        self.s = store
        self.r = rules
        self.policy = policy
        self.cost = cost          # (aR, bR, aB, bB, fbar) for CERTIS2
        self.learned = {}
        self.stats = {"R": 0, "B": 0, "P": 0}
        self.counters = {"probe": 0, "scan": 0, "rows": 0, "refresh_rows": 0,
                         "veto": 0}

    # --- the predicate, shared by every path so none gets a cheaper one
    def _viol_unit(self, lids, cap):
        """Violation under a cap: judged quantity above the cap and the edit not
        overridden.  The second term is what makes a band a superset."""
        if len(lids) == 0:
            return np.empty(0, np.int32)
        v = self.s.unit[lids]
        m = v > cap
        if not m.any():
            return np.empty(0, np.int32)
        l = lids[m]
        return l[~self.s.override[self.s.gid[l]]]

    def _viol_amt(self, lids, tau):
        if len(lids) == 0:
            return np.empty(0, np.int32)
        v = self.s.amt[lids]
        m = v > tau
        if not m.any():
            return np.empty(0, np.int32)
        l = lids[m]
        return l[~self.s.override[self.s.gid[l]]]

    @staticmethod
    def _xor(a, b):
        if len(a) == 0:
            return b.astype(np.int32)
        if len(b) == 0:
            return a.astype(np.int32)
        return np.setxor1d(a, b).astype(np.int32)

    def _scan_cap_pair(self, code, cap_a, cap_b):
        """One contiguous pass over the materialised scope, both states judged.
        Two separate passes would be the easy way to write this and would
        inflate path R by 2x, which is exactly the sort of accident that makes
        an index look good."""
        lo, hi = self.s.cptr[code], self.s.cptr[code + 1]
        self.counters["scan"] += 1
        n = hi - lo
        self.counters["rows"] += n
        if n == 0:
            return np.empty(0, np.int32)
        v = self.s.unit_c[lo:hi]
        m = (v > cap_a) != (v > cap_b)
        if not m.any():
            return np.empty(0, np.int32)
        lid = self.s.cline[lo:hi][m]
        return lid[~self.s.override[self.s.gid[lid]]].astype(np.int32)

    def _band_cap_pair(self, code, cap_a, cap_b, cand=None):
        """Verify the band members.  `cand` is passed in when the decision
        already paid for the probe, so the decision and the execution share one
        query instead of two -- otherwise the policy is charged twice for
        knowing what it is about to do."""
        if cand is None:
            cand = self.s.band_u(code, int(min(cap_a, cap_b)), int(max(cap_a, cap_b)))
            self.counters["probe"] += 1
        self.counters["rows"] += len(cand)
        if len(cand) == 0:
            return np.empty(0, np.int32)
        v = self.s.unit[cand]
        m = (v > cap_a) != (v > cap_b)
        if not m.any():
            return np.empty(0, np.int32)
        l = cand[m]
        return l[~self.s.override[self.s.gid[l]]].astype(np.int32)

    def _pair_groups(self, c1, c2):
        """Groups holding both codes; a paired-code edit does not fire on an
        overridden claim, which is why the flag is filtered here and in the
        oracle identically."""
        l1, h1 = self.s.cptr[c1], self.s.cptr[c1 + 1]
        l2, h2 = self.s.cptr[c2], self.s.cptr[c2 + 1]
        self.counters["scan"] += 1
        self.counters["rows"] += (h1 - l1) + (h2 - l2)
        if h1 <= l1 or h2 <= l2:
            return np.empty(0, np.int32)
        g2 = np.sort(self.s.gid_c[l2:h2])
        g1 = self.s.gid_c[l1:h1]
        pos = np.searchsorted(g2, g1)
        pos_c = np.clip(pos, 0, len(g2) - 1)
        hit = g2[pos_c] == g1
        if not hit.any():
            return np.empty(0, np.int32)
        g = np.unique(g1[hit])
        return g[~self.s.override[g]].astype(np.int32)

    # --- paths
    def _path_R(self, corr):
        if corr.kind == "cap":
            return self._scan_cap_pair(corr.code, corr.old, corr.new)
        if corr.kind == "pair":
            return self._pair_groups(corr.c1, corr.c2)
        if corr.kind == "mai":
            s1 = self.s.scope(corr.code)
            self.counters["scan"] += 1
            self.counters["rows"] += len(s1)
            v = self.s.unit[s1]
            m = v > (corr.old if corr.old is not None else 0)
            if not m.any():
                return np.empty(0, np.int32)
            g = np.unique(self.s.gid[s1[m]])
            return g[~self.s.override[g]].astype(np.int32)
        if corr.kind == "rel":
            lo_c, hi_c = self.s.cptr[corr.code], self.s.cptr[corr.code + 1]
            self.counters["scan"] += 1
            mu = self.s.mean_amt(corr.code)
            self.counters["refresh_rows"] += hi_c - lo_c
            self.counters["rows"] += hi_c - lo_c
            if hi_c <= lo_c:
                return np.empty(0, np.int32)
            v = self.s.amt_c[lo_c:hi_c]
            m = (v > mu * corr.old) != (v > mu * corr.new)
            if not m.any():
                return np.empty(0, np.int32)
            l = self.s.cline[lo_c:hi_c][m]
            return l[~self.s.override[self.s.gid[l]]].astype(np.int32)
        if corr.kind == "global":
            mu = self.s.mean_amt(None)
            self.counters["refresh_rows"] += self.s.n_l
            self.counters["scan"] += 1
            self.counters["rows"] += self.s.n_l
            v = self.s.amt
            m = (v > mu * corr.old) != (v > mu * corr.new)
            if not m.any():
                return np.empty(0, np.int32)
            l = np.flatnonzero(m).astype(np.int32)
            return l[~self.s.override[self.s.gid[l]]].astype(np.int32)
        raise ValueError(corr.kind)

    def _path_B(self, corr, cand=None):
        if corr.kind == "cap":
            return self._band_cap_pair(corr.code, corr.old, corr.new, cand)
        if corr.kind == "pair":
            # |B| = |scope| for an added pair: every group holding both codes
            # flips, so no band can be narrower.  Priced as R when B is forced,
            # which is the honest cost of the losing side.
            return self._path_R(corr)
        raise ValueError(corr.kind)

    def _path_P(self, corr):
        """Refresh the aggregate, then take the band.  Theta(|scope|) for (b),
        Theta(|D|) for (c).  Prediction (ii) says this is what sets the p99."""
        if corr.kind == "rel":
            s = self.s.scope(corr.code)
            mu = self.s.mean_amt(corr.code)
            self.counters["refresh_rows"] += len(s)
            if len(s) == 0:
                return np.empty(0, np.int32)
            t_a, t_b = mu * corr.old, mu * corr.new
            cand = self.s.band_a(corr.code, min(t_a, t_b), max(t_a, t_b))
            self.counters["probe"] += 1
            self.counters["rows"] += len(cand)
            v = self.s.amt[cand]
            m = (v > t_a) != (v > t_b)
            if not m.any():
                return np.empty(0, np.int32)
            l = cand[m]
            return l[~self.s.override[self.s.gid[l]]].astype(np.int32)
        if corr.kind == "global":
            mu = self.s.mean_amt(None)
            self.counters["refresh_rows"] += self.s.n_l
            t_a, t_b = mu * corr.old, mu * corr.new
            cand = self.s.band_a_global(min(t_a, t_b), max(t_a, t_b))
            self.counters["probe"] += 1
            self.counters["rows"] += len(cand)
            v = self.s.amt[cand]
            m = (v > t_a) != (v > t_b)
            if not m.any():
                return np.empty(0, np.int32)
            l = cand[m]
            return l[~self.s.override[self.s.gid[l]]].astype(np.int32)
        raise ValueError(corr.kind)

    # --- the decision
    def _decide(self, corr):
        """Returns (path, cand, frac).  `cand` is the band the decision already
        paid for, so execution can reuse it.

        The ratio |B|/|scope| is *not* free, but it is cheap: two bisections on
        a sorted key give the band's cardinality without touching its members.
        That distinction -- counting the band is O(log n), gathering it is
        O(|B|) -- is the whole reason a policy can route on the ratio at all.
        """
        regime = self.r.regime(corr.kind)
        if regime in ("b", "c"):
            return "P", None, float("nan")
        if self.policy == "R-always":
            return "R", None, float("nan")
        if self.policy == "B-always":
            cand = self.s.band_u(corr.code, int(min(corr.old, corr.new)),
                                 int(max(corr.old, corr.new))) \
                if corr.kind == "cap" else None
            self.counters["probe"] += 1
            return "B", cand, float("nan")
        if corr.kind != "cap":
            # a pair rule's band is its scope: ratio 1.0, no query needed
            self.learned.setdefault(corr.key(), 1.0)
            return "R", None, 1.0
        # --- the criterion, with the veto sitting in front of the probe ------
        # Two policies, one difference: CERTIS-veto answers the metadata
        # question first (|scope| <= N0_PRE) and therefore pays no probe, while
        # CERTIS-crit probes every cap correction and would have been the
        # "router" the paper describes.  Keeping both is what lets the
        # comparison be "veto vs no veto" and not "veto vs a strawman".
        if self.policy in ("CERTIS-crit", "CERTIS-veto"):
            ns = self.s.scope_len(corr.code)
            if self.policy == "CERTIS-veto" and ns <= N0_PRE:
                self.counters["veto"] += 1
                return "R", None, float("nan")
            lo, hi = sorted((corr.old, corr.new))
            cand = self.s.band_u(corr.code, int(lo), int(hi))
            self.counters["probe"] += 1
            k_n = int(len(cand))
            frac = k_n / max(ns, 1)
            self.learned[corr.key()] = frac
            if k_n + DP < KAPPA * ns:
                return "B", cand, frac
            return "R", None, frac
        k = corr.key()
        if self.policy == "CERTIS2":
            # The two-parameter comparison S5.7 says should replace the constant,
            # but in the variable that is actually free: |scope| is O(1) to read
            # while |B| needs two bisections, so routing on |scope| costs nothing
            # and routing on |B| costs a probe.  The band's size is taken from
            # this rule's own history, falling back to the mix mean.
            aR, bR, aB, bB, fbar = self.cost
            ns = self.s.scope_len(corr.code)
            f = self.learned.get(k, fbar)
            if aB * (f * ns) + bB < aR * ns + bR:
                lo, hi = sorted((corr.old, corr.new))
                cand = self.s.band_u(corr.code, int(lo), int(hi))
                self.counters["probe"] += 1
                if ns > 0:
                    self.learned[k] = len(cand) / ns
                return "B", cand, f
            return "R", None, f
        if k in self.learned:
            frac = self.learned[k]
            if frac < C_HAT:
                cand = self.s.band_u(corr.code, int(min(corr.old, corr.new)),
                                     int(max(corr.old, corr.new)))
                self.counters["probe"] += 1
                return "B", cand, frac
            return "R", None, frac
        # first sighting: count the band (two bisections), remember the ratio
        lo, hi = sorted((corr.old, corr.new))
        ns = self.s.scope_len(corr.code)
        nb = int(np.searchsorted(self.s.key_u, int(corr.code) * BIG + int(hi), side="right")
                 - np.searchsorted(self.s.key_u, int(corr.code) * BIG + int(lo), side="right"))
        self.counters["probe"] += 1
        frac = nb / max(ns, 1)
        self.learned[k] = frac
        if frac < C_HAT:
            cand = self.s.band_u(corr.code, int(lo), int(hi))
            return "B", cand, frac
        return "R", None, frac

    def apply(self, corr, path=None):
        """One correction end to end.  Returns (flips, path, seconds)."""
        t0 = time.perf_counter()
        if path is None:
            chosen, cand, frac = self._decide(corr)
        else:
            chosen, cand, frac = path, None, float("nan")
        if chosen == "R":
            flips = self._path_R(corr)
        elif chosen == "B":
            flips = self._path_B(corr, cand)
        else:
            flips = self._path_P(corr)
        dt = time.perf_counter() - t0
        self.stats[chosen] += 1
        return flips, chosen, dt

    def reset_counters(self):
        for k in self.counters:
            self.counters[k] = 0


# ------------------------------------------------------- ground truth (verify)

def ground_truth(store, corr):
    """The flip set with the whole table scanned and every rule evaluated: what
    the system would cost if nothing were certifiable.  O(|D|) per correction
    and deliberately dumb -- it is the oracle, not an implementation."""
    s = store
    if corr.kind == "cap":
        lid = np.flatnonzero(s.code == corr.code).astype(np.int32)
        lo, hi = sorted((corr.old, corr.new))
        v = s.unit[lid]
        m = (v > lo) & (v <= hi)
        lid = lid[m]
        return np.sort(lid[~s.override[s.gid[lid]]]).astype(np.int32)
    if corr.kind == "pair":
        g1 = set(s.gid[s.code == corr.c1].tolist())
        g2 = set(s.gid[s.code == corr.c2].tolist())
        g = np.array(sorted(g1 & g2), dtype=np.int32)
        if len(g) == 0:
            return g
        return g[~s.override[g]].astype(np.int32)
    if corr.kind == "rel":
        mu = s.mean_amt(corr.code)
        lid = np.flatnonzero(s.code == corr.code).astype(np.int32)
        return _gt_amt(s, lid, mu * corr.old, mu * corr.new)
    if corr.kind == "global":
        mu = s.mean_amt(None)
        lid = np.arange(s.n_l, dtype=np.int32)
        return _gt_amt(s, lid, mu * corr.old, mu * corr.new)
    if corr.kind == "mai":
        lid = np.flatnonzero(s.code == corr.code).astype(np.int32)
        m = s.unit[lid] > (corr.old if corr.old is not None else 0)
        return np.unique(s.gid[lid[m]]).astype(np.int32)
    raise ValueError(corr.kind)


def _gt_amt(s, lid, t_old, t_new):
    lo, hi = sorted((t_old, t_new))
    v = s.amt[lid]
    m = (v > lo) & (v <= hi)
    lid = lid[m]
    return np.sort(lid[~s.override[s.gid[lid]]]).astype(np.int32)


def delta_complete(system_flips, truth):
    """Missed flips.  Must be exactly zero: a missed flip is a claim reported as
    unchanged, which looks exactly like a correct answer."""
    if len(truth) == 0:
        return 0, len(system_flips)
    if len(system_flips) == 0:
        return int(len(truth)), 0
    missed = np.setdiff1d(truth, system_flips)
    extra = np.setdiff1d(system_flips, truth)
    return int(len(missed)), int(len(extra))


# ------------------------------------------------------------------ generator

class Stream:
    """Generator v1: four axes under control, statistics taken from the corpus.

    Axes are the paper's (S5.1): |D|, |R|, pi_c, family mix.  What is *not*
    free is the correction mix -- it is set from the measured corpus (53:4
    up/down, median step 2x, 40/57 moving a small integer cap), because a stream
    we invented would make the end-to-end numbers a statement about the
    generator rather than about the rule set.
    """

    def __init__(self, store, rules, rng, cap_codes, pair_codes):
        self.s = store
        self.r = rules
        self.rng = rng
        self.cap_codes = np.asarray(cap_codes, dtype=np.int32)
        self.pair_codes = pair_codes

    def _pick_code(self, n_active):
        """Draw among the n_active most-scoped codes: that is what '10^3 live
        rules' means, and drawing from the whole code universe instead would
        make the rule count decoration."""
        k = int(min(n_active, len(self.cap_codes)))
        return int(self.cap_codes[int(self.rng.integers(0, k))])

    def cap_move(self, code, position="as-is", mode="rel"):
        s = self.s.scope(code)
        if len(s) < 4:
            return None
        v = self.s.unit[s]
        old = int(np.percentile(v, 90)) if position == "high" else int(np.median(v))
        old = max(old, 1)
        if mode == "abs":
            # fixed ABSOLUTE step: |delta| = 1 whatever the threshold is, which
            # is the arm Thm. 3's staircase is about
            new = old + 1
        else:
            if self.rng.random() < 0.93:                   # 53:4 up-biased
                step = float(self.rng.choice([1.0, 2.0, 2.0, 3.0]))
                new = max(old + 1, int(round(old * step)))
            else:
                new = max(1, old - max(1, int(round(old * 0.5))))
            if new == old:
                new = old + 1
        return Correction("cap", code=code, old=old, new=new, delta=new - old)

    def pair_add(self):
        i = int(self.rng.integers(0, len(self.pair_codes)))
        c1, c2 = self.pair_codes[i]
        return Correction("pair", c1=int(c1), c2=int(c2), old=0, new=1)

    def rel_move(self, code):
        k_old = float(self.r.rel_k)
        k_new = k_old * float(self.rng.choice([0.9, 1.1, 1.25]))
        return Correction("rel", code=code, old=k_old, new=k_new, delta=k_new - k_old)

    def global_move(self):
        k_old = float(self.r.glob_k)
        k_new = k_old * float(self.rng.choice([0.95, 1.05, 1.2]))
        return Correction("global", old=k_old, new=k_new, delta=k_new - k_old)

    def batch(self, n_rules, pi_c, mix, cap_mode="rel"):
        """int(pi_c * n_rules) corrections, codes drawn distinct within the batch
        so the batch's flip set is the union of its members' (a real delta file
        does not correct one rule twice in the same instant either)."""
        n = max(1, int(round(pi_c * n_rules)))
        used = set()
        out = []
        kinds = list(mix)
        probs = np.array([mix[k] for k in kinds], dtype=float)
        probs /= probs.sum()
        guard = 0
        while len(out) < n and guard < 8 * n:
            guard += 1
            kind = kinds[int(self.rng.choice(len(kinds), p=probs))]
            if kind in ("cap_low", "cap_high", "rel"):
                c = self._pick_code(n_rules)
                if c in used:
                    continue
                used.add(c)
                corr = self.cap_move(c, "high" if kind == "cap_high" else "as-is",
                                     mode=cap_mode) if kind.startswith("cap") \
                    else self.rel_move(c)
            elif kind == "pair":
                i = int(self.rng.integers(0, len(self.pair_codes)))
                if i in used:
                    continue
                used.add(i)
                corr = self.pair_add()
            else:
                corr = self.global_move()
            if corr is not None:
                out.append(corr)
        return out
