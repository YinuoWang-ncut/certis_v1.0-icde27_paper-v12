"""W8 -- Oracle partition ceiling.

Question (fixed before measurement; see the PRE-REG dict in run_w8.py):
  given a clairvoyant oracle that knows exactly which verdicts flip, how much
  faster than full recomputation can it be?

Three execution strategies are timed, because the literal one is not falsifiable:

  ORACLE   clairvoyant: touch exactly the flipped verdicts          (W8 as written)
  BAND     probe the index, evaluate every member of the band       (best realisable)
  RULE     re-evaluate the whole scope of the corrected rule        (free from the
                                                                    correction itself)

and the baseline

  FULL     evaluate every (rule, group) pair in the instance

Rule topology and correction types come from the real CMS NCCI corpus
(w3_data: PTP full tables + MUE, six seasons). The claim instance is synthetic,
because CMS publishes edits and not claims; every constant used to build it is
either measured from the corpus or stated as an assumption in the output.

Nothing here is estimated that could be measured: |scope|, |band| and |A| are
exact counts on the instance, and every timings figure is a wall-clock
measurement on this machine.
"""
import os
import zipfile
import glob
import csv
import json
import time
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W3 = os.path.normpath(os.path.join(HERE, "..", "..", "2026-11-03-20-37-57", "w3_data"))
RAW = os.path.join(W3, "raw")

PTP_Q4 = ["q2026q4_ccipra_f1.zip", "q2026q4_ccipra_f2.zip",
          "q2026q4_ccipra_f3.zip", "q2026q4_ccipra_f4.zip"]

BIG = 1 << 20          # pair key stride; > max code id, so keys are unique
MAXK = 5               # max lines per claim we generate


# ---------------------------------------------------------------- real corpus

def parse_ptp(files=PTP_Q4, verbose=True):
    """Distinct (col1, col2) edit pairs from the real NCCI PTP full tables."""
    pairs = set()
    raw_rows = 0
    for fn in files:
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            continue
        with zipfile.ZipFile(p) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
            txt = z.open(name).read().decode("cp1252", "replace")
        lines = txt.splitlines()
        hdr = None
        for i, l in enumerate(lines[:80]):
            if "Column 1" in l and "Column 2" in l:
                hdr = i
                break
        if hdr is None:
            continue
        for l in lines[hdr + 1:]:
            f = l.split("\t")
            if len(f) < 2:
                continue
            a, b = f[0].strip(), f[1].strip()
            if a and b and a != "Column 1":
                raw_rows += 1
                pairs.add((a, b))
    if verbose:
        print(f"[ptp] {raw_rows} rows -> {len(pairs)} distinct pairs")
    return pairs, raw_rows


def parse_mue(season="mue_2026q2_pra"):
    """code -> (cap, adjudication indicator). Indicator 1 = line edit,
    2/3 = date-of-service edit: that is the scope class, and it is real."""
    p = os.path.join(W3, "mue", season)
    csvs = glob.glob(os.path.join(p, "*.csv"))
    if not csvs:
        return {}
    out = {}
    with open(csvs[0], encoding="cp1252", errors="replace") as fh:
        rows = list(csv.reader(fh))
    hdr = None
    for i, r in enumerate(rows[:30]):
        if r and "HCPCS" in r[0]:
            hdr = i
            break
    if hdr is None:
        return {}
    for r in rows[hdr + 1:]:
        if len(r) < 3:
            continue
        code = r[0].strip()
        if not code:
            continue
        try:
            cap = float(r[1].strip())
        except ValueError:
            continue
        ind = r[2].strip()[:1]
        out[code] = (cap, ind if ind in "123" else "?")
    return out


def topology(pairs, mue, verbose=True):
    """Rule-set topology: universe, degree, MUE scope split."""
    codes = set()
    for a, b in pairs:
        codes.add(a); codes.add(b)
    codes |= set(mue.keys())
    codes = sorted(codes)
    cid = {c: i for i, c in enumerate(codes)}

    keys = np.empty(len(pairs), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        x, y = cid[a], cid[b]
        if x > y:
            x, y = y, x
        keys[i] = x * BIG + y
    keys.sort()

    deg = np.zeros(len(codes), dtype=np.int64)
    for a, b in pairs:
        deg[cid[a]] += 1
        deg[cid[b]] += 1

    ind_split = {}
    caps = []
    for c, (cap, ind) in mue.items():
        ind_split[ind] = ind_split.get(ind, 0) + 1
        caps.append(cap)

    st = {
        "ptp_pairs": int(len(pairs)),
        "codes": int(len(codes)),
        "mue_rules": int(len(mue)),
        "mue_indicator_split": ind_split,
        "mue_cap_median": float(np.median(caps)) if caps else None,
        "mue_cap_p90": float(np.percentile(caps, 90)) if caps else None,
        "ptp_degree_median": int(np.median(deg)),
        "ptp_degree_p90": int(np.percentile(deg, 90)),
        "ptp_degree_max": int(deg.max()),
        "ptp_density": float(len(pairs) / (len(codes) * (len(codes) - 1) / 2)),
    }
    if verbose:
        print("[topo]", json.dumps(st, indent=None))
    return cid, keys, st


# ---------------------------------------------------------------- instance

def build_instance(n_g, kmax_p, code_p, cid, rng, claims_per_ent=3.0):
    """n_g claim lines-groups; each has k>=1 lines with distinct codes.

    code_p: probability over code ids (length = len(cid)). We draw a code
    distribution from the real universe because CMS publishes no claims; two
    variants (uniform / Zipf) are reported as a bracket.
    """
    n_codes = len(cid)
    cum = np.cumsum(code_p)
    cum /= cum[-1]

    ks = np.arange(1, len(kmax_p) + 1)
    k = rng.choice(ks, size=n_g, p=kmax_p).astype(np.int32)
    tot = int(k.sum())
    gid = np.repeat(np.arange(n_g, dtype=np.int32), k)

    u = rng.random(tot).astype(np.float64)
    code = np.searchsorted(cum, u).astype(np.int32)
    np.clip(code, 0, n_codes - 1, out=code)

    order = np.lexsort((code, gid))
    gid = gid[order]; code = code[order]
    keep = np.ones(tot, dtype=bool)
    keep[1:] = (gid[1:] != gid[:-1]) | (code[1:] != code[:-1])
    gid = gid[keep]; code = code[keep]
    n_l = len(gid)

    amt = np.exp(rng.normal(math.log(120.0), 1.05, n_l)).astype(np.float32)
    unit = (1 + rng.poisson(0.6, n_l)).astype(np.int16)
    ent = (gid // claims_per_ent).astype(np.int32)
    date = rng.integers(0, 365, n_l).astype(np.int32)
    lid = np.arange(n_l, dtype=np.int32)

    gptr = np.zeros(n_g + 1, dtype=np.int64)
    np.add.at(gptr, gid.astype(np.int64) + 1, 1)
    gptr = np.cumsum(gptr)

    # code -> (line ids) CSR, sorted by code then group
    o2 = np.lexsort((gid, code))
    c_ptr = np.zeros(n_codes + 1, dtype=np.int64)
    np.add.at(c_ptr, code[o2].astype(np.int64) + 1, 1)
    c_ptr = np.cumsum(c_ptr)
    c_line = o2.astype(np.int32)          # line id, grouped by code
    c_gid = gid[o2].astype(np.int32)

    return {
        "n_g": n_g, "n_l": n_l,
        "gptr": gptr, "gid": gid, "code": code, "amt": amt,
        "unit": unit, "ent": ent, "date": date,
        "c_ptr": c_ptr, "c_line": c_line, "c_gid": c_gid,
    }


def pair_offsets(inst):
    """(line_index, line_index) arrays for every within-group code pair.
    Vectorised: within a group the lines are contiguous, so offsets are fixed."""
    gptr, gid = inst["gptr"], inst["gid"]
    cnt = np.diff(gptr)
    out = []
    for d in range(1, MAXK):
        # line i and i+d belong to the same group iff i+d < gptr[gid[i]+1]
        idx = np.arange(len(gid) - d)
        same = (gid[idx + d] == gid[idx])
        out.append((idx[same], idx[same] + d))
    return out


def match_pairs(inst, rule_keys, offsets):
    """(rule_index, group, line) for every (pair rule, group) match: the PTP part
    of the full recompute set, obtained by a hash join, which is what any engine
    would do. The line id is returned so the caller pays a real gather per match
    instead of reading a pre-gathered array."""
    code = inst["code"]; gid = inst["gid"]
    rs = []
    gs = []
    ls = []
    for (i, j) in offsets:
        a = code[i].astype(np.int64)
        b = code[j].astype(np.int64)
        lo = np.minimum(a, b) * BIG + np.maximum(a, b)
        pos = np.searchsorted(rule_keys, lo)
        pos_c = np.clip(pos, 0, len(rule_keys) - 1)
        hit = rule_keys[pos_c] == lo
        if hit.any():
            rs.append(pos_c[hit].astype(np.int32))
            gs.append(gid[i][hit])
            ls.append(i[hit].astype(np.int32))
    if not rs:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32))
    return (np.concatenate(rs), np.concatenate(gs), np.concatenate(ls))


def match_mue(inst, cap_of_code):
    """(rule_index, group, line) for every (cap rule, group) match.
    The line id is returned so the caller can gather the real value instead of
    inventing one."""
    code = inst["code"]; gid = inst["gid"]
    r = cap_of_code[code]
    hit = r >= 0
    lid = np.arange(len(code), dtype=np.int32)
    return r[hit].astype(np.int32), gid[hit], lid[hit]


# ---------------------------------------------------------------- timing

def timeit(fn, target=0.012, repeat=5, warmup=1):
    """Median per-call wall time, with repetitions auto-scaled so each sample
    lasts >= target seconds. Without this, microsecond paths are measured at
    timer resolution and the ratios are noise."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    fn()
    base = max(time.perf_counter() - t0, 1e-9)
    m = int(target / base)
    m = max(1, min(m, 20000))

    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(m):
            fn()
        ts.append((time.perf_counter() - t0) / m)
    ts.sort()
    return ts[len(ts) // 2]


def probe_cost(keys_sorted, n_probe, rng, repeat=5):
    """ns per index probe on a sorted int64 index, batched like a band query."""
    q = rng.integers(0, int(keys_sorted[-1]), size=n_probe, dtype=np.int64)
    q.sort()

    def once():
        p = np.searchsorted(keys_sorted, q)
        np.clip(p, 0, len(keys_sorted) - 1, out=p)
        return keys_sorted[p] == q

    t = timeit(once, repeat=repeat)
    return t / n_probe * 1e9
