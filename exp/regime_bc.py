#!/usr/bin/env python
"""Regimes (b) and (c), measured on the real threshold distribution.

The problem
-----------
Every threshold in the published NCCI tables is a literal, so no rule in them
has a denominator and regimes (b)/(c) are empty *in the table*.  They are not
empty in the domain: the NCCI Policy Manual, Ch.1 Sec.V, states that an MUE is
determined using "submitted claims data (100%) from a six month period", i.e.
the literal in the table is a frozen statistic of the whole claims table.  The
table stores regime (a); the rule that produced it is regime (c), or (b) if the
derivation is per adjudication class.

What this script measures
-------------------------
1. audit      -- how many threshold fields in the corpus are literals (all of
                 them); the check that makes "empty in the table" a fact.
2. stream     -- the real revision stream: 6 published quarters, so the set of
                 codes whose cap moved between consecutive quarters is observed,
                 not simulated.  Its size and shape are the ground truth a band
                 model has to predict.
3. forms      -- the two denominator forms applied to the same real population:
                 global (c) and per-MAI-tier (b).  Band = codes strictly above
                 cut(p) and at most cut(p+eps), with cut the *observed* value
                 (no interpolation: the density form's smoothness is exactly
                 what real discrete thresholds do not have).
4. records    -- the band weighted by each code's real PTP degree, i.e. |rho| in
                 claims rather than in codes.

Output exp/regime_bc.json
"""
import io
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get("CERTIS_DATA") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MUE_DIR = os.path.join(_DATA, "mue")
RAW = os.path.join(_DATA, "raw")
OUT = os.path.join(HERE, "regime_bc.json")


def read_mue(path):
    z = zipfile.ZipFile(path)
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csvs:
        return None
    text = z.open(csvs[0]).read().decode("cp1252", "replace")
    rows, started = {}, False
    for line in text.splitlines():
        if not started:
            if "CPT Code" in line and "MUE" in line:
                started = True
            continue
        f = [x.strip().strip('"') for x in line.split(",")]
        if len(f) < 3 or not f[0]:
            continue
        try:
            cap = int(float(f[1]))
        except ValueError:
            continue
        mai = f[2].split()[0] if f[2] else ""
        if mai in ("1", "2", "3"):
            rows[f[0]] = (cap, int(mai))
    return rows


def load_mue():
    out = {}
    for f in sorted(os.listdir(MUE_DIR)):
        if not f.endswith(".zip") or "delta" in f:
            continue
        rows = read_mue(os.path.join(MUE_DIR, f))
        if rows:
            out[f.replace("mue_", "").replace("_pra.zip", "")] = rows
    return out


def read_ptp():
    import glob
    deg = {}
    for p in sorted(glob.glob(os.path.join(RAW, "q2026q4_ccipra_f*.zip"))):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            if not n.lower().endswith(".txt"):
                continue
            with z.open(n) as fh:
                for line in io.TextIOWrapper(fh, encoding="cp1252", errors="replace"):
                    f0 = line.rstrip("\r\n").split("\t")
                    if len(f0) < 6:
                        continue
                    c1 = f0[0].strip()
                    if c1 and " " not in c1 and "/" not in c1:
                        deg[c1] = deg.get(c1, 0) + 1
    return deg


def cut(values_sorted, p):
    """Observed quantile: the largest observed value with CDF <= p."""
    i = int(np.floor(p * (len(values_sorted) - 1)))
    return values_sorted[max(0, min(i, len(values_sorted) - 1))]


def band(vals, p0, p1):
    """Codes whose cap is strictly above cut(p0) and at most cut(p1)."""
    v = np.sort(np.asarray(vals))
    c0, c1 = cut(v, p0), cut(v, p1)
    n = int(((v > c0) & (v <= c1)).sum())
    return n, int(c0), int(c1)


def main():
    mue = load_mue()
    quarters = sorted(mue)
    last = quarters[-1]
    rows = mue[last]
    codes = list(rows)
    caps = np.array([rows[c][0] for c in codes], dtype=np.int64)
    mais = np.array([rows[c][1] for c in codes], dtype=np.int64)
    ucap = np.unique(caps)

    res = {
        "quarter": last,
        "quarters": quarters,
        "n_codes": int(len(codes)),
        "cap_p50": int(np.median(caps)),
        "cap_p90": int(np.quantile(caps, 0.90)),
        "cap_p99": int(np.quantile(caps, 0.99)),
        "cap_max": int(caps.max()),
        "distinct_caps": int(len(ucap)),
        "mass_at_one": float((caps == 1).mean()),
        "mass_at_atoms_le_16": float((caps <= 16).mean()),
        "tier_size": {str(t): int((mais == t).sum()) for t in (1, 2, 3)},
        "tier_cap_p50": {str(t): int(np.median(caps[mais == t])) for t in (1, 2, 3)},
    }

    # ---- 1. the real revision stream between consecutive published quarters
    stream = []
    for a, b in zip(quarters, quarters[1:]):
        A, B = mue[a], mue[b]
        common = set(A) & set(B)
        changed = [c for c in common if A[c][0] != B[c][0]]
        d = np.array([B[c][0] - A[c][0] for c in changed], dtype=np.int64)
        up = int((d > 0).sum())
        stream.append({
            "from": a, "to": b,
            "codes_common": len(common),
            "added": len(set(B) - set(A)),
            "removed": len(set(A) - set(B)),
            "changed": len(changed),
            "changed_frac": len(changed) / max(len(common), 1),
            "increased": up, "decreased": int(len(d) - up),
            "median_abs_delta": float(np.median(np.abs(d))) if len(d) else 0.0,
            "max_abs_delta": int(np.abs(d).max()) if len(d) else 0,
            "at_cap_1_before": int(sum(1 for c in changed if A[c][0] == 1)),
        })
    res["revision_stream"] = stream

    # ---- 2. the two denominator forms on the real population
    grid = []
    for p in (0.50, 0.90, 0.95, 0.99):
        row = {"p": p}
        for eps in (0.01, 0.05):
            kg, c0, c1 = band(caps, p, p + eps)
            kt = sum(band(caps[mais == t], p, p + eps)[0] for t in (1, 2, 3))
            row["eps_%g" % eps] = {
                "k_global": kg, "cut": [c0, c1], "k_tier": kt,
                "ratio": (kt / kg) if kg else None,
                "predicted_ratio_if_tier_is_a_random_sample": 1 / 3,
            }
        grid.append(row)
    res["forms"] = grid

    # ---- 3. atom structure: is the band a function of eps, or a step?
    prof = []
    for p in np.round(np.arange(0.02, 0.995, 0.02), 3):
        c0 = cut(np.sort(caps), float(p))
        c1 = cut(np.sort(caps), float(p) + 0.01)
        k = int(((caps > c0) & (caps <= c1)).sum())
        prof.append({"p": float(p), "cut": int(c0), "next_cut": int(c1),
                     "k": k, "atom_mass": float((caps == c0).mean())})
    ks = [x["k"] for x in prof]
    res["atom_profile"] = {
        "points": prof,
        "zero_band_points": int(sum(1 for k in ks if k == 0)),
        "total_points": len(ks),
        "mean_k": float(np.mean(ks)),
        "max_k": int(max(ks)),
        "median_k": float(np.median(ks)),
        "eps_n_if_continuous": float(0.01 * len(caps)),
    }

    # ---- 4. the band in records, not codes
    deg = read_ptp()
    d = np.array([deg.get(c, 0) for c in codes], dtype=np.int64)
    total = int(d.sum())
    rw = []
    for p in (0.90, 0.95, 0.99):
        k, c0, c1 = band(caps, p, p + 0.01)
        sel = (caps > c0) & (caps <= c1)
        rec = int(d[sel].sum())
        live = int((d[sel] > 0).sum())
        rw.append({"p": p, "k_codes": k, "records": rec,
                   "band_codes_with_no_pair": k - live,
                   "share_codes": k / len(caps),
                   "share_records": rec / max(total, 1),
                   "over_representation": (rec / max(total, 1)) / max(k / len(caps), 1e-12)})
    res["record_weighting"] = {"n_records": total, "rows": rw}

    # ---- 5. audit: no threshold in the corpus is computed from the table
    import glob
    import re
    AGG = re.compile(r"(quant|percent|pct|%|avg|mean|median|sum|count|threshold|>|<|>=|<=)")
    scanned = literal_rows = aggregate_rows = 0
    for p in sorted(glob.glob(os.path.join(RAW, "q2026q4_ccipra_f*.zip"))) + \
            sorted(glob.glob(os.path.join(RAW, "q2026q4_ccioph_f*.zip"))):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            if not n.lower().endswith(".txt"):
                continue
            with z.open(n) as fh:
                for line in io.TextIOWrapper(fh, encoding="cp1252", errors="replace"):
                    f0 = line.rstrip("\r\n").split("\t")
                    if len(f0) < 6 or not f0[0].strip():
                        continue
                    if " " in f0[0].strip():
                        continue
                    scanned += 1
                    tail = "\t".join(f0[2:])
                    if AGG.search(tail):
                        aggregate_rows += 1
                    else:
                        literal_rows += 1
    res["audit"] = {
        "ptp_rows_scanned": scanned,
        "rows_whose_threshold_is_a_literal": literal_rows,
        "rows_whose_threshold_references_a_statistic": aggregate_rows,
        "mue_rows_scanned": int(len(codes)),
        "mue_rule_per_manual": "NCCI Policy Manual Ch.1 Sec.V: an MUE is "
        "determined using submitted claims data (100%) from a six month period, "
        "then published as a literal.",
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "atom_profile"}, indent=2)[:3000])


if __name__ == "__main__":
    main()
