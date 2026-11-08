#!/usr/bin/env python
"""Section V-D: the published delta, and the four things it does not determine.

The paper makes a claim about a *file format* -- that CMS's quarterly change
package reports the new state of a rule and not the transition -- and every part
of that claim is checkable against the package itself plus the two full tables
it sits between.  This script does the checking.

    A. counts        what each of the four PTP packs and the MUE pack report
    B. additions     the published additions against Q4 \\ Q3, both directions
    C. deletions     the 14 remain in the table, carrying Deletion Date 20260930
    D. CCMIChgs      of the 7 rows, how many are indicator flips and how many
                     are a reissue of an unchanged indicator (prose-only edits)
    E. clustering    removals are not spread over codes
    F. direction     the 31 MUE revisions: the file states the new threshold, the
                     full tables state which way it moved.  It is only in the diff.

Why B costs a full scan.  The interesting direction is the one the file cannot
report: a pair that *disappeared*.  Only a set difference over both seasons
answers that, so both tables are materialised.  Pairs are packed into int64
rather than kept as string tuples, which is what makes 2.3M x 2 affordable.

Output: exp/delta.json
"""
import collections
import glob
import io
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CERTIS_DATA") or os.path.join(os.path.dirname(HERE), "data")
RAW = os.path.join(DATA, "raw")
MUE = os.path.join(DATA, "mue")
OUT = os.path.join(HERE, "delta.json")

FULL = {
    ("2026Q3", "practitioner"): "q2026q3_ccipra_f*.zip",
    ("2026Q3", "hospital"): "medicare-ncci-2026q3-hospital-ptp-edits-*-f*.zip",
    ("2026Q4", "practitioner"): "q2026q4_ccipra_f*.zip",
    ("2026Q4", "hospital"): "q2026q4_ccioph_f*.zip",
}
PACK = {
    ("2026Q3", "practitioner"): "q3_pra_delta.zip",
    ("2026Q3", "hospital"): "q3_hosp_delta.zip",
    ("2026Q4", "practitioner"): "q4_pra_delta.zip",
    ("2026Q4", "hospital"): "q4_hosp_delta.zip",
}
MUE_PACK = "mue_2026q4_pra_delta.zip"


def recs(text, numeric=False):
    """(column1, column2, third field) per data row of a CMS change file.

    `numeric` is for the MUE packs, whose tab header is itself two fields
    ("HCPCS/CPT Code", "Practitioner Services MUE Values") and so parses as a
    data row unless the value is required to be an integer.
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().strip('"').lower().startswith("column"):
            start = i + 1
            break
    out = []
    for line in lines[start:]:
        if not line.strip():
            continue
        f = [x.strip().strip('"') for x in line.split("\t")]
        if len(f) >= 2 and f[0] and f[1]:
            if numeric:
                try:
                    int(f[1])
                except ValueError:
                    continue
            out.append((f[0], f[1], f[2] if len(f) > 2 else ""))
    return out


def load_pack(path, want, numeric=False):
    """The file inside a delta zip whose name contains `want`."""
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if want.lower() in name.lower() and name.lower().endswith(".txt"):
                return recs(z.read(name).decode("cp1252", "replace"), numeric)
    return []


class Interning:
    """code string -> dense int, shared across seasons so ids compare."""

    def __init__(self):
        self.d = {}

    def __call__(self, s):
        v = self.d.get(s)
        if v is None:
            v = len(self.d)
            self.d[s] = v
        return v


def full_table(paths, ids, keep=None):
    """Packed (c1, c2) ids for a season; optionally (deletion date, modifier) too."""
    packed, meta = [], {}
    for p in paths:
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                with z.open(name) as fh:
                    for line in io.TextIOWrapper(fh, encoding="cp1252", errors="replace"):
                        f = line.rstrip("\r\n").split("\t")
                        if len(f) < 6:
                            continue
                        c1, c2 = f[0].strip(), f[1].strip()
                        if not c1 or not c2 or " " in c1 or "/" in c1:
                            continue
                        key = ids(c1) * 1000000 + ids(c2)
                        packed.append(key)
                        if keep is not None and (c1, c2) in keep:
                            meta[(c1, c2)] = (f[4].strip(), f[5].strip())
    return np.unique(np.asarray(packed, dtype=np.int64)), meta


def mue_table(path):
    with zipfile.ZipFile(path) as z:
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
            rows[f[0]] = int(float(f[1]))
        except ValueError:
            continue
    return rows


def main():
    ids = Interning()
    res = {}

    # ---- A. what the four packs report -----------------------------------
    counts = {}
    for key, pack in PACK.items():
        p = os.path.join(RAW, pack)
        a = load_pack(p, "Additions")
        d = load_pack(p, "Deletions")
        c = load_pack(p, "CCMIChgs")
        counts["%s %s" % key] = {
            "additions": len(a), "deletions": len(d), "ccmichgs": len(c)}
    res["delta_counts"] = counts

    # ---- B. additions against the reconstruction, and the disappeared -----
    recon = {}
    for setting in ("practitioner", "hospital"):
        q3p = sorted(glob.glob(os.path.join(RAW, FULL[("2026Q3", setting)])))
        q4p = sorted(glob.glob(os.path.join(RAW, FULL[("2026Q4", setting)])))
        if not q3p or not q4p:
            print("missing shards for", setting)
            continue
        add = set((x[0], x[1]) for x in
                  load_pack(os.path.join(RAW, PACK[("2026Q4", setting)]), "Additions"))
        del4 = set((x[0], x[1]) for x in
                   load_pack(os.path.join(RAW, PACK[("2026Q4", setting)]), "Deletions"))
        cc = load_pack(os.path.join(RAW, PACK[("2026Q4", setting)]), "CCMIChgs")
        cc_pairs = set((x[0], x[1]) for x in cc)
        keep = add | del4 | cc_pairs
        k3, old_meta = full_table(q3p, ids, cc_pairs)   # (deletion, modifier) in Q3
        k4, meta = full_table(q4p, ids, keep)

        add_ids = set(ids(c1) * 1000000 + ids(c2) for c1, c2 in add)
        new = np.setdiff1d(k4, k3)          # in Q4, not in Q3
        gone = np.setdiff1d(k3, k4)         # in Q3, not in Q4
        add_arr = np.fromiter(add_ids, dtype=np.int64, count=len(add_ids))
        published_and_new = int(np.intersect1d(add_arr, new).size)

        dated = sum(1 for c in del4
                    if c in meta and meta[c][0].replace("/", "") == "20260930")
        recon[setting] = {
            "additions_published": len(add),
            "additions_also_in_q3": int(len(add_ids) - published_and_new),
            "reconstructed_new_pairs": int(new.size),
            "published_not_reconstructed": int(len(add_ids) - published_and_new),
            "disappeared_pairs": int(gone.size),
            "pairs_q3": int(k3.size), "pairs_q4": int(k4.size),
            "deletions_published": len(del4),
            "deletions_present_in_q4_with_end_date": dated,
        }
        print("%-13s add %d/%d new, gone %d, q3 %d q4 %d, del %d/%d dated"
              % (setting, published_and_new, len(add), gone.size,
                 k3.size, k4.size, dated, len(del4)))

        # ---- D. CCMIChgs: flip or reissue? --------------------------------
        # the Q3 modifier for these seven pairs came out of the pass above, so
        # "did the indicator change" is answered without a second scan
        old = dict((k, v[1]) for k, v in old_meta.items())
        flips = [x for x in cc if old.get((x[0], x[1]), None) not in (None, x[2])]
        same = [x for x in cc if old.get((x[0], x[1]), None) == x[2]]
        missing = len(cc) - len(flips) - len(same)
        recon[setting]["ccmichgs"] = {
            "rows": len(cc), "flipped": len(flips), "unchanged": len(same),
            "not_found_in_q3": missing,
            "unchanged_values": sorted(set(x[2] for x in same)),
            "false_positive_rate": round(len(same) / max(len(cc), 1), 4),
        }

    res["reconstruction"] = recon

    # ---- E. removals cluster ---------------------------------------------
    q3del = [x[1] for x in
             load_pack(os.path.join(RAW, PACK[("2026Q3", "practitioner")]), "Deletions")]
    q4del = [x[1] for x in
             load_pack(os.path.join(RAW, PACK[("2026Q4", "practitioner")]), "Deletions")]
    c3, c4 = collections.Counter(q3del), collections.Counter(q4del)
    res["clustering"] = {
        "q3_removals": len(q3del),
        "q3_top_column_two": c3.most_common(1)[0][0],
        "q3_top_count": c3.most_common(1)[0][1],
        "q3_top_share": round(c3.most_common(1)[0][1] / max(len(q3del), 1), 4),
        "q3_distinct_column_two": len(c3),
        "q4_removals": len(q4del),
        "q4_top_column_two": c4.most_common(1)[0][0],
        "q4_top_count": c4.most_common(1)[0][1],
        "q4_distinct_column_two": len(c4),
    }

    # ---- F. direction is not in the file ---------------------------------
    rv = load_pack(os.path.join(MUE, MUE_PACK), "Revisions", True)
    ad = load_pack(os.path.join(MUE, MUE_PACK), "Additions", True)
    de = load_pack(os.path.join(MUE, MUE_PACK), "Deletions", True)
    older = mue_table(os.path.join(MUE, "mue_2026q3_pra.zip"))
    newer = mue_table(os.path.join(MUE, "mue_2026q4_pra.zip"))
    moved = {}
    if older and newer:
        for code in set(older) & set(newer):
            if older[code] != newer[code]:
                moved[code] = newer[code] - older[code]
    delta_names = set(x[0] for x in rv)
    res["mue_delta"] = {
        "revisions": len(rv), "additions": len(ad), "deletions": len(de),
        "reconstructed_moved": len(moved),
        "reconstructed_increases": sum(1 for v in moved.values() if v > 0),
        "reconstructed_decreases": sum(1 for v in moved.values() if v < 0),
        "revision_codes_matched_by_file": len(delta_names & set(moved)),
        "file_states_direction": False,
        "note": "the Revisions file carries the new threshold only; "
                "the sign of every move above comes from the two full tables",
    }

    # ---- what the paper prints, asserted rather than eyeballed -------------
    assert counts["2026Q3 practitioner"]["deletions"] == 708, counts
    assert counts["2026Q4 practitioner"]["additions"] == 4249, counts
    assert counts["2026Q4 hospital"]["additions"] == 4146, counts
    for setting in ("practitioner", "hospital"):
        r = recon[setting]
        assert r["published_not_reconstructed"] == 0, r
        assert r["disappeared_pairs"] == 0, r
        assert r["deletions_present_in_q4_with_end_date"] == r["deletions_published"], r
        assert r["ccmichgs"]["flipped"] + r["ccmichgs"]["unchanged"] == 7, r
    assert res["clustering"]["q3_top_count"] == 155, res["clustering"]
    assert res["mue_delta"]["revision_codes_matched_by_file"] == \
        res["mue_delta"]["reconstructed_moved"] == 31, res["mue_delta"]
    assert res["mue_delta"]["reconstructed_decreases"] == 0, res["mue_delta"]

    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
