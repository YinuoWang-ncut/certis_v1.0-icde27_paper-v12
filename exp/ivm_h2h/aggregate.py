"""Fold the three harness runs into the table the paper prints.

The Rust harness (src/main.rs) emits one JSON per run: the raw per-repetition
timings plus that run's own summary.  The table in section VI is the *median
across runs* of those per-run figures, so a single slow process cannot move it.
This script performs exactly that reduction; it measures nothing, so it is
deterministic given the three files.

  python exp/ivm_h2h/aggregate.py

Inputs   raw_run1.json raw_run2.json raw_run3.json instance.json
Output   ivm_h2h.json

The ratios at the bottom mix in three figures that are not from this harness --
the retired DuckDB full recompute (18.282 ms), path R on this same instance
(19.7 us) and path R at the corpus point (14.9 us).  They are named here rather
than buried, because a reader should be able to see which side of a ratio is a
measurement of this run and which is a constant carried over from section V.

The column the paper prints is `vs_duckdb`, and its denominator is the one
carried-over figure that is itself a measurement on this instance: the DuckDB
full recompute of exp/ivm.json.  Nothing in that column is a prediction and
nothing in it crosses an instance.  The `*_over_pathR` entries are kept for
diagnosis only: path R was not timed here, and the paper no longer divides by
an estimate of it.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [1, 2, 3]

# carried over from the section V accounting, not measured by this harness
DUCKDB_FULL_MS = 18.282         # exp/ivm.json, the retired full recompute
PATH_R_CORPUS_US = 19.2         # path R at the corpus point, Table V, 394 rows
SUMMARY_PROBE_US = 0.31         # exp/cheap.py, partition summary


def load(i):
    return json.load(open(os.path.join(HERE, "raw_run%d.json" % i), encoding="utf-8"))


def med(v):
    return float(np.median(np.asarray(v, float)))


def main():
    R = [load(i) for i in RUNS]
    inst = json.load(open(os.path.join(HERE, "instance.json"), encoding="utf-8"))
    S = [r["summary"] for r in R]

    # per-run figure -> median across runs
    pick = lambda k: med([s[k] for s in S])
    m = {
        "dd_deliver_p50_us": pick("dd_deliver_p50_us"),
        "dd_deliver_mean_us": pick("dd_deliver_mean_us"),
        "dd_certify_p50_us": pick("dd_certify_p50_us"),
        "dd_certify_mean_us": pick("dd_certify_mean_us"),
        "dd_floor_p50_us": pick("dd_floor_p50_us"),
        "dd_floor_mean_us": pick("dd_floor_mean_us"),
        # the hand-written arm: each run already averaged its own repetitions,
        # so the reducer takes the median of those per-run means
        "hand_deliver_ns": pick("hand_deliver_mean_ns"),
        "hand_certify_ns": pick("hand_certify_mean_ns"),
        "hand_scope_recompute_ns": pick("hand_scope_mean_ns"),
        "hand_full_scan_ms": pick("hand_full_scan_ms"),
        # rebuild is [fresh, differential]; the fresh one is the conservative side
        "dd_rebuild_ms": med([r["summary"]["rebuild_ms"][0] for r in R]),
    }

    # per-band-size delivery, for the figure that shows no trend in k.
    # The band is measured every other delivery rep (15 per multiplier against
    # 30 deliveries), so the stride comes from dd_band_k, not from a guess.
    muls = R[0]["instance"]["multipliers"]
    stride = len(R[0]["dd_band_k"]) // len(muls)
    per_k = {}
    for j, mul in enumerate(muls):
        k = inst["bands"]["up%.2f" % mul]
        v = []
        for r in R:
            v += r["dd_deliver_us"][j * stride:(j + 1) * stride]
        per_k[str(k)] = {"n": len(v), "mean": round(float(np.mean(v)), 1),
                         "p50": round(float(np.median(v)), 1)}

    out = {
        "instance": R[0]["instance"],
        "runs": len(R),
        "corrections_per_run": R[0]["instance"]["reps_per_point"] * len(muls),
        # rounding is per quantity: the scan and the rebuild are printed whole,
        # the hand-written arm to the nanosecond it was measured at
        "medians": dict(
            [(k, round(m[k], 1)) for k in
             ("dd_deliver_p50_us", "dd_deliver_mean_us", "dd_certify_p50_us",
              "dd_certify_mean_us", "dd_floor_p50_us", "dd_floor_mean_us")]
            + [("hand_deliver_ns", int(round(m["hand_deliver_ns"]))),
               ("hand_certify_ns", int(round(m["hand_certify_ns"]))),
               ("hand_scope_recompute_ns", int(round(m["hand_scope_recompute_ns"]))),
               ("hand_full_scan_ms", round(m["hand_full_scan_ms"], 2)),
               ("dd_rebuild_ms", round(m["dd_rebuild_ms"]))]),
        # the column the paper prints: every ratio divides by the DuckDB arm
        # measured on this same instance, so the denominator is a measurement
        "vs_duckdb": {
            "denom_us": round(DUCKDB_FULL_MS * 1e3, 0),
            "rebuild": round(m["dd_rebuild_ms"] * 1e3 / (DUCKDB_FULL_MS * 1e3), 1),
            "deliver": round(m["dd_deliver_p50_us"] / (DUCKDB_FULL_MS * 1e3), 4),
            "certify": round(m["dd_certify_p50_us"] / (DUCKDB_FULL_MS * 1e3), 4),
            "floor": round(m["dd_floor_p50_us"] / (DUCKDB_FULL_MS * 1e3), 5),
        },
        # every "vs path R" cell in the table is one of these, and path R is
        # the corpus point (19.2 us), not this instance's 436-row scope
        "v8_retired": {
            "duckdb_full_ms": DUCKDB_FULL_MS,
            "path_r_corpus_us": PATH_R_CORPUS_US,
            "ratio": round(DUCKDB_FULL_MS * 1e3 / PATH_R_CORPUS_US),
        },
        "ratios": {
            "deliver_over_retired": round((DUCKDB_FULL_MS * 1e3) /
                                          m["dd_deliver_p50_us"], 1),
            "rebuild_over_deliver": round(m["dd_rebuild_ms"] * 1e3 /
                                          m["dd_deliver_p50_us"]),
            "certify_over_deliver": round(m["dd_certify_p50_us"] /
                                          m["dd_deliver_p50_us"], 3),
            "deliver_over_pathR": round(m["dd_deliver_p50_us"] /
                                        PATH_R_CORPUS_US, 2),
            "certify_over_pathR": round(m["dd_certify_p50_us"] /
                                        PATH_R_CORPUS_US, 2),
            "floor_over_pathR": round(m["dd_floor_p50_us"] / PATH_R_CORPUS_US, 2),
            "hand_scope_over_pathR": round(m["hand_scope_recompute_ns"] /
                                           (PATH_R_CORPUS_US * 1e3), 4),
            "hand_deliver_over_pathR": round(m["hand_deliver_ns"] /
                                             (PATH_R_CORPUS_US * 1e3), 4),
            "hand_certify_over_pathR": round(m["hand_certify_ns"] /
                                             (PATH_R_CORPUS_US * 1e3), 4),
            "rebuild_over_pathR": round(m["dd_rebuild_ms"] * 1e3 /
                                        PATH_R_CORPUS_US),
            "certify_over_summary_031us": round(m["dd_certify_p50_us"] /
                                                SUMMARY_PROBE_US),
            "hand_certify_over_deliver": round(m["hand_certify_ns"] /
                                               m["hand_deliver_ns"], 2),
        },
        "per_k_deliver_us": per_k,
        "source": "exp/ivm_h2h/src/main.rs (differential-dataflow 0.21.2/"
                  "timely 0.28, Rust 1.99.1, one worker); exp/ivm_h2h/mkdata.py "
                  "regenerates the instance from exp/ivm.py's seed",
    }

    p = os.path.join(HERE, "ivm_h2h.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("run   deliver p50   certify p50   floor p50   rebuild ms")
    for s in S:
        print("      %6.1f us     %6.1f us    %5.1f us   %8.0f"
              % (s["dd_deliver_p50_us"], s["dd_certify_p50_us"],
                 s["dd_floor_p50_us"], s["summary"]["rebuild_ms"][0]
                 if "summary" in s else s["rebuild_ms"][0]))
    print()
    print("median of runs: deliver %.1f us, certify %.1f us, floor %.1f us, "
          "rebuild %.0f ms" % (m["dd_deliver_p50_us"], m["dd_certify_p50_us"],
                               m["dd_floor_p50_us"], m["dd_rebuild_ms"]))
    print("deliver/retired %.1fx, rebuild/deliver %.0fx, certify/deliver %.2f"
          % (out["ratios"]["deliver_over_retired"],
             out["ratios"]["rebuild_over_deliver"],
             out["ratios"]["certify_over_deliver"]))
    print("vs DuckDB %.0f us: rebuild %.1fx, deliver %.4fx, certify %.4fx, "
          "floor %.5fx" % (out["vs_duckdb"]["denom_us"],
                           out["vs_duckdb"]["rebuild"],
                           out["vs_duckdb"]["deliver"],
                           out["vs_duckdb"]["certify"],
                           out["vs_duckdb"]["floor"]))
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
