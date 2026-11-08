#!/usr/bin/env python
"""The boundary arithmetic, from the fitted constants to kappa and n_0.

The paper reports kappa and n_0 for two of the three accountings of Table VIII.
Those numbers are arithmetic over four fitted constants, and this file is where
the arithmetic lives, so that the table can be checked without re-running a
timing harness:

    T_R(n) = F_R + c_R n          recompute the changed rule's scope
    T_B(k) = F_B + c_B k          retrieve the band and mask it

    band wins  iff  k < kappa (n - n_0),
    kappa = c_R / c_B,   n_0 = (F_B - F_R) / c_R.

The constants come from two places and are labelled:

  fitted_full_paths  w12 end-to-end fit, the harness behind Table V: t_R = 11.3
                     ns*|sigma| + 14.8 us, t_B = 25.6 ns*|B| + 19.9 us, on 300
                     corrections that were run down *both* paths (600 path
                     timings).  This is the accounting the paper uses: both
                     paths are charged the verdict decision.
                     Its error bar is the across-run span [0.26, 0.46] from
                     seven independent re-runs of the same stream in fresh
                     processes (exp/run_duel_big.py, exp/kappa_runs.json), not
                     the bootstrap interval [0.37, 0.52] of
                     exp/bootstrap_kappa.py: resampling the corrections of one
                     run never sees the error that dominates, which is the
                     large-band gather moving 57% between runs.  The corpus
                     density 0.462 sits at the top of the span, which is why the
                     paper calls the residue at Delta_p = 0 a tie.

  microbenchmark     exp/cost_model.py, one process, one synthetic layout, the
                     bare gather fitted against the band path.  Reproduced by
                     running that script; the numbers are read from its json
                     when it is present and otherwise from the copy below.

The third row of Table VIII (the bare-layout ratio) is a ratio of two per-element
costs with no fixed term, and is reported by cost_model.py as `v7_ratio`.

Output: exp/boundary.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "boundary.json")

FITS = {
    "fitted_full_paths": {
        "c_R_ns": 11.3, "c_B_ns": 25.6,
        "F_R_ns": 14.8e3, "F_B_ns": 19.9e3,
        # Two error bars, and they are not the same quantity.  The bootstrap
        # [0.37, 0.52] resamples the 300 corrections of one run: it measures
        # how well those scopes pin two slopes down, and it is 2.5x too narrow
        # because the scatter it does not see -- the large-|B| gather moving
        # 57% between processes -- is larger than the scatter it does see.
        # The span [0.26, 0.46] re-measures the whole stream seven times in
        # fresh processes at 3000 corrections each: that is the error a reader
        # on another afternoon would meet.  The paper prints the span.
        "kappa_ci": [0.37, 0.52],
        "kappa_span_runs": [0.2636, 0.4639],
        "runs": {
            "n": 7, "corrections_per_run": 3000,
            "kappa_median": 0.4174,
            "allowance_rows_at_dp0": [-78.1, 0.8],
            "allowance_rows_at_measured_dp": [-257, -193],
            "source": "exp/run_duel_big.py + exp/kappa_runs.json",
        },
        "source": "w12 end-to-end fit, 300 paired corrections, 600 path "
                  "timings (Table V harness); span from seven re-runs of the "
                  "same stream (exp/run_duel_big.py)",
    },
    "microbenchmark": {
        # fallback only: the shipped exp/cost_model.json overrides this row.
        # These are the corrected-predicate fit of 2026-11-06; the previous
        # numbers (c_R 3.2713, c_B 4.2949) came from the two-state conjunction
        # and gave kappa 0.762, which the paper never printed.
        "c_R_ns": 0.3279, "c_B_ns": 4.7102,
        "F_R_ns": -1582.5, "F_B_ns": 2923.9,
        "source": "exp/cost_model.py",
    },
}


def load_microbenchmark():
    """Override the baked constants with a fresh fit when cost_model.json exists."""
    p = os.path.join(HERE, "cost_model.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    fit = j.get("fit")
    if not fit:
        return None
    return {
        "c_R_ns": fit["c_R_ns_per_row"], "c_B_ns": fit["c_B_ns_per_candidate"],
        "F_R_ns": fit["F_R_ns"], "F_B_ns": fit["F_B_ns"],
        "source": "exp/cost_model.py (re-read)",
    }


def boundary(cR, cB, FR, FB):
    kappa = cR / cB
    n0 = (FB - FR) / cR
    return kappa, n0


def rows_for(kappa, n0):
    """The row count above which the band wins at a fixed k/n, for two ratios."""
    out = {}
    for rho in (0.21, 0.10):
        if rho < kappa:
            out["k_over_n=%.2f" % rho] = kappa * n0 / (kappa - rho)
    return out


def main():
    mb = load_microbenchmark()
    if mb:
        FITS["microbenchmark"] = mb

    res = {"fits": FITS, "accountings": {}}
    for name, f in FITS.items():
        kappa, n0 = boundary(f["c_R_ns"], f["c_B_ns"], f["F_R_ns"], f["F_B_ns"])
        res["accountings"][name] = {
            "kappa": kappa, "n_0_rows": n0,
            "kappa_ci_90_bootstrap": f.get("kappa_ci"),
            "kappa_span_runs": f.get("kappa_span_runs"),
            "conditional_rows": rows_for(kappa, n0),
        }

    used = res["accountings"]["fitted_full_paths"]
    # the two numbers the paper prints, asserted rather than eyeballed
    assert abs(used["kappa"] - 0.4414) < 1e-3, used["kappa"]
    assert abs(used["n_0_rows"] - 451) < 2, used["n_0_rows"]
    assert abs(used["conditional_rows"]["k_over_n=0.21"] - 862) < 3
    assert abs(used["conditional_rows"]["k_over_n=0.10"] - 584) < 3
    lo, hi = used["kappa_span_runs"]
    assert lo < used["kappa"] < hi, used["kappa_span_runs"]
    # the paper's sentence: the corpus density sits at the TOP of the span, so
    # at Delta_p = 0 the most favourable run leaves +1 row and none wins
    rho = 182.0 / 394.0
    assert lo < rho <= hi, "span no longer brackets the density"
    assert hi * 394.0 - 182.0 < 2.0, "the top of the span should leave ~0 rows"

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
