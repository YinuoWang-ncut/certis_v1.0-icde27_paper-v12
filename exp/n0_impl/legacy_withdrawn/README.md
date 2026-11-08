# Withdrawn: the 2026-11-04 scale measurement

Kept so the correction can be inspected rather than taken on faith. Nothing here
is on the `reproduce.py` path and no number in the paper comes from it.

## What it is

`scale.py`, `scale_16m.py`, `scale_16m_reps.py`, `scale.json`,
`scale_16m_runs.json` and the 50 raw runs in `scale_16m_runs/` are the previous
version of the size sweep of §V-E. They reported `kappa` = 0.3218 / 0.3357 /
0.3155 at |D| = 1M / 4M / 16M, with 45 of 50 fresh processes inside 0.30-0.35,
and the paper printed "0.32 over 50 fresh processes".

## Why it is wrong

Two defects, both of them already documented in the README's "Corrections"
section for `exp/n0_robust` and both left in place here:

1. **Predicate.** `~((m == 1) & (OMEGA == 1))` computes the same predicate as
   `~(m == 1)` at `OMEGA = 1` and costs 2.78 ns/row more. The scan itself costs
   0.26 ns/row, so `c_R` came out at 2.89 instead of 0.255 -- 11.3x, not the
   3.6x the README estimated for a slower backend. Because the excess is a
   *constant*, it also flattened `c_R` across |D| (2.923 / 2.864 / 2.898), which
   is precisely the quantity the experiment existed to test: the invariance it
   reported was manufactured by the defect, not measured.
2. **Band.** `cand = np.sort(choice(N, 8192)); cand[:k]` returns the k smallest
   identifiers. With the table value-sorted, every one of them is below the
   threshold, so path B returned 0 rows from 4096 candidates -- it timed a
   gather no band could issue -- and the ids it touched sat in a cache-resident
   prefix of the table, so the gather's per-candidate cost did not respond to
   |D| either.

## The measurement that replaces it

`../scale.py` and `../scale_reps.py`, 50 fresh processes per cell:

| cell | kappa (med) | c_R ns/row | c_B ns/cand |
|---|---|---|---|
| 1M  | 0.0337 | 0.258 | 7.62 |
| 4M  | 0.0311 | 0.258 | 8.34 |
| 16M | 0.0276 | 0.266 | 9.48 |

The scan is flat to 3%, the gather rises 24%, and `kappa` therefore falls 1.2x --
the opposite direction to what this tree reported, and the same order as the
0.030 that the paper's own Table VIII prints for the numpy layout.
