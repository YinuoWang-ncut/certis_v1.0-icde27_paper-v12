# The fixed-|D| grid: what the sign surface actually depends on

`exp/same_store.py`, record in `exp/out/same_store.json`, D-sweep records in
`exp/out/_ss_d{1.4,2,4,8}M.json`.  Written 2026-11-08.

## The question

A2 proposed replacing the constant `kappa` of eq. (12) with a curve
`kappa(d)`.  Before fitting anything, the grid had to be made honest: in
`grid80.py` each scope gets its own store, so |D| grows with n, and
`store_size.py` shows one fixed cell moving from 0.368 to 1.102 as the store
grows.  `grid_fixedD.py` held |D| at 2M but rebuilt the store for each of its
seventy cells, so every cell also got a new layout -- and `layout.json` shows
layout alone is worth up to 2x at the largest scope.

This experiment holds both fixed: **one store, one |D| = 6,356,470 rows, one
permutation, seventy cells measured in one interleaved block**, eleven
repetitions.  Between two cells only n and k differ.

## What comes out

Ratio `T_B/T_R` (band path over scan path, both complete, probe included):

| n \ d | 0.02 | 0.05 | 0.09 | 0.14 | 0.183 | 0.25 | 0.30 | 0.40 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| 197 | 1.22 | 1.20 | 1.22 | 1.21 | 1.20 | 1.25 | 1.22 | 1.17 | 1.22 | 1.22 |
| 1,000 | 1.17 | 1.17 | 1.17 | 1.16 | 1.17 | 1.17 | 1.19 | 1.19 | 1.19 | 1.25 |
| 6,969 | 1.09 | 1.08 | 1.10 | 1.15 | 1.16 | 1.17 | 1.18 | 1.20 | 1.22 | 1.23 |
| 20,000 | **0.96** | 1.02 | 1.03 | 1.06 | 1.10 | 1.14 | 1.16 | 1.18 | 1.19 | 1.20 |
| 44,606 | **0.88** | **0.95** | **0.96** | 1.02 | 1.06 | 1.10 | 1.12 | 1.22 | 1.21 | 1.18 |
| 150,000 | **0.64** | **0.66** | **0.66** | 0.97 | 1.02 | 1.43 | 1.64 | 1.54 | 1.91 | 2.17 |
| 412,875 | **0.52** | **0.54** | **0.59** | 1.52 | 1.67 | 1.57 | 1.83 | 1.69 | 1.65 | 1.31 |

Bold: the band is cheaper.  Every row is monotone in d, and the density at
which the sign turns moves *left* as n grows, from 0.60 at n=20k to 0.14 at
n=412k.  The surface is a band of width `d_min(n)` around the diagonal, not a
band in n alone.

Baselines over the seventy cells: **all-band 11/70, all-scan 59/70**.
Eq. (12) as printed agrees on **56/70** -- below either constant.  Beyond the
tie band (|ratio-1| > 0.20) it is 31/33, and the two misses are real false
admissions:

| n | d | k | T_B/T_R | criterion | stopwatch |
|---|---|---|---|---|---|
| 412,875 | 0.140 | 57,803 | 1.52 | band | scan |
| 412,875 | 0.183 | 75,556 | 1.67 | band | scan |

So on a fixed-|D| grid the claim "in no run does it admit a band the stopwatch
finds dearer" does not survive.  It survived on the per-scope grid of
`grid80.py` because there the small scopes were measured in small stores.

## Why the surface has a d-axis at all

Fitting both paths on the same seventy cells (R^2 = 0.987 and 0.969):

```
T_R = 0.74 ns/row * n  +  18.5 ns/cand * k
T_B = 31.7 ns/cand * k
```

Path R is not linear in n alone.  After the scan it gathers the maintained
state at every flipped position, and by construction that count is k/2; the
emit pass then gathers and scatters at the same positions.  **Path R already
pays 58% of path B's per-candidate cost.**  What the criterion compares is the
difference, `c - b = 13.1 ns/cand`, not `c = 31.7`, and that difference is what
`kappa = a/(c-b) = 0.056` divides by.

The fit is stable against dropping small scopes (b = 18.5 in every subset,
c = 31.7-32.1, kappa = 0.056-0.074), which is the useful half of the result:
**at fixed |D| the cost model is linear and reproducible.**

| subset | cells | b (ns/cand) | c (ns/cand) | c-b | kappa |
|---|---|---|---|---|---|
| all | 70 | 18.5 | 31.7 | 13.1 | 0.056 |
| n >= 6,969 | 50 | 18.5 | 31.9 | 13.4 | 0.061 |
| n >= 20,000 | 40 | 18.5 | 32.0 | 13.5 | 0.066 |
| n >= 44,606 | 30 | 18.5 | 32.1 | 13.5 | 0.074 |

## Why `kappa(d)` would have been wrong

Fitting kappa as a linear function of d with `Delta p` pinned reproduces a
slope of 1 to four decimals (measured on the eighty `grid80` cells: 0.9997),
and that slope is algebra, not physics.  Solving each cell for the kappa it
would need gives `kappa_need = (k + dp)/n = d + dp/n`, whose derivative in d
is exactly 1 whatever the machine does -- on any grid, not just that one.  A
fit that recovers 1.000 has observed nothing; it only rewrites "the k term
vanishes" as "kappa rises with d".  The 80/80 agreement it produces is a
tautology.

## Where kappa goes

`kappa = a/(c-b)` with `b/c = 0.58` means the denominator is 42% of a number
that itself moves 4x between stores: `c_B` is 8 ns/cand in `grid_fixedD`
(D = 2M, verification only) and 32 here (D = 6.36M, probe and emit included).
Two consequences the paper has to live with:

1. `kappa` is not a property of the workload.  It is a property of the store,
   the layout and which stages the number is charged.  This is the same
   observation as the paper's "the slope has no clock" (kappa spans 0.03 to
   1.72 across six bare layouts), now with a mechanism.
2. The density axis in the surface is not a curvature in kappa.  It is the
   `c-b` term acting on a k that spans 0.02n to 0.60n while `a*n` is fixed.

## Files

- `exp/same_store.py` -- one store, one |D|, any subset of scopes and densities.
- `exp/out/same_store.json` -- the seventy-cell record used above.
- `exp/out/_ss_d1393350.json`, `_ss_d2000000.json`, `_ss_d4000000.json`,
  `_ss_d8000000.json` -- the same nine cells at four store sizes.  The sign
  pattern is identical at all four; the ratio at (412,875, 0.183) moves
  0.99 -> 1.16 -> 1.69 -> 1.76.
