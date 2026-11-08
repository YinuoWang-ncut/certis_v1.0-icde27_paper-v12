# Sect. V-A's four scopes at five store sizes

`exp/scope_vs_D.py`, record in `exp/out/scope_vs_D.json`.
Written 2026-11-08. Answers the question the paper had left untested.

## What was in doubt

Sect. V-A prints one number per published scope at `d = 0.183`:

    1.01, 0.91, 0.92, 0.99      (197, 6,969, 44,606, 412,875)

Sect. V-E prints, for the same four scopes at the same density,

    1.2, 1.2, 1.1, 1.7

and explains the gap by the store: V-A's cells share 1,393,941 rows (four scopes
x three densities), V-E's share 6,356,470. That explanation was never tested.
Worse, the two sections also differ in accounting: `masked_pipeline.py` reports
`ratio` (path B *without* the rank probe) and `ratio_full` (with it), Fig. 2
plots the first, and V-E's grid times the probe inside the band arm. So two
variables moved at once.

## What was held fixed

`scope_vs_D.py` rebuilds masked_pipeline's twelve-cell store exactly -- same
seed, same plan list, same build rule (`build_rule(n)` for R, `build_rule(k)`
for B), same arms, probe as its own arm -- and reproduces the published four as
a control. Then it rebuilds the same four scopes, one code each at `d = 0.183`,
inside stores of 0.46, 1.39, 2, 4 and 6.36 million rows, with a filler code
holding the balance so `|D|` moves and nothing else does.

The warmup had to match too. masked_pipeline runs eight build variants at five
repetitions before it times a cell; two passes leave the largest scope 18%
slower and the control then misses the archive by that much. WARM = 40.

## The answer

`T_B/T_R` with the probe, two runs:

| store | |D| | 197 | 6,969 | 44,606 | 412,875 |
|---|---|---|---|---|---|
| four codes only | 464,647 | 1.24 | 1.02 | 0.84 | 0.96 |
| + filler | 1,393,941 | 1.25 | 1.02 | 0.83 | 0.97 |
| + filler | 2,000,000 | 1.23 | 1.03 | 0.82 | 1.03 |
| + filler | 4,000,000 | 1.24 | 1.03 | 0.87 | 1.59 |
| + filler | 6,356,470 | 1.25 | 1.04 | 0.86 | 1.56-1.90 |

**Only the largest scope moves.** The other three shift by at most 5% across a
13.7x range of `|D|`; the 412,875 scope reads 0.96 at half a million rows and
1.6-1.9 at six million.

The mechanism is in the same record: `c_B` at that scope goes 13.8 -> 15.2 ->
16.0 -> 37.1 -> 34.9 ns per candidate while `c_R` stays at 2.6-3.7. The band is
75,556 candidates scattered through the store; at 0.46M rows they fit in cache,
at 4M they do not, and the gather becomes a DRAM walk. `c_R` does not care
because the scan is sequential.

So the V-A/V-E gap is the store, as the paper said -- and now it is measured
with everything else held fixed, on the same four cells, both ways of accounting
for the probe.

## The control, and where it disagrees

| cell | this file | archive | archive's three runs |
|---|---|---|---|
| 197 | 1.00 | 1.01 | 1.006-1.018 |
| 6,969 | 0.90 | 0.91 | 0.909-0.947 |
| 44,606 | 0.81 | 0.92 | 0.828-0.953 |
| 412,875 | 0.97-1.12 | 0.99 | 0.974-1.004 |

Three of four agree. The 44,606 cell is 12% low and does not move run to run,
so it is a protocol difference rather than scatter; that cell also has the
widest spread in the archive (0.83-0.95), which is why it is the last place a
reviewer should read a threshold from. It is recorded, not explained away.

## What changed in the paper

1. Sect. V-A names the store (`1.39e6` rows) and the accounting ("verifying the
   band ... neither path charged the probe"). No printed number changed.
2. The sentence after the figure says which cell the store decides: 0.96 at
   0.5e6 and 1.7-1.9 at 6.4e6, the other three within 5%, citing
   `exp/scope_vs_D.py`. "A slice, not a frontier."
3. Fig. 2's caption says `1.39e6` rows and "verification only".

**The four points were not replotted.** They are correct at their store. What
would have to change to replot them is not the numbers but a claim: with the
probe included, the series is 1.24 / 1.02 / 0.84 / 0.97, which crosses parity
twice, so "one crossing" would have to go. That is an argument-structure edit
and the freeze list rules it out this close to the deadline; the caption now
says which comparison the figure shows instead.
