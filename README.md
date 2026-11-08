# Artifact: *When Does Band Retrieval Pay? Scope, Density, and a Measured Floor for Retroactive Rule Evaluation*

[![reproduce (no corpus)](https://github.com/YinuoWang-ncut/certis_v1.0-icde27_paper-v12/actions/workflows/reproduce.yml/badge.svg)](https://github.com/YinuoWang-ncut/certis_v1.0-icde27_paper-v12/actions/workflows/reproduce.yml)

Release `v1.0-icde27`, the tree the paper's numbers were produced from. MIT for
the code, not for the data. One row per claim, with the script and the output it
is checked against: [`docs/claim-to-code.md`](docs/claim-to-code.md). What a green
run prints on a clean machine: [`docs/EXPECTED_OUTPUT.txt`](docs/EXPECTED_OUTPUT.txt).

Everything in the paper that is a number comes from a script here, and
`reproduce.py` re-derives it. Nothing is typed in from a previous run.

## Quick start

```bash
python -m pip install -r requirements.txt
python data/fetch.py          # 27 files, ~250 MB, verifies SHA-256
python reproduce.py           # re-runs everything, exits 0 iff all checks pass
```

No corpus at hand? `python reproduce.py --no-data` still runs the eleven theorem
witnesses, re-fits the six rows of the six-layout table out of stored repetitions,
recomputes the boundary arithmetic and the decision line, and re-times every
timing block that does not need the CMS tables — which is where the paper's
claims that could be *wrong* live. The corpus-dependent checks need the tables.

```bash
make verify                   # = make audit + make check
```

`make audit` is the part a reader can check without trusting any measurement:
every script in `exp/` and `tests/` has a row in `tools/manifest.tsv`, that row
agrees with what `reproduce.py` actually calls, every script it calls carries a
claim in `docs/claim-to-code.md`, and no script the manuscript names is missing,
withdrawn or unreachable. It exits non-zero on the first disagreement. About a
minute for the pair on a laptop; CI runs both on Python 3.10, 3.12 and 3.13.

If you already have the tables, skip the download and point at them:

```bash
CERTIS_DATA=/path/to/dir/with/raw/and/mue python reproduce.py
```

## Cross-machine and scale

One quantity in the paper is a ratio of two per-element times and is therefore
clock-free, which makes it the one a reviewer wants to re-measure somewhere
else: `kappa` in (19). Two things can move it, the table's size and the host,
and neither needs the corpus.

**Size.** `python exp/n0_impl/scale.py` fits both paths at |D| = 1M, 4M and 16M
in one process -- about a minute, numpy only. The paper quotes it for one
claim: how much of the boundary is the warehouse's size. Over that 16x the scan
is flat (0.258, 0.258, 0.266 ns/row) and the gather rises 24% (7.62, 8.34, 9.48
ns/candidate), so `kappa` falls 0.0337 -> 0.0311 -> 0.0276: size is a 1.2x term
in a quantity that moves 1.8x between runs, and it is the gather that carries it.

Those three are medians, not runs. A single fit at 16.7M lands anywhere in
0.007-0.084, and at 4M in 0.017-0.048: the fitted scan slope occasionally comes
back three times its usual value, the same effect that makes the CERTIS error
bar a seven-run span rather than a confidence interval. So every cell is
measured 50 times, each in a fresh process:

    python exp/n0_impl/scale_reps.py 50        # ~60 s, all three cells

`scale_reps.py` appends to `exp/n0_impl/scale_runs/` instead of replacing it,
and reduces every run in that directory into `scale_reps.json`. The shipped file
holds 50 runs per cell; two independent batches agree on the medians to within
0.0006.

**Withdrawn.** The previous version of this cell -- one run per size, `kappa`
0.3218 / 0.3357 / 0.3155, plus a 50-run median of 0.3155 at 16.7M -- is kept in
`exp/n0_impl/legacy_withdrawn/` and is used by no number in the paper. It is
11.5x high, and its flatness was manufactured by the defect rather than
measured: see "Corrections" below.

**Machine.** Those two scripts are the whole cross-machine experiment for this
ratio: nothing to copy over, nothing to download, no corpus, no prototype, no
JDK. On a second host,

    python exp/n0_impl/scale_reps.py 50
    # then compare kappa.median and c_B.median with scale_reps.json

**Move `exp/n0_impl/scale_runs/` aside first.** `scale_reps.py` appends to that
directory and reduces *every* run in it, which is what makes a repeat batch widen
the sample on one host -- and what makes it wrong on a second one: with the 50
shipped runs per cell still there, the reported median is a mixture of the two
hosts and belongs to neither. So `mkdir ..\_box1_runs`, move the three sets of
`*_run*.json` into it, run, send `scale_reps.json` and the new run files, then
put them back and restore `scale_reps.json` -- `git checkout --` it in a clone,
or copy it back if you are working from the zip, which carries no `.git`. This is
the only reading in the paper that is meant to cross machines; nothing else needs
it.

Within one machine the median moves by about 0.001 between batches, so a
difference of 0.003 in `kappa`, or 0.3 ns in `c_B`, is the smallest that reads
as a machine effect; anything smaller is the tail doing what the tail does. The
corpus-side cells are different: `exp/run_duel_big.py` times the prototype,
which this archive does not ship (see the kappa note below), and the 27 CMS
tables must be fetched before any corpus script runs.

**What the second box returned.** `exp/cross_machine_box2.txt` is the run
summary from an AMD Ryzen 5 8640U (Zen4, six cores, 15.27 GB, Windows 11):
`reproduce.py --no-data` passes all eleven witnesses there, and P0 at n = 436
puts the band at 1.37x warm and 1.43x cold against 1.44x here, with both paths
1.6-1.7x slower in absolute terms. Table V carries that box as two further
rows (16.45 / 22.25 us, 1.35x), and the caption quotes 1.37x: the first is the
quotient of the two medians, the second the median of the three paired ratios,
and the aggregate prints both so they cannot be swapped. What does *not* cross
the gap is any engine arm -- DuckDB, dataflow and Drools were not re-run there,
and no cell in Table V compares one box against the other. The bare-layout
`kappa` of `scale_reps.py` was likewise *not* re-measured, so the ratio is
replicated and the kappa column is still one host. `exp/reproduce_box2_record.txt`
is the whole `--no-data` record block from that machine: it is identical to this
one, digit for digit, because the block re-reads the shipped `exp/*.json` rather
than timing anything -- it is a digest, not a measurement.

To make that box a *record* rather than a log, re-run it three times
(`python exp/cross_machine.py --quick --tag box2`, ~1 min each): each run now
writes `exp/cross_machine_box2-rK.json` with K auto-incrementing, so nothing
overwrites, and `python exp/cross_machine.py --merge box2` reduces them into
`exp/table_v_box2.json` plus the two lines Table V prints. The box did not send
those files back, so what ships is the summary; `--merge` fails loudly rather
than guessing when they are absent.

## What is asserted, and what is not

`expected.json` has two sections, and the split is the honest part.

**`assert`** — deterministic given the tables. Row counts, pair counts, degree
percentiles, the number of distinct MUE caps, how many of the 49 threshold
placements give a zero band, tier sizes, the two band sizes that touch zero and
3,074 edited pairs, the delta-vs-reconstruction agreement (4,249 and 4,146
additions, zero disappeared pairs, 14 end-dated deletions), the 5-of-7 indicator
flips behind the 29% figure, the 31 MUE revisions that the change file cannot
sign, the 116 published cap moves with the 98 empty bands among them, and the
arithmetic that turns four fitted constants into kappa = 0.4414 and n_0 = 716.8 rows,
the rank probe charged (451.3 without it).
These reproduce bit-for-bit on any machine.

Kappa's error bar is *not* in `assert`, and is not a confidence interval. The
bootstrap over one duel run's records gives [0.37, 0.52], but resampling within
a run cannot see the error that dominates -- the large-band gather moving between
53 and 83 us from process to process. Seven independent re-runs of the same
stream in fresh processes put kappa at 0.26-0.46. `exp/run_duel_big.py` is the
harness (it times a prototype tree this artifact does not ship; the seven runs
it recorded are in `exp/duel_big/`), and `exp/fit_kappa_runs.py` reduces them
into `exp/kappa_runs.json` -- that reduction needs nothing but the shipped runs.
`exp/bootstrap_kappa.py` is kept, marked superseded, as the record of the
narrower interval and why it was rejected.

**`recorded`** — timings. `cost_model.json`, `ivm.json`, `rank_probe.json` and
`n0_impl/scale.json` are printed next to the paper's values and never asserted,
because they are a property of the host: the paper says so in §V.
`n0_impl/scale_reps.json` is printed the same way, and **it is the file to open
if you are re-measuring kappa on another machine** -- see "Cross-machine and
scale" above. `cost_model.json`'s `derived.kappa` (0.0696) is an
artifact-internal reduction that the paper does not print at all; it read 0.76
until the same predicate was fixed in that file (see Corrections).

Table VIII (the six bare layouts) is not in `recorded`. `n0_robust.json` stores all 25 repetitions of
every point, so `fix_boot.py` re-fits the six rows from them without touching a
clock and `reproduce.py` asserts the result; the interval endpoints depend on
the bootstrap seed, which is 11, not the instance seed 20261105.

The `exp/*.json` shipped in the archive are the reference run's outputs, kept so
a reviewer can diff without first fetching 253 MB. `reproduce.py` overwrites
them; `expected.json` — not these — is what the checks read.

## Layout

```
reproduce.py             the entry point; asserts expected.json
expected.json            the golden values, split as above
data/fetch.py            downloads and verifies the 27 CMS files
data/manifest.csv|json   path, URL, byte count, SHA-256, and which script reads each
exp/degree.py            "Setup"  degree, end-dating, modifier distribution
exp/delta.py             "Real correction streams"  published delta vs reconstruction
exp/regime_bc.py         "Real correction streams"  the audit, regimes (b)/(c)
exp/boundary.py          "The boundary"  kappa and n_0 from the fitted constants
exp/cost_model.py        "The boundary"  one-process accounting, the 0.762 of §V-C
exp/rank_probe.py        "The boundary"  fetch/scan ratio (ratio only; see below)
exp/ivm.py               "Baselines"  DuckDB: full, scope, band, and delta = band
exp/ivm_h2h/             "Baselines"  the head-to-head harness (Rust), its recipe
                         (mkdata.py regenerates the 24 MB instance from exp/ivm.py's
                         seed), the three raw runs, and aggregate.py which folds them
exp/path_r_h2h/          "Baselines"  path R on the Table V instance, timed warm and
                         cold by the same harness; its json is Table V's denominator,
                         so that column is measured rather than extrapolated
exp/java/CountOracle.java "Baselines"  four count structures, one JVM
exp/run_duel_big.py      "End-to-end certification"  the seven-run harness (see kappa
                         note above; needs CERTIS_PROTO to re-time, or just reduce the
                         shipped exp/duel_big/ with exp/fit_kappa_runs.py)
exp/fit_kappa_runs.py    "End-to-end certification"  reduces the seven runs -> kappa_runs.json
certis.py                the core the two timing scripts import: classify,
                         decide, execute (R scan / B bisect+gather / P
                         refresh), verify. numpy only, ~700 lines, MIT.
                         exp/run_duel_big.py and the two scripts above
                         import it; CERTIS_PROTO overrides the location.
exp/cross_machine.py     the second-machine pass: rebuilds Table V's denominator from a
                         seed, then prints this host's constants; --merge TAG
                         reduces the r*.json of that tag into Table V's row
exp/cross_machine_box2.txt  the run summary that box returned (P0: 1.37x warm,
                         1.43x cold, 16.45 / 22.25 us); no r*.json was
                         collected, re-run the script three times for it
exp/bootstrap_kappa.py   superseded: the within-run bootstrap, kept as the record
exp/bootstrap_kappa_check.py  the delta-method cross-check on that interval
exp/run_duel_repeat.py   the discarded 31-sweep repeat route; its record is
                         exp/duel_repeat/duel_rep.json
exp/n0_robust/           the six-configuration n_0 study: n0_robust.py (rebuilds its
                         own claim databases from a seed), fix_boot.py (re-derives the
                         table from the shipped n0_robust.json), N0Bench2.java, and
                         the two reference outputs
exp/n0_impl/             the boundary arithmetic: dline.py, the 1M/4M/16M sweep
                         (scale.py), and the 50-run repetition study of its
                         largest size (scale_16m.py, scale_16m_reps.py)
exp/cheap.py             "The summary, built"  occupancy vs partition summary on the caps
exp/utilization.py       "The summary, built"  the observed population, from CMS's own
                         utilisation release (see the note below the layout)
exp/utilization.json     the reference run of utilization.py -- its ~10 GB input is not
                         redistributed, so this file is the record of §"observed"
docs/claim-to-code.md    one row per claim: script, output, asserted or recorded
docs/EXPECTED_OUTPUT.txt what `--no-data` prints on a clean machine
tools/set_url.py         fills the two repository-url placeholders; --check gates
                         `make release`, not `make check`
tools/audit.py           the ledger: disk vs reproduce.py vs claim-to-code.md;
                         exits non-zero on the first disagreement
tools/manifest.tsv       one row per script: entry | corpus | provenance |
                         legacy | helper, with the reason.  Edit ROWS in
                         tools/manifest_src.py and re-run it, never by hand.
tests/test_theorems.py   T1-T11, one witness per repaired or refuted statement
paper/main.tex            the compile unit: \input's the nine sections below.
                         main.tex + s1_intro.tex ... s7_threats.tex are the
                         source the numbers belong to (draft v5, EAB track:
                         13 pages, 12 of body); figures are TikZ except
                         fig_phase, so pdflatex it twice.  The artifact
                         sentence in §II-D carries a slug behind
                         \ifartifactset, which set_url.py fills.
```

## The eleven witnesses

Each test in `tests/test_theorems.py` does the same thing twice: it builds a
witness on which the *previous* draft's statement is false, then checks the
statement this paper makes on the same witness. That is why the file is evidence
rather than a smoke test. T1 is the one that mattered: one disjunct flipping
does not flip a disjunction, so the band is a candidate set and the flip set is
the band intersected with a support test.

## One modelled object

`cheap.py` reconstructs the claim population, which CMS does not publish: an
MUE cap is by definition the maximum over six months of submitted claims, so
the population is anchored with per-code maximum equal to the published cap and
a Zipf shape below it. Everything that depends on that reconstruction -- 7.4M
rows, 80,326 occupied cells, the 63 of 98 bands a Bloom certifies -- is printed
and **not** asserted; what is asserted is what the cap table fixes on its own:
116 moves, 98 increases, 98 empty bands, 3 probes per rejection at the median
move, and the 0.94 MB cell bound. Re-run with a different seed and the first
three move by a few percent while the asserted five do not move at all.

`exp/utilization.py` is the v9 answer to that model: it takes the population
from CMS's own utilisation release instead of inverting one, and reports, per
move, a decision that follows from the published aggregate alone (necessarily
empty / forced overtake / inside on the mean / undecided). It needs two files
`fetch.py` does not fetch -- `mup_phy_2024_prov_svc.csv` and
`mup_phy_2024_geo.csv` from `data.cms.gov`'s Medicare Physician & Other
Practitioners, CY2024 release (~10 GB unpacked; not CPT-restricted, but too
large to hash into this manifest) -- so `exp/utilization.json`, the reference
run over 9.78M rows, ships in its place.

## Two places where the artifact disagrees with a careless reading

**`rank_probe.py` reports a ratio, not a latency.** Running the probe through
CPython prices two interpreter calls per group, not two binary searches; the
compiled figure is 142 ns against the interpreter's 3,851 ns. The paper uses the
compiled figure and says where the 27x went. The ratio (fetch/scan = 17.2) is
the machine-independent part and is the part the argument uses.

**`cost_model.py` does not produce the paper's kappa.** It fits both paths on a
synthetic layout in one process and gets 0.762; the paper's 0.4414 comes from the
end-to-end harness fit (`t_R = 11.3 ns*|sigma| + 14.8 us`, `t_B = 25.6 ns*|B| +
19.9 us`), which `boundary.py` holds and computes from. §V-C quotes both, and
`dline.json` computes the decision threshold under each, because the ratio is
accounting-dependent and reporting one number as if it were the constant was a
defect of the earlier draft.

## Data

27 files, all from `cms.gov`, fetched 2026-11-03/04: 16 PTP full-table shards
(2026 Q3 and Q4, practitioner and hospital), 4 PTP quarterly delta bundles, 6
MUE quarterly tables (2025 Q1 through 2026 Q4), and 1 MUE delta bundle.

They are not redistributed: the CPT-derived tables are AMA property and CMS
distributes them under its own terms. `manifest.csv` is the contract — if a
re-fetched file's SHA-256 differs, CMS has reissued it, and the counts will move.

## Licence

MIT for the code. Not for the data. See `LICENSE`.

## Corrections (superseding `exp/n0_robust`)

The six-layout table (Table VIII in the current draft; Table VII in the v18
draft this archive shipped with) is computed from `exp/n0_holdout/`, not from
`exp/n0_robust/`. The
earlier measurement had three defects, all of which pushed `kappa` toward 1:

1. **Scan origin.** The Python backends timed `v[LOFF:LOFF+n]`, where `LOFF`
   is the midpoint of the table; `N0Bench2.java` timed `v[0:n]`. The JVM table
   is sorted by value and stored `v[0] = 0`, so the predicate passed on zero of
   65,536 rows (26,176 on the Python side). The Java arm carried no count
   assertion; the corrected `N0Bench3.java` aborts on any mismatch.
2. **Band construction.** `np.sort(choice(N, 8192))` takes the k smallest
   identifiers when sliced, so for k <= 4096 no candidate was in the band at
   all (pass rate 0%; 50% only at k = 8192). `n0_holdout` draws candidates from
   the band and keeps them unsorted.
3. **Predicate.** `~((m == 1) & (OMEGA == 1))` is semantically identical to
   `~(m == 1)` at `OMEGA = 1` but costs 4.19 against 1.16 ns/row. The penalty
   is ~2.5 ns per element: 3.6x on a scan that costs 0.3-1.2 ns/row, and 7% on
   a gather that costs ~8 ns per candidate, which is what compressed the ratio.

Effect on the six bare layouts, `kappa` old -> new: numpy 0.329 -> 0.030,
python loop 0.541 -> 0.440, sqlite 0.157 -> 0.122, JVM C2 0.149 -> 1.636,
JVM C1 0.945 -> 1.723, interpreter 0.813 -> 0.580. The split stays 3-3, the
membership does not: the python loop leaves the band side (k* = 174 against
k = 182) and JVM C2 joins it.

`n0_robust/` is kept so the change can be inspected; it is no longer on the
`reproduce.py` path.

**The same two defects, in two other files (2026-11-06).** `exp/n0_impl/scale.py`
and `exp/cost_model.py` still carried correction 3's conjunction, and `scale.py`
carried correction 2's band as well. In `scale.py` the excess (2.78 ns/row)
dominated a 0.26 ns/row scan -- 11x, not the 3.6x that correction 3 estimated
from a slower backend -- and, being a constant, it flat-lined `c_R` across |D|,
which is the one thing that script existed to test; its band arm gathered from a
cache-resident prefix and returned zero rows, so it timed rows a band cannot
return (pass rate 0 of 4096). Both are fixed, and the withdrawn tree with its 50
runs is in `exp/n0_impl/legacy_withdrawn/`. The number the paper took from it
(0.32 in §V-E) was 11.5x high and contradicted this paper's own Table VIII,
whose numpy row is 0.030; the corrected sweep gives 0.0337 / 0.0311 / 0.0276, so
the sweep now agrees with the table instead of contradicting it. `cost_model.py`
is fixed too: `derived.kappa` 0.76 -> 0.0696, a reduction the paper never
printed.

## v18 presentation revision

The claims were re-scoped without changing any measured number. Title moved from
"when retrieval loses" to "pricing band retrieval"; the abstract makes the sign of
the criterion the subject and the corpus loss a clause; contribution 4 and the
scope section were rewritten so the inequality leads; the conclusion closes on
the criterion rather than on the negative. One figure was added (Fig. 1, band vs.
ripple set) and the probe table was folded into the prose it duplicated. A
portability paragraph in V-E states the three measurement steps.

Title second pass: "One Inequality Prices Band Retrieval against Scope Recompute",
so the criterion is the subject of the sentence and the recompute is its object.
A third figure (Fig. 3) plots the six bare-layout thresholds `k*` against the
`k = 182` the corpus requires, one dot per layout with a bar for the three
further instances; the line cuts the six three and three. No number changed in
this pass.

Theorem 5(c) revised (reviewer point): the clause no longer argues from a two-row
counterexample. It now states the sharp bound, sup k = Sigma * min{1, (mean - v1)/
(theta0 - v1), (vP - mean)/(vP - theta1)}, reached by a two-atom extremal
distribution, so no bound from (|delta|, Sigma, R, mean) comes in under it and the
trivial k <= Sigma is all that remains when the mean sits inside the band. Proof by
the two-moment linear programme. No measured number changed; Table VI, the
criterion and the artifact scripts are untouched. **This statement is narrowed in
v19**: the bound is now presented as a relaxation over distributions sharing the
same first two moments, not as the exact maximum candidate count of a finite
database, and the sharpness language is withdrawn. See *v19 restructuring* below.

## v19 restructuring

The paper was rebuilt around the decision it actually supports. **No measured
number changed**: every figure, every table entry and every cost here is the same
run as in v18. What changed is what each number is allowed to claim.

*Title and abstract.* Now *Choosing between Band Retrieval and Scope Recompute for
Retroactive Rule Evaluation*. The abstract is organised as problem, three
findings, evidence, implications: the wrong comparison object (band against table
against scope), candidates are not flips, and probe plus representation move the
sign. The single inequality survives as the instrument, not as the contribution.

*Theorem 2 (candidate coverage).* The one width `delta` that mixed two corrections
is split. Holding the mean fixed and moving the threshold gives
`w = abar_l |dlambda|`; holding the tier fixed and editing one amount gives
`w = lambda |da| / n_l`, and that edited row joins the candidate set on its own.
Both cases stay covered; the second is what makes "the band is not the flip set"
checkable rather than rhetorical.

*Theorem 4 (density).* Explicitly restricted to an exogenous endpoint. It is no
longer applied to a threshold that is the mean of the rows being counted; where
that happens, the table that uses it says the density form is a heuristic.

*Theorem 5(c).* Withdrawn as a sharp bound. It is now a relaxation: over
distributions with the same first two moments, the mass a band can carry is at
most `Sigma * pbar` with
`pbar = min{1, (vbar - v1)/(theta0 - v1), (vP - vbar)/(vP - theta1)}`, the
extreme point being two atoms. That is an upper bound on a relaxation, not the
maximum candidate count of a finite database: at `theta0 = theta1` the band is
empty while `Sigma * pbar > 0`, and the upper atom `vP` is excluded by the
half-open endpoint. The atomic staircase property stands as before.

*Evidence grade.* Table IV states, per measured block, the data, the output task,
whether the number is measured or predicted, and what it bounds. The head-to-head
table (VI) is demoted to a reference comparison: its denominator is a *predicted*
19.7 us, so the ~7x is not one of this paper's results. The three-win-three-loss
split over the six bare layouts stands, with two qualifiers now in the caption: it
is measured under the bare-kernel cost model, and the 394-row instance sits below
the 512-row point where the fits begin.

*Thirteen of 98, on paper.* Section V-G derives the certificate instead of citing
it. A group with `S` units over at least `B` lines puts at most `S - B + 1` on one
line, so its band `[l, h)` is unreachable iff `S - B + 1 < l`; a move is
necessarily empty iff that holds for every observed group of the code. For the
`1 -> 2` moves that make up all thirteen this reduces to `S <= B`, and the section
works one case numerically in each direction. `exp/empty_cases.py` prints
`l - 1 - S_g + B_g` per group and writes `exp/empty_cases.csv`, one line per case
with the minimum over groups, so the count re-derives from the two published
columns without our code. It needs the CMS provider-service release and the six
MUE packs; without them it stops with the download pointer, as `exp/utilization.py`
does.

The empty-band statement is scoped accordingly: inside the single-threshold
theorem an empty candidate band implies an empty flip set, while validity
intervals, override interaction and the other corrections are outside that scope
and are not decided by a threshold band. The v18 sentence that read "if `B` is
empty then Omega is empty" without the qualifier is gone.

*A running example.* Section II carries four records through the paper: a
candidate that flips, a candidate masked by another rule (so not a flip), a
support change from 2 to 1 with no verdict change, and a 10-to-20-to-10 batch
whose net is zero. The same records price the two access models: 28.02 against
26.10 us before the probe, 28.02 against 29.10 after it.

*Wording.* Absolute claims were replaced by scoped ones. "Not worth retrieving" is
now "not worth retrieving under this complete cost profile"; clock-free
portability is now "a unified scaling leaves the criterion unchanged, a deployment
needs recalibration"; "existence is not cheaper than output" is now a statement
about the corresponding access model's worst case.

## Table and figure numbering in this version

Table I example records; II the three regimes; III ripple measurements; IV evidence
grade; V end-to-end latency; VI reference head-to-head; VII the two correction
families as published; VIII the seven fittings. Figure 1 band against ripple set;
Figure 2 the six bare-layout thresholds.
