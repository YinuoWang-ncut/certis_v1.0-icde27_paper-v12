# Where each claim comes from

One row per claim a reviewer could try to falsify. The `check` column is the honest
part:

* **assert** — `reproduce.py` compares the produced file against `expected.json`
  and exits non-zero on any mismatch. Host-free.
* **record** — a property of the machine. Printed beside the paper's value, never
  asserted. See the `recorded` section of `expected.json`.
* **†** — needs the corpus: the 27 CMS releases (`data/fetch.py`), or the ~10 GB
  CMS utilisation release, which is not redistributed.

`python reproduce.py --no-data` covers every row without a dagger.

Two conventions this table has to keep straight, because the paper's own
arguments turn on them:

* **instrument** — a row about a *bare kernel* (the two access paths with the
  mask, the probe and the overlay removed) and a row about the *pipeline* (all
  five stages) are different measurements of the same $(n,k)$.  They are marked
  as such; quoting one against the other is the error Sect. V-A is about.
* **producer** — every row names the script that prints the number, and every
  script named here is reachable from `reproduce.py`.  That property is
  machine-checked by `tools/audit.py`, not asserted here.

Section letters are the v21 draft's.  In the v5 (EAB) manuscript the
workload paragraph is §II-B, the boundary arithmetic is §V-D, the
head-to-head table is §V-B (Tab. V) and the six bare layouts are §V-C
(Tab. VI).  The script and output columns are unchanged and are what the
check reads; renumbering the § column is a 30-minute job, not a
correctness one.

| § | claim | script | output | check |
|---|---|---|---|---|
| III, Thm. 1–5 | the eleven statements, one witness each | `tests/test_theorems.py` | stdout: `11 / 11 passed` | assert (exit status) |
| IV-B | degree, end-dating, modifier distribution | `exp/degree.py` | `exp/degree.json` | assert |
| IV-B | published delta vs reconstruction; 4,249 / 4,146; zero disappeared pairs; 14 end-dated deletions; 5 of 7 indicator flips; 31 unsigned revisions | `exp/delta.py` | `exp/delta.json` | assert |
| IV-B | relative regimes (b)/(c), semi-synthetic | `exp/regime_bc.py` | `exp/regime_bc.json` | assert |
| IV-D | `kappa = 0.4414`, `n_0 = 451.3` rows from four fitted constants | `exp/boundary.py` | `exp/boundary.json` | assert |
| IV-D | the second accounting, `kappa = 0.0696` (read 0.762 before the predicate fix; the paper prints neither) | `exp/cost_model.py` | `exp/cost_model.json` | record |
| V-E | `Delta p` and the decision line solved for `a_d` | `exp/n0_impl/dline.py` | `exp/n0_impl/dline.json` | assert |
| V-E, Tab. VI | the six bare layouts: `kappa` and `k*` per layout | `exp/n0_holdout/analyze.py` | `exp/n0_holdout/n0_holdout_summary.json` | assert |
| V-E | six probe rows in the units of (6) | `exp/n0_robust/probe_rows.py` | `exp/n0_robust/probe_rows.json` | assert |
| V-E | `kappa` against \|D\|: one run per size, 0.034 / 0.0333 / 0.0270 | `exp/n0_impl/scale.py` | `exp/n0_impl/scale.json` | record |
| V-E | `kappa` over 50 fresh processes per cell: medians 0.0337, 0.0311, 0.0276; `c_R` 0.258/0.258/0.266; `c_B` 7.62/8.34/9.48 | `exp/n0_impl/scale_reps.py` | `exp/n0_impl/scale_reps.json` | record |
| V-E | the seven end-to-end re-runs, span 0.26–0.46 | `exp/fit_kappa_runs.py` | `exp/kappa_runs.json` | folded into `exp/boundary.json`, span printed not asserted |
| V-B, Tab. V | path R on the 436-row instance the engines share: 9.5 µs warm, 10.5 µs cold; the band end to end 13.7 / 15.4 µs, so 1.44x | `exp/path_r_h2h/` | `exp/path_r_h2h/path_r_h2h.json` | record |
| V-B, Tab. V | the same scope on a second microarchitecture (Zen4 mobile, 6 cores): 16.45 / 22.25 µs, 1.35x from the two medians and 1.37x as the median of three paired runs, 1.43x cold against 1.46, both paths 1.73x / 1.62x in level | `exp/cross_machine.py --quick --tag box2` x3, then `--merge box2` | `exp/cross_machine_box2.txt` | record — `--merge` recomputes the aggregate from `cross_machine_box2-r*.json` and refuses when they are absent; that box returned only its run summary, so the three per-run files are not shipped |
| V-B, Tab. V | the reference head-to-head, divided by that measured denominator | `exp/ivm.py`, `exp/ivm_h2h/`, `exp/java/CountOracle.java` | `exp/ivm.json`, `exp/ivm_h2h/ivm_h2h.json` | record |
| V-D | 116 moves, 98 empty bands, the cap-table bounds | `exp/cheap.py` | `exp/cheap.json` | assert |
| V-F | a verdict for all 116 observed moves; 13 certified empty | `exp/utilization.py` | `exp/utilization.json` | record — the 10 GB input is not shipped |
| V-F | the tightest slack per case, so the thirteen re-derive | `exp/empty_cases.py` | `exp/empty_cases.csv` | † |
| V-D | the fetch/scan ratio, 17.2, machine-independent part only | `exp/rank_probe.py` | `exp/rank_probe.json` | record |
| V-D | the 600 corrections under both protocols: `kappa = 0.214` (alternating) against `0.441` (recorded); `n_0 = 959` and `717`; `Delta p = 199` and `316`; none faster down the band in all 23, 362 slower in all 23, 28 protocol-flipped, agreement 81.0\% against 77.3\% | `exp/alt600.py` | `exp/out/alt600.json` | assert |
| V-D | the two protocols read the same 600 corrections (`stream_identical`) | `exp/alt600.py` | `exp/out/alt600.json` | assert |
| V-E | band density ceiling from two published columns: 426 in-regime moves, median ceiling 0.183, quartiles 0.042/0.401; 193 admitted, 233 excluded on density | `exp/realband.py` | `exp/realband.json` | † |
| V-E, Fig. 1 | the 80-cell per-scope grid: 8 scopes x 10 densities; agreement 47-50/80 across nine stored runs (48-50 at reps=25), all errors refusals. Not comparable across columns: each scope brought its own store | `exp/grid80.py` | `exp/out/grid80.json`, `exp/out/grid80_runs/` | record (needs the prototype) |
| V-A, Fig. 2 | the four published scopes at d = 0.183 in one store of 1.39e6 rows: 1.01, 0.91, 0.92, 0.99 (verification only, probe excluded). The largest is the store's: 0.98 at 0.5e6 and 1.59 at 6.4e6, the other three within 5% | `exp/masked_pipeline.py`, `exp/scope_vs_D.py` | `exp/out/masked_pipeline.json`, `exp/out/scope_vs_D.json` | record (asserted) |
| V-E, Fig. 1 | the fixed-|D| grid: 70 cells (7 scopes x 10 densities) in one store of 6,356,470 rows and one layout; band stops paying at d = 0.02 / 0.14 / 0.09 for n = 2e4 / 1.5e5 / 4.1e5; criterion over-admits 14 cells, refuses none; T_R carries 18.5 ns per candidate against T_B's 31.7 | `exp/same_store.py` | `exp/out/same_store.json` | record (asserted) |
| V-E, Fig. 2 | the store-size confound: at n=412,875, k=75,556 the ratio moves 0.37 -> 1.02 as the store grows 0.4M -> 6.6M lines while c_R stays 5.8 ns | `exp/store_size.py` | `exp/out/store_size.json` | record (needs the prototype) |
| III | the density-form null model, its proof and the two checks | `exp/boundary.py` | `docs/density-form.md`, `exp/boundary.json` | assert |
| V-E, Tab. V | the criterion against the stopwatch, scored with the normative order-controlled tuple: 24 cells, 18 agree; the confusion matrix 5/4/2/13, 4/9 within 20% of a tie, 14/15 beyond, and no false admission beyond the tie band | `exp/bigscope.py` | `exp/bigscope.json` | record (needs the prototype) |
| V-A, instrument: pipeline | the five stages at the published sizes, under the stated build rule: 1.01-1.02 / 0.90-0.94 / 0.83-0.98 / 0.99-1.01 times the scan at d=0.183 over three processes, Delta-complete on all four; the eight builds span 0.50-1.22 at the largest scope, which is why the range and not a point is printed | `exp/masked_pipeline.py` | `exp/out/masked_pipeline.json`, `exp/out/mp_run{1,2,3}.json` | record |
| V-A, instrument: bare kernel | the same cell priced on its own: path R 106 us with a boolean-index compaction against 59-64 with `compress`, path B 64 either way, so the compaction choice alone moves the cell from 0.63x to 1.0-1.08x; the `take` gather is 1.2-1.9x faster on a contiguous band but does not reproduce across processes (34.3 / 34.6 / 56.8 us), which is why it is not built with | `exp/build_triple.py` | `exp/out/build_triple.json` | record |
| V-B | the second microarchitecture: box 1 at 9.5/13.7 $\mu$s and box 2 at 16.45/22.25, ratios 1.44 and 1.35; box 2 run 1 is 1.3921$\times$ run 3 in level but only 1.0244$\times$ in ratio | `exp/cross_machine.py --quick --tag box2` | `exp/cross_machine/{box1,box2-r1,box2-r2,box2-r3,summary}.json` | record |
| V-E | verification versus scan on the same 600 corrections: 0.90 at the median, 1.04 in aggregate, 410 of 600; probe 6.4 against a 14.6 us scan | `exp/fullpath_band_vs_scan.py` | `exp/fullpath.json` | record (needs the prototype) |
| V-A | end-to-end ordering and `Delta`-completeness on five policies | `exp/e2e_repro.py` | `exp/e2e_repro.json` | record (needs the prototype) |
| V-A | the two primitives and their crossover: compaction by `compress` wins from 2.8e4 elements and a boolean index below 1.8e4; gathering scattered identifiers costs 221 us by fancy indexing against 321 through `take` | `exp/prim_cost.py` | `exp/out/prim_cost.json` | record |
| V-A, Fig. 2 | the ratio against scope at d=0.183, three processes of 21 repetitions, paired bootstrap intervals, with the floor and the break-even placed from the fitted constants before anything is timed | `exp/fig_ratio.py` | `exp/out/fig_ratio.json`, `exp/out/figs/` | record |
| V-A | six value/id width pairs (int8/16/32 x int32/64): the sign holds on all six at 6,969, 44,606 and 412,875 rows and does not at 197, which is the region the floor excludes | `exp/layout_sensitivity.py` | `exp/out/layout.json` | record |
| V-A | one instance, two engines: DuckDB reads the candidate list by zone map and never chooses its ART for a range predicate; SQLite's B-tree counting costs 0.137 of the scope scan at 44,606 rows and materialising 1.51, because a B-tree returns identifiers to dereference | `exp/sql_same_instance.py` | `exp/out/sql_same_instance.json` | record |
| IV-F, Prop. 4 | sigma maintenance priced in nesting order: the mask reads 2.8 ns per candidate, the update 8.1, both flat in n from k=300 to 150,000, so no term for sigma enters eq. (11) | `exp/sigma_selfpay.py` | `exp/out/sigma_selfpay.json` | record |

## What the artifact does not contain

1. **The corpus.** 27 CMS files, ~250 MB, fetched and SHA-256-verified by
   `data/fetch.py` against `data/manifest.csv`. The PTP and MUE tables carry CPT
   content that is AMA property, so they are not redistributed.
2. **The ~10 GB utilisation release.** `exp/utilization.py` and
   `exp/empty_cases.py` read it; `exp/utilization.json` is the reference run over
   its 9.78M rows, shipped in its place.
3. **Nothing, for the tables.** `certis.py` ships at the tree root and the
   prototype that drives it ships under `exp/prototype/` (`w12/run_w12.py`,
   `w8/`), so Table II, Table III and the phase split of Sect. V-E are
   re-timable from a clean checkout: `exp/e2e_repro.py`,
   `exp/fullpath_band_vs_scan.py`, `exp/bigscope.py` and
   `exp/masked_pipeline.py` find the prototype without any environment
   variable. `masked_pipeline.py` is the pipeline cell of Sect. V-A: it runs
   all five stages -- mask, dedup, overlay, selector -- at the four published
   per-code scope sizes, which is what makes Table II (401 rows) and the
   six-density sweep (up to 412,875) points on one curve instead of two
   objects. It prints 1.29x / 0.95x / 0.54x / 0.52-0.63x the scan; the first
   three repeat to 2%, the last is given as an interval for that reason. What is not shipped is the ingestion
   path that built the corpus store from the raw CMS downloads -- it is not
   needed, because the stores those three scripts build are synthetic and
   self-contained.
