# Changelog

One entry per released tree. The internal archives `ICDE2028_artifact_v1.2` …
`v1.10` were not published; `v1.0-icde27` is `v1.10` with the two scratch files
(`0`, `2`) dropped, the three reference timing outputs restored, the size sweep
re-measured on a corrected instrument (see the first entry), and the publication
furniture added (`docs/`, `CITATION.cff`, CI, `Makefile`).

## paper v8 (EAB track) — 2026-11-07 (same tree, 14-point revision, back inside 12 body pages)

The manuscript is rebuilt against the fourteen-point punch list and the ICDE
page rule (12 pages excluding references and the AI acknowledgement). It is
now 13 pages total with the body ending exactly at the bottom of page 12 and
REFERENCES opening page 13; zero overfull boxes.

* **Routing table restored with real numbers.** `tab:routing` reports, for
  the 3,000-correction replay stream, what the criterion admits under both
  calibration protocols and both accountings (recorded fixed 503--833 /
  post-probe 1,430--2,979; alternating fixed 375 / post-probe 617), computed
  from `exp/duel_big/duel_big_run1.json` and the tuples of
  `exp/kappa_runs.json` -- labelled as arithmetic on a stored stream, not an
  execution record. The withdrawn "600 rejected" reading stays withdrawn.
* **Def. 3 unifies the two scope lengths** ($n=|S_c|$, $m=|N_r|$) and the
  override flag is written $\eta(t)$ throughout.
* **The second microarchitecture keeps its rows** in Table III (16.45 /
  22.25 us, ratios against each box's own path R) and its own paragraph,
  scoped to direction-of-comparison replication.
* **Selector overhead stated as a point estimate**: 4.59 us, 12.1% of the
  mean scope-recompute cost on the merged replay, with guard variants
  separated from a complete pre-probe scope-veto.
* **Two correctness defects fixed beyond the list**: a dangling `\S\ref
  {sec:artifact}` now points at `docs/claim-to-code.md`, and the cited
  `reproduce.py --matrix` flag, which does not exist, is replaced by the
  script's actual contract (run every block in order, exit non-zero on the
  first mismatch).
* **Compression for the page rule**: the standalone findings section, the
  phase figure and two regime/example tables are gone (their numbers live in
  §I and §V prose), the display-page stretch is removed, and `reproduce.py
  --no-data` still holds every assertion.

## paper v5 (EAB track) — 2026-11-07 (same tree, resubmitted manuscript)

The manuscript is rebuilt for the ICDE 2028 EAB track: 13 pages, of which 12
are body. Two cells that v21 could only predict are now measured, and one
sentence that cited a number no file in this tree produces is gone.

* **Table V's denominator is a measurement.** `exp/path_r_h2h/` times path R
  on the exact instance `exp/ivm_h2h/mkdata.py` builds (2e6 rows, 5410
  groups, seed 20261105, 436-row largest scope): 9.5 us warm, 10.5 us cold,
  against a band of 13.7 / 15.4 us. The v21 caption predicted 19.7 us by
  extrapolating a fit whose smallest abscissa is 512 rows; the instance is
  below that range, so the extrapolation was about 2x high. Every ratio in
  Table V is recomputed against the measured value: 1,926x, 1.6e5x, 14.4x,
  16.1x, 1.0x, 1.44x. The old 7.0x is gone.
* **A sentence in §VII is withdrawn, not revised.** It read "the python loop
  measures the band 1.67x faster". No file in this tree produces 1.67, and
  it contradicted Table V's 1.44x *slower* on the same page pair. It is now
  the 5% refusal at the calibration point plus a statement that the two
  rulers part only on the task.
* **§II-A is new and opens the design section**: a four-level evidence
  contract (measured on the task / off the task / derived / reference), and
  every one of the eight table and figure captions opens with its grade.
* **The address is behind a flag again.** §II-D says the artifact travels
  with the submission; `\ifartifactset` puts the URL in once `set_url.py`
  fills it. `make check` no longer gates on the placeholder -- an empty
  placeholder before a tag is the normal state and must not read as a
  failed reproduction. The new `make release` gates on it.
* **`paper/` is v5**: nine `\input` sections (s1, s2_bench, s2_model,
  s3_boundary, s4_certis, s5_eval, s5_findings, s7_threats, s6_tail) and
  fig_phase.pdf. The v21 tree had six sections and no figure file.
* **`certis.py` now ships at the tree root.** `exp/path_r_h2h/` and
  `exp/cross_machine.py` import it; without it Table V's denominator was a
  measurement no reviewer could repeat. `CERTIS_PROTO` overrides the location
  but is no longer required, so both scripts run from a clean checkout. The
  surrounding application -- ingestion and the query surface -- is still not
  shipped; see `docs/claim-to-code.md`.
* **`exp/cross_machine.py`** rebuilds Table V's denominator from a seed on
  any machine and prints that host's constants, so a second microarchitecture
  costs one command rather than a new experiment.

## paper v21 — 2026-11-06 (same tree, revised manuscript)

The manuscript is revised against a simulated ICDE round-2 review; the artifact
tree is unchanged except for `tools/set_url.py`, which now fills both
placeholder mechanisms. Seven changes, all in `paper/`:

* The artifact address is a macro (`\artifactslug`) behind a flag
  (`\ifartifactset`). Until the flag is raised the PDF prints a red
  *[address not yet substituted]* beside the link, so a forgotten substitution
  is visible on the page instead of only on a 404. `tools/set_url.py --check`
  fails while either mechanism is still on placeholder.
* §V-E opens with the criterion divided by the scope, `rho + dp/n < kappa`,
  reports `rho` as a distribution (median 0.337, p95 0.439 over the 600
  corrections; 0.295 / 0.433 over the fitting stream) and states the working
  point `rho = 0.462` against `kappa = 0.441`. A margin rule routes to recompute
  whenever `kappa - (rho + dp/n) < 0.10`.
* §VI separates the three nearest results: [25] takes the amendment as input,
  [26] shows the representation sets the per-tuple cost, bag-semantics IVM keeps
  the view and pays `sigma`. Previously the first two shared one clause.
* The metadata veto is implemented and the 600 corrections re-run:
  `n <= p/kappa = 451` fires on 233 at `|D| = 10^6` (182 at `2.8e6`), the
  surcharge falls from 12–14% single-pass to 2.5% and 5.7%, and the smaller
  cell's interval covers zero (-0.4 to +8.4%). The same paragraph reports that
  the selector's zero band picks come from its constant 0.05 threshold, not the
  workload: routing by the criterion puts the band on 6 of 600 and 66 at the
  larger size without moving p50.
* §V-F says *empty under the CY2024 allocation* wherever the emptiness claim is
  made, including the one place in §II that still said "provably empty", and
  states the year mismatch as a premise.
* Abstract down to 199 words; §I defines `kappa` in one sentence; three
  run-on sentences split.
* Table VIII's python-loop cell and JVM-interpreter cell are refitted in 25
  fresh processes each: `kappa = 0.450` (0.313–0.463) with `k* = 178`
  (123–183), which contains the corpus's 182, so the 5% gap is noise; the
  interpreter is `kappa = 0.695` (0.504–0.760) with `k* = 274` (199–299).

Body still ends inside page 12 (references head on the lower half of page 12,
50 entries run onto page 13), which is what ICDE's "12 pages excluding
references and the AI acknowledgement" requires.

## v1.0-icde27 — 2026-11-06 (paper v20; first public release)

Nothing was published before this tag, so the release ships the paper at v20
rather than v19; §V-A of the paper cites this tag.

* §V-A gains the artifact sentence reviewers look for: the repository URL,
  tag `v1.0-icde27`, `reproduce.py --no-data`, and the fetcher — the URL is
  a placeholder slug until `tools/set_url.py` is run, and that command now patches
  the LaTeX source too.
* §V-E's size sweep is replaced with the corrected numbers
  (`kappa` 0.0337 → 0.0276, scan flat to 3%, gather +24%), measured by the
  rewritten `exp/n0_impl/scale.py`; the withdrawn 0.32 tree and its 50 runs stay
  in `exp/n0_impl/legacy_withdrawn/`, so the sweep now agrees with Table VIII's
  numpy row instead of contradicting it by 10x.

Paying for both additions: §V-E's two paragraphs are tightened, the artifact
statement is one clause inside §V-A's Metrics paragraph rather than a paragraph
of its own, float separation drops 2pt, and the conclusion's page carries two
extra baselines. Body still ends inside page 12, which is what ICDE's "12 pages
excluding references and the AI-generated content acknowledgement" requires.
The acknowledgment keeps its own one-sentence artifact inventory (it is where
the no-redistribution promise belongs); only the URL lives in §V-A.

## v1.0-icde27 — paper v19 (superseded before publication)

First assembled tree; it matches paper v19 and differs from v20 only in the two
places the section above lists.

* `docs/claim-to-code.md` — one row per claim: script, output file, asserted or recorded.
* `docs/EXPECTED_OUTPUT.txt` — the `--no-data` transcript on a clean machine, for diffing.
* `exp/cost_model.json`, `exp/ivm.json`, `exp/rank_probe.json` restored: `expected.json`
  records values from them, and the previous tree shipped the scripts without the runs.
* CI runs `reproduce.py --no-data` on Python 3.10 / 3.12 / 3.13, which is the same
  command the README gives a reviewer.
* `CITATION.cff`, `.zenodo.json`, `Makefile`, `.gitattributes` added.
* Two zero-byte files named `0` and `2`, left by a shell redirect, removed.

Verified before tagging: `reproduce.py --no-data` exits 0 — 11 of 11 witnesses,
eight asserted files `ok`, every `recorded` value identical to the reference run.
