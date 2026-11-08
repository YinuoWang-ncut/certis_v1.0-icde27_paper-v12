"""Write tools/manifest.tsv, asserting it covers the tree exactly.

Run once from the release root; the result is committed and from then on the
manifest is the source of truth and `tools/audit.py` checks it against disk.
Generating rather than hand-typing is deliberate: a 55-row table typed by hand
is a 55-row table with a missing row.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import audit  # noqa: E402

E, C, P, L, H = "entry", "corpus", "provenance", "legacy", "helper"

ROWS = {
    # ---- on the entry point, no corpus -----------------------------------
    "tests/test_theorems.py":
        (E, "eleven theorem witnesses; run first, and a failure stops the run"),
    "exp/boundary.py":
        (E, "kappa and n_0 for the two accounts; asserted against boundary.json"),
    "exp/alt600.py":
        (E, "the 600-correction replay under both protocols; asserted, not re-timed"),
    "exp/grid80.py":
        (E, "the 80-cell calibration grid; its stored run is asserted"),
    "exp/store_size.py":
        (E, "one cell held fixed while the store grows; the |D| confound"),
    "exp/masked_pipeline.py":
        (E, "the pipeline at the four published scopes under eight builds"),
    "exp/build_triple.py":
        (E, "bare kernels on that same store, under both compactions and gathers"),
    "exp/prim_cost.py":
        (E, "the two primitives and their crossover, on a synthetic array"),
    "exp/fig_ratio.py":
        (E, "Fig. 2: the ratio against scope, with paired bootstrap intervals"),
    "exp/layout_sensitivity.py":
        (E, "six value/id width pairs, standing in for a third machine"),
    "exp/sql_same_instance.py":
        (E, "DuckDB and SQLite on one instance, to price the candidate list"),
    "exp/sigma_selfpay.py":
        (E, "Prop. 4 measured in nesting order: mask, then update, then both"),
    "exp/n0_holdout/analyze.py":
        (E, "reduces the six-layout stores to Table VI; reads, does not re-time"),
    "exp/n0_robust/probe_rows.py":
        (E, "the rank probe re-expressed in the units of eq. (6)"),
    "exp/n0_impl/dline.py":
        (E, "solves eq. (6) for the decision term on the V-E table"),
    # ---- on the entry point, corpus-gated (make full) ---------------------
    "exp/degree.py":
        (C, "degree of the real edit set; needs data/mue and data/cms"),
    "exp/regime_bc.py":
        (C, "regimes (b) and (c) on the real threshold distribution; corpus"),
    "exp/delta.py":
        (C, "the published delta and the four things it does not determine"),
    "exp/cheap.py":
        (C, "the two cheaper certification structures, and the 13 of 98"),
    "exp/utilization.py":
        (C, "the CY2024 aggregate behind the emptiness certificate; corpus"),
    "exp/empty_cases.py":
        (C, "one line per certified-empty move; the slack a reader can recount"),
    # ---- run once, output stored, numbers pinned by expected.json ---------
    "exp/bigscope.py":
        (P, "bare paths at five scope sizes; its stored run is what V-D quotes"),
    "exp/realband.py":
        (P, "the band-density ceiling from published aggregates; stored"),
    "exp/fullpath_band_vs_scan.py":
        (P, "the six-density full-path comparison at 401 rows; stored"),
    "exp/grid_fixedD.py":
        (P, "the 70-cell grid at fixed |D|; supersedes the calibration confound"),
    "exp/same_store.py":
        (P, "one store, one |D|, one layout, measured side by side"),
    "exp/certificate_reach.py":
        (P, "how far the emptiness certificate reaches, and the S/B correlation"),
    "exp/corpus_point.py":
        (P, "n = 394 and k = 182 computed rather than asserted, from the corpus"),
    "exp/cost_model.py":
        (P, "one accounting for both paths; superseded in the paper by eq. (11)"),
    "exp/cross_machine.py":
        (P, "the second box; needs a second machine, so stored not re-run"),
    "exp/e2e_repro.py":
        (P, "whether the end-to-end pipeline is runnable today; a one-off probe"),
    "exp/fit_kappa_runs.py":
        (P, "turns the seven duel runs into the recorded kappa interval"),
    "exp/ivm.py":
        (P, "the IVM baseline; needs differential dataflow, so stored"),
    "exp/ivm_h2h/mkdata.py":
        (P, "dumps the instance for the Rust harness; a build step, not a run"),
    "exp/ivm_h2h/aggregate.py":
        (P, "folds three harness runs into the table the paper prints"),
    "exp/n0_holdout/n0_holdout.py":
        (P, "out-of-grid holdout for the cost model; needs a JDK, so stored"),
    "exp/n0_impl/scale.py":
        (P, "does the boundary move with |D|; stored, three sizes in one process"),
    "exp/n0_impl/scale_reps.py":
        (P, "fresh-process repetitions of the size sweep; 150 stored runs"),
    "exp/n0_robust/n0_robust.py":
        (P, "n_0 across six backends; needs the JVM and sqlite builds, stored"),
    "exp/n0_robust/fig_n0.py":
        (P, "two panels for the robustness section; reads the stored sweep"),
    "exp/n0_robust/fix_boot.py":
        (P, "re-does one bootstrap with the correct estimator; recorded as a fix"),
    "exp/path_r_h2h/path_r_h2h.py":
        (P, "path R on the instance Table V uses; needs the Rust harness"),
    "exp/rank_probe.py":
        (P, "what it costs to learn k; the source of the rank-probe term"),
    "exp/run_duel_big.py":
        (P, "the duel at larger N, seven runs; stored under exp/duel_big"),
    "exp/run_duel_repeat.py":
        (P, "the duel re-measured with repeated sweeps; the data behind kappa"),
    "exp/veto_duel_rot.py":
        (P, "the duel with arm order rotated inside each correction; 9 runs"),
    "exp/synpuf_gate.py":
        (P, "the DE-SynPUF feasibility gate; it killed that branch, and it is why"),
    # ---- superseded, kept for the record ---------------------------------
    "exp/bootstrap_kappa.py":
        (L, "superseded as the paper's error bar; the paired interval replaced it"),
    "exp/bootstrap_kappa_check.py":
        (L, "delta-method cross-check of the superseded bootstrap"),
    "exp/grid_calibration.py":
        (L, "the first 80-cell grid, whose stores grew with the scope; superseded"),
    "exp/n0_impl/legacy_withdrawn/scale_16m.py":
        (L, "withdrawn: one repetition of the 16.7M cell, wrong protocol"),
    "exp/n0_impl/legacy_withdrawn/scale_16m_reps.py":
        (L, "withdrawn with scale_16m.py; kept so the withdrawal is auditable"),
    # ---- tooling, not experiments ---------------------------------------
    "tools/audit.py":
        (H, "this ledger: three-way consistency, exits non-zero on a regression"),
    "tools/manifest_src.py":
        (H, "the editable source of manifest.tsv; edit ROWS here and re-run it"),
    "tools/set_url.py":
        (H, "fills the artifact URL into README, CITATION and .zenodo.json"),
    "tools/push.py":
        (H, "upload helper for the release archive; not part of reproduction"),
    "tools/push_api.py":
        (H, "the same upload over HTTP; not part of reproduction"),
}

disk = audit.disk_scripts()
have, want = set(disk), set(ROWS)
if have != want:
    print("MISMATCH")
    for s in sorted(have - want):
        print("  on disk, no row:   ", s)
    for s in sorted(want - have):
        print("  row, not on disk:  ", s)
    sys.exit(1)

# the manifest must not contradict reproduce.py
inv = audit.invoked_scripts()
on_entry = {p for p, (st, _) in ROWS.items() if st in ("entry", "corpus")}
if on_entry != inv:
    print("MISMATCH vs reproduce.py")
    for s in sorted(on_entry - inv):
        print("  manifest says on the entry point, never called:", s)
    for s in sorted(inv - on_entry):
        print("  called, manifest does not say so:            ", s)
    sys.exit(1)

lines = ["# path\tstatus\twhy",
         "# status: entry | corpus | provenance | legacy | helper",
         "# entry/corpus must be reachable from reproduce.py; provenance means a",
         "# stored run whose numbers expected.json pins; legacy is superseded.",
         "# prototype/ is excluded: a vendored driver tree, not part of the claim."]
for p in sorted(ROWS):
    st, why = ROWS[p]
    lines.append("%s\t%s\t%s" % (p, st, why))
with io.open(os.path.join(HERE, "manifest.tsv"), "w", encoding="utf-8",
             newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("wrote tools/manifest.tsv: %d rows (%d entry, %d corpus, %d provenance, "
      "%d legacy, %d helper)"
      % (len(ROWS),
         sum(1 for s, _ in ROWS.values() if s == "entry"),
         sum(1 for s, _ in ROWS.values() if s == "corpus"),
         sum(1 for s, _ in ROWS.values() if s == "provenance"),
         sum(1 for s, _ in ROWS.values() if s == "legacy"),
         sum(1 for s, _ in ROWS.values() if s == "helper")))
