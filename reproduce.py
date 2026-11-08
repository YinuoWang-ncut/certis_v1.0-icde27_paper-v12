#!/usr/bin/env python
"""Re-derive every number in the paper, and fail loudly if one moved.

    python reproduce.py             everything (needs the 27 CMS files)
    python reproduce.py --no-data   only what takes no corpus: the eleven
                                    witnesses and the boundary arithmetic

Two kinds of check, deliberately separated:

  assert    deterministic given the inputs.  Row counts, pair counts, the
            number of distinct MUE caps, how many of the 49 threshold
            placements give a zero band, the kappa/n_0 arithmetic.  A mismatch
            is a bug in the paper or in this code, and the exit status is 1.

  record    timings.  Printed next to the paper's values, never asserted: they
            are a property of the host, and the paper says so.  See README.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python"
TOL = 1e-6

# dotted paths with list indices, the same syntax _make_artifact.py writes
TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")


def v_of(tree, path):
    """Resolve `configs[0].kappa` against a parsed json document."""
    cur = tree
    for part in TOKEN.findall(path):
        if part.startswith("["):
            idx = int(part[1:-1])
            cur = cur[idx] if isinstance(cur, list) and len(cur) > idx else None
        else:
            cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


# --------------------------------------------------------------------------
# assertion walker
# --------------------------------------------------------------------------
def compare(expect, actual, path, fails):
    """Walk `expect` against `actual`; leaves compare with a relative tolerance."""
    if isinstance(expect, dict):
        if not isinstance(actual, dict):
            fails.append("%s: expected an object, got %r" % (path, type(actual)))
            return
        for k, v in expect.items():
            if k not in actual:
                fails.append("%s.%s: missing" % (path, k))
                continue
            compare(v, actual[k], "%s.%s" % (path, k), fails)
        return
    if isinstance(expect, list):
        if not isinstance(actual, list) or len(actual) != len(expect):
            fails.append("%s: list shape differs" % path)
            return
        for i, (e, a) in enumerate(zip(expect, actual)):
            compare(e, a, "%s[%d]" % (path, i), fails)
        return
    if isinstance(expect, bool) or isinstance(actual, bool):
        ok = expect == actual
    elif isinstance(expect, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(expect - actual) <= TOL * max(1.0, abs(expect))
    else:
        ok = expect == actual
    if not ok:
        fails.append("%s: expected %r, got %r" % (path, expect, actual))


def run(name, argv, cwd):
    print("\n$ %s" % " ".join(argv))
    r = subprocess.run(argv, cwd=cwd)
    if r.returncode != 0:
        print("-- %s exited %d" % (name, r.returncode))
    return r.returncode == 0


def have_data():
    man = os.path.join(ROOT, "data", "manifest.json")
    if not os.path.exists(man):
        return False, "no data/manifest.json"
    rows = json.load(io.open(man, encoding="utf-8"))
    base = os.environ.get("CERTIS_DATA") or os.path.join(ROOT, "data")
    missing = []
    for r in rows:
        rel = r["path"][len("data/"):] if r["path"].startswith("data/") else r["path"]
        if not os.path.exists(os.path.join(base, rel)):
            missing.append(r["path"])
    if missing:
        return False, "%d of %d files missing under %s (run data/fetch.py)" % (
            len(missing), len(rows), base)
    return True, "%d files present under %s" % (len(rows), base)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true",
                    help="skip everything that needs the CMS tables")
    a = ap.parse_args()

    ok = True
    print("=" * 72)
    print("witnesses and arithmetic -- no corpus needed")
    print("=" * 72)
    ok &= run("test_theorems", [PY, os.path.join("tests", "test_theorems.py")], ROOT)
    ok &= run("boundary", [PY, os.path.join("exp", "boundary.py")], ROOT)

    if a.no_data:
        print("\n--no-data: skipping the corpus-dependent steps")
    else:
        ready, why = have_data()
        print("\ncorpus: %s" % why)
        if not ready:
            print("cannot run the corpus steps; see README")
            ok = False
        else:
            for script in ("degree.py", "regime_bc.py", "delta.py", "cheap.py",
                           "utilization.py", "empty_cases.py"):
                ok &= run(script, [PY, os.path.join("exp", script)], ROOT)

    # Table VII and the decision line of Sect. V-E, from numbers already stored:
    # n0_holdout/analyze.py reduces the corrected bare-layout measurement to the
    # summary Table VII prints, probe_rows.py re-expresses the rank probe in the
    # units of (6), and dline.py solves (6) for the decision term.  None of the
    # three re-times anything, so they belong to the corpus-free set.  To re-time
    # the six layouts instead of reading the stored run, see exp/n0_holdout
    # (needs a JDK; about 90 s).
    print("\n" + "=" * 72)
    print("Table VI, the probe rows, and the decision line -- no corpus needed")
    print("=" * 72)
    ok &= run("n0_holdout/analyze", [PY, os.path.join("exp", "n0_holdout", "analyze.py")], ROOT)
    ok &= run("n0_robust/probe_rows", [PY, os.path.join("exp", "n0_robust", "probe_rows.py")], ROOT)
    ok &= run("n0_impl/dline", [PY, os.path.join("exp", "n0_impl", "dline.py")], ROOT)

    # The build triple goes first among the timing blocks, before grid80 and
    # store_size have allocated their stores.  On a box that has just walked
    # several gigabytes the two compactions converge -- `compress` is
    # bandwidth-bound and the boolean index is not -- and the spread this block
    # exists to measure shrinks from 0.32-1.08 to 0.26-0.72.  That is a real
    # property of the measurement, not noise, and the honest place to take it is
    # on the idle box.
    print("\n" + "=" * 72)
    print("Sect. V-A, the build triple, on the idle box")
    print("=" * 72)
    ok &= run("build_triple", [PY, os.path.join("exp", "build_triple.py"),
                               "--reps", "25",
                               "--out", os.path.join("exp", "out", "_repro_build_triple.json")], ROOT)

    # The build triple: the same store, with the mask and the probe removed, so
    # the primitive choice can be priced on its own.  Sect. V-A quotes these
    # numbers beside the pipeline's and the two are different instruments; a
    # script per instrument is the only way to keep that straight.
    for label, script, args in (
            ("Sect. V-A, the two primitives and the crossover", "prim_cost.py",
             ["--out", os.path.join("exp", "out", "_repro_prim_cost.json")]),
            ("Fig. 2, the ratio against scope", "fig_ratio.py",
             ["--out", os.path.join("exp", "out", "_repro_fig_ratio.json"),
              "--figdir", os.path.join("exp", "out", "_repro_figs"),
              "--no-copy"])):
        print()
        print("=" * 72)
        print(label)
        print("=" * 72)
        ok &= run(script[:-3], [PY, os.path.join("exp", script)] + args, ROOT)

    # The build, the engines, the widths: three blocks added for the ICDE
    # revision.  masked_pipeline re-times (recorded, not asserted); the two
    # engine and width blocks need duckdb / sqlite3 from the standard library.
    for label, script, args in (
            ("Sect. V-A, the pipeline under eight builds", "masked_pipeline.py",
             ["--reps", "15", "--sweep-reps", "7",
              "--out", os.path.join("exp", "out", "_repro_masked15.json")]),
            ("Sect. V-A, one instance and two engines", "sql_same_instance.py",
             ["--reps", "15",
              "--out", os.path.join("exp", "out", "_repro_sql_same.json")]),
            ("Sect. V-A, six width pairs", "layout_sensitivity.py",
             ["--reps", "15",
              "--out", os.path.join("exp", "out", "_repro_layout.json")])):
        print()
        print("=" * 72)
        print(label)
        print("=" * 72)
        ok &= run(script[:-3], [PY, os.path.join("exp", script)] + args, ROOT)

    # Prop. 4: what sigma maintenance costs, measured in nesting order so each
    # delta is one more pass over the same identifiers.  Re-times, so recorded.
    print("\n" + "=" * 72)
    print("Prop. 4, sigma maintenance priced by stage (re-times)")
    print("=" * 72)
    ok &= run("sigma_selfpay", [PY, os.path.join("exp", "sigma_selfpay.py"),
                                "--reps", "9",
                                "--out", os.path.join("exp", "out", "_repro_sigma_selfpay.json")], ROOT)

    # The pipeline at the published scope sizes: the cell Sec. V-A and Sec. V-D
    # are two ends of.  fullpath_band_vs_scan times the five stages at 401 rows,
    # bigscope times bare paths up to 412,875; this runs the five stages --
    # mask, dedup, overlay, selector -- at all four published sizes, so the two
    # are points on one curve rather than two objects.  It re-times, so it is
    # recorded and not asserted.
    print("\n" + "=" * 72)
    print("Sec. V-A, the pipeline at the published scope sizes (re-times)")
    print("=" * 72)
    # --out sends the re-run to a scratch file.  Without it this step overwrote
    # the curated exp/out/masked_pipeline.json, so every smoke run replaced the
    # numbers the paper prints with one draw from a ±2%-per-cell distribution
    # and the record stopped matching Sect. V-A.
    ok &= run("masked_pipeline", [PY, os.path.join("exp", "masked_pipeline.py"),
                                  "--reps", "13",
                                  "--out", os.path.join("exp", "out", "_repro_masked.json")], ROOT)

    # The two-protocol replay of Sect. V-D: alt600.py reduces the 30 stored
    # per-process run records (7 recorded-order, 23 alternating-order) to the
    # per-correction verdicts and the tuple the paper prints.  It re-times
    # nothing, so it belongs to the corpus-free set; exp/out/alt600.json is
    # asserted below, which is what makes "kappa = 0.214, n_0 = 959" checkable.
    # The 80-cell calibration grid and the store-size confound.  Both re-time
    # the pipeline, so their ratio fields go to `recorded`; what is asserted is
    # the structure -- the grid's shape, the delta-completeness of every cell,
    # and the monotonicity of the confound.
    print("\n" + "=" * 72)
    print("Sect. V-E, the 80-cell grid and the store-size confound")
    print("=" * 72)
    # --out keeps the curated out/grid80.json intact: a low-repetition re-run
    # is a smoke test of the harness, not a replacement for the stored record,
    # and the cell-level agreement it prints moves by +-2 cells between runs.
    ok &= run("grid80", [PY, os.path.join("exp", "grid80.py"), "--reps", "5",
                        "--out", os.path.join("exp", "out", "_repro_grid80.json")], ROOT)
    # The fixed-|D| grid behind Fig. 1 (left) and the density edge of Sect. V-D.
    # Its store is built once and every cell shares it, so n is the only thing
    # that moves between two columns; grid80.py above builds one store per
    # scope and therefore cannot answer the same question.  --out keeps the
    # curated record, whose sign counts are what the text prints.
    # Sect. V-A's four scopes at five store sizes.  The four published numbers
    # are a slice at 1.39e6 rows; this asks how much of the ratio is the store,
    # and it shares masked_pipeline's builder, arms and build rule so the
    # 1.39e6 case is a control and not a second experiment.
    ok &= run("scope_vs_D", [PY, os.path.join("exp", "scope_vs_D.py"), "--reps", "11",
                             "--out", os.path.join("exp", "out", "_repro_scope_vs_D.json")], ROOT)
    ok &= run("same_store", [PY, os.path.join("exp", "same_store.py"), "--reps", "11",
                             "--out", os.path.join("exp", "out", "_repro_same_store.json")], ROOT)
    ok &= run("store_size", [PY, os.path.join("exp", "store_size.py"),
                             "--out", os.path.join("exp", "out", "_repro_store_size.json")], ROOT)

    print("\n" + "=" * 72)
    print("Sect. V-D, the 600 corrections under both protocols -- no corpus needed")
    print("=" * 72)
    ok &= run("alt600", [PY, os.path.join("exp", "alt600.py")], ROOT)

    # ----------------------------------------------------------------------
    path = os.path.join(ROOT, "expected.json")
    if not os.path.exists(path):
        print("no expected.json; nothing to assert")
        return 0 if ok else 1
    exp = json.load(io.open(path, encoding="utf-8"))

    fails = []
    print("\n" + "=" * 72)
    print("assert: deterministic values")
    print("=" * 72)
    for rel, tree in sorted(exp.get("assert", {}).items()):
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            if a.no_data:
                print("%-24s skipped (no corpus)" % rel)
                continue
            fails.append("%s: not produced (was the step skipped?)" % rel)
            continue
        got = json.load(io.open(fp, encoding="utf-8"))
        before = len(fails)
        compare(tree, got, rel, fails)
        print("%-24s %s" % (rel, "ok" if len(fails) == before else "MISMATCH"))

    print("\n" + "=" * 72)
    print("record: host-dependent, printed not asserted")
    print("=" * 72)
    print("  %-42s %-12s %s" % ("value", "recorded", "this host"))
    for rel, tree in sorted(exp.get("recorded", {}).items()):
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            print("%-24s absent -- run %s to record this host's values"
                  % (rel, rel[:-5] + ".py"))
            continue
        got = json.load(io.open(fp, encoding="utf-8"))
        for k in sorted(tree):
            v = v_of(got, k)
            if v is None:
                # The key is not in the stored file.  Printing an empty column
                # here is how four stale keys survived a release: the table
                # looked like a host that had drifted, not like a typo.
                fails.append("%s: no key %r (stale key, or the producer "
                             "changed schema)" % (rel, k))
                print("  %-42s %-12s %s" % (k, tree[k], "*** NO SUCH KEY ***"))
                continue
            print("  %-42s %-12s %s" % (k, tree[k], v))

    if fails:
        print("\n%d assertion(s) failed:" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    print("\nall assertions hold")
    if not ok:
        # An assertion can hold while the block that produced its input exited
        # non-zero -- a missing optional dependency, a crashed driver.  Without
        # this line the log reads as a pass and the exit code says otherwise.
        print("but at least one block above exited non-zero; see the lines "
              "beginning '--'")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
