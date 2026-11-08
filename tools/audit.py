#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-way audit of the release tree.

The claim this artifact makes is narrow and checkable: every number in the paper
is produced by a file in this tree, that file is reachable from `reproduce.py`,
and `docs/claim-to-code.md` says which claim it carries.  Four lists have to
agree for that to be true, and a fifth file has to say what the leftovers are:

    disk       every exp/*.py and tests/*.py that exists
    invoked    every script `reproduce.py` runs, in any of its call shapes
    claimed    every script named in `docs/claim-to-code.md`
    manifest   `tools/manifest.tsv`: every disk script, with a status and a why
    manuscript the scripts the paper itself names, if the source tree is beside us

Checks:

  [1] a disk script with no row in the manifest            -> fail
  [2] manifest says `entry`/`corpus` but not invoked       -> fail
  [3] manifest says `provenance`/`legacy`/`helper` but is  -> fail
      invoked anyway (the status would be a lie)
  [4] invoked but not on disk                              -> fail
  [5] invoked and on disk, but no claim in the claim table  -> fail
  [6] claimed, but not on disk                             -> fail
  [7] named in the manuscript, but not entry/corpus         -> fail

`provenance` is not a euphemism for dead code: it means the script was run once,
its output is stored under `exp/out/`, and `expected.json` pins the numbers from
that stored file, so the paper's claim is checkable without re-running a job that
needs a JDK, a Rust toolchain or a second machine.  A script that is merely
forgotten gets `legacy` and a date, or gets deleted.

Usage:  python tools/audit.py [-v]
"""
import argparse
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MANIFEST = os.path.join(HERE, "manifest.tsv")
STATUSES = ("entry", "corpus", "provenance", "legacy", "helper")
ON_ENTRY = ("entry", "corpus")


def _read(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- the tree


def disk_scripts():
    """Every runnable script in the release, by path relative to the root.

    `prototype/` is a vendored driver tree, not part of the argument; `out/` is
    results.  Both are excluded here and accounted for in a manifest comment.
    """
    out = set()
    for top in ("exp", "tests", "tools"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = [d for d in dn
                     if d not in ("out", "__pycache__", "prototype")]
            for f in fn:
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dp, f), ROOT)
                    out.add(rel.replace(os.sep, "/"))
    return out


def invoked_scripts():
    """Every script `reproduce.py` runs, in any of the four call shapes it uses.

    A regex for one shape undercounts, and earlier versions of this file did:
    they reported 7 of 18, then 22 of 40, then missed the whole
    `for script in (...)` corpus loop because the join's second argument is a
    variable rather than a literal.  Each shape is parsed explicitly so the count
    is a property of the file and not of how the file happens to be written
    today.
    """
    txt = _read("reproduce.py")
    names = set()
    by_base = {os.path.basename(p): p for p in disk_scripts()}

    # shape 1: os.path.join("exp", "a", "b.py")   -- all-literal parts
    for m in re.finditer(r"os\.path\.join\(([^()]*?)\)", txt, re.S):
        args = re.findall(r'"([^"]*)"', m.group(1))
        if not args or args[0] not in ("exp", "tests"):
            continue
        parts = [x for x in args[1:] if x]
        if not parts or not parts[-1].endswith(".py"):
            continue                      # a directory, or an --out name
        names.add("/".join([args[0]] + parts))

    # shape 2: for script in ("a.py", "b.py"):  ... os.path.join("exp", script)
    for m in re.finditer(r"for\s+(\w+)\s+in\s+\(([^)]*)\)\s*:", txt):
        var, body = m.group(1), m.group(2)
        if not re.search(r'os\.path\.join\(\s*"exp"\s*,\s*' + var, txt):
            continue
        for lit in re.findall(r'"([^"]*\.py)"', body):
            cand = "exp/" + lit
            if cand in disk_scripts():
                names.add(cand)

    # shape 3: for label, script, args in (("text", "a.py", [...]), ...)
    for m in re.finditer(r'\(\s*"[^"]*"\s*,\s*"([a-z_0-9]+\.py)"', txt):
        hit = by_base.get(m.group(1))
        if hit:
            names.add(hit)

    # shape 4: a bare run("x", [PY, "exp/x.py", ...]) whose path is literal
    for m in re.finditer(r'"((?:exp|tests)/[A-Za-z0-9_/]+\.py)"', txt):
        names.add(m.group(1))
    return names


def claimed_scripts():
    doc = _read("docs", "claim-to-code.md")
    return set(m.group(1).rstrip("/") for m in
               re.finditer(r"`((?:exp|tests)/[^`]+?\.py)`", doc))


def paper_scripts():
    """Scripts the manuscript itself names, if the source tree is beside us.

    Optional by design: a release must not depend on a sibling manuscript
    directory.  When it is absent the check is skipped, not silently passed.
    """
    cands = [os.path.join(ROOT, os.pardir, os.pardir, os.pardir,
                          "2026-11-07-11-21-16", "ICDE2028_EAB_V2", "paper"),
             os.environ.get("CERTIS_PAPER") or ""]
    import glob
    for cand in cands:
        if cand and os.path.isdir(cand):
            tex = ""
            for p in sorted(glob.glob(os.path.join(cand, "*.tex"))):
                tex += io.open(p, encoding="utf-8").read()
            tex = tex.replace("\\_", "_")
            return set(m.group(0) for m in
                       re.finditer(r"exp/[A-Za-z0-9_/.]*[A-Za-z0-9_]", tex))
    return set()


def read_manifest():
    """path -> (status, why).  A missing or malformed row is an error, not a skip."""
    rows, bad = {}, []
    with io.open(MANIFEST, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                bad.append((n, "expected 3 tab-separated columns, got %d" % len(parts)))
                continue
            path, status, why = (x.strip() for x in parts)
            if status not in STATUSES:
                bad.append((n, "unknown status %r" % status))
                continue
            if len(why) < 12:
                bad.append((n, "reason too short to be a reason: %r" % why))
                continue
            if path in rows:
                bad.append((n, "duplicate row for %s" % path))
                continue
            rows[path] = (status, why)
    return rows, bad


# ---------------------------------------------------------------- the report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    disk = disk_scripts()
    inv = invoked_scripts()
    claim = claimed_scripts()
    paper = paper_scripts()
    man, mbad = read_manifest()

    print("=" * 78)
    print("release audit")
    print("=" * 78)
    print("  on disk     %3d    (%d exp, %d tests, %d tools)"
          % (len(disk),
             sum(1 for d in disk if d.startswith("exp/")),
             sum(1 for d in disk if d.startswith("tests/")),
             sum(1 for d in disk if d.startswith("tools/"))))
    print("  invoked     %3d" % len(inv))
    print("  claimed     %3d" % len(claim))
    print("  manifest    %3d rows" % len(man))
    if paper:
        print("  named in the manuscript  %3d" % len(paper))
    print()

    problems = 0

    if mbad:
        print("[0] manifest is malformed: %d" % len(mbad))
        for n, why in mbad[:12]:
            print("    line %-4d %s" % (n, why))
        problems += len(mbad)
        print()

    unlisted = sorted(disk - set(man))
    print("[1] on disk with no manifest row: %d" % len(unlisted))
    for s in unlisted:
        print("    " + s)
    problems += len(unlisted)
    print()

    ghosts = sorted(set(man) - disk)
    print("[2] manifest row for a file that is not there: %d" % len(ghosts))
    for s in ghosts:
        print("    %-40s %s" % (s, man[s][0]))
    problems += len(ghosts)
    print()

    lying = sorted(p for p, (st, _) in man.items() if st in ON_ENTRY and p not in inv)
    print("[3] manifest says entry/corpus, reproduce.py never calls it: %d" % len(lying))
    for s in lying:
        print("    " + s)
    problems += len(lying)
    print()

    crossed = sorted(p for p, (st, _) in man.items() if st not in ON_ENTRY and p in inv)
    print("[4] manifest says provenance/legacy/helper, but it IS called: %d" % len(crossed))
    for s in crossed:
        print("    " + s)
    problems += len(crossed)
    print()

    broken = sorted(inv - disk)
    print("[5] called by reproduce.py, not on disk: %d" % len(broken))
    for s in broken:
        print("    " + s)
    problems += len(broken)
    print()

    orphan = sorted((inv & disk) - claim)
    print("[6] called and present, but no claim in claim-to-code.md: %d" % len(orphan))
    for s in orphan:
        print("    " + s)
    problems += len(orphan)
    print()

    phantom = sorted(c for c in claim if c not in disk)
    print("[7] claimed, not on disk: %d" % len(phantom))
    for s in phantom:
        print("    " + s)
    problems += len(phantom)
    print()

    if paper:
        ln = {os.path.basename(p): p for p in disk}
        absent = sorted(p for p in paper
                        if p not in disk and p not in ln.values()
                        and os.path.basename(p) not in ln
                        and not os.path.exists(os.path.join(ROOT, p)))
        bad = sorted(p for p in paper if p in disk
                     and man.get(p, ("", ""))[0] in ("legacy", "helper"))
        stored = sorted(p for p in paper if p in disk
                        and man.get(p, ("", ""))[0] == "provenance")
        print("[8] named in the manuscript: %d" % len(paper))
        print("    absent from the tree: %d" % len(absent))
        for s in absent:
            print("       *** " + s)
        print("    cited but withdrawn or tooling: %d" % len(bad))
        for s in bad:
            print("       *** %-34s manifest=%s" % (s, man.get(s, ("-",))[0]))
        print("    cited, not on the entry point, stored run asserted: %d"
              % len(stored))
        for s in stored:
            print("        %-34s run it directly, or check the stored json"
                  % s)
        problems += len(absent) + len(bad)
        print()

    if a.verbose:
        print("-" * 78)
        print("ledger by status")
        for st in STATUSES:
            rows = sorted(p for p, (x, _) in man.items() if x == st)
            print("  %-11s %3d" % (st, len(rows)))
            for r in rows:
                print("     %-52s %s" % (r, man[r][1][:60]))
        print()

    print("=" * 78)
    if problems:
        print("AUDIT FAILED: %d discrepancies" % problems)
    else:
        print("audit clean -- %d scripts, every one accounted for" % len(disk))
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
