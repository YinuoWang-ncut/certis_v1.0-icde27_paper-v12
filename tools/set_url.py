#!/usr/bin/env python
"""Fill in the repository URL everywhere a placeholder stands in for it.

    python tools/set_url.py https://github.com/<user>/certis
    python tools/set_url.py --check        # exit 1 if any placeholder survives
    python tools/set_url.py --clear        # put the paper back on the placeholder

Two placeholder mechanisms exist, and they have to be filled together, or the
PDF ships with an address the repository does not answer at:

  * ``__REPO_URL__`` / ``__REPO_SLUG__`` --- plain text tokens in README.md,
    CITATION.cff, .zenodo.json, docs/, and any other text file that carries one.
  * ``\\artifactslug`` / ``\\ifartifactset`` in ``paper/main.tex`` --- the paper
    prints a red ``[address not yet substituted]`` next to the link until the
    flag is flipped, so a forgotten substitution shows on the page rather than
    only on a 404.

Run ``--check`` before tagging; the CI workflow runs the same check, so a
forgotten placeholder fails the build rather than the review.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = ("__REPO_URL__", "__REPO_SLUG__")
TEXT_EXT = {".md", ".txt", ".tex", ".cff", ".json", ".yml", ".yaml", ".cfg",
            ".toml", ".py", ".bib", ".rst", ".html"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "_build"}
SKIP_FILES = {os.path.join(ROOT, "tools", "set_url.py")}

RE_URL = re.compile(r"(\\newcommand\{\\artifacturl\}\{)([^}]*)(\})")
RE_DISPLAY = re.compile(r"(\\newcommand\{\\artifactdisplay\}\{)([^}]*)(\})")
RE_FLAG = re.compile(r"(\\newif\\ifartifactset\s*\\artifactset)(true|false)")
PLACEHOLDER = "YOUR-GITHUB-USER/certis"


def paper_path():
    """Where main.tex is.  It is not always inside the release tree: the
    manuscript and the artifact are separate deliverables, and a release must
    not fail to check merely because they sit in different directories."""
    cands = [os.environ.get("CERTIS_PAPER") or "",
             os.path.join(ROOT, "paper", "main.tex"),
             os.path.join(ROOT, os.pardir, os.pardir, os.pardir,
                          "2026-11-07-11-21-16", "ICDE2028_EAB_V2",
                          "paper", "main.tex")]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def display_of(url):
    r"""`https://github.com/a/b_c` -> `github.com/\allowbreak a/\allowbreak b\_c`."""
    body = url.split("://", 1)[-1].rstrip("/")
    parts = body.split("/")
    esc = [p.replace("_", r"\_") for p in parts]
    if len(esc) >= 3:
        return "%s/\\allowbreak %s/\\allowbreak %s" % (esc[0], esc[1], esc[2])
    return "/".join(esc)


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def text_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1].lower() in TEXT_EXT and path not in SKIP_FILES:
                out.append(path)
    return sorted(out)


def paper_state():
    """(url, flag_bool) as written in the manuscript's main.tex, or None."""
    p = paper_path()
    if p is None:
        return None
    s = read(p)
    mu, mf = RE_URL.search(s), RE_FLAG.search(s)
    if not mu or not mf:
        return None
    return mu.group(2), mf.group(2) == "true"


def set_paper(url, flag):
    p = paper_path()
    if p is None:
        return False
    s = read(p)
    s, n1 = RE_URL.subn(lambda m: m.group(1) + url + m.group(3), s, count=1)
    s, n2 = RE_DISPLAY.subn(lambda m: m.group(1) + display_of(url) + m.group(3),
                            s, count=1)
    s, n3 = RE_FLAG.subn(lambda m: m.group(1) + ("true" if flag else "false"),
                         s, count=1)
    if (n1, n2, n3) != (1, 1, 1):
        print("%s: could not rewrite (url %d, display %d, flag %d)"
              % (p, n1, n2, n3))
        return False
    write(p, s)
    return True


def check():
    bad = []
    for p in text_files():
        s = read(p)
        for t in TOKENS:
            if t in s:
                bad.append("%s: %s" % (os.path.relpath(p, ROOT), t))
    st = paper_state()
    if st is None:
        bad.append("main.tex: no \\artifacturl / \\ifartifactset found, or the "
                   "manuscript is not on CERTIS_PAPER")
    elif not st[1]:
        bad.append("main.tex: \\artifactsetfalse (%s)" % st[0])
    if bad:
        print("placeholder(s) still present:")
        for b in bad:
            print("  " + b)
        return 1
    print("no placeholders left")
    return 0


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    arg = argv[1].strip().rstrip("/")

    if arg == "--check":
        return check()

    if arg == "--clear":
        if paper_path() is None:
            print("main.tex not found; set CERTIS_PAPER to its directory")
            return 2
        ok = set_paper(PLACEHOLDER, False)
        print("paper on placeholder; recompile to see the red warning again")
        return 0 if ok else 1

    if not arg.startswith("http"):
        print("expected an http(s) URL, got %r" % arg)
        return 2
    slug = "/".join(arg.rstrip("/").split("/")[-2:])

    changed = []
    for p in text_files():
        s = read(p)
        n = s.replace("__REPO_URL__", arg).replace("__REPO_SLUG__", slug)
        if n != s:
            write(p, n)
            changed.append(os.path.relpath(p, ROOT))

    if paper_path() is not None:
        before = paper_state()
        if before is None:
            print("main.tex: no \\artifacturl / \\ifartifactset found")
            return 1
        set_paper(arg, True)
        if before != (arg, True):
            changed.append("main.tex")

    for c in changed:
        print("updated %s" % c)
    print("repo url: %s" % arg)
    print("now: pdflatex three times (paper/), then python tools/set_url.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
