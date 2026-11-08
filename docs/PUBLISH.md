# Publishing this artifact for ICDE 2027

The manuscript (v10) points at `\artifactaddress`. Until that substitution is
made, the PDF prints a red `[address not yet substituted]` and the paper has no
retrievable artifact — which the ICDE 2027 call makes disqualifying for the
Experiment, Analysis and Benchmark category, with no exceptions.

Everything below is on this machine already. The tree is committed and tagged
`v1.0-icde27`, the archive built from that tag passes `reproduce.py --no-data`
from a clean extraction, and the only thing missing is a public URL.

## 0. What the URL must point at, and what it must not

The paper says *"the release is the git tree tagged `v1.0-icde27`"*. A fixed
archive cannot satisfy that sentence: a `.zip` has no tag, no history, and no
way for a reader to check out the bytes that were reviewed. So the address is
the **repository** or the **tagged tree**, or a **DOI** of that release — never
a `.zip`/`.tar`, and never a link that needs a login.

`tools/set_url.py` refuses archive URLs for exactly this reason, so the mistake
cannot reach the PDF.

| what you paste into `set_url.py` | what a reviewer sees |
|---|---|
| `https://github.com/<you>/certis` | the repository, tag selectable in the branch menu |
| `https://github.com/<you>/certis/tree/v1.0-icde27` | exactly the reviewed tree |
| `https://doi.org/10.5281/zenodo.<id>` | permanent record of that tree |

The tag URL is the safest of the three: it lands on the reviewed bytes and
survives later commits to `main`.

An archive is still worth publishing as an **extra mirror** — reviewers behind
a firewall, or artifact chairs who want one download — but it is not the
address, and nothing in the paper should claim it is.

## 0b. The state of the published repository, and the one thing that must change

As of paper v13 the remote holds **one commit containing one file**, the
archive `certis_v1.0-icde27_paper-v12.zip`, and **no tag**. That satisfies
"openly accessible" but not the sentence the paper prints ("the git tree tagged
`v1.0-icde27`"), and it has a sharper problem: the archive there was built
*before* Sect. V-E gained its 80-cell grid, so it does not contain
`exp/grid80.py`, `exp/store_size.py`, `docs/density-form.md` or
`exp/out/grid80_runs/`. A reader who downloads it cannot find the scripts the
paper cites.

Two ways to fix it. Either is acceptable; the first is better.

**A. Push the tree and the tag** (makes the paper's sentence literally true, and
gives a browsable repository):

```bash
cd release/certis
git remote add origin https://github.com/YinuoWang-ncut/certis_v1.0-icde27_paper-v12.git
git push --force origin main          # replaces the one-commit, one-file history
git push --force origin v1.0-icde27   # the tag the paper names
git ls-remote --tags origin           # must list refs/tags/v1.0-icde27
```

`--force` is required because the remote history is unrelated to this one. The
replaced commit stays reachable in GitHub's reflog for 90 days.

**B. Replace the file, keep the repository as it is** (one minute, works with the
URL already in the paper):

Upload the rebuilt `certis_v1.0-icde27_paper-v12.zip` over the old one **with
the same filename**, so the repository path does not change. The URL in the
paper keeps working and the bytes behind it become the reviewed ones.

Either way, check that the download contains the four paths above before
submitting: `unzip -l certis_v1.0-icde27_paper-v12.zip | grep -E 'grid80|store_size|density-form|grid80_runs'`.

## 0c. State after the URL was filled (paper v13)

The paper now prints `github.com/YinuoWang-ncut/certis_v1.0-icde27_paper-v12`
and `tools/set_url.py --check` exits 0. Two things remain on your side.

1. **The archive behind that URL is the v12 build, not the reviewed one.** It
   predates the 80-cell grid, so it lacks `exp/grid80.py`,
   `exp/store_size.py`, `docs/density-form.md` and `exp/out/grid80_runs/`.
   Upload the rebuilt `certis_v1.0-icde27_paper-v12.zip` over it under the same
   filename; the URL then serves the reviewed bytes with no change to the paper.
2. **The repository still has no tag.** The paper says "the tree released as
   `v1.0-icde27`", which the archive filename supports, but a reviewer who looks
   for a tag will not find one. Push the tree and the tag (section 1) to make it
   verifiable; nothing in the paper needs to change either way.

## 0d. Two ways to push, because one of them is blocked from some networks

`git push` speaks smart-HTTP to **github.com**. On a machine behind an egress
proxy that host is intermittently blocked -- measured here as 5 successes out of
6 tries in one minute, the failure being
`CONNECT tunnel failed, response 502` -- while **api.github.com** stays
reachable. That is a property of the network, not of the repository, and it has
an easy test:

```bash
python tools/push.py --check     # resolves the remote, compares trees, changes nothing
```

Two scripts, same result: a commit whose tree is the reviewed one, and the tag.

| script | talks to | needs | use when |
|---|---|---|---|
| `tools/push.py` | github.com (git) | a logged-in credential helper | normal network, e.g. your own shell |
| `tools/push_api.py` | api.github.com (Git Data API) | `GITHUB_TOKEN`, fine-grained, Contents: read and write | `git push` fails with 502, or no credential helper |

Both end with the same check, and it is the only one worth trusting: the tree
GitHub reports for `v1.0-icde27` has to equal `git rev-parse v1.0-icde27^{tree}`
locally. `push_api.py` performs that comparison **before** it moves any ref, so a
mismatch leaves the branch untouched instead of publishing something close.

```bash
# ordinary path
python tools/push.py

# restricted-network path; the token is read from the environment, never echoed
export GITHUB_TOKEN=github_pat_...
python tools/push_api.py --plan     # 386 blobs -> 1 tree -> 1 commit -> tag
python tools/push_api.py
```

The token needs no other scope: Contents read/write is enough for blobs, trees,
commits, refs and tags. Revoke it once the push is done; nothing in the artifact
stores it.

## 1. Push the commit and the tag

Already done locally: commit `619ae0e` and annotated tag `v1.0-icde27`. What
remains is a remote.

```bash
cd release/certis

git remote add origin https://github.com/<you>/certis.git
git push -u origin main
git push origin v1.0-icde27
```

Check that the tag is on the remote before filling the URL in:
`git ls-remote --tags origin` should list `refs/tags/v1.0-icde27`.

## 2. Archive it and get a DOI (recommended, not required)

Zenodo can mint a DOI from the tag and `.zenodo.json` is already in the tree:

1. Sign in to zenodo.org with GitHub, enable the repository under *Settings → GitHub*.
2. Publish a release from tag `v1.0-icde27`.
3. Copy the DOI badge URL.

A DOI survives repository deletion and is what an artifact evaluator expects;
a bare GitHub URL is acceptable under the call (`"standard, openly accessible
file sharing services"`) but the DOI is better.

## 3. Put the address in the paper

```bash
cd ../..                       # back to ICDE2028_EAB_V2
python tools/set_url.py https://github.com/<you>/certis/tree/v1.0-icde27
python tools/set_url.py --check     # must exit 0
python build.py                     # prints the pages and the overfull count
```

`set_url.py` rewrites `\artifactslug` and raises `\artifactsettrue`, so the PDF
prints a live link instead of the red slot. `--check` exits non-zero while the
slot is unfilled: wire it into whatever builds the submission PDF and the
mistake becomes impossible rather than merely unlikely.

If the DOI is preferred over the repository URL, pass the DOI URL instead —
the script does not care which.

## 4. Verify before submitting

```bash
cd release/certis
python reproduce.py --no-data      # must end with "all assertions hold", exit 0
```

That run covers the eleven theorem witnesses, the boundary arithmetic, the six
bare layouts, the probe rows, the decision line, and the two-protocol replay of
Sect. V-D — every number the paper prints that does not need the corpus. With
the 27 CMS files fetched (`python data/fetch.py`), plain `python reproduce.py`
adds the corpus blocks.

Three blocks are printed rather than asserted, because they are timings:
`exp/e2e_repro.py`, `exp/fullpath_band_vs_scan.py` and `exp/bigscope.py`. They
need no environment variable — they find the prototype under `exp/prototype/`
on their own — and each takes about a minute.

## 5. What is not in the tree, and why

* **The 27 CMS releases (~250 MB).** CPT content is AMA property, so they are
  fetched and SHA-256-verified by `data/fetch.py` against `data/manifest.csv`
  rather than redistributed. The manifest is the guarantee, not the bytes.
* **The ~10 GB utilisation release.** `exp/utilization.json` is the reference
  run over its 9.78M rows, shipped in place of the file.
* **The ingest path from raw CMS downloads to the corpus store.** Not needed:
  every store the shipped scripts build is synthetic and self-contained.

## 6. Sanity checks already done here

| check | result |
|---|---|
| `python reproduce.py --no-data` | exit 0, "all assertions hold" |
| every script the Artifact paragraph names | present under `exp/` |
| no absolute path or drive letter in any shipped script | confirmed by grep |
| `exp/alt600.py` from the shipped records | κ 0.214 / 0.441, n₀ 959 / 717 — matches the paper |
| `reproduce.py --no-data` from a clean extraction of the v12 archive | exit 0, "all assertions hold", 10 of 10 assert blocks ok |
| archive built with `git archive v1.0-icde27` | no `_repro.log`, no `.git`, 388 entries, 2.0 MB |
| `set_url.py` with a `.zip` URL | exits 1 with an explanation, so an archive cannot become the address |
| tree size, excluding `.git` | ~6 MB plus 2.2 MB of run records |
