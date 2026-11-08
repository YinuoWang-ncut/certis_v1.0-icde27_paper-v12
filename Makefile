PY ?= python

.PHONY: help check audit verify url fetch full clean

help:
	@echo "make url URL=https://github.com/<user>/certis   # fill the repository URL in"
	@echo "make check   # reproduce.py --no-data: no corpus, no downloads, ~1 min"
	@echo "make audit   # every script accounted for: disk vs reproduce vs claims"
	@echo "make verify  # audit, then check -- what a reviewer should run"
	@echo "make release # verify, after failing if a placeholder survives"
	@echo "make fetch   # download and SHA-256 verify the 27 CMS releases (~250 MB)"
	@echo "make full    # fetch, then everything, including the corpus steps"
	@echo "make clean   # drop caches"

url:
	@test -n "$(URL)" || (echo "usage: make url URL=https://github.com/<user>/certis"; exit 2)
	$(PY) tools/set_url.py $(URL)

# What a reviewer runs, and what CI runs on every push: no corpus, no
# network, 63 s on the author's box -- the timing blocks are most of it.
# The URL placeholder is deliberately not checked here --
# it is empty until the release is tagged, and an empty placeholder must not
# read as a failed reproduction.
check:
	$(PY) reproduce.py --no-data

# The ledger: every exp/ and tests/ script has a row in tools/manifest.tsv, the
# row agrees with what reproduce.py actually calls, every script it calls has a
# claim in docs/claim-to-code.md, and every script the manuscript names is
# reachable.  Exits non-zero on the first disagreement, so a new script that
# nobody wired in cannot ship quietly.
audit:
	$(PY) tools/audit.py

# What a reviewer should run, and what CI runs on every push.
verify: audit check

# What has to pass before a public tag.
release:
	$(PY) tools/set_url.py --check
	$(PY) tools/audit.py
	$(PY) reproduce.py --no-data

fetch:
	$(PY) data/fetch.py

full: fetch
	$(PY) reproduce.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
