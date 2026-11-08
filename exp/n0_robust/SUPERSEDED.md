# Superseded

Table VII is now computed from `../n0_holdout/`. This directory is kept only so
the change can be inspected: the JVM arm timed a value-sorted table from row 0
(pass count 26,176 vs 0), the band was drawn as the k smallest identifiers
(pass rate 0% for k <= 4096), and the predicate expression was 3.6x slower than
its equivalent. See README.md, section *Corrections*.

`fix_boot.py` still re-fits these numbers; `reproduce.py` no longer calls it.
