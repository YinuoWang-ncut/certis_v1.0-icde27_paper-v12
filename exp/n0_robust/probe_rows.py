# -*- coding: utf-8 -*-
"""probe_rows.py -- the rank probe expressed in the units of eq. (6).

n_0 is defined by  k < kappa(n - n_0),  i.e.  (a_p+a_d)k + F_B < (a_s+a_d)n + F_R.
Charging a probe P on the band side shifts n_0 by P/(a_p+a_d) -- the band's own
per-candidate cost -- and NOT by P/a_s, which is what n0_eff in n0_robust_fix.json
does.  This script re-expresses every configuration in the correct unit so the
paper's sentence ("moves n_0 by ... rows") is reproducible from the artifact.

Since a_p = a_s/kappa in the bare fits, P/a_p = kappa * (P/a_s) = kappa * (n0_eff - n0).
"""
import io, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "n0_robust_fix.json")
OUT = os.path.join(HERE, "probe_rows.json")


def main():
    d = json.load(io.open(SRC, encoding="utf-8"))
    rows = []
    for c in d["configs"]:
        kappa = c["kappa"]
        n0 = c["n0"]
        n0_eff_asj = c["n0_eff"]              # P / a_s  (wrong unit)
        shift_asj = n0_eff_asj - n0
        shift_p = kappa * shift_asj           # P / a_p  (eq. 6 unit)
        rows.append({
            "backend": c["backend"],
            "kappa": kappa,
            "n0": n0,
            "probe_rows_asj": shift_asj,
            "probe_rows_correct": shift_p,
            "n0_with_probe": n0 + shift_p,
            "kstar_with_probe": kappa * (d["corpus"]["n"] - (n0 + shift_p)),
            "win_with_probe": kappa * (d["corpus"]["n"] - (n0 + shift_p)) > d["corpus"]["k"],
        })
    out = {"corpus": d["corpus"], "note": __doc__.strip(), "rows": rows}
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    for r in rows:
        print("%-16s kappa %.3f  shift %7.1f (was %7.1f)  k*_probe %7.1f  win %s"
              % (r["backend"], r["kappa"], r["probe_rows_correct"], r["probe_rows_asj"],
                 r["kstar_with_probe"], r["win_with_probe"]))
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
