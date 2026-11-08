"""Cross-check the kappa interval with analytic (delta-method) standard errors.

The bootstrap answers "would kappa move if the corpus were redrawn"; the delta
method answers "how precisely did the regression pin the two slopes".  If the
two agree, the interval is not a bootstrap artefact.

Also prices the two decisions in rows, because that is the unit the paper's
argument is actually in:

  gap_full  = kappa*n - k - Delta_p     (the criterion as written, n=394, k=182)
  gap_zero  = kappa*n - k               (the same criterion with the surcharge
                                         set to zero, i.e. an infinitely fast
                                         band implementation)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "2026-11-04-00-12-59", "w12",
                                    "w12_results.json"))
D = json.load(open(SRC, encoding="utf-8"))
du = D["duel_cell"]["duel"]
xr = np.array([d["n_scope"] for d in du], float)
xb = np.array([d["n_band"] for d in du], float)
yr = np.array([d["tR"] for d in du], float) * 1e9
yb = np.array([d["tB"] for d in du], float) * 1e9
n = len(du)
N, K, RHO = 394.0, 182.0, 182.0 / 394.0


def ols(x, y):
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, X, r


(cR, FR), Xr, rr = ols(xr, yr)
(cB, FB), Xb, rb = ols(xb, yb)


def cov_hc(X, r, kind):
    XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    if kind == "ols":
        s2 = r @ r / (len(r) - 2)
        return s2 * XtXi
    if kind == "hc3":
        w = (r / (1.0 - h)) ** 2
        return XtXi @ (X.T * w) @ X @ XtXi
    raise ValueError(kind)


def kappa_se(Vr, Vb, cov=0.0):
    k = cR / cB
    var = k * k * (Vr[0, 0] / cR ** 2 + Vb[0, 0] / cB ** 2 - 2 * cov / (cR * cB))
    return k, np.sqrt(max(var, 0.0))


print("c_R = %.3f ns   c_B = %.3f ns   kappa = %.4f" % (cR, cB, cR / cB))
print("F_R = %.2f us   F_B = %.2f us   Delta_p = %.1f" % (FR / 1e3, FB / 1e3, (FB - FR) / cB))

for kind in ("ols", "hc3"):
    Vr, Vb = cov_hc(Xr, rr, kind), cov_hc(Xb, rb, kind)
    # slopes are estimated on the same corrections; the paired bootstrap is the
    # only place that correlation is observable, so take it from there
    boot = json.load(open(os.path.join(HERE, "bootstrap_kappa.json"), encoding="utf-8"))
    sd_pair = boot["resamplings"]["paired"]["kappa"]["sd"]
    for lbl, se in (("independent", kappa_se(Vr, Vb)[1]),):
        print("  %-4s se(c_R)=%.3f ns  se(c_B)=%.3f ns  se(kappa) indep=%.4f" %
              (kind, np.sqrt(Vr[0, 0]), np.sqrt(Vb[0, 0]), se))
    print("       paired-bootstrap se(kappa)=%.4f" % sd_pair)

k = cR / cB
se_ind = kappa_se(cov_hc(Xr, rr, "hc3"), cov_hc(Xb, rb, "hc3"))[1]
boot = json.load(open(os.path.join(HERE, "bootstrap_kappa.json"), encoding="utf-8"))
P = boot["resamplings"]

def ci(name):
    d = P[name]["kappa"]
    return d["p5"], d["p95"]

print()
print(f"90pct interval, independent slopes (HC3): [{k-1.645*se_ind:.4f}, {k+1.645*se_ind:.4f}]")
for name in ("paired", "block_batch", "residual"):
    lo, hi = ci(name)
    print(f"90pct interval, {name:<12}: [{lo:.4f}, {hi:.4f}]")

dp = (FB - FR) / cB
gap_full = k * N - K - dp
gap_zero = k * N - K
print()
print(f"rows of slack at corpus: full {gap_full:.1f}   surcharge zeroed {gap_zero:.1f}")
print(f"the surcharge is worth {dp:.1f} rows")
print(f"kappa needed at Delta_p=0: {K/N:.4f}   measured {k:.4f}")
print(f"shortfall {RHO-k:.4f} = {100*(RHO-k)/RHO:.1f} pct of rho")
for name in ("paired", "block_batch"):
    lo, hi = ci(name)
    h = (hi - lo) / 2
    print(f"{name:<12} half-width {h:.4f} = {100*h/RHO:.1f} pct of rho = {h/(RHO-k):.1f}x the shortfall")
