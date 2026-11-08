# The density form: the null model this paper refutes

This note is the technical report behind one sentence of §III. It is not part of
the submitted paper: the theorem is a *null model* — the estimate an
implementation would assume — and the paper only needs the fact that it fails on
an atomic fee schedule, which takes a sentence. The statement, the proof, and
the two checks that it is not vacuous live here.

## Statement

Let the values of $y_r$ in $N_r$ be i.i.d. draws from a distribution with density
$f_r$, continuous and bounded above and away from zero in a neighbourhood of
$\theta_0$, with $\theta_0$ exogenous — set by the rule, and not a quantile or a
mean of the rows being counted — and let $m\to\infty$ with $wm\to\infty$ and
$w\to0$ along the sequence. Then

$$\mathbb{E}[k_r] = m\,(F_r(\theta_0+w)-F_r(\theta_0)) = m\,w f_r(\theta_0)(1+o(1)),$$

with $\operatorname{Var} k_r = \Theta(wm)$, and $k_r/m \to w f_r(\theta_0)$ in
probability. Substituting the two widths of the paper's eq. (5) separates what
the scope does in each case:

$$\text{(i) } \mathbb{E}[k_r] = m(\Delta\lambda/\lambda)\,\theta_0 f_r(\theta_0),
\qquad
\text{(ii) } \mathbb{E}[k_r] = \lambda|\Delta a| f_r(\theta_0),$$

a constant fraction of the scope in (i) and a constant in (ii).

## Proof

$k_r$ is a sum of $m$ indicators with probability $p = w f_r(\theta_0)(1+o(1))$;
a sum of independent indicators has mean $mp$ and variance at most $mp$, so
Chebyshev gives the statement. The $o(1)$ is the density assumption, not an error
term we absorb.

## Why the paper states it and then sets it aside

It is the model an implementation would assume: a density-based cardinality
estimate is the default the estimators of the related-work section supply. The
corpus is where it fails — at $\varepsilon = 10^{-2}$ the continuous form
predicts 152 codes where 44 of 49 placements hold none (§V-C). As a null model it
earns its place; as a claim about fee schedules it would be false, which is the
impossibility argument's point.

Two further qualifications, both of which the paper compresses into the same
sentence:

1. When $\theta = \lambda\bar a$ the endpoint is a mean of the counted rows, so
   the density at $\theta_0$ moves with the band and the closed form must read
   $f_r$ there rather than quote it from a table.
2. The expectation is not an identity and not a statement about emptiness: with
   $\mathbb{E}[k_r]=1$ the band is empty with probability tending to $e^{-1}$, so
   an estimator that gets the mean right is wrong about the question one time in
   three.

## The two checks that it is not vacuous

* The exponent of $m$ is 1 and of $|D|$ at fixed $m$ is 0 — verified by exact
  counts, not by the asymptotic form.
* At equal scope the two closed forms agree to 2.2%, and the supremum over
  placement matches $\varepsilon/(c\sqrt{2\pi})$ to 2% with no free parameter.

Reproduce with `exp/boundary.py` (writes `exp/boundary.json`, asserted by
`reproduce.py`).
