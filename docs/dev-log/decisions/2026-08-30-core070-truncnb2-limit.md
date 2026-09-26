# Truncated NB2 near the Poisson limit: diagnostic algebra

**Status 2026-09-25:** landed. The engine repair this diagnosis called for
landed in commit `376044e7f` ("Stabilize truncated NB2 scalar kernel with
derivative regressions"), with derivative regression tests
(`test/test_truncnb2_precision.jl`) and evidence in
`docs/dev-log/core070/truncnb2-kernel-evidence.json`. The "engine repair and
derivative validation pending" line below is stale.

Status: VERIFIED_SCALAR_DIAGNOSIS; engine repair and derivative validation pending.

For integer y>=1, mean mu>0 and dispersion r>0, NB2 mass is

    Gamma(r+y)/(Gamma(r) y!) [r/(r+mu)]^r [mu/(r+mu)]^y.

Using Gamma(r+y)/Gamma(r)=r^y product(k=0..y-1)(1+k/r), its log mass becomes

    sum(k=0..y-1) log1p(k/r) + y log(mu) - log(y!)
      - (r+y) log1p(mu/r).

The zero mass has log p0=-r log1p(mu/r). Thus the truncated log density subtracts
`log(-expm1(-r*log1p(mu/r)))`. This avoids forming a probability rounded close to
one. The candidate is exact integer-count algebra, not a Poisson approximation,
cap on r or change to the fitted model. Its naive O(y) cost is explicit; a general
production implementation needs a reviewed stable large-count strategy.

The existing kernel forms `r/(r+mu)` for Distributions.NegativeBinomial and again
for p0. At very large r this loses mean information. Sixty fixed (r,mu,y) cells
were checked against a256-bit recurrence reference. Current maximum absolute
log-density error0.0151883 at the stress grid; candidate maximum2.8422e-14.
At the fitted native r=6.25309e6 the grid maximum current error8.80575e-8;
at the R r=9.87351e7 it is2.33291e-8. Do not attribute the stress maximum to the
actual fitted data or claim this proves the sole cause of either optimizer flag.

The original seed58 fit has fifteen parameters. Tight public R controls retain
code1 (singular convergence); native finite differences give max gradient
5.39642e-4, changing by4.20488e-4 when the step doubles. The log likelihoods still
differ only1.37079e-6. This is evidence against trusting fit flags/likelihood
closeness alone, not permission to weaken the health gate.

Next repair must cover density, p0/normalization, score/curvature coherence,
small-mu and large-r limits, exact integer support and large counts; compare
FD/AD derivatives and known moments before rerunning the original fixture.
No native source, R oracle, original data or tolerance changed in this diagnosis.
Evidence and runnable checks: `tools/core070_truncnb2_precision.jl`,
`tools/core070_truncnb2_health.jl`, and `docs/dev-log/core070/truncnb2-diagnosis-evidence.json`.
