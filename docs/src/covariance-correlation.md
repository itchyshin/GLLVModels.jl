# Covariance and correlation

```@raw html
<div class="gllvm-route gllvm-route--interpret">
  <div>
    <span class="gllvm-route__eyebrow">Interpret the shared structure</span>
    <p>Start with model-implied covariance, correlation, and shared variance; those quantities remain meaningful when a loading axis rotates.</p>
  </div>
</div>
```

A fitted Gaussian GLLVM gives you more than latent ordination axes — it gives
the **among-response covariance** those axes imply, and summaries of
how much of each response's variation is
*shared* (communality), and which responses *move together* (correlation).

## The model-implied covariance

For the simple Gaussian fit below, with one latent source, `K` factors and
loadings `Λ`, the responses at a site have covariance

```math
\Sigma_y = \Lambda \Lambda^{\top} + \sigma_\varepsilon^2 I_p.
```

Here `ΛΛᵀ` is the **shared** latent part and `σ_eps² I_p` is independent
observation noise. Additional diagonal variance components enter when
requested in the model. Three extractors describe this covariance:

```julia
using GLLVModels, Random
Random.seed!(1)
p, n, K = 6, 200, 2
Λtrue = 0.8 .* randn(p, K)
Y = Λtrue * randn(K, n) .+ 0.5 .* randn(p, n)   # p × n responses

fit = fit_gaussian_gllvm(Y; K = K)

Σ  = sigma_y_site(fit)    # p×p covariance, including observation noise
c² = communality(fit)     # per-response shared fraction (ΛΛᵀ)ₜₜ / Σₜₜ ∈ [0,1]
R  = correlation(fit)     # p×p cross-response correlation derived from Σ_y
```

## Reading the results

- **`communality(fit)`** — for each response, the fraction of its variance
  assigned to the shared latent factors. With `c² ≈ 0.8`, the fitted factors
  account for about 80% of that response's site-specific variance; with
  `c² ≈ 0.1`, they account for about 10%. This describes a variance split,
  not the cause of that variation.
- **`correlation(fit)`** — the model's estimate of which responses co-vary. A
  positive entry indicates that two responses tend to be high or low
  together after accounting for fitted predictors; a negative entry indicates
  that they tend to vary in opposite directions.
- **`sigma_y_site(fit)`** — covariance on the Gaussian response scale,
  including observation noise. In a structured fit it excludes the shared
  phylogenetic block `B`; the full covariance at one site is then `A + B`,
  where this function returns `A` (see [Model](model.md)).

Residual association has several possible causes, including unmeasured
environmental conditions, shared history, sampling effects, and biological
interactions. Its sign alone is not proof of competition, facilitation, or a
trait trade-off. Those interpretations need additional biological evidence
and an appropriate study design.

## When you need `unique`

If some responses carry their own variance component beyond the shared factors
(the gllvmTMB `unique()` case), that variance enters the diagonal of `Σ_y`
in addition to observation noise, and `communality` reports the smaller
shared fraction.

Do not confuse this with `extract_communality(fit)`, whose default is the
selected source alone. Without a diagonal variance within that source, it
returns `1.0` for responses with positive source variance, even when
observation noise remains. Use `communality(fit)` or
`extract_communality(fit; level = :total)` for the fraction including that
noise. [Post-fit extractors](postfit-extractors.md) explains the denominators.

## Uncertainty on derived quantities

The extractors above are point estimates. For an interval on a *derived*
quantity — a `Σ_y` entry, a communality, a cross-response correlation — use the
derived-quantity confidence intervals (`confint_derived`, and the
transformed-scale Wald intervals for `[0,1]`- and `[−1,1]`-bounded quantities),
which provide profile-likelihood and parametric-bootstrap intervals.

See also: [Get started](quickstart.md) · [Model](model.md) · [Reference](api.md).
