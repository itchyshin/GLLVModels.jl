# Model

```@raw html
<div class="gllvm-route gllvm-route--interpret">
  <div>
    <span class="gllvm-route__eyebrow">General model guide</span>
    <p>Follow each term to the covariance it contributes before comparing estimates, latent axes, or fitted models.</p>
  </div>
</div>
```

For the Gaussian `fit_gaussian_gllvm` model, `Y` has `p` response rows
(such as species) and `n` site columns. The vector `y_s = Y[:, s]` contains
all responses at site `s`. It combines a fixed predictor, latent variation
that differs among sites, response-specific variation, and an optional
structured effect shared across sites:

```math
y_s = X_s\beta + \Lambda_B\eta_s + \Lambda_W\eta_{W,s} + e_s + u.
```

Here `η_s`, `η_{W,s}` and `e_s` are independent between sites, whereas the
same structured vector `u` enters every site. All random terms are Gaussian
and independent of one another. The model integrates them out analytically.
The W-tier term `Λ_W η_{W,s}` is present only when `K_W > 0`. The entries of
`e_s` are independent of each other, with variances `d_total` defined below;
`u` has covariance `B` and is absent in an unstructured fit.

## Terms

**Fixed effects** `X_s β` — the standard linear predictor for trait or
intercept effects per species. `X_s` is the per-site design matrix
constructed by the caller (or by the R-side fixture generator); `β` is
estimated by ML jointly with the variance components.

**Latent factor block** `Λ_B η_B[s]` — the rank-`K` ordination axes
shared across species. `Λ_B` is a `p × K` loading matrix and
`η_B[s] ∼ N(0, I_K)` is the latent gradient at site `s`. Marginal
contribution to `Σ_y_site`: `Λ_B Λ_B'`.

**Predictor-informed latent-score mean** `Λ_B (X_lv[s] α_lv)` — a C1
ordinary unit-tier extension matching the current R `gllvmTMB` Design 73
surface for Gaussian, Poisson (log link), shared-dispersion NB2, shared-shape
Gamma, shared-precision Beta, complete-response binomial logit/probit/cloglog,
and shared-cutpoint Ordinal logit point fits. With `fit_gaussian_gllvm(...; X_lv = X_lv)`,
`fit_poisson_gllvm(...; X_lv = X_lv)`, `fit_nb_gllvm(...; X_lv = X_lv)`,
`fit_gamma_gllvm(...; X_lv = X_lv)`, `fit_beta_gllvm(...; X_lv = X_lv)`,
`fit_binomial_gllvm(...; X_lv = X_lv)`, or
`fit_ordinal_gllvm(...; X_lv = X_lv)`, the unit score is decomposed as
`η_B[s] = X_lv[s] α_lv + z_s`, where `z_s ∼ N(0, I_K)`.
The raw `α_lv` coefficients are the familiar constrained-ordination axis effects
(CLV-style coefficients), but they depend on the latent-axis orientation. The
rotation-stable induced trait-effect matrix is `B_lv = Λ_B α_lv'`, returned by
`extract_lv_effects(fit)`. `confint_lv_effects()` targets this `B_lv` product
only; it does not supply SEs for the raw axis-effect table. Profile-likelihood
calls can be limited to selected entries with `profile_indices`, which index
`vec(B_lv)` in column-major order. Response masks, fixed-effect `X` plus
`X_lv`, per-trait ordinal bridge parity, W-tier, and
phylogenetic/source-specific extensions remain separate validation gates.

**Unit-observation loadings** `Λ_W η_W[s]` — a second reduced-rank block
with a `p × K_W` loading matrix `Λ_W` and its own scores
`η_W[s] ∼ N(0, I_{K_W})`. There is one score vector per site, shared by all
responses at that site, as in the `gllvmTMB` C++ engine. Marginal
contribution to `Σ_y_site`: the full `Λ_W Λ_W'`, off-diagonal entries
included. With one column of `Y` per site, this block cannot be told apart
from `Λ_B`; see the Identifiability section below.

**Site-tier diagonal random effects** `s_B[:, s] ∼ N(0, diag(σ²_B))` —
per-species independent random effects at the site tier. The marginal
contribution to `Σ_y_site` is `diag(σ²_B)`.

**Unit-obs diagonal random effects** `s_W[:, s] ∼ N(0, diag(σ²_W))` —
the per-site version, contributing `diag(σ²_W)` to `Σ_y_site`.

**Structured component** `u ∼ N(0, B)` — `Σ_phy` is a supplied covariance
among the response rows. With structured loadings and/or per-response
structured standard deviations, the model constructs
`B = (Λ_phy_aug * Λ_phy_aug') .* Σ_phy`, where `Λ_phy_aug` combines
`Λ_phy` with the column `σ_phy` when both are present. With `σ_phy` alone,
`B = (σ_phy * σ_phy') .* Σ_phy`. The same `u` enters every site, so `B`
contributes both within a site and between different sites. See
[Structured dependence](structured-dependence.md) for the data layout.

**Observation noise** `ε[:, s] ∼ N(0, σ²_eps I_p)` — the iid residual
term.

## Closed-form Gaussian marginal

Without the structured effect, integrating out the random terms gives

```math
y_s \sim \mathcal{N}\!\left(X_s\,\beta,\; \Lambda_B\,\Lambda_B^\top + \Lambda_W\,\Lambda_W^\top + \mathrm{diag}(d_{\text{total}})\right),
```

where `d_total[t] = σ²_B[t] + σ²_W[t] + σ²_eps`, with absent components
(including `Λ_W` when `K_W = 0`) set to zero. Call this covariance `A`.
The sites are independent in this case, so their log-likelihoods add.
With a structured effect, a single site's marginal covariance is `A + B`
and the covariance between two different sites is `B`; their joint
likelihood must account for that dependence.

`sigma_y_site(fit)` returns `A`, including observation noise but excluding
`B`. It is not the full marginal covariance `A + B` of a structured fit.

The negative log-marginal-likelihood is evaluated via Woodbury so the
expensive `p × p` operations are reduced to `K × K` inversions plus a
`p`-vector solve, which is the same trick `MixedModels.jl` uses for
random-effect blocks. With a W tier, the Woodbury step uses the stacked
loadings `[Λ_B Λ_W]`, so `K` there is `K + K_W`.

## Rotation trick for phylogenetic terms

For this row-structured model, the covariance of `vec(Y)` (all `p`
responses from the first site, then the second, and so on) is

```math
\Sigma_{y,\text{full}} \;=\; I_n \otimes A \;+\; J_n \otimes B,
```

where `A` contains the site-specific variation, `B` is the structured
covariance defined above, and `J_n` is the `n × n` all-ones matrix.
Each diagonal site block is `A + B`; each off-diagonal site block is `B`.
Thus the full covariance is block-diagonal only when `B` is zero (or there
is just one site). Diagonalising in the site-dimension (which
amounts to rotating into the `1_n / √n` versus orthogonal-complement
basis) decomposes the determinant and quadratic form into the rank-1
component `A + n·B` and the `(n − 1)` copies of `A`, reducing the
phylogenetic likelihood evaluation to *one* `p × p` Cholesky plus
`(n − 1)` reuses of the iid Cholesky. The engine does this once per
gradient evaluation; the marginal log-likelihood remains closed-form.

## Identifiability

The loading matrix `Λ_B` is identified only up to an orthogonal rotation
in `K`-space — the marginal covariance `Λ_B Λ_B'` is invariant under
`Λ_B → Λ_B Q` for any orthogonal `Q`. The engine uses the standard
lower-triangular packing (matching the R-side `gllvmTMB::rr_theta_len(p,
K)`) as the identifying constraint at the optimum. The latent scores
`η_B[s]` are not estimated; they are integrated out.

**Per-species residual variances (`fit_gaussian_pervar_gllvm`).** When each
response carries its own diagonal residual ψ (Quick-start `pervar = true`),
separating Λ from ψ additionally requires **(p − K)² ≥ p + K** (Ledermann
bound). The shared-σ Gaussian fitter uses one scalar `σ_eps` and does not
apply this bound; its `K < p` requirement is only for the PPCA warm start.

**The two tiers are not separated by this fitter.** In `fit_gaussian_gllvm`
each column of `Y` is one site observed once, so `η_s` and `η_W[s]` vary at
the same level. The likelihood depends on the two loading blocks only
through `Λ_B Λ_B' + Λ_W Λ_W'`. Any orthogonal rotation of the `K + K_W`
columns of `[Λ_B Λ_W]`, including one that mixes the two blocks, leaves the
fit unchanged. In the same way, `σ²_B`, `σ²_W` and `σ²_eps` enter only
through their sum `d_total`. A fit with `K_W > 0` therefore reaches the same
maximised log-likelihood and the same `sigma_y_site` as a single-tier fit
with `K + K_W` axes, and how the variation is split between the tiers comes
from the starting values, not from the data. `gllvmTMB` behaves the same way
when each unit is observed once.

With `K_W > 0`, do not interpret summaries that report one tier on its own:
`communality`, `proportions`, `extract_Sigma` at `level = :unit` or
`:unit_obs`, the default (tier-scoped) results of `extract_communality`,
`extract_correlations`, `extract_proportions` and `extract_ICC_site`, the
latent scores from `getLV`, and the angle from
`diagnose_kernel_separability`. Wald standard errors for single entries of
`Λ_B` and `Λ_W` are not usable either; in our checks they were `NaN` or in
the thousands. With `has_diag = true`, the same caution applies to anything
that separates `σ²_B`, `σ²_W` and `σ²_eps`: `proportions` with
`component = :unique_B`, `:unique_Wd` or `:residual`, and `extract_Sigma` and
the tier-scoped extractors at `level = :unit` or `:unit_obs`. The
log-likelihood, `sigma_y_site`, `correlation` and
`extract_Sigma(fit; level = :site)` use only the identified total and are
safe to report. To separate between-unit from within-unit variation, the
data need repeated observations of each unit: use `fit_twolevel_gaussian`,
which takes an `individual` grouping vector.

Whether the structured variance can be separated from other components
depends on the design and the covariance patterns they imply. In this
model, the structured effect is shared across sites while observation
noise is independent, even if `Σ_phy` is the identity. Inspect convergence
and uncertainty for the fitted design; the form of the tree alone does
not guarantee precise variance estimates.
