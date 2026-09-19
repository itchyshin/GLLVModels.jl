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
y_s = X_s\beta + \Lambda_B\eta_s + e_s + u.
```

Here `η_s` and `e_s` are independent between sites, whereas the same
structured vector `u` enters every site. All random terms are Gaussian and
independent of one another. The model integrates them out analytically.
The entries of `e_s` have variances `d_total`, defined below; `u` has
covariance `B` and is absent in an unstructured fit.

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

**Unit-observation loadings** `Λ_W` — for this fitter, these contribute
response-specific variances: response `t` receives `sum(Λ_W[t, :] .^ 2)`.
Only these diagonal entries enter the site covariance; the fitter does not
add the off-diagonal entries of `Λ_W Λ_W'` to it.

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
y_s \sim \mathcal{N}\!\left(X_s\,\beta,\; \Lambda_B\,\Lambda_B^\top + \mathrm{diag}(d_{\text{total}})\right),
```

where `d_total[t] = sum(Λ_W[t, :] .^ 2) + σ²_B[t] + σ²_W[t] + σ²_eps`,
with absent components set to zero. Call this covariance `A`.
The sites are independent in this case, so their log-likelihoods add.
With a structured effect, a single site's marginal covariance is `A + B`
and the covariance between two different sites is `B`; their joint
likelihood must account for that dependence.

`sigma_y_site(fit)` returns `A`, including observation noise but excluding
`B`. It is not the full marginal covariance `A + B` of a structured fit.

The negative log-marginal-likelihood is evaluated via Woodbury so the
expensive `p × p` operations are reduced to `K × K` inversions plus a
`p`-vector solve, which is the same trick `MixedModels.jl` uses for
random-effect blocks.

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

Whether the structured variance can be separated from other components
depends on the design and the covariance patterns they imply. In this
model, the structured effect is shared across sites while observation
noise is independent, even if `Σ_phy` is the identity. Inspect convergence
and uncertainty for the fitted design; the form of the tree alone does
not guarantee precise variance estimates.
