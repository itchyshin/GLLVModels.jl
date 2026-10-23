# Tweedie power contracts for Core 0.7.0

**Status 2026-09-25:** landed. The numerical outcomes this contract proposed
were carried out and merged: PR #378 (shared-power second-order cell, option
A) and PR #391 (shared + species estimated-power second-order cells,
PARTIAL). The "numerical outcomes pending" line just below is stale; see
those PRs and `docs/dev-log/core070/second-order-holdouts-2026-09-04.md` for
the current disposition.

## Status

**Proposed implementation contract — numerical outcomes pending.**  Frozen R
reference: `gllvmTMB` `b4d5fee64def88bc768dda1f1f77c29b295edd86`.

## Symbolic model

For trait `t` and site `s`, with the log link,

\[
  \mu_{ts}=\exp(\beta_t+\Lambda_t z_s+o_{ts}),\qquad
  Y_{ts}\sim\operatorname{Tw}(\mu_{ts},\phi_{g(t)},p_t),\qquad
  \operatorname{Var}(Y_{ts})=\phi_{g(t)}\mu_{ts}^{p_t},
\]

where `group[t] = g(t)`, `\phi_g > 0`, and `1 < p_t < 2`.  A masked cell is
absent from both its site likelihood and the warm start; its numerical
placeholder cannot affect the objective.  Offsets remain additive on the log
mean scale.  The full Tweedie density, including the zero atom and the
positive-part normalising series, remains in the objective.  The Laplace
log-determinant uses `hessian = :observed` by default; `:fisher` is an explicit
alternative for a deliberately matched objective.  This choice affects the
Laplace correction only, not the Fisher-scored inner mode search.

The power transform is always `p_t = 1 + logistic(\xi_t)`.  Thus the three
admitted contracts are distinct parameter spaces:

| Contract | API | Power coordinates | Free parameter count |
| --- | --- | --- | --- |
| fixed common | `power = p0` | none; `p_t = p0` for every trait | `p + rr_theta_len(p,K) + G` |
| estimated shared | `power = nothing, power_group = :shared` | one `\xi` | `p + rr_theta_len(p,K) + G + 1` |
| estimated per species | `power = nothing, power_group = :species` | `\xi_1,\ldots,\xi_p` | `p + rr_theta_len(p,K) + G + p` |

The fixed-common comparison is intentionally not a surrogate for either
estimated model.  The R reference makes `log_phi_tweedie` and
`logit_p_tweedie` trait vectors, maps non-Tweedie rows out, and fixes a
user-supplied scalar `tweedie(p = p0)` independently for every Tweedie trait.
For this all-Tweedie Julia surface, `:species` is the corresponding estimated
model; it uses the existing `disp_group = :species` vocabulary. `:shared`
preserves the existing Julia grouped estimand explicitly. The frozen R source
does not expose shared-power estimation as its default; that Julia option is an
explicit compatibility-preserving model, not a claim about R's default.

## API and fit objects

`fit_tweedie_gllvm_grouped` gains `power = nothing` and
`power_group = :shared`.  Passing a finite scalar `power` strictly inside
`(1,2)` fixes a common power and returns `TweedieGroupedFit` with explicit
fixed-status metadata. Controls are validated before selection: an invalid
`power_group` is rejected even when `power` is fixed, so it cannot be silently
ignored. With `power = nothing`, `power_group = :shared` estimates the
historical single power and returns `TweedieGroupedFit`; `power_group =
:species` estimates one power per species and returns
`TweediePerTraitPowerFit`. The latter exposes
`power::Vector{Float64}` and has its own `_nparams`/StatsAPI degrees-of-freedom
methods, avoiding a type-unstable union in likelihood-adjacent code and leaving
the existing shared fit object intact.

Each free power coordinate is subject to the same interior, finite-objective,
and scaled-gradient verdict as the existing scalar coordinate.  A per-trait
fit fails its verdict if any coordinate runs to an endpoint.  The fitter
validates the link, group layout, power control, and initial powers before
optimization; it does not silently replace an invalid or masked parameter with
a new model.  The fixture used for external comparison remains seed 82,
`p = 5`, `K = 1`, `n = 150`.

## Alignment table

| Symbol | Julia control / stored field | Fixture / observation rule | Verification |
| --- | --- | --- | --- |
| `\beta,\Lambda,z` | existing packed lower-triangular loadings | original seed-82 `p=5,K=1,n=150` fixture | unchanged latent model and log density |
| `\phi_{g(t)}` | `group`, `φ` | grouped / per-trait dispersion stays unchanged | `G` nuisance coordinates and density constants retained |
| `p_0` | `power = p0` | common fixed power | R and Julia have zero power coordinates |
| `p` | `power_group = :shared` | shared-power qualified comparison | exactly one power coordinate and dof increment |
| `p_t` | `power_group = :species`, `power::Vector` | R-default qualified comparison | exactly `p` power coordinates and dof increment |
| `o_{ts}`, mask | existing `offset`, `mask` | same observed cells / placeholders | objective invariant to masked placeholders |
| Laplace curvature | `hessian = :observed` / `:fisher` | same selected objective on both sides | no curvature-mode substitution in parity proof |

## Acceptance boundary

Local deterministic tests establish the parameter-vector length, fixed-power
identity and mask/offset propagation.  A bounded Totoro run then executes the
seed-82 fixed-common, shared-estimated, and per-trait-estimated models.  The
R-qualified rows require an immutable installed-library provenance receipt;
without it their status is **PARTIAL**, never a pass.  No full-AGHQ or general
core-parity claim follows from these checks.
