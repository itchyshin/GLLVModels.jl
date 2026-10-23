# Design spec: porting gllvmTMB's phylogenetic latent structure (`phylo_latent()`) to GLLVModels.jl

Status: DRAFT for maintainer review. No code. Gate rows A14 and A15 of the signed
0.7.0-programme gate-tier list (`docs/dev-log/core070/true-parity-gate-tier-2026-09-05.md:50-51`,
scoreboard `docs/dev-log/core070/true-parity-gate-tier-scoreboard-2026-09-25.md:39-40`).

Scope authority: maintainer decision D-295 (2026-09-27) places the phylogenetic latent
structure inside the P1 boundary. P1 is gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9`
(DESCRIPTION `Version: 0.7.1`). Every R citation below is `file:line` at that commit, read
with `git show 9539352f6:<path>`; the R tree was never checked out or edited. Every Julia
citation is `file:line` at `origin/main` `97e11be0475ca8c40abf2dc916db7a7131e919fa`.

Two earlier decisions bind this spec:

- The phylo transport questions Q1 to Q4 carry Ada defaults awaiting ratification
  (`docs/dev-log/core070/phylo-transport-questions-2026-09-02.md:76-95`; decision packet
  `docs/dev-log/owed/2026-09-25-true-parity-decision-packet.md:44`, row 8, B-05, which
  estimated the build at 5 to 25 days).
- Public intervals for `phylo_latent + lv = ~x` (Phylo Model A) are **rejected** for
  advertising (`docs/design/capability-status.md:33`, rows at `:58` and `:99`; the brief dates
  the maintainer decision 2026-08-28). This spec does not reopen it: no `lv = ~x` surface,
  no Model A interval, no bridge advertising of either.

The one-sentence read-back: build the Julia twin of R's *bare* `phylo_latent(species, d = K)`
Gaussian model on the existing `PrecisionPhy` plus `precision_multivariate` path (which is
already the R-shaped model), give it R's argument names and refusals, pair it with R at P1 on
the two ledger cases and one realistic-size cell, and keep every native Julia phylogenetic
design (Hadamard species-row loadings, signed `sigma_phy`, contrasts, edge-incidence) as a
documented extra rather than a parity claim.

---

## 1. The model, its parameterisation, and identifiability (A14 and A15)

### 1.1 What the two rows are

| Row | Ledger id | Requires (tier key `true-parity-gate-tier-2026-09-05.md:17-26`) | Route |
|---|---|---|---|
| A14 | `covariance/COV-PHYLO-LATENT-1FO` | `phylo_latent()` bare, first-order paired receipt: logLik, point estimates, cross-objective (each engine's fit evaluated under the other engine's objective) | NAT (native Julia; the bridge cannot honestly carry it, `bridge_precision_multivariate.jl:113`) |
| A15 | `covariance/COV-PHYLO-LATENT-RSZ` | the same model at realistic size: p ≥ 20, n ≥ 500, cond(H) recorded | NAT |

The ledger row `covariance/COV-PHYLO-LATENT` has zero executable cases; its three planned
cases are `STRUCT-PHY-TREE-RR`, `STRUCT-PHY-DENSE-RR` and `STRUCT-PHY-TREE-PROPTO`, all
`PREPARED_REFERENCE_NUMERICS_UNPAID` (`docs/dev-log/core070/required-source-case-map.json`,
row 75; `docs/dev-log/core070/structured-required-case-plan.json`). The reference calls are:

```r
# STRUCT-PHY-TREE-RR
gllvmTMB(value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree), df,
         cluster = "species", family = stats::gaussian(), control = control)
# STRUCT-PHY-DENSE-RR
gllvmTMB(value ~ 0 + trait + phylo_latent(species, d = 1, vcv = C), df,
         cluster = "species", family = stats::gaussian(), control = control)
```

with free parameters `b_fix: 3, log_sigma_eps: 1, theta_rr_phy: 3` and random `g_phy`.
The third case, `STRUCT-PHY-TREE-PROPTO`, is `phylo_scalar()`, a different keyword (a scalar
phylogenetic variance shared across traits, `R/brms-sugar.R:1046`), listed under A14's own ledger
row. Whether A14 can be promoted with that planned case left `UNPAID` is the maintainer's call
(OQ-10); this spec recommends fencing it and says so there rather than fencing it unilaterally.

"Bare" means the R default `unique = FALSE` (`brms-sugar.R:771-782`), no `rho`, no slope
LHS, no companion `latent()` at the species tier. NEWS at P1 confirms this is the
loadings-only reading: "intercept-only `phylo_latent()` (default `unique = FALSE`; it emits no
Psi companion at all)" (`NEWS.md:1064-1065`).

### 1.2 The R model at P1

For long-format row `o` with trait `t(o)` and species `s(o)`:

```
y_o = b_{t(o)} + sum_{k=1}^{K} Lambda_phy[t(o), k] * g_k[aug(s(o))] + eps_o
eps_o ~ N(0, sigma_eps^2)                     (one shared Gaussian residual)
g_k ~ N(0, A)  independently over k = 1..K,   A = (Ainv_phy_rr)^{-1}
```

- Linear predictor: `eta(o) += Lambda_phy(t, k) * g_phy(species_aug_id(o), k)`
  (`src/gllvmTMB.cpp:3181-3189`).
- Prior on each factor column, evaluated through the sparse precision:
  `nll += 0.5 * (n_aug_phy * log(2 pi) + log_det_A_phy_rr + g_k' Ainv_phy_rr g_k)`
  (`src/gllvmTMB.cpp:2153-2176`). `Sigma_phy = Lambda_phy Lambda_phy'` is `REPORT`ed
  (`:2181-2183`).
- Residual: exactly one Gaussian `log_sigma_eps` slot (`src/gllvmTMB.cpp:3262-3265`,
  `expected_sigma_slots == 1` unless lognormal rows co-occur), `sigma_eps = exp(log_sigma_eps)`.
- Fixed effects: `0 + trait` gives one intercept per trait (`b_fix`, length T).
- The phylogenetic scale is fixed at one and absorbed by `Lambda_phy`; `A` is the
  correlation-form tree covariance (unit root-to-tip height) because the fit path hard-codes
  `correlation = TRUE` (`R/fit-multi.R:4677`, `R/phylo-tree-precision.R:223-227`).

Marginally, stacking species within trait, `Cov(vec Y_species-by-trait) = Lambda_phy
Lambda_phy' (x) C + sigma_eps^2 I`, the trait-by-species Kronecker form, with `C` the tip
covariance implied by `A`. (This is not the form of the native Julia J3 block; see section 3.1.)

### 1.3 How the tree enters

Three inputs, one internal bundle `(Ainv_phy_rr, log_det_A_phy_rr, n_aug_phy, species_aug_id)`
(`R/fit-multi.R:4656-4755`):

| Input | R route | Facts that fix the twin |
|---|---|---|
| `tree =` (an `ape::phylo`; canonical) | `.gllvm_phylo_tree_precision(phylo_tree, correlation = TRUE)` (`R/fit-multi.R:4677`; builder `R/phylo-tree-precision.R:183-249`) | Ultrametric gate within `sqrt(.Machine$double.eps) * scale` (`:139-146`, message "tree must be ultrametric"); positive branch lengths (`:193-195`); root dropped, nodes ordered internal-first tips-last (`:203-207`), so `n_aug = n_tip + Nnode - 1` (`2p - 2` for a bifurcating tree); precision entries `1/edge_length` scaled by `height` (`:214-227`); `log_det_precision = n_aug * log(scale) - sum(log(edge_length))` (`:240`); `log_det_A_phy_rr = -log_det_precision` (`R/fit-multi.R:4679`); tip-to-row map by tip label (`:4684-4690`, 0-based for C++). |
| `vcv =` / `A =` (dense tip-only matrix; legacy aliases of each other) | dense path (`R/fit-multi.R:4724-4755`) | Rownames required (`:4729-4733`); rows must cover the species levels (`:4735-4737`); `Aphy + diag(1e-8)` ridge then dense `solve()` (`:4740-4752`); `n_aug_phy = n_species`, identity tip map (`:4754-4755`). The roxygen states the two routes therefore agree only to roughly `1e-5` in log-density (`R/brms-sugar.R:689-693`). |
| `Ainv =` (sparse precision, tip-only or with ancestors) | sparse direct route (`R/fit-multi.R:4691-4723`) | Rownames required; full precision kept when ancestors are present because subsetting a precision would condition, not marginalise (`:4700-4704`). |

Per-term `tree =` / `vcv =` is harvested into the global slot (Phase L, `R/fit-multi.R:4318-4345`);
a global tree with no phylogenetic term aborts (`:4611-4627`). Species are matched by
**factor level**, not tip order (`R/brms-sugar.R:695-712`); levels not covered by the tree
abort, and the message names `droplevels()` when the missing level has no observations
(`R/fit-multi.R:609-636`). Tips that are in the tree but unobserved are legal (they are
marginalised through the augmented precision).

### 1.4 Parameterisation and identifiability

- `Lambda_phy` is unpacked from `theta_rr_phy` by `gll_unpack_rr_loadings`
  (`src/gllvmTMB.cpp:34-59`): `K` diagonal entries first, then strict-lower entries column by
  column; upper triangle zero; **no positivity transform on the diagonal** (signed). Length
  `T*K - K(K-1)/2`.
- Identified estimand: `Sigma_phy = Lambda_phy Lambda_phy'`. `Lambda_phy` itself is identified
  only up to column rotation and sign; R's own test compares `Sigma_phy`, not `Lambda_phy`
  (`tests/testthat/test-phylo-hadfield.R:104-107`). The twin therefore pairs `Sigma_phy`, `b_fix`,
  `sigma_eps` and the log-likelihood, never raw loadings.
- Rank bound: `d <= n_traits`, refusal "phylo_latent(d = ...) exceeds the number of traits"
  (`R/fit-multi.R:2536-2542`); `d == n_traits` fits (`test-latent-rank-guard.R:162`).
- Scale: with `correlation = TRUE`, `sigma2_phy` per trait is `diag(Sigma_phy)` on the
  unit-height scale. A raw-branch-length Julia fit would differ by exactly `height`
  (`phylo-transport-questions-2026-09-02.md:19-23`).
- Not bare, recorded for completeness: `unique = TRUE` adds `Psi_phy = diag(exp(2 *
  log_sd_phy_diag))` with the same `A` (`src/gllvmTMB.cpp:2186-2223`; positive log link), giving
  `Sigma_phy = Lambda Lambda' + Psi_phy`; a lone `phylo_unique()` is rerouted to `phylo_rr` with
  `d = T` and a diagonal `lambda_constraint` (`R/fit-multi.R:2376-2388`, `:2525-2533`,
  `:6503-6527`), so its per-trait SDs are the **signed** diagonal of `Lambda_phy`.

### 1.5 R's own claims about this surface (quoted)

- Capability status: "phylogenetic × latent (`phylo_latent()`) | implemented" backed by
  `PHY-01, PHY-02, PHY-03, PHY-09, PHY-10, PHY-17` (`docs/design/capability-status.md:55`);
  "phylo_latent + `lv = ~ x` (Phylo Model A public intervals) | planned" (`:65`).
- Register: PHY-01 "Hadfield & Nakagawa sparse A⁻¹ | `covered` | `test-phylo-hadfield.R`"
  with the note that the tree route passes `log_det_A_phy_rr = -log_det_precision` so the
  sparse and dense paths agree on logLik as well as MLEs
  (`docs/design/35-validation-debt-register.md:234`); PHY-09 mode dispatch `covered`
  (`:242`); PHY-10 optional `phyloVCV` `covered` (`:243`); LV-08 (Model A) `blocked`
  (`:794`).
- Intervals: the roxygen of `extract_phylo_signal()` says "The point estimates are the
  supported claim. When `ci = TRUE`, the interval methods are provided for exploration: their
  empirical coverage is not certified for this estimand" (`R/extract-omega.R:356-360`).
  `confint(fit, parm = "phylo_signal")` routes to profile, Wald or bootstrap helpers
  (`R/z-confint-gllvmTMB.R:894-934`). Communality Wald CIs at the phylo tier require the
  folded `unique = TRUE` contract (`R/communality-ci.R:73-87`). None of these is inside the
  A14 or A15 bar (first order only).

---

## 2. Public API contract for the Julia twin

Principle (from the brief): twin R's public API at R's scope; port R's semantics; Julia
designs stay documented extras. GLLVModels.jl has no `@formula` term recogniser for
`phylo_latent(...)` (`src/formula.jl:1-12`; ledger row 571 records "0 hits for a term-name
dispatch"), and formula v1 rejects `FunctionTerm` (`src/phylo_dep.jl:6`). The repo's
convention for structured cells is a named matrix fitter (`fit_phylo_dep_gllvm`,
`fit_animal_latent_gllvm`, `docs/design/capability-status.md:133-135`). The twin follows it.

### 2.1 Entry point

```julia
fit_phylo_latent_gllvm(Y::AbstractMatrix, species::AbstractVector{<:AbstractString};
    d::Integer = 1,
    tree = nothing,        # AugmentedPhy (from augmented_phy / make_phy)
    vcv = nothing,         # dense p x p tip covariance, labelled through tip_labels
    A = nothing,           # alias of vcv (R: "alias of vcv =")
    Ainv = nothing,        # PrecisionPhy, or (I, J, V, labels) sparse triplets
    tip_labels = nothing,  # labels for vcv / A rows; required with vcv / A
    unique::Bool = false,  # true is a documented extra (mode :explicitunique), not the twin
    rho = 1,               # anything other than 1 is refused by a Julia scope fence (OQ-12)
    X = nothing,           # optional site-level design; default is 0 + trait intercepts
    start = nothing, g_tol = 1e-5, iterations = 400)
```

- `Y` is traits-by-observations, the layout `fit_precision_multivariate` already takes
  (`src/precision_multivariate_fit.jl:184-193`). `species[o]` is the label of observation `o`;
  labels are matched to tip labels by name, which is R's factor-level rule.
- Exactly one of `tree`, `vcv`/`A`, `Ainv` must be given (R: "Supply one of `tree`, `vcv`, or
  `A` / `Ainv`", `R/brms-sugar.R:731-733`; both `vcv` and `A` abort,
  `test-phylo-vcv-A-aliases.R:107-119`).
- `tree` is converted with `PrecisionPhy(tree; correlation = true)`
  (`src/phylo_precision.jl:122-160`), which already carries R's unit-height rescale and the
  ultrametric gate (`GJL-GATE-PHYLO-NONULTRAMETRIC`, `:116`). Polytomies: R accepts them,
  with `n_aug = n_tip + Nnode - 1`, smaller than `2p - 2` (`src/gllvmTMB.cpp:920-922`), but
  `PrecisionPhy(::AugmentedPhy)` asserts `n_aug == 2p - 2` (`src/phylo_precision.jl:147-148`)
  and `augmented_phy` itself is bifurcating-only (`src/edge_incidence.jl:181`). The twin
  therefore builds the R-convention precision for a tree **directly** from the edge list
  (R's rule at `R/phylo-tree-precision.R:209-227`, one row per non-root node) and wraps it with
  the raw-triplet constructor (`src/phylo_precision.jl:190`), so polytomies are admitted
  exactly as R admits them; `PrecisionPhy(::AugmentedPhy)` is not on the twin path. A tree
  with a node of out-degree one is refused with `GJL-GATE-PHYLO-LATENT-TREE`, as R refuses
  it through `.gllvm_validate_phylo_tree`. This entry sets
  `correlation = true` **because it is the twin of R's fit path**; the native constructors
  keep the Q1 default (opt-in `false`). No native default changes.
- `vcv`/`A` follows the R dense route byte for byte: reorder rows to the species label order,
  add `1e-8` to the diagonal, invert by Cholesky, `log_det` from the ridged matrix, tip-only
  `PrecisionPhy` with `n_aug = p` and identity tip map (`R/fit-multi.R:4740-4755`). Q2's
  condition-number warning (κ > 1e8) is emitted, never a refusal, so R-accepted inputs are
  not refused.
- Residual: one shared `sigma_eps` (`residual_mode = :shared`), matching R's single slot.
  `residual_mode = :trait` stays reachable through `fit_gllvm(...; phylo = ...)` as an extra.
- Return: the existing `PrecisionMultivariateFit` (`src/precision_multivariate_fit.jl:134-152`)
  extended with `species_labels::Vector{String}` and `tip_labels::Vector{String}`; no new
  fit type. `fit.loading` is `T x d` with zeros above the diagonal (R's `Lambda_phy` shape,
  `test-stage35-phylo-rr.R:44-48`), `fit.beta` has length `T`, `fit.residual_variance` is
  `sigma_eps^2` repeated `T` times (the existing `:shared` layout), `fit.loglik`,
  `fit.converged`, `fit.gradient_norm`, `fit.hessian_condition_number`.

`fit_gllvm(Y; phylo = ::PrecisionPhy, ...)` (`src/families/fit_gllvm.jl:163-190`) is left
unchanged and keeps routing to `fit_precision_multivariate`; the new entry is a thin wrapper
around that one fitter.

### 2.2 Extractors (R names at R's scope)

| R | Julia twin | Behaviour ported |
|---|---|---|
| `extract_Sigma(fit, level = "phy", part = "total"/"shared"/"unique")` (`R/extract-sigma.R:628-655`) | `extract_Sigma(fit::PrecisionMultivariateFit; level = :phy, part = :total)` | `:shared` = `loading * loading'`; `:unique` = `Diagonal(phylo_unique_variance)` or zeros when `nothing`; `:total` = sum. Any other `level` refuses with "unsupported level". |
| `extract_phylo_signal(fit)` (`R/extract-omega.R:468-530`, definition `:325-333`, components `:668-679`) | `extract_phylo_signal(fit::PrecisionMultivariateFit)` | Returns a table with `trait, H2, C2_non, Psi, V_eta`. For a bare fit `Sigma_non = 0` and `Psi_non = 0`, so `H2 = 1` for every trait and `V_eta = diag(Sigma_phy)`; no advisory for Gaussian (`test-phylo-signal-categorical.R:117`). Replaces the current `_pmv_phylogenetic_signal` refusal (`src/precision_multivariate_fit.jl:307-320`) for this fit type only; see open question OQ-5. `ci = true` refuses: "phylogenetic signal intervals are outside the A14/A15 first-order scope" (R's own claim is exploratory, `R/extract-omega.R:356-360`). |
| `logLik`, `coef`, `nobs`, `dof` | already defined (`src/precision_multivariate_fit.jl:348-351`) | unchanged. |
| `precision_multivariate_intervals` | unchanged, documented extra | transformed Wald on `beta`, rotation-invariant `phylo_cov[i,j]`, shared residual (`:322-347`). Not a parity claim. |

### 2.3 Refusals (ported messages)

Julia throws `ArgumentError` whose message starts with the R sentence, so a twin test can
assert the same substring on both sides. cli markup is rendered to plain text.

| Condition | R message (rendered) and site | Julia tag |
|---|---|---|
| `d > n_traits` | "phylo_latent(d = D) exceeds the number of traits (T); the latent rank must satisfy d <= n_traits." (`R/fit-multi.R:2537-2540`) | `GJL-GATE-PHYLO-LATENT-RANK` |
| no `tree`, `vcv`, `A`, `Ainv` | "phylo_latent() / phylo_slope() found in formula but `phylo_vcv` (or `phylo_tree`) is NULL." (`:4726-4728`) | `GJL-GATE-PHYLO-LATENT-SOURCE` |
| two sources given | "phylo_*() with both vcv and A" (tested `test-phylo-vcv-A-aliases.R:107-131`) | `GJL-GATE-PHYLO-LATENT-SOURCE` |
| `tree` not a phylogeny | "`tree` must be a phylogeny object." (`R/phylo-tree-precision.R:73-77`) | `GJL-GATE-PHYLO-LATENT-TREE` |
| non-ultrametric tree | "`tree` must be ultrametric. Root-to-tip distances differ by more than {tol}." (`:141-146`) | existing `GJL-GATE-PHYLO-NONULTRAMETRIC` |
| `vcv` without labels | "phylo_vcv must have rownames matching levels of `species`." (`R/fit-multi.R:4729-4733`) | `GJL-GATE-PHYLO-LATENT-LABELS` |
| species label not in tree / vcv rows | "{what} do not cover all species levels." plus the `droplevels()` bullet when the level is unobserved (`:609-636`) | `GJL-GATE-PHYLO-LATENT-COVERAGE` |
| `unique = true` | not refused; routed to `mode = :explicitunique` and labelled a documented extra in the fit's `admission_scope` | none |
| `rho != 1` | **No R analogue.** R accepts `rho` on a bare `phylo_latent()`: the guard at `R/fit-multi.R:4639-4641` aborts only when `use_phylo_rr` is false or a slope/mi/kernel term co-occurs, and the dense route attenuates `Aphy` (`:4742-4750`). The twin refuses with its own scope fence: "rho is outside the A14/A15 scope of the Julia twin; fit rho = 1 or use R." (OQ-12) | `GJL-GATE-PHYLO-LATENT-RHO` |
| non-Gaussian `family` | "explicit precision fitting currently requires Gaussian responses" (`src/families/fit_gllvm.jl:167`) | existing |

Formula-only guards have no analogue in a matrix API and are fenced in section 4: a global
`phylo_tree` with no consuming term (`R/fit-multi.R:4611-4627`), the `unit != species`
foot-gun for `phylo_latent + latent(species)` (`:3058-3066`), and duplicate-term errors.

---

## 3. Julia design

### 3.1 Which existing path, and why

GLLVModels.jl has two different phylogenetic Gaussian models. Only one is R's.

| Path | Model | Fit for the twin? |
|---|---|---|
| Native J3 block in `fit_gaussian_gllvm` (`src/likelihood.jl:132-150`, `:379-395`) and its consumers `fit_phylo_dep_gllvm`, `fit_animal_latent_gllvm`, `em_phylo.jl` | Loadings live on the **rows of Y** (the entities on the tree); marginal is the Hadamard form `B = (Lambda_phy Lambda_phy') .* Sigma_phy` (`src/extract_gamma.jl:9-17`: "not R's trait⊗species Kronecker form"). `sigma_phy` has a signed identity link (`src/likelihood.jl:379-383`). | **No.** Different model. Stays a documented extra. |
| `PrecisionPhy` + `multivariate_phylo_precision_loglik` + `fit_precision_multivariate` (`src/phylo_precision.jl`, `src/precision_multivariate.jl:8-35`, `src/precision_multivariate_fit.jl:184-275`) | Loadings on **traits**, one sparse augmented field per factor with prior precision `Q`, shared Gaussian residual, exact marginal by one sparse Cholesky solve of the joint `(K * n_aug)` system; positive log links for unique and residual variances (`:41-64`, `exp.(2 .* ...)`) | **Yes.** It is R's `phylo_rr` block in Julia, root dropped and tips last, with R's `log_det` checksum (`src/phylo_precision.jl:1-27`, `precision_logdet_check` at `:215`). |

Cost. One objective evaluation is one CHOLMOD factorisation of a sparse system of order
`K * n_aug` plus residual terms; for a tree `n_aug = 2p - 2` so the fill is linear in the
number of species. The gradient is finite-difference (`_pmv_fd_gradient!`,
`src/precision_multivariate_fit.jl:104-118`): `2 * n_par` evaluations per gradient, with
`n_par = T + T*K - K(K-1)/2 + 1`. CHOLMOD blocks ForwardDiff, stated at `src/fit_phylo.jl:16-17`
and in the fitter docstring (`:197-199`); the sparse phylogenetic path in the native engine
carries the same limitation (`src/GLLVModels.jl:52`). At the A15 shape (T = 20, K = 2, 100
species, 5 replicates) `n_par = 60`, so a gradient costs about 120 factorisations of an order
about 400 system, which is milliseconds each. A hand-coded gradient through the Takahashi
selected inverse (`src/takahashi_selinv.jl`, the pattern in `src/sparse_phy_grad.jl`) can
follow later; A14 and A15 do not need it.

Convergence rule. The fitter already requires `Optim.converged`, a valid final objective and
`gradient_norm <= g_tol` (`:261`), which is the rule packet row 5 (#485) asks every fitter
to adopt.

### 3.2 New files and edits

| File | Kind | Content |
|---|---|---|
| `src/phylo_latent.jl` | new | `fit_phylo_latent_gllvm`, the three source admissions (tree, dense vcv/A with the `1e-8` ridge, sparse Ainv), label matching, the refusal table of section 2.3. Wraps `fit_precision_multivariate`. |
| `src/phylo_latent_postfit.jl` | new | `extract_Sigma(::PrecisionMultivariateFit; level, part)`, `extract_phylo_signal(::PrecisionMultivariateFit)`, `show`. |
| `src/precision_multivariate_fit.jl` | edit, additive | two fields appended to `PrecisionMultivariateFit` (`species_labels`, `tip_labels`), filled through the outer constructor at `src/precision_multivariate_fit.jl:173`, which is the only construction site, so existing callers and tests do not change. |
| `src/GLLVModels.jl` | edit | two `include`s after line 177; export `fit_phylo_latent_gllvm`. |
| `test/test_phylo_latent_twin.jl` | new | the red-first twins of section 4. |
| `test/test_phylo_latent_paired_p1.jl` | new | replays the P1 receipts of section 5 from committed JSON; skips, never passes, when a receipt is absent. |
| `tools/phylo_latent/r_reference_p1.R`, `tools/phylo_latent/compare_phylo_latent_p1.jl` | new | adapted from `tools/destination_b/phylo_gaussian_reference.R` and `compare_phylo_gaussian_reference.jl`, re-pinned to P1. |
| `docs/dev-log/core070/phylo-latent-p1/` | new | receipts. |
| `docs/design/capability-status.md`, `docs/src/tutorial.md`, `docs/src/api.md`, `ROADMAP.md`, `docs/dev-log/check-log.md`, `docs/dev-log/decisions/<date>-phylo-latent-parameterisation.md` | edit | convention-change cascade (AGENTS.md); the decisions note is required by design rule 4 because the twin fixes the phylogenetic scale at one on the unit-height tree. |

Not touched, by the brief: the shared `_laplace_mode`, `src/families/mixed.jl`,
`src/families/grouped_dispersion.jl`, `src/model_selection.jl`, `src/cv.jl`, the Normal branch
of `src/formula.jl`, and the `confint` entry points (`src/confint.jl`, `src/confint_profile.jl`
term builders) owned by the Gaussian-intercepts lane (#519). The twin's intervals live in
`precision_multivariate_intervals`, so it does not need any of them.

### 3.3 Entry points mirroring R's names

| R (P1) | Julia |
|---|---|
| `phylo_latent(species, d, tree, vcv, A, Ainv, unique, rho)` (`R/brms-sugar.R:771-782`) | `fit_phylo_latent_gllvm(Y, species; d, tree, vcv, A, Ainv, unique, rho)` |
| `.gllvm_phylo_tree_precision(tree, correlation = TRUE)` (`R/phylo-tree-precision.R:183`) | `PrecisionPhy(tree; correlation = true)` (`src/phylo_precision.jl:141`) |
| `extract_Sigma(level = "phy")` | `extract_Sigma(fit; level = :phy)` |
| `extract_phylo_signal(fit)` | `extract_phylo_signal(fit)` |
| `fit$report$Lambda_phy`, `fit$report$Sigma_phy` | `fit.loading`, `extract_Sigma(fit; level = :phy, part = :shared)` |
| `fit$opt$par["log_sigma_eps"]` | `log(sqrt(fit.residual_variance[1]))` |

---

## 4. Block-by-block map of in-scope R phylo tests to red-first Julia twins

Method. Every `tests/testthat/test-*.R` file at P1 that mentions `phylo` was parsed for
`test_that()` blocks and `expect_*()` calls (202 blocks, 573 expectations across the 25 files
examined per block; a further 208 blocks in whole-file fences, listed in 4.4). "Twin" means a
Julia `@testset` written red first with the stated tolerance. "Deferred" means inside the
boundary but outside the A14/A15 bar or dependent on a later slice. "Fenced" means it will
not be twinned, with one of three reasons: out of boundary (another keyword, family, or
surface), R-internal (parser, print, dotted helper, TMB slot), or dev-only (behind
`GLLVMTMB_HEAVY_TESTS`, `tests/testthat/setup.R:16-21`).

Tolerances follow the signed contracts: logLik `rtol 1e-6`, objective re-evaluation `1e-8`,
point estimates `rtol 1e-4` (`docs/dev-log/core070/phylo-transport-design.md` section 3;
`delta-matched-contract.md:17`); structural equalities are exact.

### 4.1 Twinned (26 R blocks, 21 Julia testsets)

| R block (file:line, expectations) | Julia twin | Tolerance | Reason |
|---|---|---|---|
| `test-phylo-hadfield.R:53` sparse Ainv (3) | tree route gives `n_aug > p`, `nnz(Q) < 0.1 n_aug^2` and `< 600` at p = 50 | exact | same construction (`2p - 2` rows, `1/edge` entries) |
| `test-phylo-hadfield.R:70` dense path (2) | vcv route gives `n_aug == p`, `nnz > 5p` | exact | R's dense inverse |
| `test-phylo-hadfield.R:85` tree vs vcv identical MLE (6) | both routes converge; `loglik`, `Sigma_phy`, `beta` agree | `1e-4` (R's own tolerance; ridge makes the two models differ at `1e-5`, `R/brms-sugar.R:689-693`) | same marginal model after integrating internal nodes |
| `test-phylo-hadfield.R:114` tip map (3) | `species_aug_id` values in `1:n_aug`, one row per species | exact | 1-based in Julia |
| `test-phylo-hadfield.R:134` non-phylo tree (1) and `test-gllvmTMB-args.R:202` (1) | `tree = "not a tree"` refuses with the R sentence | substring | one testset serves both |
| `test-phylo-tree-precision.R:6` precision inverts to `ape::vcv` on tips (6) | `inv(Q)[tips, tips]` equals the path-sum covariance divided by height | `1e-10` | already proven for one fixed tree (`test/test_destination_b_tree_precision.jl:3-30`); generalise to the twin's fixture |
| `test-phylo-tree-precision.R:42` species-to-node map (3) | `species_aug_id[tip_index]` equals the label position | exact | |
| `test-stage35-phylo-rr.R:35` fit and shapes (8) | converged; `size(fit.loading) == (4, 2)`; `Sigma_phy` is `4 x 4` with positive diagonal; `fit.loading[1, 2] == 0` | exact | `unpack_lambda` already lower-triangular (`src/packing.jl:47-61`) |
| `test-stage35-phylo-rr.R:51` requires a source (1) | no source refuses with the R sentence | substring | |
| `test-phylo-vcv-A-aliases.R:72`, the `phylo_latent` arm only (1 of 1) | `A = C` and `vcv = C` give byte-identical fits | exact | the other four keywords are fenced |
| `test-phylo-vcv-A-aliases.R:107` (1), `:119` (1) | two sources refuse | substring | |
| `test-phylo-latent-unique-fold.R:176` `unique = FALSE` loadings-only (3) | `fit.phylo_unique_variance === nothing`; `part = :unique` is zero | exact | |
| `test-phylo-signal-categorical.R:117` no advisory for Gaussian (1) | `extract_phylo_signal` on a bare fit logs nothing | exact | |
| `test-phylo-signal-categorical.R:137` typed refusal, no phylo tier (1) | `extract_phylo_signal` on a `GllvmFit` without phylo refuses | substring | R message "Phylogenetic signal requires a phylogenetic component." (`R/phylo-signal-ci.R:55-60`) |
| `test-latent-rank-guard.R:128` `d = T + 1` refuses (1) | refuse with the rank sentence | substring | |
| `test-latent-rank-guard.R:162` `d == T` fits (3) | converged at `d = T` on a small fixture | exact flag | R's version is heavy-gated; the twin is not |
| `test-species-unused-levels-guard.R:47` (1), `:62` (1), `:78` (1) | tree, vcv and Ainv routes: a label present in `species` but absent from the source refuses; the message names `droplevels()` when the label has no observations | substring | three routes, one testset with three cases |
| `test-species-unused-levels-guard.R:116` observed species missing from tree (4) | message lists the missing label as a "genuine mismatch" | substring | |
| `test-species-unused-levels-guard.R:143` clean fit unaffected (2) | a fit with every tip observed converges | exact flag | |
| `test-gllvmTMB-args.R:175` vcv without rownames (1) | `vcv` without `tip_labels` refuses | substring | |
| `test-gllvmTMB-args.R:187` vcv rows do not cover species (1) | refuse | substring | |
| `test-phylo-tree-unused-guard.R:93` a real term does not trigger the guard (1) | covered by every successful fit above | none | counted, no separate testset |

Julia-only red-first tests added alongside (not twins of an R block, required by the design
rules): `PrecisionPhy(tree; correlation = true)` log-det checksum on the twin fixture;
`fit_phylo_latent_gllvm` at `d = 1` recovers a planted `Sigma_phy` on an ADEMP fixture with
StableRNGs (AGENTS.md design rule 1, recovery test for the new public route).

### 4.2 Deferred (16 R blocks)

| R block | Why deferred | Where it lands |
|---|---|---|
| `test-stage35-phylo-rr.R:61` `phylo_latent + latent(site) + diag` (4) | needs a joint phylo-plus-rank-K-site-latent objective; `fit_joint_phylo_grouped_gaussian` admits `:indep` terms only (`src/joint_phylo_grouped_fit.jl:15-16`) | later slice, after A14/A15 |
| `test-phylo-latent-unique-fold.R:149` `unique = TRUE` equals `phylo_latent + phylo_unique` (4) | `unique = TRUE` is a documented extra here; `mode = :explicitunique` exists and uses R's positive link | `unique = TRUE` slice |
| `test-phylo-tree-unused-guard.R:236` `unique = TRUE` does not warn (1) | same | same |
| `test-phylo-signal-ci.R:147` (10), `:184` (12), `:265` (4) H2 Wald and bootstrap CIs | second order and three-piece; R calls them exploratory | 2SO row if ever opened |
| `test-phylo-signal-species-level.R:44` (6), `:64` (2), `:81` (5) crossed design `q_it` | three-piece decomposition (PHY-03), not bare | later slice |
| `test-derived-phylo-ci-audit.R:56` (10), `:79` (12), `:121` (2) | intervals on derived phylo quantities | 2SO |
| `test-confint-derived.R:217` (1), `:229` (1), `:243` (1), `:256` (1) `confint(parm = "phylo_signal")` | `confint` entry points are owned by #519; intervals are 2SO | after #519 lands |

### 4.3 Fenced, enumerated (76 R blocks)

| File | Blocks | Reason |
|---|---|---|
| `test-phylo-mode-dispatch.R` | 6 (`:33,:56,:76,:96,:113,:127`) | R-internal: `phylo(mode = ...)` alias grammar and byte-identity of parser rewrites; no formula surface in the twin |
| `test-phylo-vcv-optional.R` | 6 | R-internal: legacy `phylo()` keyword and global `phylo_vcv` argument grammar |
| `test-phylo-vcv-A-aliases.R` | 2 (`:36,:55`) | out of boundary: `phylo_scalar` |
| `test-phylo-tree-unused-guard.R` | 10 (`:50,:71`; `:142,:160,:186,:209,:263,:309,:325,:341`) | `:50,:71` formula-only global-argument guard; the rest are `phylo_indep`, `phylo_unique`, `phylo_scalar`, `phylo_dep` (out of boundary) |
| `test-phase56-1-phylo-augmented-stub.R` | 1 | out of boundary: augmented slopes (PHY-17) |
| `test-phase56-3-phylo-unique-parser.R` | 5 | R-internal parser; `phylo_unique` slopes |
| `test-phylo-latent-unique-fold.R` | 6 (`:16,:30,:43,:65,:78,:188`) | R-internal parser markers and duplicate-term error |
| `test-phylo-q-decomposition.R` | 5 | dev-only (`skip_if_not_heavy`) and three-piece |
| `test-phylo-signal-ci.R` | 2 (`:81,:112`) | R-internal dotted helpers; the public refusal is twinned via `test-phylo-signal-categorical.R:137` |
| `test-latent-rank-guard.R` | 3 (`:102,:114,:141`) | out of boundary: ordinary `latent()` ranks and the slope path |
| `test-species-unused-levels-guard.R` | 1 (`:96`) | out of boundary: `phylo_scalar` |
| `test-extract-sigma-augmented-unique.R` | 3 | out of boundary: augmented slopes |
| `test-comparator-gllvm.R` | 6 | out of boundary: R-vs-`gllvm` comparator, non-Gaussian, not phylogenetic |
| `test-canonical-keywords.R` | 6 (`:47,:361,:429,:471,:508,:572`) | R-internal alias recognition and `print`; `unit != species` foot-gun is formula-only; `phylo_indep`/`phylo_dep` out of boundary |
| `test-gllvmTMB-args.R` | 1 (`:166`) | out of boundary: `propto()` |
| `test-phyloscalar-binary.R` | 1 | out of boundary: `phylo_scalar`, binomial, dev-only |
| `test-phylodepindep-binary.R` | 2 | out of boundary: `phylo_indep`/`phylo_dep`, binomial, dev-only |
| `test-m1-7-extract-omega-phylo-signal-mixed-family.R` | 3 | out of boundary: mixed families, dev-only |
| `test-phylo-signal-categorical.R` | 7 (`:61,:78,:96,:146,:197,:239,:291`) | out of boundary: ordinal and multinomial liability H2 |

### 4.4 Owned by the temporal spec (23 R blocks)

Temporal is inside P1 (D-295). The four temporal-phylo files at P1 contain no `phylo_latent()`
call (census: 0 mentions in each) and belong to the temporal port spec (PR #535). They are
recorded there, not fenced here: `test-temporal-program-phylo-replicated.R` (5 blocks; gated on
`TMB`/`ape` only) is **deferred** to #535; `test-temporal-phylo-damped-newton.R` (8),
`test-temporal-phylo-optimizer-qualification.R` (6) and `test-temporal-phylo-third-pass.R` (4)
are **dev-only** there (they skip when the temporal programme files are absent from an
installed check, `test-temporal-phylo-damped-newton.R:8`).

### 4.5 Fenced, whole files (185 R blocks)

Block counts from the census; the reason is given per group.

| Surface | Files (blocks) |
|---|---|
| Non-Gaussian phylo matrix cells: **dev-only, non-Gaussian, not bare** for the 8 blocks that call `phylo_latent()` (the paired `phylo_latent + phylo_unique` block in each of `test-matrix-{poisson,nbinom2,gamma,beta,ordinal}-phylo.R`, all `skip_if_not_heavy`, and the three multinomial equivalence blocks `test-matrix-multinomial-phylo.R:315,376,456`, heavy); **out of boundary (other keyword)** for the remaining 27 blocks (`phylo_scalar`, `phylo_indep`, `phylo_dep`, animal and kernel keywords). Whether non-Gaussian `phylo_latent()` is inside P1 is OQ-11. | 20 blocks in the five family files, 15 in the multinomial file |
| Augmented slopes (PHY-11 to PHY-18): out of boundary. D-297 classes the slope and column-coefficient family (`slope`, `kernel_slope`, `spatial_slope`, `*_coef`) `outside_boundary`, revisited at P2; `phylo_slope` and the augmented `phylo_*(1 + x \| sp)` forms predate P1 and are fenced by A14's bare scope with D-297 as the class precedent. 58 of these 82 blocks are also `skip_if_not_heavy` (dev-only). | `test-matrix-slope-phylo-latent.R` (7), `test-phylo-latent-slope-gaussian.R` (5), `test-matrix-slope-phylo-dep.R` (14), `test-matrix-slope-phylo-indep.R` (6), `test-phylo-dep-slope-gaussian.R` (8), `test-phylo-dep-slope-s2-gaussian.R` (6), `test-phylo-indep-slope-gaussian.R` (2), `-nongaussian.R` (5), `-spike.R` (3), `test-phylo-unique-slope-{gaussian,binomial-logit,binomial-probit}.R` (3 each), `test-phylo-column-slope-indep.R` (8), `test-phylo-slope.R` (2), `test-phylo-slope-rhs-routing.R` (3), `test-ordinary-column-slope-phylo-coexistence.R` (4) |
| Column coefficients | `test-column-coef-phylo-fixed-rho.R` (10), `test-column-coef-phylo-estimated-rho.R` (14) |
| R-internal | `test-missing-predictor-phylo.R` (12): `mi()` covariate-model grammar with zero `phylo_latent()` calls (census); `test-va-r3-structured-phylo.R` (11): internal VA/KL machinery (`Stage 7 does NOT open the public route`, `:574`) |
| Other keywords: out of boundary | `test-stage3-propto-equalto.R` (7), `test-relmat-*-slope-gaussian.R` (3, 4, 3, 3), `test-funcphylo-spatial-recovery.R` (1) |

Animal, kernel, structured-rho and multinomial-fence files that mention `phylo` only through
shared helpers are outside this ledger row and are not counted.

### 4.6 Counts

| Class | R blocks | Notes |
|---|---|---|
| Twinned | 26 | 21 Julia testsets plus 2 Julia-only red-first tests |
| Deferred here | 16 | inside the boundary, outside the A14/A15 bar |
| Owned by the temporal spec (#535) | 23 | 5 deferred there, 18 dev-only there |
| Fenced, enumerated | 76 | 21 R-internal, 5 dev-only, 50 out of boundary |
| Fenced, whole files | 185 | 23 R-internal, 8 dev-only (non-Gaussian, not bare), 154 out of boundary (82 of them slope, 58 of those also heavy) |
| Total examined | 326 | |

---

## 5. Receipts per case-map row

One receipt directory `docs/dev-log/core070/phylo-latent-p1/<case>/` per row, following the
existing frozen-R schema (`docs/dev-log/decisions/destination-b-phylo-gaussian-reference-schema.md`)
re-pinned to P1. The existing Destination B evidence was recorded against gllvmTMB 0.7.0
`b4d5fee64` and is precedent, not a P1 receipt: cross-evaluation deltas of `0` and
`1.07e-14`, own-optimum NLL delta about `6.4e-14` on the tree cell, matched Wald covariance
relative error `3.9e-6` (`destination-b-s3b-pilot/README.md`, `destination-b-tree/README.md`,
`destination-b-uncertainty/README.md`). Those numbers show the kernel pairs; they do not
discharge A14 or A15 at P1, by the receipt carry rule (D-295 rule 1; `docs/dev-log/core070/true-parity-latest/GATES.md:129`): a 0.7.0 receipt
counts at P1 only if every source-pin file is byte-identical, otherwise `PARTIAL_STALE_AT_P1`.
Between `b4d5fee64` and `9539352f6`, `src/gllvmTMB.cpp` changed by +866 lines and
`R/fit-multi.R` by +2375 (`git diff --stat`), so every phylo receipt is stale at P1 and must be
re-run.

| Case | Model | R side (P1) | Julia side | Paired quantities and tolerance |
|---|---|---|---|---|
| `STRUCT-PHY-TREE-RR` (A14) | 3 traits, 8-tip ultrametric tree of height 4 (the Destination B fixture, `destination-b-tree/precision-reference.json`), 2 replicates, `d = 1`, `tree =` | `gllvmTMB(value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree), cluster = "species", family = gaussian())` from a private library built at `9539352f6`; DLL SHA-256, `sessionInfo()`, refined nlminb `rel.tol = 1e-12`; `se = FALSE` | `fit_phylo_latent_gllvm(Y, species; d = 1, tree = phy)`, Julia 1.10 (reference platform, decision X-03), `origin/main` SHA recorded | logLik at each own optimum `rtol 1e-6`; Julia objective at R's `theta_hat` and R objective at Julia's `theta_hat` (`abs 1e-8` after coordinate reorder `beta, theta_rr_phy, log_sigma_eps`); `Sigma_phy`, `beta`, `sigma_eps^2` `rtol 1e-4`; `log_det` checksum `abs 1e-8`; gradient at the other engine's optimum `<= 1e-4` |
| `STRUCT-PHY-DENSE-RR` (A14) | same data, `vcv = ape::vcv(tree, corr = TRUE)` | same call with `vcv = C` | same with `vcv = C, tip_labels` | same table; additionally tree-vs-vcv agreement on each side `1e-4`, matching `test-phylo-hadfield.R:99-107` |
| `COV-PHYLO-LATENT-RSZ` (A15) | T = 20 traits, 100 species (`ape::rcoal`, seed fixed), 5 replicates per species (n = 500 observations, 10 000 long rows), `d = 2`, `tree =` | same call; `cond(H)` from `sdreport` `cov.fixed` recorded | same; `fit.hessian_condition_number` recorded | same table; `cond(H)` on both sides recorded, not gated; wall time on both sides recorded |
| `STRUCT-PHY-TREE-PROPTO` | `phylo_scalar()` | not run under this spec | not run | maintainer question OQ-10; recommended disposition is a written fence with the case left `UNPAID` on the row |

Each receipt JSON carries: `source_pin = 9539352f66f2db2cc26b1c393e67212a359b60c9`,
`r_version = 0.7.1`, DLL hash, response hash (Float64 little-endian, `digits = 17`), the
`(i, j, x)` triplets of `Ainv_phy_rr` and `log_det_A_phy_rr`, `species_aug_id` (0-based), the
R `theta_hat`, R logLik, R gradient, the Julia `theta_hat`, Julia logLik, both cross
objectives, and `qualified = false` until the maintainer signs. The replay test
`test/test_phylo_latent_paired_p1.jl` recomputes the Julia side from the stored response and
asserts every tolerance; when the R JSON is absent it `skip`s and the skip is counted, not
passed.

Compute. A14 runs in seconds locally. A15: the Julia side is estimated at under five minutes
(section 3.1) and the R side at a similar order; both under the three-hour line
(D-287), so no pre-run gate, but the wall time is recorded in the receipt. Totoro's pinned
library (X-03) is the authority when a local build and the Totoro build disagree.

---

## 6. Symbolic alignment table

| Symbol | Meaning | R at P1 | Julia | Check |
|---|---|---|---|---|
| `T` | traits | `n_traits` | `size(Y, 1)` (`d` in `_pmv_layout`) | shapes |
| `K` | phylogenetic rank | `d_phy` (`R/fit-multi.R:2525`) | `rank` | `1 <= K <= T` both sides |
| `Lambda_phy` (`T x K`) | loadings, lower-triangular, signed diagonal | `gll_unpack_rr_loadings(theta_rr_phy, T, K)` (`src/gllvmTMB.cpp:34-59`) | `unpack_lambda(theta, T, K)` (`src/packing.jl:47-61`; index rule derived from the C++ at `:28-36`) | same packing order (diagonals first, then strict-lower column-wise), same length `TK - K(K-1)/2` |
| `Sigma_phy = Lambda Lambda'` | identified estimand | `REPORT(Sigma_phy)` (`:2182-2183`) | `loading * loading'` | paired `rtol 1e-4` |
| `A`, `Q = A^{-1}` | tree covariance and precision over `n_aug` nodes | `Ainv_phy_rr`, built at `R/phylo-tree-precision.R:209-227` | `PrecisionPhy.Q` (`src/phylo_precision.jl:141-188`, from `AugmentedPhy` by root drop and permutation) | `log_det` checksum `abs 1e-8`; `inv(Q)[tips,tips]` equals the path-sum covariance / height |
| `scale` | unit-height rescale | `scale = height` when `correlation = TRUE` (`:223`) | `PrecisionPhy.scale` with `correlation = true` (`:122-160`) | `scale == height` |
| `n_aug` | rows of `Q` | `n_tip + Nnode - 1` (`src/gllvmTMB.cpp:918-923`) | `n_aug == 2p - 2` for bifurcating trees | exact |
| `species_aug_id` | observation to node row | 0-based (`R/fit-multi.R:4690`) | 1-based `species_aug_id[species_id[o]]` | `julia == r + 1` |
| `g_k` | field for factor `k` | `g_phy(:, k) ~ N(0, A)` (`:2154`) | integrated exactly by the sparse Cholesky (`src/precision_multivariate.jl:24-30`) | not a parameter; enters only through the marginal |
| `b` | per-trait intercepts | `b_fix` from `0 + trait` | `beta` from `_trait_mean_design` | `rtol 1e-4` |
| `sigma_eps^2` | one shared residual | `exp(2 * log_sigma_eps)`, one slot (`:3262-3266`) | `exp(2 * theta[residual])`, `residual_mode = :shared` (`src/precision_multivariate_fit.jl:41-64`) | `rtol 1e-4` |
| `sigma2_phy` | phylogenetic scale | fixed at 1, absorbed by `Lambda` (`R/fit-multi.R:4666-4668` comment) | `sigma2_phy = 1.0` fixed (`:73-99`) | identical convention |
| `Psi_phy` (extra) | per-trait phylogenetic unique variance | `exp(2 * log_sd_phy_diag)` (`:2186-2203`) | `exp.(2 .* theta[unique])`, `mode = :explicitunique` | same positive link; not bare |
| `H2_t` | phylogenetic signal | `Sigma_phy[t,t] / (Sigma_phy[t,t] + Sigma_non[t,t] + Psi_non[t,t])` (`R/extract-omega.R:329-333`) | same formula; `1` for a bare fit | exact |
| logLik | marginal Gaussian log-likelihood | `-fit$opt$objective` (Laplace is exact here) | `fit.loglik` | `rtol 1e-6`, cross-objective `abs 1e-8` |

---

## 7. How #136 and #129 interact with twinning

Facts.

- #136: the native Julia J3 block gives `sigma_phy` a signed identity link
  (`src/likelihood.jl:379-383`), with a sign-flip restart and a global sign anchor in `fit.jl`,
  whereas R's co-fitted `phylo_diag` uses `exp(log_sd_phy_diag)` (`src/gllvmTMB.cpp:2199`,
  `:3198`). Note the nuance at P1: R's **lone** `phylo_unique()` is rerouted to `phylo_rr`
  with a diagonal `Lambda_phy` whose diagonal is signed (`R/fit-multi.R:2376-2388`, `:2525-2533`,
  `:6527`), so R is positive-linked only in the co-fit. Because R's diagonal `Lambda` has no
  cross-loadings, the sign never reaches a cross-trait covariance there; in Julia's native
  model it does, because `sigma_phy` multiplies a shared species-row factor.
- #129: `sigma_phy` is tagged `:linear` in the Wald term builder (`src/confint.jl:102-107`) and
  `:log_sd` in the profile term builder (`src/confint_profile.jl:117-122`), so `profile_ci`
  exponentiates a natural-scale value. Confirmed live (packet row 35).

Interaction with the twin. Neither issue touches the twin's code path. The twin is built on
`precision_multivariate`, where unique and residual variances already use R's positive log
link (`src/precision_multivariate_fit.jl:41-64`) and intervals are transformed Wald in
`precision_multivariate_intervals`, not `confint`. So:

1. **Port R's parameterisation for the twin** (done by construction), and **keep the native
   signed `sigma_phy` as a documented extra** with its own name in the capability matrix. This
   is the packet's recommendation for #136 (row 25: "document the divergence rather than unify
   the link"). Recommendation: adopt it; write the divergence into the new decisions note and
   into `docs/src/tutorial.md` beside the `PrecisionPhy` section (`:482-515`), and do not
   change the native engine in this programme.
2. **Fix #129 as its own small PR** (tag `sigma_phy` `:linear` in `_profile_all_term_names`,
   packet row 35), sequenced after #519 because both edit the `confint` term builders that
   lane owns. The twin does not wait for it. Until it lands, any gate-tier row that reports a
   native phylo-unique profile CI is wrong, and the twin must not cite one.
3. The R lone-`phylo_unique` signed-diagonal fact belongs in the `unique = TRUE` slice's
   symbolic table, not here; it is recorded now so the later slice does not rediscover it.

---

## 8. Open questions for the maintainer, with recommendations

| Id | Question | Recommendation | Reply to paste |
|---|---|---|---|
| OQ-1 | Ratify the Q1 to Q4 transport defaults (B-05)? The twin entry sets `correlation = true` on its own path while native constructors keep opt-in `false` (Q1); dense `vcv` is admitted with a κ warning (Q2); kernel stays split (Q3); non-ultrametric trees are refused on the twin path, accepted natively (Q4). | Ratify all four as written; this spec depends on Q1 and Q4 only. | "OQ-1: ratify Q1-Q4; twin path uses correlation = true." |
| OQ-2 | A15 shape. The tier key says p ≥ 20, n ≥ 500. Read as T = 20 traits and n = 500 observations (100 species × 5 replicates), or as 500 species? | 100 species × 5 replicates × 20 traits. Replication is what identifies the shared residual against `Sigma_phy` (the R roxygen makes the same point for `Psi_phy`, `R/brms-sugar.R:1103-1105`); 500 tips with one draw each is a boundary-collapse regime (`phylo-transport-s3-julia-fit.md`). | "OQ-2: A15 = 100 species × 5 replicates × 20 traits, d = 2." |
| OQ-3 | Species matching. Port R's label semantics (observed label absent from the source refuses; unobserved tips are marginalised) or keep integer `species_id` only? | Port labels; keep `species_id` reachable through `fit_gllvm(; phylo = ...)`. | "OQ-3: labels on the twin entry." |
| OQ-4 | Dense `vcv`: replicate R's `1e-8` ridge exactly (needed for `STRUCT-PHY-DENSE-RR` to pair at `1e-6`) even though it is a numerical artefact? | Yes, exactly, and say so in the docstring; pairing is the point of the row. | "OQ-4: replicate the ridge." |
| OQ-5 | `extract_phylo_signal` on a bare fit returns `H2 = 1` per trait (R semantics) rather than the current Julia refusal `estimand_not_admitted`. Port or keep the refusal? | Port, with the same `V_eta` column so the number is interpretable, and keep the refusal for `ci = true`. A twin that refuses where R answers is not a twin. | "OQ-5: port H2 = 1 with V_eta; refuse ci." |
| OQ-6 | Entry-point name. `fit_phylo_latent_gllvm` (repo convention) versus only extending `fit_gllvm(; phylo = ...)`. | The named entry, plus the untouched `fit_gllvm` route. R's keyword name should be findable by an R user. | "OQ-6: named entry." |
| OQ-7 | Who lands the #129 fix, and when? | Its own PR after #519 merges, by whichever lane holds `confint_profile.jl` then. | "OQ-7: #129 after #519, separate PR." |
| OQ-8 | Gradient. Accept finite differences for A14 and A15, with the selected-inverse gradient as a later performance item? | Yes. Record wall time in the A15 receipt; open the gradient item only if A15 exceeds one hour on Totoro. | "OQ-8: FD now; gradient later if A15 > 1 h." |
| OQ-9 | Sign-off evidence for A14/A15 promotion: a dated maintainer block in the receipt PR body (the B-01/B-02 pattern)? | Yes. | "OQ-9: dated maintainer block in the PR body." |
| OQ-10 | `STRUCT-PHY-TREE-PROPTO` (`phylo_scalar()`) sits in A14's own ledger row. Fence it (scalar shared phylogenetic variance is a different model from the rank-K loadings model) and promote A14 with that planned case `UNPAID`, or require it before promotion? | Fence it, with the reason written on the row; A14 promotes on the two `phylo_latent` cases. If required later, it is a small extra slice on the same `PrecisionPhy` kernel (`rank = 1` with a shared loading and the variance as the free parameter). | "OQ-10: fence PROPTO; A14 promotes on the two phylo_latent cases." |
| OQ-11 | Is non-Gaussian `phylo_latent()` (Poisson, NB2, Gamma, Beta, ordinal, multinomial; 8 heavy R blocks) inside P1? | Inside the boundary by D-295's wording, but a later slice, not this spec: it needs a Laplace or AGHQ outer loop over the augmented field, which is `src/phylo_glm.jl`'s territory, and R's own evidence there is heavy-gated and partly failed (multinomial rail rates, `NEWS.md:1101-1112`). | "OQ-11: non-Gaussian phylo_latent is a later slice via phylo_glm.jl." |
| OQ-12 | R accepts `rho` (source-strength attenuation) on a bare `phylo_latent()`; the twin refuses `rho != 1` with its own scope fence. Accept the fence for A14/A15, or require `rho` pairing? | Accept the fence: `rho` is a separate estimand (`R/brms-sugar.R:762-770`, "not a variance-share summary") with its own R test files (`test-structured-rho-*.R`), and no ledger case in A14/A15 names it. | "OQ-12: accept the rho scope fence." |

---

## 9. Estimate and its basis

Basis. Packet row 8 (B-05) estimated 5 to 25 days before anyone had checked what exists.
The evidence above narrows it: the R-shaped kernel, the fitter, the precision consumer, the
admission validator, the reference runner, the comparison checker and the receipt schema all
exist and are tested (`test/test_precision_multivariate.jl:49-218`,
`test/test_precision_multivariate_fit.jl:120-253`, `tools/destination_b/`), and the kernel
has already paired with 0.7.0 to `1e-14`. What is missing is the public twin surface, the
P1 re-pin, the twin tests, the receipts and the documentation cascade.

| Slice | Work | Agent-days |
|---|---|---|
| S1 | `fit_phylo_latent_gllvm`, three source admissions, label matching, refusal table, two struct fields | 1.5 |
| S2 | 21 red-first twin testsets and 2 Julia-only tests (section 4.1) | 1.5 |
| S3 | P1 private R build (`9539352f6`), re-pinned reference runner and checker, DLL receipt | 1.0 (plus Totoro) |
| S4 | A14 receipts, two cases, both optima and both cross objectives | 1.0 |
| S5 | A15 receipt, `cond(H)` and wall time on both sides | 1.0 |
| S6 | Postfit extractors, docs cascade (capability matrix, tutorial, api, ROADMAP, check-log, decisions note), after-task report | 1.0 |
| Reserve | R build friction, an unexpected optimiser stall on A15, review rounds | 1.5 |
| Total | | about 8.5 agent-days, roughly the low end of B-05 |

Compute: A14 seconds; A15 minutes on Totoro on both sides (section 3.1), under the D-287
three-hour line. Excluded from this estimate: `unique = TRUE`, the joint site-latent objective,
any interval pairing (2SO), the bridge, and every fenced surface in section 4.

Verification checks per slice, in the plan sense: S1 passes the refusal table; S2 is red
before S1 and green after; S3 produces a receipt whose response hash Julia reproduces; S4 and
S5 pass `test/test_phylo_latent_paired_p1.jl` with no skips; S6 passes `Pkg.test()` in full
and the Rose claim-versus-evidence read of the capability matrix.
