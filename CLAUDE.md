# CLAUDE.md — guidance for AI coding agents in this repo

This file orients AI coding agents (Claude Code, Codex, etc.) to the GLLVModels.jl repo conventions and current state. The maintainer is Shinichi Nakagawa (itchyshin).

## What this package is

GLLVModels.jl is a Julia implementation of the Generalised Linear Latent Variable Model (GLLVM) class across a broad, status-tracked GLM response-family surface, plus phylogenetic and spatial (SPDE/Matérn) random-effect structures. It is a from-scratch reimplementation of R's `gllvm`/`gllvmTMB`, prioritising fitting speed at moderate-to-large p (species count) and rigorous inference. Headline result on the Gaussian path: median 265.1× (range 161–698×) on the published Gaussian closed-form profile grid over R/`gllvmTMB` while reproducing point estimates and likelihoods to at least six significant digits on the benchmarked single-variance grid (worst case `|Δ logLik| = 2.343e-07`; Core070 FREE=0 ≠ true parity — see `docs/src/gllvmtmb-parity.md`).

**Status**: v0.3.0 development has broad engine coverage, but public R-user capability is governed by the capability/bridge matrix rather than a blanket parity claim. The Gaussian path uses the closed-form marginal; the non-Gaussian families use a shared Laplace-approximated marginal (with a Gaussian-variational/ELBO alternative for several). Implemented or scaffolded engine rows include Gaussian (incl. per-species variance), Poisson, NB2, NB1, Binomial/Bernoulli, beta-binomial, Beta, Gamma, Exponential, Ordinal (logit/probit), Tweedie, Conway–Maxwell–Poisson, and the two-part/zero families (Delta-lognormal, Delta-Gamma, Hurdle-Poisson, Hurdle-NB, beta-hurdle, ordered-beta, ZIP, ZINB, ZIB). Plus per-species/grouped dispersion, fixed/species-specific covariates, fourth-corner, fixed and random row effects, quadratic response, the ordination trio (unconstrained / concurrent `num.lv.c` / constrained `num.RR`), a phylogenetic GLM, and an SPDE/Matérn spatial latent field. Treat each row as `covered`, `partial`, `experimental`, `planned`, or `unsupported` according to the live matrix and the `gllvmTMB` bridge gate.

**Size**: 78 commits, 77 source files, 108 test files as of v0.3.0; CI green across Julia 1.10 + stable on Linux/macOS/Windows.

## Working with this repo

### Dev environment

- Julia ≥ 1.10 (project compat). The maintainer's local Julia binary is at `~/.juliaup/bin/julialauncher`; `julia` may not be on every shell's PATH.
- Install deps: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- **Full suite (canonical)**: `julia --project=. -e 'using Pkg; Pkg.test()'` — verified working on macOS + all CI runners (the earlier `can not merge projects` problem is resolved; root `Project.toml` carries no conflicting `[extras]`/`[targets]`). This is what CI runs, and the only invocation that loads the test-only quality tools (Aqua, JET) declared in `test/Project.toml`.
- **Quick core run**: `julia --project=. test/runtests.jl` runs the core suite directly — faster, no test-env setup, but skips the quality tools (they live in the test environment).

- **Known sandbox-only failures (#875)**: on a non-standard local build (Julia 1.10.4 from conda-forge with SuiteSparse 7.2.1 swapped in), shard 2/4 shows 3 failures that pass on CI's standard Julia binaries: `test/test_phylo_latent_paired_p1.jl:135-136` (A15 live refit `converged` / `gradient_norm <= 1e-5`) and `test/test_phylo_precision_animal.jl:218` (animal `negll` vs dense, 1e-8). They reproduce identically on unmodified `main` in that environment. Treat them as an environment exception, not a package regression, unless they also fail on CI's standard binaries. Whether this is SuiteSparse/BLAS ordering noise or a real numerical-path difference has not been confirmed; do not loosen the tests to hide it.

### Design discipline (apply throughout)

- **Surgical changes**: touch only what the task requires. Don't refactor adjacent code. Match existing style.
- **Verifiable goals**: state the goal as a test or measurable check before writing code.
- **No silent tolerance widening**: if a test breaks, fix the cause, not the test.
- **Stage by name**: `git add path/to/file` — **never** `git add -A` or `git add .` (sweeps in unintended files; we have agents working on disjoint files in parallel).
- **One concern per commit**: keep engineering changes, cosmetic renames, and chores in separate commits.
- **Verify before claiming**: run the full test suite and paste the actual pass/fail tally before reporting work as done.
- **Honest reporting**: if a result is mixed or negative, say so plainly. Don't oversell.

### Source layout (src/)

The listing below covers the main groups; it is representative, not exhaustive (77 source files). The non-Gaussian families live under `src/families/`.

Core engine:
- `GLLVModels.jl` — module + includes
- `packing.jl` — Λ pack/unpack (lower-triangular convention)
- `likelihood.jl` — Gaussian marginal log-likelihood (closed-form, dense reference)
- `lowrank_cholesky.jl` — Woodbury-based Cholesky for ΛΛᵀ + diag
- `ppca_init.jl` — Probabilistic PCA closed-form initialisation (Tipping & Bishop 1999)
- `em_fa.jl` — EM-FA solver (alternative; Rubin & Thayer 1982)
- `profile.jl` — σ_eps analytic profile-out (lme4 / MixedModels.jl pattern)
- `fit.jl` — `fit_gaussian_gllvm` (Optim LBFGS + PPCA warm-start)
- `simulate.jl` — Julia-side data simulator

Phylogenetic representations (all compute the identical log-likelihood to machine precision; differ in cost and AD compatibility):
- `sparse_phy.jl` + `likelihood_sparse_phy.jl` — augmented-state sparse phylogenetic precision (Hadfield & Nakagawa 2010 *JEB* appendix), via CHOLMOD — **the fastest path**, ~O(p), but CHOLMOD blocks generic forward-mode AD
- `phylo_contrasts.jl` + `likelihood_contrasts.jl` — Felsenstein independent contrasts (Felsenstein 1985; Lande 1979)
- `edge_incidence.jl` + `likelihood_edge_incidence.jl` — edge-node incidence representation: matrix-free Q = B·W·Bᵀ; per-branch evolution rates naturally on `diag(W)`

Fitting at scale (closes the fast-and-fittable gap):
- `sparse_phy_grad.jl` — hand-coded analytic gradient for the sparse phylo path (TMB-style; the maintainer-approved Takahashi O(p) selected-inverse swap is the next planned optimisation, currently O(p²))
- `em_phylo.jl` — gradient-free EM fit using the fast sparse phylo solves in the E-step; conditional means double as ancestral-state BLUPs
- `em_squarem.jl` — SQUAREM extrapolation accelerator for EM (Varadhan & Roland 2008)
- `relaxed_clock.jl` — per-branch evolution-rate prototype on the edge-incidence substrate (with the hierarchical-prior identifiability caveat)

Confidence intervals:
- `confint.jl` — Wald via observed information (Hessian; log-scale back-transform for SDs, identity for signed loadings)
- `confint_profile.jl` — profile likelihood (LRT inversion + bracket-then-bisect)
- `confint_bootstrap.jl` — parametric bootstrap
- `confint_derived.jl` — profile + bootstrap CIs for derived quantities (Σ_y entries, communality c², cross-trait correlation, phylogenetic signal H²)
- `confint_derived_wald.jl` — transformed-scale Wald (Fisher-z for [−1,1] correlations, logit for [0,1] communality / ICC / H²); matches the bootstrap to within MC error at one-Hessian cost for interior-valued bounded quantities
- `confint_family.jl` — Wald / profile / bootstrap CIs for the non-Gaussian families

Non-Gaussian families (`src/families/`):
- `links.jl` — link functions (Identity, Log, Logit, Probit, CLogLog)
- `laplace.jl` — generic family-dispatched Laplace marginal core, shared across families (the `<family>_marginal_loglik_laplace` pattern)
- per-family pieces + fitters: `poisson.jl`, `negbin.jl`/`negbin1.jl`, `binomial.jl`, `beta.jl`, `gamma.jl`, `exponential.jl`, `ordinal.jl`, `tweedie.jl`, `com_poisson.jl`, `beta_binomial.jl`; two-part/zero families in `twopart.jl`, `beta_hurdle.jl`, `ordered_beta.jl`
- `grouped_dispersion.jl` — per-species / grouped dispersion (`disp.group`)
- model-structure add-ons: `covariates.jl`, `species_covariates.jl`, `fourthcorner.jl`, `row_effects.jl`, `row_random.jl`, `constrained_ordination.jl`, `rrr.jl`, `quadratic.jl`
- `variational*.jl` — Gaussian-variational (VA/ELBO) marginal alternative for several families
- `fit_gllvm.jl` — unified `fit_gllvm(Y; family)` dispatcher; the top-level `gllvm()` API wraps it

Spatial / non-Gaussian phylo:
- `spde.jl`, `spde_mesh.jl`, `spde_delaunay.jl`, `spde_fit.jl`, `spde_latent*.jl` — SPDE / Matérn-GMRF FEM spatial field, as a fitted model and as a latent variable inside a non-Gaussian GLLVM
- `phylo_glm.jl` — phylogenetic GLLVM for non-Gaussian families (augmented-state joint Laplace)

Post-fit & front-end:
- `postfit.jl`, `ordination.jl`, `model_selection.jl`, `summary_table.jl`, `simulate_fit.jl`, `formula.jl` — `predict`/`fitted`/`residuals`/`aic`/`bic`, ordination output, `select_lv`, `coef_table`, family-aware simulation, and the `@formula` front-end

## Planned next

The GLM-family expansion from the v0.1.0 Gaussian pilot is broad but still status-gated by row (see `ROADMAP.md` for the live checklist and remaining gaps to R `gllvm`/`gllvmTMB`). The Laplace marginal core is a single family-dispatched function (`src/families/laplace.jl`), each family adds `<family>_marginal_loglik_laplace` pieces and a `fit_<family>_gllvm` driver, and current routes are reachable through the unified `fit_gllvm` / `gllvm()` API — study that pattern before adding a family.

Current direction:
- **Julia General registration** (v0.3.0; first registration is a manual merge — see the registration notes).
- Performance: the maintainer-approved Takahashi O(p) selected-inverse swap for the sparse phylo gradient (`sparse_phy_grad.jl`, `takahashi_selinv.jl`).
- Keep extending parity / inference coverage as tracked in `ROADMAP.md`.

## Hard boundaries

- **No engine surgery on R's `gllvmTMB`** from this repo. That R package is a read-only reference; do not modify it.
- **No push without an explicit instruction** from the maintainer. Always commit locally first; ask before pushing.
- A separate benchmark/comparison repo exists locally (mixed R + Julia, comparison-final.md report, ADEMP simulation infrastructure). It is **not** in this repo and is intentionally separate.

## Key references

- Tipping & Bishop 1999 (PPCA closed form, *JRSSB*)
- Bates et al. 2015 (lme4 / MixedModels.jl; σ_eps profile-out, sparse Z'Z, *J Stat Soft*)
- Rubin & Thayer 1982 (EM for factor analysis, *Psychometrika*)
- Hadfield & Nakagawa 2010 (augmented-state sparse phylogenetic precision, *J Evol Biol* appendix)
- Felsenstein 1985; Lande 1979 (phylogenetic contrasts)
- Kristensen et al. 2016 (TMB; the sparse-Cholesky analytic-adjoint pattern, *J Stat Soft*)
- Morris et al. 2019; Williams et al. 2024 (ADEMP simulation reporting framework)
- Takahashi 1973; Erisman & Tinney 1975 (selected-inverse recursion for sparse matrices)
