# phylo_latent twin: parameterisation, provenance and two P1 findings (2026-09-27)

Scope: `fit_phylo_latent_gllvm` (`src/phylo_latent.jl`,
`src/phylo_latent_postfit.jl`), gate rows A14 (`COV-PHYLO-LATENT-1FO`) and A15
(`COV-PHYLO-LATENT-RSZ`). Design: `docs/design/phylo-latent-port-spec.md`
(draft PR #545, head `c0fd5f5be`). Signed scope: vault decision D-300
(packet 1d, answers 1 to 12 as recommended).

## Parameterisation (design rule 4)

The twin fits R's `phylo_rr` block exactly as gllvmTMB 0.7.1 does at
`9539352f66f2db2cc26b1c393e67212a359b60c9`:

    y[t, o] = b[t] + sum_k Lambda[t, k] g_k[aug(s(o))] + eps[t, o]
    g_k ~ N(0, A), independently over k;  eps ~ N(0, sigma_eps^2)

* `A` is the correlation-form (unit root-to-tip height) tree covariance over
  every non-root node, entered through its sparse precision. The phylogenetic
  scale is fixed at one and absorbed by `Lambda` (R hard-codes
  `correlation = TRUE`, `R/fit-multi.R:4677`). A raw-branch-length fit would
  differ by exactly the tree height.
* `Lambda` is `T x K` lower-triangular with a **signed** diagonal, packed
  diagonals first, then strict-lower entries column by column (`unpack_lambda`,
  identical to `gll_unpack_rr_loadings`, `src/gllvmTMB.cpp:34-59`). Only
  `Sigma_phy = Lambda Lambda'` is identified.
* One shared residual, `sigma_eps^2 = exp(2 * log_sd)`, R's single
  `log_sigma_eps` slot. Coordinate order differs only by position: Julia
  `[b; theta_rr; log_sd]`, R `[b_fix; log_sigma_eps; theta_rr_phy]`.
* The dense `vcv` / `A` route adds R's `1e-8` diagonal ridge before inversion
  and records `log_det = -logdet(A + 1e-8 I)` (`R/fit-multi.R:4740-4752`),
  replicated exactly (D-300 answer 4). The tree and dense routes therefore
  differ at about `1e-5` in log-density, as R documents.
* The tree precision follows R's builder (`R/phylo-tree-precision.R:183-249`):
  internal nodes first, tips last, entries `height / edge_length`, root row
  dropped, `log_det = n_aug log(height) - sum(log(edge_length))`. Polytomies
  give `n_aug = n_tip + Nnode - 1` as in R, through the raw-triplet
  `PrecisionPhy` constructor; `PrecisionPhy(::AugmentedPhy)` is not on the
  twin path.

The native Julia designs (Hadamard species-row loadings, signed `sigma_phy`,
contrasts, edge-incidence) are unchanged and stay documented extras.

## Provenance (design rule 9)

Ported from gllvmTMB P1, read only with `git show 9539352f6:<path>`:
the tree precision rule and ultrametric tolerance
(`R/phylo-tree-precision.R:70-249`), the dense ridge route and species-level
coverage diagnostics (`R/fit-multi.R:600-650`, `:4656-4755`), the rank guard
sentence (`:2536-2542`), the alias refusals (`R/brms-sugar.R:3297-3319`), and
the bare-fit `extract_phylo_signal()` columns (`R/extract-omega.R:468-640`).
Refusal messages start with R's rendered sentence (cli markup to plain text)
followed by a Julia tag.

## Two P1 findings that differ from the spec

1. **`Ainv =` route.** The spec maps the Julia `Ainv` keyword to R's sparse
   direct route (`R/fit-multi.R:4691-4723`). At P1 that route is reached only
   by a global sparse `phylo_vcv`; the in-keyword `phylo_latent(Ainv = X)` is
   rewritten to `vcv = solve(as.matrix(X))` (`R/brms-sugar.R:3311-3319`), the
   dense ridged route. Per the build brief (stop where R contradicts the
   spec), `Ainv` is refused by `GJL-GATE-PHYLO-LATENT-AINV` until the
   maintainer chooses which R behaviour the keyword twins.
2. **Unary nodes.** The spec says a node of out-degree one is refused "as R
   refuses it through `.gllvm_validate_phylo_tree`". At P1 that validator
   has no such check, and R's precision rule is well defined for unary nodes.
   No unary-node refusal was added; the builder follows R's rule.

## Other deviations, recorded

* `extract_Sigma(::PrecisionMultivariateFit)` already existed
  (`src/destination_b_postfit.jl`); `level = :phy` was added as R's name
  beside the existing `:phylo`, rather than a second method.
* `species_levels` was added to the entry so R's declared-but-unobserved
  factor levels (and the `droplevels()` branch of the coverage refusal) have a
  Julia analogue.
* `extract_phylo_signal` on a `GllvmFit` without a phylogenetic term still
  returns `NaN` (existing behaviour); twinning R's refusal there
  (`test-phylo-signal-categorical.R:137`) would change that public method and
  is deferred.
* The bridge (`_bridge_fit_precision_multivariate`) still reports the signal
  as `:estimand_not_admitted`; only the public extractor answers.
