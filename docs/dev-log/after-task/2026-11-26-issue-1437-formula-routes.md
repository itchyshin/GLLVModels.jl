# After-task: #1437 spider NB2 formula vs `fit_gllvm_cov` (2026-11-26)

## Goal

Diagnose whether `gllvm(@formula(...), …; family = NegativeBinomial())` and
`fit_gllvm_cov(...; X = X)` should return the same log-likelihood on eSpider-style data.

## Verdict

**Case (a) — different models by design.** The formula route calls
`fit_nb_gllvm_grouped_cov` (per-species `r`, default `group = 1:p`, `G = p`). `fit_gllvm_cov`
with `NegativeBinomial()` estimates one shared scalar `r`. Not a convergence defect or
likelihood bug.

## Evidence (eSpider, 12 × 28, scaled ConWate/BareSand/CovMoss; Julia 1.10.14, `iterations = 60`)

| Quantity | Value |
| --- | --- |
| `gllvm(@formula…)` logLik | −686.167995178808 (`NBGroupedCovFit`) |
| `fit_nb_gllvm_grouped_cov` logLik | −686.167995178808 (Δ = 0) |
| `fit_gllvm_cov` logLik | −705.327711463305 (`GllvmCovFit`, Δ ≈ +19.16 vs formula) |
| `fit_gllvm_cov`, `iterations = 500` | −705.327711463194 (unchanged optimum) |
| NB1 Δ(formula − shared) | ≈ +12.74 |

Three start seeds (`20261126`, `42`, `99001`) gave identical likelihoods (deterministic optimiser path on this data).

## User-facing change

Tutorial §5 corrected: the `@formula` route no longer described as “`fit_gllvm_cov` otherwise”; added explicit NB2/NB1/Beta/Gamma dispatch note and matched-route guidance.

## Tests

Docs-only; no new test file. Diagnostic runs were local tools scripts (not committed).

## Rose verdict

**PASS WITH NOTES** — documentation fix only; engine behaviour unchanged and already documented in `fit_gllvm_cov` / `formula.jl` docstrings. Tutorial had been stale.