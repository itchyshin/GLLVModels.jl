# after-task: confint `objective = :va` for ExponentialFit (#1186)

## Goal

Enable Wald `confint(...; objective = :va)` on `ExponentialFit` VA fits via the ELBO Hessian, matching Poisson/Gamma/Delta-Gamma; default Laplace path unchanged.

## Files

- `src/confint_family.jl`: `_family_ci(::ExponentialFit)` VA branch; allow-list; docstring family list; unified ELBO objective check via `_family_ci(...; objective = :va)`.
- `test/test_variational_exponential.jl`: VA Wald / `coef_table` testset.

## Local tests (Julia 1.10.14, cloud VM)

- `julia --project=. test/test_variational_exponential.jl`: **39 passed**, 0 failed, ~15.5s.
- `julia --project=. test/test_confint_family.jl`: **345 passed**, 0 failed, ~8m06s.
- Issue reproducer (`fit_exponential_gllvm_va` + `confint(...; objective = :va)`): OK.

Full `Pkg.test()` not run (time budget).

## Rose verdict

**PASS WITH NOTES** — behaviour change is inference-only on an opt-in keyword; docstring list updated in-source; no R-parity or coverage claim.