# #1556 — `simulate(fit, n)` on `GllvmCovFit` (recommendation)

`GllvmCovFit` supports `simulate(fit, X)` only; site count `n` comes from `size(X, 2)`, so `simulate(fit, n::Integer)` is not part of the API.
The package has no existing refusal-only `simulate(fit, n::Integer)` pattern for design-required fits (unlike the one-argument `bic` `ArgumentError` guards in PR #1527 for `GllvmFit`).
Defer an informative `ArgumentError` stub until a package-wide convention exists for covariate/design-required fits; do not add a drawable `simulate(fit, n)` overload for `GllvmCovFit` alone.