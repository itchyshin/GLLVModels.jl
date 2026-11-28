# After-task: masked Laplace residuals are NaN (#1151)

**Claim.** `residuals` on a Poisson or Binomial Laplace fit returns `NaN` at
cells where `mask` is false, for both `:pearson` and `:dunnsmyth`. The
observed cells do not depend on the number stored in a masked cell.

**Files.** `src/postfit.jl` (`_nan_unobserved!` after the Laplace residual is
built). `test/test_masked_residuals_1151.jl`, wired in `test/runtests.jl`.

**Tests.** That file on Julia 1.10.13, 1 thread. Before: 8 passed, 8 failed.
After: 16 passed, 0 failed. The AGHQ branches already returned `NaN` and were
not changed. Full `Pkg.test` was not run.

**Rose verdict:** PASS WITH NOTES — the two Laplace routes named in the issue
now match the AGHQ and grouped-NB2 rule; other families were not probed.
