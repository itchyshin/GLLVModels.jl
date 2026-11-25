# Gamma covariate fits with an unidentifiable predictor

## Source and results

The full suite on `2c647aed8745dbf4ac246fc3b02fb31159648b1f` completed on Totoro with 43,731 passing assertions, two failures, no errors, and 116 broken tests. Both failures were in `test/test_cov_mode_search_gamma.jl`. The test supplied an all-zero covariate, so its slope was unidentifiable. The objective and fitted log likelihood were finite, but the Hessian was not positive definite. Under the current fit contract, the result is `degenerate`, not converged.

The test now checks the finite fit, completed optimizer, small relative gradient, non-positive-definite Hessian, and `degenerate` status. The exact revised source tree is `ef4f207f1d87badc6c7eae257a93a6a8e4b54e2b` (commit `c1014d1728a2b6e7c1c868a53cc24f6fdd625c0c`). Its focused Gamma mode-search test passed 41/41 assertions on Totoro in 53 seconds. The full suite and the public-table derivative probe have not yet been rerun on this revision.

## Next checks

Run the complete Julia suite and the public-table derivative probe on this exact source. Keep the prior full-suite result as a recorded failure; it is not replaced by the focused pass.
