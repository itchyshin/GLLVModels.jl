# Student-t parity boundary

Student-t fits use a location-scale t likelihood with an identity link. A
numeric `nu` fixes the degrees of freedom. The public default `nu = nothing`
estimates degrees of freedom, and `disp_group = :species` estimates one scale
and one degree-of-freedom parameter for each trait.

Fixed `nu` must be finite and positive, either a number or one value per trait.
`Inf`, `NaN`, and nonpositive values are rejected by the fitter. Julia also
admits fixed `0 < nu <= 1`; this is an extension beyond the frozen R 0.7.0
constructor, which requires `df > 1`. Estimated degrees of freedom remain
greater than one. The Gaussian limiting statement does not make `nu = Inf`
a valid Student-t input.

The two calls answer different questions:

```julia
# Fixed degrees of freedom control
fit_studentt_gllvm(Y; K = 1, nu = 4.0, disp_group = :species)

# Public estimated-degrees-of-freedom route
fit_studentt_gllvm(Y; K = 1, nu = nothing, disp_group = :species)
```

The fixed call is useful for checking that both engines evaluate the same
scale-grouped model. It does not replace evidence for the estimated-ν route.
When a trait approaches the Gaussian limit, the likelihood can be very flat in
ν: two healthy fits can have very different large ν values while having nearly
the same log likelihood. Parity therefore compares fit health and the common
log likelihood; it does not force ν estimates to be equal.

For the recorded seed-71 test case, the fixed-ν control agrees to machine
precision and the estimated-ν log likelihood is within `0.001`. The frozen R
fit currently reports `nlminb` code 1 (`false convergence (8)`) for that
estimated-ν model. This comparison remains limited until both engines converge
cleanly.

For large degrees of freedom, the Float64 density now evaluates its normalizing
constant without subtracting two large log-gamma values. The calculation also
preserves automatic first and second derivatives near the Gaussian limit.
This numerical repair does not cap `nu`, change the model, or turn a failed
optimizer diagnostic into a successful fit.

The frozen oracle also has a numerical limit. Same-parameter checks with its
installed TMB 1.9.21 show loss of precision in the Student-t density at very
large degrees of freedom. On the recorded test case, inner modes and curvature
agree closely, but the density discrepancy changes the reported likelihood.
Thus a small difference between two separately optimized log likelihoods is
not sufficient evidence of parity. The original comparison and fit-health
checks remain required; no df cap or weakened tolerance has been substituted.

The numerical record retains the exact oracle provenance alongside the source
used for the comparison.

A public fixed-to-free warm start on this same test case improves likelihood
agreement but still fails the raw-gradient and same-parameter density checks.
The final free-df fit remains unqualified even though its optimizer returns
code0. Fixed initializer parameters have not replaced the original free model.
