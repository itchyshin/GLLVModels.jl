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

In a small reproducible comparison, the fixed-ν model agrees to machine
precision and the estimated-ν log likelihood differs by less than `0.001`.
For the estimated-ν model, the reference R fit reports false convergence.
Treat that result as a limitation of the comparison: it does not yet show that
both engines converge cleanly for estimated degrees of freedom.

For large degrees of freedom, the Float64 density now evaluates its normalizing
constant without subtracting two large log-gamma values. The calculation also
preserves automatic first and second derivatives near the Gaussian limit.
This numerical repair does not cap `nu`, change the model, or turn a failed
optimizer diagnostic into a successful fit.

The reference R calculation also has a numerical limit. With TMB 1.9.21,
same-parameter calculations lose precision in the Student-t density at very
large degrees of freedom. In the comparison above, the inner modes and
curvature agree closely, but this density difference changes the reported
likelihood. A small difference between two separately optimized likelihoods is
therefore not enough evidence of parity. We have not added an upper limit for
`nu` or relaxed the comparison tolerance to hide this limitation.

Starting the estimated-ν fit from a fixed-ν fit can improve likelihood
agreement, but the resulting fit still has an unsatisfactory gradient and a
same-parameter density mismatch. An optimizer's convergence message alone
does not validate this estimated-ν model.
