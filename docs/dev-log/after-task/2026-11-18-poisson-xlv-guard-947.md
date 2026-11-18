# After-task: Poisson phylo-xlv objective guard (#947)

**Claim.** The objective `_fit_phylo_poisson_xlv` hands to Optim turns an
exception or a non-finite value into the `1e12` sentinel and rethrows
`InterruptException`, matching the NB2, binomial and Gamma xlv guards.

**Files.** `src/phylo_poisson_xlv.jl` (new `_phylo_poisson_xlv_nll_guarded`,
used by the fit and the profile refit). `test/test_phylo_xlv_verdict.jl`.

**Tests.** `test/test_phylo_xlv_verdict.jl` on Julia 1.10.13, 1 thread.
Before the guard: new testset 1 passed, 1 failed, 1 errored. After: the
file is 91 passed, 0 failed, 0 errored. Full `Pkg.test`, JET, Aqua and
R-parity were not run. No likelihood value on a finite evaluation changed:
the guard only catches what the packed objective already threw.

**Rose verdict:** PASS WITH NOTES — the Poisson objective now matches the
sibling guards; the full suite was not run.
