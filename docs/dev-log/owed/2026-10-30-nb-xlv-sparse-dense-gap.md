# GitHub issue (not filed)

Paste the block below as a new issue. Do not use closing keywords.
Do not try to fix the NB path in the #948 wiring PR.

**Title:** NB phylo xlv sparse-augmented loglik disagrees with the dense-leaf oracle by 1.3e-2

**Labels:** `bug`

---

## Where

- Sparse-augmented evaluator: `src/phylo_nb_xlv.jl` (`_phylo_nb_xlv_marginal_loglik`)
- Dense-leaf oracle in the test: `test/test_phylo_nb_xlv.jl` (`_dense_leaf_phylo_nb_xlv_loglik`, lines 4-82)
- Broken identity: `test/test_phylo_nb_xlv.jl` (S1 likelihood testset, `@test_broken` at atol = 1e-6)

The binomial sibling of this identity already passes at `atol = 1e-6` in `test/test_phylo_binomial_xlv.jl`.

## What is wrong

Both log-likelihoods are evaluated at the same parameters. This is a deterministic identity, not an optimiser or BLAS cell. On Julia 1.10.12 the gap is about 1.3e-2:

- `ll_sparse = -112.424`
- `ll_dense = -112.412`

That is about 10^4 times `atol = 1e-6`. Either the sparse-augmented NB path or the dense-leaf oracle is wrong. The 1e-6 contract must not be widened.

The check used to sit behind `@test_skip` and `GLLVM_XLV_CANARY=1`, so CI never tracked the gap. It is now `@test_broken` so CI flags it once the identity holds.

## Minimal repro

Julia 1.10.12. Seed and tree are the ones in `test/test_phylo_nb_xlv.jl`.

```julia
using GLLVModels, Test, Random, LinearAlgebra
using Distributions: NegativeBinomial
include("test/test_phylo_nb_xlv.jl")
```

Or, inside the S1 testset after `ll_sparse_aug` and `ll_dense_leaf` are computed:

```julia
println(ll_sparse_aug, " ", ll_dense_leaf, " ",
        abs(ll_sparse_aug - ll_dense_leaf))
@assert isapprox(ll_sparse_aug, ll_dense_leaf; atol = 1e-6)
```

Observed on this snapshot: absolute gap about 1.3e-2 (values above).

## What a user observes

The NB phylo xlv sparse path and the test's dense-leaf Laplace disagree at a fixed theta. The binomial xlv counterpart agrees. Wiring #948 did not change the NB evaluator.

## What the code would need to guarantee

`_phylo_nb_xlv_marginal_loglik` and `_dense_leaf_phylo_nb_xlv_loglik` agree to `atol = 1e-6` at the same `(Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)`. Then flip `@test_broken` to `@test`.

Do not widen atol. Do not fold the fix into the #948 wiring PR.

## Files

- `src/phylo_nb_xlv.jl`
- `test/test_phylo_nb_xlv.jl`
- sibling that already holds: `test/test_phylo_binomial_xlv.jl`

## Severity

Medium. Likelihood identity at fixed parameters; reachable on the six-leaf tree in the test file.

Refs #948
Refs #1005
