# animal × dep — Gaussian matrix fitter (K_phy = p full-rank Λ_phy on Σ_animal).
#
# Row-relatedness model: the rows of Y are the individuals indexed by A and the
# animal block is (Λ_phy Λ_phy') .* Σ_animal. This is NOT gllvmTMB's animal_dep,
# which is Σ_trait ⊗ A; the Julia model with that covariance is
# fit_kernel_dep_gllvm(Y, A, groups) (SourceCovariance :dep). Same path as
# phylo_dep (#817). Reuses fit_gaussian_gllvm J3 block. No @formula
# animal_dep() sugar (formula v1 rejects FunctionTerm).

"""
    fit_animal_dep_gllvm(Y, A; family = Normal(), Σ_animal = nothing, jitter = 1e-6, kwargs...)

Standalone **row-relatedness** Gaussian fit: the `p` rows of `Y` are the
individuals indexed by `A`, and the relatedness block of the row covariance is the
Hadamard product `(Λ_phy Λ_phy') .* A`, with full-rank loadings (`K_phy = p`,
packed lower-triangular Λ_phy with `p(p + 1)/2` parameters), a fixed p × p
relatedness / GRM matrix passed as `Σ_phy` internally, and a minimal unit-tier
factor (`K = 1`).

This is **not** the estimand of gllvmTMB's `animal_dep(0 + trait | id, A = A)`.
R's `animal_dep` puts traits and individuals on separate axes, with covariance
`Σ_trait ⊗ A` (an unstructured trait covariance Kronecker the relatedness `A`).
The Julia model with that covariance is
[`fit_kernel_dep_gllvm`](@ref)`(Y, A, groups)`, with `Y` traits × units and
`groups` mapping each unit to its row of `A`.

This is a **Gaussian matrix** fitter. `@formula` `animal_dep()` syntax is not
currently available.

`A` must be a square p × p (precomputed) numerator-relationship or genomic-
relationship matrix. `Y` is individuals × sites with `size(Y, 1) == size(A, 1)`.
When `Σ_animal` is omitted, [`relatedness_cov`](@ref)`(A; jitter = jitter)`
validates and symmetrizes `A` for the engine.

`K`, `num_lv`, `K_W`, `has_diag`, `K_phy`, and `has_phy_unique` are fail-loud —
they are not knobs on this path.

```julia
A = ...  # p × p NRM or GRM from kinship2 / nadiv / rrBLUP / etc.
Y = randn(size(A, 1), 30)
fit = fit_animal_dep_gllvm(Y, A)
```
"""
function fit_animal_dep_gllvm(Y::AbstractMatrix, A::AbstractMatrix;
                              family = Normal(),
                              Σ_animal = nothing,
                              jitter::Real = 1e-6,
                              kwargs...)
    p = size(Y, 1)
    size(A, 1) == p && size(A, 2) == p ||
        throw(ArgumentError(
            "fit_animal_dep_gllvm: A must be p × p; got $(size(A)) for p = $p"))
    family isa Normal || throw(ArgumentError(
        "fit_animal_dep_gllvm: this slice is a Gaussian wrapper only; got " *
        "$(nameof(typeof(family))). No formula animal_dep() sugar."))
    for sym in (:K, :num_lv, :K_W, :has_diag, :K_phy, :has_phy_unique)
        if haskey(kwargs, sym)
            throw(ArgumentError(
                "fit_animal_dep_gllvm: `$sym` is not a knob on animal × dep " *
                "(animal × dep forces K = 1, K_phy = p = $p)."))
        end
    end
    Σ_phy = if Σ_animal === nothing
        relatedness_cov(A; jitter = jitter)
    else
        size(Σ_animal, 1) == p && size(Σ_animal, 2) == p ||
            throw(ArgumentError(
                "Σ_animal must be p × p; got $(size(Σ_animal)) for p = $p"))
        Σ_animal
    end
    return fit_gaussian_gllvm(Y; K = 1, K_phy = p, Σ_phy = Σ_phy, kwargs...)
end

function fit_animal_dep_gllvm(Y::AbstractMatrix, phy::AugmentedPhy; kwargs...)
    throw(ArgumentError(
        "fit_animal_dep_gllvm: AugmentedPhy is phylogenetic input; " *
        "use fit_phylo_dep_gllvm(Y, phy) or pass a relatedness matrix A."))
end
