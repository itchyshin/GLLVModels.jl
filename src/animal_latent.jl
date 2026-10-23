# animal × latent — Gaussian matrix fitter (rank-d Λ_phy on Σ_animal).
#
# Row-relatedness model: the rows of Y are the individuals indexed by A and the
# animal block is (Λ_phy Λ_phy') .* Σ_animal. This is NOT gllvmTMB's
# animal_latent, which is Λ Λ' ⊗ A; the Julia model with that covariance is
# fit_kernel_latent_gllvm(Y, A, groups, d). Same path as animal_dep (#823).
# Reuses fit_gaussian_gllvm J3 block. No @formula animal_latent() sugar
# (formula v1 rejects FunctionTerm).

"""
    fit_animal_latent_gllvm(Y, A, d; family = Normal(), unique = false,
                            Σ_animal = nothing, jitter = 1e-6, kwargs...)

Standalone **row-relatedness** Gaussian fit: the `p` rows of `Y` are the
individuals indexed by `A`, and the relatedness block of the row covariance is the
Hadamard product `(Λ_phy Λ_phy') .* A` with rank-`d` loadings (`K_phy = d`, packed
Λ_phy), a fixed p × p GRM / NRM passed as engine `Σ_phy`, and a minimal unit-tier
factor (`K = 1`). When `d == p` this is [`fit_animal_dep_gllvm`](@ref).

This is **not** the estimand of gllvmTMB's `animal_latent(0 + trait | id, d = d,
A = A)`. R's `animal_latent` puts traits and individuals on separate axes, with
covariance `Λ Λ' ⊗ A` (a rank-`d` trait covariance Kronecker the relatedness
`A`), not `Σ_trait ⊗ A` with an unstructured trait covariance. The Julia model with
the Kronecker form is [`fit_kernel_latent_gllvm`](@ref)`(Y, A, groups, d)`, with `Y`
traits × units and `groups` mapping each unit to its row of `A`.

`unique = false` (default) is loadings-only on the animal block (`has_phy_unique =
false`). `unique = true` (twin `animal_latent(..., unique = TRUE)`) is not yet
available and raises an error.

This is a **Gaussian matrix** fitter. `@formula` `animal_latent()` syntax is not
currently available.

`A` must be square p × p. `Y` is individuals × sites with `size(Y, 1) == size(A, 1)`.
`d` must satisfy `1 ≤ d ≤ p`.

`K`, `num_lv`, `K_W`, `has_diag`, `K_phy`, and `has_phy_unique` are fail-loud —
use positional `d` instead of `K_phy`.

```julia
A = ...  # p × p NRM or GRM
Y = randn(size(A, 1), 30)
fit = fit_animal_latent_gllvm(Y, A, 1)
```
"""
function fit_animal_latent_gllvm(Y::AbstractMatrix, A::AbstractMatrix, d::Integer;
                                 family = Normal(),
                                 unique::Bool = false,
                                 Σ_animal = nothing,
                                 jitter::Real = 1e-6,
                                 kwargs...)
    p = size(Y, 1)
    size(A, 1) == p && size(A, 2) == p ||
        throw(ArgumentError(
            "fit_animal_latent_gllvm: A must be p × p; got $(size(A)) for p = $p"))
    (1 <= d <= p) || throw(ArgumentError(
        "fit_animal_latent_gllvm: d must satisfy 1 ≤ d ≤ p = $p; got d = $d"))
    family isa Normal || throw(ArgumentError(
        "fit_animal_latent_gllvm: this slice is a Gaussian wrapper only; got " *
        "$(nameof(typeof(family))). No formula animal_latent() sugar."))
    unique && throw(ArgumentError(
        "fit_animal_latent_gllvm: unique = true (animal × latent with diag(psi) " *
        "on the animal tier) is not available; use unique = false."))
    for sym in (:K, :num_lv, :K_W, :has_diag, :K_phy, :has_phy_unique)
        if haskey(kwargs, sym)
            throw(ArgumentError(
                "fit_animal_latent_gllvm: `$sym` is not a knob; animal × latent " *
                "forces K = 1 and K_phy = d = $d (pass rank as the third argument)."))
        end
    end
    if d == p
        return fit_animal_dep_gllvm(Y, A;
            family = family, Σ_animal = Σ_animal, jitter = jitter, kwargs...)
    end
    Σ_phy = if Σ_animal === nothing
        relatedness_cov(A; jitter = jitter)
    else
        size(Σ_animal, 1) == p && size(Σ_animal, 2) == p ||
            throw(ArgumentError(
                "Σ_animal must be p × p; got $(size(Σ_animal)) for p = $p"))
        Σ_animal
    end
    return fit_gaussian_gllvm(Y; K = 1, K_phy = d, Σ_phy = Σ_phy, kwargs...)
end

function fit_animal_latent_gllvm(Y::AbstractMatrix, phy::AugmentedPhy, d::Integer; kwargs...)
    throw(ArgumentError(
        "fit_animal_latent_gllvm: AugmentedPhy is phylogenetic input; " *
        "pass a relatedness matrix A or use a phylogenetic fitter on AugmentedPhy."))
end
