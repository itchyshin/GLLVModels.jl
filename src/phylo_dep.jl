# phylo × dep — Gaussian matrix fitter (K_phy = p full-rank Λ_phy on Σ_phy).
#
# Row-phylogeny model: the rows of Y are the tree's tips and the phylogenetic
# block is (Λ_phy Λ_phy') .* Σ_phy. This is NOT gllvmTMB's phylo_dep, which is
# Σ_trait ⊗ A; the twin of that is fit_phylo_latent_gllvm(...; d = n_traits).
# Reuses the fit_gaussian_gllvm J3 phylo_latent block. No @formula phylo_dep()
# sugar (formula v1 rejects FunctionTerm).

"""
    fit_phylo_dep_gllvm(Y, phy; family = Normal(), Σ_phy = nothing, σ²_phy = 1.0, kwargs...)

Standalone **row-phylogeny** Gaussian fit: the `p` rows of `Y` are the tips of
the tree, and the phylogenetic block of the row covariance is the Hadamard
product `(Λ_phy Λ_phy') .* Σ_phy`, with full-rank loadings (`K_phy = p`, packed
lower-triangular Λ_phy with `p(p + 1)/2` parameters), a fixed `p × p` tip
covariance `Σ_phy`, and a minimal unit-tier factor (`K = 1`). The marginal
covariance of `vec(Y)` is `I_n ⊗ A_unit + J_n ⊗ B`, with `A_unit` the
unit-tier covariance and `B` that Hadamard block (see
`GLLVModels.gaussian_marginal_loglik`).

This is **not** the estimand of gllvmTMB's `phylo_dep(0 + trait | species)`.
R's `phylo_dep` puts traits and species on separate axes, with covariance
`Σ_trait ⊗ A` (an unstructured trait covariance Kronecker the phylogenetic
correlation `A`). The Julia twin of R's `phylo_dep` is
[`fit_phylo_latent_gllvm`](@ref)`(Y, species; d = n_traits, ...)`, with `Y`
traits × observations.

This is a **Gaussian matrix** fitter. `@formula` `phylo_dep()` syntax is not
currently available.

`phy` must be an [`AugmentedPhy`](@ref). `Y` is tips × sites with
`size(Y, 1) == phy.n_leaves`. When `Σ_phy` is omitted, a dense leaf covariance
is built via [`sigma_phy_dense`](@ref) (small trees / tests only).

`K`, `num_lv`, `K_W`, `has_diag`, `K_phy`, and `has_phy_unique` are fail-loud —
they are not knobs on this path.

```julia
phy = augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
Y = randn(phy.n_leaves, 30)
fit = fit_phylo_dep_gllvm(Y, phy)
```
"""
function fit_phylo_dep_gllvm(Y::AbstractMatrix, phy::AugmentedPhy;
                             family = Normal(),
                             Σ_phy = nothing,
                             σ²_phy::Real = 1.0,
                             kwargs...)
    p = size(Y, 1)
    p == phy.n_leaves ||
        throw(ArgumentError(
            "fit_phylo_dep_gllvm: size(Y, 1) ($p) must equal phy.n_leaves " *
            "($(phy.n_leaves))"))
    family isa Normal || throw(ArgumentError(
        "fit_phylo_dep_gllvm: this slice is a Gaussian wrapper only; got " *
        "$(nameof(typeof(family))). No formula phylo_dep() sugar."))
    for sym in (:K, :num_lv, :K_W, :has_diag, :K_phy, :has_phy_unique)
        if haskey(kwargs, sym)
            throw(ArgumentError(
                "fit_phylo_dep_gllvm: `$sym` is not a knob on phylo × dep " *
                "(phylo × dep forces K = 1, K_phy = p = $p)."))
        end
    end
    if Σ_phy === nothing
        Σ_phy = sigma_phy_dense(phy; σ²_phy = σ²_phy)
    else
        size(Σ_phy, 1) == p && size(Σ_phy, 2) == p ||
            throw(ArgumentError(
                "Σ_phy must be p × p; got $(size(Σ_phy)) for p = $p"))
    end
    return fit_gaussian_gllvm(Y; K = 1, K_phy = p, Σ_phy = Σ_phy, kwargs...)
end

function fit_phylo_dep_gllvm(Y::AbstractMatrix, phy::PrecisionPhy; kwargs...)
    throw(ArgumentError(
        "fit_phylo_dep_gllvm: PrecisionPhy transport is not available; " *
        "pass AugmentedPhy or supply Σ_phy with a dense AugmentedPhy wrapper."))
end
