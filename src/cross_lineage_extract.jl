# extract_Gamma and extract_coevolution_modules for a fitted kernel tier of a
# GaussianSourcesFit (true-parity Wave 1, slice W1-4).
#
# Julia counterparts of gllvmTMB::extract_Gamma and
# gllvmTMB::extract_coevolution_modules (R/extract-sigma.R at P1,
# 9539352f66f2db2cc26b1c393e67212a359b60c9). Both R functions start from
# extract_Sigma(fit, level, part = "shared", link_residual = "none"), the
# trait-by-trait shared covariance Lambda Lambda' of the named tier, and slice
# it by trait. Here the same matrix is the fitted trait covariance of the named
# source of a `fit_gaussian_sources` / `fit_kernel_latent_gllvm` fit, whose rows
# and columns are the traits (the rows of Y). This is R's orientation: rows of
# Gamma are row_traits, columns are col_traits, in the order given.
#
# The older `extract_Gamma(fit::GllvmFit; ...)` method (src/extract_gamma.jl) is
# a different estimand: it slices the phylo loadings of the Hadamard fit, whose
# rows are the stacked species, not the traits. It is left unchanged.

# The shared (Lambda Lambda') trait covariance of the source named `level`.
function _sources_shared_covariance(fit::GaussianSourcesFit, level)
    (level isa Symbol || level isa AbstractString) && !isempty(string(level)) ||
        throw(ArgumentError("`level` must be one non-empty source name (Symbol or String)."))
    name = Symbol(level)
    k = findfirst(s -> s.name === name, fit.sources)
    k === nothing && throw(ArgumentError(
        "no source named `$(name)` in the fit; available: " *
        join(string.(getfield.(fit.sources, :name)), ", ") * "."))
    s = fit.sources[k]
    shared = (s.mode === :latent && !s.unique) || s.mode === :dep
    shared || throw(ArgumentError(
        "source `$(name)` has no separable shared covariance (mode = :$(s.mode)" *
        (s.unique ? ", unique = true" : "") * "); refit it as a latent tier with " *
        "unique = false (for example `fit_kernel_latent_gllvm`)."))
    return Matrix{Float64}(fit.trait_covariances[k])
end

# Resolve trait selectors (positional integers, or names looked up in `trait_names`)
# to row indices of the p x p shared covariance. Returns (indices, labels).
function _cross_lineage_traits(x, arg::AbstractString, p::Integer, trait_names)
    (x isa AbstractVector && !isempty(x)) ||
        throw(ArgumentError("`$arg` must be a non-empty vector of trait names or indices."))
    allunique(x) || throw(ArgumentError("`$arg` must not contain duplicate traits."))
    if all(t -> t isa Integer && !(t isa Bool), x)
        all(t -> 1 <= t <= p, x) ||
            throw(ArgumentError("`$arg` indices must lie in 1:$p (the traits, rows of Y)."))
        return collect(Int, x), collect(Int, x)
    end
    all(t -> t isa AbstractString || t isa Symbol, x) ||
        throw(ArgumentError("`$arg` must be all integers or all trait names."))
    trait_names === nothing && throw(ArgumentError(
        "`$arg` gives trait names; pass `trait_names` (the names of the rows of Y, in order)."))
    length(trait_names) == p ||
        throw(ArgumentError("`trait_names` must have length $p (one per row of Y)."))
    names = string.(trait_names)
    labels = string.(x)
    idx = [findfirst(==(t), names) for t in labels]
    missing_names = labels[idx .=== nothing]
    isempty(missing_names) || throw(ArgumentError(
        "`$arg` not found: $(join(missing_names, ", ")). Available traits: $(join(names, ", "))."))
    return Int.(idx), labels
end

function _cross_lineage_scale(scale)
    s = scale isa AbstractString ? Symbol(scale) : scale
    s === :shape && return nothing
    s === :effect && throw(ArgumentError(
        "scale = :effect needs the fixed cross-lineage rho recorded on the fitted tier; " *
        "a GaussianSourcesFit does not record rho (make_cross_kernel returns a plain matrix). " *
        "Use scale = :shape and multiply by your rho."))
    throw(ArgumentError("`scale` must be :shape (:effect is not available here)."))
end

"""
    extract_Gamma(fit::GaussianSourcesFit; level, row_traits, col_traits,
                  trait_names = nothing, scale = :shape) -> Matrix{Float64}

Cross-lineage block `Gamma_shape = Lambda_row Lambda_colᵀ` of a fitted kernel tier,
the counterpart of `gllvmTMB::extract_Gamma(fit, level, row_traits, col_traits)`.

`level` names the source (for example `:cross` in
`fit_kernel_latent_gllvm(Y, K, groups, d; name = :cross)`, R's `kernel_latent(...,
name = "cross")`). The source's shared trait covariance `Lambda Lambdaᵀ` (R's
`extract_Sigma(fit, level, part = "shared")`) is sliced to rows `row_traits` and
columns `col_traits`, in the order given; either set may contain any trait. Traits
are positional indices into the rows of `Y`, or names when `trait_names` (the names
of the rows of `Y`, in order) is given. Duplicates and unknown traits are errors.

The source must be a latent tier with `unique = false` (or the full-rank tier that
`fit_kernel_latent_gllvm` builds when `d` equals the number of traits). `scale` must be
`:shape`; R's `scale = "effect"` multiplies by the `rho` it records on the fitted kernel,
which a `GaussianSourcesFit` does not record, so `:effect` raises an `ArgumentError`.
Point estimates only: no intervals.

```julia
fit = fit_kernel_latent_gllvm(Y, K, groups, 3; name = :cross)
Gamma = extract_Gamma(fit; level = :cross, row_traits = ["h_size", "h_defence"],
                      col_traits = ["p_size", "p_attack"],
                      trait_names = ["h_size", "h_defence", "p_size", "p_attack"])
```
"""
function extract_Gamma(fit::GaussianSourcesFit; level, row_traits, col_traits,
                       trait_names = nothing, scale = :shape)
    _cross_lineage_scale(scale)
    Sigma = _sources_shared_covariance(fit, level)
    p = size(Sigma, 1)
    ri, _ = _cross_lineage_traits(row_traits, "row_traits", p, trait_names)
    ci, _ = _cross_lineage_traits(col_traits, "col_traits", p, trait_names)
    return Sigma[ri, ci]
end

# gllvmTMB's .matrix_inv_sqrt: symmetric generalised inverse square root with a
# tolerance relative to max(max |eigenvalue|, 1).
function _matrix_inv_sqrt_rel(x::AbstractMatrix, tol::Real, arg::AbstractString)
    xs = Symmetric((x .+ transpose(x)) ./ 2)
    eg = eigen(xs)
    sc = max(maximum(abs, eg.values), 1.0)
    minimum(eg.values) < -tol * sc && throw(ArgumentError(
        "cannot standardise the cross-lineage block: the shared covariance block for " *
        "`$arg` is not positive semidefinite."))
    keep = eg.values .> tol * sc
    any(keep) || throw(ArgumentError(
        "cannot standardise the cross-lineage block: the shared covariance block for " *
        "`$arg` is numerically zero."))
    vals = [k ? 1 / sqrt(max(v, 0.0)) : 0.0 for (v, k) in zip(eg.values, keep)]
    return eg.vectors * Diagonal(vals) * transpose(eg.vectors)
end

function _coevolution_axes(M, labels, modules, component, side)
    nt, nm = length(labels), length(modules)
    # `module` is a Julia keyword, so the NamedTuple is built from its field names.
    return NamedTuple{(:component, :side, :module, :trait, :loading)}((
        fill(component, nt * nm), fill(side, nt * nm), repeat(modules; inner = nt),
        repeat(labels, nm), vec(M[:, 1:nm])))
end

"""
    extract_coevolution_modules(fit::GaussianSourcesFit; level, row_traits, col_traits,
        trait_names = nothing, scale = :shape, n_modules = nothing,
        tol = sqrt(eps(Float64))) -> NamedTuple

Coupled trait axes of a fitted kernel tier, the counterpart of
`gllvmTMB::extract_coevolution_modules(fit, level, row_traits, col_traits)`.

From the source's shared trait covariance `Sigma` (as in [`extract_Gamma`](@ref)),
`R = Sigma_row^(-1/2) Gamma Sigma_col^(-1/2)` with `Sigma_row = Sigma[row_traits,
row_traits]`, `Sigma_col = Sigma[col_traits, col_traits]` and `Gamma = Sigma[row_traits,
col_traits]`, followed by the singular-value decomposition `R = U Diagonal(d) Vᵀ`. The
inverse square roots keep eigenvalues above `tol * max(max |eigenvalue|, 1)` (R's rule)
and throw on a block that is not positive semidefinite or is numerically zero.

Returns `(R, modules, row_axes, col_axes, level, scale, row_traits, col_traits)`, in R's
layout: `modules` has columns `component` (the level name), `module` (`"module_1"`, ...),
`singular_value` and `squared_share` (`d_k^2 / sum(d^2)` over all axes); `row_axes` and
`col_axes` are long tables `(component, side, module, trait, loading)` with the trait
varying fastest, from `U` and `V`. `n_modules` keeps only the first axes. The sign of
each pair of axes (`U[:, k]`, `V[:, k]`) is not identified, as in R. Traits are given as
in [`extract_Gamma`](@ref); `scale` must be `:shape`.
"""
function extract_coevolution_modules(fit::GaussianSourcesFit; level, row_traits, col_traits,
                                     trait_names = nothing, scale = :shape,
                                     n_modules = nothing, tol::Real = sqrt(eps(Float64)))
    _cross_lineage_scale(scale)
    if n_modules !== nothing
        (n_modules isa Integer && !(n_modules isa Bool) && n_modules >= 1) ||
            throw(ArgumentError("`n_modules` must be a positive integer."))
    end
    (isfinite(tol) && tol > 0) || throw(ArgumentError("`tol` must be one positive number."))
    Sigma = _sources_shared_covariance(fit, level)
    p = size(Sigma, 1)
    ri, rl = _cross_lineage_traits(row_traits, "row_traits", p, trait_names)
    ci, cl = _cross_lineage_traits(col_traits, "col_traits", p, trait_names)
    R = _matrix_inv_sqrt_rel(Sigma[ri, ri], tol, "row_traits") * Sigma[ri, ci] *
        _matrix_inv_sqrt_rel(Sigma[ci, ci], tol, "col_traits")
    F = svd(R)
    n_axis = n_modules === nothing ? length(F.S) : min(length(F.S), n_modules)
    component = string(level)
    modnames = ["module_$k" for k in 1:n_axis]
    sv = F.S[1:n_axis]
    sq = sum(abs2, F.S)
    share = sq > 0 ? sv .^ 2 ./ sq : fill(NaN, n_axis)
    modules = NamedTuple{(:component, :module, :singular_value, :squared_share)}((
        fill(component, n_axis), modnames, sv, share))
    return (R = Matrix(R), modules = modules,
            row_axes = _coevolution_axes(F.U, rl, modnames, component, "row"),
            col_axes = _coevolution_axes(F.V, cl, modnames, component, "column"),
            level = component, scale = :shape, row_traits = rl, col_traits = cl)
end
