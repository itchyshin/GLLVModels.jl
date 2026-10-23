# kernel × dep — Gaussian matrix fitter (full-rank trait loadings on fixed dense K).
#
# Twin estimand (Identity 2026-09-14; pin gllvmTMB kernel_dep roxygen b4d5fee6):
# unstructured cross-trait covariance on a user-supplied between-unit matrix K,
# a true Kronecker model Σ_trait ⊗ K (source_fit.jl: kron(S, B)). It matches R's
# phylo_dep(..., vcv = K) / animal_dep, NOT the Julia fit_phylo_dep_gllvm /
# fit_animal_dep_gllvm, which are Hadamard row models (#817, #823). Reuses fit_gaussian_sources / SourceCovariance mode :dep — not the J3 Σ_phy block.
# No @formula kernel_dep() sugar (formula recognizer exists separately).

"""
    fit_kernel_dep_gllvm(Y, K, groups; name = :kernel, kwargs...)

Standalone **kernel × dep** Gaussian fit: full-rank trait loadings (`mode = :dep`,
rank = p) on a fixed dense between-unit covariance matrix `K`, with units projected
via one-based `groups` indices into the rows/columns of `K`.

This is a true **Kronecker** model: the covariance of `vec(Y)` carries
`Σ_trait ⊗ K` (an unstructured trait covariance Kronecker the projected `K`). It is
the same estimand class as twin `kernel_dep(unit, K = K, name = "…")` and as R's
`phylo_dep(0 + trait | unit, vcv = K)` / `animal_dep(0 + trait | id, A = A)` on the
dense supplied-kernel route (documentary keyword only on the R side). It is **not**
equivalent to [`fit_phylo_dep_gllvm`](@ref) or [`fit_animal_dep_gllvm`](@ref), which
are Hadamard row models `(Λ_phy Λ_phy') .* Σ`.

This is a **Gaussian matrix** fitter via [`fit_gaussian_sources`](@ref). It does
**not** estimate `K`, apply `rho` attenuation, or parse long-format grouping
columns. `@formula` `kernel_dep()` syntax is not currently available.

`Y` is traits × units (`size(Y, 2) == length(groups)`). `K` must be square PD;
`groups[i]` maps unit `i` to a row of `K` (integer in `1:size(K, 1)`).

Engine knobs `K`, `num_lv`, `K_phy`, `Σ_phy`, and `sources` are fail-loud. Non-
default twin `rho` is not available. `common` is not a knob on kernel × dep.

```julia
p, n = 3, 12
Y = randn(p, n)
groups = repeat(1:4; inner = 3)
L = randn(4, 4); K = L * L' + 0.5I
fit = fit_kernel_dep_gllvm(Y, K, groups; name = :known)
```
"""
function fit_kernel_dep_gllvm(Y::AbstractMatrix, K::AbstractMatrix, groups;
                              name::Union{Symbol, AbstractString} = :kernel,
                              rho = 1,
                              kwargs...)
    p, n = size(Y)
    m = size(K, 1)
    size(K, 1) == size(K, 2) ||
        throw(ArgumentError(
            "fit_kernel_dep_gllvm: K must be square; got $(size(K))"))
    length(groups) == n ||
        throw(ArgumentError(
            "fit_kernel_dep_gllvm: length(groups) must equal n = $n; got $(length(groups))"))
    ids = collect(groups)
    all(g -> g isa Integer && !(g isa Bool) && 1 <= g <= m, ids) ||
        throw(ArgumentError(
            "fit_kernel_dep_gllvm: groups must be one-based indices in 1:$m"))
    sym_name = name isa Symbol ? name : Symbol(name)
    if !(rho === 1 || rho == 1)
        throw(ArgumentError(
            "fit_kernel_dep_gllvm: rho attenuation is not available; omit rho or pass 1."))
    end
    for sym in (:K, :num_lv, :K_W, :has_diag, :K_phy, :has_phy_unique, :Σ_phy, :sources,
                :family, :common)
        if haskey(kwargs, sym)
            throw(ArgumentError(
                "fit_kernel_dep_gllvm: `$sym` is not a knob on kernel × dep."))
        end
    end
    source = SourceCovariance(K; groups = ids, name = sym_name, mode = :dep)
    return fit_gaussian_sources(Y; sources = [source], kwargs...)
end

function fit_kernel_dep_gllvm(Y::AbstractMatrix, K::AbstractMatrix; kwargs...)
    throw(ArgumentError(
        "fit_kernel_dep_gllvm: pass unit-to-source `groups` (length n = $(size(Y, 2))) " *
        "as the third argument; kernel × dep does not infer grouping from Y alone."))
end
