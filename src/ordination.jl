# Ordination output — the headline ecology-facing summary of a fitted GLLVM.
#
# An ordination places SITES (samples/rows) and SPECIES (responses/columns) in a
# shared low-dimensional latent space. The two coordinate sets are:
#   * sites   = conditional latent-variable scores (see `getLV`) — the ordination
#               point cloud, one row per site;
#   * species = the fitted loadings Λ (or Λc for the two-part families) — the
#               "arrows" / response directions, one row per species.
#
# Loadings/scores are identified only up to a K×K orthogonal rotation: the linear
# predictor contribution S·Lᵀ (and hence Σ_y) is invariant under (S,L) → (S·R, L·R)
# for any R with RᵀR = I, since (S·R)(L·R)ᵀ = S·R·Rᵀ·Lᵀ = S·Lᵀ. We therefore fix a
# canonical PRINCIPAL orientation: R = V from the SVD of the centered site scores,
# so the rotated site axes are ordered by decreasing spread (principal axes first).

# Loadings accessor is defined in src/postfit.jl

"""
    ordination(fit, Y; rotate=true, kwargs...) -> (sites, species, rotation)

Ordination of a fitted GLLVM: site and species coordinates in the shared `K`-
dimensional latent space, returned as a `NamedTuple` `(sites, species, rotation)`.

- `sites`    — `n×K` matrix of latent SITE scores (ordination point cloud), the
               conditional latent-variable scores `getLV(fit, Y; rotate=false)`.
- `species`  — `p×K` matrix of SPECIES loadings (the ordination "arrows" / response
               directions), the fitted `Λ` (or `Λc` for the two-part families).
- `rotation` — the `K×K` orthogonal matrix `R` applied to both sets.

With `rotate=true` (default) a canonical PRINCIPAL rotation is applied: `R = V`, the
right singular vectors of the centered site scores `Sc = S .- mean(S; dims=1)`, so
the returned `sites = S*R` are ordered by decreasing spread (principal axes first)
and `species = L*R`. With `rotate=false`, `R = I` and the raw `sites`/`species` are
returned unrotated.

The rotation is orthogonal, so the fit is preserved exactly: the linear-predictor
contribution `sites * species' == S * L'` for any `R` with `R'R = I`.

`Y` (the `p×n` response matrix) must match what was passed to the fitting call — the
fit does not store the data. Any other keyword (`X`, `N`, `X_lv`, `mask`, ...) is
forwarded to `getLV`, so pass the same design the fit used: a `GllvmFit` with fixed
effects needs `X = X`, a binomial fit with trials needs `N = N`. A `GllvmFit` that
estimated `β` but does not retain its design throws when `X` is omitted, rather
than returning scores computed at a zero mean.
"""
function ordination(fit, Y; rotate::Bool = true, kwargs...)
    S = getLV(fit, Y; rotate = false, kwargs...)      # n×K site scores
    L = _loadings(fit)                      # p×K species loadings
    K = size(L, 2)
    if !rotate
        return (sites = S, species = L, rotation = Matrix{Float64}(I, K, K))
    end
    Sc = S .- mean(S; dims = 1)
    R = Matrix(svd(Sc).V)                   # K×K right singular vectors (principal axes)
    return (sites = S * R, species = L * R, rotation = R)
end

"""
    ordiplot(fit, Y; rotate=true, biplot=true, site_labels=nothing,
             species_labels=nothing, kwargs...)

Tidy, plot-ready ordination DATA for a fitted GLLVM, mirroring R `gllvm`'s
`ordiplot` / `getLV` interface but as a pure data layer (no plotting backend, no
plotting dependency): you feed the returned coordinates into whichever backend you
like. Returns a `NamedTuple` with fields:

- `sites`          — `n×K` matrix of latent SITE scores (the ordination point
                     coordinates), identical to `ordination(fit, Y; rotate).sites`.
- `species`        — `p×K` matrix of SPECIES loadings (the biplot "arrows" /
                     response directions) when `biplot=true`; an empty `0×K`
                     `Matrix{Float64}` when `biplot=false`.
- `site_labels`    — `Vector{String}` of point labels (the supplied
                     `site_labels`, or the defaults `"site 1"`, `"site 2"`, …).
- `species_labels` — `Vector{String}` of arrow labels (the supplied
                     `species_labels`, or the defaults `"sp 1"`, `"sp 2"`, …);
                     empty when `biplot=false`.
- `axis_prop`      — `Vector{Float64}` of length `K`: the proportion of latent
                     variance carried by each ordination axis, from the SVD of the
                     centred site scores (`s = svd(sites .- mean).S`;
                     `axis_prop = s.^2 ./ sum(s.^2)`). These are the "% variance per
                     axis" figures conventionally printed on ordination-axis labels.

`Y` (the `p×n` response matrix) must match what was passed to the fitting call — the
fit does not store the data. With `rotate=true` (default) the canonical PRINCIPAL
rotation is applied (see [`ordination`](@ref)); other keywords (`X`, `N`, ...) are
forwarded to `ordination`.

You plot with any backend, e.g.

```julia
using Plots
o = ordiplot(fit, Y)
scatter(o.sites[:, 1], o.sites[:, 2];                       # site point cloud
        xlabel = "LV1 (\$(round(100o.axis_prop[1]; digits=1))%)",
        ylabel = "LV2 (\$(round(100o.axis_prop[2]; digits=1))%)")
# overlay species loadings as biplot arrows from the origin:
for t in 1:size(o.species, 1)
    plot!([0, o.species[t, 1]], [0, o.species[t, 2]])
end
```
"""
function ordiplot(fit, Y; rotate::Bool = true, biplot::Bool = true,
                  site_labels = nothing, species_labels = nothing, kwargs...)
    ord = ordination(fit, Y; rotate = rotate, kwargs...)
    S = ord.sites                                   # n×K
    n = size(S, 1)
    K = size(S, 2)
    p = size(ord.species, 1)

    species = biplot ? ord.species : Matrix{Float64}(undef, 0, K)

    slabels = site_labels === nothing ?
        ["site $i" for i in 1:n] : collect(String, site_labels)
    splabels = if !biplot
        String[]
    elseif species_labels === nothing
        ["sp $t" for t in 1:p]
    else
        collect(String, species_labels)
    end

    # Proportion of latent variance per ordination axis, from the SVD of the
    # centred site scores (the "% variance per axis" shown on ordination plots).
    Sc = S .- mean(S; dims = 1)
    s = svd(Sc).S
    tot = sum(abs2, s)
    axis_prop = tot > 0 ? (s .^ 2) ./ tot : fill(1.0 / K, K)

    return (sites = S, species = species,
            site_labels = slabels, species_labels = splabels,
            axis_prop = axis_prop)
end
