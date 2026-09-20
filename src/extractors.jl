# Post-fit extractor family mirroring R gllvmTMB's extract_*() / get*()
# surface (core070 missing-surface work order, Cluster 1).
#
# These are almost all THIN wrappers: the underlying quantities are already
# computed elsewhere in the engine —
#   * src/postfit.jl        — getLoadings, getLV, rotation, _loadings
#   * src/ordination.jl     — ordination (sites, species, rotation)
#   * src/confint_derived.jl — sigma_y_site, communality, correlation,
#                              proportions, phylo_signal (GllvmFit)
#   * src/link_residual.jl  — sigma_y_site, communality, correlation
#                              (non-Gaussian families, latent-scale)
#   * src/twolevel.jl       — repeatability, communality_B/W,
#                              correlation_B/W (TwoLevelFit)
#   * src/extract_gamma.jl  — extract_Gamma (cross-lineage coevolution)
#
# This file supplies the R-name-mirroring public surface (`extract_*`,
# `get*`) that dispatches onto that existing machinery, plus a handful of
# genuinely new thin readouts (extract_Sigma, extract_Sigma_table,
# extract_cutpoints, extract_ICC_site, extract_Omega, extract_rotated_loadings,
# extract_cross_correlations, extract_residual_cov/cor + get* aliases).
#
# R oracle source (frozen 0.7.0 readback):
#   .unlazy/core070-aghq/oracle-source/readback/R/extract-sigma.R
#   .unlazy/core070-aghq/oracle-source/readback/R/extract-sigma-table.R
#   .unlazy/core070-aghq/oracle-source/readback/R/extract-correlations.R
#   .unlazy/core070-aghq/oracle-source/readback/R/extract-cutpoints.R
#   .unlazy/core070-aghq/oracle-source/readback/R/extract-omega.R
#   .unlazy/core070-aghq/oracle-source/readback/R/rotate-loadings.R
#   .unlazy/core070-aghq/oracle-source/readback/R/output-methods.R
#   .unlazy/core070-aghq/oracle-source/readback/R/extractors.R
#
# Loading-sign convention: GLLVModels.jl's `getLoadings(fit; rotate=true)` fixes
# signs via the largest-magnitude-entry convention (src/postfit.jl); rotated
# loadings are never compared or canonicalised against R's own sign choice.
# Quantities compared against a Gaussian closed form here are all
# rotation/sign-INVARIANT (Σ, communality, correlation, Ω, ICC).

using LinearAlgebra: diagm

const _GllvmOrTwoLevel = Union{GllvmFit, TwoLevelFit}

# ---------------------------------------------------------------------------
# Level normalisation. GLLVModels.jl uses :unit / :unit_obs for the two ordinary
# tiers (mirrors R's canonical spelling); the legacy :B / :W aliases are
# accepted for gllvm-familiarity, matching R's `.normalise_level()`.
# ---------------------------------------------------------------------------
function _canonical_level(level::Symbol)
    level === :unit || level === :B      ? :unit :
    level === :unit_obs || level === :W  ? :unit_obs :
    level === :site                      ? :site :
    throw(ArgumentError("level must be one of :unit, :unit_obs, :site " *
                         "(or legacy :B, :W); got $(level)"))
end

# ---------------------------------------------------------------------------
# extract_Sigma / extract_Sigma_table — unified covariance API.
# ---------------------------------------------------------------------------

# Between-unit (species/site "B") and within-unit ("W") covariance pieces
# for a Gaussian GllvmFit, mirroring the v_B,t / v_W,t split documented in
# extract-repeatability.R: v_B,t = (Λ_B Λ_B')_tt + σ²_B,t;
# v_W,t = (Λ_W Λ_W')_tt + σ²_W,t + σ²_eps.
function _sigma_unit(fit::GllvmFit)
    Λ = fit.pars.Λ
    Σ = Λ * Λ'
    if fit.model.has_diag && fit.pars.σ²_B !== nothing
        Σ = Σ + diagm(collect(Float64, fit.pars.σ²_B))
    end
    return Matrix(Σ)
end

function _sigma_unit_obs(fit::GllvmFit)
    p = fit.model.p
    Σ = zeros(Float64, p, p)
    if fit.model.K_W > 0 && fit.pars.Λ_W !== nothing
        Λ_W = fit.pars.Λ_W
        Σ = Σ + Λ_W * Λ_W'
    end
    if fit.model.has_diag && fit.pars.σ²_W !== nothing
        Σ = Σ + diagm(collect(Float64, fit.pars.σ²_W))
    end
    Σ = Σ + (fit.pars.σ_eps^2) * I
    return Matrix(Σ)
end

# cov2cor: standardise a covariance to a correlation (diagonal exactly 1;
# NaN on a non-positive diagonal entry rather than a divide-by-zero Inf).
function _cov2cor(Σ::AbstractMatrix)
    p = size(Σ, 1)
    R = Matrix{Float64}(undef, p, p)
    @inbounds for j in 1:p, i in 1:p
        d = Σ[i, i] * Σ[j, j]
        R[i, j] = (Σ[i, i] > 0 && Σ[j, j] > 0) ? Σ[i, j] / sqrt(d) : NaN
    end
    return R
end

"""
    extract_Sigma(fit::GllvmFit; level::Symbol = :unit, part::Symbol = :total)
        -> NamedTuple

Implied trait covariance at one tier of a fitted Gaussian GLLVM, mirroring
`gllvmTMB::extract_Sigma()`. `level`:

  - `:unit`     — between-unit (species/site, "B") tier,
                  `Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)`.
  - `:unit_obs` — within-unit (observation, "W") tier,
                  `Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W) + σ²_eps·I`.
  - `:site`     — the full per-site covariance `sigma_y_site(fit)` (a GLLVModels.jl
                  extension not present in the R tier vocabulary; combines
                  both tiers' diagonal contributions with the Gaussian
                  residual, excluding the phylogenetic block).

Legacy aliases `:B`/`:W` are accepted for `:unit`/`:unit_obs`.

`part`:
  - `:total`  (default) — the full `Σ = Λ Λᵀ + Ψ` sum. Returns `(Sigma, R,
              level, part)` where `R` is the corresponding correlation matrix.
  - `:shared` — the `Λ Λᵀ` term alone (no diagonal). Returns `(Sigma, level,
              part)` (no `R`, matching R's `part = "shared"` return, which
              omits the correlation entry).
  - `:unique` — the diagonal `Ψ` term alone, as a length-`p` vector `s`.
              Returns `(s, level, part)`.

Deviation from R: GLLVModels.jl has no `phy`/`spatial`/`*_slope` tiers yet (those
augmented-block tiers belong to structured-term recognizers, Cluster 3 of
the missing-surface work order); requesting them throws `ArgumentError`.
"""
function extract_Sigma(fit::GllvmFit; level::Symbol = :unit, part::Symbol = :total)
    lvl = _canonical_level(level)
    if lvl === :site
        part === :total || throw(ArgumentError(
            "level = :site only supports part = :total"))
        Σ = sigma_y_site(fit)
        return (Sigma = Σ, R = _cov2cor(Σ), level = lvl, part = part)
    end
    Σ_shared = lvl === :unit ? fit.pars.Λ * fit.pars.Λ' :
               (fit.model.K_W > 0 && fit.pars.Λ_W !== nothing ?
                    fit.pars.Λ_W * fit.pars.Λ_W' : zeros(Float64, fit.model.p, fit.model.p))
    if part === :shared
        return (Sigma = Matrix(Σ_shared), level = lvl, part = part)
    elseif part === :unique
        p = fit.model.p
        s = if lvl === :unit
            fit.model.has_diag && fit.pars.σ²_B !== nothing ?
                collect(Float64, fit.pars.σ²_B) : zeros(Float64, p)
        else
            s_W = fit.model.has_diag && fit.pars.σ²_W !== nothing ?
                collect(Float64, fit.pars.σ²_W) : zeros(Float64, p)
            s_W .+ fit.pars.σ_eps^2
        end
        return (s = s, level = lvl, part = part)
    elseif part === :total
        Σ = lvl === :unit ? _sigma_unit(fit) : _sigma_unit_obs(fit)
        return (Sigma = Σ, R = _cov2cor(Σ), level = lvl, part = part)
    else
        throw(ArgumentError("part must be one of :total, :shared, :unique; got $(part)"))
    end
end

"""
    extract_Sigma(fit::TwoLevelFit; level::Symbol = :unit, part::Symbol = :total)
        -> NamedTuple

`TwoLevelFit` twin of [`extract_Sigma(::GllvmFit)`](@ref): `:unit` reads
`fit.Σ_B`/`fit.Λ_B`/`fit.σ²_B`, `:unit_obs` reads `fit.Σ_W`/`fit.Λ_W`/`fit.σ²_W`.
"""
function extract_Sigma(fit::TwoLevelFit; level::Symbol = :unit, part::Symbol = :total)
    lvl = _canonical_level(level)
    lvl === :site && throw(ArgumentError("TwoLevelFit has no :site tier"))
    Λ, σ², Σ = lvl === :unit ? (fit.Λ_B, fit.σ²_B, fit.Σ_B) : (fit.Λ_W, fit.σ²_W, fit.Σ_W)
    if part === :shared
        return (Sigma = Matrix(Λ * Λ'), level = lvl, part = part)
    elseif part === :unique
        return (s = collect(Float64, σ²), level = lvl, part = part)
    elseif part === :total
        return (Sigma = Matrix(Σ), R = _cov2cor(Σ), level = lvl, part = part)
    else
        throw(ArgumentError("part must be one of :total, :shared, :unique; got $(part)"))
    end
end

"""
    extract_Sigma_table(fit; level::Symbol = :unit, part::Symbol = :total)
        -> Vector{<:NamedTuple}

Tidy long-format companion to [`extract_Sigma`](@ref): one row per
upper-triangular `(i, j)` pair (`i ≤ j`) with fields `(trait_i, trait_j,
value)`, mirroring `gllvmTMB::extract_Sigma_table()`'s tidy-data-frame
return but as a `Vector{NamedTuple}` rather than an R `data.frame`. `part`
must be `:total` or `:shared` (the two matrix-valued parts); `:unique` is a
vector, not a table — call [`extract_Sigma`](@ref) directly for it.
"""
function extract_Sigma_table(fit::_GllvmOrTwoLevel; level::Symbol = :unit, part::Symbol = :total)
    part === :unique && throw(ArgumentError(
        "extract_Sigma_table is for matrix-valued parts (:total, :shared); " *
        "for :unique call extract_Sigma(fit; level=level, part=:unique) directly"))
    out = extract_Sigma(fit; level = level, part = part)
    Σ = out.Sigma
    p = size(Σ, 1)
    rows = NamedTuple{(:trait_i, :trait_j, :value), Tuple{Int, Int, Float64}}[]
    for i in 1:p, j in i:p
        push!(rows, (trait_i = i, trait_j = j, value = Float64(Σ[i, j])))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Loadings.
# ---------------------------------------------------------------------------

"""
    extract_loadings(fit; rotate::Bool = true) -> p×K matrix

Canonical snake_case accessor for the fitted species/trait loadings.
Forwards to [`getLoadings`](@ref) (`gllvm`-style spelling), mirroring
`gllvmTMB::extract_loadings()`'s forwarding to `getLoadings()`.

Deviation from R: no `level` tier argument (GLLVModels.jl's `_loadings` reads
the single loadings matrix each fit type carries; there is no Julia-bridge
rotation gate to special-case).
"""
extract_loadings(fit; rotate::Bool = true) = getLoadings(fit; rotate = rotate)

"""
    extract_rotated_loadings(fit) -> (Λ = p×K matrix, R = K×K matrix)

Report-ready companion to [`extract_loadings`](@ref): the canonically
rotated loadings `Λ` together with the `K×K` orthogonal rotation `R` that
produced them (`Λ = Λ_raw · R`), mirroring
`gllvmTMB::extract_rotated_loadings_table()`. `R'R == I`.
"""
extract_rotated_loadings(fit) = (Λ = getLoadings(fit; rotate = true), R = rotation(fit))

# ---------------------------------------------------------------------------
# Communality / correlation — forward to the existing per-family generics.
# ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## R-tier-scoped composition helpers (maintainer decision, round 1, item 3:
## docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md —
## "align to R's tier-scoped decomposition as the DEFAULT, mirroring R's
## level= arguments; keep the total-variance variant behind an explicit
## option"). Traced against R source
## (.unlazy/core070-aghq/oracle-source/readback/R/{extract-sigma,extract-
## omega}.R) and the wave5 R-oracle repair notes
## (docs/dev-log/core070/surface-conversion-notes.md, Repairs 3-5):
## gllvmTMB's B/W/phy tier system NEVER folds `sigma_eps²` (the Gaussian
## observation residual) into any tier total — `link_residual_per_trait()`
## returns exactly 0 for the gaussian family (fid 0), and σ_eps is not one
## of the B/W/phy component tiers R's `extract_Sigma()`/`extract_Omega()`
## track at all. These helpers reproduce that convention exactly. They are
## intentionally distinct from `_sigma_unit_obs` above (`extract_Sigma`'s own
## `:unit_obs` total, which folds `σ_eps²` in unconditionally as GLLVModels.jl's
## own useful extension per its docstring, and is unchanged by this decision
## — `extract_Sigma`'s public contract was never part of the estimand-
## alignment ledger row).
## ---------------------------------------------------------------------------

# Tier total with NO σ_eps² folded in, mirroring R's B/W tier Sigma
# (`extract_Sigma(fit, level, part="total", link_residual="none")` on a
# Gaussian fit — Gaussian's link_residual contributes 0 regardless).
function _r_tier_total(fit::GllvmFit, lvl::Symbol)
    p = fit.model.p
    if lvl === :unit
        Σ = fit.pars.Λ * fit.pars.Λ'
        if fit.model.has_diag && fit.pars.σ²_B !== nothing
            Σ = Σ + diagm(collect(Float64, fit.pars.σ²_B))
        end
        return Matrix(Σ)
    else # :unit_obs
        Σ = zeros(Float64, p, p)
        if fit.model.K_W > 0 && fit.pars.Λ_W !== nothing
            Σ = Σ + fit.pars.Λ_W * fit.pars.Λ_W'
        end
        if fit.model.has_diag && fit.pars.σ²_W !== nothing
            Σ = Σ + diagm(collect(Float64, fit.pars.σ²_W))
        end
        return Matrix(Σ)
    end
end

# Tier shared (ΛΛᵀ) block alone — no diagonal, no σ_eps².
function _r_tier_shared(fit::GllvmFit, lvl::Symbol)
    p = fit.model.p
    lvl === :unit && return Matrix(fit.pars.Λ * fit.pars.Λ')
    (fit.model.K_W > 0 && fit.pars.Λ_W !== nothing) ?
        Matrix(fit.pars.Λ_W * fit.pars.Λ_W') : zeros(Float64, p, p)
end

# Whether the fit genuinely carries a given tier at all (mirrors R's
# `fit$use$rr_B || fit$use$diag_B` / `rr_W || diag_W` tier-presence gate).
function _r_tier_present(fit::GllvmFit, lvl::Symbol)
    lvl === :unit && return fit.model.K > 0 ||
        (fit.model.has_diag && fit.pars.σ²_B !== nothing)
    fit.model.K_W > 0 || (fit.model.has_diag && fit.pars.σ²_W !== nothing)
end

function _validate_communality_level(level::Symbol)
    level === :total && return :total
    lvl = _canonical_level(level)
    lvl === :site && throw(ArgumentError(
        "level = :site has no R tier-scoped analogue; use :unit, :unit_obs, or :total"))
    return lvl
end

"""
    extract_communality(fit::GllvmFit; level::Symbol = :unit) -> Vector

Per-trait communality at ONE tier, `c²_t = (Λ_tier Λ_tierᵀ)_tt /
Σ_tier,total_tt`, mirroring `gllvmTMB::extract_communality(level = ...)`.
The default `level = :unit` matches R's
tier-scoped denominator exactly: `σ_eps²` (the Gaussian observation
residual) never enters, because it is not one of R's `B`/`W`/`phy` tier
components. `level = :unit_obs` is the within-unit (W) twin.

On a fit with no diagonal Ψ_tier component (e.g. `has_diag = false`), the
shared and total tiers coincide exactly and `c²_t` degenerates to `1.0` for
every trait, matching `gllvmTMB` for the same model structure.

`level = :total` returns the total-variance estimand
(forwards to [`communality`](@ref)): shared / `sigma_y_site(fit)`, i.e.
every non-phylo tier the fit carries plus `σ_eps²`. The two estimands agree
only when `σ_eps == 0` and there is no W-tier.
"""
function extract_communality(fit::GllvmFit; level::Symbol = :unit)
    lvl = _validate_communality_level(level)
    lvl === :total && return communality(fit)
    shared = diag(_r_tier_shared(fit, lvl))
    total = diag(_r_tier_total(fit, lvl))
    return [(isfinite(t) && t > 0) ? s / t : NaN for (s, t) in zip(shared, total)]
end

"""
    extract_communality(fit::TwoLevelFit; level::Symbol = :unit) -> Vector

`level = :unit` returns [`communality_B`](@ref); `level = :unit_obs` returns
[`communality_W`](@ref).
"""
function extract_communality(fit::TwoLevelFit; level::Symbol = :unit)
    lvl = _canonical_level(level)
    return lvl === :unit ? communality_B(fit) : communality_W(fit)
end

"""
    extract_communality(fit, Y::AbstractMatrix; kwargs...) -> Vector

Non-Gaussian twin: forwards to `communality(fit, Y; kwargs...)`
(src/link_residual.jl) for `PoissonFit`, `NBFit`, `BetaFit`, `GammaFit`,
`BinomialFit`, `OrdinalFit`, `OrdinalPerTraitFit`.
"""
extract_communality(fit::Union{_NonGaussianLatentFit, BinomialFit, OrdinalFit,
                                OrdinalPerTraitFit},
                    Y::AbstractMatrix; kwargs...) = communality(fit, Y; kwargs...)

"""
    extract_correlations(fit::GllvmFit; level::Symbol = :unit) -> Matrix

Cross-trait correlation at ONE tier, `cov2cor(Σ_tier,total)`, mirroring
`gllvmTMB::extract_correlations(tier = ...)`'s point-only route
(`extract_Sigma(fit, level = tier, part = "total")\$R`). This is now the
DEFAULT (`level = :unit`): `σ_eps²` never enters the
tier total, matching R exactly (same tier-scoping as
[`extract_communality`](@ref)). `level = :unit_obs` is the within-unit (W)
twin.

`level = :total` recovers GLLVModels.jl's original TOTAL-variance estimand
(forwards to [`correlation`](@ref)): `ρ_ij = Σ_y_site,ij / √(Σ_y_site,ii ·
Σ_y_site,jj)`, standardising by every non-phylo tier plus `σ_eps²`. The two
estimands agree only when `σ_eps == 0` and there is no W-tier.
"""
function extract_correlations(fit::GllvmFit; level::Symbol = :unit)
    lvl = _validate_communality_level(level)
    lvl === :total && return correlation(fit)
    return _cov2cor(_r_tier_total(fit, lvl))
end

"""
    extract_correlations(fit::TwoLevelFit; level::Symbol = :unit) -> Matrix

`level = :unit` returns [`correlation_B`](@ref); `level = :unit_obs` returns
[`correlation_W`](@ref).
"""
function extract_correlations(fit::TwoLevelFit; level::Symbol = :unit)
    lvl = _canonical_level(level)
    return lvl === :unit ? correlation_B(fit) : correlation_W(fit)
end

"""
    extract_correlations(fit, Y::AbstractMatrix; kwargs...) -> Matrix

Non-Gaussian twin, forwards to `correlation(fit, Y; kwargs...)`.
"""
extract_correlations(fit::Union{_NonGaussianLatentFit, BinomialFit, OrdinalFit,
                                 OrdinalPerTraitFit},
                     Y::AbstractMatrix; kwargs...) = correlation(fit, Y; kwargs...)

"""
    extract_cross_correlations(fit; level::Symbol = :unit,
                                traits_i, traits_j) -> Matrix

Cross-trait correlation SUBMATRIX `R[traits_i, traits_j]` between two named
groups of traits (by positional integer index), mirroring the block-slicing
intent of `gllvmTMB::extract_cross_correlations()`. Deviation from R: no
Fisher-z confidence band and no name-based trait subsetting (positional
integer indices only, matching GLLVModels.jl's convention elsewhere — see
[`extract_Gamma`](@ref)); the CI band is Cluster 2 (derived-CI surfaces).

`level = :unit` is the only value accepted for `fit::GllvmFit` (it forwards
to [`extract_correlations`](@ref)'s own `level` default, R-tier-scoped as
of the estimand-alignment decision — see `extract_correlations`'s
docstring); any other value is validated and thrown, matching
[`bootstrap_Sigma`](@ref)'s validate-and-throw pattern, rather than being
silently ignored.
"""
function extract_cross_correlations(fit::GllvmFit; level::Symbol = :unit,
                                    traits_i::AbstractVector{<:Integer},
                                    traits_j::AbstractVector{<:Integer})
    lvl = _canonical_level(level)
    lvl === :unit || throw(ArgumentError(
        "extract_cross_correlations(::GllvmFit) currently supports level = :unit only " *
        "(GLLVModels.jl computes one site-level correlation tier for GllvmFit); got :$level"))
    R = extract_correlations(fit)
    return Matrix(R[traits_i, traits_j])
end
function extract_cross_correlations(fit::TwoLevelFit; level::Symbol = :unit,
                                    traits_i::AbstractVector{<:Integer},
                                    traits_j::AbstractVector{<:Integer})
    R = extract_correlations(fit; level = level)
    return Matrix(R[traits_i, traits_j])
end

# ---------------------------------------------------------------------------
# Residual covariance / correlation — thin re-slice of extract_Sigma, plus
# the gllvm-style get* aliases.
# ---------------------------------------------------------------------------

"""
    extract_residual_cov(fit; level::Symbol = :unit) -> Matrix

Implied trait covariance at the given tier — the `Sigma` component of
[`extract_Sigma`](@ref) (`part = :total`). Mirrors
`gllvmTMB::extract_residual_cov()`.
"""
extract_residual_cov(fit::_GllvmOrTwoLevel; level::Symbol = :unit) =
    extract_Sigma(fit; level = level, part = :total).Sigma

"""
    extract_residual_cor(fit; level::Symbol = :unit) -> Matrix

Implied trait correlation at the given tier — the `R` component of
[`extract_Sigma`](@ref) (`part = :total`). Mirrors
`gllvmTMB::extract_residual_cor()`.
"""
extract_residual_cor(fit::_GllvmOrTwoLevel; level::Symbol = :unit) =
    extract_Sigma(fit; level = level, part = :total).R

"""
    getResidualCov(fit; level::Symbol = :unit) -> Matrix

`gllvm`-style compatibility spelling of [`extract_residual_cov`](@ref).
"""
getResidualCov(fit::_GllvmOrTwoLevel; level::Symbol = :unit) = extract_residual_cov(fit; level = level)

"""
    getResidualCor(fit; level::Symbol = :unit) -> Matrix

`gllvm`-style compatibility spelling of [`extract_residual_cor`](@ref).
"""
getResidualCor(fit::_GllvmOrTwoLevel; level::Symbol = :unit) = extract_residual_cor(fit; level = level)

# ---------------------------------------------------------------------------
# Ordination.
# ---------------------------------------------------------------------------

"""
    extract_ordination(fit, Y; rotate::Bool = true) -> (sites, species, rotation)

Canonical snake_case accessor forwarding to [`ordination`](@ref), mirroring
`gllvmTMB::extract_ordination()`.
"""
extract_ordination(fit, Y::AbstractMatrix; rotate::Bool = true) = ordination(fit, Y; rotate = rotate)

# ---------------------------------------------------------------------------
# Ordinal cutpoints.
# ---------------------------------------------------------------------------

"""
    extract_cutpoints(fit::OrdinalFit) -> (τ = Vector, C = Int)

Shared ordered cutpoints `τ₁ < … < τ_{C-1}` (common across species) from a
fitted ordinal GLLVM. Mirrors `gllvmTMB::extract_cutpoints()`.
"""
extract_cutpoints(fit::OrdinalFit) = (τ = copy(fit.τ), C = fit.C)

"""
    extract_cutpoints(fit::OrdinalPerTraitFit) -> (τ = Matrix, C = Vector{Int})

Per-trait ordered cutpoints (native `gllvmTMB` parity shape): `τ` is
`p × max(C_t - 1)`, padded with `NaN` past each trait's own cutpoint count;
`C` is the per-trait category count.
"""
extract_cutpoints(fit::OrdinalPerTraitFit) = (τ = copy(fit.τ), C = copy(fit.C))

# ---------------------------------------------------------------------------
# Variance-proportion decomposition and phylogenetic signal — forward to the
# existing GllvmFit generics in src/confint_derived.jl.
# ---------------------------------------------------------------------------

"""
    extract_proportions(fit::GllvmFit; component::Symbol = :shared,
                        level::Symbol = :unit) -> Vector

Per-trait variance-share decomposition. For `component = :shared` (the
default), mirrors `gllvmTMB::extract_proportions()`'s `shared_unit` row:
numerator `(Λ_B Λ_Bᵀ)_tt`, denominator the sum of every R tier component the
fit genuinely carries (`shared_unit` + `unique_unit` + `shared_unit_obs` +
`unique_unit_obs`, whichever are present — `σ_eps²` is never one of them,
same tier-scoping as [`extract_communality`](@ref) /
[`extract_correlations`](@ref)). This tier-scoped `:shared` route is now the
DEFAULT; on a fit with only a `:unit`
tier and no diagonal (e.g. `has_diag = false`), it degenerates to `1.0` for
every trait, matching R's own degenerate behaviour on such a fit.

Any other `component` (`:unique_W`, `:unique_B`, `:unique_Wd`, `:residual`),
or `level = :total`, forwards unchanged to the existing [`proportions`](@ref)
generic (GLLVModels.jl's original TOTAL-variance composition, `sigma_y_site(fit)`
denominator) — those components/level are not part of this alignment slice.
"""
function extract_proportions(fit::GllvmFit; component::Symbol = :shared, level::Symbol = :unit)
    (component !== :shared || level === :total) && return proportions(fit; component = component)
    total = zeros(Float64, fit.model.p)
    _r_tier_present(fit, :unit) && (total .+= diag(_r_tier_total(fit, :unit)))
    _r_tier_present(fit, :unit_obs) && (total .+= diag(_r_tier_total(fit, :unit_obs)))
    shared_unit = diag(_r_tier_shared(fit, :unit))
    return [(isfinite(t) && t > 0) ? s / t : NaN for (s, t) in zip(shared_unit, total)]
end

"""
    extract_phylo_signal(fit::GllvmFit; Σ_phy = nothing) -> Vector

Per-trait phylogenetic signal `H²`; forwards to the existing
[`phylo_signal`](@ref) generic. Mirrors `gllvmTMB::extract_phylo_signal()`
(point estimate only — the CI band is Cluster 2 / `confint_derived*.jl`).
"""
extract_phylo_signal(fit::GllvmFit; Σ_phy::Union{Nothing, AbstractMatrix} = nothing) =
    phylo_signal(fit; Σ_phy = Σ_phy)

# ---------------------------------------------------------------------------
# Repeatability / ICC.
# ---------------------------------------------------------------------------

"""
    extract_repeatability(fit::TwoLevelFit) -> Vector

Per-trait repeatability (ICC) `R_t = Σ_B[t,t] / (Σ_B[t,t] + Σ_W[t,t])`;
forwards to the existing [`repeatability`](@ref) generic. Mirrors
`gllvmTMB::extract_repeatability()` (point estimate only — Wald/bootstrap CI
columns are Cluster 2).
"""
extract_repeatability(fit::TwoLevelFit) = repeatability(fit)

"""
    extract_ICC_site(fit::GllvmFit) -> Vector

Per-trait unit-level intraclass correlation `ICC_t = v_B,t / (v_B,t + v_W,t)`,
`v_B,t = diag(extract_Sigma(fit; level=:unit).Sigma)`,
`v_W,t = diag(extract_Sigma(fit; level=:unit_obs).Sigma)`. Mirrors
`gllvmTMB::extract_ICC_site()` (`link_residual = "none"`; GLLVModels.jl's
`GllvmFit` is Gaussian-only so there is no implicit link residual to add).
NaN where `v_B,t + v_W,t` is not finite-positive, matching R's
`.safe_icc_ratio()`.
"""
function extract_ICC_site(fit::GllvmFit)
    vB = diag(extract_Sigma(fit; level = :unit, part = :total).Sigma)
    vW = diag(extract_Sigma(fit; level = :unit_obs, part = :total).Sigma)
    denom = vB .+ vW
    return [((isfinite(denom[t]) && denom[t] > 0) ? vB[t] / denom[t] : NaN) for t in eachindex(denom)]
end

"""
    extract_ICC_site(fit::TwoLevelFit) -> Vector

`TwoLevelFit` twin of [`extract_ICC_site(::GllvmFit)`](@ref); identical to
[`extract_repeatability(::TwoLevelFit)`](@ref) — the unit-level ICC and the
repeatability coincide for a two-level fit.
"""
extract_ICC_site(fit::TwoLevelFit) = extract_repeatability(fit)

# ---------------------------------------------------------------------------
# Omega — sum of the fitted covariance tiers.
# ---------------------------------------------------------------------------

"""
    extract_Omega(fit::GllvmFit; level::Symbol = :auto) -> Matrix

Total implied trait covariance, summing only the covariance TIERS the fit
genuinely carries — the phylogenetic block (`Λ_phy_aug Λ_phy_augᵀ` when
`K_phy > 0 || has_phy_unique`), the `:unit` tier (when `K > 0` or a diagonal
`σ²_B` is present), and the `:unit_obs` tier (when `K_W > 0` or a diagonal
`σ²_W` is present) — using each present tier's R-tier-scoped total (no
`σ_eps²`, see [`extract_communality`](@ref)). Mirrors
`gllvmTMB::extract_Omega()` with `tiers = NULL` (auto-detected) and
`link_residual = "none"` (Gaussian `GllvmFit` has no implicit link residual
to add). The default `level = :auto` includes only tiers present in the fit
and excludes the Gaussian observation residual `σ_eps²`.

`level = :total` returns the unconditional-sum estimand
(`Σ_unit + Σ_unit_obs` via `extract_Sigma`, `:unit_obs` always including
`σ_eps²·I` regardless of W-tier presence).
"""
function extract_Omega(fit::GllvmFit; level::Symbol = :auto)
    p = fit.model.p
    Ω = if level === :total
        Matrix(extract_Sigma(fit; level = :unit, part = :total).Sigma .+
               extract_Sigma(fit; level = :unit_obs, part = :total).Sigma)
    else
        level === :auto || throw(ArgumentError("level must be :auto or :total; got $(level)"))
        acc = zeros(Float64, p, p)
        _r_tier_present(fit, :unit) && (acc .+= _r_tier_total(fit, :unit))
        _r_tier_present(fit, :unit_obs) && (acc .+= _r_tier_total(fit, :unit_obs))
        acc
    end
    has_phy = (fit.model.K_phy > 0) || fit.model.has_phy_unique
    if has_phy
        Λ_phy_aug = if fit.pars.Λ_phy !== nothing && fit.pars.σ_phy !== nothing
            hcat(fit.pars.Λ_phy, reshape(collect(Float64, fit.pars.σ_phy), p, 1))
        elseif fit.pars.Λ_phy !== nothing
            fit.pars.Λ_phy
        else
            reshape(collect(Float64, fit.pars.σ_phy), p, 1)
        end
        Ω = Ω .+ Λ_phy_aug * Λ_phy_aug'
    end
    return Matrix(Ω)
end

# ---------------------------------------------------------------------------
# Still blocked (no stub — see docs/dev-log/core070/extractors-slice-notes.md
# for the full accounting):
#   * extract_residual_split — needs the per-family link-residual bank wired
#     to an explicit OLRE fit tag; GLLVModels.jl's K_W tier is not that tag.
#   * extract_coevolution_modules — needs a module/eigen-decomposition of Γ
#     that no coevolution fit type currently computes.
#   * getREsd — needs TMB-sdreport-style marginal SDs of the random effects
#     from the joint precision; no such accessor exists in GLLVModels.jl yet (it
#     is Hessian/SE machinery, i.e. Cluster 2 territory, not a point readout).
# ---------------------------------------------------------------------------
