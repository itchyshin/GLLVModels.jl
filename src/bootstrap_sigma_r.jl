# bootstrap_Sigma_r: parametric bootstrap of the pinned-R `bootstrap_Sigma` behaviour.
#
# Oracle (read-only): gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9, R/bootstrap-sigma.R.
# Spec: docs/dev-log/plans/2026-11-01-bootstrap-sigma-checkpoint-ledger.md ("Spec").
#
# Contract in one paragraph. ONE batch of `n_boot` datasets is drawn first, in the parent,
# from ONE stream (`seed` -> MersenneTwister, else the default RNG); the refit loop draws
# nothing from it. Each replicate is refit exactly once and every requested summary is
# extracted from that one refit. A replicate FAILS if the refit throws or returns a
# non-converged fit; failed replicates are NaN-shaped and excluded from the percentiles.
# Bounds are type-7 percentiles (`_derived_percentile`) of the finite draws, no minimum
# survivor rule. This function never calls `extract_Sigma(:unit_obs)` or `extract_ICC_site`
# (both add sigma_eps^2 unconditionally). The GllvmFit tier B is `_r_tier_total(fit, :unit)`:
# LambdaLambda' only when `!has_diag`, but the identified total `sigma_y_site(fit)` (which
# DOES include sigma_eps^2 and sigma^2_W) when `has_diag && K_W == 0`. That second meaning
# rests on a Julia comment (extractors.jl:270-276), not on R output: UNVERIFIED against R.
#
# WIRING: included by src/GLLVModels.jl after extractors.jl, confint_derived.jl and twolevel.jl
# (PR #1056). The legacy `bootstrap_Sigma` (confint_derived_wald.jl) is not touched and is a
# different estimand.

"""
    BootstrapSigmaR

Result of [`bootstrap_Sigma_r`](@ref); mirrors the fields of the R `bootstrap_Sigma` list.
`point_est`, `ci_lower`, `ci_upper` are `NamedTuple`s in R's entry order (per tier in
`level` order: `Sigma_*`, `R_*`, `communality_*`; then `ICC_site`). `n_effective` has
entries for vector summaries only (R limitation, kept). `level` holds the internal tier
names (`:B`, `:W`, `:phy`), never `:unit`/`:unit_obs`.
"""
struct BootstrapSigmaR
    point_est::NamedTuple
    ci_lower::NamedTuple
    ci_upper::NamedTuple
    n_effective::NamedTuple
    boot_median::NamedTuple
    ci_method::String
    link_residual::Symbol
    conf::Float64
    n_boot::Int
    coverage_ceiling::Float64
    n_failed::Int
    level::Vector{Symbol}
    what::Vector{Symbol}
    draws::Union{Nothing, Vector{NamedTuple}}
end

function Base.show(io::IO, r::BootstrapSigmaR)
    print(io, "BootstrapSigmaR(level=", r.level, ", what=", r.what, ", n_boot=", r.n_boot,
          ", n_failed=", r.n_failed, ", conf=", r.conf, ", ci_method=", r.ci_method, ")")
end

const _BSIG_LEVELS_ALL = (:unit, :unit_obs, :phy, :B, :W)
const _BSIG_WHAT_ALL = (:Sigma, :R, :communality, :ICC, :cross_corr)

_bsig_refuse(msg::AbstractString) = throw(ArgumentError("bootstrap_Sigma_r: " * msg))

# ---------------------------------------------------------------- argument handling
_bsig_symvec(x::Symbol) = Symbol[x]
_bsig_symvec(x) = Symbol[Symbol(s) for s in x]

# R: `.normalise_level` (unit -> B, unit_obs -> W) after match.arg; then unique, first-seen.
function _bsig_normalise_level(level)
    lv = _bsig_symvec(level)
    isempty(lv) && _bsig_refuse("level is empty")
    out = Symbol[]
    for l in lv
        l in _BSIG_LEVELS_ALL || _bsig_refuse(
            "unknown level :$l; choose from $(_BSIG_LEVELS_ALL)")
        c = l === :unit ? :B : l === :unit_obs ? :W : l
        c in out || push!(out, c)
    end
    return out
end

function _bsig_check_what(what)
    wh = _bsig_symvec(what)
    isempty(wh) && _bsig_refuse("what is empty")
    for w in wh
        w in _BSIG_WHAT_ALL || _bsig_refuse(
            "unknown summary :$w; choose from $(_BSIG_WHAT_ALL)")
    end
    return unique(wh)
end

# ---------------------------------------------------------------- per-class adapters
# A "tier pair" is (Sigma_t, shared_t): the tier total and its shared Lambda Lambda' block
# (nothing when no rr tier is used). For a GllvmFit the total is Lambda Lambda' when
# `!has_diag` and sigma_y_site(fit) (incl. sigma_eps^2) when `has_diag && K_W == 0`;
# the latter is UNVERIFIED against R output (see the docstring of `bootstrap_Sigma_r`).
_bsig_available_tiers(fit::GllvmFit) =
    _r_tier_present(fit, :unit) ? Symbol[:B] : Symbol[]
_bsig_available_tiers(::TwoLevelFit) = Symbol[:B, :W]

function _bsig_pair(fit::GllvmFit, tier::Symbol)
    tier === :B || throw(ArgumentError("internal: GllvmFit has only tier :B in v1"))
    Σ = _r_tier_total(fit, :unit)
    shared = fit.model.K >= 1 ? _r_tier_shared(fit, :unit) : nothing
    return Σ, shared
end

function _bsig_pair(fit::TwoLevelFit, tier::Symbol)
    tier === :B && return fit.Σ_B, fit.Λ_B * fit.Λ_B'
    tier === :W && return fit.Σ_W, fit.Λ_W * fit.Λ_W'
    throw(ArgumentError("internal: TwoLevelFit has no tier :$tier"))
end

# R `.safe_icc_ratio`: NaN where the denominator is not finite or <= 0.
function _bsig_icc(ΣB::AbstractMatrix, ΣW::AbstractMatrix)
    vB = diag(ΣB); vW = diag(ΣW)
    return [(isfinite(b + w) && (b + w) > 0) ? b / (b + w) : NaN for (b, w) in zip(vB, vW)]
end

# All summary entries of one fit, in R order. Returned as Vector{Pair{Symbol,Any}}.
function _bsig_entries(fit, tiers::Vector{Symbol}, what::Vector{Symbol})
    out = Pair{Symbol, Any}[]
    pairs = Dict{Symbol, Any}()
    for t in tiers
        Σ, shared = _bsig_pair(fit, t)
        pairs[t] = (Σ, shared)
        :Sigma in what && push!(out, Symbol(:Sigma_, t) => Matrix{Float64}(Σ))
        :R in what && push!(out, Symbol(:R_, t) => _cov2cor(Σ))
        if :communality in what && shared !== nothing
            dS = diag(shared); dT = diag(Σ)
            push!(out, Symbol(:communality_, t) =>
                  Float64[(isfinite(d) && d > 0) ? s / d : NaN for (s, d) in zip(dS, dT)])
        end
    end
    if :ICC in what && :B in tiers && :W in tiers
        push!(out, :ICC_site => _bsig_icc(pairs[:B][1], pairs[:W][1]))
    end
    return out
end

_bsig_nan_like(v::AbstractMatrix) = fill(NaN, size(v))
_bsig_nan_like(v::AbstractVector) = fill(NaN, length(v))

# ---------------------------------------------------------------- class-specific setup
# Each returns (simulate_batch(rng) -> Vector{Matrix}, refit(Y) -> fit).
function _bsig_setup(fit::GllvmFit, n_boot::Int, y, n_sites, X, _refit)
    m = fit.model
    p = m.p
    n = n_sites !== nothing ? Int(n_sites) :
        X !== nothing ? size(X, 2) :
        y !== nothing ? size(y, 2) :
        _bsig_refuse("needs n_sites. Pass one of `y = ...`, `X = ...`, or `n_sites = ...`.")
    q = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    Xm = _mean_X(fit, X, n)
    q > 0 && Xm === nothing && _bsig_refuse(
        "the fitted model has q = $q fixed effects; pass X.")
    μ̂ = Xm === nothing ? zeros(Float64, p, n) : _derived_mean_from_X(fit, Xm)
    A = _derived_site_cov(fit)
    L_site = cholesky(Symmetric((A + A') ./ 2)).L
    refit_fn = _refit === nothing ? fit_gaussian_gllvm : _refit
    simulate = function (rng)
        Ys = Vector{Matrix{Float64}}(undef, n_boot)
        for b in 1:n_boot
            yb = Matrix{Float64}(undef, p, n)
            _derived_simulate!(rng, yb, μ̂, L_site, nothing, nothing)
            Ys[b] = yb
        end
        return Ys
    end
    refit = Y -> refit_fn(Y; K = m.K, K_W = m.K_W, has_diag = m.has_diag,
                          K_phy = m.K_phy, has_phy_unique = m.has_phy_unique,
                          Σ_phy = nothing, X = Xm)
    return simulate, refit
end

function _bsig_setup(fit::TwoLevelFit, n_boot::Int, individual, center::Bool, _refit)
    individual === nothing && _bsig_refuse(
        "a TwoLevelFit needs `individual` (the fit stores only nindiv).")
    p = size(fit.Λ_B, 1)
    K_B = size(fit.Λ_B, 2); K_W = size(fit.Λ_W, 2)
    codes, _ = _code_grouping(individual)
    maximum(codes) == fit.nindiv || _bsig_refuse(
        "`individual` has $(maximum(codes)) groups but the fit has nindiv = $(fit.nindiv).")
    ind_idx = [findall(==(g), codes) for g in 1:maximum(codes)]
    ntot = length(individual)
    LB = _psd_sqrt_factor(fit.Σ_B); LW = _psd_sqrt_factor(fit.Σ_W)
    refit_fn = _refit === nothing ? fit_twolevel_gaussian : _refit
    simulate = function (rng)
        Ys = Vector{Matrix{Float64}}(undef, n_boot)
        for b in 1:n_boot
            yb = Matrix{Float64}(undef, p, ntot)
            for idx in ind_idx
                bi = LB * randn(rng, p)
                for j in idx
                    yb[:, j] = bi .+ LW * randn(rng, p)
                end
            end
            Ys[b] = yb
        end
        return Ys
    end
    refit = Y -> refit_fn(Y, individual; K_B = K_B, K_W = K_W, center = center)
    return simulate, refit
end

# ---------------------------------------------------------------- refusals
function _bsig_check_fit(fit::GllvmFit, Σ_phy)
    _has_gaussian_record(fit) && _bsig_refuse(
        "a GllvmFit with a retained Gaussian record is refused in v1 (its refit route is " *
        "seeded per replicate, not an R-style single batch).")
    m = fit.model
    m.K_W > 0 && _bsig_refuse(
        "K_W > 0: the B/W split of a GllvmFit is unidentified (one unit per column of y), " *
        "so no tier-scoped Sigma exists to bootstrap.")
    (m.K_phy > 0 || m.has_phy_unique) && _bsig_refuse(
        "a fit with a phylogenetic block is refused in v1 (no phy tier, no verified phy refit route).")
    Σ_phy === nothing || _bsig_refuse("Σ_phy is only meaningful for phylogenetic fits, refused in v1.")
    _check_unpinned_refit(fit, "bootstrap_Sigma_r")
    return nothing
end
_bsig_check_fit(::TwoLevelFit, Σ_phy) = nothing

# ---------------------------------------------------------------- driver
"""
    bootstrap_Sigma_r(fit::Union{GllvmFit,TwoLevelFit}; n_boot=999,
        level=(:unit,:unit_obs,:phy,:B,:W), what=(:Sigma,:R,:communality,:ICC,:cross_corr),
        conf=0.95, seed=nothing, n_cores=1, progress=true, keep_draws=false,
        link_residual=:auto, y=nothing, n_sites=nothing, X=nothing, Σ_phy=nothing,
        individual=nothing, center=true) -> BootstrapSigmaR

Parametric percentile bootstrap of covariance summaries, following the pinned R
`bootstrap_Sigma` (gllvmTMB 9539352f6). `GllvmFit` (no W tier, no pins, no phylo block,
no retained record): tier `:B` only; `Sigma`, `R`, `communality`; `ICC` is dropped and
`cross_corr` is always skipped (R needs a multinomial trait). `TwoLevelFit` (pass
`individual`): tiers `:B`, `:W` and `ICC_site`. Everything else throws `ArgumentError`.
`n_cores > 1` is refused. See the ledger Spec for the full contract; the legacy
`bootstrap_Sigma` is a different estimand and is unchanged.

# What tier `:B` means for a `GllvmFit` (it changes with `has_diag`)

- `has_diag = false` (rr-only): `Sigma_B = ΛΛ'` only. There is no diagonal and no
  `σ_eps²`. `Sigma_B` is rank-deficient (rank `K`), and for `K = 1` `R_B` is degenerate
  (all off-diagonals are `±1`). `communality_B` is **exactly 1 with a zero-width interval**
  (shared `ΛΛ'` over total `ΛΛ'`); it carries no information, and an `@info` says so.
- `has_diag = true` and `K_W == 0`: `Sigma_B` is the identified total
  `sigma_y_site(fit) = ΛΛ' + diag(σ²_B + σ²_W + σ_eps²)`, so it DOES include `σ_eps²` and
  `σ²_W`; `communality_B = diag(ΛΛ') ./ diag(Sigma_B)` uses that total as denominator.
  **UNVERIFIED against R output:** that this equals R's unit tier for such a model rests on a
  Julia comment (`extractors.jl:270-276`) and was never compared with an R run.

`Sigma_B`, `R_B` and `communality_B` are therefore different estimands in the two cases;
do not compare them across `has_diag` settings.

# Other behaviour worth knowing

- `TwoLevelFit` does not store `center`; the refit assumes `center = true` unless the
  keyword says otherwise. Simulated data are zero-mean.
- Draws are filtered with `isfinite`; R's `quantile(na.rm = TRUE)` drops only `NA`/`NaN`,
  so a `±Inf` draw is dropped here but would be kept by R (not expected to arise).
- A refit that throws or returns a non-converged fit is a failed replicate. An error while
  extracting summaries from a converged refit blanks every entry of that replicate to `NaN`
  and is NOT counted in `n_failed` (R skips per level; here one or two tiers make the
  difference small). `InterruptException` is never swallowed.
- `n_boot = 999` and `progress = true` (one `@info` per replicate) mirror R and need
  maintainer agreement as Julia defaults.
"""
function bootstrap_Sigma_r(fit::Union{GllvmFit, TwoLevelFit};
        n_boot::Integer = 999,
        level = _BSIG_LEVELS_ALL,
        what = _BSIG_WHAT_ALL,
        conf::Real = 0.95,
        seed::Union{Nothing, Integer} = nothing,
        n_cores::Integer = 1,
        progress::Bool = true,
        keep_draws::Bool = false,
        link_residual::Symbol = :auto,
        y = nothing, n_sites = nothing, X = nothing, Σ_phy = nothing,
        individual = nothing, center::Bool = true,
        _refit::Union{Nothing, Function} = nothing)

    # 1. class-specific refusals (R: class and structural asserts come first)
    _bsig_check_fit(fit, Σ_phy)
    # 2-4. level, what, link_residual, conf, n_boot (R order)
    lev = _bsig_normalise_level(level)
    wh = _bsig_check_what(what)
    link_residual in (:auto, :none) || _bsig_refuse(
        "link_residual must be :auto or :none; got :$link_residual")
    (0 < conf < 1) || _bsig_refuse("conf must be in (0, 1); got $conf")
    n_boot >= 1 || _bsig_refuse("n_boot must be >= 1; got $n_boot")
    n_cores == 1 || _bsig_refuse("n_cores = $n_cores is unsupported in v1 (use 1).")
    nb = Int(n_boot)

    # 5a. coverage ceiling warnings (R:252-277). Never both; neither throws.
    min_boot = ceil(Int, 2 / (1 - conf)) - 1
    ceiling = (nb - 1) / (nb + 1)
    if nb < min_boot
        @warn "bootstrap_Sigma_r: n_boot = $nb is below min_boot = $min_boot for conf = $conf; " *
              "the achievable coverage is capped at (n_boot-1)/(n_boot+1) = $(round(ceiling; digits = 4))."
    elseif nb < 999
        @warn "bootstrap_Sigma_r: n_boot = $nb < 999; percentile bounds are noisy (R recommends at least 999)."
    end

    # 5b. tier availability (R:280-296)
    avail = _bsig_available_tiers(fit)
    kept = Symbol[t for t in lev if t in avail]
    dropped = Symbol[t for t in lev if !(t in avail)]
    isempty(kept) && _bsig_refuse(
        "None of the requested level(s) are present in the fit. Available: " *
        (isempty(avail) ? "(none)" : join(avail, ", ")))
    isempty(dropped) || @info "bootstrap_Sigma_r: level(s) $(join(dropped, ", ")) not present in this fit; dropped."
    # 5c. ICC needs both B and W (R:299-306)
    if :ICC in wh && !(:B in kept && :W in kept)
        @info "bootstrap_Sigma_r: ICC requires both B and W tiers; dropping ICC from what."
        wh = Symbol[w for w in wh if w !== :ICC]
    end

    # point estimate (original fit); refuse before any simulation if nothing can be reported
    pe_pairs = _bsig_entries(fit, kept, wh)
    isempty(pe_pairs) && _bsig_refuse(
        "none of the requested summaries ($(join(wh, ", "))) is available for this fit and level; " *
        "nothing to bootstrap.")
    keys_pe = Symbol[k for (k, _) in pe_pairs]

    # rr-only GllvmFit: B = ΛΛ' only, so communality_B is identically 1 (zero-width interval)
    if fit isa GllvmFit && !fit.model.has_diag && :communality in wh && :B in kept
        @info "bootstrap_Sigma_r: communality_B is exactly 1 with a zero-width interval for an " *
              "rr-only fit (has_diag = false): the B tier is ΛΛ' with no diagonal, so shared/total = 1. " *
              "It carries no information; refit with has_diag = true for an informative communality."
    end

    # 6. one batch drawn first, then one refit per replicate
    simulate, refit = fit isa GllvmFit ?
        _bsig_setup(fit, nb, y, n_sites, X, _refit) :
        _bsig_setup(fit, nb, individual, center, _refit)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    Ys = simulate(rng)

    draws = Vector{Dict{Symbol, Any}}(undef, nb)
    failed = fill(false, nb)
    for b in 1:nb
        progress && @info "bootstrap_Sigma_r: replicate $b / $nb"
        fb = try
            refit(Ys[b])
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        if fb === nothing || !(fb isa typeof(fit)) || !fb.converged
            failed[b] = true
            draws[b] = Dict{Symbol, Any}(k => _bsig_nan_like(v) for (k, v) in pe_pairs)
            continue
        end
        ent = try
            Dict{Symbol, Any}(_bsig_entries(fb, kept, wh))
        catch e
            e isa InterruptException && rethrow()
            nothing   # R: an extraction error skips the entry; not counted as a failed replicate
        end
        draws[b] = Dict{Symbol, Any}(k => (ent !== nothing && haskey(ent, k) ?
                                           ent[k] : _bsig_nan_like(v)) for (k, v) in pe_pairs)
    end
    n_failed = count(failed)

    # 7. aggregation: type-7 percentiles of the finite draws, no minimum survivor rule
    α = 1 - conf
    lo = Dict{Symbol, Any}(); hi = Dict{Symbol, Any}(); neff = Dict{Symbol, Vector{Int}}()
    for (k, v) in pe_pairs
        L = fill(NaN, size(v)); U = fill(NaN, size(v)); N = zeros(Int, size(v))
        for idx in eachindex(v)
            vals = Float64[d[k][idx] for d in draws if isfinite(d[k][idx])]
            N[idx] = length(vals)
            if !isempty(vals)
                L[idx] = _derived_percentile(vals, α / 2)
                U[idx] = _derived_percentile(vals, 1 - α / 2)
            end
        end
        lo[k] = L; hi[k] = U
        v isa AbstractVector && (neff[k] = vec(N))
    end

    # 8. attrition warning over vector entries only (R:465-486, R limitation kept)
    if !isempty(neff)
        worst_k = :none; worst = Inf
        for k in keys_pe
            haskey(neff, k) || continue
            f = minimum(neff[k]) / nb
            f < worst && (worst = f; worst_k = k)
        end
        worst < 0.8 && @warn "bootstrap_Sigma_r: attrition. Summary $worst_k kept only " *
            "$(round(100 * worst; digits = 1))% of $nb replicates (below 80% retained); its bounds rest on the survivors."
    end

    nt(d, ks) = NamedTuple{Tuple(ks)}(Tuple(d[k] for k in ks))
    nkeys = Symbol[k for k in keys_pe if haskey(neff, k)]
    return BootstrapSigmaR(
        NamedTuple{Tuple(keys_pe)}(Tuple(v for (_, v) in pe_pairs)),
        nt(lo, keys_pe), nt(hi, keys_pe),
        NamedTuple{Tuple(nkeys)}(Tuple(neff[k] for k in nkeys)),
        NamedTuple(), "percentile", link_residual, Float64(conf), nb, ceiling, n_failed,
        kept, wh,
        keep_draws ? NamedTuple[nt(d, keys_pe) for d in draws] : nothing)
end

# Any other class: refuse explicitly (no verified simulate-and-refit route).
bootstrap_Sigma_r(fit; kwargs...) = _bsig_refuse(
    "unsupported fit class $(typeof(fit)). v1 supports GllvmFit and TwoLevelFit only " *
    "(PrecisionMultivariateFit, JointPhyloGroupedGaussianFit and non-Gaussian fits have no " *
    "verified simulate-and-refit route).")
