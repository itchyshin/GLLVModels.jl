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
# PrecisionMultivariateFit (tier :phy only) follows the same contract: the batch is drawn from the
# fitted law with every phylogenetic effect REDRAWN (R `simulate(fit)` unconditional path, R/methods-
# gllvmTMB.R at the pinned SHA), the loadings held fixed, one cold-start refit per replicate (R has no warm
# start), failure = throw or `!converged`. Evidence level: STRUCTURAL (code read against the pinned R plus
# tests and mutants); no numerical comparison with an R run exists. See the docstring.
#
# JointPhyloGroupedGaussianFit (tier :phy only, Stage 2) follows the same contract plus the ordinary
# grouping terms redrawn through their incidence matrices. Evidence level: STRUCTURAL plus JULIA
# DESCRIPTIVE (a pre-run on fixtures measured the failure fractions; no R run was compared). Documented
# DEVIATIONS: a replicate whose refit throws, does not converge OR is not component-identified is a failure
# (R's rule is convergence only); B and W are dropped (R would report them; the mapping of the ordinary
# terms to R's B/W is unverified); the per-replicate identification @warn of the fitter is silenced and
# counted. Converged replicates with a non-positive-definite Hessian are kept (as in R) and counted.
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
names (`:B`, `:W`, `:phy`), never `:unit`/`:unit_obs`. `what` echoes the request as in R (after `:ICC` is
dropped when B and W are not both present), so it can list `:cross_corr`, which is never produced for a
Gaussian fit: the keys of `point_est` are the authoritative list of summaries. `diagnostics` is an empty `NamedTuple` except for a
`JointPhyloGroupedGaussianFit`, where it carries the failure breakdown and the Hessian-curvature count
(see [`bootstrap_Sigma_r`](@ref)).
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
    diagnostics::NamedTuple
end

# `diagnostics` is empty for GllvmFit / TwoLevelFit / PrecisionMultivariateFit. For a
# JointPhyloGroupedGaussianFit it holds n_thrown, n_not_converged, n_non_identified (their sum is
# `n_failed`), n_converged (converged replicates, identified or not), n_hessian_not_pd (of those) and
# original_hessian_pd (Bool).
BootstrapSigmaR(pe, lo, hi, ne, bm, ci, lr, conf, nb, ceil, nf, level, what, draws) =
    BootstrapSigmaR(pe, lo, hi, ne, bm, ci, lr, conf, nb, ceil, nf, level, what, draws, NamedTuple())

function Base.show(io::IO, r::BootstrapSigmaR)
    print(io, "BootstrapSigmaR(level=", r.level, ", what=", r.what, ", n_boot=", r.n_boot,
          ", n_failed=", r.n_failed, ", conf=", r.conf, ", ci_method=", r.ci_method, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::BootstrapSigmaR)
    show(io, r)
    println(io)
    fmt(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : "NaN"
    for k in keys(r.point_est)
        pe = r.point_est[k]; lo = r.ci_lower[k]; hi = r.ci_upper[k]
        if pe isa AbstractVector
            neff = haskey(r.n_effective, k) ? r.n_effective[k] : nothing
            println(io, "  ", k, " (estimate [lower, upper], n_effective):")
            for i in eachindex(pe)
                println(io, "    [", i, "] ", fmt(pe[i]), " [", fmt(lo[i]), ", ", fmt(hi[i]), "]",
                        neff === nothing ? "" : ", n_effective = " * string(neff[i]))
            end
        else
            println(io, "  ", k, " (", join(size(pe), "x"), " matrix; diagonal estimate [lower, upper]):")
            for i in 1:min(size(pe)...)
                println(io, "    [", i, ",", i, "] ", fmt(pe[i, i]), " [", fmt(lo[i, i]), ", ", fmt(hi[i, i]), "]")
            end
        end
    end
    r.n_failed > 0 && println(io, "  ", r.n_failed, " of ", r.n_boot, " replicates failed (excluded).")
    if haskey(r.diagnostics, :n_hessian_not_pd)
        dg = r.diagnostics
        r.n_failed > 0 && println(io, "  (", dg.n_thrown, " thrown, ", dg.n_not_converged, " non-converged, ",
                                  dg.n_non_identified, " non-identified).")
        println(io, "  non-positive-definite Hessian in ", dg.n_hessian_not_pd, " of ", dg.n_converged,
                " converged replicates (kept; see ?bootstrap_Sigma_r).")
    end
    print(io, "  Full matrices: r.point_est, r.ci_lower, r.ci_upper.")
end

const _BSIG_LEVELS_ALL = (:unit, :unit_obs, :phy, :B, :W)
const _BSIG_WHAT_ALL = (:Sigma, :R, :communality, :ICC, :cross_corr)

_bsig_refuse(msg::AbstractString) = throw(ArgumentError("bootstrap_Sigma_r: " * msg))

# ---------------------------------------------------------------- argument handling
_bsig_symvec(x::Symbol) = Symbol[x]
_bsig_symvec(x::AbstractString) = Symbol[Symbol(x)]   # a String is not iterated as characters
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
# R keeps `phy` when phylo_rr || phylo_diag is fit; the class has no B/W tier. R has only B, W, phy.
_bsig_available_tiers(::PrecisionMultivariateFit) = Symbol[:phy]
# JointPhyloGroupedGaussianFit: phy only. R would also report B/W for a comparable multi-tier fit; the
# mapping of the ordinary grouping terms to R's B/W is UNVERIFIED, so they are not offered (dropped with an @info).
_bsig_available_tiers(::JointPhyloGroupedGaussianFit) = Symbol[:phy]

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

# PrecisionMultivariateFit tier :phy. Total = LL' + diag(unique) (R `extract_Sigma(level = "phy",
# part = "total")` for a Gaussian fit: NO residual variance); shared = LL' (R phylo_rr part). With
# `:barelowrank` (R `phylo_latent` alone) unique = 0, so total == shared and communality is exactly 1.
function _bsig_pair(fit::PrecisionMultivariateFit, tier::Symbol)
    tier === :phy || throw(ArgumentError("internal: PrecisionMultivariateFit has only tier :phy"))
    shared, _, total = _destination_b_covariance_parts(fit, :phy)
    return total, shared
end

# JointPhyloGroupedGaussianFit tier :phy. Total = `fit.phylo_covariance` = LL' + diag(unique) (NO residual and
# NO ordinary-term variance: the ordinary terms are separate sources, never added to the phy tier);
# shared = LL'. `:barelowrank` gives total == shared, so communality is exactly 1.
function _bsig_pair(fit::JointPhyloGroupedGaussianFit, tier::Symbol)
    tier === :phy || throw(ArgumentError("internal: JointPhyloGroupedGaussianFit has only tier :phy"))
    parts = _joint_phylo_grouped_covariance_parts(fit, :phylo)
    return parts.total, parts.shared
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
    y === nothing || ndims(y) == 2 || _bsig_refuse(
        "`y` must be a p x n matrix (p = $p traits); got a $(ndims(y))-d array.")
    n >= 2 || _bsig_refuse("n_sites must be >= 2; got $n")
    y === nothing || size(y, 1) == p || _bsig_refuse(
        "`y` has $(size(y, 1)) rows but the fit has p = $p traits; `y` must be p x n " *
        "(did you pass n x p?). Only size(y, 2) is used; its values are ignored.")
    X === nothing || ndims(X) == 3 || _bsig_refuse(
        "`X` must be a p x n_sites x q array (as passed to fit_gaussian_gllvm); got $(ndims(X))-d.")
    X === nothing || size(X, 2) == n || _bsig_refuse(
        "`X` has $(size(X, 2)) sites but n_sites = $n; they must agree.")
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
    # Built-in refit only (a user `_refit` keeps its own controls). A has_diag / K_W fit has a flat
    # ridge (sigma2_B, sigma2_W and sigma_eps^2 enter only as a sum), and a Heywood draw runs along
    # it: at the stock 500 iterations / f_tol = 0, LBFGS hits the cap in about 0.4% of fixture-c
    # replicates (logLik 1e-3 to 2e-3 below the optimum, dropped as 'failed'). Needing ~600-1800
    # iterations, they converge with a relative objective test, as R's nlminb does. Measured: 0 of
    # 1998 refits fail, logLik within 1e-4 of R, and it is not slower than stock (mean 60 vs 77 ms).
    # f_tol = 1e-10 stops too early (logLik up to 1.3e-3 low); do not loosen it. Never applied to
    # phylo fits (a relative f_tol stopped those early); a phylogenetic GllvmFit is refused in v1, and the
    # PrecisionMultivariateFit refit also keeps the default controls.
    # DEVIATION from the pinned-R behaviour (optimizer settings), see the PR description.
    refit_opts = (_refit === nothing && (m.has_diag || m.K_W > 0)) ?
        (iterations = 2000, f_tol = 1e-12) : NamedTuple()
    refit = Y -> refit_fn(Y; K = m.K, K_W = m.K_W, has_diag = m.has_diag,
                          K_phy = m.K_phy, has_phy_unique = m.has_phy_unique,
                          Σ_phy = nothing, X = Xm, refit_opts...)
    return simulate, refit
end

function _bsig_setup(fit::TwoLevelFit, n_boot::Int, individual, center::Bool, _refit)
    individual === nothing && _bsig_refuse(
        "a TwoLevelFit needs `individual` (the fit stores only nindiv).")
    p = size(fit.Λ_B, 1)
    K_B = size(fit.Λ_B, 2); K_W = size(fit.Λ_W, 2)
    length(individual) >= 2 || _bsig_refuse("`individual` is too short (needs one entry per observation).")
    codes, _ = _code_grouping(individual)
    maximum(codes) == fit.nindiv || _bsig_refuse(
        "`individual` has $(maximum(codes)) groups but the fit has nindiv = $(fit.nindiv). " *
        "(`individual` must have one entry per observation of the original data; its length " *
        "cannot be checked because the fit does not store it.)")
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

# PrecisionMultivariateFit, tier :phy. Batch simulator: for each replicate every phylogenetic effect
# is REDRAWN from its fitted law, loadings held fixed, as R's unconditional `simulate(fit)` does
# (R/methods-gllvmTMB.R at the pinned SHA: phylo_rr = one A-covariance draw per latent axis; phylo_diag
# = one A-covariance draw per TRAIT, independent across traits, scaled by the fitted unique SD).
# A draw with covariance A = Q^-1 on the augmented node set is `U \ z`, z ~ N(0, I), Q = U'U. Each
# observation reads the node `phy.species_aug_id[species_id[o]]`; the mean is `mean_design * beta`.
# Draw order per replicate: g (n_aug x rank), then h (n_aug x d; only when the fit has a unique
# component), then the residual. The RNG stream cannot equal R's (different language and generator):
# the law is the same, the draws are not. Dense Cholesky of Q: O(n_aug^3), fixtures and modest trees only.
function _bsig_pmv_simulate_batch(rng, fit::PrecisionMultivariateFit, n_boot::Integer)
    phy = fit.phy
    d, m = fit.response_shape
    U = cholesky(Symmetric(Matrix(phy.Q))).U
    nodes = phy.species_aug_id[fit.species_id]
    μ = reshape(fit.mean_design * fit.beta, d, m)
    r = size(fit.loading, 2)
    sd_res = sqrt.(fit.residual_variance)
    sd_uni = fit.phylo_unique_variance === nothing ? nothing : sqrt.(fit.phylo_unique_variance)
    Ys = Vector{Matrix{Float64}}(undef, n_boot)
    for b in 1:n_boot
        g = U \ randn(rng, phy.n_aug, r)
        yb = μ .+ fit.loading * transpose(g[nodes, :])
        if sd_uni !== nothing
            h = U \ randn(rng, phy.n_aug, d)
            yb .+= sd_uni .* transpose(h[nodes, :])
        end
        yb .+= sd_res .* randn(rng, d, m)
        Ys[b] = yb
    end
    return Ys
end

# Refit: the SAME tree object, rank, mode, residual mode, observation-to-tip map and complete mean design
# (`fit.mean_design` is accepted back as `X`; for a default-means fit it reproduces them exactly), default
# optimiser controls, and NO `start` (R refits cold; a warm start would shrink failure rates and tails).
# No relative `f_tol` (see the GllvmFit comment above: it stopped phylogenetic fits early).
function _bsig_setup(fit::PrecisionMultivariateFit, n_boot::Int, _refit)
    refit_fn = _refit === nothing ? fit_precision_multivariate : _refit
    simulate = rng -> _bsig_pmv_simulate_batch(rng, fit, n_boot)
    # The fitter's structural-redundancy @warn fires on every likelihood evaluation (about 1e3 per refit for
    # d = 2 rank 1 :explicitunique), so it is silenced here; the driver counts and announces failed refits
    # itself, and an exception is NOT silenced (it propagates to the driver's catch).
    refit = Y -> Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        refit_fn(Y, fit.phy; rank = fit.rank, mode = fit.mode,
                 residual_mode = fit.residual_mode, species_id = fit.species_id,
                 X = fit.mean_design, coefficient_names = fit.coefficient_names)
    end
    return simulate, refit
end

# JointPhyloGroupedGaussianFit, tier :phy. Batch simulator: the PMV draw (phylogenetic effects redrawn,
# loadings held fixed) PLUS, for each ordinary term k, one independent N(0, diag(ordinary_covariances[k]))
# draw per GROUP, added to each observation through its incidence row (`fit.incidences[k]` is n x groups).
# The ordinary terms must be redrawn: otherwise the phy estimate is bootstrapped on the wrong data law.
# Draw order per replicate: g (n_aug x rank), h (n_aug x d, only with a unique component), the ordinary
# terms in term order (d x groups each), then the residual. The RNG stream cannot equal R's. Dense Cholesky
# of Q: O(n_aug^3), fixtures and modest trees only.
function _bsig_jpg_simulate_batch(rng, fit::JointPhyloGroupedGaussianFit, n_boot::Integer)
    phy = fit.phy
    d, m = fit.response_shape
    U = cholesky(Symmetric(Matrix(phy.Q))).U
    nodes = phy.species_aug_id[fit.species_id]
    μ = reshape(fit.mean_design * fit.beta, d, m)
    r = size(fit.loading, 2)
    sd_res = sqrt.(fit.residual_variance)
    sd_uni = fit.phylo_unique_variance === nothing ? nothing : sqrt.(fit.phylo_unique_variance)
    sd_ord = [sqrt.(diag(C)) for C in fit.ordinary_covariances]
    incT = [sparse(transpose(inc)) for inc in fit.incidences]      # groups x n, so `z * incT[k]` is the fast dense-by-sparse product
    Ys = Vector{Matrix{Float64}}(undef, n_boot)
    for b in 1:n_boot
        g = U \ randn(rng, phy.n_aug, r)
        yb = μ .+ fit.loading * transpose(g[nodes, :])
        if sd_uni !== nothing
            h = U \ randn(rng, phy.n_aug, d)
            yb .+= sd_uni .* transpose(h[nodes, :])
        end
        for (k, inc) in enumerate(fit.incidences)
            z = sd_ord[k] .* randn(rng, d, size(inc, 2))
            yb .+= z * incT[k]
        end
        yb .+= sd_res .* randn(rng, d, m)
        Ys[b] = yb
    end
    return Ys
end

# Group labels for the refit, recovered from the stored incidences (one nonzero per row; the label is its
# column), keyed by the TERM NAME (`unit`, `unit_obs`, `cluster`, `cluster2`), never by position. The fitter
# validates `unit_obs` nesting against `unit` labels, which a fit without a `:unit` term does not store; the
# unit_obs partition itself is then a valid stand-in (every unit_obs level nested in itself) and adds no term.
function _bsig_jpg_group_labels(fit::JointPhyloGroupedGaussianFit)
    n = fit.response_shape[2]
    pairs = Pair{Symbol, Vector{Int}}[]
    for (term, inc) in zip(fit.terms, fit.incidences)
        rows, cols, _ = findnz(inc)
        lab = zeros(Int, n); lab[rows] = cols
        push!(pairs, term.name => lab)
    end
    names = Symbol[first(pr) for pr in pairs]
    if :unit_obs in names && !(:unit in names)
        push!(pairs, :unit => copy(last(pairs[findfirst(==(:unit_obs), names)])))
    end
    return NamedTuple(pairs)
end

# Refit: the SAME tree object, rank, phylo mode, ordinary terms, recovered labels, observation-to-tip map and
# complete mean design, default optimiser controls (g_tol 1e-5, 200 iterations), NO `start`, no `f_tol`. The
# fitter's per-replicate identification @warn is silenced here (the status is checked and counted by the driver
# instead, so n_boot warnings do not spam); an exception is NOT silenced, it propagates to the driver's catch.
function _bsig_setup(fit::JointPhyloGroupedGaussianFit, n_boot::Int, _refit)
    refit_fn = _refit === nothing ? fit_joint_phylo_grouped_gaussian : _refit
    labels = _bsig_jpg_group_labels(fit)
    simulate = rng -> _bsig_jpg_simulate_batch(rng, fit, n_boot)
    refit = Y -> Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        refit_fn(Y, fit.phy; rank = fit.rank, phylo_mode = fit.phylo_mode, terms = fit.terms,
                 species_id = fit.species_id, X = fit.mean_design,
                 coefficient_names = fit.coefficient_names, labels...)
    end
    return simulate, refit
end

# One dispatching call per class (the driver passes the same argument bundle to all of them).
_bsig_build(fit::GllvmFit, nb, a, _refit) = _bsig_setup(fit, nb, a.y, a.n_sites, a.X, _refit)
_bsig_build(fit::TwoLevelFit, nb, a, _refit) = _bsig_setup(fit, nb, a.individual, a.center, _refit)
_bsig_build(fit::PrecisionMultivariateFit, nb, a, _refit) = _bsig_setup(fit, nb, _refit)
_bsig_build(fit::JointPhyloGroupedGaussianFit, nb, a, _refit) = _bsig_setup(fit, nb, _refit)

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
function _bsig_check_fit(::PrecisionMultivariateFit, Σ_phy)
    Σ_phy === nothing || _bsig_refuse(
        "Σ_phy is not accepted for a PrecisionMultivariateFit: the tree is stored in the fit.")
    return nothing
end

# JointPhyloGroupedGaussianFit: refuse unless the ORIGINAL fit converged and its covariance components are
# identified (the phy tier is not separately interpretable otherwise, and replicates inherit the problem).
function _bsig_check_fit(fit::JointPhyloGroupedGaussianFit, Σ_phy)
    Σ_phy === nothing || _bsig_refuse(
        "Σ_phy is not accepted for a JointPhyloGroupedGaussianFit: the tree is stored in the fit.")
    fit.converged || _bsig_refuse(
        "the original JointPhyloGroupedGaussianFit did not converge (stopping reason :$(fit.stopping_reason)); " *
        "a parametric bootstrap around a non-converged fit is refused.")
    reason = _joint_phylo_grouped_component_identification(fit).reason
    reason === :identified || _bsig_refuse(
        "the original JointPhyloGroupedGaussianFit is not component-identified (reason :$reason): its " *
        "phylogenetic and ordinary covariance components are not separately interpretable, so a phy-tier " *
        "bootstrap is refused.")
    return nothing
end

# Per-class refusal of arguments that would otherwise be silently ignored. These classes keep their response
# shape, mean design and tree, so `y`, `n_sites`, `X` and `individual` mean nothing.
_bsig_check_args(fit, a) = nothing
function _bsig_check_args(fit::Union{PrecisionMultivariateFit, JointPhyloGroupedGaussianFit}, a)
    for k in (:y, :n_sites, :X, :individual)
        getfield(a, k) === nothing || _bsig_refuse(
            "`$k` is not accepted for a $(nameof(typeof(fit))) (the fit stores its response " *
            "shape, mean design and tree); it would be silently ignored.")
    end
    return nothing
end

# Status of a CONVERGED replicate: `:ok`, or `:non_identified` (JointPhyloGroupedGaussianFit only; DEVIATION
# from R, whose rule is convergence only). A throw while diagnosing counts as non-identified.
_bsig_replicate_status(fit, fb) = :ok
function _bsig_replicate_status(::JointPhyloGroupedGaussianFit, fb)
    reason = try
        _joint_phylo_grouped_component_identification(fb).reason
    catch e
        e isa InterruptException && rethrow()
        :invalid
    end
    return reason === :identified ? :ok : :non_identified
end

# ---------------------------------------------------------------- driver
"""
    bootstrap_Sigma_r(fit::Union{GllvmFit,TwoLevelFit,PrecisionMultivariateFit,
        JointPhyloGroupedGaussianFit}; n_boot=999,
        level=(:unit,:unit_obs,:phy,:B,:W), what=(:Sigma,:R,:communality,:ICC,:cross_corr),
        conf=0.95, seed=nothing, n_cores=1, progress=true, keep_draws=false,
        link_residual=:auto, y=nothing, n_sites=nothing, X=nothing, Σ_phy=nothing,
        individual=nothing, center=true) -> BootstrapSigmaR

Parametric percentile bootstrap of covariance summaries, following the pinned R
`bootstrap_Sigma` (gllvmTMB 9539352f6). `GllvmFit` (no W tier, no pins, no phylo block,
no retained record): tier `:B` only; `Sigma`, `R`, `communality`; `ICC` is dropped and
`cross_corr` is always skipped (R needs a multinomial trait). `TwoLevelFit` (pass
`individual`): tiers `:B`, `:W` and `ICC_site`. `PrecisionMultivariateFit` and
`JointPhyloGroupedGaussianFit`: tier `:phy` only (see below). Everything else throws `ArgumentError`.
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

# `PrecisionMultivariateFit` (tier `:phy` only)

Maps to R's phylogenetic tier: `mode = :barelowrank` is `phylo_latent` alone, `:explicitunique` is
`phylo_latent + phylo_unique`. `Sigma_phy = ΛΛ' + diag(unique)` with NO residual variance (R's Gaussian
`Sigma_phy`), `R_phy` its correlation, `communality_phy = diag(ΛΛ') ./ diag(Sigma_phy)` (R: link residual
`"none"`). B, W and `ICC_site` do not exist for this class (dropped with an `@info` at the default
`level`; `level = :B` alone throws); `cross_corr` is never produced; `y`, `n_sites`, `X`, `individual`
and `Σ_phy` are refused (the fit stores its tree, response shape and mean design). `mode = :barelowrank`
gives `communality_phy` exactly 1 with a zero-width interval (flagged by an `@info`).

Each replicate redraws every phylogenetic effect from its fitted law with the loadings held fixed (R's
unconditional `simulate`: one tree-correlated draw per latent axis, and, with a unique component, one per
trait), adds the fixed effects `mean_design * beta` and residual noise, and is refit once from default
starts (no warm start, no relative `f_tol`). This is a parametric bootstrap conditional on the tree,
`species_id`, mean design and fitted parameters; it is not a bootstrap of tips. Caveats:

- The RNG stream cannot match R's: parity is distributional only.
- `Sigma_phy` is on the scale of the supplied tree (`sigma2_phy` is fixed at 1); compare across trees
  through `R_phy` or `communality_phy` only.
- DEVIATION in the failure rule: R fails a replicate on `opt\$convergence != 0`; here a replicate fails
  when the refit throws or `!converged`, and `converged` also needs a fresh finite-difference gradient
  norm below `g_tol`. Failure rates are therefore not comparable with R's. The Hessian is checked by
  neither implementation.
- The simulator uses a dense Cholesky of the tree precision (O(n_aug^3)): fixtures and modest trees only.
- Small tip counts give biased, boundary-truncated percentile intervals for variance components (R
  shares this). Coverage has not been assessed.
- EVIDENCE LEVEL: structural. The contract was read from the pinned R source and is pinned by tests and
  mutation checks; the simulator law is checked against the model covariance and the fitter's own
  likelihood. NO numerical comparison with an R `bootstrap_Sigma` run exists, so "matches R output" is
  not claimed. (R's own test-suite has no phy-tier bootstrap test.) `center` is ignored.

# `JointPhyloGroupedGaussianFit` (tier `:phy` only; refused unless the ORIGINAL fit is identified and converged)

`Sigma_phy = ΛΛ' + diag(unique)` (`fit.phylo_covariance`): it contains NEITHER the residual variance NOR any
ordinary grouping-term variance; those are separate sources and are never added to the phy tier. `R_phy` and
`communality_phy = diag(ΛΛ') ./ diag(Sigma_phy)` follow. Each replicate redraws every phylogenetic effect as for
`PrecisionMultivariateFit` AND, for each ordinary term, one independent `N(0, diag(ordinary_covariances[k]))`
draw per group, added through the stored incidence matrix; loadings, fixed effects and variances are held at
the fit. Each replicate is refit once from default starts (no warm start, no relative `f_tol`; the fitter's
defaults `g_tol = 1e-5`, `iterations = 200`), with the same tree, rank, phylo mode, terms, `species_id` and
mean design. Group labels are recovered from the stored incidences and passed under each TERM's keyword
(`unit`, `unit_obs`, `cluster`, `cluster2`), so only the partitions (not the original label values) are kept.
B and W are dropped with an `@info` (default `level`; `level = :B` alone throws) and `ICC` is dropped.

DEVIATIONS from the pinned R, all counted and reported:

- **Failure rule.** A replicate FAILS (is in `n_failed`, NaN-shaped, excluded from the percentiles) when the
  refit throws, does not converge (`converged` also needs a finite-difference gradient norm below `g_tol`,
  unlike R's `nlminb` convergence code), OR its covariance components are not identified
  (`_joint_phylo_grouped_component_identification(refit).reason !== :identified`). R's rule is convergence only
  and has no identification check. The breakdown is in `result.diagnostics` (`n_thrown`, `n_not_converged`,
  `n_non_identified`; they sum to `n_failed`). Failure rates are therefore not comparable with R's.
- **B/W omitted.** R would report B/W tiers for a comparable multi-tier fit. How this class's ordinary terms
  map to R's B/W is unverified, so they are not bootstrapped.
- **Silenced fitter warning.** The fitter's per-replicate "not separately identifiable" `@warn` is silenced in
  the loop and counted in `diagnostics.n_non_identified`. Exceptions are not silenced.

Curvature. A converged replicate with a non-positive-definite Hessian is KEPT (as in R, which checks only
convergence, and as the percentile method needs only point estimates), and counted: `diagnostics.n_converged`
(converged replicates, identified or not) and `diagnostics.n_hessian_not_pd` (of those), reported in an `@info`.
Such replicates often have a variance on its boundary (a unique or ordinary variance near 0), so percentile ends
of quantities that depend on it can be truncated there. If the ORIGINAL fit's Hessian is not positive definite
(`diagnostics.original_hessian_pd == false`) a `@warn` says so; the run proceeds. In a pre-run on fixtures the
non-PD fraction among converged replicates was 0% to 71%: 33.7% and 71.4% in the two explicit-unique cases.

EVIDENCE LEVEL: structural plus Julia-descriptive. The contract is read from the pinned R source for the shared
parts and pinned by tests and mutation checks. The only other evidence is a diagnostic pre-run (six fixtures, one
seed each, `n_boot = 200`, `g_tol` 1e-5 to 2e-4, up to 200 iterations) in which 0% to 2.5% of replicates failed
(thrown, non-converged or non-identified); with `n_boot = 200` a count of 5 does not exclude a true failure rate
above 5%. Not tested: more than one ordinary term in a real refit, rank above 1 in a real refit, `:unit_obs` and
`:cluster2` in a real refit, covariates beyond trait intercepts, larger trees, unbalanced trees. NO numerical
comparison with an R run exists and NO R parity is claimed; `link_residual` and `center` are ignored.

# Other behaviour worth knowing

- `X` is the p x n_sites x q design array used at fit time (a 2-d matrix is refused) and
  must agree with `n_sites` when both are given. For a fit from
  `fit_gllvm(Y; family = Normal(), K)` (trait intercepts) no `X` is needed, but `n_sites`
  (or `y`) still is. `y` must be p x n; only `size(y, 2)` is read, its values are ignored.
  `n_sites` must be at least 2.
- `link_residual` is validated and stored on the result but has no effect for these Gaussian
  fits. `boot_median` is accepted by the R list but is always an empty `NamedTuple` here
  (not computed). For a `TwoLevelFit`, `y`, `n_sites`, `X` and `Σ_phy` are ignored; for a
  `GllvmFit`, `individual` and `center` are ignored.
- A warning always fires when ANY replicate is lost (refit failed, or the summaries could
  not be extracted), whatever `what` is requested; it names the count out of `n_boot`. `n_failed` itself counts
  failed refits only: an extraction error on a converged refit blanks that replicate but is
  not in `n_failed` (see below).
- `progress = true` emits one `@info` per replicate (about 1000 at the default) and the
  default `level` always logs a "level(s) ... dropped" `@info`; pass `progress = false` and an
  explicit `level` for quiet output.
- Correlation bounds (`R_*`) are clamped to [-1, 1] only when they overshoot by rounding
  (at most 1e-8); a larger excursion is left as is so a genuine error stays visible.
- A replicate summary of the wrong shape becomes a NaN draw and is counted in the "lost"
  warning (not in `n_failed`), like an extraction error.
- `n_sites` between 2 and p is accepted, but a fit with fewer sites than traits usually fails
  every refit: the result is then all-NaN, announced only by the "lost" warning.
- For `has_diag = true` the built-in refit (not a user `_refit`) uses `iterations = 2000,
  f_tol = 1e-12` (R's nlminb stops on a relative objective change); a deviation from the pinned
  R. It rescues the replicates that hit the stock 500-iteration cap and moves even
  already-converged replicates slightly (bounds by about 1e-5 in the fixtures measured).
- Entries whose point estimate is itself non-finite (for example a zero-variance trait) are
  not counted in the attrition warning.
- `TwoLevelFit` does not store `center`; the refit assumes `center = true` unless the
  keyword says otherwise. Simulated data are zero-mean.
- `boot_median` is an empty `NamedTuple`, as in R: the pinned R fills `boot_median` only for
  `multiple_r_*` entries (R/bootstrap-sigma.R:717-720), which Gaussian fits never have.
- Draws are filtered with `isfinite`; R's `quantile(na.rm = TRUE)` drops only `NA`/`NaN`,
  so a `±Inf` draw is dropped here but would be kept by R (not expected to arise).
- A refit that throws or returns a non-converged fit is a failed replicate. An error while
  extracting summaries from a converged refit blanks every entry of that replicate to `NaN`
  and is NOT counted in `n_failed` (R skips per level; here one or two tiers make the
  difference small). `InterruptException` is never swallowed.
- `n_boot = 999` and `progress = true` (one `@info` per replicate) mirror R and need
  maintainer agreement as Julia defaults.
"""
function bootstrap_Sigma_r(fit::Union{GllvmFit, TwoLevelFit, PrecisionMultivariateFit, JointPhyloGroupedGaussianFit};
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
    args = (y = y, n_sites = n_sites, X = X, individual = individual, center = center)
    _bsig_check_args(fit, args)
    # 2-4. level, what, link_residual, conf, n_boot (R order)
    lev = _bsig_normalise_level(level)
    wh = _bsig_check_what(what)
    link_residual in (:auto, :none) || _bsig_refuse(
        "link_residual must be :auto or :none; got :$link_residual")
    (0 < conf < 1) || _bsig_refuse("conf must be in (0, 1); got $conf")
    (seed === nothing || seed >= 0) || _bsig_refuse("seed must be >= 0; got $seed")
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
    isempty(dropped) || @info "bootstrap_Sigma_r: level(s) $(join(dropped, ", ")) not present in this fit; dropped." *
        (fit isa JointPhyloGroupedGaussianFit ?
         " For a JointPhyloGroupedGaussianFit R would report B/W tiers for a comparable multi-tier fit; the " *
         "mapping of this class's ordinary terms to R's B/W is unverified, so they are not bootstrapped " *
         "(a coverage difference from R)." : "")
    # 5c. ICC needs both B and W (R:299-306)
    if :ICC in wh && !(:B in kept && :W in kept)
        @info "bootstrap_Sigma_r: ICC requires both B and W tiers; dropping ICC from what."
        wh = Symbol[w for w in wh if w !== :ICC]
    end

    # point estimate (original fit); refuse before any simulation if nothing can be reported
    pe_pairs = _bsig_entries(fit, kept, wh)
    isempty(pe_pairs) && _bsig_refuse(
        "none of the requested summaries ($(isempty(wh) ? "none left after dropping ICC" : join(wh, ", "))) " *
        "is available for this fit and level; nothing to bootstrap (a GllvmFit, PrecisionMultivariateFit or " *
        "JointPhyloGroupedGaussianFit supports what = :Sigma, :R or :communality; :ICC needs a TwoLevelFit " *
        "with levels :B and :W; :cross_corr is never available, R needs a multinomial trait).")
    keys_pe = Symbol[k for (k, _) in pe_pairs]

    # rr-only GllvmFit: B = ΛΛ' only, so communality_B is identically 1 (zero-width interval)
    if fit isa GllvmFit && !fit.model.has_diag && :communality in wh && :B in kept
        @info "bootstrap_Sigma_r: communality_B is exactly 1 with a zero-width interval for an " *
              "rr-only fit (has_diag = false): the B tier is ΛΛ' with no diagonal, so shared/total = 1. " *
              "It carries no information; refit with has_diag = true for an informative communality."
    end

    # PrecisionMultivariateFit / JointPhyloGroupedGaussianFit :barelowrank (R `phylo_latent` alone):
    # phy total = LL' = shared
    if (fit isa PrecisionMultivariateFit || fit isa JointPhyloGroupedGaussianFit) &&
       fit.phylo_unique_variance === nothing && :communality in wh
        mk = fit isa JointPhyloGroupedGaussianFit ? "phylo_mode" : "mode"
        @info "bootstrap_Sigma_r: communality_phy is exactly 1 with a zero-width interval for a fit " *
              "without a phylogenetic unique component ($mk = :barelowrank): the phy tier is LL' with " *
              "no diagonal, so shared/total = 1. It carries no information; refit with " *
              "$mk = :explicitunique for an informative communality."
    end

    # JointPhyloGroupedGaussianFit: a non-PD Hessian at the original fit does not stop the run (the percentile
    # bootstrap needs only point estimates, as in R) but the interval ends can be boundary-truncated.
    if fit isa JointPhyloGroupedGaussianFit && !fit.hessian_positive_definite
        @warn "bootstrap_Sigma_r: the Hessian of the original fit is not positive definite (smallest " *
              "eigenvalue $(fit.hessian_min_eigenvalue)); a variance component is probably on its boundary. " *
              "The bootstrap runs, but interval ends of quantities that depend on that component can be " *
              "truncated at the boundary."
    end

    # 6. one batch drawn first, then one refit per replicate
    simulate, refit = _bsig_build(fit, nb, args, _refit)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    Ys = simulate(rng)

    draws = Vector{Dict{Symbol, Any}}(undef, nb)
    failed = fill(false, nb)
    n_extract_err = 0
    n_thrown = 0; n_not_conv = 0; n_non_ident = 0; n_conv = 0; n_nonpd = 0
    for b in 1:nb
        progress && @info "bootstrap_Sigma_r: replicate $b / $nb"
        fb = try
            refit(Ys[b])
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        status = fb === nothing ? :thrown :
                 (!(fb isa typeof(fit)) || !fb.converged) ? :not_converged : :converged
        if status === :converged
            n_conv += 1
            fb isa JointPhyloGroupedGaussianFit && !fb.hessian_positive_definite && (n_nonpd += 1)
            status = _bsig_replicate_status(fit, fb)         # :ok or :non_identified
        end
        status === :thrown && (n_thrown += 1)
        status === :not_converged && (n_not_conv += 1)
        status === :non_identified && (n_non_ident += 1)
        if status !== :ok
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
        # a summary of the wrong shape is treated like an extraction error (NaN draw, counted as lost)
        bad_shape = ent !== nothing && any(haskey(ent, k) && size(ent[k]) != size(v) for (k, v) in pe_pairs)
        (ent === nothing || bad_shape) && (n_extract_err += 1)
        draws[b] = Dict{Symbol, Any}(k => (ent !== nothing && haskey(ent, k) && size(ent[k]) == size(v) ?
                                           ent[k] : _bsig_nan_like(v)) for (k, v) in pe_pairs)
    end
    n_failed = count(failed)
    # replicates lost to failed refits or to extraction errors (the latter are not in n_failed)
    n_lost = n_failed + n_extract_err
    n_lost > 0 && @warn "bootstrap_Sigma_r: $n_lost of $nb replicates were lost " *
        "($n_failed failed refits, $n_extract_err extraction errors); bounds rest on the survivors" *
        (n_lost == nb ? " and are all NaN." : ".") *
        (fit isa JointPhyloGroupedGaussianFit ?
         " Failed refits: $n_thrown thrown, $n_not_conv non-converged, $n_non_ident non-identified." : "")
    if fit isa JointPhyloGroupedGaussianFit && n_nonpd > 0
        @info "bootstrap_Sigma_r: $n_nonpd of $n_conv converged replicates have a non-positive-definite " *
              "Hessian. They are kept (R checks convergence only and the percentile bootstrap needs only " *
              "point estimates), but such replicates often sit on a variance boundary, so interval ends of " *
              "quantities that depend on the unique or ordinary variances can be truncated there."
    end

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
        if startswith(String(k), "R_")   # interpolation can overshoot +-1 by an ulp
            # only a rounding overshoot is clamped; a genuine |R| > 1 + 1e-8 is left visible (it is a bug upstream)
            L .= ifelse.(abs.(L) .<= 1 + 1e-8, clamp.(L, -1.0, 1.0), L)
            U .= ifelse.(abs.(U) .<= 1 + 1e-8, clamp.(U, -1.0, 1.0), U)
        end
        lo[k] = L; hi[k] = U
        v isa AbstractVector && (neff[k] = vec(N))
    end

    # 8. attrition warning over vector entries only (R:465-486, R limitation kept)
    if !isempty(neff)
        worst_k = :none; worst = Inf
        for (k, v) in pe_pairs
            haskey(neff, k) || continue
            ok = [i for i in eachindex(v) if isfinite(v[i])]   # structural NaN is not attrition
            isempty(ok) && continue
            f = minimum(neff[k][ok]) / nb
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
        keep_draws ? NamedTuple[nt(d, keys_pe) for d in draws] : nothing,
        fit isa JointPhyloGroupedGaussianFit ?
            (n_thrown = n_thrown, n_not_converged = n_not_conv, n_non_identified = n_non_ident,
             n_converged = n_conv, n_hessian_not_pd = n_nonpd,
             original_hessian_pd = fit.hessian_positive_definite) : NamedTuple())
end

# Any other class: refuse explicitly (no verified simulate-and-refit route).
bootstrap_Sigma_r(fit; kwargs...) = _bsig_refuse(
    "unsupported fit class $(typeof(fit)). Supported: GllvmFit, TwoLevelFit, " *
    "PrecisionMultivariateFit and JointPhyloGroupedGaussianFit (tier :phy only for the last two). " *
    "Non-Gaussian fits have no verified simulate-and-refit route.")
